import re
import time
import random
from dataclasses import dataclass, asdict
from typing import Optional, List, Set, Dict
from urllib.parse import urljoin, urlencode, urlparse, parse_qs, urlunparse

import requests
from bs4 import BeautifulSoup
import pandas as pd

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE = "https://bama.ir"

# Seed pages to help you collect enough unique "Samand" listings without relying on pagination only.
# (All of these are still Samand family pages on bama.ir)
SEED_LISTING_URLS = [
    "https://bama.ir/car/samand",
    "https://bama.ir/car/samand-lx",
    "https://bama.ir/car/samand-se",
    "https://bama.ir/car/samand-soren",
    "https://bama.ir/car/samand-soren-elx",
    "https://bama.ir/car/samand-soren-plus",
    "https://bama.ir/car/samand-soren-pluscng",
    "https://bama.ir/car/samand?transmission=automatic",
]

TARGET_COUNT = 50
MIN_YEAR = 1385  # "after 1385" => year > 1385


# --- helpers: digits + parsing ------------------------------------------------
PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

def normalize_text(s: str) -> str:
    s = s.replace("\u200c", " ").strip()  # remove ZWNJ
    s = s.translate(PERSIAN_DIGITS).translate(ARABIC_DIGITS)
    s = s.replace("٬", ",")  # Arabic thousands separator -> comma
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_int_from_text(s: Optional[str]) -> Optional[int]:
    if not s:
        return None
    s = normalize_text(s)
    # pick digits+commas
    m = re.search(r"(\d[\d,]*)", s)
    if not m:
        return None
    num = m.group(1).replace(",", "")
    try:
        return int(num)
    except ValueError:
        return None

def find_first_iran_year(strings: List[str]) -> Optional[int]:
    """
    Finds the first token that looks like an Iranian production year (13xx or 14xx).
    Avoids tokens with slashes like 1404/11/14.
    """
    for t in strings:
        t = normalize_text(t)
        if "/" in t:
            continue
        if re.fullmatch(r"(13|14)\d{2}", t):
            try:
                return int(t)
            except ValueError:
                pass
    return None

def adjacent_value(strings: List[str], label: str) -> Optional[str]:
    label = normalize_text(label)
    for i, s in enumerate(strings):
        if normalize_text(s) == label and i + 1 < len(strings):
            return normalize_text(strings[i + 1])
    return None

def extract_price(strings: List[str]) -> Optional[str]:
    """
    Tries to extract the 'main' price shown near the top.
    Handles both standard price and installment-style pages (پیش پرداخت/ماهانه).
    """
    # define a boundary so we don't accidentally grab other numbers deep in the page
    stop_markers = {"نمایش شماره", "تماس با فروشنده", "کارکرد"}
    header: List[str] = []
    for s in strings:
        ns = normalize_text(s)
        if ns in stop_markers:
            break
        header.append(ns)

    # installment case
    if "پیش پرداخت" in header:
        start = header.index("پیش پرداخت")
        return " ".join(header[start:])

    # normal price case
    # Sometimes "تومان" is attached, sometimes separated.
    for s in header:
        if s == "توافقی":
            return s
        if "تومان" in s:
            return s

    for i, s in enumerate(header):
        if s == "تومان" and i > 0:
            return f"{header[i-1]} تومان"

    return None

def normalize_transmission(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    raw = normalize_text(raw)
    if "اتومات" in raw:
        return "automatic"
    if "دنده" in raw:
        return "manual"
    return raw  # fallback


# --- HTTP session with retries ------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()

    retries = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # Headers matter on many sites
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "fa-IR,fa;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    })
    return s

def fetch_html(session: requests.Session, url: str, timeout=(20, 40)) -> str:
    r = session.get(url, timeout=timeout)
    # If you get 403/401, it’s a blocking/anti-bot issue; connect timeout is different.
    r.raise_for_status()
    if not r.encoding:
        r.encoding = "utf-8"
    return r.text


# --- scraping logic -----------------------------------------------------------
def add_or_replace_query_param(url: str, key: str, value: str) -> str:
    """
    Adds ?key=value or replaces it if it exists.
    """
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    q[key] = [value]
    new_query = urlencode(q, doseq=True)
    return urlunparse(parsed._replace(query=new_query))

def extract_detail_links(listing_html: str) -> List[str]:
    soup = BeautifulSoup(listing_html, "html.parser")
    links: Set[str] = set()

    # Common pattern for bama detail pages: /car/detail-xxxx-...
    for a in soup.select('a[href^="/car/detail-"], a[href^="https://bama.ir/car/detail-"]'):
        href = a.get("href")
        if not href:
            continue
        href = href.split("#")[0]
        full = urljoin(BASE, href)
        links.add(full)

    return sorted(links)

def parse_detail_page(detail_html: str, url: str) -> Dict[str, Optional[str]]:
    soup = BeautifulSoup(detail_html, "html.parser")
    strings = [normalize_text(s) for s in soup.stripped_strings]

    year = find_first_iran_year(strings)
    mileage_raw = adjacent_value(strings, "کارکرد")
    color = adjacent_value(strings, "رنگ بدنه")
    transmission_raw = adjacent_value(strings, "گیربکس")
    description = adjacent_value(strings, "توضیحات")
    price = extract_price(strings)

    mileage_km = None
    if mileage_raw:
        mr = normalize_text(mileage_raw)
        if "صفر" in mr:
            mileage_km = 0
        else:
            mileage_km = parse_int_from_text(mr)

    return {
        "url": url,
        "price": price,
        "mileage_km": mileage_km,
        "mileage_raw": mileage_raw,
        "color": color,
        "production_year": year,
        "transmission": normalize_transmission(transmission_raw),
        "transmission_raw": transmission_raw,
        "description": description,
    }

def scrape_samand_after_1385(
    target_count: int = TARGET_COUNT,
    min_year: int = MIN_YEAR,
    max_pages_per_seed: int = 10,
    sleep_range=(0.8, 1.6),
    output_xlsx: str = "samand_after_1385.xlsx",
) -> pd.DataFrame:
    session = make_session()

    # 1) Collect candidate detail URLs
    candidate_urls: List[str] = []
    seen_urls: Set[str] = set()

    for seed in SEED_LISTING_URLS:
        for page in range(1, max_pages_per_seed + 1):
            page_url = seed if page == 1 else add_or_replace_query_param(seed, "page", str(page))

            try:
                html = fetch_html(session, page_url)
            except requests.RequestException as e:
                print(f"[WARN] listing fetch failed: {page_url} -> {type(e).__name__}: {e}")
                break

            links = extract_detail_links(html)
            new_links = [u for u in links if u not in seen_urls]

            if not new_links:
                # likely no pagination or we reached the end
                break

            for u in new_links:
                seen_urls.add(u)
                candidate_urls.append(u)

            # If we already have plenty, stop expanding
            if len(candidate_urls) >= target_count * 3:
                break

            time.sleep(random.uniform(*sleep_range))

        if len(candidate_urls) >= target_count * 3:
            break

    print(f"[INFO] collected {len(candidate_urls)} candidate detail URLs")

    # 2) Visit detail pages and extract fields
    rows: List[Dict[str, Optional[str]]] = []
    visited: Set[str] = set()

    for url in candidate_urls:
        if url in visited:
            continue
        visited.add(url)

        try:
            html = fetch_html(session, url)
            data = parse_detail_page(html, url)
        except requests.RequestException as e:
            print(f"[WARN] detail fetch failed: {url} -> {type(e).__name__}: {e}")
            continue
        except Exception as e:
            print(f"[WARN] detail parse failed: {url} -> {type(e).__name__}: {e}")
            continue

        year = data.get("production_year")
        if isinstance(year, int) and year > min_year:
            rows.append(data)

        if len(rows) >= target_count:
            break

        time.sleep(random.uniform(*sleep_range))

    df = pd.DataFrame(rows)

    # 3) Save Excel
    df.to_excel(output_xlsx, index=False)
    print(f"[OK] saved {len(df)} rows to {output_xlsx}")

    return df


if __name__ == "__main__":
    scrape_samand_after_1385()
