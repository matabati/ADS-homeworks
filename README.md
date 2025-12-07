# ADS-homeworks
I gadder my Applied Data Science (ADS) home works for my master degree here.

# Anime Ratings Analysis & Recommender System

## Project Overview
This project focuses on analyzing anime ratings data to understand factors influencing anime popularity and viewer ratings, ultimately laying the groundwork for a recommender system. It involves comprehensive data analysis, feature engineering, and the implementation of various machine learning models for both regression (predicting continuous ratings) and classification (predicting rating categories).

## Dataset
The analysis utilizes two primary datasets:

*   **`anime.csv`**: Contains detailed information about various anime titles.
    *   `anime_id`: Unique identifier for each anime on MyAnimeList.net.
    *   `name`: Full title of the anime.
    *   `genre`: Comma-separated list of genres (e.g., Action, Drama, Sci-Fi).
    *   `type`: The format of the anime (e.g., Movie, TV, OVA).
    *   `episodes`: Number of episodes (1 for movies).
    *   `rating`: Average rating out of 10 for the anime.
    *   `members`: Number of community members following the anime.

*   **`rating.csv`**: Contains user-assigned ratings for anime.
    *   `user_id`: Non-identifiable randomly generated user ID.
    *   `anime_id`: The anime that this user has rated.
    *   `rating`: Rating out of 10 assigned by the user (-1 if watched but not rated).

## Analysis & Methodology
The project follows a structured approach:

### 1. Data Loading and Initial Exploration
-   Loading `anime.csv` and `rating.csv` into Pandas DataFrames.
-   Initial inspection of data structures, head rows, and basic statistics.

### 2. Data Merging and Cleaning
-   Merging `anime` and `rating` DataFrames on `anime_id`.
-   Filtering out `-1` ratings from the `rating` dataset, which indicate watched but unrated anime.
-   Recalculating average ratings and total valid ratings for each anime.
-   Handling duplicate entries in the datasets.

### 3. Dataset Analysis
-   Identifying top animes based on recalculated average ratings (with a minimum rating threshold).
-   Identifying top animes based on the sheer number of valid ratings.
-   Exploring the impact of different genres on average ratings.
-   Investigating the relationship between episode count and average ratings by categorizing episodes into bins.

### 4. Data Visualization
-   Visualizing anime type distribution, average rating by episode count, top genres by average rating, and the relationship between average rating and total valid ratings.
-   Generating an interactive Plotly bubble chart for episodes vs. average rating (sized by members).
-   Creating a grouped bar chart comparing average rating by anime type for top genres.

### 5. Feature Engineering
-   **Mathematical Transformations**: Applying log transformations to skewed numerical features like `members` and `total_valid_ratings`.
-   **Ratios and Combinations**: Creating new features such as `members_per_episode` and `rating_x_members`.
-   **Aggregation Statistics**: Counting the number of genres associated with each anime.
-   **Mutual Information**: Performing feature selection to identify features most relevant to the `average_rating`.
-   **Dimensionality Reduction**: Preparing for PCA (though not explicitly implemented in detail, mentioned as a next step).

### 6. Regression Models
-   Splitting data into training and testing sets.
-   Implementing and evaluating a suite of regression models to predict `average_rating`:
    -   Linear Regression
    -   Ridge Regression
    -   Lasso Regression
    -   Kernel Ridge Regression
    -   Polynomial Regression
    -   Bayesian Ridge Regression
    -   Elastic Net Regression
    -   Decision Tree Regression
    -   Support Vector Regressor (SVR)
-   Evaluating models using Mean Squared Error (MSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and R-squared (R²).
-   Discussion on the Kernel Trick, limitations of Time Series models (ARIMA/SARIMA) for this dataset, and the applicability of Locally Weighted Regression.
-   Justification for the best regression metric (MAE) for this specific problem.

### 7. Binary Classification Models
-   Creating a binary target variable (`is_high_rated`) by applying a threshold to `average_rating`.
-   Splitting data into training and testing sets with stratified sampling.
-   Implementing and evaluating various classification models:
    -   Logistic Regression
    -   Support Vector Machine (Linear SVM)
    -   Kernel SVM (RBF Kernel)
    -   K-Nearest Neighbors (KNN) with hyperparameter tuning.
    -   Decision Tree Classifier with hyperparameter tuning.
    -   Random Forest Classifier
-   Evaluating models using Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC Curve, and AUC.
-   Discussion on the best classification metric (F1-Score and AUC-ROC) and Decision Tree regularization techniques.

### 8. Multiclass Classification Models
-   Creating a multiclass target variable (`rating_category`) by discretizing `average_rating` into four categories (e.g., 'below_average', 'average', 'above_average', 'excellent') using quantiles.
-   Splitting data into training and testing sets with stratified sampling.
-   Handling missing values in features using `SimpleImputer`.
-   Implementing and evaluating various multiclass classification models:
    -   Logistic Regression (Multinomial)
    -   Decision Tree Classifier with hyperparameter tuning.
    -   Random Forest Classifier
    -   Support Vector Machine (Multiclass SVC)
    -   K-Nearest Neighbors (KNN) with hyperparameter tuning.
    -   XGBoost Classifier
-   Evaluating models using Accuracy, per-class Precision, Recall, Macro/Micro/Weighted F1-Score, Log Loss, and Confusion Matrix.
-   Justification for the best multiclass classification metrics (Weighted F1-Score and Log Loss).
-   Discussion on how KNN and Decision Trees can be extended for multi-label classification.

## Libraries Used
-   `pandas` for data manipulation and analysis.
-   `numpy` for numerical operations.
-   `matplotlib` and `seaborn` for static visualizations.
-   `plotly.express` for interactive visualizations.
-   `sklearn` (scikit-learn) for preprocessing, feature selection, regression models, and classification models.
-   `xgboost` for gradient boosting classification.

This project provides a comprehensive overview of machine learning techniques applied to anime rating prediction, offering insights into model performance and feature importance across different problem formulations.
