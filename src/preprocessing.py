import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from typing import Union, Optional

def map_rating_to_scale(rating: Union[str, float, None]) -> Union[float, Optional[float]]:
    """
    Map a rating to a numerical scale.

    This function takes a rating as input and maps it to a numerical scale
    based on a predefined dictionary.

    Args:
        rating (Union[str, float, None]): The rating to be mapped. It can be a
        string (e.g., "S+", "Buy"), a float (e.g., 7.5), or None.

    Returns:
        Union[float, Optional[float]]: The mapped rating on a numerical scale,
        or None if the input rating is None or not found in the mapping.
    """
    rating_mapping = {
        "S+": 10.0, "S": 9.5, "S-": 9.0, "A+": 8.5, "A": 8.0,
        "A-": 7.5, "B+": 7.0, "B": 6.5, "B-": 6.0, "C+": 5.5,
        "C": 5.0, "C-": 4.5, "D+": 4.0, "D": 3.5, "D-": 3.0,
        "F": 0.0, np.nan: np.nan, "Strong Buy": 10.0,
        "Buy": 7.5, "Neutral": 5.0, "Sell": 2.5, "Strong Sell": 0.
    }
    return rating_mapping.get(rating, None)

def preprocessing(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input dataset.

    This function preprocesses the input dataset by performing various data cleaning
    and transformation operations.

    Args:
        dataset (pd.DataFrame): The input dataset containing financial data.

    Returns:
        pd.DataFrame: The preprocessed and transformed dataset.
    """
    # Keeping only rows with target variable not NaN
    dataset.dropna(subset=["price"], inplace=True)

    # Deleting estimated columns to remove human bias, deleting links
    for col in dataset.columns:
        if "estimated" in col or "Estimated" in col or "Link" in col or "link" in col:
            dataset.drop(columns=[col], inplace=True)
    dataset.drop(columns=["acceptedDate"], inplace=True)
    print("Dropped estimated columns.")

    # Handle 'missing' values
    for column in dataset.columns:
        if 'missing' in dataset[column].to_list():
            dataset[column] = dataset[column].apply(
        lambda x: float(x) if x != 'missing' else np.nan
    )
    print("Handled missing values.")

    # Map ratings to a numerical scale
    for column in ["rating", "ratingDetailsDCFRecommendation", "ratingDetailsDERecommendation",
                   "ratingDetailsPBRecommendation", "ratingDetailsPERecommendation", "ratingDetailsROARecommendation",
                   "ratingDetailsROERecommendation", "ratingRecommendation"]:
        dataset[column] = dataset[column].apply(map_rating_to_scale)
    dataset.drop(columns=["cik", "fillingDate", "period"], inplace=True)
    print("Mapped ratings to numerical scale.")

    # Calculate correlation matrix and remove highly correlated features
    correlation_matrix = dataset.corr().abs()
    highly_correlated = (correlation_matrix > 0.9) & (correlation_matrix < 1)
    correlated_features = set()

    for i in range(len(highly_correlated.columns)):
        for j in range(i):
            if highly_correlated.iloc[i, j]:
                feature_i = highly_correlated.columns[i]
                feature_j = highly_correlated.columns[j]
                if feature_i == 'price' or feature_j == 'price':
                    continue
                correlated_features.add((feature_i, feature_j))

    features_to_drop = set()
    for feature_i, feature_j in correlated_features:
        features_to_drop.add(feature_j)

    dataset.drop(columns=list(features_to_drop), inplace=True)
    print("Dropped highly correlated features.")
    dataset.reset_index(inplace=True)
    dataset.drop(columns=["index", "calendarYear", "week"], inplace=True)
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset["year"] = dataset["date"].dt.year
    dataset["month"] = dataset["date"].dt.month
    dataset['day'] = dataset['date'].dt.day

    threshold = 0.25 * dataset.shape[0]
    dataset.dropna(thresh=threshold, axis=1, inplace=True)
    dataset.drop(columns=["date"], inplace=True)
    print("Kept columns with at least 25% of non-missing values.")

    # One-hot encode the currency and symbol columns
    dataset = pd.get_dummies(dataset, columns=["reportedCurrency"], prefix=["reportedCurrency"])
    dataset = pd.get_dummies(dataset, columns=["symbol"], prefix=["symbol"])

    # Fill missing values with KNN Imputer
    imputer = KNNImputer(n_neighbors=5)
    dataset_preprocessed_imputed = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
    print("Applied KNNImputer.")

    # Scale columns, discarding currency and symbol columns. Use robust scaler to limit outlier influence
    reported_columns = [col for col in dataset_preprocessed_imputed.columns if "reportedCurrency" in col or "symbol" in col]
    columns_to_scale = [col for col in dataset_preprocessed_imputed.columns if col not in reported_columns]
    scaler = RobustScaler()
    dataset_preprocessed_imputed_scaled = pd.DataFrame(scaler.fit_transform(dataset_preprocessed_imputed[columns_to_scale]), columns=columns_to_scale)
    dataset_preprocessed_imputed_scaled = pd.concat([dataset_preprocessed_imputed_scaled, dataset_preprocessed_imputed[reported_columns]], axis=1)
    dataset_preprocessed_imputed_scaled = dataset_preprocessed_imputed
    print("Scaled data, preprocessing done.")

    return dataset_preprocessed_imputed_scaled