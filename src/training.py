import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def train_lasso(dataset: pd.DataFrame) -> Lasso:
    """
    Train a Lasso regression model with hyperparameter tuning.

    This function trains a Lasso regression model on the input dataset and performs
    hyperparameter tuning using GridSearchCV to find the best alpha parameter.

    Args:
        dataset (pd.DataFrame): The input dataset containing features and the target variable.

    Returns:
        Lasso: The best Lasso regression model with the optimal alpha.
    """
    # Define the features matrix and the target variable
    X = dataset.drop(columns=["price"])
    Y = dataset["price"]

    # Create a Lasso estimator
    lasso = Lasso(max_iter=100000)
    param_grid = {'alpha': [0.1, 1.0, 10.0]}

    # Define scoring metrics for grid search
    scoring = {
        'MSE': make_scorer(mean_squared_error),
        'MAE': make_scorer(mean_absolute_error),
        'R2': make_scorer(r2_score)
    }

    # Perform time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring=scoring, cv=tscv, refit=False)
    grid_search.fit(X, Y)

    # Get the best Lasso model with optimal alpha
    best_lasso = grid_search.best_estimator_

    return best_lasso