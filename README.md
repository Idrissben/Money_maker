# Stock Price Prediction Repository

This repository contains code for predicting the stock price of a company based on its fundamental economic values. The prediction model is built using machine learning techniques and aims to provide insights into potential stock price movements.

## Team JIMAMS
This project was developed by Team JIMAMS during a Hackathon with Quinten. The team members include:

- Idriss Bennis
- Ajouad Akjouj
- Mathieu Péharpré
- Joseph Moussa
- Samuel Berrebi
- Marie-Sophie Richard

## Functions

### `calculate_volatility(df: pd.DataFrame) -> pd.DataFrame`

This function calculates the volatility of stock prices for valid stocks. It filters stocks with more than 10 data points, calculates the standard deviation (volatility), and returns a DataFrame with two columns: 'Symbol' and 'Volatility', sorted by volatility.

### `train_lasso(dataset: pd.DataFrame) -> Lasso`

This function trains a Lasso regression model on the input dataset and performs hyperparameter tuning using GridSearchCV to find the best alpha parameter. It returns the best Lasso regression model with the optimal alpha.

### `map_rating_to_scale(rating: Union[str, float, None]) -> Union[float, Optional[float]]`

This function maps a rating to a numerical scale based on a predefined dictionary. It can handle ratings as strings (e.g., "S+," "Buy"), floats (e.g., 7.5), or None, and returns the mapped rating on a numerical scale or None if the input rating is not found in the mapping.

### `preprocessing(dataset: pd.DataFrame) -> pd.DataFrame`

This function preprocesses the input dataset by performing various data cleaning and transformation operations. It handles missing values, maps ratings to a numerical scale, removes highly correlated features, one-hot encodes categorical columns, fills missing values using KNN Imputer, scales the data, and returns the preprocessed dataset.

## Usage

You can use the functions provided in this repository to perform the following tasks:

1. Calculate the volatility of stock prices for valid stocks using `calculate_volatility`.
2. Train a Lasso regression model for stock price prediction using `train_lasso`.
3. Map ratings to a numerical scale using `map_rating_to_scale`.
4. Preprocess your financial dataset for stock price prediction using `preprocessing`.

## Example

```python
import pandas as pd

# Load your financial dataset
financial_data = pd.read_csv("your_financial_data.csv")

# Calculate volatility
volatility_df = calculate_volatility(financial_data)

# Train Lasso regression model
lasso_model = train_lasso(financial_data)

# Preprocess your financial dataset
preprocessed_data = preprocessing(financial_data)
```

## Contributors

We would like to thank all contributors for their valuable input and effort in creating this repository.

If you have any questions or suggestions, please feel free to reach out to us.

Happy hacking and happy stock price prediction!
