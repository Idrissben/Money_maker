import pandas as pd

def calculate_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the volatility of stock prices for valid stocks.

    This function calculates the volatility of stock prices for stocks that have
    more than 10 data points.

    Args:
        df (pd.DataFrame): The input DataFrame containing stock price data.

    Returns:
        pd.DataFrame: A DataFrame with two columns, 'Symbol' and 'Volatility', sorted by volatility.
    """
    # Count the occurrences of each stock symbol
    stock_counts = df['symbol'].value_counts()

    # Get the symbols of valid stocks with more than 10 data points
    valid_stocks = stock_counts[stock_counts > 10].index

    # Filter the DataFrame based on valid stocks
    filtered_df = df[df['symbol'].isin(valid_stocks)]

    # Group by 'Symbol' and calculate the standard deviation (volatility) of 'Price'
    result = filtered_df.groupby('symbol')['price'].std().sort_values().reset_index()

    # Rename the columns for clarity
    result.columns = ['Symbol', 'Volatility']

    return result