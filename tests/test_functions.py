"""
This file contains unit tests for the functions in src/.
"""
import unittest
import pandas as pd
from sklearn.linear_model import Lasso
from ..src.utils import calculate_volatility
from ..src.training import train_lasso
from ..src.preprocessing import map_rating_to_scale, preprocessing


class TestStockPredictionFunctions(unittest.TestCase):
    """
    Test the functions in src/utils.py, src/training.py, and src/preprocessing.py.
    """
    def setUp(self):
        """
        Set up the unit tests by loading the sample data. This requires having a sample data file in the same directory.
        """
        # Sample financial data for testing
        self.financial_data = pd.read_csv('sample_data.csv')

    def test_calculate_volatility(self):
        """
        Test the calculate_volatility function.
        """
        # Test if calculate_volatility returns a DataFrame
        result = calculate_volatility(self.financial_data)
        self.assertIsInstance(result, pd.DataFrame)

        # Test if the resulting DataFrame has the correct columns
        self.assertTrue(all(col in result.columns for col in ["Symbol", "Volatility"]))

    def test_train_lasso(self):
        """ 
        Test the train_lasso function.
        """
        # Test if train_lasso returns a Lasso model
        lasso_model = train_lasso(self.financial_data)
        self.assertIsInstance(
            lasso_model, Lasso
        )  # Make sure to import Lasso from sklearn

        # Add more test cases for hyperparameter tuning if needed

    def test_map_rating_to_scale(self):
        """
        Test the map_rating_to_scale function.
        """
        # Test mapping of valid ratings
        self.assertEqual(map_rating_to_scale("S+"), 10.0)
        self.assertEqual(map_rating_to_scale("Buy"), 7.5)
        self.assertEqual(map_rating_to_scale("Neutral"), 5.0)

        # Test mapping of invalid rating
        self.assertIsNone(map_rating_to_scale("UnknownRating"))

        # Test handling of None input
        self.assertIsNone(map_rating_to_scale(None))

    def test_preprocessing(self):
        """
        Test the preprocessing function.
        """
        # Test if preprocessing returns a DataFrame
        preprocessed_data = preprocessing(self.financial_data)
        self.assertIsInstance(preprocessed_data, pd.DataFrame)

        # Add more specific tests for preprocessing if needed


if __name__ == "__main__":
    unittest.main()
