"""Test Dataset module."""

import os
import shutil

import pandas as pd
import pytest

from skfolio.datasets import (
    load_factors_dataset,
    load_ftse100_dataset,
    load_nasdaq_dataset,
    load_sp500_dataset,
    load_sp500_implied_vol_dataset,
    load_sp500_index,
)
from skfolio.datasets._base import clear_data_home, get_data_home


class TestGetDataHome:
    #  Returns the default path to skfolio data directory if no argument is passed
    def test_default_path(self):
        assert get_data_home() == os.path.expanduser(os.path.join("~", "skfolio_data"))

    #  Creates the skfolio data directory if it does not exist
    def test_create_directory(self):
        data_home = os.path.expanduser(os.path.join("~", "skfolio_data"))
        shutil.rmtree(data_home, ignore_errors=True)
        get_data_home()
        assert os.path.exists(data_home)


class TestClearDataHome:
    #  Deletes all content of data home cache when given a valid path.
    def test_delete_content_valid_path(self):
        # Set up
        data_home = "valid/path"
        os.makedirs(data_home)
        with open(os.path.join(data_home, "file1.txt"), "w") as f:
            f.write("test")

        # Execute
        clear_data_home(data_home)

        # Assert
        assert not os.path.exists(data_home)

    #  Deletes all content of default data home cache when no path is given.
    def test_delete_content_default_path(self):
        clear_data_home()
        # Set up
        data_home = os.path.expanduser(os.path.join("~", "skfolio_data"))
        os.makedirs(data_home)
        with open(os.path.join(data_home, "file1.txt"), "w") as f:
            f.write("test")

        # Execute
        clear_data_home()

        # Assert
        assert not os.path.exists(data_home)

    #  Does not raise an error when given a non-existent path.
    def test_no_error_nonexistent_path(self):
        # Set up
        data_home = "nonexistent/path"

        # Execute and assert
        try:
            clear_data_home(data_home)
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")


class TestLoadSp500Dataset:
    #  Loads the S&P 500 dataset successfully
    def test_load_sp500_dataset_success(self):
        # Call the load_sp500_dataset function
        df = load_sp500_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_sp500_dataset_shape(self):
        # Call the load_sp500_dataset function
        df = load_sp500_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (8313, 20)

    #  DataFrame has the correct column names
    def test_load_sp500_dataset_columns(self):
        # Call the load_sp500_dataset function
        df = load_sp500_dataset()

        # Define the expected column names
        expected_columns = [
            "AAPL",
            "AMD",
            "BAC",
            "BBY",
            "CVX",
            "GE",
            "HD",
            "JNJ",
            "JPM",
            "KO",
            "LLY",
            "MRK",
            "MSFT",
            "PEP",
            "PFE",
            "PG",
            "RRC",
            "UNH",
            "WMT",
            "XOM",
        ]

        # Check if the column names of the DataFrame are correct
        assert list(df.columns) == expected_columns


class TestLoadSp500Index:
    def test_load_sp500_index_success(self):
        df = load_sp500_index()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_sp500_index_shape(self):
        df = load_sp500_index()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (8313, 1)

    #  DataFrame has the correct column names
    def test_load_sp500_dataset_columns(self):
        df = load_sp500_index()

        # Define the expected column names
        expected_columns = ["SP500"]

        # Check if the column names of the DataFrame are correct
        assert list(df.columns) == expected_columns


class TestLoadFactorsDataset:
    def test_load_factors_dataset_success(self):
        df = load_factors_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_factors_dataset_shape(self):
        df = load_factors_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (2264, 5)

    def test_load_factors_dataset_columns(self):
        df = load_factors_dataset()

        # Define the expected column names
        expected_columns = ["MTUM", "QUAL", "SIZE", "USMV", "VLUE"]

        # Check if the column names of the DataFrame are correct
        assert list(df.columns) == expected_columns


class TestLoadFtse100Dataset:
    def test_load_ftse100_dataset_success(self):
        df = load_ftse100_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_ftse100_dataset_shape(self):
        df = load_ftse100_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (5960, 64)


class TestNasdaqDataset:
    def test_load_nasdaq_dataset_success(self):
        df = load_nasdaq_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_nasdaq_dataset_shape(self):
        df = load_nasdaq_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (1362, 1455)


class TestSp500ImpliedVolDataset:
    def test_load_sp500_implied_vol_dataset_success(self):
        df = load_sp500_implied_vol_dataset()

        # Check if the returned object is a pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)

        # Check if the DataFrame is not empty
        assert not df.empty

    #  Returns a pandas DataFrame with the correct shape
    def test_load_sp500_implied_vol_dataset_shape(self):
        df = load_sp500_implied_vol_dataset()

        # Check if the shape of the DataFrame is correct
        assert df.shape == (3270, 20)
