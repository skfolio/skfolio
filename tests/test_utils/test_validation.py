"""Tests for validation utilities."""

import numpy as np
import pandas as pd
import pytest
import sklearn.utils.validation as skv
from numpy.testing import assert_array_equal
from sklearn import config_context
from sklearn.base import BaseEstimator

from skfolio.pre_selection import DropCorrelated
from skfolio.utils.validation import validate_cross_sectional_data


class DummyEstimator(BaseEstimator):
    """Dummy estimator for testing validation."""

    pass


def test_validate_data(X):
    """Test sklearn's validate_data with skfolio estimator."""
    with config_context(transform_output="pandas"):
        model = DropCorrelated()
        _ = skv.validate_data(model, X)

    model = DropCorrelated()
    _ = skv.validate_data(model, X)


class TestValidateCrossSectionalData:
    """Test suite for validate_cross_sectional_data function."""

    def test_numpy_3d_array_only_X(self):
        """Test validation with 3D NumPy array for X only."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)

        X_val = validate_cross_sectional_data(estimator, X=X)

        assert X_val.shape == (10, 5, 3)
        assert estimator.n_features_in_ == 3
        assert not hasattr(estimator, "assets_")
        assert not hasattr(estimator, "factors_")
        assert not hasattr(estimator, "index_")

    def test_numpy_arrays_X_and_y(self):
        """Test validation with both X and y as NumPy arrays."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 5)

        X_val, y_val = validate_cross_sectional_data(estimator, X=X, y=y)

        assert X_val.shape == (10, 5, 3)
        assert y_val.shape == (10, 5)
        assert estimator.n_features_in_ == 3

    def test_dataframe_X_with_multiindex(self):
        """Test validation with DataFrame having MultiIndex columns."""
        estimator = DummyEstimator()
        dates = pd.date_range("2020-01-01", periods=10)
        assets = ["AAPL", "MSFT", "GOOGL"]
        factors = ["momentum", "value"]

        columns = pd.MultiIndex.from_product(
            [assets, factors], names=["asset", "factor"]
        )
        X_df = pd.DataFrame(np.random.randn(10, 6), index=dates, columns=columns)

        X_val = validate_cross_sectional_data(estimator, X=X_df)

        assert X_val.shape == (10, 3, 2)
        assert estimator.n_features_in_ == 2
        assert_array_equal(estimator.assets_, np.array(assets))
        assert_array_equal(estimator.factors_, np.array(factors))
        assert_array_equal(estimator.index_, dates)

    def test_dataframe_X_and_y(self):
        """Test validation with both X and y as DataFrames."""
        estimator = DummyEstimator()
        dates = pd.date_range("2020-01-01", periods=10)
        assets = ["AAPL", "MSFT", "GOOGL"]
        factors = ["momentum", "value"]

        columns = pd.MultiIndex.from_product(
            [assets, factors], names=["asset", "factor"]
        )
        X_df = pd.DataFrame(np.random.randn(10, 6), index=dates, columns=columns)
        y_df = pd.DataFrame(np.random.randn(10, 3), index=dates, columns=assets)

        X_val, y_val = validate_cross_sectional_data(estimator, X=X_df, y=y_df)

        assert X_val.shape == (10, 3, 2)
        assert y_val.shape == (10, 3)

    def test_y_dataframe_reindexed_to_match_X(self):
        """Test that y DataFrame is reindexed to match X's index and assets."""
        estimator = DummyEstimator()
        dates_X = pd.date_range("2020-01-01", periods=10)
        dates_y = pd.date_range("2020-01-01", periods=12)  # More dates
        assets = ["AAPL", "MSFT", "GOOGL"]
        factors = ["momentum", "value"]

        columns = pd.MultiIndex.from_product(
            [assets, factors], names=["asset", "factor"]
        )
        X_df = pd.DataFrame(np.random.randn(10, 6), index=dates_X, columns=columns)
        y_df = pd.DataFrame(
            np.random.randn(12, 4),
            index=dates_y,
            columns=["AAPL", "MSFT", "GOOGL", "TSLA"],  # Extra asset
        )

        X_val, y_val = validate_cross_sectional_data(estimator, X=X_df, y=y_df)

        assert X_val.shape == (10, 3, 2)
        assert y_val.shape == (10, 3)  # Reindexed to match X

    def test_reset_false_does_not_update_attributes(self):
        """Test that reset=False doesn't update estimator attributes but validates consistency."""
        estimator = DummyEstimator()
        X1 = np.random.randn(10, 5, 3)
        X2 = np.random.randn(8, 4, 3)  # Same n_factors=3

        # First call with reset=True
        validate_cross_sectional_data(estimator, X=X1, reset=True)
        assert estimator.n_features_in_ == 3

        # Second call with reset=False and same n_factors - should work
        validate_cross_sectional_data(estimator, X=X2, reset=False)
        assert estimator.n_features_in_ == 3  # Should not change

        # Third call with reset=False and different n_factors - should raise error
        X3 = np.random.randn(8, 4, 2)  # Different n_factors=2
        with pytest.raises(ValueError, match="Expected n_factors=3"):
            validate_cross_sectional_data(estimator, X=X3, reset=False)

    def test_allows_nan_values(self):
        """Test that NaN values are allowed in inputs."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 5)

        # Insert some NaNs
        X[0, 0, 0] = np.nan
        y[1, 1] = np.nan

        X_val, y_val = validate_cross_sectional_data(estimator, X=X, y=y)

        assert np.isnan(X_val[0, 0, 0])
        assert np.isnan(y_val[1, 1])

    def test_no_validation_both_raises_error(self):
        """Test that skipping validation for both X and y raises error."""
        estimator = DummyEstimator()

        with pytest.raises(ValueError, match="Validation should be done on X, y"):
            validate_cross_sectional_data(estimator)

    def test_no_validation_X_with_y_raises_error(self):
        """Test that providing y without X raises error."""
        estimator = DummyEstimator()
        y = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="X must be provided"):
            validate_cross_sectional_data(estimator, y=y)

    def test_X_not_3d_raises_error(self):
        """Test that 2D array for X raises error."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5)  # 2D instead of 3D

        with pytest.raises(ValueError, match="must be a 3D array"):
            validate_cross_sectional_data(estimator, X=X)

    def test_X_4d_raises_error(self):
        """Test that 4D array for X raises error."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3, 2)  # 4D

        with pytest.raises(ValueError, match="must be a 3D array"):
            validate_cross_sectional_data(estimator, X=X)

    def test_y_not_2d_raises_error(self):
        """Test that 1D array for y raises error."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10)  # 1D

        with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
            validate_cross_sectional_data(estimator, X=X, y=y)

    def test_incompatible_shapes_raises_error(self):
        """Test that incompatible X and y shapes raise error."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 6)  # Wrong number of assets

        with pytest.raises(ValueError, match="Incompatible shapes"):
            validate_cross_sectional_data(estimator, X=X, y=y)

    def test_X_dataframe_not_multiindex_raises_error(self):
        """Test that DataFrame without MultiIndex columns raises error."""
        estimator = DummyEstimator()
        X_df = pd.DataFrame(np.random.randn(10, 6), columns=list("ABCDEF"))

        with pytest.raises(ValueError, match="must have MultiIndex columns"):
            validate_cross_sectional_data(estimator, X=X_df)

    def test_X_dataframe_wrong_nlevels_raises_error(self):
        """Test that MultiIndex with wrong number of levels raises error."""
        estimator = DummyEstimator()
        columns = pd.MultiIndex.from_tuples(
            [("A", "X", 1), ("B", "Y", 2)], names=["asset", "factor", "extra"]
        )
        X_df = pd.DataFrame(np.random.randn(10, 2), columns=columns)

        with pytest.raises(ValueError, match="exactly 2 levels"):
            validate_cross_sectional_data(estimator, X=X_df)

    def test_preserves_multiindex_names(self):
        """Test that MultiIndex column names are preserved."""
        estimator = DummyEstimator()
        assets = ["AAPL", "MSFT"]
        factors = ["momentum", "value"]
        columns = pd.MultiIndex.from_product(
            [assets, factors], names=["stock", "feature"]
        )
        X_df = pd.DataFrame(np.random.randn(5, 4), columns=columns)

        validate_cross_sectional_data(estimator, X=X_df)

        # Check that names were preserved during reindexing
        assert_array_equal(estimator.assets_, np.array(assets))
        assert_array_equal(estimator.factors_, np.array(factors))

    def test_handles_missing_combinations_in_multiindex(self):
        """Test that missing (asset, factor) combinations raise an error."""
        estimator = DummyEstimator()
        # Create incomplete MultiIndex (missing some combinations)
        columns = pd.MultiIndex.from_tuples(
            [("AAPL", "momentum"), ("AAPL", "value"), ("MSFT", "momentum")],
            # Missing ("MSFT", "value")
            names=["asset", "factor"],
        )
        X_df = pd.DataFrame(np.random.randn(5, 3), columns=columns)

        # Should raise an error for incomplete combinations
        with pytest.raises(
            ValueError, match="X must have all \\(asset, factor\\) combinations"
        ):
            validate_cross_sectional_data(estimator, X=X_df)

    def test_list_input_converted_to_array(self):
        """Test that list inputs are converted to arrays."""
        estimator = DummyEstimator()
        X_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]  # 2x2x2 list
        y_list = [[1, 2], [3, 4]]  # 2x2 list

        X_val, y_val = validate_cross_sectional_data(estimator, X=X_list, y=y_list)

        assert isinstance(X_val, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        assert X_val.shape == (2, 2, 2)
        assert y_val.shape == (2, 2)

    def test_dtype_conversion(self):
        """Test that inputs are validated and converted to numeric dtype."""
        estimator = DummyEstimator()
        # check_array with dtype="numeric" preserves int if no conversion needed
        X = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float)
        y = np.array([[1.0, 2.0]], dtype=float)

        X_val, y_val = validate_cross_sectional_data(estimator, X=X, y=y)

        # Verify numeric dtypes are preserved
        assert np.issubdtype(X_val.dtype, np.floating)
        assert np.issubdtype(y_val.dtype, np.floating)

    def test_multiple_observations_different_sizes(self):
        """Test with various observation/asset/factor combinations."""
        test_cases = [
            (5, 3, 2),  # Small
            (100, 50, 10),  # Large
            (1, 1, 1),  # Minimal
            (20, 10, 5),  # Medium
        ]

        for n_obs, n_assets, n_factors in test_cases:
            estimator = DummyEstimator()
            X = np.random.randn(n_obs, n_assets, n_factors)
            y = np.random.randn(n_obs, n_assets)

            X_val, y_val = validate_cross_sectional_data(estimator, X=X, y=y)

            assert X_val.shape == (n_obs, n_assets, n_factors)
            assert y_val.shape == (n_obs, n_assets)
            assert estimator.n_features_in_ == n_factors

    def test_y_none_equivalent_to_no_validation(self):
        """Test that y=None behaves the same as y='no_validation'."""
        estimator1 = DummyEstimator()
        estimator2 = DummyEstimator()
        X = np.random.randn(10, 5, 3)

        X_val1 = validate_cross_sectional_data(estimator1, X=X, y=None)
        X_val2 = validate_cross_sectional_data(estimator2, X=X)

        assert_array_equal(X_val1, X_val2)
        assert estimator1.n_features_in_ == estimator2.n_features_in_

    def test_consecutive_calls_with_different_shapes(self):
        """Test behavior with consecutive calls having different input shapes."""
        estimator = DummyEstimator()

        # First call
        X1 = np.random.randn(10, 5, 3)
        validate_cross_sectional_data(estimator, X=X1, reset=True)
        assert estimator.n_features_in_ == 3

        # Second call with different shape, reset=True (should update)
        X2 = np.random.randn(8, 4, 7)
        validate_cross_sectional_data(estimator, X=X2, reset=True)
        assert estimator.n_features_in_ == 7

        # Third call with different shape, reset=False (should raise error)
        X3 = np.random.randn(5, 3, 2)
        with pytest.raises(ValueError, match="Expected n_factors=7"):
            validate_cross_sectional_data(estimator, X=X3, reset=False)
        assert estimator.n_features_in_ == 7  # Still 7

    def test_copy_false_no_copy_if_correct_dtype(self):
        """Test that copy=False doesn't copy if dtype is already correct."""
        estimator = DummyEstimator()
        X = np.random.randn(5, 10, 3).astype(np.float64)
        y = np.random.randn(5, 10).astype(np.float64)

        X_validated, y_validated = validate_cross_sectional_data(
            estimator, X=X, y=y, copy=False
        )

        # With copy=False and correct dtype, should share memory
        assert np.shares_memory(X, X_validated)
        assert np.shares_memory(y, y_validated)

    def test_copy_false_preserves_integer_dtype(self):
        """Test that copy=False with integer dtype doesn't convert or copy."""
        estimator = DummyEstimator()
        X = np.random.randint(0, 10, (5, 10, 3), dtype=np.int64)
        y = np.random.randint(0, 10, (5, 10), dtype=np.int64)

        X_validated, y_validated = validate_cross_sectional_data(
            estimator, X=X, y=y, copy=False
        )

        # Integer is numeric, so no conversion needed with copy=False
        # Should share memory
        assert np.shares_memory(X, X_validated)
        assert np.shares_memory(y, y_validated)
        # Dtype should be preserved
        assert X_validated.dtype == np.int64
        assert y_validated.dtype == np.int64

    def test_copy_true_always_copies(self):
        """Test that copy=True always creates a copy."""
        estimator = DummyEstimator()
        X = np.random.randn(5, 10, 3).astype(np.float64)
        y = np.random.randn(5, 10).astype(np.float64)

        X_validated, y_validated = validate_cross_sectional_data(
            estimator, X=X, y=y, copy=True
        )

        # Even with correct dtype, copy=True should create a copy
        assert not np.shares_memory(X, X_validated)
        assert not np.shares_memory(y, y_validated)
        # Values should be equal
        np.testing.assert_array_equal(X, X_validated)
        np.testing.assert_array_equal(y, y_validated)

    def test_copy_parameter_with_dataframe(self):
        """Test copy parameter with DataFrame inputs."""
        estimator = DummyEstimator()

        # Create DataFrame X with MultiIndex
        dates = pd.date_range("2020-01-01", periods=5)
        assets = ["A", "B", "C"]
        factors = ["F1", "F2"]
        columns = pd.MultiIndex.from_product(
            [assets, factors], names=["asset", "factor"]
        )
        X_df = pd.DataFrame(np.random.randn(5, 6), index=dates, columns=columns)

        # Create DataFrame y
        y_df = pd.DataFrame(np.random.randn(5, 3), index=dates, columns=assets)

        # Test with copy=False (DataFrame conversion always creates array)
        X_val1, y_val1 = validate_cross_sectional_data(
            estimator, X=X_df, y=y_df, copy=False
        )

        # Test with copy=True
        X_val2, y_val2 = validate_cross_sectional_data(
            estimator, X=X_df, y=y_df, copy=True
        )

        # Both should be valid arrays with same shape
        assert X_val1.shape == (5, 3, 2)
        assert X_val2.shape == (5, 3, 2)
        assert y_val1.shape == (5, 3)
        assert y_val2.shape == (5, 3)

    def test_copy_only_X(self):
        """Test copy parameter when validating only X."""
        estimator = DummyEstimator()
        X = np.random.randn(5, 10, 3).astype(np.float64)

        # copy=False
        X_val1 = validate_cross_sectional_data(estimator, X=X, copy=False)
        assert np.shares_memory(X, X_val1)

        # copy=True
        X_val2 = validate_cross_sectional_data(estimator, X=X, copy=True)
        assert not np.shares_memory(X, X_val2)
        np.testing.assert_array_equal(X, X_val2)
