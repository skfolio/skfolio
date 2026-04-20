"""Tests for validation utilities."""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.utils.validation as skv
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
    """Test suite for validate_cross_sectional_data."""

    def test_numpy_3d_array_only_X(self):
        """Test validation with 3D NumPy array for X only."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)

        X_val = validate_cross_sectional_data(estimator, X=X)

        assert X_val.shape == (10, 5, 3)
        assert estimator.n_features_in_ == 3

    def test_numpy_arrays_X_and_y(self):
        """Test validation with both X and y as NumPy arrays."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 5)

        X_val, y_val, w_val = validate_cross_sectional_data(estimator, X=X, y=y)

        assert X_val.shape == (10, 5, 3)
        assert y_val.shape == (10, 5)
        assert w_val.shape == (10, 5)
        np.testing.assert_allclose(w_val, np.ones((10, 5)))
        assert estimator.n_features_in_ == 3

    def test_y_none_returns_only_X(self):
        """Test that y=None returns only validated X for unsupervised estimators."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)

        result = validate_cross_sectional_data(estimator, X=X, y=None)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5, 3)

    def test_no_validation_default_skips_tag_check(self):
        """Test that omitting y (default 'no_validation') works on supervised
        estimators, mirroring sklearn's predict path."""

        class SupervisedEstimator(BaseEstimator):
            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.target_tags.required = True
                return tags

        estimator = SupervisedEstimator()
        X = np.random.randn(10, 5, 3)

        result = validate_cross_sectional_data(estimator, X=X)

        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5, 3)

    def test_y_none_raises_when_target_required_by_tag(self):
        """Test that y=None raises for estimators with target_tags.required."""

        class SupervisedEstimator(BaseEstimator):
            def __sklearn_tags__(self):
                tags = super().__sklearn_tags__()
                tags.target_tags.required = True
                return tags

        estimator = SupervisedEstimator()
        X = np.random.randn(10, 5, 3)

        with pytest.raises(ValueError, match="requires y to be passed"):
            validate_cross_sectional_data(estimator, X=X, y=None)

    def test_cs_weights_without_y_raises_error(self):
        """Test that providing cs_weights without y raises error."""
        estimator = DummyEstimator()
        X = np.random.randn(5, 4, 3)
        weights = np.abs(np.random.randn(5, 4))

        with pytest.raises(ValueError, match="cs_weights cannot be provided without y"):
            validate_cross_sectional_data(estimator, X=X, cs_weights=weights)

    def test_uniform_weights_when_cs_weights_is_none(self):
        """Test that uniform weights are returned when cs_weights=None."""
        estimator = DummyEstimator()
        X = np.random.randn(3, 2, 4)
        y = np.random.randn(3, 2)

        X_val, _, w_val = validate_cross_sectional_data(
            estimator, X=X, y=y, cs_weights=None
        )

        assert X_val.shape == (3, 2, 4)
        assert w_val.shape == (3, 2)
        assert w_val.dtype == np.float64
        np.testing.assert_allclose(w_val, np.ones((3, 2)))

    def test_reset_false_does_not_update_attributes(self):
        """Test that reset=False validates consistency without updating."""
        estimator = DummyEstimator()
        X1 = np.random.randn(10, 5, 3)
        X2 = np.random.randn(8, 4, 3)

        validate_cross_sectional_data(estimator, X=X1, reset=True)
        assert estimator.n_features_in_ == 3

        validate_cross_sectional_data(estimator, X=X2, reset=False)
        assert estimator.n_features_in_ == 3

        X3 = np.random.randn(8, 4, 2)
        with pytest.raises(ValueError, match="expecting 3 features as input"):
            validate_cross_sectional_data(estimator, X=X3, reset=False)

    def test_allows_nan_values(self):
        """Test that NaN values are allowed in X and y."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 5)

        X[0, 0, 0] = np.nan
        y[1, 1] = np.nan

        X_val, y_val, _ = validate_cross_sectional_data(estimator, X=X, y=y)

        assert np.isnan(X_val[0, 0, 0])
        assert np.isnan(y_val[1, 1])

    def test_infinite_values_raise_error(self):
        """Test that infinite values are rejected in X and y."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 5)

        X[0, 0, 0] = np.inf
        with pytest.raises(ValueError, match="contains infinity"):
            validate_cross_sectional_data(estimator, X=X, y=y)

        X[0, 0, 0] = 0.0
        y[1, 1] = -np.inf
        with pytest.raises(ValueError, match="contains infinity"):
            validate_cross_sectional_data(estimator, X=X, y=y)

    def test_X_not_3d_raises_error(self):
        """Test that 2D array for X raises error."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5)

        with pytest.raises(ValueError, match="must be a 3D array"):
            validate_cross_sectional_data(estimator, X=X)

    def test_X_4d_raises_error(self):
        """Test that 4D array for X raises error."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3, 2)

        with pytest.raises(ValueError, match="must be a 3D array"):
            validate_cross_sectional_data(estimator, X=X)

    def test_y_not_2d_raises_error(self):
        """Test that 1D array for y raises error."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10)

        with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
            validate_cross_sectional_data(estimator, X=X, y=y)

    def test_incompatible_shapes_raises_error(self):
        """Test that incompatible X and y shapes raise error."""
        estimator = DummyEstimator()
        X = np.random.randn(10, 5, 3)
        y = np.random.randn(10, 6)

        with pytest.raises(ValueError, match="y must have shape"):
            validate_cross_sectional_data(estimator, X=X, y=y)

    def test_list_input_converted_to_array(self):
        """Test that list inputs are converted to arrays."""
        estimator = DummyEstimator()
        X_list = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        y_list = [[1, 2], [3, 4]]

        X_val, y_val, w_val = validate_cross_sectional_data(
            estimator, X=X_list, y=y_list
        )

        assert isinstance(X_val, np.ndarray)
        assert isinstance(y_val, np.ndarray)
        assert isinstance(w_val, np.ndarray)
        assert X_val.shape == (2, 2, 2)
        assert y_val.shape == (2, 2)
        assert w_val.shape == (2, 2)

    def test_dtype_conversion(self):
        """Test that inputs are validated and converted to numeric dtype."""
        estimator = DummyEstimator()
        X = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=float)
        y = np.array([[1.0, 2.0]], dtype=float)

        X_val, y_val, _ = validate_cross_sectional_data(estimator, X=X, y=y)

        assert np.issubdtype(X_val.dtype, np.floating)
        assert np.issubdtype(y_val.dtype, np.floating)

    def test_multiple_observations_different_sizes(self):
        """Test with various observation/asset/feature combinations."""
        test_cases = [
            (5, 3, 2),
            (100, 50, 10),
            (1, 1, 1),
            (20, 10, 5),
        ]

        for n_obs, n_assets, n_features in test_cases:
            estimator = DummyEstimator()
            X = np.random.randn(n_obs, n_assets, n_features)
            y = np.random.randn(n_obs, n_assets)

            X_val, y_val, _ = validate_cross_sectional_data(estimator, X=X, y=y)

            assert X_val.shape == (n_obs, n_assets, n_features)
            assert y_val.shape == (n_obs, n_assets)
            assert estimator.n_features_in_ == n_features

    def test_consecutive_calls_with_different_shapes(self):
        """Test behavior with consecutive calls having different input shapes."""
        estimator = DummyEstimator()

        X1 = np.random.randn(10, 5, 3)
        validate_cross_sectional_data(estimator, X=X1, reset=True)
        assert estimator.n_features_in_ == 3

        X2 = np.random.randn(8, 4, 7)
        validate_cross_sectional_data(estimator, X=X2, reset=True)
        assert estimator.n_features_in_ == 7

        X3 = np.random.randn(5, 3, 2)
        with pytest.raises(ValueError, match="expecting 7 features as input"):
            validate_cross_sectional_data(estimator, X=X3, reset=False)
        assert estimator.n_features_in_ == 7

    def test_copy_false_no_copy_if_correct_dtype(self):
        """Test that copy=False doesn't copy if dtype is already correct."""
        estimator = DummyEstimator()
        X = np.random.randn(5, 10, 3).astype(np.float64)
        y = np.random.randn(5, 10).astype(np.float64)

        X_validated, y_validated, _ = validate_cross_sectional_data(
            estimator, X=X, y=y, copy=False
        )

        assert np.shares_memory(X, X_validated)
        assert np.shares_memory(y, y_validated)

    def test_copy_false_preserves_integer_dtype(self):
        """Test that copy=False with integer dtype doesn't convert or copy."""
        estimator = DummyEstimator()
        X = np.random.randint(0, 10, (5, 10, 3), dtype=np.int64)
        y = np.random.randint(0, 10, (5, 10), dtype=np.int64)

        X_validated, y_validated, _ = validate_cross_sectional_data(
            estimator, X=X, y=y, copy=False
        )

        assert np.shares_memory(X, X_validated)
        assert np.shares_memory(y, y_validated)
        assert X_validated.dtype == np.int64
        assert y_validated.dtype == np.int64

    def test_copy_true_always_copies(self):
        """Test that copy=True always creates a copy."""
        estimator = DummyEstimator()
        X = np.random.randn(5, 10, 3).astype(np.float64)
        y = np.random.randn(5, 10).astype(np.float64)

        X_validated, y_validated, _ = validate_cross_sectional_data(
            estimator, X=X, y=y, copy=True
        )

        assert not np.shares_memory(X, X_validated)
        assert not np.shares_memory(y, y_validated)
        np.testing.assert_array_equal(X, X_validated)
        np.testing.assert_array_equal(y, y_validated)

    def test_copy_only_X(self):
        """Test copy parameter when validating only X."""
        estimator = DummyEstimator()
        X = np.random.randn(5, 10, 3).astype(np.float64)

        X_val1 = validate_cross_sectional_data(estimator, X=X, copy=False)
        assert np.shares_memory(X, X_val1)

        X_val2 = validate_cross_sectional_data(estimator, X=X, copy=True)
        assert not np.shares_memory(X, X_val2)
        np.testing.assert_array_equal(X, X_val2)

    def test_negative_weights_raise_error(self):
        """Test that negative cs_weights raise an error."""
        estimator = DummyEstimator()
        X = np.random.randn(3, 5, 2)
        y = np.random.randn(3, 5)
        weights = np.abs(np.random.randn(3, 5))
        weights[0, 0] = -1.0

        with pytest.raises(ValueError, match="Negative values"):
            validate_cross_sectional_data(estimator, X=X, y=y, cs_weights=weights)

    def test_nan_weights_raise_error(self):
        """Test that NaN cs_weights raise an error."""
        estimator = DummyEstimator()
        X = np.random.randn(3, 5, 2)
        y = np.random.randn(3, 5)
        weights = np.abs(np.random.randn(3, 5))
        weights[0, 0] = np.nan

        with pytest.raises(ValueError, match="NaN"):
            validate_cross_sectional_data(estimator, X=X, y=y, cs_weights=weights)

    def test_inf_weights_raise_error(self):
        """Test that infinite cs_weights raise an error."""
        estimator = DummyEstimator()
        X = np.random.randn(3, 5, 2)
        y = np.random.randn(3, 5)
        weights = np.abs(np.random.randn(3, 5))
        weights[0, 0] = np.inf

        with pytest.raises(ValueError, match="infinity"):
            validate_cross_sectional_data(estimator, X=X, y=y, cs_weights=weights)

    def test_wrong_weight_shape_raises_error(self):
        """Test that incorrect weight shape raises an error."""
        estimator = DummyEstimator()
        X = np.random.randn(3, 5, 2)
        y = np.random.randn(3, 5)
        weights = np.abs(np.random.randn(3, 6))

        with pytest.raises(ValueError, match="must have shape"):
            validate_cross_sectional_data(estimator, X=X, y=y, cs_weights=weights)

    def test_X_y_and_weights(self):
        """Test returning all three: X, y, and cs_weights."""
        estimator = DummyEstimator()
        X = np.random.randn(5, 4, 3)
        y = np.random.randn(5, 4)
        weights = np.abs(np.random.randn(5, 4))

        X_val, y_val, w_val = validate_cross_sectional_data(
            estimator, X=X, y=y, cs_weights=weights
        )

        assert X_val.shape == (5, 4, 3)
        assert y_val.shape == (5, 4)
        assert w_val.shape == (5, 4)
        np.testing.assert_allclose(w_val, weights)
