"""Tests for EmpiricalVariance estimator."""

import numpy as np
import pytest

from skfolio.moments import EmpiricalCovariance, EmpiricalVariance


class TestEmpiricalVariance:
    def test_basic(self, X):
        """Test basic functionality with default parameters."""
        model = EmpiricalVariance()
        model.fit(X)

        assert model.variance_.shape == (20,)
        assert model.location_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)

    def test_matches_numpy(self, X):
        """Test that variance matches numpy's variance calculation."""
        model = EmpiricalVariance(ddof=1)
        model.fit(X)

        expected_variance = np.var(X, axis=0, ddof=1)
        expected_mean = np.asarray(X).mean(axis=0)

        np.testing.assert_allclose(model.variance_, expected_variance, rtol=1e-10)
        np.testing.assert_allclose(model.location_, expected_mean, rtol=1e-10)

    def test_ddof_zero(self, X):
        """Test with ddof=0 (biased estimator)."""
        model = EmpiricalVariance(ddof=0)
        model.fit(X)

        expected_variance = np.var(X, axis=0, ddof=0)
        np.testing.assert_allclose(model.variance_, expected_variance, rtol=1e-10)

    def test_invalid_ddof(self):
        """Invalid ddof values raise ValueError."""
        X = np.random.randn(5, 3)

        with pytest.raises(ValueError, match="ddof must be a non-negative integer"):
            EmpiricalVariance(ddof=-1).fit(X)

        with pytest.raises(ValueError, match="ddof must be a non-negative integer"):
            EmpiricalVariance(ddof=1.5).fit(X)

        with pytest.raises(
            ValueError,
            match="ddof must be strictly less than the number of observations",
        ):
            EmpiricalVariance(ddof=5).fit(X)

    def test_assume_centered_true(self, X):
        """Test with assume_centered=True."""
        model = EmpiricalVariance(assume_centered=True)
        model.fit(X)

        X_arr = np.asarray(X)
        n_obs = X_arr.shape[0]
        expected_variance = np.sum(X_arr**2, axis=0) / (n_obs - 1)

        np.testing.assert_allclose(model.variance_, expected_variance, rtol=1e-10)
        np.testing.assert_array_equal(model.location_, np.zeros(20))

    def test_assume_centered_produces_different_results(self, X):
        """Test that assume_centered produces different results."""
        model_centered = EmpiricalVariance(assume_centered=True)
        model_not_centered = EmpiricalVariance(assume_centered=False)

        model_centered.fit(X)
        model_not_centered.fit(X)

        # Results should differ (unless data is truly centered)
        assert not np.allclose(model_centered.variance_, model_not_centered.variance_)

    def test_window_size(self, X):
        """Test with window_size parameter."""
        model_full = EmpiricalVariance()
        model_window = EmpiricalVariance(window_size=50)

        model_full.fit(X)
        model_window.fit(X)

        # Window model should use only last 50 observations
        X_arr = np.asarray(X)
        expected_variance = np.var(X_arr[-50:], axis=0, ddof=1)
        np.testing.assert_allclose(
            model_window.variance_, expected_variance, rtol=1e-10
        )

        # Results should differ from full data
        assert not np.allclose(model_full.variance_, model_window.variance_)

    def test_sklearn_api(self, X):
        """Test sklearn-compatible API."""
        model = EmpiricalVariance()

        # fit should return self
        result = model.fit(X)
        assert result is model

        # n_features_in_ should be set
        assert model.n_features_in_ == 20

    def test_single_asset(self):
        """Test with single asset."""
        np.random.seed(42)
        X_single = np.random.randn(100, 1) * 0.02

        model = EmpiricalVariance()
        model.fit(X_single)

        assert model.variance_.shape == (1,)
        assert model.location_.shape == (1,)
        np.testing.assert_allclose(
            model.variance_, np.var(X_single, axis=0, ddof=1), rtol=1e-10
        )

    def test_exact_values(self, X):
        """Regression test with exact expected values."""
        model = EmpiricalVariance()
        model.fit(X)

        expected_variance = np.array(
            [0.00033685, 0.00139232, 0.0003929, 0.00061523, 0.00036072]
        )
        expected_location = np.array(
            [0.00104344, 0.00190157, 0.00058076, 0.00073676, 0.00050673]
        )

        np.testing.assert_allclose(model.variance_[:5], expected_variance, rtol=1e-5)
        np.testing.assert_allclose(model.location_[:5], expected_location, rtol=1e-5)

    def test_matches_empirical_covariance_diagonal(self, X):
        """Variance matches diagonal of EmpiricalCovariance."""
        model_var = EmpiricalVariance(ddof=1, assume_centered=False)
        model_var.fit(X)

        model_cov = EmpiricalCovariance(ddof=1, assume_centered=False, nearest=False)
        model_cov.fit(X)

        np.testing.assert_allclose(
            model_var.variance_, np.diag(model_cov.covariance_), rtol=1e-10
        )
        np.testing.assert_allclose(model_var.location_, model_cov.location_, rtol=1e-10)

    def test_matches_empirical_covariance_diagonal_centered(self, X):
        """Centered variance matches diagonal of centered EmpiricalCovariance."""
        model_var = EmpiricalVariance(ddof=1, assume_centered=True)
        model_var.fit(X)

        model_cov = EmpiricalCovariance(ddof=1, assume_centered=True, nearest=False)
        model_cov.fit(X)

        np.testing.assert_allclose(
            model_var.variance_, np.diag(model_cov.covariance_), rtol=1e-10
        )
        np.testing.assert_array_equal(model_var.location_, model_cov.location_)
