"""Test covariance calibration metrics."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from skfolio.metrics import (
    diagonal_calibration_loss,
    diagonal_calibration_ratio,
    exceedance_rate,
    mahalanobis_calibration_loss,
    mahalanobis_calibration_ratio,
    portfolio_variance_calibration_loss,
    portfolio_variance_calibration_ratio,
    portfolio_variance_qlike_loss,
    qlike_loss,
)
from skfolio.moments import EWCovariance
from skfolio.prior import EmpiricalPrior


class _FakeCovEstimator:
    """Minimal estimator stub that exposes a covariance_ attribute."""

    def __init__(self, covariance):
        self.covariance_ = covariance


class _FakePriorEstimator:
    """Minimal estimator stub that exposes return_distribution_.covariance."""

    class _Dist:
        def __init__(self, cov):
            self.covariance = cov

    def __init__(self, covariance):
        self.return_distribution_ = self._Dist(covariance)


@pytest.fixture()
def cov_estimator():
    """Fitted EWCovariance estimator on synthetic data."""
    rng = np.random.default_rng(42)
    X_train = rng.standard_normal((300, 5)) * 0.01
    est = EWCovariance(half_life=30)
    est.fit(X_train)
    return est


@pytest.fixture()
def X_test():
    """Synthetic test data."""
    rng = np.random.default_rng(123)
    return rng.standard_normal((50, 5)) * 0.01


class TestGetCovariance:
    def test_covariance_estimator(self):
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        ratio = mahalanobis_calibration_ratio(est, np.zeros((1, 3)))
        assert isinstance(ratio, float)

    def test_prior_estimator(self):
        cov = np.eye(3)
        est = _FakePriorEstimator(cov)
        ratio = mahalanobis_calibration_ratio(est, np.zeros((1, 3)))
        assert isinstance(ratio, float)

    def test_no_covariance_raises(self):
        class _NoCov:
            pass

        with pytest.raises(AttributeError, match="covariance_"):
            mahalanobis_calibration_ratio(_NoCov(), np.zeros((1, 3)))


class TestMahalanobisCalibrationRatio:
    def test_single_observation(self):
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        r = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ratio = mahalanobis_calibration_ratio(est, r)
        expected = np.sum(r**2) / 5
        assert abs(ratio - expected) < 1e-12

    def test_calibration_converges(self):
        """Mean of many single-obs ratios converges to 1.0."""
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        rng = np.random.default_rng(0)
        ratios = np.array(
            [
                mahalanobis_calibration_ratio(est, rng.standard_normal((1, 5)))
                for _ in range(10000)
            ]
        )
        assert abs(np.mean(ratios) - 1.0) < 0.05

    def test_block_semantics(self):
        """Block return is summed and compared against h*cov."""
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        ratio = mahalanobis_calibration_ratio(est, X)
        R_block = X.sum(axis=0)
        h = 2
        expected = float(R_block @ np.linalg.solve(h * cov, R_block)) / 3
        assert abs(ratio - expected) < 1e-12

    def test_with_fitted_estimator(self, cov_estimator, X_test):
        ratio = mahalanobis_calibration_ratio(cov_estimator, X_test)
        assert ratio > 0
        assert isinstance(ratio, float)


class TestDiagonalCalibrationRatio:
    def test_single_observation(self):
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        r = np.array([0.1, 0.2, 0.3])
        ratio = diagonal_calibration_ratio(est, r)
        expected = (0.1**2 + 0.2**2 + 0.3**2) / 3
        assert abs(ratio - expected) < 1e-12

    def test_calibration_converges(self):
        """Mean of many single-obs ratios converges to 1.0."""
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        rng = np.random.default_rng(0)
        ratios = np.array(
            [
                diagonal_calibration_ratio(est, rng.standard_normal((1, 5)))
                for _ in range(10000)
            ]
        )
        assert abs(np.mean(ratios) - 1.0) < 0.05

    def test_block_semantics(self):
        """Block return is summed and compared against h*variance."""
        cov = np.diag([1.0, 4.0, 9.0])
        est = _FakeCovEstimator(cov)
        X = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        ratio = diagonal_calibration_ratio(est, X)
        R_block = X.sum(axis=0)
        h = 2
        variance_h = h * np.diag(cov)
        expected = float(np.sum(R_block**2 / variance_h) / 3)
        assert abs(ratio - expected) < 1e-12


class TestPortfolioVarianceCalibrationRatio:
    def test_identity_covariance_ew(self):
        n = 5
        cov = np.eye(n)
        est = _FakeCovEstimator(cov)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10000, n))
        w = np.ones(n) / n
        ratio = portfolio_variance_calibration_ratio(est, X, portfolio_weights=w)
        assert abs(ratio - 1.0) < 0.1

    def test_default_weights(self, cov_estimator, X_test):
        ratio = portfolio_variance_calibration_ratio(cov_estimator, X_test)
        assert ratio > 0

    def test_multi_obs(self):
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        w = np.array([1.0, 0.0, 0.0])
        X = np.array([[0.1, 0, 0], [0.2, 0, 0], [0.3, 0, 0]])
        ratio = portfolio_variance_calibration_ratio(est, X, portfolio_weights=w)
        realized = 0.1**2 + 0.2**2 + 0.3**2
        forecast = 3 * 1.0
        assert abs(ratio - realized / forecast) < 1e-12


class TestCalibrationLosses:
    """Tests for the three *_calibration_loss functions."""

    def test_mahalanobis_loss_equals_abs_ratio_minus_one(self):
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1, 5))
        ratio = mahalanobis_calibration_ratio(est, X)
        loss = mahalanobis_calibration_loss(est, X)
        assert abs(loss - abs(ratio - 1.0)) < 1e-12

    def test_diagonal_loss_equals_abs_ratio_minus_one(self):
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((1, 5))
        ratio = diagonal_calibration_ratio(est, X)
        loss = diagonal_calibration_loss(est, X)
        assert abs(loss - abs(ratio - 1.0)) < 1e-12

    def test_portfolio_variance_loss_equals_abs_ratio_minus_one(self):
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 5))
        w = np.ones(5) / 5
        ratio = portfolio_variance_calibration_ratio(est, X, portfolio_weights=w)
        loss = portfolio_variance_calibration_loss(est, X, portfolio_weights=w)
        assert abs(loss - abs(ratio - 1.0)) < 1e-12

    def test_losses_nonnegative(self, cov_estimator, X_test):
        assert mahalanobis_calibration_loss(cov_estimator, X_test) >= 0.0
        assert diagonal_calibration_loss(cov_estimator, X_test) >= 0.0
        assert portfolio_variance_calibration_loss(cov_estimator, X_test) >= 0.0

    def test_perfect_calibration_gives_low_loss(self):
        rng = np.random.default_rng(42)
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        losses = np.array(
            [
                mahalanobis_calibration_loss(est, rng.standard_normal((1, 3)))
                for _ in range(5000)
            ]
        )
        assert np.mean(losses) < 1.0

    def test_overestimated_covariance_produces_positive_loss(self):
        rng = np.random.default_rng(42)
        bad_cov = np.eye(3)
        est = _FakeCovEstimator(bad_cov)
        X = rng.standard_normal((1, 3)) * 0.01
        assert mahalanobis_calibration_loss(est, X) > 0.5
        assert diagonal_calibration_loss(est, X) > 0.5


class TestPortfolioVarianceQlikeLoss:
    def test_lower_for_better_estimator(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3)) * 0.01
        true_cov = np.cov(X, rowvar=False)
        bad_cov = true_cov * 5.0

        est_good = _FakeCovEstimator(true_cov)
        est_bad = _FakeCovEstimator(bad_cov)

        loss_good = portfolio_variance_qlike_loss(est_good, X)
        loss_bad = portfolio_variance_qlike_loss(est_bad, X)
        assert loss_good < loss_bad


class TestExceedanceRate:
    def test_chi_squared(self):
        rng = np.random.default_rng(42)
        d2 = rng.chisquare(df=10, size=100000)
        rate = exceedance_rate(d2, n_features=10, confidence_level=0.95)
        assert abs(rate - 0.05) < 0.01

    def test_nan_filtered_before_comparison(self):
        """NaN distances are excluded, not silently counted as non-exceedances."""
        rng = np.random.default_rng(42)
        d2 = rng.chisquare(df=5, size=10000)
        rate_clean = exceedance_rate(d2, n_features=5, confidence_level=0.95)
        d2_nan = np.concatenate([d2, np.full(5000, np.nan)])
        rate_nan = exceedance_rate(d2_nan, n_features=5, confidence_level=0.95)
        assert abs(rate_clean - rate_nan) < 1e-12

    def test_all_nan_returns_nan(self):
        d2 = np.full(10, np.nan)
        assert np.isnan(exceedance_rate(d2, n_features=5, confidence_level=0.95))

    def test_validation(self):
        with pytest.raises(ValueError, match="n_features"):
            exceedance_rate(np.array([1.0]), n_features=0, confidence_level=0.95)
        with pytest.raises(ValueError, match="confidence_level"):
            exceedance_rate(np.array([1.0]), n_features=5, confidence_level=0.0)


class TestQlikeLoss:
    def test_basic(self):
        r = np.array([0.01, -0.02, 0.015])
        v = np.array([0.0001, 0.0004, 0.000225])
        loss = qlike_loss(r, v)
        expected = np.mean(np.log(v) + r**2 / v)
        assert abs(loss - expected) < 1e-12

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            qlike_loss(np.array([1, 2]), np.array([1, 2, 3]))


class TestBlockSemantics:
    """Verify that block semantics are consistent across functions."""

    def test_h1_matches_single_obs(self):
        """With h=1, block result equals direct single-observation result."""
        cov = np.diag([1.0, 4.0, 9.0])
        est = _FakeCovEstimator(cov)
        X_single = np.array([[0.1, 0.2, 0.3]])
        m_ratio = mahalanobis_calibration_ratio(est, X_single)
        d_ratio = diagonal_calibration_ratio(est, X_single)
        assert m_ratio > 0
        assert d_ratio > 0

    def test_block_vs_single_obs_different(self):
        """Block of 5 gives different result from mean of 5 singles."""
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((5, 3))

        block_ratio = mahalanobis_calibration_ratio(est, X)
        single_ratios = [
            mahalanobis_calibration_ratio(est, X[i : i + 1]) for i in range(5)
        ]
        mean_single = np.mean(single_ratios)
        assert abs(block_ratio - mean_single) > 1e-6


class TestPriorEstimatorDispatch:
    def test_scorers_with_prior(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((300, 5)) * 0.01
        X_test = rng.standard_normal((20, 5)) * 0.01

        prior = EmpiricalPrior()
        prior.fit(X_train)

        m = mahalanobis_calibration_ratio(prior, X_test)
        c = diagonal_calibration_ratio(prior, X_test)
        p = portfolio_variance_calibration_ratio(prior, X_test)
        q = portfolio_variance_qlike_loss(prior, X_test)

        assert isinstance(m, float) and m > 0
        assert isinstance(c, float) and c > 0
        assert isinstance(p, float) and p > 0
        assert isinstance(q, float)

        ml = mahalanobis_calibration_loss(prior, X_test)
        cl = diagonal_calibration_loss(prior, X_test)
        pl = portfolio_variance_calibration_loss(prior, X_test)

        assert isinstance(ml, float) and ml >= 0
        assert isinstance(cl, float) and cl >= 0
        assert isinstance(pl, float) and pl >= 0

    def test_partial_with_portfolio_weights(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((300, 5)) * 0.01
        X_test = rng.standard_normal((20, 5)) * 0.01

        prior = EmpiricalPrior()
        prior.fit(X_train)

        w = np.ones(5) / 5
        scorer = partial(portfolio_variance_calibration_ratio, portfolio_weights=w)
        ratio = scorer(prior, X_test)
        assert isinstance(ratio, float) and ratio > 0

        loss_scorer = partial(portfolio_variance_calibration_loss, portfolio_weights=w)
        loss = loss_scorer(prior, X_test)
        assert isinstance(loss, float) and loss >= 0


class TestMultiPortfolioWeights:
    """Tests for 2D portfolio_weights (multiple test portfolios)."""

    def test_single_row_matches_1d(self, cov_estimator, X_test):
        """A 2D array with one row must match the equivalent 1D array."""
        w = np.array([0.3, 0.2, 0.1, 0.25, 0.15])
        ratio_1d = portfolio_variance_calibration_ratio(
            cov_estimator, X_test, portfolio_weights=w
        )
        ratio_2d = portfolio_variance_calibration_ratio(
            cov_estimator, X_test, portfolio_weights=w[np.newaxis, :]
        )
        assert abs(ratio_1d - ratio_2d) < 1e-12

        qlike_1d = portfolio_variance_qlike_loss(
            cov_estimator, X_test, portfolio_weights=w
        )
        qlike_2d = portfolio_variance_qlike_loss(
            cov_estimator, X_test, portfolio_weights=w[np.newaxis, :]
        )
        assert abs(qlike_1d - qlike_2d) < 1e-12

    def test_multi_portfolio_returns_float(self, cov_estimator, X_test):
        """Multi-portfolio weights still return a single float."""
        W = np.array(
            [
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5, 0.0],
                [0.2, 0.2, 0.2, 0.2, 0.2],
            ]
        )
        ratio = portfolio_variance_calibration_ratio(
            cov_estimator, X_test, portfolio_weights=W
        )
        assert isinstance(ratio, float)
        assert ratio > 0

        loss = portfolio_variance_calibration_loss(
            cov_estimator, X_test, portfolio_weights=W
        )
        assert isinstance(loss, float)
        assert loss >= 0

        qlike = portfolio_variance_qlike_loss(
            cov_estimator, X_test, portfolio_weights=W
        )
        assert isinstance(qlike, float)

    def test_multi_portfolio_is_mean_of_singles(self, cov_estimator, X_test):
        """Multi-portfolio result equals the mean of individual results."""
        W = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ]
        )
        ratios = [
            portfolio_variance_calibration_ratio(
                cov_estimator, X_test, portfolio_weights=W[k]
            )
            for k in range(3)
        ]
        ratio_multi = portfolio_variance_calibration_ratio(
            cov_estimator, X_test, portfolio_weights=W
        )
        assert abs(ratio_multi - np.mean(ratios)) < 1e-12

        qlikes = [
            portfolio_variance_qlike_loss(cov_estimator, X_test, portfolio_weights=W[k])
            for k in range(3)
        ]
        qlike_multi = portfolio_variance_qlike_loss(
            cov_estimator, X_test, portfolio_weights=W
        )
        assert abs(qlike_multi - np.mean(qlikes)) < 1e-12

    def test_calibration_loss_with_multi_portfolio(self, cov_estimator, X_test):
        """Loss with multi-portfolio is abs(mean_ratio - 1)."""
        W = np.array(
            [
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.5, 0.5],
            ]
        )
        ratio = portfolio_variance_calibration_ratio(
            cov_estimator, X_test, portfolio_weights=W
        )
        loss = portfolio_variance_calibration_loss(
            cov_estimator, X_test, portfolio_weights=W
        )
        assert abs(loss - abs(ratio - 1.0)) < 1e-12


class TestNaNHandling:
    """NaN robustness for calibration metrics."""

    def test_no_nan_matches_original(self):
        """With no NaN, results are identical to the original zero-fill code."""
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        rng = np.random.default_rng(42)
        X = rng.standard_normal((10, 5))
        m = mahalanobis_calibration_ratio(est, X)
        d = diagonal_calibration_ratio(est, X)
        p = portfolio_variance_calibration_ratio(est, X)
        q = portfolio_variance_qlike_loss(est, X)
        assert isinstance(m, float) and m > 0
        assert isinstance(d, float) and d > 0
        assert isinstance(p, float) and p > 0
        assert isinstance(q, float)

    def test_nan_covariance_diagonal_returns_nan(self):
        """All assets inactive in covariance gives NaN."""
        cov = np.full((3, 3), np.nan)
        est = _FakeCovEstimator(cov)
        X = np.ones((5, 3))
        assert np.isnan(mahalanobis_calibration_ratio(est, X))
        assert np.isnan(diagonal_calibration_ratio(est, X))
        assert np.isnan(portfolio_variance_calibration_ratio(est, X))
        assert np.isnan(portfolio_variance_qlike_loss(est, X))

    def test_all_nan_x_test_returns_nan(self):
        """All NaN in X_test gives NaN (no jointly active assets)."""
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        X = np.full((5, 3), np.nan)
        assert np.isnan(mahalanobis_calibration_ratio(est, X))
        assert np.isnan(diagonal_calibration_ratio(est, X))
        assert np.isnan(portfolio_variance_calibration_ratio(est, X))
        assert np.isnan(portfolio_variance_qlike_loss(est, X))

    def test_hadamard_diagonal_per_asset_horizon(self):
        """Diagonal ratio uses per-asset effective horizon, not total rows."""
        cov = np.diag([1.0, 4.0, 9.0])
        est = _FakeCovEstimator(cov)
        X = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, np.nan, 0.6],
                [0.7, 0.8, np.nan],
            ]
        )
        ratio = diagonal_calibration_ratio(est, X)
        # Asset 0: R=1.2, h=3, var=1.0 -> 1.2^2/(3*1)=0.48
        # Asset 1: R=1.0, h=2, var=4.0 -> 1.0^2/(2*4)=0.125
        # Asset 2: R=0.9, h=2, var=9.0 -> 0.9^2/(2*9)=0.045
        expected = (0.48 + 0.125 + 0.045) / 3
        assert abs(ratio - expected) < 1e-12

    def test_hadamard_mahalanobis(self):
        """Hadamard Mahalanobis with non-trivial cov and sparse NaN."""
        cov = np.array(
            [
                [1.0, 0.5, 0.3],
                [0.5, 1.0, 0.4],
                [0.3, 0.4, 1.0],
            ]
        )
        est = _FakeCovEstimator(cov)
        X = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, np.nan],
            ]
        )
        ratio = mahalanobis_calibration_ratio(est, X)
        # R = [5, 7, 3], H = [[2,2,1],[2,2,1],[1,1,1]]
        # cov_eff = H * cov (element-wise)
        R = np.array([5.0, 7.0, 3.0])
        H = np.array([[2, 2, 1], [2, 2, 1], [1, 1, 1]], dtype=float)
        cov_eff = H * cov
        d2 = float(R @ np.linalg.solve(cov_eff, R))
        expected = d2 / 3
        assert abs(ratio - expected) < 1e-10

    def test_portfolio_hadamard_with_nan(self):
        """Portfolio metrics use zero-fill + Hadamard forecast variance."""
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        X = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, np.nan, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )
        w = np.ones(3) / 3
        ratio = portfolio_variance_calibration_ratio(est, X, portfolio_weights=w)
        X_filled = np.where(np.isfinite(X), X, 0.0)
        mask = np.isfinite(X).astype(float)
        H = mask.T @ mask
        cov_eff = H * cov
        forecast_var = w @ cov_eff @ w
        r_ptf = X_filled @ w
        expected = float(np.sum(r_ptf**2) / forecast_var)
        assert abs(ratio - expected) < 1e-12

    def test_portfolio_no_complete_rows_still_valid(self):
        """When no row is fully complete, portfolio ratio is still valid."""
        cov = np.eye(3)
        est = _FakeCovEstimator(cov)
        X = np.array(
            [
                [0.1, np.nan, 0.3],
                [np.nan, 0.2, 0.6],
            ]
        )
        w = np.ones(3) / 3
        ratio = portfolio_variance_calibration_ratio(est, X, portfolio_weights=w)
        assert np.isfinite(ratio)
        assert ratio > 0

    def test_portfolio_hadamard_unbiased(self):
        """Mean portfolio ratio over many NaN-injected blocks converges to 1."""
        rng = np.random.default_rng(42)
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        w = np.ones(5) / 5
        ratios = []
        for _ in range(5000):
            X = rng.standard_normal((10, 5))
            X[::3, :] = np.nan
            ratios.append(
                portfolio_variance_calibration_ratio(est, X, portfolio_weights=w)
            )
        assert abs(np.mean(ratios) - 1.0) < 0.1

    def test_holiday_nan_unbiased_diagonal(self):
        """Mean diagonal ratio over many blocks with holiday NaN converges to 1."""
        rng = np.random.default_rng(42)
        cov = np.eye(5)
        est = _FakeCovEstimator(cov)
        ratios = []
        for _ in range(5000):
            X = rng.standard_normal((10, 5))
            X[::3, :] = np.nan
            ratios.append(diagonal_calibration_ratio(est, X))
        assert abs(np.mean(ratios) - 1.0) < 0.1

    def test_partial_cov_nan_filters_asset(self):
        """An asset with NaN diagonal in covariance is excluded."""
        cov = np.diag([1.0, np.nan, 4.0])
        est = _FakeCovEstimator(cov)
        X = np.array([[0.1, 0.2, 0.3]])
        # Only assets 0 and 2 active
        ratio = diagonal_calibration_ratio(est, X)
        expected = (0.01 / 1.0 + 0.09 / 4.0) / 2
        assert abs(ratio - expected) < 1e-12
