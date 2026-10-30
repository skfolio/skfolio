"""Tests for BaseOnlineOptimization base class: predict, fit_predict, predict_online.

Uses FollowTheWinner(strategy='eg') as the lightest concrete subclass.
Structured in TDD order: happy paths first, edge cases second.
"""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from skfolio.optimization.online._winner import FollowTheWinner
from skfolio.portfolio import (
    MultiPeriodPortfolio,
    Portfolio,
)


@pytest.fixture
def X_synthetic():
    """Small synthetic net-returns matrix: 30 periods, 3 assets."""
    rng = np.random.default_rng(42)
    return rng.normal(0.001, 0.02, size=(30, 3))


@pytest.fixture
def estimator():
    """Simplest online estimator: Exponentiated Gradient."""
    return FollowTheWinner(strategy="eg")


class TestPredict:
    """Tests for the inherited BaseOptimization.predict method."""

    def test_returns_portfolio(self, estimator, X_synthetic):
        model = estimator
        model.fit(X_synthetic)
        result = model.predict(X_synthetic)
        assert isinstance(result, Portfolio)

    def test_weights_shape(self, estimator, X_synthetic):
        model = estimator
        model.fit(X_synthetic)
        result = model.predict(X_synthetic)
        assert result.weights.shape == (X_synthetic.shape[1],)

    def test_weights_sum_to_budget(self, estimator, X_synthetic):
        model = estimator
        model.fit(X_synthetic)
        result = model.predict(X_synthetic)
        assert result.weights.sum() == pytest.approx(1.0, abs=1e-6)

    def test_weights_nonnegative(self, estimator, X_synthetic):
        model = estimator
        model.fit(X_synthetic)
        result = model.predict(X_synthetic)
        assert np.all(result.weights >= -1e-10)


class TestFitPredict:
    """Tests for the BaseOnlineOptimization.fit_predict override."""

    def test_returns_mpp(self, estimator, X_synthetic):
        result = estimator.fit_predict(X_synthetic)
        assert isinstance(result, MultiPeriodPortfolio)

    def test_one_portfolio_per_period(self, estimator, X_synthetic):
        result = estimator.fit_predict(X_synthetic)
        assert len(result.portfolios) == X_synthetic.shape[0]

    def test_each_portfolio_is_portfolio(self, estimator, X_synthetic):
        result = estimator.fit_predict(X_synthetic)
        for ptf in result.portfolios:
            assert isinstance(ptf, Portfolio)

    def test_each_portfolio_has_one_observation(self, estimator, X_synthetic):
        result = estimator.fit_predict(X_synthetic)
        for ptf in result.portfolios:
            assert ptf.returns.shape[0] == 1

    def test_weights_valid_in_trajectory(self, estimator, X_synthetic):
        result = estimator.fit_predict(X_synthetic)
        for ptf in result.portfolios:
            assert ptf.weights.sum() == pytest.approx(1.0, abs=1e-6)
            assert np.all(ptf.weights >= -1e-10)

    def test_fit_predict_sets_all_weights(self, estimator, X_synthetic):
        estimator.fit_predict(X_synthetic)
        assert hasattr(estimator, "all_weights_")
        assert estimator.all_weights_.shape == X_synthetic.shape


class TestPredictOnline:
    """Tests for the BaseOnlineOptimization.predict_online method."""

    def test_returns_mpp(self, estimator, X_synthetic):
        estimator.fit(X_synthetic[:10])
        result = estimator.predict_online(X_synthetic[10:])
        assert isinstance(result, MultiPeriodPortfolio)

    def test_trajectory_length(self, estimator, X_synthetic):
        estimator.fit(X_synthetic[:10])
        X_test = X_synthetic[10:]
        result = estimator.predict_online(X_test)
        assert len(result.portfolios) == X_test.shape[0]

    def test_does_not_mutate_state(self, estimator, X_synthetic):
        estimator.fit(X_synthetic[:10])
        weights_before = estimator.weights_.copy()
        history_before = estimator.all_weights_.copy()
        gradient_sum_before = estimator._foco_engine._G_sum.copy()
        t_before = estimator._t

        estimator.predict_online(X_synthetic[10:])

        np.testing.assert_array_equal(estimator.weights_, weights_before)
        assert not hasattr(estimator, "wealth_")
        np.testing.assert_array_equal(estimator.all_weights_, history_before)
        np.testing.assert_array_equal(
            estimator._foco_engine._G_sum, gradient_sum_before
        )
        assert estimator._t == t_before

    def test_repeated_calls_give_same_result(self, estimator, X_synthetic):
        estimator.fit(X_synthetic[:10])
        X_test = X_synthetic[10:]

        result1 = estimator.predict_online(X_test)
        result2 = estimator.predict_online(X_test)

        for p1, p2 in zip(result1.portfolios, result2.portfolios, strict=True):
            np.testing.assert_array_equal(p1.weights, p2.weights)

    def test_weights_valid(self, estimator, X_synthetic):
        estimator.fit(X_synthetic[:10])
        result = estimator.predict_online(X_synthetic[10:])
        for ptf in result.portfolios:
            assert ptf.weights.sum() == pytest.approx(1.0, abs=1e-6)
            assert np.all(ptf.weights >= -1e-10)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPredictOnlineEdgeCases:
    """Edge cases and error paths for predict_online."""

    def test_unfitted_raises(self, X_synthetic):
        model = FollowTheWinner(strategy="eg")
        with pytest.raises(NotFittedError):
            model.predict_online(X_synthetic)

    def test_wrong_features_raises(self, estimator, X_synthetic):
        estimator.fit(X_synthetic[:10])
        X_bad = np.random.default_rng(0).normal(size=(5, X_synthetic.shape[1] + 1))
        with pytest.raises(ValueError, match="features"):
            estimator.predict_online(X_bad)

    def test_1d_input_raises(self, estimator, X_synthetic):
        estimator.fit(X_synthetic[:10])
        with pytest.raises(ValueError):
            estimator.predict_online(X_synthetic[10])


class TestFitPredictEdgeCases:
    """Edge cases for fit_predict."""

    def test_single_row(self, estimator):
        X_one = np.array([[0.01, -0.005, 0.003]])
        result = estimator.fit_predict(X_one)
        assert isinstance(result, MultiPeriodPortfolio)
        assert len(result.portfolios) == 1


class TestPredictEdgeCases:
    """Edge cases for the inherited predict method."""

    def test_unfitted_raises(self, X_synthetic):
        model = FollowTheWinner(strategy="eg")
        with pytest.raises(NotFittedError):
            model.predict(X_synthetic)
