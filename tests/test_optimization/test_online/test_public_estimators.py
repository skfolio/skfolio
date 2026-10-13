"""Public API, independent algorithm references and execution boundaries."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.special import softmax
from sklearn.base import clone

from skfolio.model_selection import OnlineGridSearch, online_predict
from skfolio.optimization.online import (
    AdaGrad,
    ExponentiatedGradient,
    OnlineMovingAverageReversion,
)
from skfolio.optimization.online._foco import FirstOrderOCO
from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._mirror_maps import AdaptiveMahalanobisMap
from skfolio.prior import BasePrior, ReturnDistribution


@pytest.mark.parametrize(
    "factory", [ExponentiatedGradient, AdaGrad, OnlineMovingAverageReversion]
)
def test_public_clone_reset_blocks_and_execution(factory):
    X = np.random.default_rng(261).normal(0, 0.015, (12, 3))
    model = factory()
    assert clone(model).get_params() == model.get_params()
    batch = clone(model).fit(X)
    streamed = clone(model)
    for row in X:
        streamed.partial_fit(row)
    assert_allclose(batch.weights_, streamed.weights_, atol=1e-10)
    assert batch.n_observations_ == 12
    assert not hasattr(batch, "all_weights_")
    assert not hasattr(batch, "wealth_")
    batch.fit(X[:4])
    assert_allclose(batch.weights_, clone(model).fit(X[:4]).weights_)
    evaluation = online_predict(model, X, warmup_size=4, test_size=1)
    expected = []
    for row in X[4:]:
        expected.append(row @ batch.weights_)
        batch.partial_fit(row)
    assert_allclose(evaluation.returns, expected)


@pytest.mark.parametrize("mode", ["omd", "ftrl"])
def test_eg_matches_independent_logwealth_update(mode):
    weights = np.array([0.7, 0.3])
    model = ExponentiatedGradient(
        initial_weights=weights.copy(), learning_rate=0.2, update_mode=mode
    )
    gradient_sum = np.zeros(2)
    for row in np.array([[0.2, -0.1], [-0.03, 0.15], [0.01, 0.02]]):
        gradient = -row / (1 + weights @ row)
        gradient_sum += gradient
        weights = (
            softmax(np.log(weights) - 0.2 * gradient)
            if mode == "omd"
            else softmax(-0.2 * gradient_sum)
        )
        model.partial_fit(row)
        assert_allclose(model.weights_, weights, atol=1e-12)
    assert isinstance(model._foco_engine, FirstOrderOCO)


def test_adagrad_matches_independent_weighted_simplex_update():
    weights = np.array([0.5, 0.5])
    sums = np.zeros(2)
    model = AdaGrad(learning_rate=0.01)
    for row in np.array([[0.1, -0.05], [-0.02, 0.03], [0.04, 0.02]]):
        gradient = -row / (1 + weights @ row)
        sums += gradient**2
        curvature = np.sqrt(sums + model.adagrad_eps)
        candidate = weights - 0.01 * gradient / curvature
        multiplier = (candidate.sum() - 1) / (1 / curvature).sum()
        weights = candidate - multiplier / curvature
        model.partial_fit(row)
        assert_allclose(model.weights_, weights, atol=2e-6)
    assert isinstance(model._foco_engine.map, AdaptiveMahalanobisMap)


def test_olmar_window_tuning_and_first_full_window():
    X = np.array([[1.0, -0.5], [-0.1, 0.2], [0.2, -0.1], [0.1, 0.2]])
    model = OnlineMovingAverageReversion(window=5).set_params(window=2)
    assert clone(model).window == 2
    model.partial_fit(X[:1])
    # MA/current price = (1 + initial/current price)/2 = [0.75, 1.5].
    # The PA step reaches the second-asset corner after this first full window.
    assert_allclose(model.weights_, [0, 1], atol=1e-10)
    model.fit(X)
    assert_allclose(
        model.weights_, OnlineMovingAverageReversion(window=2).fit(X).weights_
    )
    assert len(model._predictor._history) <= 1


@pytest.mark.parametrize(
    "factory", [ExponentiatedGradient, AdaGrad, OnlineMovingAverageReversion]
)
def test_named_initial_allocation_and_constraints(factory):
    X = pd.DataFrame([[0.1, -0.1, 0.02], [0.02, 0.01, -0.01]], columns=["a", "b", "c"])
    model = factory(
        initial_weights={"a": 0.2, "b": 0.3, "c": 0.5},
        linear_constraints=["a + b <= 0.7"],
        max_weights={"a": 0.5, "b": 0.5, "c": 0.8},
    ).fit(X)
    assert_allclose(model.initial_weights_, [0.2, 0.3, 0.5], atol=1e-7)
    assert model.weights_[:2].sum() <= 0.7 + 1e-7
    assert model.weights_[0] <= 0.5 + 1e-7


@pytest.mark.parametrize(
    "factory,grid",
    [
        (ExponentiatedGradient, {"learning_rate": [0.01, 0.1]}),
        (AdaGrad, {"learning_rate": [0.01, 0.1]}),
        (OnlineMovingAverageReversion, {"window": [2, 3]}),
    ],
)
def test_public_online_search(factory, grid):
    X = np.random.default_rng(17).normal(0, 0.01, (12, 3))
    result = OnlineGridSearch(
        factory(), param_grid=grid, warmup_size=4, test_size=1
    ).fit(X)
    assert result.best_estimator_.n_observations_ == len(X)
    with pytest.raises(ValueError, match="test_size=1"):
        online_predict(factory(), X, warmup_size=4, test_size=2)


class ChangingPrior(BasePrior):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        raise AssertionError("Use causal updates")

    def partial_fit(self, X, y=None):
        self.count_ = getattr(self, "count_", 0) + 1
        covariance = np.diag([1.0, 0.01]) if self.count_ > 1 else np.diag([0.01, 0.01])
        self.return_distribution_ = ReturnDistribution(
            mu=np.zeros(2), covariance=covariance, returns=np.asarray(X)
        )
        return self


@pytest.mark.parametrize("window", [2, 10])
def test_olmar_inactive_predictor_still_respects_updated_risk_limit(window):
    model = OnlineMovingAverageReversion(
        window=window, prior_estimator=ChangingPrior(), max_variance=0.1
    )
    model.partial_fit([0.0, 0.0]).partial_fit([0.0, 0.0])
    covariance = model.prior_estimator_.return_distribution_.covariance
    assert model.weights_ @ covariance @ model.weights_ <= 0.1 + 1e-7


def test_internal_mean_reversion_md_starts_from_explicit_initial_target():
    model = FollowTheLoser(
        strategy="olmar", update_mode="md", olmar_window=2, initial_weights=[0.8, 0.2]
    )
    model.fit([[0.0, 0.0]])
    assert_allclose(model.weights_, [0.8, 0.2], atol=1e-10)


@pytest.mark.parametrize(
    "factory", [ExponentiatedGradient, AdaGrad, OnlineMovingAverageReversion]
)
def test_diagnostic_loss_includes_uninvested_cash(factory):
    model = factory(budget=0.5, store_weights=True).fit([[0.1, 0.2]])
    expected = -np.log1p(model.initial_weights_ @ np.array([0.1, 0.2]))
    assert_allclose(model.loss_, expected)


@pytest.mark.parametrize(
    "factory", [ExponentiatedGradient, AdaGrad, OnlineMovingAverageReversion]
)
def test_zero_turnover_retains_exact_execution_holdings(factory):
    anchor = np.array([0.4, 0.6])
    model = factory(previous_weights=anchor, max_turnover=0, max_weights=0.8)
    model.partial_fit([0.1, -0.1])
    np.testing.assert_array_equal(model.weights_, anchor)
    assert_allclose(model.predict([[0.1, -0.1]]).transaction_costs, 0)


@pytest.mark.parametrize(
    "factory", [ExponentiatedGradient, AdaGrad, OnlineMovingAverageReversion]
)
@pytest.mark.parametrize("limits", [(None, 0.7), (1.2, None), (0.6, 0.8)])
def test_public_budget_bounds(factory, limits):
    """Budget bounds apply to initialization and every streamed target."""
    lower, upper = limits
    model = factory(budget=None, min_budget=lower, max_budget=upper, store_weights=True)
    assert clone(model).get_params() == model.get_params()
    returns = np.array([[0.02, -0.01, 0.01], [-0.01, 0.03, 0.02]])
    model.fit(returns)
    streamed = clone(model)
    for row in returns:
        streamed.partial_fit(row)
    assert_allclose(model.all_weights_, streamed.all_weights_, atol=1e-6)
    targets = np.vstack([model.initial_weights_, model.all_weights_])
    totals = targets.sum(axis=1)
    if lower is not None:
        assert np.all(totals >= lower - 1e-6)
    if upper is not None:
        assert np.all(totals <= upper + 1e-6)


@pytest.mark.parametrize(
    "factory", [ExponentiatedGradient, AdaGrad, OnlineMovingAverageReversion]
)
@pytest.mark.parametrize("bound", ["min_budget", "max_budget"])
def test_public_budget_bound_rejects_equality_budget(factory, bound):
    with pytest.raises(ValueError, match="cannot be provided at the same time"):
        factory(**{bound: 0.8}).fit([[0.01, -0.01]])


@pytest.mark.parametrize(
    "factory", [ExponentiatedGradient, AdaGrad, OnlineMovingAverageReversion]
)
@pytest.mark.parametrize(
    "bounds",
    [
        {"min_budget": 0.9, "max_budget": 0.8},
        {"min_budget": 0.9, "max_weights": 0.2},
    ],
)
def test_public_infeasible_budget_bounds(factory, bounds):
    with pytest.raises(RuntimeError, match="Convex projection failed"):
        factory(budget=None, **bounds).fit([[0.01, -0.01]])
