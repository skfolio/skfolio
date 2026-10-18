"""Tests for online hindsight benchmarks."""

import numpy as np
import pytest

from skfolio.measures import PerfMeasure, RiskMeasure
from skfolio.optimization.online._autograd_objectives import MEASURE_PROPERTIES
from skfolio.optimization.online._benchmark import (
    BestAsset,
    BestConstantRebalancedPortfolio,
    ConstantRebalancedPortfolio,
)

allowed_measures = list(MEASURE_PROPERTIES.keys())


def test_constant_rebalanced_with_custom_weights(X_small):
    """Test ConstantRebalancedPortfolio with custom weights."""
    n_assets = X_small.shape[1]
    weights = np.ones(n_assets) / n_assets
    constant_rebalanced = ConstantRebalancedPortfolio(weights=weights)
    constant_rebalanced.fit(X_small)


def test_comparator_uniform_weights(X_small):
    model = ConstantRebalancedPortfolio(weights=None).fit(X_small)
    expected_returns = np.asarray(X_small).mean(axis=1)
    np.testing.assert_allclose(model.weights_, 1 / X_small.shape[1])
    np.testing.assert_allclose(
        model.all_wealth_, np.r_[1.0, np.cumprod(1 + expected_returns)]
    )
    np.testing.assert_allclose(model.all_weights_, 1 / X_small.shape[1])
    model.partial_fit(X_small.iloc[:1])
    np.testing.assert_allclose(model.weights_, 1 / X_small.shape[1])


def test_best_asset_picks_best_performer(X_small):
    """Test BestAsset picks the asset with highest cumulative log return."""
    bs = BestAsset()
    bs.fit(X_small)

    # Verify one-hot vector
    assert np.sum(bs.weights_) == pytest.approx(1.0)
    assert np.sum(bs.weights_ > 0) == 1

    # Verify it picks the best asset
    relatives = 1.0 + X_small.values
    log_returns = np.sum(np.log(relatives), axis=0)
    best_idx = np.argmax(log_returns)
    assert bs.weights_[best_idx] == pytest.approx(1.0)


def test_best_constant_default_is_log_wealth(X_small):
    """Test BestConstantRebalancedPortfolio defaults to log-wealth maximization."""
    best_constant = BestConstantRebalancedPortfolio()
    best_constant.fit(X_small)

    assert best_constant.objective_measure == PerfMeasure.LOG_WEALTH
    assert hasattr(best_constant, "weights_")
    assert best_constant.weights_.shape == (X_small.shape[1],)
    assert np.sum(best_constant.weights_) == pytest.approx(1.0)


def test_best_constant_log_wealth_explicit(X_small):
    """Test explicit LOG_WEALTH specification."""
    best_constant = BestConstantRebalancedPortfolio(
        objective_measure=PerfMeasure.LOG_WEALTH
    )
    best_constant.fit(X_small)

    assert best_constant.objective_measure == PerfMeasure.LOG_WEALTH
    assert np.sum(best_constant.weights_) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "objective, minimum, maximum",
    [
        pytest.param(PerfMeasure.LOG_WEALTH, None, 0.5, id="log-wealth"),
        pytest.param(RiskMeasure.VARIANCE, 0.05, 0.4, id="variance"),
    ],
)
def test_best_constant_box_constraints(X_small, objective, minimum, maximum):
    min_weight = 1 / X_small.shape[1] if minimum is None else minimum
    model = BestConstantRebalancedPortfolio(
        objective_measure=objective,
        min_weights=min_weight,
        max_weights=maximum,
    ).fit(X_small)
    assert np.all(model.weights_ >= min_weight - 1e-6)
    assert np.all(model.weights_ <= maximum + 1e-6)
    assert np.sum(model.weights_) == pytest.approx(1.0)


@pytest.mark.parametrize("objective_measure", allowed_measures)
def test_best_constant_risk_measures(objective_measure, X_small):
    """Test BestConstantRebalancedPortfolio with CVaR minimization."""
    best_constant = BestConstantRebalancedPortfolio(
        objective_measure=objective_measure,
    )
    best_constant.fit(X_small)

    assert np.sum(best_constant.weights_) == pytest.approx(1.0)
    # Conic solvers satisfy bounds to numerical tolerance, not exact arithmetic.
    assert np.all(best_constant.weights_ >= -1e-8)


def test_best_constant_rejects_invalid_measure(X_small):
    """Test BestConstantRebalancedPortfolio raises error for invalid measure type."""
    with pytest.raises(ValueError, match="objective_measure must be RiskMeasure"):
        best_constant = BestConstantRebalancedPortfolio(objective_measure="invalid")
        best_constant.fit(X_small)


def test_best_constant_solver_rejects_infeasible_nonconvergence():
    """An inaccurate infeasible allocation cannot serve as a regret comparator."""
    from skfolio.exceptions import ConvexOptimizationError
    from skfolio.optimization.online._regret import _solve_best_constant_weights

    # Construct data where BestConstantRebalancedPortfolio is far from uniform (one asset dominates),
    # so the solver needs many iterations to converge from uniform init.
    rng = np.random.default_rng(42)
    relatives = 1.0 + rng.standard_normal((100, 5)) * 0.02
    relatives[:, 0] += 0.05  # asset 0 strongly dominates

    with pytest.warns(UserWarning, match="inaccurate"):
        with pytest.raises(ConvexOptimizationError, match="infeasible"):
            _solve_best_constant_weights(relatives, solver_params={"max_iter": 1})


@pytest.mark.parametrize("weights", [[0.8, 0.2], [0.4, 0.1], [1.2, -0.2]])
def test_constant_wealth_preserves_cash_and_short_positions(weights):
    returns = np.array([[0.1, -0.05], [-0.02, 0.03], [0.04, 0.01]])
    model = ConstantRebalancedPortfolio(weights=weights).fit(returns)
    # Independent account calculation: cash earns zero and targets are restored
    # before each observation, including when an asset is held short.
    value = 1.0
    expected = [value]
    for row in returns:
        cash = value * (1.0 - sum(weights))
        positions = value * np.array(weights)
        value = cash + sum(positions * (1.0 + row))
        expected.append(value)
    np.testing.assert_allclose(model.all_wealth_, expected)
    assert model.wealth_ == pytest.approx(value)
    np.testing.assert_allclose(model.all_weights_, [weights] * len(returns))
    model.all_weights_[0, 0] = 99
    assert model.all_weights_[1, 0] == weights[0]
    assert model.weights_[0] == weights[0]


def test_best_asset_wealth_and_refit():
    model = BestAsset().fit([[0.1, 0.05], [-0.2, 0.05]])
    np.testing.assert_array_equal(model.weights_, [0.0, 1.0])
    np.testing.assert_allclose(model.all_wealth_, [1.0, 1.05, 1.1025])
    model.fit([[0.2, 0.1]])
    np.testing.assert_array_equal(model.weights_, [1.0, 0.0])
    np.testing.assert_allclose(model.all_wealth_, [1.0, 1.2])
    assert model.all_weights_.shape == (1, 2)


def test_optimized_wealth_preserves_initial_capital_and_cash(X_small):
    model = BestConstantRebalancedPortfolio(initial_wealth=250.0, budget=0.5).fit(
        X_small
    )
    value = 250.0
    expected = [value]
    for row in np.asarray(X_small):
        cash = value * (1.0 - model.weights_.sum())
        value = cash + sum(value * model.weights_ * (1.0 + row))
        expected.append(value)
    np.testing.assert_allclose(model.all_wealth_, expected)
    assert model.wealth_ == pytest.approx(value)


@pytest.mark.parametrize(
    "factory",
    [
        BestAsset,
        BestConstantRebalancedPortfolio,
        lambda: ConstantRebalancedPortfolio(weights=None),
    ],
)
@pytest.mark.parametrize(
    "returns", [[], [[np.nan, 0.0]], [[np.inf, 0.0]], [[-1.0, 0.0]], [0.1, 0.2]]
)
def test_benchmarks_reject_invalid_returns(factory, returns):
    with pytest.raises(ValueError):
        factory().fit(returns)


@pytest.mark.parametrize("initial_wealth", [0.0, -1.0, np.nan, np.inf, [1.0, 2.0]])
def test_initial_wealth_must_be_positive_scalar(initial_wealth, X_small):
    with pytest.raises(
        ValueError, match="initial_wealth must be a finite positive scalar"
    ):
        BestConstantRebalancedPortfolio(initial_wealth=initial_wealth).fit(X_small)
