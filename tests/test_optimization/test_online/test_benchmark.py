"""Tests for benchmark portfolio classes (BestConstantRebalancedPortfolio, ConstantRebalancedPortfolio, UniformConstantRebalanced, BestAsset)"""

import numpy as np
import pytest

from skfolio.measures import PerfMeasure, RiskMeasure
from skfolio.optimization.online._autograd_objectives import MEASURE_PROPERTIES
from skfolio.optimization.online._benchmark import (
    BestAsset,
    BestConstantRebalancedPortfolio,
    ConstantRebalancedPortfolio,
    UniformConstantRebalanced,
)

allowed_measures = list(MEASURE_PROPERTIES.keys())


def test_constant_rebalanced_with_custom_weights(X_small):
    """Test ConstantRebalancedPortfolio with custom weights."""
    n_assets = X_small.shape[1]
    weights = np.ones(n_assets) / n_assets
    constant_rebalanced = ConstantRebalancedPortfolio(weights=weights)
    constant_rebalanced.fit(X_small)


def test_uniform_constant_uniform_weights(X_small):
    """Test UniformConstantRebalanced produces uniform weights."""
    n_assets = X_small.shape[1]
    uniform_constant = UniformConstantRebalanced()
    uniform_constant.fit(X_small)
    expected = np.ones(n_assets) / n_assets
    np.testing.assert_array_almost_equal(uniform_constant.weights_, expected)


def test_constant_rebalanced_none_defaults_to_uniform(X_small):
    """Test ConstantRebalancedPortfolio with weights=None defaults to uniform."""
    n_assets = X_small.shape[1]
    constant_rebalanced = ConstantRebalancedPortfolio(weights=None)
    constant_rebalanced.fit(X_small)
    expected = np.ones(n_assets) / n_assets
    np.testing.assert_array_almost_equal(constant_rebalanced.weights_, expected)


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


def test_best_constant_log_wealth_with_constraints(X_small):
    """Test BestConstantRebalancedPortfolio log-wealth with box constraints."""
    n_assets = X_small.shape[1]
    min_weight, max_weight = 1 / n_assets, 0.5
    best_constant = BestConstantRebalancedPortfolio(
        objective_measure=PerfMeasure.LOG_WEALTH,
        min_weights=min_weight,
        max_weights=max_weight,
    )
    best_constant.fit(X_small)

    assert np.all(best_constant.weights_ >= min_weight - 1e-6)
    assert np.all(best_constant.weights_ <= max_weight + 1e-6)
    assert np.sum(best_constant.weights_) == pytest.approx(1.0)


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


def test_best_constant_with_box_constraints(X_small):
    """Test BestConstantRebalancedPortfolio respects min/max weight constraints."""
    min_weight, max_weight = 0.05, 0.4
    best_constant = BestConstantRebalancedPortfolio(
        objective_measure=RiskMeasure.VARIANCE,
        min_weights=min_weight,
        max_weights=max_weight,
    )
    best_constant.fit(X_small)

    assert np.all(best_constant.weights_ >= min_weight - 1e-6)
    assert np.all(best_constant.weights_ <= max_weight + 1e-6)
    assert np.sum(best_constant.weights_) == pytest.approx(1.0)


def test_best_constant_fit_dynamic_produces_all_weights(X_small):
    """Test fit_dynamic produces weights for each time step."""
    best_constant = BestConstantRebalancedPortfolio()
    best_constant.fit_dynamic(X_small)

    assert hasattr(best_constant, "all_weights_")
    assert best_constant.all_weights_.shape == X_small.shape


def test_best_constant_solver_warns_on_nonconvergence():
    """_solve_best_constant_constant must emit a warning when max_iter is reached."""
    from skfolio.optimization.online._regret import _solve_best_constant_constant

    # Construct data where BestConstantRebalancedPortfolio is far from uniform (one asset dominates),
    # so the solver needs many iterations to converge from uniform init.
    rng = np.random.default_rng(42)
    relatives = 1.0 + rng.standard_normal((100, 5)) * 0.02
    relatives[:, 0] += 0.05  # asset 0 strongly dominates

    with pytest.warns(UserWarning, match="not converge"):
        _solve_best_constant_constant(relatives, max_iter=1)


def test_dynamic_comparator_respects_first_period_constraints():
    X = np.array([[0.1, 0.01], [0.02, 0.03], [0.03, -0.01]])
    model = BestConstantRebalancedPortfolio(max_weights=0.6).fit_dynamic(X)
    assert np.all(model.all_weights_ <= 0.6 + 1e-7)
    assert np.all(model.all_weights_ >= -1e-7)


def test_dynamic_comparator_named_first_period_constraint():
    import pandas as pd

    X = pd.DataFrame([[0.1, 0.01], [0.02, 0.03], [0.03, -0.01]], columns=["A", "B"])
    model = BestConstantRebalancedPortfolio(
        linear_constraints=["A <= 0.6"]
    ).fit_dynamic(X)
    assert np.all(model.all_weights_[:, 0] <= 0.6 + 1e-7)


def test_dynamic_risk_objective_rejects_undefined_first_period():
    with pytest.raises(ValueError, match="risk cannot be estimated"):
        BestConstantRebalancedPortfolio(
            objective_measure=RiskMeasure.VARIANCE
        ).fit_dynamic([[0.1, 0.2]])
