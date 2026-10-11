"""Tests for benchmark portfolio classes (BCRP, CRP, UCRP, BestStock)"""

import numpy as np
import pytest

from skfolio.measures import PerfMeasure, RiskMeasure
from skfolio.optimization.online._autograd_objectives import MEASURE_PROPERTIES
from skfolio.optimization.online._benchmark import BCRP, CRP, UCRP, BestStock

allowed_measures = list(MEASURE_PROPERTIES.keys())


def test_crp_with_custom_weights(X_small):
    """Test CRP with custom weights."""
    n_assets = X_small.shape[1]
    weights = np.ones(n_assets) / n_assets
    crp = CRP(weights=weights)
    crp.fit(X_small)


def test_ucrp_uniform_weights(X_small):
    """Test UCRP produces uniform weights."""
    n_assets = X_small.shape[1]
    ucrp = UCRP()
    ucrp.fit(X_small)
    expected = np.ones(n_assets) / n_assets
    np.testing.assert_array_almost_equal(ucrp.weights_, expected)


def test_crp_none_defaults_to_uniform(X_small):
    """Test CRP with weights=None defaults to uniform."""
    n_assets = X_small.shape[1]
    crp = CRP(weights=None)
    crp.fit(X_small)
    expected = np.ones(n_assets) / n_assets
    np.testing.assert_array_almost_equal(crp.weights_, expected)


def test_best_stock_picks_best_performer(X_small):
    """Test BestStock picks the asset with highest cumulative log return."""
    bs = BestStock()
    bs.fit(X_small)

    # Verify one-hot vector
    assert np.sum(bs.weights_) == pytest.approx(1.0)
    assert np.sum(bs.weights_ > 0) == 1

    # Verify it picks the best asset
    relatives = 1.0 + X_small.values
    log_returns = np.sum(np.log(relatives), axis=0)
    best_idx = np.argmax(log_returns)
    assert bs.weights_[best_idx] == pytest.approx(1.0)


def test_bcrp_default_is_log_wealth(X_small):
    """Test BCRP defaults to log-wealth maximization."""
    bcrp = BCRP()
    bcrp.fit(X_small)

    assert bcrp.objective_measure == PerfMeasure.LOG_WEALTH
    assert hasattr(bcrp, "weights_")
    assert bcrp.weights_.shape == (X_small.shape[1],)
    assert np.sum(bcrp.weights_) == pytest.approx(1.0)


def test_bcrp_log_wealth_explicit(X_small):
    """Test explicit LOG_WEALTH specification."""
    bcrp = BCRP(objective_measure=PerfMeasure.LOG_WEALTH)
    bcrp.fit(X_small)

    assert bcrp.objective_measure == PerfMeasure.LOG_WEALTH
    assert np.sum(bcrp.weights_) == pytest.approx(1.0)


def test_bcrp_log_wealth_with_constraints(X_small):
    """Test BCRP log-wealth with box constraints."""
    n_assets = X_small.shape[1]
    min_weight, max_weight = 1 / n_assets, 0.5
    bcrp = BCRP(
        objective_measure=PerfMeasure.LOG_WEALTH,
        min_weights=min_weight,
        max_weights=max_weight,
    )
    bcrp.fit(X_small)

    assert np.all(bcrp.weights_ >= min_weight - 1e-6)
    assert np.all(bcrp.weights_ <= max_weight + 1e-6)
    assert np.sum(bcrp.weights_) == pytest.approx(1.0)


@pytest.mark.parametrize("objective_measure", allowed_measures)
def test_bcrp_risk_measures(objective_measure, X_small):
    """Test BCRP with CVaR minimization."""
    bcrp = BCRP(
        objective_measure=objective_measure,
    )
    bcrp.fit(X_small)

    assert np.sum(bcrp.weights_) == pytest.approx(1.0)
    # Conic solvers satisfy bounds to numerical tolerance, not exact arithmetic.
    assert np.all(bcrp.weights_ >= -1e-8)


def test_bcrp_rejects_invalid_measure(X_small):
    """Test BCRP raises error for invalid measure type."""
    with pytest.raises(ValueError, match="objective_measure must be RiskMeasure"):
        bcrp = BCRP(objective_measure="invalid")
        bcrp.fit(X_small)


def test_bcrp_with_box_constraints(X_small):
    """Test BCRP respects min/max weight constraints."""
    min_weight, max_weight = 0.05, 0.4
    bcrp = BCRP(
        objective_measure=RiskMeasure.VARIANCE,
        min_weights=min_weight,
        max_weights=max_weight,
    )
    bcrp.fit(X_small)

    assert np.all(bcrp.weights_ >= min_weight - 1e-6)
    assert np.all(bcrp.weights_ <= max_weight + 1e-6)
    assert np.sum(bcrp.weights_) == pytest.approx(1.0)


def test_bcrp_fit_dynamic_produces_all_weights(X_small):
    """Test fit_dynamic produces weights for each time step."""
    bcrp = BCRP()
    bcrp.fit_dynamic(X_small)

    assert hasattr(bcrp, "all_weights_")
    assert bcrp.all_weights_.shape == X_small.shape


def test_bcrp_solver_warns_on_nonconvergence():
    """_solve_bcrp_constant must emit a warning when max_iter is reached."""
    from skfolio.optimization.online._regret import _solve_bcrp_constant

    # Construct data where BCRP is far from uniform (one asset dominates),
    # so the solver needs many iterations to converge from uniform init.
    rng = np.random.default_rng(42)
    relatives = 1.0 + rng.standard_normal((100, 5)) * 0.02
    relatives[:, 0] += 0.05  # asset 0 strongly dominates

    with pytest.warns(UserWarning, match="not converge"):
        _solve_bcrp_constant(relatives, max_iter=1)


def test_dynamic_comparator_respects_first_period_constraints():
    X = np.array([[0.1, 0.01], [0.02, 0.03], [0.03, -0.01]])
    model = BCRP(max_weights=0.6).fit_dynamic(X)
    assert np.all(model.all_weights_ <= 0.6 + 1e-7)
    assert np.all(model.all_weights_ >= -1e-7)


def test_dynamic_comparator_named_first_period_constraint():
    import pandas as pd

    X = pd.DataFrame([[0.1, 0.01], [0.02, 0.03], [0.03, -0.01]], columns=["A", "B"])
    model = BCRP(linear_constraints=["A <= 0.6"]).fit_dynamic(X)
    assert np.all(model.all_weights_[:, 0] <= 0.6 + 1e-7)


def test_dynamic_risk_objective_rejects_undefined_first_period():
    with pytest.raises(ValueError, match="risk cannot be estimated"):
        BCRP(objective_measure=RiskMeasure.VARIANCE).fit_dynamic([[0.1, 0.2]])
