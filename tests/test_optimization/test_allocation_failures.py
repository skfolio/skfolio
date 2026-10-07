"""Allocation failures at numerical and composite estimator boundaries."""

import cvxpy as cp
import numpy as np
import pytest

from skfolio.cluster import HierarchicalClustering
from skfolio.exceptions import ConvexOptimizationError, OptimizationError
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    NestedClustersOptimization,
    StackingOptimization,
)
from skfolio.prior import EmpiricalPrior, ReturnDistribution


@pytest.fixture
def returns():
    return np.random.default_rng(3).normal(0, 0.01, (40, 4)) * [1, 2, 3, 4]


def test_inverse_volatility_rejects_undefined_normalization(returns, monkeypatch):
    def fit_zero_variance_prior(self, X, y=None):
        self.return_distribution_ = ReturnDistribution(
            mu=X.mean(axis=0), covariance=np.diag([0, 1, 2, 3]), returns=X
        )
        return self

    monkeypatch.setattr(EmpiricalPrior, "fit", fit_zero_variance_prior)
    with np.errstate(divide="ignore", invalid="ignore"):
        with pytest.raises(OptimizationError, match="non-finite weights"):
            InverseVolatility().fit(returns)
        model = InverseVolatility(fallback=EqualWeighted()).fit(returns)
    np.testing.assert_array_equal(model.weights_, np.full(4, 0.25))


@pytest.mark.parametrize("failure", ["solver", "nonfinite"])
def test_herc_convex_failure_type_and_cause(returns, monkeypatch, failure):
    backend_error = cp.SolverError("Constraint adjustment failed")
    solve = cp.Problem.solve

    def fail_solve(problem, *args, **kwargs):
        if failure == "solver":
            raise backend_error
        solve(problem, *args, **kwargs)
        problem.variables()[0].save_value(np.full(4, np.nan))

    monkeypatch.setattr(cp.Problem, "solve", fail_solve)
    model = HierarchicalEqualRiskContribution(min_weights=0.25, max_weights=0.25)
    with pytest.raises(ConvexOptimizationError) as caught:
        model.fit(returns)
    if failure == "solver":
        assert isinstance(caught.value.__cause__, cp.SolverError)
        assert caught.value.__cause__.__cause__ is backend_error
    else:
        assert "non-finite weights" in str(caught.value)


@pytest.mark.parametrize("composite", ["nco", "stacking"])
@pytest.mark.parametrize("failed_child", ["inner", "outer"])
def test_composite_rejects_missing_child_allocation(returns, composite, failed_child):
    failed = MeanRisk(min_return=10, raise_on_failure=False)
    inner = failed if failed_child == "inner" else EqualWeighted()
    outer = failed if failed_child == "outer" else EqualWeighted()
    if composite == "nco":
        model = NestedClustersOptimization(
            inner_estimator=inner,
            outer_estimator=outer,
            clustering_estimator=HierarchicalClustering(max_clusters=2),
            cv="ignore",
        )
    else:
        model = StackingOptimization(
            estimators=[("inner", inner)], final_estimator=outer, cv="ignore"
        )
    with (
        pytest.warns(UserWarning, match="Solver 'CLARABEL' failed"),
        pytest.raises(OptimizationError, match="returned no allocation weights"),
    ):
        model.fit(returns)


@pytest.mark.parametrize("budget", [0, 0.5])
def test_finite_allocations_do_not_require_unit_budget(returns, budget):
    model = MeanRisk(min_weights=-1, budget=budget).fit(returns)
    assert np.isfinite(model.weights_).all()
    assert model.weights_.sum() == pytest.approx(budget, abs=1e-8)


@pytest.mark.parametrize("risk", [0.0, np.nan, np.inf])
@pytest.mark.parametrize("stage", ["asset", "cluster"])
def test_hrp_rejects_undefined_risk_splits(returns, monkeypatch, risk, stage):
    if stage == "asset":
        monkeypatch.setattr(
            HierarchicalRiskParity, "_unitary_risks", lambda *a, **kw: np.full(4, risk)
        )
    else:
        monkeypatch.setattr(HierarchicalRiskParity, "_risk", lambda *a, **kw: risk)
    with pytest.raises(OptimizationError, match="HRP cannot split"):
        HierarchicalRiskParity().fit(returns)
    model = HierarchicalRiskParity(fallback=EqualWeighted()).fit(returns)
    np.testing.assert_array_equal(model.weights_, np.full(4, 0.25))
