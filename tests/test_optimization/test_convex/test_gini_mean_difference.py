"""GMD sorting-network tests against an independent pairwise LP."""

from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np
import pytest

from skfolio import RiskMeasure
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk, ObjectiveFunction, RiskBudgeting
from skfolio.optimization.convex._base import _solve, _validated_factor_value
from skfolio.optimization.convex._gini_mean_difference import (
    _gmd_sorting_network,
    _odd_even_merge_sort_comparators,
)
from skfolio.prior import EmpiricalPrior


def _make_returns(n_observations=40, n_assets=6, seed=11):
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_observations, 3))
    loadings = rng.normal(0.5, 0.15, size=(3, n_assets))
    returns = factors @ loadings + 0.6 * rng.normal(size=(n_observations, n_assets))
    returns -= returns.mean(axis=0)
    returns /= returns.std(axis=0, ddof=1)
    return returns * np.linspace(0.006, 0.016, n_assets) + np.linspace(
        0.0002, 0.0012, n_assets
    )


def _pairwise_gmd(values):
    differences = np.abs(values[:, None] - values[None, :])
    return differences.sum() / (len(values) * (len(values) - 1))


class _PairwiseMeanRisk(MeanRisk):
    """Independent quadratic-size risk formulation for small oracle problems."""

    def _gini_mean_difference_risk(self, return_distribution, w):
        returns = return_distribution.returns
        n_observations = len(returns)
        row, col = np.triu_indices(n_observations, k=1)
        differences = (returns[row] - returns[col]) @ w
        return 2 * cp.norm1(differences) / (n_observations * (n_observations - 1)), []


@pytest.mark.parametrize("n_observations", range(2, 11))
def test_sorting_network_sorts_all_binary_inputs(n_observations):
    # The zero-one principle also covers arbitrary real inputs.
    comparators = _odd_even_merge_sort_comparators(n_observations)
    for inputs in itertools.product((0, 1), repeat=n_observations):
        outputs = list(inputs)
        for i, j in comparators:
            outputs[i], outputs[j] = (
                min(outputs[i], outputs[j]),
                max(outputs[i], outputs[j]),
            )
        assert outputs == sorted(inputs)


@pytest.mark.parametrize("n_observations", [2, 3, 5, 8, 17, 31])
@pytest.mark.parametrize("distribution", ["random", "ties", "constant"])
def test_relaxed_network_matches_pairwise_oracle(n_observations, distribution):
    rng = np.random.default_rng(17)
    values = rng.normal(size=n_observations)
    if distribution == "ties":
        values = rng.integers(-2, 3, size=n_observations).astype(float)
    elif distribution == "constant":
        values[:] = -0.5
    equality, inequality, coefficients = _gmd_sorting_network(n_observations)
    z = cp.Variable(equality.shape[1])
    inputs = cp.Parameter(n_observations)
    problem = cp.Problem(
        cp.Minimize(coefficients @ z),
        [z[:n_observations] == inputs, equality @ z == 0, inequality @ z <= 0],
    )
    for shift, scale in [(0.0, 1.0), (7.0, 0.01), (-3.0, 10.0)]:
        inputs.value = shift + scale * values
        problem.solve(
            solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10
        )
        assert problem.status == cp.OPTIMAL
        np.testing.assert_allclose(
            problem.value, _pairwise_gmd(inputs.value), atol=2e-9, rtol=1e-8
        )


@pytest.mark.parametrize("n_observations", [0, 1])
def test_sorting_network_requires_two_observations(n_observations):
    with pytest.raises(ValueError, match="at least two observations"):
        _gmd_sorting_network(n_observations)


def test_cached_network_is_immutable_and_subquadratic():
    equality, inequality, coefficients = _gmd_sorting_network(5000)
    cached = _gmd_sorting_network(5000)
    assert all(
        a is b
        for a, b in zip((equality, inequality, coefficients), cached, strict=True)
    )
    assert equality.shape[0] + inequality.shape[0] < 1_000_000
    assert equality.nnz + inequality.nnz < 3_000_000
    for matrix in (equality, inequality):
        for values in (matrix.data, matrix.indices, matrix.indptr):
            assert not values.flags.writeable
    assert not coefficients.flags.writeable


@pytest.mark.parametrize("objective", list(ObjectiveFunction))
@pytest.mark.parametrize(
    ("layout", "scale", "unrestricted", "solver"),
    [
        ("random", 1.0, False, "CLARABEL"),
        ("ties", 1.0, False, "CLARABEL"),
        ("collinear", 1.0, False, "CLARABEL"),
        ("random", 0.01, False, "CLARABEL"),
        ("random", 100.0, False, "CLARABEL"),
        ("random", 1.0, True, "CLARABEL"),
        ("random", 1.0, False, "SCS"),
        ("random", 1.0, True, "SCS"),
    ],
)
def test_portfolio_objectives_match_pairwise_oracle(
    objective, layout, scale, unrestricted, solver
):
    if solver not in cp.installed_solvers():
        pytest.skip(f"{solver} is not installed")
    returns = _make_returns(31, 5)
    if layout == "ties":
        returns = np.repeat(returns, 2, axis=0)
    elif layout == "collinear":
        returns[:, 1] = returns[:, 0]
    returns *= scale
    risk_limit = (
        1.2 * _pairwise_gmd(returns.mean(axis=1))
        if objective == ObjectiveFunction.MAXIMIZE_RETURN
        else None
    )
    settings = dict(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=objective,
        risk_aversion=3.0,
        max_gini_mean_difference=risk_limit,
        min_weights=None if unrestricted else 0,
        max_weights=None if unrestricted else 1,
        scale_objective=1 / scale,
        scale_constraints=1 / scale,
    )
    reference = _PairwiseMeanRisk(**settings).fit(returns)
    model = MeanRisk(
        **settings,
        solver=solver,
        solver_params={"eps": 1e-8} if solver == "SCS" else None,
    ).fit(returns)
    risk, reference_risk = (
        _pairwise_gmd(returns @ fitted.weights_) for fitted in (model, reference)
    )
    mean, reference_mean = (
        returns.mean(axis=0) @ fitted.weights_ for fitted in (model, reference)
    )
    np.testing.assert_allclose(model.problem_values_["risk"], risk, atol=1e-12)
    if objective == ObjectiveFunction.MAXIMIZE_RATIO:
        np.testing.assert_allclose(
            mean / risk, reference_mean / reference_risk, rtol=2e-5
        )
    elif objective == ObjectiveFunction.MAXIMIZE_UTILITY:
        np.testing.assert_allclose(
            (mean - 3 * risk) / scale,
            (reference_mean - 3 * reference_risk) / scale,
            atol=5e-8,
        )
    elif objective == ObjectiveFunction.MINIMIZE_RISK:
        np.testing.assert_allclose(risk / scale, reference_risk / scale, atol=5e-8)
    else:
        assert risk <= risk_limit + 5e-8 * scale
        np.testing.assert_allclose(mean / scale, reference_mean / scale, atol=5e-8)


@pytest.mark.parametrize("target", [False, True])
@pytest.mark.parametrize("risk_limit", [None, 1.0, [1.0, 2.0]])
def test_maximum_return_reports_empirical_gmd_with_nonbinding_cap(target, risk_limit):
    returns = _make_returns()
    target_weights = np.full(6, 1 / 6) if target else None
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_gini_mean_difference=risk_limit,
        target_weights=target_weights,
    ).fit(returns)
    results = (
        zip(model.weights_, model.problem_values_, strict=True)
        if isinstance(risk_limit, list)
        else [(model.weights_, model.problem_values_)]
    )
    for weights, values in results:
        active_weights = weights - target_weights if target else weights
        np.testing.assert_allclose(
            values["risk"], _pairwise_gmd(returns @ active_weights), atol=1e-12
        )
        np.testing.assert_allclose(
            values["expected_return"], returns.mean(axis=0).max(), atol=1e-8
        )


@pytest.mark.parametrize(
    "objective",
    [ObjectiveFunction.MINIMIZE_RISK, ObjectiveFunction.MAXIMIZE_RATIO],
)
def test_target_weights_use_normalized_active_returns(objective):
    returns = _make_returns()
    target = np.arange(1, 7, dtype=float)
    target /= target.sum()
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=objective,
        target_weights=target,
    ).fit(returns)
    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(returns @ (model.weights_ - target)),
        atol=1e-12,
    )


@pytest.mark.parametrize("offset", [1e6, 1e7, 1e8])
def test_ratio_with_large_return_offset(offset):
    returns = offset + 0.05 * _make_returns()
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    ).fit(returns)
    assert 0 < model.problem_values_["factor"] < 1.1e-3
    np.testing.assert_allclose(model.weights_.sum(), 1.0, atol=1e-8)
    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(returns @ model.weights_),
        atol=2e-8,
    )


@pytest.mark.parametrize("n_assets", [20, 50])
def test_500_observations_use_one_sparse_solver_call(monkeypatch, n_assets):
    calls = []
    original_solve = cp.Problem.solve

    def solve(problem, *args, **kwargs):
        calls.append(problem)
        return original_solve(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", solve)
    returns = np.random.default_rng(42).normal(0.0005, 0.01, (500, n_assets))
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE, save_problem=True
    ).fit(returns)
    equality, inequality, _ = _gmd_sorting_network(500)
    assert calls == [model.problem_]
    assert (
        model.problem_.size_metrics.num_scalar_variables == n_assets + equality.shape[1]
    )
    assert model.problem_.size_metrics.num_scalar_leq_constr == (
        2 * n_assets + inequality.shape[0]
    )
    np.testing.assert_allclose(
        model.problem_values_["objective"],
        _pairwise_gmd(returns @ model.weights_),
        atol=2e-8,
    )


@pytest.mark.parametrize(
    "risk_measure", [RiskMeasure.GINI_MEAN_DIFFERENCE, RiskMeasure.VARIANCE]
)
@pytest.mark.parametrize("targets", [[0.0002, 1.0, 0.0003], [0.0002, 0.0003, 1.0]])
def test_sweep_reuses_problem_and_restores_last_success(
    monkeypatch, risk_measure, targets
):
    calls = []
    solver_stats = []
    original_solve = cp.Problem.solve

    def solve(problem, *args, **kwargs):
        calls.append(problem)
        result = original_solve(problem, *args, **kwargs)
        solver_stats.append(problem.solver_stats)
        return result

    monkeypatch.setattr(cp.Problem, "solve", solve)
    returns = _make_returns()
    model = MeanRisk(
        risk_measure=risk_measure,
        max_gini_mean_difference=0.02,
        min_return=targets,
        raise_on_failure=False,
        save_problem=True,
    )
    with pytest.warns(UserWarning, match="Solver 'CLARABEL' failed"):
        model.fit(returns)
    assert calls == [model.problem_] * 3
    assert model.problem_.status == cp.OPTIMAL
    last = max(i for i, error in enumerate(model.error_) if error is None)
    assert model.problem_.solver_stats is solver_stats[last]
    assert all(
        np.isnan(weights).all()
        for weights, error in zip(model.weights_, model.error_, strict=True)
        if error
    )
    assert any(float(p.value) == targets[last] for p in model.problem_.parameters())
    saved = next(v for v in model.problem_.variables() if v.shape == (6,))
    np.testing.assert_allclose(
        saved.value, model.weights_[last] * model.problem_values_[last]["factor"]
    )
    for constraint in model.problem_.constraints:
        assert np.max(constraint.violation()) <= 1e-7


@pytest.mark.parametrize("optimizer", [MeanRisk, RiskBudgeting])
def test_costs_and_fees_do_not_change_gmd(optimizer):
    returns = _make_returns()
    model = optimizer(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        transaction_costs=0.0001,
        management_fees=np.linspace(0, 0.0001, 6),
        previous_weights=np.full(6, 1 / 6),
    ).fit(returns)
    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(returns @ model.weights_),
        atol=1e-12,
    )
    if optimizer is RiskBudgeting:
        contributions = model.predict(returns).contribution(
            RiskMeasure.GINI_MEAN_DIFFERENCE
        )
        np.testing.assert_allclose(
            contributions, contributions.mean(), rtol=2e-3, atol=2e-6
        )


@pytest.mark.skipif(
    "SCIP" not in cp.installed_solvers(), reason="SCIP is not installed"
)
@pytest.mark.parametrize(
    "objective",
    [
        ObjectiveFunction.MINIMIZE_RISK,
        ObjectiveFunction.MAXIMIZE_RATIO,
        ObjectiveFunction.MAXIMIZE_RETURN,
    ],
)
def test_scip_cardinality(objective):
    returns = _make_returns(30, 5)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=objective,
        max_gini_mean_difference=0.009
        if objective == ObjectiveFunction.MAXIMIZE_RETURN
        else None,
        cardinality=3,
        solver="SCIP",
    ).fit(returns)
    assert np.count_nonzero(np.abs(model.weights_) > 1e-8) <= 3
    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(returns @ model.weights_),
        atol=1e-12,
    )


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_repeated_fits_use_new_returns(method):
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=20),
            covariance_estimator=EWCovariance(half_life=20),
        ),
    )
    for returns in [_make_returns(40), _make_returns(70, seed=2)]:
        getattr(model, method)(returns)
        fitted = model.prior_estimator_.return_distribution_.returns
        np.testing.assert_allclose(
            model.problem_values_["risk"],
            _pairwise_gmd(fitted @ model.weights_),
            atol=1e-12,
        )


@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_truly_unbounded_problem_is_rejected(raise_on_failure):
    base = np.random.default_rng(4).normal(0, 0.01, 20)
    returns = np.column_stack((base, base + 0.001))
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        min_weights=None,
        max_weights=None,
        max_gini_mean_difference=0.02,
        raise_on_failure=raise_on_failure,
    )
    if raise_on_failure:
        with pytest.raises(cp.SolverError, match="status 'unbounded"):
            model.fit(returns)
    else:
        with pytest.warns(UserWarning, match="status 'unbounded"):
            model.fit(returns)
        assert model.weights_ is None


@pytest.mark.parametrize("status", [cp.OPTIMAL_INACCURATE, cp.USER_LIMIT])
def test_solver_status_handling(monkeypatch, status):
    original_solve = cp.Problem.solve

    def solve(problem, *args, **kwargs):
        result = original_solve(problem, *args, **kwargs)
        problem._status = status
        return result

    monkeypatch.setattr(cp.Problem, "solve", solve)
    model = MeanRisk(risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE)
    if status == cp.OPTIMAL_INACCURATE:
        with pytest.warns(UserWarning, match="Solution may be inaccurate"):
            model.fit(_make_returns())
    else:
        with pytest.raises(cp.SolverError, match="status 'user_limit'"):
            model.fit(_make_returns())


@pytest.mark.parametrize("factor", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_ratio_factor_is_rejected(factor):
    with pytest.raises(cp.SolverError, match="invalid homogeneous"):
        _validated_factor_value(factor)


@pytest.mark.parametrize("invalid_weight", [np.nan, np.inf, -np.inf])
def test_non_finite_solver_weights_are_rejected(monkeypatch, invalid_weight):
    weights = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(weights)))

    def solve(**kwargs):
        weights.save_value(np.array([invalid_weight, 1.0]))
        problem._status = cp.OPTIMAL

    monkeypatch.setattr(problem, "solve", solve)
    with pytest.raises(cp.SolverError, match="non-finite weights"):
        _solve(
            w=weights,
            factor=cp.Constant(1),
            expressions={},
            problem=problem,
            solver="CLARABEL",
            solver_params={},
            risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
            scale_objective=cp.Constant(1),
        )


def test_empirical_gmd_normalization_overflow_is_rejected():
    weights = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(weights)), [cp.sum(weights) == 1])
    with pytest.raises(cp.SolverError, match="normalized empirical GMD is non-finite"):
        _solve(
            w=weights,
            factor=cp.Constant(1e-11),
            expressions={},
            problem=problem,
            solver="CLARABEL",
            solver_params={},
            risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
            scale_objective=cp.Constant(1),
            risk_returns=cp.Constant(np.array([1e308, -1e308])),
        )
