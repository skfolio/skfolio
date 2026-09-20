"""Tests for GMD permutation-cut constraint generation."""

from __future__ import annotations

import itertools

import cvxpy as cp
import numpy as np
import pytest
from sklearn.base import clone

from skfolio import RiskMeasure
from skfolio.measures import owa_gmd_weights
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk, ObjectiveFunction, RiskBudgeting
from skfolio.optimization.convex._base import _solve, _validated_factor_value
from skfolio.optimization.convex._gini_mean_difference import _GiniMeanDifference
from skfolio.optimization.convex._mean_risk import _optimal_homogenization_factor
from skfolio.prior import EmpiricalPrior


def _make_returns(
    n_observations: int = 40, n_assets: int = 6, seed: int = 11
) -> np.ndarray:
    """Create deterministic correlated returns with distinct expected returns."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(n_observations, 3))
    loadings = rng.normal(0.5, 0.15, size=(3, n_assets))
    returns = factors @ loadings + 0.6 * rng.normal(size=(n_observations, n_assets))
    returns -= returns.mean(axis=0)
    returns /= returns.std(axis=0, ddof=1)
    return returns * np.linspace(0.006, 0.016, n_assets) + np.linspace(
        0.0002, 0.0012, n_assets
    )


def _pairwise_gmd(values: np.ndarray) -> float:
    """Independent quadratic-size empirical GMD oracle."""
    values = np.asarray(values, dtype=float)
    differences = np.abs(values[:, None] - values[None, :])
    return float(differences.sum() / (len(values) * (len(values) - 1)))


def _scaled_ratio_returns(scale: float) -> np.ndarray:
    """Create a ratio problem with a small homogeneous normalization factor."""
    return scale + 0.05 * _make_returns()


def _pairwise_cvxpy_risk(
    returns: np.ndarray, weights: cp.Variable
) -> tuple[cp.Expression, list[cp.Constraint]]:
    """Independent pairwise-linear GMD formulation for small test problems."""
    n_observations = returns.shape[0]
    row, col = np.triu_indices(n_observations, k=1)
    # Centering subtracts one observation-independent scalar and makes the
    # independent pairwise model robust to the extreme ratio-scale regressions.
    portfolio_returns = (returns - returns.mean(axis=0)) @ weights
    differences = cp.Variable(len(row), nonneg=True)
    constraints = [
        differences >= portfolio_returns[row] - portfolio_returns[col],
        differences >= portfolio_returns[col] - portfolio_returns[row],
    ]
    risk = 2 * cp.sum(differences) / (n_observations * (n_observations - 1))
    return risk, constraints


def _solve_pairwise_reference(
    returns: np.ndarray,
    context: str,
    *,
    risk_aversion: float = 3.0,
    risk_limit: float | None = None,
    unrestricted: bool = False,
) -> tuple[np.ndarray, float, float]:
    """Solve a small independent reference problem for one MeanRisk context."""
    n_assets = returns.shape[1]
    mu = returns.mean(axis=0)
    weights = cp.Variable(n_assets)
    constraints: list[cp.Constraint] = [] if unrestricted else [weights >= 0]
    ratio_scale = 1.0

    if context == "maximum_ratio":
        factor = cp.Variable()
        constraints += [cp.sum(weights) == factor]
        mu_scale = np.max(np.abs(mu))
        homogenization_factor = _optimal_homogenization_factor(mu)
        ratio_scale = mu_scale / homogenization_factor
        constraints += [(mu / mu_scale) @ weights == homogenization_factor / mu_scale]
    else:
        factor = cp.Constant(1)
        constraints += [cp.sum(weights) == 1]

    risk, risk_constraints = _pairwise_cvxpy_risk(returns, weights)
    constraints += risk_constraints
    if risk_limit is not None and context != "maximum_return_under_risk":
        constraints += [risk * ratio_scale <= risk_limit * factor * ratio_scale]

    if context == "minimum_risk":
        objective: cp.Objective = cp.Minimize(risk)
    elif context == "maximum_utility":
        objective = cp.Maximize(mu @ weights - risk_aversion * risk)
    elif context == "maximum_ratio":
        objective = cp.Minimize(risk * ratio_scale)
    elif context == "maximum_return_under_risk":
        assert risk_limit is not None
        constraints += [risk <= risk_limit * factor]
        objective = cp.Maximize(mu @ weights)
    else:  # pragma: no cover - protected by test parametrization
        raise ValueError(f"Unknown context: {context}")

    problem = cp.Problem(objective, constraints)
    problem.solve(solver="CLARABEL", tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    normalized_weights = np.asarray(weights.value / factor.value, dtype=float)
    exact_risk = _pairwise_gmd(returns @ normalized_weights)
    expected_return = float(mu @ normalized_weights)
    return normalized_weights, exact_risk, expected_return


@pytest.mark.parametrize("seed", range(5))
def test_owa_gmd_matches_pairwise_oracle(seed):
    values = np.random.default_rng(seed).normal(size=25)
    np.testing.assert_allclose(
        owa_gmd_weights(len(values)) @ np.sort(values),
        _pairwise_gmd(values),
        rtol=1e-14,
        atol=1e-14,
    )


def test_two_observation_optimization_matches_pairwise_oracle():
    returns = np.array([[0.01, 0.02, -0.01], [-0.02, 0.01, 0.03]])
    model = MeanRisk(risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE).fit(returns)

    np.testing.assert_allclose(
        model.problem_values_["risk"], _pairwise_gmd(returns @ model.weights_)
    )


def test_permutation_cut_is_reduced_to_asset_coefficients():
    returns = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])
    weights = cp.Variable(2)
    generator = _GiniMeanDifference(returns, weights, cp.Constant(3.0))
    permutation = (2, 1, 0)
    constraint = generator._create_cut(permutation, normalization_factor=1.0)

    candidate = np.array([0.3, 0.7])
    weights.value = candidate
    generator.expression.value = 0.0
    expected = owa_gmd_weights(3) @ (returns @ candidate)[list(permutation)]

    assert constraint.args[0].size == 1
    np.testing.assert_allclose(constraint.violation(), max(3 * expected, 0))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"returns": np.ones(3), "weights": cp.Variable(1)}, "2D array"),
        (
            {"returns": np.ones((1, 2)), "weights": cp.Variable(2)},
            "at least two observations",
        ),
        (
            {"returns": np.array([[1.0], [np.nan]]), "weights": cp.Variable(1)},
            "finite values",
        ),
        (
            {"returns": np.ones((2, 2)), "weights": cp.Variable(3)},
            "one expression per asset",
        ),
        (
            {
                "returns": np.ones((2, 2)),
                "weights": cp.Variable(2),
                "absolute_tolerance": -1.0,
            },
            "tolerances",
        ),
    ],
)
def test_generator_validates_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _GiniMeanDifference(scale_constraints=cp.Constant(1.0), **kwargs)


def test_separation_oracle_finds_maximizing_permutation():
    returns = _make_returns(n_observations=5, n_assets=3)
    weights = cp.Variable(3)
    generator = _GiniMeanDifference(returns, weights, cp.Constant(1.0))
    candidate = np.array([1.0, 0.0, 0.0])
    portfolio_returns = returns @ candidate
    owa_weights = owa_gmd_weights(len(returns))

    enumerated = max(
        owa_weights @ portfolio_returns[list(permutation)]
        for permutation in itertools.permutations(range(len(returns)))
    )
    weights.value = candidate
    generator.expression.value = enumerated - 1e-4
    cuts = {generator._initial_permutation: 1.0}

    assert (
        generator._separate(normalization_factor=1.0, cut_normalization_factors=cuts)
        is not None
    )
    assert generator._stable_permutation(portfolio_returns) in cuts
    np.testing.assert_allclose(generator.evaluate(1.0), enumerated, atol=1e-14)


def test_stable_ties_and_duplicate_violated_permutation():
    assert _GiniMeanDifference._stable_permutation(np.array([1.0, 1.0, 0.0, 1.0])) == (
        2,
        0,
        1,
        3,
    )

    returns = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])
    weights = cp.Variable(2)
    generator = _GiniMeanDifference(returns, weights, cp.Constant(1.0))
    weights.value = np.array([1.0, 0.0])
    generator.expression.value = 0.0
    cuts = {generator._initial_permutation: 1.0}

    assert (
        generator._separate(normalization_factor=1.0, cut_normalization_factors=cuts)
        is not None
    )
    with pytest.raises(cp.SolverError, match="duplicate violated permutation"):
        generator._separate(normalization_factor=1.0, cut_normalization_factors=cuts)


def test_small_duplicate_violation_is_treated_as_master_feasibility_error():
    returns = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])
    weights = cp.Variable(2)
    generator = _GiniMeanDifference(returns, weights, cp.Constant(1.0))
    weights.value = np.array([1.0, 0.0])
    generator.expression.value = 0.0
    cuts = {generator._initial_permutation: 1.0}

    assert (
        generator._separate(normalization_factor=1.0, cut_normalization_factors=cuts)
        is not None
    )
    exact_value = _pairwise_gmd(returns @ weights.value)
    generator.expression.value = exact_value - 2e-8

    assert (
        generator._separate(normalization_factor=1.0, cut_normalization_factors=cuts)
        is None
    )


def test_missing_weight_values_are_not_silent():
    returns = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])
    missing_weights = cp.Variable(2)
    generator = _GiniMeanDifference(returns, missing_weights, cp.Constant(1.0))
    generator.expression.value = 0.0
    cuts = {generator._initial_permutation: 1.0}
    with pytest.raises(cp.SolverError, match="invalid weight values"):
        generator._separate(normalization_factor=1.0, cut_normalization_factors=cuts)


def test_missing_or_non_finite_separation_values_are_not_silent(monkeypatch):
    returns = np.array([[0.0, 1.0], [1.0, 0.0]])
    weights = cp.Variable(2)
    generator = _GiniMeanDifference(returns, weights, cp.Constant(1.0))
    weights.value = np.array([0.5, 0.5])
    cuts = {generator._initial_permutation: 1.0}

    with pytest.raises(cp.SolverError, match=r"epigraph.*finite scalar"):
        generator._separate(normalization_factor=1.0, cut_normalization_factors=cuts)

    generator.expression.value = 0.0
    generator._returns[0, 0] = np.inf
    with pytest.raises(cp.SolverError, match="invalid portfolio returns"):
        generator._separate(normalization_factor=1.0, cut_normalization_factors=cuts)

    generator._returns[0, 0] = 0.0
    monkeypatch.setattr(
        generator,
        "_candidate_returns",
        lambda normalization_factor=1.0: np.array([-1e308, 1e308]),
    )
    with np.errstate(over="ignore"):
        with pytest.raises(cp.SolverError, match="non-finite separation value"):
            generator._separate(
                normalization_factor=1.0, cut_normalization_factors=cuts
            )


def test_non_finite_normalized_gmd_is_rejected():
    returns = np.array([[-1e300], [1e300]])
    weights = cp.Variable(1)
    generator = _GiniMeanDifference(returns, weights, cp.Constant(1.0))
    weights.value = np.ones(1)
    generator.expression.value = 0.0
    cuts = {generator._initial_permutation: 1.0}

    # Homogeneous GMD and normalized returns are finite, but normalized GMD
    # overflows when the ordered weighted sum combines both observations.
    with np.errstate(over="ignore"):
        with pytest.raises(cp.SolverError, match="non-finite normalized value"):
            generator._separate(
                normalization_factor=1e-8, cut_normalization_factors=cuts
            )


@pytest.mark.parametrize(
    ("context", "objective_function"),
    [
        ("minimum_risk", ObjectiveFunction.MINIMIZE_RISK),
        ("maximum_utility", ObjectiveFunction.MAXIMIZE_UTILITY),
        ("maximum_ratio", ObjectiveFunction.MAXIMIZE_RATIO),
    ],
)
def test_mean_risk_matches_independent_pairwise_formulation(
    context, objective_function
):
    returns = _make_returns()
    reference_weights, reference_risk, reference_return = _solve_pairwise_reference(
        returns, context
    )
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=objective_function,
        risk_aversion=3.0,
    )
    model.fit(returns)

    np.testing.assert_allclose(model.weights_, reference_weights, rtol=2e-3, atol=3e-4)
    np.testing.assert_allclose(
        model.problem_values_["risk"], reference_risk, rtol=5e-5, atol=2e-8
    )
    np.testing.assert_allclose(
        model.problem_values_["expected_return"],
        reference_return,
        rtol=5e-5,
        atol=5e-8,
    )


def test_maximum_gmd_constraint_matches_pairwise_formulation():
    returns = _make_returns()
    minimum = MeanRisk(risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE).fit(returns)
    concentrated_risk = max(
        _pairwise_gmd(returns[:, asset]) for asset in range(returns.shape[1])
    )
    risk_limit = minimum.problem_values_["risk"] + 0.35 * (
        concentrated_risk - minimum.problem_values_["risk"]
    )

    reference_weights, reference_risk, reference_return = _solve_pairwise_reference(
        returns, "maximum_return_under_risk", risk_limit=risk_limit
    )
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_gini_mean_difference=risk_limit,
    ).fit(returns)

    np.testing.assert_allclose(model.weights_, reference_weights, rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(
        model.problem_values_["risk"], reference_risk, rtol=3e-6, atol=2e-8
    )
    np.testing.assert_allclose(
        model.problem_values_["expected_return"],
        reference_return,
        rtol=3e-6,
        atol=2e-8,
    )
    assert _pairwise_gmd(returns @ model.weights_) <= risk_limit + 2e-8


def test_gmd_constraint_with_different_primary_risk():
    returns = _make_returns()
    minimum = MeanRisk(risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE).fit(returns)
    risk_limit = minimum.problem_values_["risk"] * 1.2
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_gini_mean_difference=risk_limit,
    ).fit(returns)

    assert _pairwise_gmd(returns @ model.weights_) <= risk_limit + 2e-8
    assert np.isfinite(model.problem_values_["risk"])


def test_maximum_ratio_separates_in_homogeneous_variable_space():
    returns = _make_returns(n_observations=60)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        save_problem=True,
    ).fit(returns)

    factor = float(model.problem_values_["factor"])
    exact_normalized = _pairwise_gmd(returns @ model.weights_)
    exact_homogeneous = _pairwise_gmd(returns @ (model.weights_ * factor))
    epigraph = next(
        variable
        for variable in model.problem_.variables()
        if variable.name() == "gmd_epigraph"
    )
    normalized_violation = (exact_homogeneous - float(epigraph.value)) / factor
    tolerance = 1e-8 + 1e-8 * abs(exact_normalized)

    np.testing.assert_allclose(
        model.problem_values_["risk"], exact_normalized, rtol=1e-12, atol=1e-12
    )
    assert normalized_violation <= tolerance


@pytest.mark.parametrize("scale", [1e6, 1e7, 1e8])
def test_maximum_ratio_small_factor_matches_pairwise_reference(scale):
    returns = _scaled_ratio_returns(scale)
    _, reference_risk, reference_return = _solve_pairwise_reference(
        returns, "maximum_ratio"
    )
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        save_problem=True,
    ).fit(returns)

    factor = float(model.problem_values_["factor"])
    empirical_risk = _pairwise_gmd(returns @ model.weights_)
    production_ratio = returns.mean(axis=0) @ model.weights_ / empirical_risk
    reference_ratio = reference_return / reference_risk
    epigraph = next(
        variable
        for variable in model.problem_.variables()
        if variable.name() == "gmd_epigraph"
    )
    homogeneous_risk = _pairwise_gmd(returns @ (model.weights_ * factor))
    normalized_violation = (homogeneous_risk - float(epigraph.value)) / factor

    assert factor < 1.1e-3
    assert production_ratio >= reference_ratio * (1 - 2e-4)
    np.testing.assert_allclose(
        model.problem_values_["risk"], empirical_risk, atol=2e-8, rtol=5e-5
    )
    assert normalized_violation <= 1.1e-8


def test_maximum_ratio_small_factor_respects_maximum_gmd():
    returns = 0.05 * _make_returns() + np.array([1e8] * 5 + [1e9])
    risk_limit = 5e-4
    unconstrained = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    ).fit(returns)
    _, reference_risk, reference_return = _solve_pairwise_reference(
        returns, "maximum_ratio", risk_limit=risk_limit
    )
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        max_gini_mean_difference=risk_limit,
        save_problem=True,
    ).fit(returns)

    empirical_risk = _pairwise_gmd(returns @ model.weights_)
    production_ratio = returns.mean(axis=0) @ model.weights_ / empirical_risk
    reference_ratio = reference_return / reference_risk
    assert _pairwise_gmd(returns @ unconstrained.weights_) > risk_limit + 3e-4
    assert float(model.problem_values_["factor"]) < 2e-5
    assert empirical_risk <= risk_limit + 1e-8
    assert production_ratio >= reference_ratio * (1 - 1e-3)
    np.testing.assert_allclose(
        model.problem_values_["risk"], empirical_risk, atol=2e-8, rtol=5e-5
    )


def test_small_factor_maximum_gmd_with_different_primary_risk():
    returns = 0.05 * _make_returns() + np.array([1e8] * 5 + [1e9])
    risk_limit = 4e-4
    unconstrained = MeanRisk(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    ).fit(returns)
    model = MeanRisk(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        max_gini_mean_difference=risk_limit,
    ).fit(returns)

    assert _pairwise_gmd(returns @ unconstrained.weights_) > 8e-4
    assert float(model.problem_values_["factor"]) < 1e-5
    # The active risk limit and the epigraph facet are separate solver constraints.
    # Allow the bounded master-feasibility residual in addition to separation error.
    assert _pairwise_gmd(returns @ model.weights_) <= risk_limit + 5e-8


def test_invalid_ratio_factor_uses_solver_failure_lifecycle():
    weights = cp.Variable(2)
    factor = cp.Variable(nonneg=True)
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(weights)),
        [cp.sum(weights) == 0, factor == 0],
    )

    with pytest.raises(cp.SolverError, match="Solver 'CLARABEL' failed"):
        _solve(
            w=weights,
            factor=factor,
            expressions={"factor": factor},
            problem=problem,
            solver="CLARABEL",
            solver_params={},
            risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
            scale_objective=cp.Constant(1),
        )


@pytest.mark.parametrize("invalid_weight", [np.nan, np.inf, -np.inf])
def test_non_finite_solver_weights_are_rejected(monkeypatch, invalid_weight):
    weights = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(weights)))

    def solve(**kwargs):
        # Bypass CVXPY's assignment validation to simulate invalid solver output.
        weights.save_value(np.array([invalid_weight, 1.0]))

    monkeypatch.setattr(problem, "solve", solve)
    with pytest.raises(cp.SolverError, match="Solver 'CLARABEL' failed"):
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


@pytest.mark.parametrize("factor", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_ratio_factor_values_are_rejected(factor):
    with pytest.raises(cp.SolverError, match="invalid homogeneous"):
        _validated_factor_value(factor)


@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_user_limit_master_is_never_accepted(raise_on_failure):
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        solver_params={"max_iter": 1},
        raise_on_failure=raise_on_failure,
    )

    if raise_on_failure:
        with pytest.raises(cp.SolverError, match="status 'user_limit'"):
            model.fit(_make_returns())
    else:
        with pytest.warns(UserWarning, match="status 'user_limit'"):
            model.fit(_make_returns())
        assert model.weights_ is None
        assert "status 'user_limit'" in model.error_


def test_construction_failure_cannot_leak_generator_state():
    def fail_objective(_):
        raise RuntimeError("forced construction failure")

    returns = _make_returns()
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        add_objective=fail_objective,
    )
    with pytest.raises(TypeError, match="add_objective"):
        model.fit(returns)

    cloned = clone(model).set_params(add_objective=None)
    cloned.fit(returns[::-1])
    np.testing.assert_allclose(
        cloned.problem_values_["risk"], _pairwise_gmd(returns[::-1] @ cloned.weights_)
    )

    model.set_params(risk_measure=RiskMeasure.VARIANCE, add_objective=None)
    model.fit(returns)
    assert np.isfinite(model.weights_).all()


def test_costs_fees_and_target_weights_preserve_gmd_semantics():
    returns = _make_returns()
    n_assets = returns.shape[1]
    previous_weights = np.full(n_assets, 1 / n_assets)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_aversion=0.2,
        transaction_costs=np.linspace(0.0001, 0.0006, n_assets),
        management_fees=np.linspace(0.0002, 0.0012, n_assets),
        previous_weights=previous_weights,
    ).fit(returns)
    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(returns @ model.weights_),
        rtol=1e-12,
        atol=1e-12,
    )

    target_weights = np.arange(1, n_assets + 1, dtype=float)
    target_weights /= target_weights.sum()
    target_model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        target_weights=target_weights,
    ).fit(returns)
    active_risk = _pairwise_gmd(returns @ (target_model.weights_ - target_weights))
    np.testing.assert_allclose(target_model.problem_values_["risk"], active_risk)
    assert active_risk <= 2e-8


def test_target_weights_maximum_ratio_uses_normalized_active_weights():
    returns = _make_returns(n_observations=60)
    target_weights = np.arange(1, returns.shape[1] + 1, dtype=float)
    target_weights /= target_weights.sum()
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        target_weights=target_weights,
    ).fit(returns)

    active_risk = _pairwise_gmd(returns @ (model.weights_ - target_weights))
    np.testing.assert_allclose(
        model.problem_values_["risk"], active_risk, rtol=2e-7, atol=2e-10
    )


def test_short_selling_linear_and_other_risk_constraints():
    returns = _make_returns()
    n_assets = returns.shape[1]
    variance_limit = float(np.var(returns @ np.full(n_assets, 1 / n_assets), ddof=1))
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        min_weights=-0.2,
        max_weights=0.7,
        min_return=0.0004,
        left_inequality=np.array([[1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]),
        right_inequality=np.array([0.6]),
        max_variance=variance_limit * 1.05,
    ).fit(returns)

    assert np.all(model.weights_ >= -0.2 - 1e-8)
    assert np.all(model.weights_ <= 0.7 + 1e-8)
    assert model.weights_[:2].sum() <= 0.6 + 1e-8
    assert returns.mean(axis=0) @ model.weights_ >= 0.0004 - 1e-8
    assert np.var(returns @ model.weights_, ddof=1) <= variance_limit * 1.05 + 1e-8
    np.testing.assert_allclose(
        model.problem_values_["risk"], _pairwise_gmd(returns @ model.weights_)
    )


def test_market_neutral_maximum_ratio():
    returns = _make_returns()
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        budget=0,
        min_weights=-1,
        max_weights=1,
    ).fit(returns)

    np.testing.assert_allclose(model.weights_.sum(), 0, atol=1e-9)
    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(returns @ model.weights_),
        rtol=1e-7,
        atol=1e-10,
    )


def test_parameter_sweep_continues_after_solver_failure():
    returns = _make_returns()
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        min_return=[0.0002, 1.0, 0.0003],
        raise_on_failure=False,
    )
    with pytest.warns(UserWarning, match="Solver 'CLARABEL' failed"):
        model.fit(returns)

    assert model.weights_.shape == (3, returns.shape[1])
    assert not np.isnan(model.weights_[[0, 2]]).any()
    assert np.isnan(model.weights_[1]).all()
    assert model.error_[0] is None
    assert model.error_[1] is not None
    assert model.error_[2] is None
    for index in (0, 2):
        np.testing.assert_allclose(
            model.problem_values_[index]["risk"],
            _pairwise_gmd(returns @ model.weights_[index]),
        )


def test_efficient_frontier_uses_independent_target_masters():
    returns = _make_returns(n_observations=60)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        efficient_frontier_size=4,
        save_problem=True,
    ).fit(returns)

    assert model.weights_.shape == (4, returns.shape[1])
    for weights, values in zip(model.weights_, model.problem_values_, strict=True):
        np.testing.assert_allclose(
            values["risk"], _pairwise_gmd(returns @ weights), rtol=1e-12, atol=1e-12
        )
    assert any(
        variable.name() == "gmd_epigraph" for variable in model.problem_.variables()
    )


@pytest.mark.parametrize(
    ("targets", "last_successful_target"),
    [
        ([0.0002, 1.0, 0.0003], 0.0003),
        ([0.0002, 0.0003, 1.0], 0.0003),
    ],
)
def test_save_problem_keeps_final_successful_parameter_target(
    targets, last_successful_target
):
    returns = _make_returns()
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        min_return=targets,
        raise_on_failure=False,
        save_problem=True,
    )
    with pytest.warns(UserWarning, match="Solver 'CLARABEL' failed"):
        model.fit(returns)

    assert model.problem_.status == cp.OPTIMAL
    assert float(model.problem_.parameters()[0].value) == last_successful_target
    successful_index = max(i for i, error in enumerate(model.error_) if error is None)
    expected_weights = model.weights_[successful_index]
    saved_weights = next(
        variable
        for variable in model.problem_.variables()
        if variable.shape == (returns.shape[1],)
    )
    np.testing.assert_allclose(
        saved_weights.value,
        expected_weights * model.problem_values_[successful_index]["factor"],
    )


def test_save_problem_all_parameter_targets_fail_cleanly():
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        raise_on_failure=False,
        save_problem=True,
    )
    returns = _make_returns()
    model.fit(returns)
    assert model.problem_.status == cp.OPTIMAL

    model.set_params(min_return=[1.0, 2.0])
    with pytest.warns(UserWarning, match="All 2 optimizations failed"):
        model.fit(returns)

    assert model.weights_ is None
    assert not hasattr(model, "problem_")


def test_failed_parameter_target_cuts_do_not_poison_next_target(monkeypatch):
    original_solve = _GiniMeanDifference.solve
    original_separate = _GiniMeanDifference._separate
    solve_count = 0
    calls_for_target = 0
    base_constraint_counts = []

    def solve(self, problem, solver, solver_params, factor):
        nonlocal solve_count, calls_for_target
        solve_count += 1
        calls_for_target = 0
        base_constraint_counts.append(len(problem.constraints))
        return original_solve(self, problem, solver, solver_params, factor)

    def fail_second_target_after_a_cut(
        self, normalization_factor, cut_normalization_factors
    ):
        nonlocal calls_for_target
        calls_for_target += 1
        if solve_count == 2:
            if calls_for_target == 1:
                permutation = tuple(np.roll(np.arange(len(self._returns)), 1))
                return self._create_cut(permutation, normalization_factor)
            raise cp.SolverError("forced failure after generated cut")
        return original_separate(self, normalization_factor, cut_normalization_factors)

    monkeypatch.setattr(_GiniMeanDifference, "solve", solve)
    monkeypatch.setattr(
        _GiniMeanDifference, "_separate", fail_second_target_after_a_cut
    )
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        min_return=[0.0002, 0.0003, 0.0004],
        raise_on_failure=False,
    )
    with pytest.warns(UserWarning, match="forced failure after generated cut"):
        model.fit(_make_returns())

    assert solve_count == 3
    assert len(set(base_constraint_counts)) == 1
    assert model.error_[0] is None
    assert model.error_[1].endswith("forced failure after generated cut")
    assert model.error_[2] is None
    assert np.isnan(model.weights_[1]).all()
    assert np.isfinite(model.weights_[[0, 2]]).all()


def test_partial_fit_rebuilds_gmd_generator_for_updated_returns():
    returns = _make_returns(n_observations=80)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=20),
            covariance_estimator=EWCovariance(half_life=20),
        ),
    )
    model.partial_fit(returns[:40])
    first_weights = model.weights_.copy()
    model.partial_fit(returns[40:])

    assert not np.array_equal(model.weights_, first_weights)
    fitted_returns = model.prior_estimator_.return_distribution_.returns
    np.testing.assert_allclose(
        model.problem_values_["risk"], _pairwise_gmd(fitted_returns @ model.weights_)
    )


def test_independent_fits_do_not_reuse_old_return_matrix_cuts():
    first_returns = _make_returns(n_observations=40, seed=1)
    second_returns = _make_returns(n_observations=70, seed=2)
    model = MeanRisk(risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE)
    model.fit(first_returns)
    first_weights = model.weights_.copy()
    model.fit(second_returns)

    assert not np.array_equal(model.weights_, first_weights)
    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(second_returns @ model.weights_),
    )


def test_exact_return_ties_converge_deterministically():
    base = _make_returns(n_observations=20)
    returns = np.repeat(base, 2, axis=0)
    first = MeanRisk(risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE).fit(returns)
    second = MeanRisk(risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE).fit(returns)

    np.testing.assert_allclose(first.weights_, second.weights_)
    np.testing.assert_allclose(
        first.problem_values_["risk"], _pairwise_gmd(returns @ first.weights_)
    )


def test_save_problem_contains_final_cuts_without_dense_variables():
    returns = _make_returns(n_observations=500, n_assets=10)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE, save_problem=True
    ).fit(returns)

    gmd_constraints = [
        constraint
        for constraint in model.problem_.constraints
        if any(variable.name() == "gmd_epigraph" for variable in constraint.variables())
    ]
    assert 1 < len(gmd_constraints) < returns.shape[0]
    assert model.problem_.size_metrics.num_scalar_variables <= returns.shape[1] + 1
    assert model.problem_.size_metrics.num_scalar_leq_constr < 5 * returns.shape[0]
    np.testing.assert_allclose(
        model.problem_values_["risk"], _pairwise_gmd(returns @ model.weights_)
    )


def test_risk_budgeting_with_costs_and_fees():
    returns = _make_returns(n_observations=80)
    n_assets = returns.shape[1]
    model = RiskBudgeting(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        transaction_costs=0.0001,
        management_fees=np.linspace(0.0, 0.001, n_assets),
        previous_weights=np.full(n_assets, 1 / n_assets),
    ).fit(returns)
    portfolio = model.predict(returns)

    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(returns @ model.weights_),
        rtol=1e-12,
        atol=1e-12,
    )
    contributions = portfolio.contribution(measure=RiskMeasure.GINI_MEAN_DIFFERENCE)
    np.testing.assert_allclose(
        contributions, np.full(n_assets, contributions.mean()), rtol=2e-3, atol=2e-6
    )


@pytest.mark.skipif(
    "SCIP" not in cp.installed_solvers(), reason="SCIP is not installed"
)
def test_scip_cardinality_compatibility():
    returns = _make_returns(n_observations=40)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        cardinality=3,
        solver="SCIP",
    ).fit(returns)

    assert np.count_nonzero(np.abs(model.weights_) > 1e-8) <= 3
    np.testing.assert_allclose(
        model.problem_values_["risk"],
        _pairwise_gmd(returns @ model.weights_),
        rtol=1e-10,
        atol=1e-10,
    )


@pytest.mark.skipif(
    "SCIP" not in cp.installed_solvers(), reason="SCIP is not installed"
)
@pytest.mark.parametrize(
    ("objective_function", "risk_limit"),
    [
        (ObjectiveFunction.MAXIMIZE_RATIO, None),
        (ObjectiveFunction.MAXIMIZE_RETURN, 0.009),
    ],
)
def test_scip_cardinality_ratio_and_maximum_gmd(objective_function, risk_limit):
    returns = _make_returns(n_observations=30, n_assets=5)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=objective_function,
        max_gini_mean_difference=risk_limit,
        cardinality=3,
        solver="SCIP",
    ).fit(returns)

    empirical_risk = _pairwise_gmd(returns @ model.weights_)
    assert np.count_nonzero(np.abs(model.weights_) > 1e-8) <= 3
    np.testing.assert_allclose(
        model.problem_values_["risk"], empirical_risk, rtol=1e-9, atol=1e-10
    )
    if risk_limit is not None:
        assert empirical_risk <= risk_limit + 1e-8
        np.testing.assert_allclose(empirical_risk, risk_limit, atol=1e-10)


@pytest.mark.skipif(
    "SCIP" not in cp.installed_solvers(), reason="SCIP is not installed"
)
def test_scip_time_limited_master_fails_cleanly():
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        cardinality=3,
        solver="SCIP",
        solver_params={"scip_params": {"limits/time": 0.0}},
        raise_on_failure=False,
    )
    with pytest.warns(UserWarning, match="Solver 'SCIP' failed"):
        model.fit(_make_returns(n_observations=40))

    assert model.weights_ is None
    assert "Solver 'SCIP' failed" in model.error_


def test_constraint_generation_failure_uses_existing_failure_lifecycle(monkeypatch):
    def fail_separation(self, normalization_factor, cut_normalization_factors):
        raise cp.SolverError("forced GMD separation failure")

    monkeypatch.setattr(_GiniMeanDifference, "_separate", fail_separation)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE, raise_on_failure=False
    )
    with pytest.warns(UserWarning, match="forced GMD separation failure"):
        model.fit(_make_returns())

    assert model.weights_ is None
    assert model.error_.endswith("forced GMD separation failure")
    assert not hasattr(model, "problem_values_")


def test_default_gmd_converges_beyond_former_cut_limit(monkeypatch):
    # Hugo's regression required 558 master solves, exceeding the former 500-cut
    # ceiling. A bounded problem must keep the asset-space formulation throughout.
    returns = np.random.default_rng(42).normal(0.0005, 0.01, (500, 20))

    def unexpected_fallback(self):
        pytest.fail("A bounded GMD master should not need a quadratic fallback")

    monkeypatch.setattr(
        _GiniMeanDifference, "_pairwise_constraints", unexpected_fallback
    )
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE, save_problem=True
    ).fit(returns)

    assert model.problem_.status == cp.OPTIMAL
    assert model.problem_.size_metrics.num_scalar_variables == returns.shape[1] + 1
    exact_risk = _pairwise_gmd(returns @ model.weights_)
    np.testing.assert_allclose(model.problem_values_["risk"], exact_risk, atol=1e-12)
    np.testing.assert_allclose(
        model.problem_values_["objective"], exact_risk, atol=5e-8
    )
    np.testing.assert_allclose(model.weights_.sum(), 1.0, atol=1e-8)
    assert model.weights_.min() >= -1e-8


@pytest.mark.parametrize(
    ("context", "objective_function", "risk_limit"),
    [
        ("maximum_return_under_risk", ObjectiveFunction.MAXIMIZE_RETURN, 0.01),
        ("maximum_utility", ObjectiveFunction.MAXIMIZE_UTILITY, None),
    ],
)
@pytest.mark.parametrize(
    "solver",
    [
        "CLARABEL",
        pytest.param(
            "SCIP",
            marks=pytest.mark.skipif(
                "SCIP" not in cp.installed_solvers(), reason="SCIP is not installed"
            ),
        ),
    ],
)
def test_unbounded_relaxation_matches_finite_pairwise_optimum(
    monkeypatch, context, objective_function, risk_limit, solver
):
    returns = np.random.default_rng(42).normal(0.0005, 0.01, (40, 6))
    _, reference_risk, reference_return = _solve_pairwise_reference(
        returns, context, risk_limit=risk_limit, risk_aversion=1.0, unrestricted=True
    )
    original_fallback = _GiniMeanDifference._pairwise_constraints
    fallback_count = 0

    def fallback(self):
        nonlocal fallback_count
        fallback_count += 1
        return original_fallback(self)

    monkeypatch.setattr(_GiniMeanDifference, "_pairwise_constraints", fallback)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=objective_function,
        min_weights=None,
        max_weights=None,
        max_gini_mean_difference=risk_limit,
        risk_aversion=1.0,
        save_problem=True,
        solver=solver,
    ).fit(returns)

    assert fallback_count == 1
    assert model.problem_.status == cp.OPTIMAL
    actual_risk = _pairwise_gmd(returns @ model.weights_)
    actual_return = returns.mean(axis=0) @ model.weights_
    np.testing.assert_allclose(model.problem_values_["risk"], actual_risk, atol=1e-12)
    if risk_limit is None:
        np.testing.assert_allclose(
            actual_return - actual_risk, reference_return - reference_risk, atol=1e-8
        )
    else:
        assert actual_risk <= risk_limit + 1e-8
        np.testing.assert_allclose(actual_return, reference_return, atol=1e-8)


@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_truly_unbounded_gmd_problem_is_not_accepted(raise_on_failure):
    base = np.random.default_rng(4).normal(0, 0.01, 20)
    # The long-short direction has positive return and zero GMD.
    returns = np.column_stack((base, base + 0.001))
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        min_weights=None,
        max_weights=None,
        max_gini_mean_difference=0.02,
        raise_on_failure=raise_on_failure,
        save_problem=True,
    )
    if raise_on_failure:
        with pytest.raises(cp.SolverError, match="status 'unbounded"):
            model.fit(returns)
    else:
        with pytest.warns(UserWarning, match="status 'unbounded"):
            model.fit(returns)
        assert model.weights_ is None
        assert "status 'unbounded" in model.error_
    assert not hasattr(model, "problem_")


@pytest.mark.parametrize("risk_limits", [[0.01, 0.001], [0.01, 0.001, 0.012]])
def test_pairwise_fallback_is_target_local_and_restores_saved_solution(
    monkeypatch, risk_limits
):
    returns = np.random.default_rng(42).normal(0.0005, 0.01, (40, 6))
    original_fallback = _GiniMeanDifference._pairwise_constraints
    fallback_count = 0

    def fallback(self):
        nonlocal fallback_count
        fallback_count += 1
        return original_fallback(self)

    monkeypatch.setattr(_GiniMeanDifference, "_pairwise_constraints", fallback)
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        min_weights=None,
        max_weights=None,
        max_gini_mean_difference=risk_limits,
        raise_on_failure=False,
        save_problem=True,
    )
    with pytest.warns(UserWarning, match="status 'infeasible"):
        model.fit(returns)

    assert fallback_count == len(risk_limits)
    assert model.error_[0] is None
    assert model.error_[1] is not None
    assert np.isnan(model.weights_[1]).all()
    successful_index = 0 if len(risk_limits) == 2 else 2
    assert model.error_[successful_index] is None
    last_limit = risk_limits[successful_index]
    assert model.problem_.status == cp.OPTIMAL
    assert float(model.problem_.parameters()[0].value) == last_limit
    weights = model.weights_[successful_index]
    np.testing.assert_allclose(_pairwise_gmd(returns @ weights), last_limit, atol=1e-8)
    # A GMD constraint must not replace the selected variance risk report. Its
    # epigraph can be loose because this problem maximizes return.
    reported_risk = model.problem_values_[successful_index]["risk"]
    assert reported_risk >= np.var(returns @ weights, ddof=1) - 1e-9
    assert reported_risk != pytest.approx(last_limit)
    for constraint in model.problem_.constraints:
        assert np.max(constraint.violation()) <= 1e-7


@pytest.mark.parametrize("factor_value", [0.0, 1e-14, -1.0])
@pytest.mark.parametrize("variable_factor", [False, True])
def test_gmd_solve_rejects_invalid_normalization_factor(factor_value, variable_factor):
    returns = _make_returns(n_observations=5, n_assets=3)
    weights = cp.Variable(3)
    gmd = _GiniMeanDifference(returns, weights, cp.Constant(1.0))
    factor = cp.Variable() if variable_factor else cp.Constant(factor_value)
    constraints = [weights == np.full(3, 1 / 3), gmd.initial_constraint]
    if variable_factor:
        constraints.append(factor == factor_value)
    problem = cp.Problem(
        cp.Minimize(gmd.expression),
        constraints,
    )
    with pytest.raises(
        cp.SolverError, match="invalid homogeneous normalization factor"
    ):
        gmd.solve(problem, "CLARABEL", {}, factor)


def test_inaccurate_master_warns_after_successful_separation(monkeypatch):
    original_solve = cp.Problem.solve

    def solve(self, *args, **kwargs):
        result = original_solve(self, *args, **kwargs)
        if self.status == cp.OPTIMAL:
            self._status = cp.OPTIMAL_INACCURATE
        return result

    monkeypatch.setattr(cp.Problem, "solve", solve)
    returns = _make_returns(n_observations=20)
    model = MeanRisk(risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE, save_problem=True)
    with pytest.warns(UserWarning, match="Solution may be inaccurate"):
        model.fit(returns)

    assert model.problem_.status == cp.OPTIMAL_INACCURATE
    exact_risk = _pairwise_gmd(returns @ model.weights_)
    np.testing.assert_allclose(model.problem_values_["risk"], exact_risk, atol=1e-12)
    np.testing.assert_allclose(
        model.problem_values_["objective"], exact_risk, atol=5e-8
    )


def test_unrestricted_ratio_recovers_from_nonpositive_relaxation_factor(monkeypatch):
    rng = np.random.default_rng(20260)
    returns = rng.normal(size=(18, 3))
    returns = (
        (returns - returns.mean(axis=0)) * 0.01 + np.linspace(0.0002, 0.0012, 3)
    ) * 0.01
    _, reference_risk, reference_return = _solve_pairwise_reference(
        returns, "maximum_ratio", unrestricted=True
    )
    original_fallback = _GiniMeanDifference._pairwise_constraints
    fallback_count = 0

    def fallback(self):
        nonlocal fallback_count
        fallback_count += 1
        return original_fallback(self)

    monkeypatch.setattr(_GiniMeanDifference, "_pairwise_constraints", fallback)
    model = MeanRisk(
        risk_measure=RiskMeasure.GINI_MEAN_DIFFERENCE,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        min_weights=None,
        max_weights=None,
        save_problem=True,
    ).fit(returns)

    # Another solver version may choose a normalizable point in the initial flat
    # relaxation and need no fallback. Either path must recover the same optimum.
    assert fallback_count <= 1
    assert model.problem_values_["factor"] > 0
    actual_risk = _pairwise_gmd(returns @ model.weights_)
    actual_return = returns.mean(axis=0) @ model.weights_
    np.testing.assert_allclose(
        actual_return / actual_risk, reference_return / reference_risk, rtol=1e-6
    )
    np.testing.assert_allclose(model.problem_values_["risk"], actual_risk, atol=1e-12)
    np.testing.assert_allclose(model.weights_.sum(), 1.0, atol=1e-8)
