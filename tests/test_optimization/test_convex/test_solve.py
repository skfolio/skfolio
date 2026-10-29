"""Shared solver validation and reporting, independent of the risk formulation."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from skfolio import Portfolio, RiskMeasure
from skfolio.exceptions import ConvexOptimizationError
from skfolio.optimization import (
    DistributionallyRobustCVaR,
    MeanRisk,
    ObjectiveFunction,
    RiskBudgeting,
)
from skfolio.optimization.convex._base import _solve, _solve_error_message


@pytest.mark.parametrize("params_string", ["", " with parameters 0.1"])
@pytest.mark.parametrize(
    ("status", "expected", "reduced_accuracy"),
    [
        (cp.INFEASIBLE, "no solution satisfies all constraints", False),
        (cp.UNBOUNDED, "the objective can improve without limit", False),
        (cp.INFEASIBLE_INACCURATE, "reports infeasibility", True),
        (cp.UNBOUNDED_INACCURATE, "reports unboundedness", True),
        (
            cp.settings.INFEASIBLE_OR_UNBOUNDED,
            "but cannot distinguish between the two",
            False,
        ),
        (cp.SOLVER_ERROR, "Solver 'CLARABEL' failed", False),
        (cp.USER_LIMIT, "Solver 'CLARABEL' failed", False),
        (None, "Solver 'CLARABEL' failed", False),
    ],
)
def test_solve_error_message_depends_on_status(
    status, expected, reduced_accuracy, params_string
):
    message = _solve_error_message("CLARABEL", status, params_string)
    assert expected in message
    assert "Solver 'CLARABEL'" in message
    if params_string:
        assert params_string in message
    else:
        assert "with parameters" not in message
    assert ("reduced numerical accuracy" in message) == reduced_accuracy
    if reduced_accuracy:
        assert "Check solver settings and problem scaling" in message
        assert "try another solver" in message


@pytest.fixture
def solve_kwargs():
    weights = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(weights)), [cp.sum(weights) == 1])
    return dict(
        w=weights,
        factor=cp.Constant(1),
        expressions={},
        problem=problem,
        solver="CLARABEL",
        solver_params={},
    )


@pytest.mark.parametrize("status", [cp.OPTIMAL_INACCURATE, cp.USER_LIMIT])
def test_usable_nonoptimal_solution_warns(solve_kwargs, monkeypatch, status):
    problem = solve_kwargs["problem"]
    original_solve = problem.solve

    def solve(**kwargs):
        result = original_solve(**kwargs)
        problem._status = status
        return result

    monkeypatch.setattr(problem, "solve", solve)
    with pytest.warns(UserWarning, match="Solution may be inaccurate"):
        weights, _ = _solve(**solve_kwargs)
    np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-8)


@pytest.mark.parametrize(
    "value", [None, 0.0, -1.0, 1e-14, 1e-12, np.nan, np.inf, -np.inf]
)
def test_invalid_normalization_factor_is_rejected(solve_kwargs, value):
    factor = cp.Variable()
    factor.save_value(None if value is None else np.asarray(value))
    solve_kwargs["factor"] = factor
    with pytest.raises(ConvexOptimizationError, match="invalid homogeneous"):
        _solve(**solve_kwargs)


@pytest.mark.parametrize(
    "risk_measure", [RiskMeasure.VARIANCE, RiskMeasure.GINI_MEAN_DIFFERENCE]
)
@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_ratio_with_zero_normalization_factor_is_rejected(
    risk_measure, raise_on_failure
):
    returns = np.random.default_rng(2).normal(0, 0.01, (20, 3))
    returns += np.array([0.001, 0.002, 0.004]) - returns.mean(axis=0)
    # Together with the default budget, this equality forces factor == 0.
    # A solver can return a tiny positive residual instead of exact zero.
    model = MeanRisk(
        risk_measure=risk_measure,
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        min_weights=None,
        max_weights=None,
        linear_constraints=["A + B + C == 0"],
        raise_on_failure=raise_on_failure,
        save_problem=True,
    )
    expectation = (
        pytest.raises(ConvexOptimizationError, match="invalid homogeneous")
        if raise_on_failure
        else pytest.warns(UserWarning, match="invalid homogeneous")
    )
    with expectation:
        model.fit(pd.DataFrame(returns, columns=["A", "B", "C"]))
    assert model.problem_values_ is None
    if not raise_on_failure:
        assert model.weights_ is None
    assert isinstance(model.problem_, cp.Problem)


@pytest.mark.parametrize("invalid_weight", [np.nan, np.inf, -np.inf])
def test_non_finite_solver_weights_are_rejected(
    solve_kwargs, monkeypatch, invalid_weight
):
    problem = solve_kwargs["problem"]

    def solve(**kwargs):
        solve_kwargs["w"].save_value(np.array([invalid_weight, 1.0]))
        problem._status = cp.OPTIMAL

    monkeypatch.setattr(problem, "solve", solve)
    with pytest.raises(ConvexOptimizationError, match="non-finite weights"):
        _solve(**solve_kwargs)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_reported_values_are_rejected(solve_kwargs, value):
    solve_kwargs["expressions"] = {"risk": cp.Constant(value)}
    with pytest.raises(ConvexOptimizationError, match="non-finite 'risk'"):
        _solve(**solve_kwargs)


def test_reported_value_normalization_overflow_is_rejected(solve_kwargs):
    factor = cp.Constant(1e-11)
    solve_kwargs["factor"] = factor
    solve_kwargs["expressions"] = {"risk": cp.Constant(1e308) / factor}
    with pytest.raises(ConvexOptimizationError, match="non-finite 'risk'"):
        _solve(**solve_kwargs)


def test_quadratic_risk_normalization_overflow_is_rejected(solve_kwargs):
    factor = cp.Constant(1e-5)
    solve_kwargs["factor"] = factor
    solve_kwargs["expressions"] = {"risk": cp.Constant(1e300) / factor / factor}
    with pytest.raises(ConvexOptimizationError, match="non-finite 'risk'"):
        _solve(**solve_kwargs)


def test_reporting_expression_does_not_enter_problem(solve_kwargs):
    weights = solve_kwargs["w"]
    factor = cp.Constant(2)
    solve_kwargs["factor"] = factor
    solve_kwargs["expressions"] = {
        "reported": cp.norm1(weights - [0.2, 0.8]) / factor,
        "risk": cp.Constant(3),
        "factor": factor,
    }
    result, values = _solve(**solve_kwargs)
    np.testing.assert_allclose(result, [0.25, 0.25], atol=1e-8)
    np.testing.assert_allclose(values["reported"], 0.3, atol=1e-8)
    assert values["risk"] == 3
    assert values["factor"] == 2
    assert cp.norm1 not in solve_kwargs["problem"].atoms()


@pytest.fixture
def reporting_returns():
    returns = np.random.default_rng(7).normal(0, 0.01, (32, 4))
    return returns + np.linspace(0.001, 0.003, 4) - returns.mean(axis=0)


@pytest.mark.parametrize("optimizer", [MeanRisk, RiskBudgeting])
@pytest.mark.parametrize(
    "risk_measure",
    [
        RiskMeasure.VARIANCE,
        RiskMeasure.SEMI_VARIANCE,
        RiskMeasure.STANDARD_DEVIATION,
        RiskMeasure.CVAR,
        RiskMeasure.GINI_MEAN_DIFFERENCE,
    ],
)
def test_reporting_uses_normalized_portfolio(
    reporting_returns, optimizer, risk_measure
):
    target = np.full(4, 0.05) if optimizer is MeanRisk else np.zeros(4)
    params = (
        dict(
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            target_weights=target,
            l1_coef=0.0001,
            l2_coef=0.00005,
        )
        if optimizer is MeanRisk
        else {}
    )
    model = optimizer(
        risk_measure=risk_measure, scale_objective=3, save_problem=True, **params
    ).fit(reporting_returns)
    values = model.problem_values_
    portfolio = model.predict(reporting_returns)
    active_portfolio = Portfolio(X=reporting_returns, weights=model.weights_ - target)
    np.testing.assert_allclose(values["expected_return"], portfolio.mean, atol=1e-12)
    np.testing.assert_allclose(
        values["risk"],
        getattr(active_portfolio, risk_measure.value),
        rtol=1e-4,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        values["objective"], model.problem_.value / 3, atol=1e-12
    )
    if optimizer is MeanRisk:
        # Preserve the existing convention for the transformed L2 penalty.
        regularization = model.l1_coef * np.abs(model.weights_).sum()
        regularization += (
            model.l2_coef * values["factor"] * np.square(model.weights_).sum()
        )
        np.testing.assert_allclose(values["regularization"], regularization, atol=1e-12)


def test_objective_reported_without_additional_expressions(reporting_returns):
    model = DistributionallyRobustCVaR(scale_objective=3, save_problem=True).fit(
        reporting_returns
    )
    assert set(model.problem_values_) == {"objective"}
    np.testing.assert_allclose(
        model.problem_values_["objective"], model.problem_.value / 3, atol=1e-12
    )
