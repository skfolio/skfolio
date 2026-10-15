"""Projection failures must preserve solver choices and meaningful errors."""

import cvxpy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online import ExponentiatedGradient
from skfolio.optimization.online._projection import project_convex


@pytest.mark.parametrize(
    "error",
    [
        cp.SolverError("backend failed"),
        TypeError("invalid option"),
        ValueError("invalid model"),
        cp.DCPError("not DCP"),
    ],
)
def test_solver_exception_never_retries(monkeypatch, error):
    calls = []

    def solve(problem, **kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(cp.Problem, "solve", solve)
    options = {"solver": "CLARABEL", "max_iter": 9}
    expected = RuntimeError if isinstance(error, cp.SolverError) else type(error)
    with pytest.raises(expected) as caught:
        project_convex(np.array([0.5, 0.5]), solver_params=options)
    assert calls == [{"solver": "CLARABEL", "verbose": False, "max_iter": 9}]
    assert options == {"solver": "CLARABEL", "max_iter": 9}
    if isinstance(error, cp.SolverError):
        assert caught.value.__cause__ is error
        assert "CLARABEL" in str(caught.value)
    else:
        assert caught.value is error


@pytest.mark.parametrize(
    "status", [cp.INFEASIBLE, cp.UNBOUNDED, cp.USER_LIMIT, cp.INFEASIBLE_INACCURATE]
)
def test_unsuccessful_status_rejects_available_weights(monkeypatch, status):
    def solve(problem, **kwargs):
        problem._status = status
        problem.variables()[0].save_value(np.array([0.5, 0.5]))

    monkeypatch.setattr(cp.Problem, "solve", solve)
    with pytest.raises(RuntimeError, match=status):
        project_convex(np.array([0.5, 0.5]))


@pytest.mark.parametrize(
    "weights", [None, np.array([np.nan, 0.5]), np.array([np.inf, 0.5])]
)
def test_nonfinite_or_missing_solution_is_rejected(monkeypatch, weights):
    def solve(problem, **kwargs):
        problem._status = cp.OPTIMAL
        problem.variables()[0].save_value(weights)

    monkeypatch.setattr(cp.Problem, "solve", solve)
    with pytest.raises(RuntimeError, match="finite weights"):
        project_convex(np.array([0.5, 0.5]))


@pytest.mark.parametrize("status", [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
def test_usable_status_is_preserved(monkeypatch, status):
    def solve(problem, **kwargs):
        problem._status = status
        problem.variables()[0].save_value(np.array([0.4, 0.6]))

    monkeypatch.setattr(cp.Problem, "solve", solve)
    assert_allclose(project_convex(np.array([0.5, 0.5])), [0.4, 0.6])


def test_unavailable_solver_is_not_replaced():
    with pytest.raises(RuntimeError, match="NOT_INSTALLED") as caught:
        project_convex(np.array([0.5, 0.5]), solver_params={"solver": "NOT_INSTALLED"})
    assert isinstance(caught.value.__cause__, cp.SolverError)


def test_bad_option_is_not_discarded():
    with pytest.raises(TypeError, match="unrecognized"):
        project_convex(
            np.array([0.5, 0.5]),
            solver_params={"solver": "CLARABEL", "not_a_solver_option": 1},
        )


def test_prior_failure_contract_applies_to_solver_failures():
    model = ExponentiatedGradient(add_constraints=lambda w: w[0] <= 0.6).fit([[0.1, 0]])
    model.set_params(solver="NOT_INSTALLED")
    with pytest.raises(RuntimeError, match="NOT_INSTALLED"):
        model.partial_fit([0.1, 0])
    with pytest.raises(RuntimeError, match="reset"):
        model.partial_fit([0.1, 0])
    model.set_params(solver="CLARABEL").fit([[0.1, 0]])
    assert model.weights_[0] <= 0.6 + 1e-6
