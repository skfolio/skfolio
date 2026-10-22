"""Regret uses the same comparator and solver options as direct benchmarking."""

import numpy as np
import pytest

from skfolio.optimization.convex import MeanRisk, ObjectiveFunction
from skfolio.optimization.online._regret import (
    _log_wealth_expr,
    _solve_best_constant_weights,
    _universal_dynamic_weights,
)


def test_regret_comparator_matches_direct_fit():
    returns = np.random.default_rng(421).normal(0.001, 0.02, (20, 3))
    parameters = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}
    direct = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        overwrite_expected_return=_log_wealth_expr,
        solver_params=parameters,
    ).fit(returns)
    actual = _solve_best_constant_weights(1 + returns, solver_params=parameters)
    np.testing.assert_allclose(actual, direct.weights_, atol=1e-7)


def test_zero_path_forwards_options_and_solver_errors():
    relatives = np.array([[1.01, 0.99], [0.99, 1.01]])
    with pytest.raises(ValueError, match=r"solver|Solver"):
        _universal_dynamic_weights(relatives, path_length=0, solver="NOT_A_SOLVER")
    with pytest.raises(TypeError, match=r"invalid|unrecognized"):
        _universal_dynamic_weights(
            relatives, path_length=0, solver_params={"invalid_option": 1}
        )


def test_missing_comparator_weights_raise_clear_error(monkeypatch):
    from skfolio.exceptions import ConvexOptimizationError

    def fit_without_weights(self, X):
        self.weights_ = None
        return self

    monkeypatch.setattr(MeanRisk, "fit", fit_without_weights)
    with pytest.raises(ConvexOptimizationError, match="infeasible weights"):
        _solve_best_constant_weights(np.array([[1.01, 0.99], [0.99, 1.01]]))


def test_best_constant_matches_independent_log_wealth_program():
    import cvxpy as cp

    returns = np.array([[0.2, -0.1], [-0.1, 0.2], [0.03, 0.01]])
    weights = cp.Variable(2)
    problem = cp.Problem(
        cp.Maximize(cp.sum(cp.log(1 + returns @ weights))),
        [weights >= 0, weights <= 1, cp.sum(weights) == 1],
    )
    problem.solve(solver="CLARABEL")
    actual = _solve_best_constant_weights(1 + returns)
    np.testing.assert_allclose(actual, weights.value, atol=1e-4)
    np.testing.assert_allclose(
        np.log1p(returns @ actual).sum(), problem.value, atol=1e-8
    )


@pytest.mark.parametrize("relatives", [[[0.0, 1.0]], [[-0.1, 1.0]], [[np.inf, 1.0]]])
def test_best_constant_preserves_invalid_return_rejection(relatives):
    with pytest.raises(ValueError):
        _solve_best_constant_weights(np.array(relatives))
