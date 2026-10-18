"""Regret uses the same comparator and solver options as direct benchmarking."""

import numpy as np
import pytest

from skfolio.optimization.online._benchmark import BestConstantRebalancedPortfolio
from skfolio.optimization.online._regret import (
    _solve_best_constant_weights,
    _universal_dynamic_weights,
)


def test_regret_comparator_matches_direct_fit():
    returns = np.random.default_rng(421).normal(0.001, 0.02, (20, 3))
    parameters = {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9, "tol_feas": 1e-9}
    direct = BestConstantRebalancedPortfolio(solver_params=parameters).fit(returns)
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
