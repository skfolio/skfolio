"""Regression checks for the tolerances used by online test assertions."""

import numpy as np
import pytest

from .utils import assert_box_budget, assert_simplex_trajectory


@pytest.mark.parametrize("kind", ["budget", "lower", "upper"])
@pytest.mark.parametrize("helper", [assert_box_budget, assert_simplex_trajectory])
def test_constraint_tolerance_is_respected(helper, kind):
    delta = 5e-8
    weights = {
        "budget": np.array([0.5, 0.5 + delta]),
        "lower": np.array([-delta, 0.5, 0.5 + delta]),
        "upper": np.array([1 + delta, -delta / 2, -delta / 2]),
    }[kind]
    if helper is assert_simplex_trajectory:
        weights = weights[None, :]
    helper(weights, atol=1e-7)
    with pytest.raises(AssertionError):
        helper(weights, atol=1e-9)
