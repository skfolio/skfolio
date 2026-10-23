"""External projection input validation."""

import numpy as np
import pandas as pd
import pytest

from skfolio.optimization.parametric._projection_context import projection_context


def test_align_named_covariance_and_holdings():
    cov, held = projection_context(
        ["a", "b"],
        [0],
        pd.DataFrame([[2, 0], [0, 1]], index=["b", "a"], columns=["b", "a"]),
        pd.DataFrame([[0.3, 0.7]], index=[0], columns=["b", "a"]),
    )
    np.testing.assert_array_equal(cov[0], np.diag([1, 2]))
    np.testing.assert_allclose(held, [[0.7, 0.3]])


@pytest.mark.parametrize(
    "cov", [[[1, 2], [0, 1]], [[1, 2], [2, 1]], [[1, np.nan], [np.nan, 1]]]
)
def test_reject_invalid_covariance(cov):
    with pytest.raises(ValueError):
        projection_context(["a", "b"], [0], covariance=cov)


def test_holdings_cannot_broadcast():
    with pytest.raises(ValueError, match="per decision"):
        projection_context(["a", "b"], [0, 1], holdings=[0.5, 0.5])


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_singular_covariance_roundoff(scale):
    original = scale * np.outer([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    before = original.copy()
    actual, _ = projection_context(["a", "b", "c"], [0, 1], covariance=original)
    np.testing.assert_allclose(
        actual, np.broadcast_to(original, (2, 3, 3)), rtol=1e-13, atol=scale * 1e-16
    )
    assert np.linalg.eigvalsh(actual).min() >= -scale * 1e-16
    np.testing.assert_array_equal(original, before)


@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_covariance_tolerance_is_relative(scale):
    with pytest.raises(ValueError, match="positive semidefinite"):
        projection_context(["a", "b"], [0], covariance=scale * np.diag([1.0, -1e-6]))


def test_zero_covariance_is_valid():
    actual, _ = projection_context(["a", "b"], [0], covariance=np.zeros((2, 2)))
    np.testing.assert_array_equal(actual, np.zeros((1, 2, 2)))
