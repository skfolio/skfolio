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
