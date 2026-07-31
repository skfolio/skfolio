import numpy as np
import pytest

from skfolio.prior import CovarianceSqrt


def test_covariance_sqrt_validation_accepts_diagonal_only():
    sqrt = CovarianceSqrt(diagonal=np.ones(3))

    assert sqrt.components == ()
    np.testing.assert_array_equal(sqrt.diagonal, np.ones(3))


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "At least one covariance square root component is required."),
        (
            {"components": (np.ones(3),)},
            "Covariance square root components must be 2D arrays.",
        ),
        (
            {"components": (np.ones((3, 0)),)},
            "Covariance square root components cannot be empty.",
        ),
        (
            {"components": (np.ones((3, 2)), np.ones((4, 2)))},
            "Covariance square root components must have matching row counts.",
        ),
        (
            {"diagonal": np.ones((3, 1))},
            "Covariance square root diagonal must be a 1D array.",
        ),
        (
            {"diagonal": np.array([])},
            "Covariance square root diagonal cannot be empty.",
        ),
        (
            {"components": (np.ones((3, 2)),), "diagonal": np.ones(4)},
            "Covariance square root diagonal must match component row counts.",
        ),
    ],
)
def test_covariance_sqrt_validation_rejects_invalid_shapes(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CovarianceSqrt(**kwargs)
