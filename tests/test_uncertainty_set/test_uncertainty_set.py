"""Tests for uncertainty-set value objects."""

import numpy as np
import pytest

from skfolio.uncertainty_set import CompactCovarianceUncertaintySet, UncertaintySet


@pytest.mark.parametrize(
    ("norm", "expected"),
    [(1, np.inf), (2, 2.0), (np.inf, 1.0)],
)
def test_uncertainty_set_dual_norm(norm, expected):
    """The support-function norm is dual to the uncertainty-set norm."""
    uncertainty_set = UncertaintySet(radius=1, geometry=np.eye(2), norm=norm)

    assert uncertainty_set.dual_norm == expected
    assert isinstance(uncertainty_set.radius, float)
    assert isinstance(uncertainty_set.norm, float)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"geometry": np.ones(2)}, "geometry.*2D"),
        ({"radius": -1}, "radius.*non-negative"),
        ({"norm": 0.5}, "norm.*greater than or equal to 1"),
    ],
)
def test_uncertainty_set_rejects_invalid_parameters(kwargs, match):
    """Norm-ball parameters must describe a valid non-negative ball."""
    params = {"radius": 1, "geometry": np.eye(2), "norm": 2}
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        UncertaintySet(**params)


def test_compact_covariance_uncertainty_set_converts_inputs():
    """Compact covariance inputs are stored as floating NumPy arrays."""
    uncertainty_set = CompactCovarianceUncertaintySet(
        radius=2,
        metric_sqrt=[1, 2],
        basis=[[1], [0]],
    )

    assert uncertainty_set.radius == 2.0
    assert isinstance(uncertainty_set.metric_sqrt, np.ndarray)
    assert isinstance(uncertainty_set.basis, np.ndarray)
    assert np.issubdtype(uncertainty_set.metric_sqrt.dtype, np.floating)
    assert np.issubdtype(uncertainty_set.basis.dtype, np.floating)
    np.testing.assert_array_equal(uncertainty_set.metric_sqrt, [1.0, 2.0])
    np.testing.assert_array_equal(uncertainty_set.basis, [[1.0], [0.0]])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"radius": -1}, "radius.*non-negative"),
        ({"metric_sqrt": [[1, 2]]}, "metric_sqrt.*1D"),
        ({"metric_sqrt": [1, -1]}, "finite non-negative"),
        ({"metric_sqrt": [1, np.inf]}, "finite non-negative"),
        ({"basis": [1, 0]}, "basis.*2D"),
        ({"basis": np.ones((3, 1))}, "same number of assets"),
        ({"basis": [[1], [np.inf]]}, "basis.*finite"),
    ],
)
def test_compact_covariance_uncertainty_set_rejects_invalid_parameters(kwargs, match):
    """Compact covariance parameters must be finite and dimensionally aligned."""
    params = {
        "radius": 1,
        "metric_sqrt": [1, 2],
        "basis": [[1], [0]],
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=match):
        CompactCovarianceUncertaintySet(**params)
