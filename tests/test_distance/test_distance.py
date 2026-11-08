"""Contracts shared by distance estimators."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.distance import (
    CovarianceDistance,
    KendallDistance,
    PearsonDistance,
    SpearmanDistance,
)
from skfolio.distance._utils import _corr_to_distance


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("power_type", [float, np.float64])
@pytest.mark.parametrize(
    "absolute,power,expected_codependence,expected_distance",
    [
        (False, 1, -0.5, np.sqrt(0.75)),
        (True, 1, 0.5, np.sqrt(0.5)),
        (False, 2, 0.25, np.sqrt(0.75)),
        (False, 3, -0.125, 0.75),
        (True, 0.5, np.sqrt(0.5), np.sqrt(1 - np.sqrt(0.5))),
    ],
)
def test_distance_conversion_preserves_input(
    dtype, power_type, absolute, power, expected_codependence, expected_distance
):
    power = power_type(power)
    corr = np.array(
        [[1, -0.5, np.nan], [-0.5, 1, np.nan], [np.nan, np.nan, np.nan]], dtype=dtype
    )
    original = corr.copy()
    corr.flags.writeable = False
    codependence, distance = _corr_to_distance(corr, absolute=absolute, power=power)

    np.testing.assert_allclose(
        codependence[:2, :2],
        [[1, expected_codependence], [expected_codependence, 1]],
        rtol=2e-7,
    )
    np.testing.assert_allclose(
        distance[:2, :2], [[0, expected_distance], [expected_distance, 0]], rtol=2e-7
    )
    assert codependence.dtype == corr.dtype
    # Match the existing NumPy promotion when the distance is scaled.
    assert distance.dtype == (np.float64(1) * codependence).dtype
    for result in [codependence, distance]:
        assert np.isnan(result[2]).all()
        assert np.isnan(result[:, 2]).all()
        assert not np.shares_memory(result, corr)
    assert not np.shares_memory(codependence, distance)
    codependence[0, 0] = 0
    np.testing.assert_array_equal(corr, original)


@pytest.mark.parametrize("absolute,power", [(False, 1), (True, 1), (False, 2)])
def test_distance_conversion_clips_roundoff(absolute, power):
    corr = np.array([[1 + 1e-14, -1 - 1e-14], [-1 - 1e-14, 1 + 1e-14]])
    _, distance = _corr_to_distance(corr, absolute=absolute, power=power)
    expected = [[0, 1], [1, 0]] if not absolute and power == 1 else np.zeros((2, 2))
    np.testing.assert_array_equal(distance, expected)


@pytest.mark.parametrize(
    "estimator",
    [PearsonDistance, KendallDistance, SpearmanDistance, CovarianceDistance],
)
@pytest.mark.parametrize(
    "absolute,power",
    [
        (False, 0),
        (True, 0),
        (False, -1),
        (True, -1),
        (False, np.nan),
        (True, np.inf),
        (False, 0.5),
    ],
)
def test_invalid_power(estimator, absolute, power):
    X = np.random.default_rng(0).normal(size=(20, 3))
    with pytest.raises(ValueError, match="power"):
        estimator(absolute=absolute, power=power).fit(X)
