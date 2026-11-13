"""Tests for PearsonDistance."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.distance import PearsonDistance


class TestPearsonDistance:
    def test_pearson_distance(self, X):
        distance = PearsonDistance()
        distance.fit(X)
        assert distance.codependence_.shape == (20, 20)
        assert distance.distance_.shape == (20, 20)
        np.testing.assert_almost_equal(distance.codependence_, np.corrcoef(X.T))
        np.testing.assert_almost_equal(
            distance.distance_, np.sqrt(0.5 * (1 - np.corrcoef(X.T)))
        )
        assert np.all(distance.distance_ >= 0) and np.all(distance.distance_ <= 1)

        distance = PearsonDistance(absolute=True)
        distance.fit(X)
        assert distance.codependence_.shape == (20, 20)
        assert distance.distance_.shape == (20, 20)
        np.testing.assert_almost_equal(distance.codependence_, np.abs(np.corrcoef(X.T)))
        np.testing.assert_almost_equal(
            distance.distance_, np.sqrt(1 - np.abs(np.corrcoef(X.T)))
        )
        assert np.all(distance.distance_ >= 0) and np.all(distance.distance_ <= 1)

        distance = PearsonDistance(power=2)
        distance.fit(X)
        assert distance.codependence_.shape == (20, 20)
        assert distance.distance_.shape == (20, 20)
        np.testing.assert_almost_equal(distance.codependence_, np.corrcoef(X.T) ** 2)
        np.testing.assert_almost_equal(
            distance.distance_, np.sqrt(1 - np.corrcoef(X.T) ** 2)
        )
        assert np.all(distance.distance_ >= 0) and np.all(distance.distance_ <= 1)

    #  PearsonDistance can be instantiated with default parameters
    def test_instantiation_with_default_parameters(self):
        pd = PearsonDistance()
        assert pd.absolute is False
        assert pd.power == 1

    #  PearsonDistance raises an error when fitting an empty array
    def test_fitting_empty_array(self):
        pd = PearsonDistance()
        with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
            pd.fit([])

    #  PearsonDistance raises an error when fitting an array with NaN values
    def test_fitting_array_with_nan_values(self):
        pd = PearsonDistance()
        X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        with pytest.raises(ValueError, match="Input X contains NaN"):
            pd.fit(X)
