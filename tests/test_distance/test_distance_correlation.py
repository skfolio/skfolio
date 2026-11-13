"""Tests for DistanceCorrelation."""

from __future__ import annotations

import numpy as np

from skfolio.distance import DistanceCorrelation


class TestDistanceCorrelation:
    def test_distance_correlation(self, X):
        distance = DistanceCorrelation()
        distance.fit(X.iloc[:500])
        assert distance.codependence_.shape == (20, 20)
        assert distance.distance_.shape == (20, 20)
        assert np.all(distance.distance_ >= 0) and np.all(distance.distance_ <= 1)

    #  Fit the estimator with valid input data.
    def test_fit_valid_input_data(self):
        X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dc = DistanceCorrelation()
        dc.fit(X)
        assert np.array_equal(dc.codependence_, np.ones((3, 3)))
        assert np.array_equal(dc.distance_, np.zeros((3, 3)))
