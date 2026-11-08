"""Tests for KendallDistance."""

from __future__ import annotations

import numpy as np

from skfolio.distance import KendallDistance


class TestKendallDistance:
    def test_kendall_distance(self, X):
        distance = KendallDistance()
        distance.fit(X)
        assert distance.codependence_.shape == (20, 20)
        assert distance.distance_.shape == (20, 20)
        assert np.all(distance.distance_ >= 0) and np.all(distance.distance_ <= 1)

    #  KendallDistance can be instantiated with default parameters
    def test_instantiation_with_default_parameters(self):
        kd = KendallDistance()
        assert kd.absolute is False
        assert kd.power == 1
