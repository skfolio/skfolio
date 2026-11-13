"""Tests for SpearmanDistance."""

from __future__ import annotations

import numpy as np

from skfolio.distance import SpearmanDistance


class TestSpearmanDistance:
    def test_spearman_distance(self, X):
        distance = SpearmanDistance()
        distance.fit(X)
        assert distance.codependence_.shape == (20, 20)
        assert distance.distance_.shape == (20, 20)
        assert np.all(distance.distance_ >= 0) and np.all(distance.distance_ <= 1)

    #  SpearmanDistance can be initialized with default parameters.
    def test_initialized_with_default_parameters(self):
        distance = SpearmanDistance()
        assert distance.absolute is False
        assert distance.power == 1
