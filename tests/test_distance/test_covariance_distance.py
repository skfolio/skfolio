"""Tests for CovarianceDistance."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context
from sklearn.exceptions import UnsetMetadataPassedError

from skfolio.distance import CovarianceDistance
from skfolio.moments import GerberCovariance, ImpliedCovariance


class TestCovarianceDistance:
    def test_covariance_distance(self, X):
        distance = CovarianceDistance()
        distance.fit(X)
        assert distance.codependence_.shape == (20, 20)
        assert distance.distance_.shape == (20, 20)
        assert np.all(distance.distance_ >= 0) and np.all(distance.distance_ <= 1)

    #  fitting the estimator with default parameters
    def test_fit_with_default_parameters(self, X):
        distance = CovarianceDistance()
        distance.fit(X)
        assert isinstance(distance.covariance_estimator_, GerberCovariance)
        assert isinstance(distance.codependence_, np.ndarray)
        assert isinstance(distance.distance_, np.ndarray)
        assert distance.absolute is False
        assert distance.power == 1

    def test_metadata_routing_errors(self, X, implied_vol):
        distance = CovarianceDistance(covariance_estimator=ImpliedCovariance())

        with pytest.raises(ValueError, match="`implied_vol` cannot be None"):
            distance.fit(X)

        with pytest.raises(
            UnsetMetadataPassedError,
            match="are passed but are not explicitly set as requested",
        ):
            distance.fit(X, implied_vol=implied_vol)

        with pytest.raises(ValueError, match="`implied_vol` cannot be None"):
            with config_context(enable_metadata_routing=True):
                distance.fit(X)

        with pytest.raises(
            UnsetMetadataPassedError,
            match="are passed but are not explicitly set as requested",
        ):
            with config_context(enable_metadata_routing=True):
                distance.fit(X, implied_vol=implied_vol)

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            distance = CovarianceDistance(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )

            with pytest.raises(ValueError, match="`implied_vol` cannot be None"):
                distance.fit(X)

            distance.fit(X, implied_vol=implied_vol)

        assert distance.covariance_estimator_.r2_scores_.shape == (20,)
        assert distance.distance_.shape == (20, 20)
