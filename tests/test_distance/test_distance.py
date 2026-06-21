"""Test Distance module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.exceptions import UnsetMetadataPassedError

from skfolio.distance import (
    CovarianceDistance,
    DistanceCorrelation,
    GraphDistance,
    GraphMode,
    KendallDistance,
    MutualInformation,
    NBinsMethod,
    PearsonDistance,
    SpearmanDistance,
)
from skfolio.moments import GerberCovariance, ImpliedCovariance


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
        with pytest.raises(ValueError):
            pd.fit([])

    #  PearsonDistance raises an error when fitting an array with NaN values
    def test_fitting_array_with_nan_values(self):
        pd = PearsonDistance()
        X = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            pd.fit(X)


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

        with pytest.raises(ValueError):
            distance.fit(X)

        with pytest.raises(UnsetMetadataPassedError):
            distance.fit(X, implied_vol=implied_vol)

        with pytest.raises(ValueError):
            with config_context(enable_metadata_routing=True):
                distance.fit(X)

        with pytest.raises(UnsetMetadataPassedError):
            with config_context(enable_metadata_routing=True):
                distance.fit(X, implied_vol=implied_vol)

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            distance = CovarianceDistance(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )

            with pytest.raises(ValueError):
                distance.fit(X)

            distance.fit(X, implied_vol=implied_vol)

        # noinspection PyUnresolvedReferences
        assert distance.covariance_estimator_.r2_scores_.shape == (20,)
        assert distance.distance_.shape == (20, 20)


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


# Generated by CodiumAI


class TestMutualInformation:
    def test_mutual_information(self, X):
        distance = MutualInformation()
        distance.fit(X)
        assert distance.codependence_.shape == (20, 20)
        assert distance.distance_.shape == (20, 20)
        assert np.all(distance.distance_ >= 0) and np.all(distance.distance_ <= 1)

    #  fitting the estimator with default parameters
    def test_default_parameters(self, X):
        distance = MutualInformation()
        assert distance.n_bins_method == NBinsMethod.FREEDMAN
        assert distance.n_bins is None
        assert distance.normalize is True


class TestGraphDistance:
    def test_graph_distance(self, X):
        adjacency_matrix = np.array(
            [
                [2.0, 1.0, 0.0],
                [1.0, 2.0, 0.5],
                [0.0, 0.5, 2.0],
            ]
        )
        distance = GraphDistance()
        distance.fit(X.iloc[:, :3], adjacency_matrix=adjacency_matrix)

        expected_codependence = adjacency_matrix / 2.0
        np.testing.assert_almost_equal(distance.codependence_, expected_codependence)
        np.testing.assert_almost_equal(
            distance.distance_, np.sqrt(1 - expected_codependence)
        )
        assert distance.codependence_.shape == (3, 3)
        assert distance.distance_.shape == (3, 3)
        assert np.all(distance.codependence_ >= 0)
        assert np.all(distance.codependence_ <= 1)
        assert np.all(distance.distance_ >= 0)
        assert np.all(distance.distance_ <= 1)

    def test_default_parameters(self):
        distance = GraphDistance()
        assert distance.mode == GraphMode.ASSET

    def test_adjacency_matrix_required(self, X):
        distance = GraphDistance()
        with pytest.raises(ValueError, match="adjacency_matrix"):
            distance.fit(X.iloc[:, :3])

    def test_adjacency_matrix_shape_must_match_assets(self, X):
        distance = GraphDistance()
        with pytest.raises(ValueError, match="shape"):
            distance.fit(X.iloc[:, :3], adjacency_matrix=np.ones((2, 2)))

        with pytest.raises(ValueError, match="shape"):
            distance.fit(X.iloc[:, :3], adjacency_matrix=np.ones((3, 2)))

    def test_adjacency_matrix_labels_must_match_assets_order(self, X):
        assets = list(X.columns[:3])
        adjacency_matrix = pd.DataFrame(
            np.eye(3), index=assets[::-1], columns=assets[::-1]
        )
        distance = GraphDistance()
        with pytest.raises(ValueError, match="assets order"):
            distance.fit(X.iloc[:, :3], adjacency_matrix=adjacency_matrix)

    def test_constant_adjacency_matrix(self, X):
        adjacency_matrix = np.ones((3, 3))
        distance = GraphDistance()
        distance.fit(X.iloc[:, :3], adjacency_matrix=adjacency_matrix)

        np.testing.assert_array_equal(distance.codependence_, np.ones((3, 3)))
        np.testing.assert_array_equal(distance.distance_, np.zeros((3, 3)))

    def test_zero_diagonal_adjacency_matrix(self, X):
        adjacency_matrix = np.array(
            [
                [0.0, 2.0, 1.0],
                [2.0, 0.0, 0.5],
                [1.0, 0.5, 0.0],
            ]
        )
        distance = GraphDistance()
        distance.fit(X.iloc[:, :3], adjacency_matrix=adjacency_matrix)

        np.testing.assert_array_equal(np.diag(distance.codependence_), np.ones(3))
        np.testing.assert_array_equal(np.diag(distance.distance_), np.zeros(3))
