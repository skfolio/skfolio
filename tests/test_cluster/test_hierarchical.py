"""Test Hierarchical module."""

import numpy as np
import pandas as pd
import pytest

from skfolio.cluster import HierarchicalClustering
from skfolio.datasets import load_sp500_dataset
from skfolio.distance import PearsonDistance
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices)
    distance_estimator = PearsonDistance()
    distance_estimator.fit(X)
    X = distance_estimator.distance_
    X = pd.DataFrame(X, columns=prices.columns)
    return X


@pytest.fixture(
    scope="module",
    params=[None, *list(range(2, 10))],
)
def max_clusters(request):
    return request.param


def test_plot_dendrogram(X):
    model = HierarchicalClustering()
    model.fit(X)
    assert model.plot_dendrogram(heatmap=True)
    assert model.plot_dendrogram(heatmap=False)

    model = HierarchicalClustering()
    model.fit(np.asarray(X))
    assert model.plot_dendrogram(heatmap=True)
    assert model.plot_dendrogram(heatmap=False)


def test_default_hierarchical_clustering(X):
    model = HierarchicalClustering()
    model.fit(X)
    np.testing.assert_almost_equal(
        model.linkage_matrix_,
        np.array(
            [
                [4.0, 19.0, 0.32061356, 2.0],
                [2.0, 8.0, 0.35777062, 2.0],
                [11.0, 14.0, 0.47003656, 2.0],
                [9.0, 13.0, 0.4721542, 2.0],
                [7.0, 22.0, 0.48811653, 3.0],
                [6.0, 18.0, 0.49292738, 2.0],
                [10.0, 24.0, 0.49983388, 4.0],
                [5.0, 21.0, 0.51391337, 3.0],
                [15.0, 23.0, 0.52944542, 3.0],
                [0.0, 12.0, 0.54435806, 2.0],
                [1.0, 29.0, 0.58293335, 3.0],
                [3.0, 25.0, 0.60127824, 3.0],
                [16.0, 20.0, 0.63669722, 3.0],
                [17.0, 26.0, 0.64352264, 5.0],
                [30.0, 31.0, 0.67599117, 6.0],
                [28.0, 33.0, 0.67799022, 8.0],
                [27.0, 34.0, 0.71550808, 9.0],
                [32.0, 36.0, 0.80955466, 12.0],
                [35.0, 37.0, 0.87957535, 20.0],
            ]
        ),
    )


def test_hierarchical_clustering(X, max_clusters, linkage_method):
    model = HierarchicalClustering(
        max_clusters=max_clusters, linkage_method=linkage_method
    )
    model.fit(X)
    assert model.n_clusters_ == max(model.labels_) + 1
    if max_clusters is not None:
        assert model.n_clusters_ <= max_clusters

    assert model.linkage_matrix_.shape == (19, 4)
