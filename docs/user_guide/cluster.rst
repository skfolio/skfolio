.. _cluster:
.. _hierarchical_clustering:

.. currentmodule:: skfolio.cluster

*********************
Clustering Estimators
*********************

The `skfolio.cluster` module complements `sklearn.cluster` with additional clustering
estimators including the :class:`HierarchicalClustering` that forms hierarchical
clusters from a distance matrix. It is used in the following portfolio optimizations:

    * :class:`~skfolio.optimization.HierarchicalRiskParity`
    * :class:`~skfolio.optimization.HierarchicalEqualRiskContribution`
    * :class:`~skfolio.optimization.NestedClustersOptimization`


**Example:**

.. code-block:: python

    from skfolio.cluster import HierarchicalClustering
    from skfolio.datasets import load_sp500_dataset
    from skfolio.distance import PearsonDistance
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    distance_estimator = PearsonDistance()
    distance_estimator.fit(X)
    distance = distance_estimator.distance_

    model = HierarchicalClustering()
    model.fit(distance)
    print(model.linkage_matrix_)

