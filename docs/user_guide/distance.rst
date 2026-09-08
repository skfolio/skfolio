.. _distance:

.. currentmodule:: skfolio.distance

******************
Distance Estimator
******************

A :ref:`distance estimator <distance_ref>` estimates the codependence and distance
matrix of the assets.

It follows the same API as scikit-learn's `estimator`: the `fit` method takes `X` as the
assets returns and stores the codependence and distance matrix in its `codependence_`
and `distance_` attributes.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc.)


Available estimators are:
    * :class:`PearsonDistance`
    * :class:`KendallDistance`
    * :class:`SpearmanDistance`
    * :class:`CovarianceDistance`
    * :class:`DistanceCorrelation`
    * :class:`MutualInformation`
    * :class:`GraphDistance`

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.distance import PearsonDistance
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = PearsonDistance()
    model.fit(X)
    print(model.codependence_)
    print(model.distance_)

**Graph adjacency example:**

:class:`GraphDistance` supports asset-to-asset adjacency matrices only. Dependency
graph mode is not implemented.

.. code-block:: python

    import numpy as np

    from skfolio.distance import GraphDistance

    adjacency_matrix = np.array(
        [
            [1.0, 0.8, 0.2],
            [0.8, 1.0, 0.5],
            [0.2, 0.5, 1.0],
        ]
    )

    model = GraphDistance()
    model.fit(X.iloc[:, :3], adjacency_matrix=adjacency_matrix)
    print(model.codependence_)
    print(model.distance_)
