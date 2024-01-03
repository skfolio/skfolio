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