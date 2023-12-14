.. _covariance_estimator:

.. currentmodule:: skfolio.moments

********************
Covariance Estimator
********************

A :ref:`covariance estimator <covariance_ref>` estimates the covariance matrix of the
assets.

It follows the same API as scikit-learn's `estimator`: the `fit` method takes `X` as the
assets returns and stores the covariance in its `covariance_` attribute.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc...)


Available estimators are:
    * :class:`EmpiricalCovariance`
    * :class:`EWCovariance`
    * :class:`GerberCovariance`
    * :class:`DenoiseCovariance`
    * :class:`DenoteCovariance`
    * :class:`LedoitWolf`
    * :class:`OAS`
    * :class:`ShrunkCovariance`
    * :class:`GraphicalLassoCV`

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.moments import EmpiricalCovariance
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = EmpiricalCovariance()
    model.fit(X)
    print(model.covariance_)

