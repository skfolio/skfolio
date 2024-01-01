.. _mu_estimator:

.. currentmodule:: skfolio.moments

*************************
Expected Return Estimator
*************************

An :ref:`expected return estimator <mu_ref>` estimates the expected returns (`mu`) of
the assets.

It follows the same API as scikit-learn's `estimator`: the `fit` method takes `X` as
the assets returns and stores the expected returns in its  `mu_` attribute.
`X` can be any array-like structure (numpy array, pandas DataFrame, etc.)


Available estimators are:
    * :class:`EmpiricalMu`
    * :class:`EWMu`
    * :class:`EquilibriumMu`
    * :class:`ShrunkMu`

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.moments import EmpiricalMu
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = EmpiricalMu()
    model.fit(X)
    print(model.mu_)