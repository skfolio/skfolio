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

`mu_` is expressed in the periodicity of `X` (daily returns give daily expected
returns) and is consumed as is by the optimizers, without annualization (see
:ref:`Periodicity Convention <periodicity_convention>`).


Available estimators are:
    * :class:`EmpiricalMu`
    * :class:`EWMu`
    * :class:`EquilibriumMu`
    * :class:`ShrunkMu`

For online learning and streaming workflows, :class:`EWMu` supports
incremental updates with `partial_fit`. It also supports NaN-aware updates with
`active_mask`, which helps distinguish assets that belong to the universe but have
missing returns (e.g. holidays), from assets outside the universe (e.g. pre-listing
or post-delisting periods).
See :ref:`online_learning` for the full online workflow, including online
portfolio optimization evaluation with incremental moments.
See :ref:`Missing Data and Changing Universes <missing_data>` for the full convention
on NaNs, universe membership, estimator warmup and investability.

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
