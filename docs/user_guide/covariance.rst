.. _covariance_estimator:

.. currentmodule:: skfolio.moments

********************
Covariance Estimator
********************

A :ref:`covariance estimator <covariance_ref>` estimates the covariance matrix of the
assets.

It follows the same API as scikit-learn's `estimator`: the `fit` method takes `X` as the
assets returns and stores the covariance in its `covariance_` attribute.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc.)


Available estimators are:
    * :class:`EmpiricalCovariance`
    * :class:`EWCovariance`
    * :class:`RegimeAdjustedEWCovariance`
    * :class:`GerberCovariance`
    * :class:`DenoiseCovariance`
    * :class:`DetoneCovariance`
    * :class:`LedoitWolf`
    * :class:`OAS`
    * :class:`ShrunkCovariance`
    * :class:`GraphicalLassoCV`
    * :class:`ImpliedCovariance`

For online learning and streaming workflows, :class:`EWCovariance` and
:class:`RegimeAdjustedEWCovariance` support incremental updates with
`partial_fit`. They also support NaN-aware updates with `active_mask`, which
helps distinguish assets that belong to the universe but have missing returns,
for example on holidays, from assets outside the universe, such as during
pre-listing or post-delisting periods.
See :ref:`online_learning` for the full online workflow, including covariance
forecast evaluation and online hyper-parameter tuning.
See :ref:`missing_data` for the full convention on NaNs, universe membership, estimator
warmup and investability.

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

