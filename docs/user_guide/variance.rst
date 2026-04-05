.. _variance_estimator:

.. currentmodule:: skfolio.moments

******************
Variance Estimator
******************

A :ref:`variance estimator <variance_ref>` estimates the variance vector of the
assets.

It follows the same API as scikit-learn's `estimator`: the `fit` method takes `X` as
the assets returns and stores the variances in its `variance_` attribute.

Variance estimators are useful when only marginal volatility is needed, for example
when modelling idiosyncratic risk or working with orthogonalized return series.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc.)


Available estimators are:
    * :class:`EmpiricalVariance`
    * :class:`EWVariance`
    * :class:`RegimeAdjustedEWVariance`

For online learning and streaming workflows, :class:`EWVariance` and
:class:`RegimeAdjustedEWVariance` support incremental updates with
`partial_fit`. They also support NaN-aware updates with `active_mask`, which
helps distinguish active assets with missing returns, for example on holidays,
from structurally inactive assets such as pre-listing or post-delisting
periods.

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.moments import EmpiricalVariance
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    model = EmpiricalVariance()
    model.fit(X)
    print(model.variance_)
