.. _portfolio:

.. currentmodule:: skfolio.portfolio

.. role:: python(code)
   :language: python

=========
Portfolio
=========

`Portfolio` classes implement a large set of attributes and methods intended for
portfolio analysis. They are returned by the `predict` method of
:ref:`portfolio optimizations <optimization>`.

They are also data-containers (calling
:python:`np.asarray(portfolio)` returns the portfolio returns) making them compatible
with `sklearn.model_selection` tools.

They use `slots` for improved performance.

Base Portfolio
**************
:class:`BasePortfolio` directly takes a portfolio returns array as input and implements
a large set of attributes and methods.

**Example:**

.. code-block:: python

    import datetime as dt
    from skfolio import BasePortfolio

    portfolio = BasePortfolio(
        returns=[0.002, -0.001, 0.0015],
        observations=[dt.date(2022, 1, 1), dt.date(2022, 1, 2), dt.date(2022, 1, 3)],
    )


Attributes and Methods
----------------------
More than 40 attributes and methods are available, including all the
:ref:`measures <measures_ref>` (Mean, Variance, Sharpe Ratio, CVaR, CDaR, Drawdowns,
etc.). The attributes are computed only when requested, then cached in `slots` for
enhanced performance.

Measures are computed on the per-observation return series, in the periodicity of
the returns. The annualized variants (e.g. `annualized_sharpe_ratio`,
`annualized_mean`) scale them for reporting using the `annualization_factor`
parameter (default 252). Optimization inputs are never annualized, only reported
measures are (see :ref:`Periodicity Convention <periodicity_convention>`).

**Example:**

.. code-block:: python

    from skfolio import RatioMeasure

    # attributes
    portfolio.mean
    portfolio.variance
    portfolio.sharpe_ratio
    portfolio.sortino_ratio
    portfolio.cdar
    portfolio.max_drawdown
    portfolio.cumulative_returns
    portfolio.drawdowns
    portfolio.returns_df
    portfolio.cumulative_returns_df

    # methods
    portfolio.summary()
    portfolio.dominates(other_portfolio)
    portfolio.rolling_measure(measure=RatioMeasure.SHARPE_RATIO)

    # plots
    portfolio.plot_cumulative_returns()
    portfolio.plot_rolling_measure(measure=RatioMeasure.SHARPE_RATIO)


It is also an array container:

.. code-block:: python

    np.asarray(portfolio)
    >>> array([ 0.002 , -0.001 ,  0.0015])


Finally, portfolios can be compared together using domination:

.. code-block:: python

    portfolio == other_portfolio
    portfolio >= other_portfolio
    portfolio > other_portfolio


The measures used in the domination are controlled using `fitness_measures`. The default
is to use the list `[PerfMeasure.MEAN, RiskMeasure.VARIANCE]`.


Portfolio
*********
:class:`Portfolio` inherits from :class:`BasePortfolio`. Under the default
constant-weight convention, portfolio returns are the dot product of the asset weights
and asset returns minus costs:

.. math::

   r_p = R \cdot w^{T} - c^{T} \cdot |w-w_{prev}| - f^{T} \cdot w

with :math:`r_p` the vector of portfolio returns, :math:`R` the matrix of asset
returns, :math:`w` the vector of asset weights, :math:`c` the vector of asset
transaction costs, :math:`f` the vector of asset management fees and
:math:`w_{prev}` the previous asset weights.

By default, each observation is evaluated at the target weights, consistent with
the optimizer's linear portfolio return definition. This convention evaluates
**allocation skill**, the expected return and risk of the selected target allocation,
independently of subsequent weight drift.

Because the same target weights are applied to every observation, reordering the
observations does not change the resulting distribution of portfolio returns.

Economically, evaluating every observation at the target weights is equivalent to
restoring those weights after each observation. The transaction costs of these
implicit within-window trades are not charged. The transaction-cost term in the
formula above instead applies to the trade from `previous_weights` to the target
weights when the `Portfolio` is created.

With `weight_drift=True`, the portfolio starts at the target weights and holds the
resulting positions throughout the observation window of `X`. Position values change
with asset returns, so portfolio weights evolve with the relative performance of the
assets.

In a long-only portfolio, an asset's weight increases when its return exceeds the
portfolio return and decreases when its return falls below it. More generally, the
weights applied to later observations depend on earlier asset returns, so the
resulting portfolio return series depends on the order of the observations.

With `weight_drift=True` and `compounded=True`, returns computed from drifted weights
are compounded into a wealth index for evaluating **realized capital growth** and
other path-dependent measures.

Both weight conventions use the same transaction-cost and management-fee formulas.
In a sequential evaluation, transaction costs are computed relative to the previous
target weights with `weight_drift=False`, or the previous period's ending weights,
including drift, with `weight_drift=True`. Management fees use the target weights
under both settings.

`weight_drift` therefore changes the observation-level portfolio return series,
while `compounded` changes how that series is accumulated. See
:ref:`backtesting_and_evaluation` for a detailed discussion of the evaluation
objectives and the choice between `weight_drift=False` and `weight_drift=True`.

**Example:**

.. code-block:: python

    from skfolio import Portfolio

    X = [
        [0.003, -0.001],
        [-0.001, 0.002],
        [0.0015, 0.004],
    ]

    weights = [0.6, 0.4]

    portfolio = Portfolio(X=X, weights=weights)

    print(portfolio.returns)
    >>> array([0.0014, 0.0002, 0.0025])

    drifted_portfolio = Portfolio(X=X, weights=weights, weight_drift=True)
    drifted_portfolio.weights_per_observation
    drifted_portfolio.ending_weights


`X` can be any data-container including numpy array and pandas DataFrame:

.. code-block:: python

    import datetime as dt
    import pandas as pd

    X = pd.DataFrame(
        data=[[0.003, -0.001], [-0.001, 0.002], [0.0015, 0.004]],
        columns=["Asset A", "Asset B"],
        index=[dt.date(2022, 1, 1), dt.date(2022, 1, 2), dt.date(2022, 1, 3)],
    )

    print(X)
    >>>
                Asset A  Asset B
    2022-01-01   0.0030   -0.001
    2022-01-02  -0.0010    0.002
    2022-01-03   0.0015    0.004

    weights = [0.6, 0.4]

    portfolio = Portfolio(X=X, weights=weights, name="my_portfolio")

    print(portfolio.returns)
    >>> array([0.0014, 0.0002, 0.0025])


Attributes and Methods
----------------------
:class:`Portfolio` inherits all the attributes and methods from :class:`BasePortfolio`.
It also provides methods for analyzing portfolio weights:

.. code-block:: python

    from skfolio import RatioMeasure

    portfolio.contribution(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO)
    >>> array([-3.04203502,  3.04203503])

    portfolio.composition
    >>>
                  my_portfolio
    asset
    Asset A           0.6
    Asset B           0.4

    portfolio.get_weight("Asset A")
    >>> 0.6

    # Weight paths and trading diagnostics
    portfolio.weights_per_observation
    portfolio.ending_weights
    portfolio.turnover

    # Plots
    portfolio.plot_contribution()
    portfolio.plot_composition()



Multi Period Portfolio
**********************
:class:`MultiPeriodPortfolio` inherits from :class:`BasePortfolio` and is composed of a
list of :class:`Portfolio`. Its return series concatenates the return series of those
portfolios in list order. Its performance and risk measures are computed from that
concatenated series.

A `MultiPeriodPortfolio` is returned by
:func:`~skfolio.model_selection.cross_val_predict`.
Its `turnover` series holds the turnover of each `Portfolio`, and
`ending_weights_dict` maps each `Portfolio` to its weights at the end of its observation
window. With `weight_drift=False`, these are the target weights. With
`weight_drift=True`, they are the held weights after applying the final observation's
asset returns.

For example, calling `cross_val_predict` with
:class:`~skfolio.model_selection.WalkForward` will return a `MultiPeriodPortfolio`
composed of multiple test `Portfolio`, each corresponding to a train/test fold.

.. code-block:: python

    from skfolio import MultiPeriodPortfolio

    portfolio = MultiPeriodPortfolio(portfolios=[ptf1, ptf2, ptf3])
