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
:class:`Portfolio` inherits from :class:`BasePortfolio`. The portfolio returns are the
dot product of the assets weights with the assets returns minus costs:

    .. math::   r_p = R \cdot w^{T} - c^{T} \cdot | w - w_{prev} | - f^{T} \cdot w

with :math:`r_p` the vector of portfolio returns , :math:`R` the matrix of assets
returns, :math:`w` the vector of assets weights, :math:`c` the vector of assets
transaction costs, :math:`f` the vector of assets management fees and :math:`w_{prev}`
the assets previous weights.

.. warning::

    The :class:`Portfolio` formulation is **homogeneous** to the convex optimization
    problems for coherent analysis. It's important to note that this portfolio
    formulation is **not perfectly replicable** due to weight drift when asset prices
    move. The only case where it would be perfectly replicable is with periodic
    rebalancing with zero costs. In practice, portfolios are
    rebalanced frequently enough, so this weight drift becomes negligible in regards to
    model analysis and selection. Before trading, a full replicability analysis should
    be performed, which is another topic left to the investor.

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
In addition, it also implements weights related methods:

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

    # Plots
    portfolio.plot_contribution()
    portfolio.plot_composition()



Multi Period Portfolio
**********************
:class:`MultiPeriodPortfolio` inherits from :class:`BasePortfolio` and is composed of a
list of :class:`Portfolio`. The multi-period portfolio returns are the sum of all its
underlying :class:`Portfolio` returns.
A `MultiPeriodPortfolio` is returned by :func:`~skfolio.model_selection.cross_val_predict`.

For example, calling `cross_val_predict` with :class:`~skfolio.model_selection.WalkForward`
will return a `MultiPeriodPortfolio` composed of multiple test `Portfolio`, each
corresponding to a train/test fold.

.. code-block:: python

    from skfolio import MultiPeriodPortfolio

    portfolio = MultiPeriodPortfolio(portfolios=[ptf1, ptf2, ptf3])

