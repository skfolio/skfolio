.. _data_preparation:

.. currentmodule:: skfolio.datasets

****************
Data Preparation
****************

Most `fit` methods of `skfolio` estimators take the assets returns as input `X`.
Therefore, the choice of methodology to convert prices to returns is left to the user.
For datasets with missing returns or a changing asset universe, see
:ref:`Missing Data and Changing Universes <missing_data>`.

There are two different notions of return:

Linear return
=============
Linear return (or simple return) is defined as:

.. math:: R^{Lin}_{t} = \frac{S_{t}}{S_{t-1}} - 1

**Linear returns aggregate across securities**, meaning that the linear return
of a portfolio is the sum of the weighted linear returns of its components:

.. math:: R^{Lin}_{t} = \sum_{i=1}^{N} w_{i} \times  R^{Lin}_{i,t}


This property is needed to properly compute portfolio return and risk [1]_.
However, linear returns cannot be aggregated across time.

Logarithmic return
==================
Logarithmic return (or continuously compounded return) is defined as:

.. math:: R^{Log}_{t} = ln\Biggl(\frac{S_{t}}{S_{t-1}}\Biggr)

**Logarithmic returns aggregate across time**, meaning that the logarithmic return over
k periods is the sum of all single-period logarithmic returns:

.. math:: R^{Log}_{t..k} = ln\Biggl(\frac{S_{t+k}}{S_{t}}\Biggr) = \sum_{j=1}^{k} ln\Biggl(\frac{S_{t+j}}{S_{t+j-1}}\Biggr)= \sum_{j=1}^{k-1} R^{Log}_{t+j}


Given this property, it is easy to scale logarithmic return from one time period to another.
However, logarithmic return cannot be aggregated across securities:

.. math:: R^{Log}_{t} = ln\Biggl(\frac{S_{t}}{S_{t-1}}\Biggr) = ln\Biggl(1+\sum_{i=1}^{N} w_{i} \times  R^{Lin}_{i,t}\Biggr)

Pitfall in Portfolio Optimization
=================================
Given the similarities of linear and logarithmic returns in the short run, they are
sometimes used interchangeably.
It is not uncommon to witness the following steps [2]_, [3]_, [4]_:

#. Take the daily prices :math:`S_{t}, S_{t+1}, ...,` for all the n securities
#. Transform the daily prices to daily logarithmic returns
#. Estimate the expected returns vector :math:`\mu` and covariance matrix :math:`\Sigma` from the daily logarithmic returns
#. Determine the investment horizon, for example k = 252 days
#. Project the expected returns and covariance to the horizon using the square-root rule: :math:`\mu_{k} ≡ k \times \mu` and :math:`\Sigma_{k} ≡ k \times \Sigma`
#. Compute the mean-variance efficient frontier :math:`\max_{w} \Biggl\{ w^T \mu - \lambda \times w^T \Sigma w \Biggr\}`

The above approach is incorrect. First, the square-root rule in (5) only applies under
the assumption that the logarithmic returns are invariants (they behave identically and
independently across time). It is approximately true for stocks; long-term dependence
requires additional care in optimization [5]_. It is not true for bonds nor most
derivatives like options.
Secondly, even for stocks, the optimization (6) is ill-posed: :math:`w^T \mu`
is not the expected return of the portfolio over the horizon and :math:`w^T \Sigma w`
is not its variance.
These would lead to suboptimal allocations and the efficient frontier would not depend on
the investment horizon.

The correct approach
====================
The correct general approach is the following:

#. Find the market invariants (logarithmic return for stocks, change in yield to maturity for bonds, etc.)
#. Estimate the joint distribution of the market invariant over the time period of estimation
#. Project the distribution of invariants to the time period of investment
#. Map the distribution of invariants into the distribution of security prices at the investment horizon through a pricing function
#. Compute the distribution of linear returns from the distribution of prices

Example for stocks
==================

#. Take the prices :math:`S_{t}, S_{t+1}, ...,` (for example daily) for all the n securities
#. Transform the daily prices to daily logarithmic returns. Note that linear return is also a market invariant for stock, however logarithmic return is going to simplify step 3) and 4).
#. Estimate the joint distribution of market invariants by fitting parametrically the daily logarithmic returns to a multivariate normal distribution: estimate the joint distribution parameters :math:`\mu^{Log}_{daily}` and :math:`\Sigma^{Log}_{daily}`
#. Project the distribution of invariants to the time period of investment (for example one year i.e. 252 business days). Because logarithmic returns are additive across time, we have [6]_, [7]_:

        * .. math:: \mu^{Log}_{yearly} = 252 \times \mu^{Log}_{daily}
        * .. math:: \Sigma^{Log}_{yearly} = 252 \times \Sigma^{Log}_{daily}
#. Compute the distribution of linear returns at the investment horizon. Using the characteristic function of the normal distribution, and the pricing function :math:`S_{yearly} = S_{0} e^{R^{Log}_{yearly}}`, we get:

        * .. math:: \mathbb{E}(S_{yearly}) = \pmb{s}_{0} \circ exp\Biggl(\pmb{\mu}^{Log}_{yearly} + \frac{1}{2} diag\Biggl(\pmb{\Sigma}^{Log}_{yearly}\Biggr)\Biggr)
        * .. math:: Cov(S_{yearly}) = \mathbb{E}(S_{yearly})\mathbb{E}(S_{yearly})^T \circ \Biggl(exp\Biggl(\pmb{\Sigma}^{Log}_{yearly}\Biggr)-1\Biggr)

   From which we can estimate the moments of the linear returns at the time horizon:

        * .. math:: \pmb{\mu}^{Lin}_{yearly} = \frac{1}{\pmb{s}_{0} } \circ \mathbb{E}(S_{yearly}) -1
        * .. math:: \pmb{\Sigma}^{Lin}_{yearly} = \frac{1}{\pmb{s}_{0}\pmb{s}_{0}^{T} } \circ Cov(S_{yearly})

Where :math:`\circ` denotes the Hadamard product (element-wise product).

Note that we could have derived the distribution of linear returns from the distribution of logarithmic returns directly in this case.
Here we demonstrated the general procedure.

In skfolio
==========
In `skfolio`, the above can be achieved using :class:`~skfolio.prior.EmpiricalPrior`
by setting `is_log_normal` to `True` and providing `investment_horizon`. The input `X`
must be linear returns. The conversion to logarithmic returns is performed inside the estimator.

However, as seen in the example :ref:`sphx_glr_auto_examples_data_preparation_plot_1_investment_horizon.py`,
for frequently rebalanced portfolios (investment horizon less than a year), the general procedure and the
below simplified one will give very close results:

#. Take the prices :math:`S_{t}, S_{t+1}, ...,` (for example daily) for all the n securities
#. Transform the daily prices to daily linear returns
#. Estimate the expected returns vector :math:`\mu` and covariance matrix :math:`\Sigma` from the daily linear returns
#. Compute the mean-variance efficient frontier :math:`\max_{w} \Biggl\{w^T \mu - \lambda \times w^T \Sigma w\Biggr\}`

This simplified procedure is the default one used in all `skfolio` examples as most
portfolios are rebalanced with a frequency less than a year.

**In both cases, it is highly recommended to use linear return for the input `X`**
If you need to estimate the moments from logarithmic returns, the conversion from linear
to logarithmic returns should be reformed inside the estimator.

For bonds and options, the general procedure will be implemented in a future release. In the meantime
you can use your own custom :ref:`prior estimator <prior>`.

.. _periodicity_convention:

Periodicity Convention
======================
skfolio estimators work in the periodicity of the input `X`. With daily returns,
moment estimators produce daily expected returns and covariance, prior estimators
produce daily return scenarios, and optimizers consume these per-period inputs
directly. Nothing is projected to the investment horizon inside the optimization.
When horizon projection is needed, it is performed inside the prior estimator (see
:class:`~skfolio.prior.EmpiricalPrior` with `investment_horizon` above), not by
scaling the optimization inputs.

This convention has several benefits:

* It avoids the incorrect moment projection described in the pitfall above.
* Variance-based and scenario-based risk measures stay consistent within a single
  optimization: scenario-based measures (e.g. CVaR, CDaR) consume the return
  scenarios directly, and scenarios have no meaningful horizon-scaled equivalent.
* For ratio objectives such as maximizing the Sharpe ratio, consistent scaling of
  the return-based inputs changes the objective value but not the optimal weights,
  so projecting the inputs adds conversion risk without changing the solution.
* Walk-forward and online evaluation rebalance in units of observations, so
  per-period inputs compose with any rebalancing frequency without rescaling.

Quantities that are paid once rather than earned per period must be converted to
the periodicity of `X`. A transaction cost is paid once per rebalancing while a
position earns its expected return on every period it is held, so the one-off cost
is divided by the expected investment duration: for a 10 basis point cost, daily
returns and a one-month expected holding period, `transaction_costs=0.001 / 21`
(see :ref:`sphx_glr_auto_examples_mean_risk_plot_6_transaction_costs.py`).
Management fees accrue with holding time, so a stated annual fee converts directly
to the return periodicity (e.g. `0.02 / 252` for a 2% annual fee on daily returns).

Annualization happens only at reporting time. :class:`~skfolio.portfolio.Portfolio`
computes measures on the per-observation return series and scales them for display
in the annualized variants (e.g. `annualized_sharpe_ratio`), using its
`annualization_factor` parameter.


.. rubric:: References

.. [1] Note on simple and logarithmic return, Panna Miskolczi (2017)

.. [2] Quant nugget 2: linear vs. compounded returns – common pitfalls in portfolio management, GARP Risk Professional, Meucci (2010)

.. [3] Quant nugget 4: annualization and general projection of skewness, kurtosis and all summary statistics, GARP Risk Professional, Meucci (2010)

.. [4] Quant nugget 5: return calculations for leveraged securities and portfolios, GARP Risk Professional, Meucci (2010)

.. [5] Portfolio optimization and long-term dependence, Carlos León and Alejandro Reveiz

.. [6] Efficient Asset Management: A Practical Guide to Stock Portfolio Optimization and Asset Allocation, Oxford University Press, Richard Michaud and Robert Michaud.

.. [7] Portfolio Optimization Cookbook, Mosek
