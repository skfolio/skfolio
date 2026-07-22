r"""
============================================
Factor-Constrained Portfolio and Attribution
============================================

This tutorial shows how to build a dollar-neutral long-short portfolio with
factor tilts, using the characteristics-based cross-sectional factor model
:class:`~skfolio.prior.CharacteristicsFactorModel` and the optimizer
:class:`~skfolio.optimization.MeanRisk`.
The methodology is covered in the :ref:`Portfolio Construction
<factor_model_portfolio_construction>` and :ref:`Attribution
<factor_model_attribution>` sections of the user guide.

We will:

* optimize a portfolio with explicit factor exposure constraints
* jointly tune the optimizer and factor model with online search
* backtest it with monthly walk-forward rebalancing
* perform ex-ante and ex-post attribution of exposures, risk and performance
"""

# %%
# Data
# ====
# We reuse the synthetic characteristics panel from the :ref:`previous tutorial
# <sphx_glr_auto_examples_factor_models_plot_1_characteristics_factor_model.py>`.
# It covers 500 assets over 1,500 trading days and includes late listings,
# delistings, holidays and missing characteristics:
import numpy as np
from plotly.io import show

from skfolio.datasets import make_synthetic_characteristics

panel = make_synthetic_characteristics(
    n_assets=500, n_observations=1500, random_state=0
)

# %%
# Model Definition
# ================
# Next, we rebuild the 24-factor model with one global market factor, 10
# industry factors and 13 style factors built from 29 descriptors. See the
# :ref:`previous tutorial
# <sphx_glr_auto_examples_factor_models_plot_1_characteristics_factor_model.py>`
# for more details:
from skfolio.descriptor import (
    AssetTurnover,
    AssetsGrowthRate,
    BookLeverage,
    BookToPrice,
    CapexToAssetsChangeInIntensity,
    CashFlowToAssets,
    CashFlowToPrice,
    DebtToAssets,
    DividendToPrice,
    EWAmihudIlliquidity,
    EWMarketBeta,
    EWMomentum,
    EWResidualVolatility,
    EWShareTurnover,
    EWVolatility,
    EarningsChangeToPrice,
    EarningsToPrice,
    EbitdaToEnterpriseValue,
    ForwardEarningsToPrice,
    GrossMargin,
    GrossProfitability,
    IssuanceGrowthRate,
    LogMarketCap,
    MarketLeverage,
    ReturnOnAssets,
    ReturnOnEquity,
    SalesGrowthRate,
    SalesToPrice,
    ShareholderYield,
)
from skfolio.factor_exposure import (
    DerivedFactor,
    FixedWeightedFactor,
    GlobalFactor,
    OneHotCategoricalFactors,
)
from skfolio.moments import EWMu, RegimeAdjustedEWCovariance
from skfolio.prior import CharacteristicsFactorModel, EmpiricalPrior

month = 21
quarter = 3 * month
half_year = 6 * month
year = 12 * month

global_factor = GlobalFactor(family="market")

industry_factors = OneHotCategoricalFactors(category="industry", family="industry")

beta_factor = FixedWeightedFactor(
    descriptors=[("market_beta", EWMarketBeta(half_life=year))],
    transform_by_group="industry",
)

momentum_factor = FixedWeightedFactor(
    descriptors=[("momentum", EWMomentum(half_life=half_year, skip=month))],
    transform_by_group="industry",
)

size_factor = FixedWeightedFactor(
    descriptors=[("log_mcap", LogMarketCap())], transform_by_group="industry"
)

non_linear_size_factor = DerivedFactor(
    source="size", func=lambda x: x**3, transform_by_group="industry"
)

value_factor = FixedWeightedFactor(
    descriptors=[
        ("book_to_price", BookToPrice()),
        ("sales_to_price", SalesToPrice()),
        ("cash_flow_to_price", CashFlowToPrice()),
    ],
    weights=[0.8, 0.1, 0.1],
    transform_by_group="industry",
)

earnings_yield_factor = FixedWeightedFactor(
    descriptors=[
        ("fwd_earnings_to_price", ForwardEarningsToPrice()),
        ("earnings_to_price", EarningsToPrice()),
        ("enterprise_multiple", EbitdaToEnterpriseValue()),
    ],
    transform_by_group="industry",
)

growth_factor = FixedWeightedFactor(
    descriptors=[
        ("earnings_change_to_price", EarningsChangeToPrice(lag=year)),
        ("sales_growth", SalesGrowthRate(lag=year)),
    ],
    transform_by_group="industry",
)

profitability_factor = FixedWeightedFactor(
    descriptors=[
        ("asset_turnover", AssetTurnover()),
        ("gross_profitability", GrossProfitability()),
        ("gross_margin", GrossMargin()),
        ("return_on_assets", ReturnOnAssets()),
        ("return_on_equity", ReturnOnEquity()),
        ("cash_flow_to_assets", CashFlowToAssets()),
    ],
    transform_by_group="industry",
)

investment_factor = FixedWeightedFactor(
    descriptors=[
        ("asset_growth", AssetsGrowthRate(lag=year)),
        ("issuance_growth", IssuanceGrowthRate(lag=year)),
        ("capex_growth", CapexToAssetsChangeInIntensity(lag=year)),
    ],
    transform_by_group="industry",
)

dividend_yield_factor = FixedWeightedFactor(
    descriptors=[
        ("dividend_to_price", DividendToPrice()),
        ("shareholder_yield", ShareholderYield()),
    ],
    weights=[0.7, 0.3],
    transform_by_group="industry",
)

leverage_factor = FixedWeightedFactor(
    descriptors=[
        ("market_leverage", MarketLeverage()),
        ("debt_to_assets", DebtToAssets()),
        ("book_leverage", BookLeverage()),
    ],
    transform_by_group="industry",
)

liquidity_factor = FixedWeightedFactor(
    descriptors=[
        ("share_turnover", EWShareTurnover(half_life=quarter)),
        ("amihud_illiquidity", EWAmihudIlliquidity()),
    ],
    transform_by_group="industry",
)

volatility_factor = FixedWeightedFactor(
    descriptors=[
        ("vol", EWVolatility(half_life=quarter)),
        (
            "residual_vol",
            EWResidualVolatility(half_life=quarter, beta_half_life=quarter),
        ),
    ],
    transform_by_group="industry",
)

model = CharacteristicsFactorModel(
    factors=[
        ("market", global_factor),
        ("industry", industry_factors),
        ("beta", beta_factor),
        ("momentum", momentum_factor),
        ("size", size_factor),
        ("non_linear_size", non_linear_size_factor),
        ("value", value_factor),
        ("earnings_yield", earnings_yield_factor),
        ("growth", growth_factor),
        ("profitability", profitability_factor),
        ("investment", investment_factor),
        ("dividend_yield", dividend_yield_factor),
        ("leverage", leverage_factor),
        ("liquidity", liquidity_factor),
        ("volatility", volatility_factor),
    ],
    neutralize_against={
        "non_linear_size": ["size"],
        "volatility": ["beta"],
    },
    constrained_families=[("industry", None)],
    exposure_lag=1,
    inv_idio_variance_weight_shrinkage=0.5,
    factor_prior_estimator=EmpiricalPrior(
        covariance_estimator=RegimeAdjustedEWCovariance(
            half_life=half_year, corr_half_life=year, regime_half_life=month
        ),
        mu_estimator=EWMu(half_life=year),
    ),
    n_jobs=-1,
)

# %%
# Factor-Constrained Optimization
# ===============================
# The factor model is a prior estimator, so we can pass it to any skfolio
# optimizer through `prior_estimator`. The optimizer fits it internally and
# consumes its expected returns, covariance and scenarios, while scikit-learn
# metadata routing forwards the `characteristics` panel to the prior.
#
# Let's define the portfolio. We maximize the Sharpe ratio under explicit
# factor exposure constraints. The synthetic generator gives momentum and
# dividend yield positive premia and investment a weak premium, so we
# target:
#
# * long momentum, with exposure at least 1.0,
# * long dividend yield, set exactly to 1.5,
# * short investment, set exactly to -1.0.
#
# We also set beta, size, volatility and every industry exposure to exactly
# zero, and bound each remaining style within :math:`\pm 0.05`:
from sklearn import set_config

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction

set_config(enable_metadata_routing=True)

X = panel.to_dataframe(fields="returns")
industry_names = panel.fields["industry"].levels

bounded_styles = [
    "non_linear_size",
    "value",
    "earnings_yield",
    "growth",
    "profitability",
    "leverage",
    "liquidity",
]

mvo = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.VARIANCE,
    prior_estimator=model,
    max_weights=0.05,  # limit individual positions to 5%
    min_weights=-0.05,  # allow short positions and limit to -5%
    budget=0.0,  # dollar neutral
    max_long=1.0,  # cap long exposure at 100%
    linear_constraints=[
        # Style exposures
        "momentum >= 1.0",
        "dividend_yield == 1.5",
        "investment == -1.0",
        # Styles set to zero
        "beta == 0",
        "size == 0",
        "volatility == 0",
        # Remaining styles within +/- 0.05
        *[f"{name} <= 0.05" for name in bounded_styles],
        *[f"{name} >= -0.05" for name in bounded_styles],
        # Industries set to zero
        *[f"{name} == 0" for name in industry_names],
    ],
)

mvo.fit(X, characteristics=panel)

print(f"Long positions: {(mvo.weights_ > 1e-8).sum()}")
print(f"Short positions: {(mvo.weights_ < -1e-8).sum()}")
print(f"Gross exposure: {np.abs(mvo.weights_).sum():.2f}")

# %%
# `budget=0.0` makes the portfolio dollar neutral and `max_long=1.0` caps the
# long exposure at 100%. Dollar neutrality implies an equally sized short
# exposure, so the gross exposure can reach 200%. Individual positions are
# limited to :math:`\pm 5\%`. We will add transaction costs and a fallback in
# the walk-forward backtest below, where each rebalancing starts from the
# previous allocation.
#
# In `linear_constraints`, an expression on a factor name (e.g.
# `"momentum >= 1.0"`) applies to the portfolio exposure to that factor.
# Industry neutrality needs one constraint per industry factor. A single
# `"industry == 0"` on the family name would only force industry exposures to
# offset each other. For the market factor, `budget=0.0` is equivalent to a
# `"market == 0"` constraint on the intercept, so the explicit constraint is
# unnecessary.
#
# The factor model fitted inside the optimizer is available through
# `prior_estimator_`. We keep a reference to it for attribution below:
factor_model = mvo.prior_estimator_.factor_model_

# %%
# Ex-Ante Attribution
# ===================
# Ex-ante attribution reports the portfolio's factor exposures and decomposes
# its forecast risk and expected return into systematic and idiosyncratic
# contributions. We obtain it from the predicted portfolio with
# `predicted_attribution`:
portfolio = mvo.predict(X)
predicted_attrib = portfolio.predicted_attribution(factor_model=factor_model)
# Equivalent lower-level call on the factor model:
# factor_model.predicted_attribution(weights=mvo.weights_)

# %%
# First, we verify that the optimizer delivered the targeted exposures:
predicted_attrib.plot_exposure(top_n=15)

# %%
# The dividend-yield factor sits at its 1.5 target and the investment factor
# at its -1.0 target. Momentum is bounded below by its 1.0 floor and can
# exceed it when the Sharpe-maximizing objective concentrates in the factor
# with the strongest forecast premium. The market, industry and neutralized
# style exposures are zero, and the remaining styles stay within their
# :math:`\pm 0.05` bands.
#
# Next, we look at where the predicted risk comes from:
predicted_attrib.plot_vol_contrib(top_n=15)

# %%
# Each factor's volatility contribution is its exposure times its standalone
# volatility times its correlation with the portfolio (the
# exposure-volatility-correlation decomposition). The idiosyncratic
# contribution comes from the part of the allocation that moves into
# orthogonal directions once market and industry exposures are forced to
# zero.
#
# Let's do the same for the expected return:
predicted_attrib.plot_return_contrib(top_n=15)

# %%
# The targeted factors drive the expected return. The idiosyncratic
# contribution is exactly zero because no alpha estimator is attached, so
# the model forecasts no return in the orthogonal space.
#
# We can also plot each factor's expected return contribution against its
# volatility contribution:
predicted_attrib.plot_return_vs_vol_contrib(top_n=15)

# %%
# The targeted factors drive both dimensions, while the idiosyncratic
# component lies on the zero-return axis, carrying risk without forecast
# reward.
#
# The same information is also available as DataFrames. Let's start with the
# summary: it reports volatility and return contributions for the systematic,
# idiosyncratic and total components:
predicted_attrib.summary_df()

# %%
# Each volatility contribution is the corresponding variance contribution
# divided by total volatility, so the systematic and idiosyncratic rows sum
# to the total volatility forecast.
#
# The family breakdown aggregates exposures and contributions by factor
# family. As imposed by the constraints, we can see that market and
# industries carry no risk or return:
predicted_attrib.families_df()

# %%
# The factor breakdown reports per-factor exposures, standalone statistics,
# and contributions to portfolio risk and expected return:
predicted_attrib.factors_df()

# %%
# Finally, the asset breakdown reports per-asset weights and volatility and
# return contributions, split into total, systematic and idiosyncratic parts.
# We show only the first rows:
predicted_attrib.assets_df().head()

# %%
# Online Hyperparameter Search
# ============================
# The optimizer, factor model and uncertainty-set estimator follow the
# scikit-learn parameter convention. Their parameters can therefore be searched
# jointly through the nested estimator hierarchy. Here, we search over:
#
# * the half-life used to estimate expected factor returns
# * the shrinkage toward inverse-idiosyncratic-variance regression weights
# * the regularization strength applied to covariance in directions orthogonal
#   to the factor span (see :ref:`Orthogonal Space Regularization
#   <factor_model_orthogonal_space_regularization>`)
#
# We sample each parameter from a continuous distribution. The half-life and uncertainty
# radius use log-uniform distributions because they are positive scale parameters, while
# the shrinkage coefficient is sampled uniformly over its valid interval.
#
# :class:`~skfolio.model_selection.OnlineRandomizedSearch` draws 100
# configurations and evaluates each candidate in a single incremental pass using
# `partial_fit`. Candidates are ranked by portfolio-level annualized Sharpe ratio
# over non-overlapping monthly validation windows, with the same transaction-cost
# and entry conventions used in the backtest. We use the first four years for
# model selection and reserve the remaining observations, approximately two
# years, for a final holdout evaluation.
#
# .. code-block:: python
#
#     from scipy.stats import loguniform, uniform
#     from sklearn.base import clone
#
#     from skfolio.measures import RatioMeasure
#     from skfolio.model_selection import OnlineRandomizedSearch
#     from skfolio.uncertainty_set import OrthogonalCovarianceUncertaintySet
#
#     search_size = 4 * year
#     search_estimator = clone(mvo).set_params(
#         covariance_uncertainty_set_estimator=(
#             OrthogonalCovarianceUncertaintySet()
#         ),
#         transaction_costs=0.001 / month,
#         fallback="previous_weights",
#     )
#
#     search = OnlineRandomizedSearch(
#         estimator=search_estimator,
#         param_distributions={
#             "prior_estimator__factor_prior_estimator__mu_estimator__half_life": (
#                 loguniform(half_year, year)
#             ),
#             "prior_estimator__inv_idio_variance_weight_shrinkage": uniform(
#                 0.0, 1.0
#             ),
#             "covariance_uncertainty_set_estimator__radius": loguniform(
#                 0.1, 10.0
#             ),
#         },
#         n_iter=100,
#         scoring=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
#         warmup_size=2 * year + month,
#         test_size=month,
#         random_state=0,
#         entry_rebalancing_params={
#             "transaction_costs": 0.0,
#             "fallback": None,
#         },
#         n_jobs=-1,
#     )
#     search.fit(
#         X.iloc[:search_size],
#         characteristics=panel[:search_size],
#     )
#
#     print(search.best_params_)
#     print(search.best_score_)
#
# `best_score_` is the aggregate validation score used for model selection, not
# the performance of the final two-year holdout. `cv_results_` contains every
# sampled parameter set, its score, rank and fit time. `best_estimator_` exposes
# the selected estimator after its complete online pass. For a small set of
# discrete candidates, :class:`~skfolio.model_selection.OnlineGridSearch` can be
# used instead.

# %%
# Walk-Forward Backtest
# =====================
# Now let's backtest the strategy.
# :func:`~skfolio.model_selection.online_predict` walks forward through the
# data, updates the model with `partial_fit` and builds one portfolio per
# test window, in a single pass over the data. We rebalance monthly and
# reserve two years plus one month of warmup for the descriptors and
# estimators warmups (see :ref:`Warmup Periods <factor_model_warmup>`).
#
# We also add:

# * `transaction_costs=0.001 / month`: skfolio deducts transaction costs
#   directly from expected returns, which are expressed per observation
#   period (here daily). The 10 basis points are paid once per rebalancing
#   while a position earns its return on every day it is held, so we
#   amortize the cost over the one-month holding period to convert it to a
#   daily cost (see :ref:`Periodicity Convention <periodicity_convention>`).
# * `fallback="previous_weights"` keeps the latest valid allocation when a
#   rebalancing problem is infeasible (see
#   :ref:`sphx_glr_auto_examples_mean_risk_plot_17_failure_and_fallbacks.py`).
# * `entry_rebalancing_params` overrides estimator parameters only for the
#   first portfolio, which starts from cash. Setting `transaction_costs=0.0` at
#   entry avoids charging costs on the full initial ramp-up and lets the first
#   rebalance reach its target allocation instead of building exposure over
#   several rebalancings.
#
# Borrow costs and market impact can be added through the optimizer's
# `add_objective` and `add_constraints` parameters, with native support
# planned for a future release:
from skfolio.model_selection import online_predict

warmup = 2 * year + month
mvo.set_params(transaction_costs=0.001 / month, fallback="previous_weights")

mpp = online_predict(
    estimator=mvo,
    X=X,
    warmup_size=warmup,
    test_size=month,
    params={"characteristics": panel},
    entry_rebalancing_params={"transaction_costs": 0.0, "fallback": None},
)

print(f"Fallback portfolios: {mpp.n_fallback_portfolios}")
print(mpp.summary())

# %%
# `online_predict` returns a
# :class:`~skfolio.portfolio.MultiPeriodPortfolio` with one portfolio per
# rebalancing. `n_fallback_portfolios` counts the rebalancings that fell back
# to the previous weights.
#
# Let's plot the out-of-sample performance:
mpp.plot_cumulative_returns()

# %%
# Next, we check the long, short, net and gross exposures through time. Net
# exposure remains zero and gross exposure stays within the 200% cap:
mpp.plot_long_short_exposure()

# %%
# Ex-Post Attribution
# ===================
# Now that we have the backtest, let's find out which factors drove the
# realized performance. `realized_attribution` decomposes the walk-forward
# portfolio, whose weights vary through time, using the realized factor
# returns, exposures and idiosyncratic returns. It also reports standard
# errors that separate genuine contributions from estimation noise:
realized_attrib = mpp.realized_attribution(factor_model=factor_model)

# %%
# As in the ex-ante section, we start with the exposures. For each factor we
# plot the mean exposure over the backtest and its standard deviation through
# time:
realized_attrib.plot_exposure(top_n=15)

# %%
# Dividend yield remains close to its 1.5 target, while investment stays
# negative but averages less short than its -1.0 rebalancing target as
# realized exposures drift between monthly rebalances. Momentum averages well
# above its 1.0 floor and has the widest variation.
#
# Next, we look at where the realized risk came from:
realized_attrib.plot_vol_contrib(top_n=15)

# %%
# Realized volatility is split between intended factor tilts and orthogonal
# risk. Idiosyncratic risk is the largest single contribution, while dividend
# yield and momentum are the largest systematic contributors.
#
# Next, we decompose realized return into factor and idiosyncratic contributions:
fig = realized_attrib.plot_return_contrib(top_n=15)
show(fig)

# %%
# |
# The error bars show 95% confidence intervals on annualized mean return
# contributions. Momentum and dividend yield are the main positive factor
# contributors. Because no alpha estimator is attached, any realized
# idiosyncratic contribution is uncompensated risk.
#
# Finally, we plot each factor's realized return contribution against its
# volatility contribution. Marker sizes are proportional to the absolute
# portfolio exposure, and the idiosyncratic component displays as a
# fixed-size diamond:
realized_attrib.plot_return_vs_vol_contrib(top_n=15)

# %%
# The same results are available as DataFrames. Compared with the ex-ante
# summary, the realized breakdown adds `unexplained`, the residual between
# observed portfolio returns and model-attributed returns. The portfolio
# returns are net of transaction costs while the factor decomposition
# explains gross returns, so the cost drag falls into this residual, as
# would management fees, slippage and model misspecification:
realized_attrib.summary_df()

# %%
# The per-factor breakdown reports each factor's average realized exposure,
# standalone statistics and contributions:
realized_attrib.factors_df().head()

# %%
# Rolling Attribution
# ===================
# Finally, let's see how the exposures evolved through time. Rolling
# attribution repeats the realized attribution over rolling windows, by
# default 60 observations stepped by 21:
rolling_realized_attrib = mpp.rolling_realized_attribution(
    factor_model=factor_model,
    compute_asset_breakdowns=False,
)

rolling_realized_attrib.plot_exposure(top_n=15)

# %%
# Dividend yield stays close to 1.5 throughout the backtest. Momentum varies
# more because 1.0 is a minimum exposure rather than an exact target, allowing
# the optimizer to increase it when this improves the forecast Sharpe ratio.
# Investment remains negative, while the other exposures stay close to zero.
# The shaded areas show one standard deviation of exposure within each rolling
# window.

# %%
# Conclusion
# ==========
# We optimized a dollar-neutral portfolio with explicit factor tilts,
# verified the targeted exposures ex ante, backtested it with
# monthly walk-forward rebalancing and attributed the realized risk and
# performance to the factors ex post.
#
# The next tutorial attaches an alpha estimator to the factor model and
# builds a factor-neutral portfolio whose return comes from the orthogonal
# alpha component.
#
# .. seealso::
#       The :ref:`Portfolio Construction
#       <factor_model_portfolio_construction>` and :ref:`Attribution
#       <factor_model_attribution>` sections of the :ref:`Factor Models
#       <factor_models>` user guide cover the methodology in depth.

# %%
# References
# ==========
# .. [1] R. C. Grinold and R. N. Kahn, *Active Portfolio Management: A
#    Quantitative Approach for Producing Superior Returns and Controlling Risk*,
#    McGraw-Hill (1999).
#
# .. [2] G. A. Paleologo, *The Elements of Quantitative Investing*, Wiley
#    Finance (2025).
#
# .. [3] D. P. Palomar, *Portfolio Optimization: Theory and Application*,
#    Chapters 3 and 14, Cambridge University Press (2025).
#    `doi:10.1017/9781009428095 <https://doi.org/10.1017/9781009428095>`_.
#
# .. [4] D. Goldfarb and G. Iyengar, "Robust Portfolio Selection Problems",
#    *Mathematics of Operations Research*, vol. 28, no. 1, pp. 1-38 (2003).
#    `doi:10.1287/moor.28.1.1.14260
#    <https://doi.org/10.1287/moor.28.1.1.14260>`_.
