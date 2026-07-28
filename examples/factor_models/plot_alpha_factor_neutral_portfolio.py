r"""
===========================================
Alpha Research and Factor-Neutral Portfolio
===========================================

This tutorial shows how to research an alpha signal that forecasts the
idiosyncratic returns of the characteristics-based cross-sectional factor
model :class:`~skfolio.prior.CharacteristicsFactorModel`, and how to trade it
in a factor-neutral long-short portfolio. The methodology is covered in the
:ref:`Alpha Estimators <factor_model_alpha>` and :ref:`Portfolio Construction
<factor_model_portfolio_construction>` sections of the user guide.

We will:

* define an alpha signal that forecasts the factor model's idiosyncratic returns
* evaluate its forecast quality with IC, portfolio and factor-correlation
  diagnostics
* integrate the alpha estimator into the factor model
* optimize a factor-neutral portfolio that allocates to the orthogonal alpha
* jointly tune the optimizer, factor model and alpha estimator with online
  search
* evaluate the strategy over a walk-forward test period
* run ex-ante and ex-post attribution of exposures, risk and performance,
  verifying that the return comes from the orthogonal alpha rather than factor
  premia
"""

# %%
# Data
# ====
# We reuse the synthetic characteristics panel from the :ref:`first tutorial
# <sphx_glr_auto_examples_factor_models_plot_characteristics_factor_model.py>`.
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
# :ref:`first tutorial
# <sphx_glr_auto_examples_factor_models_plot_characteristics_factor_model.py>`
# for more details:
from skfolio.descriptor import (
    AnalystDispersionToPrice,
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
    ShortInterest,
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
# Alpha Research
# ==============
# We now build the alpha signal, a cross-sectional forecast of relative
# idiosyncratic performance across assets at each date. The factor model
# decomposes asset returns into systematic and idiosyncratic components, and
# the signal targets the idiosyncratic returns. With raw returns as target,
# the cross-sectional variation would also include each asset's factor
# exposures times the factor returns, so a signal correlated with the
# exposures would pick up factor premia already captured by the factor model.
# Targeting idiosyncratic returns removes this component and
# keeps the forecast asset-specific (see :ref:`Alpha Estimators
# <factor_model_alpha>`).
#
# We use the first three years for factor-model estimation and alpha
# development and reserve the remaining three years for walk-forward
# evaluation. After fitting the factor model, we use `enrich_asset_panel` to
# add idiosyncratic returns, idiosyncratic variances, regression weights and
# factor exposures to the training panel. We can then iterate on the alpha
# estimator without refitting the factor model:
train_size = 3 * year
panel_train = panel[:train_size]

model.fit(characteristics=panel_train)
factor_model = model.factor_model_

panel_train_enriched = factor_model.enrich_asset_panel(panel_train)

# %%
# Alpha Signal
# ------------
# We build the signal from two characteristics, short interest and analyst
# forecast dispersion. Empirical studies have associated high short interest
# [1]_ and high analyst forecast dispersion [2]_ with lower subsequent
# returns. We therefore combine the two descriptors with equal negative
# weights, so assets with higher values receive lower alpha forecasts.
#
# On real data, such a relationship would have to be discovered and tested.
# Here the data is synthetic, so the relationship is built into the
# generator: a persistent bearish component raises short interest and analyst
# forecast dispersion and lowers future idiosyncratic returns. The
# :class:`~skfolio.descriptor.ShortInterest` and
# :class:`~skfolio.descriptor.AnalystDispersionToPrice` descriptors observe
# this component with noise. Because the true signal is known by
# construction, we can verify at the end of the tutorial that the workflow
# recovers it from the observable characteristics.
#
# We use :class:`~skfolio.alpha.FixedWeightedAlpha` because the descriptor
# directions and relative weights are specified in advance, which keeps the
# focus on the alpha evaluation and portfolio-construction workflow. skfolio
# also provides :class:`~skfolio.alpha.EWSharpeOptimalAlpha` to estimate a
# linear descriptor combination from historical idiosyncratic returns and
# :class:`~skfolio.alpha.PredictorAlpha` to use any ML predictor for
# more flexible relationships:
from skfolio.alpha import FixedWeightedAlpha, alpha_forecast_evaluation
from skfolio.preprocessing import CSGaussianRankScaler
from skfolio.utils.stats import CSWeighting

holding_period = 10

alpha_estimator = FixedWeightedAlpha(
    descriptors=[
        ("short_interest", ShortInterest()),
        ("analyst_dispersion", AnalystDispersionToPrice()),
    ],
    weights=[-1.0, -1.0],
    forecast_scale=7.5e-5,
    scoring_transformer=CSGaussianRankScaler(),
    n_jobs=-1,
)

# %%
# :class:`~skfolio.preprocessing.CSGaussianRankScaler` maps each descriptor and
# the final composite to cross-sectional Gaussian rank scores. This places the
# two descriptors on a comparable scale and limits the influence of extreme
# values. :class:`~skfolio.alpha.FixedWeightedAlpha` normalizes the weights by
# their absolute sum, so `[-1.0, -1.0]` assigns an effective weight of
# :math:`-0.5` to each descriptor.
#
# `forecast_scale` converts one composite-score unit into expected
# idiosyncratic return. Here, `7.5e-5` represents 0.75 basis points of expected
# daily idiosyncratic return per score unit. The alpha forecast should be expressed in
# expected-return units when it is combined with expected factor returns or used
# in an optimization alongside return-denominated quantities such as transaction
# costs, turnover constraints or return targets.
#
# Alpha Forecast Diagnostics
# --------------------------
# :func:`~skfolio.alpha.alpha_forecast_evaluation` fits the estimator on the
# enriched training panel and compares each historical forecast with the mean
# idiosyncratic return over the next ten trading days. `signal_lag=1` pairs a
# forecast observed at :math:`t` with returns beginning at :math:`t+1`. The
# default evaluation step equals the holding period, producing non-overlapping
# target windows. `n_forward_periods=4` extends the decay analysis across four
# consecutive ten-day windows.
#
# We use regression weights for the Pearson IC, calibration and linear
# factor-correlation diagnostics, while the Spearman IC evaluates
# cross-sectional rank ordering and does not use them:
evaluation = alpha_forecast_evaluation(
    alpha_estimator,
    panel_train_enriched,
    holding_period=holding_period,
    signal_lag=1,
    n_forward_periods=4,
    cs_weighting=CSWeighting.REGRESSION,
)

evaluation.ic_summary()

# %%
# The Spearman IC measures how well the forecast orders assets by their
# future idiosyncratic return. The Pearson IC measures the linear relationship
# between forecast magnitudes and future idiosyncratic returns. The ICIR, t-statistic and hit
# rate summarize consistency through time. The mean Spearman IC of 0.045 shows a
# positive rank association between the forecast and subsequent idiosyncratic
# returns. Its 84.3% hit rate means that this association is positive in 84.3%
# of the evaluation windows. The mean Pearson IC of 0.034 and its 78.0% hit rate
# show a weaker but persistent linear relationship. ICIRs of 0.97 and 0.71, with
# t-statistics above 5, show that the signal is consistent through time rather
# than driven by a few windows.

# %%
# Next, we evaluate the signal at the portfolio level with `portfolio_summary`.
# It reports annualized statistics for simple 200% gross long-short portfolios
# formed directly from forecast ranks and values. These portfolios isolate
# signal quality before covariance, constraints and costs are introduced. The
# turnover impact on performance is introduced in the optimizer and the
# walk-forward backtest below, where transaction costs are applied.
evaluation.portfolio_summary()

# %%
# The rank-weighted and z-score-weighted portfolios produce annualized returns
# of 6.83% and 7.36%, with information ratios of 13.64 and 14.27. The high
# information ratios reflect diversification of idiosyncratic noise across the
# broad cross-section. Both portfolios have an 84.3% hit rate. Mean one-way
# turnover of 93.6% and 105.8% per rebalance also shows that implementation costs
# will be important.
#
# Next, we check the choice of `forecast_scale` with `calibration_summary`.
# The `calibration_slope` is the slope from a weighted zero-intercept
# regression of realized targets on forecasts. A value near one indicates that
# `forecast_scale` expresses the alpha in the correct daily-return units:
evaluation.calibration_summary()

# %%
# The calibration slope of 1.003 indicates that the forecast scale is closely
# aligned with idiosyncratic returns. The mean forecast is close to zero
# because the signal is centered cross-sectionally.
#
# We now turn to the plots, starting with the IC accumulated through time. A
# steadily rising cumulative IC indicates that the predictive relationship is
# distributed through the training sample rather than concentrated in a few
# dates:
evaluation.plot_cumulative_ic()

# %%
# Both curves rise steadily with no prolonged flat or negative stretch,
# consistent with the high hit rates of the IC summary.
#
# Next, we check how quickly the signal decay. `plot_ic_decay` re-evaluates
# each forecast over consecutive, disjoint ten-day windows:
evaluation.plot_ic_decay()

# %%
# The IC is strongest in the first window and weakens over the following ones.
# The latent bearish component is highly persistent, so part of its predictive
# power extends beyond the ten-day holding period.
#
# Finally, we check whether the forecast overlaps with the risk factors.
# `plot_factor_correlation` shows the contemporaneous cross-sectional
# correlation between the raw alpha forecast and each factor exposure:
evaluation.plot_factor_correlation()

# %%
# All correlations are small, so the forecast is close to factor neutral.
# Such small overlaps are not a concern for the portfolio below because the
# factor model separates the forecast into spanned alpha and orthogonal alpha
# and the optimization constraints keep the portfolio's factor exposures near
# zero. Unwanted tilts can also be removed at the alpha estimator level with
# `neutralize_against`.

# %%
# Alpha Integration
# =================
# After defining and evaluating the signal, we attach the alpha estimator to
# the factor model:
model.set_params(alpha_estimator=alpha_estimator)

# %%
# With `alpha_estimator` configured, the factor model enriches the input panel,
# fits the estimator and decomposes its forecast into spanned alpha and orthogonal
# alpha. The default `spanned_alpha_shrinkage=1.0` uses only factor-implied asset
# expected returns. The alpha forecast therefore contributes to expected returns
# only through the orthogonal alpha.
# `orthogonal_alpha_confidence` controls shrinkage of the orthogonal alpha
# toward zero (see :ref:`Spanned and Orthogonal Alpha
# <factor_model_spanned_alpha>`).
#
# Factor-Neutral Optimization
# ===========================
# The :ref:`previous tutorial
# <sphx_glr_auto_examples_factor_models_plot_factor_constrained_portfolio.py>`
# captured factor premia through explicit exposure targets. Here we do the
# opposite: we constrain market, industry and style exposures close to zero and
# use the orthogonal alpha as the modeled expected return, an approach common in
# statistical arbitrage.
#
# We maximize mean-variance utility so that the optimizer balances expected
# return against variance and, in the backtest below, transaction costs. We use
# a risk-aversion coefficient of 0.2, bound each of the 13 style exposures
# within :math:`\pm 0.05`, set every industry exposure to zero and limit
# individual positions to :math:`\pm 1.5\%`:
from sklearn import set_config

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction

set_config(enable_metadata_routing=True)

X = panel.to_dataframe(fields="returns")
X_train = X.iloc[:train_size]
industry_names = panel.fields["industry"].levels

style_factors = [
    "beta",
    "momentum",
    "size",
    "non_linear_size",
    "value",
    "earnings_yield",
    "growth",
    "profitability",
    "investment",
    "dividend_yield",
    "leverage",
    "liquidity",
    "volatility",
]

mvo = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    risk_measure=RiskMeasure.VARIANCE,
    risk_aversion=0.2,
    prior_estimator=model,
    max_weights=0.015,  # Limit long positions to 1.5%.
    min_weights=-0.015,  # Limit short positions to -1.5%.
    budget=0.0,  # Enforce dollar neutrality.
    max_long=1.5,  # Cap long exposure at 150%.
    linear_constraints=[
        # Bound style exposures within +/- 0.05.
        *[f"{name} <= 0.05" for name in style_factors],
        *[f"{name} >= -0.05" for name in style_factors],
        # Set industry exposures to zero.
        *[f"{name} == 0" for name in industry_names],
    ],
)

mvo.fit(X_train, characteristics=panel_train)

print(f"Long positions: {(mvo.weights_ > 1e-8).sum()}")
print(f"Short positions: {(mvo.weights_ < -1e-8).sum()}")
print(f"Gross exposure: {np.abs(mvo.weights_).sum():.2f}")

# %%
# `budget=0.0` makes the portfolio dollar neutral and `max_long=1.5` caps the
# long exposure at 150%. Dollar neutrality implies an equally sized short
# exposure, so the gross exposure can reach 300%. For the market factor,
# `budget=0.0` is equivalent to a `"market == 0"` constraint on the global factor
# exposure, so the explicit constraint is unnecessary. At the gross-exposure cap, the
# 1.5% position limit requires at least 100 long and 100 short positions,
# preventing the portfolio from being concentrated in fewer names.
#
# With market and industry exposures at zero and style exposures confined to
# narrow bands, most forecast risk is idiosyncratic. In this factor model,
# orthogonal portfolios are penalized only through per-asset idiosyncratic
# variances. Model incompleteness can therefore understate their risk. In
# addition to `orthogonal_alpha_confidence`, the optimizer's uncertainty-set
# parameters provide robust optimization in the orthogonal space (see
# :ref:`Orthogonal Space Regularization
# <factor_model_orthogonal_space_regularization>`).
#
# We retrieve the factor model fitted inside the optimizer for attribution:
factor_model = mvo.prior_estimator_.factor_model_

# %%
# Ex-Ante Attribution
# ===================
# Before trading, we verify that the portfolio behaves as designed. Predicted
# attribution decomposes the optimized portfolio's factor exposures, forecast
# risk and expected return:
portfolio = mvo.predict(X_train)
predicted_attrib = portfolio.predicted_attribution(factor_model=factor_model)

# %%
# We first inspect factor exposures:
predicted_attrib.plot_exposure(top_n=15)

# %%
# Market and industry exposures are zero, while each style exposure remains
# within its :math:`\pm 0.05` constraint.
#
# We then inspect forecast volatility contributions:
predicted_attrib.plot_vol_contrib(top_n=15)

# %%
# The idiosyncratic component dominates the risk forecast, with only small
# contributions from residual style exposures.
#
# Next, we inspect expected return contributions:
predicted_attrib.plot_return_contrib(top_n=15)

# %%
# The idiosyncratic component dominates expected return. This is consistent with
# the modeled expected return coming primarily from orthogonal alpha. In the previous
# tutorial, the idiosyncratic
# expected-return contribution was zero because the factor model had no alpha
# estimator.
#
# The same decomposition is available as a DataFrame:
predicted_attrib.summary_df()

# %%
# Idiosyncratic risk contributes 95.47% of forecast variance, while orthogonal
# alpha contributes 7.00% of the 8.07% expected return. Residual systematic
# exposures within the permitted style bands contribute the remaining 1.07%.

# %%
# Online Hyperparameter Search
# ============================
# The optimizer, factor model and alpha estimator follow the scikit-learn
# parameter convention. Their parameters can therefore be searched jointly
# through the nested estimator hierarchy. Here, we search over:
#
# * the optimizer's `risk_aversion`
# * the factor model's `inv_idio_variance_weight_shrinkage`
# * the alpha estimator's descriptor `weights`
# * the alpha estimator's `scoring_transformer`
#
# We sample the scalar parameters from continuous distributions. Risk aversion
# uses a log-uniform distribution because it is a positive scale parameter,
# while the shrinkage coefficient is sampled uniformly over its valid interval.
# The descriptor weights and the scoring transformer are searched over a small
# set of candidates.
# :class:`~skfolio.model_selection.OnlineRandomizedSearch`
# draws 100 configurations and evaluates each candidate in a single incremental
# pass using `partial_fit`. The search uses only the training sample, with two
# years and one month of warmup followed by non-overlapping ten-day validation
# windows. Candidates are ranked by portfolio-level annualized Sharpe ratio
# with the same transaction-cost and entry conventions used in the backtest.
#
# .. code-block:: python
#
#     from scipy.stats import loguniform, uniform
#     from sklearn.base import clone
#
#     from skfolio.measures import RatioMeasure
#     from skfolio.model_selection import OnlineRandomizedSearch
#     from skfolio.preprocessing import CSGaussianRankScaler, CSStandardScaler
#
#     search_estimator = clone(mvo).set_params(
#         transaction_costs=0.001 / holding_period,
#         fallback="previous_weights",
#     )
#
#     search = OnlineRandomizedSearch(
#         estimator=search_estimator,
#         param_distributions={
#             "risk_aversion": loguniform(0.05, 0.8),
#             "prior_estimator__inv_idio_variance_weight_shrinkage": uniform(
#                 0.0, 1.0
#             ),
#             "prior_estimator__alpha_estimator__weights": [
#                 [-2.0, -1.0],
#                 [-1.0, -1.0],
#                 [-1.0, -2.0],
#             ],
#             "prior_estimator__alpha_estimator__scoring_transformer": [
#                 CSStandardScaler(),
#                 CSGaussianRankScaler(),
#             ],
#         },
#         n_iter=100,
#         scoring=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
#         warmup_size=2 * year + month,
#         test_size=holding_period,
#         random_state=0,
#         entry_rebalancing_params={
#             "transaction_costs": 0.0,
#             "fallback": None,
#         },
#         n_jobs=-1,
#     )
#     search.fit(X_train, characteristics=panel_train)
#
#     print(search.best_params_)
#     print(search.best_score_)
#
# `cv_results_` contains every sampled parameter set, its aggregate score,
# rank and fit time. `best_estimator_` exposes the selected estimator after its
# complete online pass.

# %%
# Walk-Forward Backtest
# =====================
# Now let's backtest the strategy.
# :func:`~skfolio.model_selection.online_predict` walks forward through the
# data, updates the model with `partial_fit` and builds one portfolio per test
# window, in a single pass over the data. The first three years, the sample
# used for the alpha research above, warm up the model before the first
# rebalancing. The strategy then trades over the remaining three years,
# rebalancing every ten trading days to match the alpha-evaluation horizon.
# The model keeps learning from each new window, so every rebalancing uses only
# information available at that date.
#
# We also add:
#
# * `transaction_costs=0.001 / holding_period`: skfolio deducts transaction costs
#   directly from expected returns, which are expressed per observation period
#   (here daily). The 10 basis points are paid once per rebalancing while a
#   position earns its return on every day it is held, so we amortize the cost
#   over the ten-day holding period to convert it to a daily cost (see
#   :ref:`Periodicity Convention <periodicity_convention>`).
# * `fallback="previous_weights"` keeps the latest valid allocation when a
#   rebalancing problem is infeasible (see
#   :ref:`sphx_glr_auto_examples_mean_risk_plot_17_failure_and_fallbacks.py`).
# * `entry_rebalancing_params` overrides estimator parameters only for the
#   first portfolio, which starts from cash. Setting `transaction_costs=0.0` at
#   entry avoids charging costs on the full initial ramp-up and lets the first
#   rebalance reach its target allocation instead of building exposure over
#   several rebalancings. Setting `fallback=None` requires the initial
#   optimization to produce a valid allocation because no previous portfolio is
#   available at entry.
#
# Borrow costs and market impact can be added through the optimizer's
# `add_objective` and `add_constraints` parameters, with native support planned
# for a future release:
from skfolio.model_selection import online_predict

mvo.set_params(transaction_costs=0.001 / holding_period, fallback="previous_weights")

mpp = online_predict(
    estimator=mvo,
    X=X,
    warmup_size=train_size,
    test_size=holding_period,
    params={"characteristics": panel},
    entry_rebalancing_params={"transaction_costs": 0.0, "fallback": None},
)

mpp.summary()

# %%
# The walk-forward strategy earns 7.59% annualized with 3.19% volatility, an
# annualized Sharpe ratio of 2.38 and a maximum drawdown of 2.86%. The backtest
# contains 74 rebalancings, with no optimization failures or fallback
# portfolios. These results include recurring transaction costs.
#
# We plot the walk-forward performance net of transaction costs:
mpp.plot_cumulative_returns()

# %%
# Next, we check the long, short, net and gross exposures through time. Net
# exposure remains zero and gross exposure stays within the 300% cap:
mpp.plot_long_short_exposure()

# %%
# Ex-Post Attribution
# ===================
# Now that we have the backtest, let's check whether realized performance was
# concentrated in idiosyncratic returns, as intended by the orthogonal alpha
# forecast. `realized_attribution` decomposes the walk-forward portfolio using
# realized factor returns, exposures and idiosyncratic returns. For this
# descriptive ex-post analysis, we refit the factor model over the completed
# sample. This fit occurs after the backtest and does not enter any portfolio
# decision:
model.fit(characteristics=panel)
realized_factor_model = model.factor_model_
realized_attrib = mpp.realized_attribution(factor_model=realized_factor_model)

# %%
# For each factor, we plot the mean realized exposure and its standard
# deviation over the backtest:
realized_attrib.plot_exposure(top_n=15)

# %%
# Mean realized exposures remain close to zero. Their standard deviations
# summarize time variation from rebalances, changing factor exposures and
# within-period weight drift.
#
# We then inspect realized return contributions:
fig = realized_attrib.plot_return_contrib(top_n=15)
# show(fig) is only used for the documentation sticker.
show(fig)

# %%
# |
#
# The error bars show 95% confidence intervals on annualized mean return
# contributions. The idiosyncratic return contribution is positive. Residual
# factor exposures make a small aggregate contribution. The realized return decomposition is
# consistent with the orthogonal alpha forecast.
#
# The summary DataFrame adds `unexplained`, the residual between
# observed portfolio returns and model-attributed returns. The portfolio
# returns are net of transaction costs while the factor decomposition
# explains gross returns, so the cost drag falls into this residual, as
# would management fees, slippage and model misspecification:

realized_attrib.summary_df()

# %%
# Idiosyncratic returns contribute 8.06% annualized and 94.93% of realized
# variance. Systematic factors contribute 0.24%, while the unexplained component
# subtracts 0.66%. Total return over the attribution sample is 7.64% annualized.

# %%
# Conclusion
# ==========
# We recovered the synthetic idiosyncratic alpha from short interest and analyst
# forecast dispersion, integrated it into the factor model and used the orthogonal
# alpha in a factor-neutral portfolio.
#
# .. seealso::
#       The :ref:`Alpha Estimators <factor_model_alpha>` section of the
#       :ref:`Factor Models <factor_models>` user guide covers the learned
#       estimators :class:`~skfolio.alpha.EWSharpeOptimalAlpha` and
#       :class:`~skfolio.alpha.PredictorAlpha`, and the :ref:`Portfolio
#       Construction <factor_model_portfolio_construction>` section covers
#       the optimizer conventions and orthogonal-space regularization.

# %%
# References
# ==========
# .. [1] H. Desai, K. Ramesh, S. R. Thiagarajan, and B. V. Balachandran,
#    "An Investigation of the Informational Role of Short Interest in the Nasdaq
#    Market", *The Journal of Finance*, vol. 57, no. 5, pp. 2263-2287 (2002).
#    `doi:10.1111/0022-1082.00495
#    <https://doi.org/10.1111/0022-1082.00495>`_.
#
# .. [2] K. B. Diether, C. J. Malloy, and A. Scherbina, "Differences of Opinion
#    and the Cross Section of Stock Returns", *The Journal of Finance*, vol. 57,
#    no. 5, pp. 2113-2141 (2002). `doi:10.1111/0022-1082.00490
#    <https://doi.org/10.1111/0022-1082.00490>`_.
#
# .. [3] R. C. Grinold and R. N. Kahn, *Active Portfolio Management: A
#    Quantitative Approach for Producing Superior Returns and Controlling Risk*,
#    McGraw-Hill (1999).
#
# .. [4] G. A. Paleologo, *The Elements of Quantitative Investing*, Wiley
#    Finance (2025).
