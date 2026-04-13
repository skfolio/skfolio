r"""
==========================================
Online Covariance Forecast Evaluation
==========================================

This tutorial shows how to evaluate online covariance estimators with
:func:`~skfolio.model_selection.online_covariance_forecast_evaluation`.

We compare :class:`~skfolio.moments.EWCovariance`, a plain EWMA covariance, against
:class:`~skfolio.moments.RegimeAdjustedEWCovariance`, its regime-adjusted counterpart
based on the Short-Term Volatility Update (STVU) [1]_.

Both support incremental updates via `partial_fit`, making them suitable for streaming
evaluation. For estimators that do not support `partial_fit`, the batch counterpart
:func:`~skfolio.model_selection.covariance_forecast_evaluation` can be used instead.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2010-01-04 up to 2022-12-28.
import numpy as np
from plotly.io import show

from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import (
    CovarianceForecastComparison,
    online_covariance_forecast_evaluation,
)
from skfolio.moments import EWCovariance, RegimeAdjustedEWCovariance, RegimeAdjustmentMethod
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X = X["2010":]

# %%
# Covariance Estimators
# =====================
# We use two covariance estimators:
#
# * :class:`~skfolio.moments.EWCovariance`
# * :class:`~skfolio.moments.RegimeAdjustedEWCovariance`
#
# `EWCovariance` can react slowly to volatility shocks.
#
# `RegimeAdjustedEWCovariance` adds a regime adjustment via the Short-Term Volatility
# Update (STVU). This applies a scalar multiplier to better align predicted and
# realized risk when volatility regimes change faster than a plain EWMA can track.
#
# We set the same variance half-life of 40 trading days for both estimators and
# a correlation half-life of 80 trading days for `RegimeAdjustedEWCovariance`. Lower
# half-life for variance allows the model to adapt faster to volatility shifts, while
# higher half-life for correlation enables more stable estimation of co-movements, which
# typically require more data for reliable inference and reduces estimation noise. This
# choice also aligns with empirical evidence that volatility tends to mean-revert faster
# than correlation.
ew_cov = EWCovariance(half_life=40)

stvu_cov = RegimeAdjustedEWCovariance(
    half_life=40,
    corr_half_life=80,
    regime_half_life=20,
    regime_method=RegimeAdjustmentMethod.RMS,
)

# %%
# Evaluate Each Estimator
# =======================
# We now evaluate each estimator with
# :func:`~skfolio.model_selection.online_covariance_forecast_evaluation`.
# This function performs a walk-forward evaluation. At each step, it updates the
# estimator with `partial_fit` and compares the one-step-ahead forecast with the next
# realized return.
#
# Here, `warmup_size=252` reserves the first year for initialization, while
# `test_size=1` evaluates the forecast one day at a time.
ew_evaluation = online_covariance_forecast_evaluation(
    ew_cov,
    X,
    warmup_size=252,
    test_size=1,
)
stvu_evaluation = online_covariance_forecast_evaluation(
    stvu_cov,
    X,
    warmup_size=252,
    test_size=1,
)

# %%
# Summary Table
# =============
# Let's display the summary of the regime-adjusted covariance forecast
# evaluation. The four rows are:
#
# * **Mahalanobis ratio** evaluates whether the full covariance structure
#   (all eigenvalue directions) is correctly specified. The target is 1.0,
#   with values above 1.0 indicating underestimated risk and values below
#   1.0 indicating overestimated risk.
# * **Diagonal ratio** evaluates the individual asset variances only, with
#   the same 1.0 target and interpretation.
# * **Portfolio standardized returns** evaluate calibration along one
#   portfolio direction rather than across all directions. Their `std`
#   column is the bias statistic, with values near 1.0 meaning
#   well-calibrated portfolio risk.
# * **Portfolio QLIKE** evaluates portfolio variance forecasts along one
#   portfolio direction by comparing the forecast portfolio variance with
#   the realized sum of squared portfolio returns over the evaluation
#   window. Lower values indicate better variance forecasts.
stvu_evaluation.summary()

# %%
# Calibration Plot
# ================
# Let's now plot the rolling calibration diagnostics: the rolling mean of the
# Mahalanobis ratio, the rolling mean of the diagonal ratio, and the rolling
# bias statistic from the portfolio standardized returns.
stvu_evaluation.plot_calibration()

# %%
# Side-by-Side Comparison
# =======================
# We now compare both evaluations with
# :class:`~skfolio.model_selection.CovarianceForecastComparison`:
comparison = CovarianceForecastComparison(
    [ew_evaluation, stvu_evaluation], names=["EWMA Cov", "STVU Cov"]
)
comparison.summary()

# %%
# Bias Statistic
# ==============
# Let's plot the bias statistic. It measures whether the portfolio risk forecast is well
# calibrated, with a target of 1.0. We expect the regime-adjusted model to remain
# closer to 1.0, especially during stress periods.
comparison.plot_calibration(diagnostics=["bias"])

# %%
# QLIKE Loss
# ==========
# Let's now plot the QLIKE loss. It compares the forecast portfolio variance
# with the realized sum of squared portfolio returns over the evaluation
# window, so lower values indicate better portfolio variance forecasts.
# Because STVU rescales the forecast toward realized risk, we generally
# expect it to achieve a lower QLIKE.
comparison.plot_qlike_loss()

# %%
# Exceedance Rates
# ================
# We can also display the exceedance summary. If the covariance forecast were perfectly
# calibrated and returns were Gaussian, the squared Mahalanobis distance would follow a
# chi-squared distribution. The exceedance rate measures how often this distance exceeds
# the chi-squared threshold at a given significance level.
#
# In practice, daily equity returns are fat-tailed, so in this example both estimators
# exceed the nominal levels. This metric is therefore more useful for comparing
# estimators than for making an absolute calibration statement.
comparison.exceedance_summary()

# %%
# Multi-Portfolio Analysis
# ========================
# In the evaluations above, we use the default `portfolio_weights=None`, which computes
# dynamic inverse-volatility weights at each step as a single default portfolio direction
# so that high-volatility assets do not dominate the diagnostics. We can also provide
# explicit test portfolios to evaluate calibration along multiple portfolio directions
# instead of only this default one. Unlike the Mahalanobis diagnostic, which tests the
# full covariance structure across all directions, these portfolio diagnostics focus on
# selected traded directions.
#
# In practice, these portfolios should be representative of the allocations you trade.
# Here, we generate random Dirichlet draws for illustration:
n_assets = X.shape[1]
rng = np.random.default_rng(42)
portfolio_weights = rng.dirichlet(np.ones(n_assets), size=30)

# %%
ew_multi_portfolio = online_covariance_forecast_evaluation(
    ew_cov,
    X,
    warmup_size=252,
    test_size=1,
    portfolio_weights=portfolio_weights,
)
stvu_multi_portfolio = online_covariance_forecast_evaluation(
    stvu_cov,
    X,
    warmup_size=252,
    test_size=1,
    portfolio_weights=portfolio_weights,
)

# %%
# Bias Statistic Distribution
# ===========================
# Let's summarize the bias statistic across the 30 portfolio directions. A tight P5-P95
# spread indicates that calibration does not depend strongly on the selected portfolio
# direction.
multi_portfolio_comparison = CovarianceForecastComparison(
    [ew_multi_portfolio, stvu_multi_portfolio], names=["EWMA Cov", "STVU Cov"]
)
multi_portfolio_comparison.bias_statistic_summary()

# %%
# The calibration plot shows the median bias across portfolios together with the P5-P95
# bands:
fig = multi_portfolio_comparison.plot_calibration(diagnostics=["bias"])
show(fig)

# %%
# |
#
# The QLIKE plot also includes the P5-P95 bands from the 30 portfolios:
multi_portfolio_comparison.plot_qlike_loss()

# %%
# Conclusion
# ==========
# This tutorial showed how to:
#
# 1. Define online covariance estimators supporting `partial_fit`.
# 2. Evaluate them with :func:`~skfolio.model_selection.online_covariance_forecast_evaluation`.
# 3. Inspect calibration diagnostics and QLIKE.
# 4. Compare multiple estimators with :class:`~skfolio.model_selection.CovarianceForecastComparison`.
# 5. Extend the analysis to multiple portfolio directions.
#
# In the :ref:`next tutorial
# <sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py>`,
# we show how to tune covariance estimator hyperparameters with online search.
#
# .. [1] G. Paleologo, "The Elements of Quantitative Investing",
#     Wiley Finance (2025).
