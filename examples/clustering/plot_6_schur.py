"""
==============================
Schur Complementary Allocation
==============================

This tutorial introduces the :class:`~skfolio.optimization.SchurComplementary`
allocation.

Schur Complementary Allocation is a portfolio allocation method developed by Peter
Cotton [1]_.

It uses Schur-complement-inspired augmentation of sub-covariance matrices,
revealing a link between Hierarchical Risk Parity (HRP) and minimum-variance
portfolios (MVP).

By tuning the regularization factor `gamma`, which governs how much off-diagonal
information is incorporated into the augmented covariance blocks, the method
smoothly interpolates from the heuristic divide-and-conquer allocation of HRP
(`gamma = 0`) to the MVP solution (`gamma -> 1`).

.. note ::
    A poorly conditioned covariance matrix can prevent convergence to the MVP solution
    as gamma approaches one. Setting `keep_monotonic=True` (the default) ensures that
    the portfolio variance decreases monotonically with respect to gamma and remains
    bounded by the variance of the HRP portfolio (`variance(Schur) <= variance(HRP)`),
    even in the presence of ill-conditioned covariance matrices. Additionally, you can
    apply shrinkage or other conditioning techniques via the `prior_estimator` parameter
    to improve numerical stability and estimation accuracy.
"""

# %%
# Data Loading
# ============
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2020-01-02 up to 2022-12-28:
import numpy as np
import scipy.stats as stats
from plotly.io import show
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from skfolio import PerfMeasure, Population, RatioMeasure, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.distance import KendallDistance, PearsonDistance
from skfolio.metrics import make_scorer
from skfolio.model_selection import MultipleRandomizedCV, WalkForward, cross_val_predict
from skfolio.moments import (
    LedoitWolf,
)
from skfolio.optimization import (
    HierarchicalRiskParity,
    MeanRisk,
    SchurComplementary,
)
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_sp500_dataset()
X = prices_to_returns(prices)

# %%
# We select **10 assets** from the 20 and split the data chronologically: 70% for
# training and 30% for testing.

# `shuffle=False` preserves chronological order, crucial for time-series data.
X_train, X_test = train_test_split(X.iloc[:, 10:], test_size=0.3, shuffle=False)

# %%
# Schur Complementary Model
# ==========================
# We start by fitting a simple :class:`~skfolio.optimization.SchurComplementary` model
# with `gamma=0.5`:

model = SchurComplementary(gamma=0.5)
model.fit(X_train)
print(model.weights_)

# %%
# Efficient Frontier Comparison
# =============================
# Let's take a closer look at how the Schur allocation behaves compared to other
# methods.
#
# To do that, we're going to fit a few different portfolio models on the
# **training set**:
#
# * Minimum-Variance (MVP)
# * Mean-Variance Efficient Frontier: 20 Markowitz portfolios spanning different risk levels (MVO)
# * Hierarchical Risk Parity (HRP)
# * 20 Schur portfolios with gamma values ranging from 0 to 1
#
# We apply a Ledoit-Wolf shrinkage estimator to regularize the covariance matrix for
# every model.
#
# Finally, we'll evaluate all these portfolios on the **test set** to see how well they
# generalize.
prior = EmpiricalPrior(covariance_estimator=LedoitWolf())

population_train = Population([])
population_test = Population([])

# 20 Schur portfolios
for gamma in np.linspace(0.0, 1.0, 20):
    schur = SchurComplementary(
        gamma=gamma,
        prior_estimator=prior,
        portfolio_params={"name": f"Schur {gamma:0.2f}", "tag": "Schur"},
    )
    # Train
    ptf = schur.fit_predict(X_train)
    population_train.append(ptf)
    # Test
    ptf = schur.predict(X_test)
    population_test.append(ptf)

# HRP portfolio
hrp = HierarchicalRiskParity(prior_estimator=prior, portfolio_params={"tag": "HRP"})
# Train
ptf = hrp.fit_predict(X_train)
population_train.append(ptf)
hrp_std = ptf.standard_deviation
# Test
ptf = hrp.predict(X_test)
population_test.append(ptf)

# 20 MVO (including MVP) portfolios
mean_variance = MeanRisk(
    prior_estimator=prior,
    efficient_frontier_size=20,
    max_standard_deviation=hrp_std,
    portfolio_params={"tag": "MVO"},
)
# Train
mv_population_train = mean_variance.fit_predict(X_train)
mv_population_train[0].tag = "MVP"
population_train += mv_population_train
# Test
mv_population_test = mean_variance.predict(X_test)
mv_population_test[0].tag = "MVP"
population_test += mv_population_test

# %%
# Plot Mean-Variance Frontiers on training set
fig = population_train.plot_measures(
    x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    y=PerfMeasure.ANNUALIZED_MEAN,
    hover_measures=[RatioMeasure.ANNUALIZED_SHARPE_RATIO],
    title="Training Set | MVO - HRP - Schur",
)
show(fig)

# %%
# |
#
# Plot Mean-Variance Frontiers on test set
population_test.plot_measures(
    x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    y=PerfMeasure.ANNUALIZED_MEAN,
    hover_measures=[RatioMeasure.ANNUALIZED_SHARPE_RATIO],
    title="Test Set | MVO - HRP - Schur",
)

# %%
# Plot portfolio compositions
population_train.filter(tags=["Schur", "MVP", "MVO"]).plot_composition()

# %%
# Analysis
# ========
# * When `gamma = 0`, the Schur portfolio is exactly equal to HRP.
# * As `gamma` increases toward 1, it gradually approaches the MVP solution,
#   **without fully reaching it**.
# * On the **training set**, both Schur and HRP portfolios are Pareto dominated
#   by the MVO portfolios, as expected, since those lie on the efficient frontier
#   by construction.
# * On the **test set**, Schur portfolios dominate the MVO portfolios.
#
# We observe a reversal in the relative frontiers (mean-variance dominance) between the
# training and test sets: MVO portfolios dominate in-sample, but their structure
# fails to hold out-of-sample versus Schur portfolios, which generalize more
# effectively.
#
# Below, we'll show how to model a more realistic train/test rebalancing strategy
# and how to find the optimal `gamma` parameter.

# %%
# Rebalancing Strategy
# ====================
# We use :class:`~skfolio.model_selection.WalkForward` to define a quarterly rebalancing
# (60 trading days), training on the prior three years (3*252 trading days):
walk_forward = WalkForward(test_size=60, train_size=252 * 3)

# %%
# Note that :class:`~skfolio.model_selection.WalkForward` also supports specific
# datetime frequencies. For example, we could use
# `walk_forward = WalkForward(test_size=3, train_size=36, freq="WOM-3FRI")` to
# rebalance quarterly on the **third Friday** (WOM-3FRI), training on the prior 36
# months.

# %%
# Hyperparameter Tuning
# =====================
# We'll tune the Schur model's `gamma` and distance metric using
# `RandomizedSearchCV`, optimizing for out-of-sample mean-CDaR Ratio:
model = SchurComplementary(prior_estimator=prior)

random_search = RandomizedSearchCV(
    estimator=model,
    cv=walk_forward,
    n_jobs=-1,
    param_distributions={
        "gamma": stats.uniform(0, 1),
        "distance_estimator": [PearsonDistance(), KendallDistance()],
    },
    n_iter=10,
    scoring=make_scorer(RatioMeasure.CDAR_RATIO),
    random_state=0,
)
random_search.fit(X_train)

# Retrieve the best estimator from the search.
schur = random_search.best_estimator_
schur


# %%
# In practice, it's recommended to increase `n_iter` to sample more parameter
# combinations, then plot those samples to ensure adequate search-space coverage and
# examine the convergence of training and test performance (see the
# :ref:`sphx_glr_auto_examples_mean_risk_plot_8_regularization.py` tutorial).
#
# Standard Walk-Forward Analysis
# ==============================
# We evaluate the MVP and tuned Schur models on the test set using standard
# walk-forward analysis:

mvo = MeanRisk(prior_estimator=prior)
pred_mvo = cross_val_predict(mvo, X_test, cv=walk_forward, n_jobs=-1)
pred_mvo.name = "MVP"

pred_schur = cross_val_predict(schur, X_test, cv=walk_forward, n_jobs=-1)
pred_schur.name = "Schur"

# Combine results for easier analysis.
population = Population([pred_schur, pred_mvo])
population.plot_cumulative_returns()

# %%
# Let's display a summary of key performance metrics:
summary = population.summary()
print(summary.loc[["Annualized Sharpe Ratio", "CDaR Ratio at 95%"]])

# %%
# A single backtest path represents one possible trajectory of cumulative
# returns under the given rebalancing scheme and parameter set. While easy
# to compute, it may understate the variability and uncertainty of real-world
# performance compared to resampling-based methods.


# %%
# Multiple Randomized Cross-Validation
# ====================================
# Using the :class:`~skfolio.model_selection.MultipleRandomizedCV` methodology of
# Palomar in [2]_, we perform resampling-based cross-validation by drawing 800
# subsamples of 10 distinct assets from the 20-asset universe and contiguous 5-year
# windows (5 x 252 trading days). We then apply our walk-forward split to each
# subsample. This approach captures both temporal and cross-sectional variability:
X_train, X_test = train_test_split(X, test_size=0.3, shuffle=False)

cv_mc = MultipleRandomizedCV(
    walk_forward=walk_forward,
    n_subsamples=800,
    asset_subset_size=10,
    window_size=5 * 252,
    random_state=0,
)

# Generate cross-validated predictions for both models.
pred_mvo_mc = cross_val_predict(
    mvo, X_test, cv=cv_mc, n_jobs=-1, portfolio_params={"tag": "MVP"}
)

pred_schur_mc = cross_val_predict(
    schur, X_test, cv=cv_mc, n_jobs=-1, portfolio_params={"tag": "Schur"}
)

# Combine results for easier analysis.
population_mc = pred_mvo_mc + pred_schur_mc

# %%
# Let's plot the distribution of out-of-sample performance metrics (e.g., Sharpe ratio,
# CDaR ratio) across all resampled subsamples for both the Schur and MVP portfolios.
# This helps assess how robust each model is across different asset combinations and
# time periods:
population_mc.plot_distribution(
    measure_list=[RatioMeasure.ANNUALIZED_SHARPE_RATIO], tag_list=["MVP", "Schur"]
)

# %%
population_mc.plot_distribution(
    measure_list=[RatioMeasure.CDAR_RATIO], tag_list=["MVP", "Schur"]
)


# %%
for pred in [pred_mvo_mc, pred_schur_mc]:
    tag = pred[0].tag
    mean_sr = pred.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO)
    std_sr = pred.measures_std(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO)
    print(f"{tag}\n{'=' * len(tag)}")
    print(f"Average Sharpe Ratio: {mean_sr:0.2f}")
    print(f"Sharpe Ratio Std Dev: {std_sr:0.2f}\n")

# %%
# In this simple example, Schur portfolios tend to outperform MVP out-of-sample,
# exhibiting higher average Sharpe and CDaR ratios.
#
# For a full tutorial on :class:`~skfolio.model_selection.MultipleRandomizedCV`,
# see :ref:`sphx_glr_auto_examples_mean_risk_plot_8_regularization.py`.
# For additional cross-validation methods, such as
# :class:`~skfolio.model_selection.CombinatorialPurgedCV` from de Prado [3]_,
# refer to :ref:`the model selection section <model_selection_examples>`.

# %%
# Conclusion
# ==========
# This short example introduced the Schur Complementary Allocation method
# and demonstrated how to use the `skfolio` API to train, evaluate, tune, and compare
# Schur portfolios with other allocation strategies.

# %%
# References
# ==========
# .. [1] "Schur Complementary Allocation: A Unification of Hierarchical Risk Parity
#         and Minimum Variance Portfolios". Peter Cotton (2024).
#
# .. [2] "Portfolio Optimization: Theory and Application", Chapter 8,
#         Daniel P. Palomar (2025)
#
# .. [3] "Advances in Financial Machine Learning",
#        Marcos López de Prado (2018)
