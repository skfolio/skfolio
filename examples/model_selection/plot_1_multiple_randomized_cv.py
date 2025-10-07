"""
====================================
Multiple Randomized Cross-Validation
====================================

This tutorial introduces :class:`~skfolio.model_selection.MultipleRandomizedCV`,
which is based on the "Multiple Randomized Backtests" methodology of Palomar in [1]_.
This cross-validation strategy performs a resampling-based evaluation by repeatedly
sampling **distinct** asset subsets (without replacement) and **contiguous** time
windows, then applying an inner walk-forward split to each subsample, capturing both
temporal and cross-sectional variability in performance.

In this example, we build a portfolio model composed of a preselection of top
performers, followed by a Hierarchical Equal Risk Contribution optimization with
covariance shrinkage. We split the dataset into training and test sets, tune
hyperparameters on the training set, and then evaluate the final portfolio models on
the test set using :class:`~skfolio.model_selection.MultipleRandomizedCV`.
"""

# %%
# Data Loading
# ============
# We load the FTSE 100 :ref:`dataset <datasets>`, which contains daily prices
# of 64 assets from the FTSE 100 index, spanning 2000-01-04 to 2023-05-31.
import scipy.stats as stats
from plotly.io import show
from sklearn import set_config
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from skfolio import Population, RatioMeasure, RiskMeasure
from skfolio.datasets import load_ftse100_dataset
from skfolio.metrics import make_scorer
from skfolio.model_selection import MultipleRandomizedCV, WalkForward, cross_val_predict
from skfolio.moments import ShrunkCovariance
from skfolio.optimization import HierarchicalEqualRiskContribution
from skfolio.pre_selection import SelectKExtremes
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

set_config(transform_output="pandas")

prices = load_ftse100_dataset()
returns = prices_to_returns(prices)

# Sequential train-test split: 67% training, 33% testing.
# `shuffle=False` preserves chronological order, crucial for time-series data.
X_train, X_test = train_test_split(returns, test_size=0.33, shuffle=False)


# %%
# Portfolio Construction
# ======================
# We build a pipeline that first selects the top-k assets by Sharpe ratio, then
# allocates weights via Hierarchical Equal Risk Contribution using a shrunk
# covariance estimator.
pre_selection = SelectKExtremes(k=10, measure=RatioMeasure.SHARPE_RATIO, highest=True)

optimization = HierarchicalEqualRiskContribution(
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ShrunkCovariance(shrinkage=0.5)
    ),
    risk_measure=RiskMeasure.VARIANCE,
)

model_bench = Pipeline(
    [
        ("pre_selection", pre_selection),
        ("optimization", optimization),
    ]
)


# %%
# Rebalancing Strategy
# ====================
# We use :class:`~skfolio.model_selection.WalkForward` to define a monthly rebalancing
# (20 trading days), training on the prior year (252 trading days):
walk_forward = WalkForward(test_size=20, train_size=252)

# %%
# Note that :class:`~skfolio.model_selection.WalkForward` also supports specific
# datetime frequencies. For examples, we could use
# `walk_forward = WalkForward(test_size=1, train_size=12, freq="WOM-3FRI")` to
# rebalance **monthly** on the **third Friday** (WOM-3FRI), training on the prior 12
# months.

# %%
# Hyperparameter Tuning
# =====================
# Initially, the number of selected assets and the shrinkage parameter were
# chosen randomly. We use `RandomizedSearchCV` to explore these parameters and
# find the combination that maximizes the mean out-of-sample CVaR ratio.
random_search = RandomizedSearchCV(
    estimator=model_bench,
    cv=walk_forward,
    n_jobs=-1,
    param_distributions={
        "pre_selection__k": stats.randint(10, 30),
        "optimization__prior_estimator__covariance_estimator__shrinkage": stats.uniform(
            0, 1
        ),
    },
    n_iter=30,
    random_state=0,
    scoring=make_scorer(RatioMeasure.CVAR_RATIO),
)
random_search.fit(X_train)

# Retrieve the best estimator from the search.
model_tuned = random_search.best_estimator_
model_tuned

# %%
# In practice, it's recommended to increase `n_iter` to sample more parameter
# combinations, then plot those samples to ensure adequate search-space coverage and
# examine the convergence of training and test performance. (see the
# :ref:`sphx_glr_auto_examples_mean_risk_plot_8_regularization.py` tutorial).
#
# Standard Walk-Forward Analysis
# ==============================
# We evaluate both the benchmark and tuned models on the test set using standard
# walk-forward analysis, which yields a single backtest path per model.
# A single backtest path represents one possible trajectory of cumulative
# returns under the given rebalancing scheme and parameter set. While easy
# to compute, it may understate the variability and uncertainty of real-world
# performance compared to resampling-based methods.
pred_bench = cross_val_predict(model_bench, X_test, cv=walk_forward)
pred_bench.name = "Benchmark Model"

pred_tuned = cross_val_predict(model_tuned, X_test, cv=walk_forward, n_jobs=-1)
pred_tuned.name = "Tuned Model"

# Combine results for easier analysis.
population = Population([pred_bench, pred_tuned])
population.plot_cumulative_returns()

# %%
# Display a summary of key performance metrics.
population.summary()


# %%
# Multiple Randomized Cross-Validation
# ====================================
# We perform resampling-based cross-validation by drawing 500 subsamples of 50 distinct
# assets and contiguous 3-year windows (3 x 252 trading days), then applying our
# walk-forward split to each subsample. This approach captures both temporal and
# cross-sectional variability.
cv_mc = MultipleRandomizedCV(
    walk_forward=walk_forward,
    n_subsamples=500,
    asset_subset_size=50,
    window_size=3 * 252,
    random_state=0,
)

# Generate cross-validated predictions for both models.
pred_bench_mc = cross_val_predict(
    model_bench,
    X_test,
    cv=cv_mc,
    n_jobs=-1,
    portfolio_params={"tag": "Benchmark Model"},
)

pred_tuned_mc = cross_val_predict(
    model_tuned, X_test, cv=cv_mc, n_jobs=-1, portfolio_params={"tag": "Tuned Model"}
)

# Combine results for easier analysis.
population_mc = pred_bench_mc + pred_tuned_mc

# %%
# Visualization and Analysis
# --------------------------
# We plot cumulative returns for the first 10 `MultiPeriodPortfolio` (resampled paths)
# of the tuned model. Each `MultiPeriodPortfolio` concatenates the test (out-of-sample)
# results from the walk-forward.
fig = pred_tuned_mc[:10].plot_cumulative_returns(use_tag_in_legend=False)
show(fig)

# %%
# |
#
# We now compute and display the distribution of out-of-sample annualized Sharpe ratios:
population_mc.plot_distribution(
    measure_list=[RatioMeasure.ANNUALIZED_SHARPE_RATIO],
    tag_list=["Benchmark Model", "Tuned Model"],
)

# %%
for pred in [pred_bench_mc, pred_tuned_mc]:
    tag = pred[0].tag
    mean_sr = pred.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO)
    std_sr = pred.measures_std(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO)
    print(f"{tag}\n{'=' * len(tag)}")
    print(f"Average Sharpe Ratio: {mean_sr:0.2f}")
    print(f"Sharpe Ratio Std Dev: {std_sr:0.2f}\n")

# %%
# Let's display the Box plot of the CVaR Ratio:
population_mc.boxplot_measure(
    measure=RatioMeasure.CVAR_RATIO, tag_list=["Benchmark Model", "Tuned Model"]
)

# %%
# We plot the asset composition for the first two `MultiPeriodPortfolio`:
pred_tuned_mc[:2].plot_composition(display_sub_ptf_name=False)

# %%
# We plot the weights evolution over time for the first `MultiPeriodPortfolio`:
pred_tuned_mc[0].plot_weights_per_observation()

# %%
# Conclusion
# ==========
# A single-path walk-forward analysis may understate the variability and uncertainty of
# real-world performance. Multiple Randomized Cross-Validation, by contrast, applies
# a resampling-based evaluation across asset subsets and time windows, yielding
# performance estimates that are more robust and less prone to overfitting.

# %%
# References
# ==========
# .. [1] "Portfolio Optimization: Theory and Application", Chapter 8
#         Daniel P. Palomar (2025)
#
