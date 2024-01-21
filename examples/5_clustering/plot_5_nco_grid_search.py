"""
=============================
NCO - Combinatorial Purged CV
=============================

The previous tutorial introduced the
:class:`~skfolio.optimization.NestedClustersOptimization`.

In this tutorial, we will perform hyperparameter search using `GridSearch` and
distribution analysis with `CombinatorialPurgedCV`.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2015-01-02 up to 2022-12-28:
from plotly.io import show
from sklearn.model_selection import GridSearchCV, train_test_split

from skfolio import Population, RatioMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_sp500_dataset
from skfolio.distance import KendallDistance, PearsonDistance
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.optimization import (
    EqualWeighted,
    MeanRisk,
    NestedClustersOptimization,
    RiskBudgeting,
)
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices["2015":]

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.5, shuffle=False)

# %%
# Model
# =====
# We create two models: the NCO and the equal-weighted benchmark:
benchmark = EqualWeighted()

model_nco = NestedClustersOptimization(
    inner_estimator=MeanRisk(), clustering_estimator=HierarchicalClustering()
)

# %%
# Parameter Tuning
# ================
# We find the model parameters that maximizes the out-of-sample Sharpe ratio using
# `GridSearchCV` with `WalkForward` cross-validation on the training set.
# The `WalkForward` are chosen to simulate a three months (60 business days) rolling
# portfolio fitted on the previous year (252 business days):
cv = WalkForward(train_size=252, test_size=60)

grid_search_hrp = GridSearchCV(
    estimator=model_nco,
    cv=cv,
    n_jobs=-1,
    param_grid={
        "inner_estimator__risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CVAR],
        "outer_estimator": [
            EqualWeighted(),
            RiskBudgeting(risk_measure=RiskMeasure.CVAR),
        ],
        "clustering_estimator__linkage_method": [
            LinkageMethod.SINGLE,
            LinkageMethod.WARD,
        ],
        "distance_estimator": [PearsonDistance(), KendallDistance()],
    },
)
grid_search_hrp.fit(X_train)
model_nco = grid_search_hrp.best_estimator_
print(model_nco)

# %%
# Prediction
# ==========
# We evaluate the two models using the same `WalkForward` object on the test set:
pred_bench = cross_val_predict(
    benchmark,
    X_test,
    cv=cv,
    portfolio_params=dict(name="Benchmark"),
)

pred_nco = cross_val_predict(
    model_nco,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(name="NCO"),
)
# %%
# Each predicted object is a `MultiPeriodPortfolio`.
# For improved analysis, we can add them to a `Population`:
population = Population([pred_bench, pred_nco])

# %%
# Let's plot the rolling portfolios compositions:
population.plot_composition(display_sub_ptf_name=False)

# %%
# Let's plot the rolling portfolios cumulative returns on the test set:
fig = population.plot_cumulative_returns()
show(fig)

# %%
# Analysis
# ========
# The NCO outperforms the Benchmark on the test set for the below measures:
# maximization:
for ptf in population:
    print("=" * 25)
    print(" " * 8 + ptf.name)
    print("=" * 25)
    print(f"Ann. Sharpe ratio : {ptf.annualized_sharpe_ratio:0.2f}")
    print(f"CVaR ratio : {ptf.cvar_ratio:0.4f}")
    print("\n")

# %%
# Combinatorial Purged Cross-Validation
# =====================================
# Only using one testing path (the historical path) may not be enough for comparing both
# models. For a more robust analysis, we can use
# :class:`~skfolio.model_selection.CombinatorialPurgedCV` to create multiple testing
# paths from different training folds combinations.
cv = CombinatorialPurgedCV(n_folds=9, n_test_folds=7)

# %%
# We choose `n_folds` and `n_test_folds` to obtain more than 30 test paths and an average
# training size of approximately 252 days:
cv.summary(X_test)

# %%
pred_nco = cross_val_predict(
    model_nco,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(tag="NCO"),
)

# %%
# The predicted object is a `Population` of `MultiPeriodPortfolio`. Each
# `MultiPeriodPortfolio` represents one testing path of a rolling portfolio.

# %%
# Distribution
# ============
# We plot the out-of-sample distribution of Sharpe Ratio for the NCO model:
pred_nco.plot_distribution(measure_list=[RatioMeasure.ANNUALIZED_SHARPE_RATIO])

# %%
# Let's print the average and standard-deviation of out-of-sample Sharpe Ratios:
print(
    "Average of Sharpe Ratio :"
    f" {pred_nco.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO):0.2f}"
)
print(
    "Std of Sharpe Ratio :"
    f" {pred_nco.measures_std(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO):0.2f}"
)

# %%
# Let's compare it with the benchmark:
pred_bench = benchmark.fit_predict(X_test)
print(pred_bench.annualized_sharpe_ratio)

# %%
# Conclusion
# ==========
# This NCO model outperforms the Benchmark in terms of Sharpe Ratio on the historical
# test set. However, the distribution analysis on the recombined (non-historical) test
# sets shows that it slightly underperforms the Benchmark in average.
#
# This was a toy example to present the API. Further analysis using different
# estimators, datasets and CV parameters should be performed to determine if the
# outperformance on the historical test set is due to chance or if this NCO model is
# able to exploit time-dependencies information lost in `CombinatorialPurgedCV`.
