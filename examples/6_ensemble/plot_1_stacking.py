"""
=====================
Stacking Optimization
=====================

This tutorial introduces the :class:`~skfolio.optimization.StackingOptimization`.

Stacking Optimization is an ensemble method that consists in stacking the output of
individual portfolio optimizations with a final portfolio optimization.

The weights are the dot-product of individual optimizations weights with the final
optimization weights.

Stacking allows to use the strength of each individual portfolio optimization by using
their output as input of a final portfolio optimization.

To avoid data leakage, out-of-sample estimates are used to fit the outer optimization.

.. note ::
    The `estimators_` are fitted on the full `X` while `final_estimator_` is trained
    using cross-validated predictions of the base estimators using `cross_val_predict`.
"""

# %%
# Data
# ====
# We load the FTSE 100 dataset. This dataset is composed of the daily prices of 64
# assets from the FTSE 100 Index composition starting from 2000-01-04 up to 2023-05-31:
from plotly.io import show
from sklearn.model_selection import GridSearchCV, train_test_split

from skfolio import Population, RatioMeasure, RiskMeasure
from skfolio.datasets import load_ftse100_dataset
from skfolio.metrics import make_scorer
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.moments import EmpiricalCovariance, LedoitWolf
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalEqualRiskContribution,
    InverseVolatility,
    MaximumDiversification,
    MeanRisk,
    ObjectiveFunction,
    StackingOptimization,
)
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_ftse100_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.50, shuffle=False)

# %%
# Stacking Model
# ==============
# Our stacking model will be composed of 4 models:
#   * Inverse Volatility
#   * Maximum Diversification
#   * Maximum Mean-Risk Utility allowing short position with L1 regularization
#   * Hierarchical Equal Risk Contribution
#
# We will stack these 4 models together using the Mean-CDaR utility maximization:

estimators = [
    ("model1", InverseVolatility()),
    ("model2", MaximumDiversification(prior_estimator=EmpiricalPrior())),
    (
        "model3",
        MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY, min_weights=-1),
    ),
    ("model4", HierarchicalEqualRiskContribution()),
]

model_stacking = StackingOptimization(
    estimators=estimators,
    final_estimator=MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=RiskMeasure.CDAR,
    ),
)

# %%
# Benchmark
# =========
# To compare the staking model, we use an equal-weighted benchmark:
benchmark = EqualWeighted()

# %%
# Parameter Tuning
# ================
# To demonstrate how parameter tuning works in a staking model, we find the model
# parameters that maximizes the out-of-sample Calmar Ratio using `GridSearchCV` with
# `WalkForward` cross-validation on the training set.
# The `WalkForward` are chosen to simulate a three months (60 business days) rolling
# portfolio fitted on the previous year (252 business days):
cv = WalkForward(train_size=252, test_size=60)

grid_search = GridSearchCV(
    estimator=model_stacking,
    cv=cv,
    n_jobs=-1,
    param_grid={
        "model2__prior_estimator__covariance_estimator": [
            EmpiricalCovariance(),
            LedoitWolf(),
        ],
        "model3__l1_coef": [0.001, 0.1],
        "model4__risk_measure": [
            RiskMeasure.VARIANCE,
            RiskMeasure.GINI_MEAN_DIFFERENCE,
        ],
    },
    scoring=make_scorer(RatioMeasure.CALMAR_RATIO),
)
grid_search.fit(X_train)
model_stacking = grid_search.best_estimator_
print(model_stacking)

# %%
# Prediction
# ==========
# We evaluate the Stacking model and the Benchmark using the same `WalkForward` object
# on the test set:
pred_bench = cross_val_predict(
    benchmark,
    X_test,
    cv=cv,
    portfolio_params=dict(name="Benchmark"),
)

pred_stacking = cross_val_predict(
    model_stacking,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(name="Stacking"),
)

# %%
# Each predicted object is a `MultiPeriodPortfolio`.
# For improved analysis, we can add them to a `Population`:
population = Population([pred_bench, pred_stacking])

# %%
# Let's plot the rolling portfolios cumulative returns on the test set:
population.plot_cumulative_returns()

# %%
# Let's plot the rolling portfolios compositions:
population.plot_composition(display_sub_ptf_name=False)

# %%
# Analysis
# ========
# The Stacking model outperforms the Benchmark on the test set for the below ratios:
for ptf in population:
    print("=" * 25)
    print(" " * 8 + ptf.name)
    print("=" * 25)
    print(f"Sharpe ratio : {ptf.annualized_sharpe_ratio:0.2f}")
    print(f"CVaR ratio : {ptf.cdar_ratio:0.5f}")
    print(f"Calmar ratio : {ptf.calmar_ratio:0.5f}")
    print("\n")

# %%
# Let's display the full summary:
population.summary()

# %%
# Combinatorial Purged Cross-Validation
# =====================================
# Only using one testing path (the historical path) may not be enough for comparing both
# models. For a more robust analysis, we can use the
# :class:`~skfolio.model_selection.CombinatorialPurgedCV` to create multiple testing
# paths from different training folds combinations:
cv = CombinatorialPurgedCV(n_folds=20, n_test_folds=18)

# %%
# We choose `n_folds` and `n_test_folds` to obtain more than 100 test paths and an
# average training size of approximately 252 days:
cv.summary(X_test)

# %%
pred_stacking = cross_val_predict(
    model_stacking,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(tag="Stacking"),
)

# %%
# The predicted object is a `Population` of `MultiPeriodPortfolio`. Each
# `MultiPeriodPortfolio` represents one test path of a rolling portfolio.

# %%
# Distribution
# ============
# Let's plot the out-of-sample distribution of Sharpe Ratio for the Stacking model:
pred_stacking.plot_distribution(
    measure_list=[RatioMeasure.ANNUALIZED_SHARPE_RATIO], n_bins=40
)

# %%
print(
    "Average of Sharpe Ratio :"
    f" {pred_stacking.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO):0.2f}"
)
print(
    "Std of Sharpe Ratio :"
    f" {pred_stacking.measures_std(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO):0.2f}"
)

# %%
# Now, let's analyze how the sub-models would have performed independently and compare
# their distribution with the Stacking model:
population = Population([])
for model_name, model in model_stacking.estimators:
    pred = cross_val_predict(
        model,
        X_test,
        cv=cv,
        n_jobs=-1,
        portfolio_params=dict(tag=model_name),
    )
    population.extend(pred)
population.extend(pred_stacking)

fig = population.plot_distribution(
    measure_list=[RatioMeasure.ANNUALIZED_SHARPE_RATIO],
    n_bins=40,
    tag_list=["Stacking", "model1", "model2", "model3", "model4"],
)
show(fig)

# %%
# Conclusion
# ==========
# The Stacking model outperforms the Benchmark on the historical test set. The
# distribution analysis on the recombined (non-historical) test sets shows that the
# Stacking model continues to outperform the Benchmark in average.
