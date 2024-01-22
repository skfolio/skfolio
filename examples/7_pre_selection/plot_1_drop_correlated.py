"""
=============================
Drop Highly Correlated Assets
=============================

This tutorial introduces the  :ref:`pre-selection transformers <pre_selection>`
:class:`~skfolio.pre_selection.DropCorrelated` to remove highly correlated assets before
the optimization.

Highly correlated assets tend to increase the instability of mean-variance optimization.

In this example, we will compare a mean-variance optimization with and without
pre-selection.
"""

# %%
# Data
# ====
# We load the FTSE 100 :ref:`dataset <datasets>` composed of the daily prices of 64
# assets from the FTSE 100 Index composition starting from 2000-01-04 up to 2023-05-31:
from plotly.io import show
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from skfolio import Population, RatioMeasure
from skfolio.datasets import load_ftse100_dataset
from skfolio.model_selection import CombinatorialPurgedCV, cross_val_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.pre_selection import DropCorrelated
from skfolio.preprocessing import prices_to_returns

prices = load_ftse100_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# First, we create a maximum Sharpe Ratio model without pre-selection and fit it on the
# training set:
model1 = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO)
model1.fit(X_train)
model1.weights_

# %%
# Pipeline
# ========
# Then, we create a maximum Sharpe ratio model with pre-selection using `Pipepline` and
# fit it on the training set:
set_config(transform_output="pandas")

model2 = Pipeline([
    ("pre_selection", DropCorrelated(threshold=0.5)),
    ("optimization", MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO)),
])
model2.fit(X_train)
model2.named_steps["optimization"].weights_

# %%
# Prediction
# ==========
# We predict both models on the test set:
ptf1 = model1.predict(X_test)
ptf1.name = "model1"
ptf2 = model2.predict(X_test)
ptf2.name = "model2"

print(ptf1.n_assets)
print(ptf2.n_assets)

# %%
# Each predicted object is a `MultiPeriodPortfolio`.
# For improved analysis, we can add them to a `Population`:
population = Population([ptf1, ptf2])

# %%
# Let's plot the portfolios cumulative returns on the test set:
population.plot_cumulative_returns()

# %%
# Combinatorial Purged Cross-Validation
# =====================================
# Only using one testing path (the historical path) may not be enough for comparing both
# models. For a more robust analysis, we can use the
# :class:`~skfolio.model_selection.CombinatorialPurgedCV` to create multiple testing
# paths from different training folds combinations:
cv = CombinatorialPurgedCV(n_folds=10, n_test_folds=6)

# %%
# We choose `n_folds` and `n_test_folds` to obtain more than 100 test paths and an average
# training size of approximately 800 days:
cv.summary(X_test)

# %%
pred_1 = cross_val_predict(
    model1,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(annualized_factor=252, tag="model1"),
)

pred_2 = cross_val_predict(
    model2,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(annualized_factor=252, tag="model2"),
)

# %%
# The predicted object is a `Population` of `MultiPeriodPortfolio`. Each
# `MultiPeriodPortfolio` represents one testing path of a rolling portfolio.
# For improved analysis, we can merge the populations of each model:
population = pred_1 + pred_2

# %%
# Distribution
# ============
# We plot the out-of-sample distribution of Sharpe ratio for both models:
fig = population.plot_distribution(
    measure_list=[RatioMeasure.SHARPE_RATIO], tag_list=["model1", "model2"], n_bins=40
)
show(fig)

# %%
# |
#
# Model 1:
print(
    "Average of Sharpe Ratio:"
    f" {pred_1.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO):0.2f}"
)
print(
    "Std of Sharpe Ratio:"
    f" {pred_1.measures_std(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO):0.2f}"
)

# %%
# Model 2:
print(
    "Average of Sharpe Ratio:"
    f" {pred_2.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO):0.2f}"
)
print(
    "Std of Sharpe Ratio:"
    f" {pred_2.measures_std(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO):0.2f}"
)
