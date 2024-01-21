"""
===========
HRP vs HERC
===========

In this tutorial, we will compare the
:class:`~skfolio.optimization.HierarchicalRiskParity` (HRP) optimization with the
:class:`~skfolio.optimization.HierarchicalEqualRiskContribution` (HERC) optimization.

For that comparison, we consider a 3 months rolling (60 business days) allocation fitted
on the preceding year of data (252 business days) that minimizes the CVaR.

We will employ `GridSearchCV` to select the optimal parameters of each model on the
training set using cross-validation that achieves the highest average out-of-sample
Mean-CVaR ratio.

Then, we will evaluate the models on the test set and compare them with the
equal-weighted benchmark.

Finally, we will use the :class:`~skfolio.model_selection.CombinatorialPurgedCV` to
analyze the stability and distribution of both models.
"""

# %%
# Data
# ====
# We load the FTSE 100 :ref:`dataset <datasets>` composed of the daily prices of 64
# assets from the FTSE 100 Index composition starting from 2000-01-04 up to 2023-05-31:
from plotly.io import show
from sklearn.model_selection import GridSearchCV, train_test_split

from skfolio import Population, RatioMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_ftse100_dataset
from skfolio.distance import KendallDistance, PearsonDistance
from skfolio.metrics import make_scorer
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
)
from skfolio.preprocessing import prices_to_returns

prices = load_ftse100_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create two models: an HRP-CVaR and an HERC-CVaR:
model_hrp = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR,
    hierarchical_clustering_estimator=HierarchicalClustering(),
)

model_herc = HierarchicalEqualRiskContribution(
    risk_measure=RiskMeasure.CVAR,
    hierarchical_clustering_estimator=HierarchicalClustering(),
)

# %%
# Parameter Tuning
# ================
# For both HRP and HERC models, we find the parameters that maximizes the average
# out-of-sample Mean-CVaR ratio using `GridSearchCV` with `WalkForward` cross-validation
# on the training set. The `WalkForward` are chosen to simulate a three months
# (60 business days) rolling portfolio fitted on the previous year (252 business days):
cv = WalkForward(train_size=252, test_size=60)

grid_search_hrp = GridSearchCV(
    estimator=model_hrp,
    cv=cv,
    n_jobs=-1,
    param_grid={
        "distance_estimator": [PearsonDistance(), KendallDistance()],
        "hierarchical_clustering_estimator__linkage_method": [
            # LinkageMethod.SINGLE,
            LinkageMethod.WARD,
            LinkageMethod.COMPLETE,
        ],
    },
    scoring=make_scorer(RatioMeasure.CVAR_RATIO),
)
grid_search_hrp.fit(X_train)
model_hrp = grid_search_hrp.best_estimator_
print(model_hrp)

# %%
#
grid_search_herc = grid_search_hrp.set_params(estimator=model_herc)
grid_search_herc.fit(X_train)
model_herc = grid_search_herc.best_estimator_
print(model_herc)

# %%
# Prediction
# ==========
# We evaluate the two models using the same `WalkForward` object on the test set:
pred_hrp = cross_val_predict(
    model_hrp,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(name="HRP"),
)

pred_herc = cross_val_predict(
    model_herc,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(name="HERC"),
)
# %%
# Each predicted object is a `MultiPeriodPortfolio`.
# For improved analysis, we can add them to a `Population`:
population = Population([pred_hrp, pred_herc])

# %%
# Let's plot the rolling portfolios compositions:
population.plot_composition(display_sub_ptf_name=False)

# %%
# Let's plot the rolling portfolios cumulative returns on the test set:
population.plot_cumulative_returns()

# %%
# Analysis
# ========
# HERC outperform HRP both in terms of CVaR minimization and Mean-CVaR ratio
# maximization:
for ptf in population:
    print("=" * 25)
    print(" " * 8 + ptf.name)
    print("=" * 25)
    print(f"CVaR : {ptf.cvar:0.2%}")
    print(f"Mean-CVaR ratio : {ptf.cvar_ratio:0.4f}")
    print("\n")

# %%
# Combinatorial Purged Cross-Validation
# =====================================
# Only using one testing path (the historical path) may not be enough to compare
# models. For a more robust analysis, we can use the
# :class:`~skfolio.model_selection.CombinatorialPurgedCV` to create multiple testing
# paths from different training folds combinations:
cv = CombinatorialPurgedCV(n_folds=16, n_test_folds=14)

# %%
# We choose `n_folds` and `n_test_folds` to obtain more than 100 test paths and an average
# training size of approximately 252 days:
cv.summary(X_test)

# %%
pred_hrp = cross_val_predict(
    model_hrp,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(tag="HRP"),
)
pred_herc = cross_val_predict(
    model_herc,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(tag="HERC"),
)

# %%
# The predicted object is a `Population` of `MultiPeriodPortfolio`. Each
# `MultiPeriodPortfolio` represents one testing path of a rolling portfolio.
# For improved analysis, we can merge the populations of each model:
population = pred_hrp + pred_herc

# %%
# Distribution
# ============
# We plot the out-of-sample distribution of Mean-CVaR Ratio for each model:
population.plot_distribution(
    measure_list=[RatioMeasure.CVAR_RATIO], tag_list=["HRP", "HERC"], n_bins=50
)

# %%
for pred in [pred_hrp, pred_herc]:
    print("=" * 25)
    print(" " * 8 + pred[0].tag)
    print("=" * 25)
    print(
        "Average Mean-CVaR ratio :"
        f" {pred.measures_mean(measure=RatioMeasure.CVAR_RATIO):0.4f}"
    )
    print(
        "Std Mean-CVaR ratio :"
        f" {pred.measures_std(measure=RatioMeasure.CVAR_RATIO):0.4f}"
    )
    print("\n")

# %%
# We can see that, in terms of Mean-CVaR Ratio distribution, the HERC model has a higher
# mean than the HRP model but also a higher standard deviation. In other words, HERC is
# less stable than HRP but performs slightly better in average.

# %%
# We can do the same analysis for other measures:
fig = population.plot_distribution(
    measure_list=[
        RatioMeasure.ANNUALIZED_SHARPE_RATIO,
        RatioMeasure.ANNUALIZED_SORTINO_RATIO,
    ],
    tag_list=["HRP", "HERC"],
    n_bins=50,
)
show(fig)
