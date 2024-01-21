"""
======================
Select Best Performers
======================

This tutorial introduces the :ref:`pre-selection transformers <pre_selection>`
:class:`~skfolio.pre_selection.SelectKExtremes` to select the `k` best or the `k` worst
assets according to a given measure before the optimization.

In this example, we will use a `Pipeline` to assemble the pre-selection step with a
minimum variance optimization. Then, we will use cross-validation to find the optimal
number of pre-selected assets to maximize the mean out-of-sample Sharpe Ratio.
"""

# %%
# Data
# ====
# We load the FTSE 100 :ref:`dataset <datasets>` composed of the daily prices of 64
# assets from the FTSE 100 Index starting from 2000-01-04 up to 2023-05-31:
import plotly.graph_objs as go
from plotly.io import show
from sklearn import set_config
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from skfolio import Population, RatioMeasure
from skfolio.datasets import load_ftse100_dataset
from skfolio.metrics import make_scorer
from skfolio.model_selection import (
    WalkForward,
    cross_val_predict,
)
from skfolio.moments import EmpiricalCovariance
from skfolio.optimization import MeanRisk
from skfolio.pre_selection import SelectKExtremes
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_ftse100_dataset()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# First, we create a Minimum Variance model without pre-selection:
benchmark = MeanRisk(
    prior_estimator=EmpiricalPrior(
        covariance_estimator=EmpiricalCovariance(nearest=True)
    ),
)
# %%
# .. note::
#   A covariance matrix is in theory positive semi-definite (PSD). However, due to
#   floating-point inaccuracies, we can end up with a covariance matrix that is just
#   slightly non-PSD. This often occurs in high dimensional problems. By setting the
#   `nearest` parameter from the covariance estimator to `True`, when the covariance
#   is not positive semi-definite (PSD), it is replaced by the nearest covariance that
#   is PSD without changing the variance.

# %%
# Pipeline
# ========
# Then, we create a Minimum Variance model with pre-selection using `Pipepline`:
set_config(transform_output="pandas")

model = Pipeline([("pre_selection", SelectKExtremes()), ("optimization", benchmark)])

# %%
# Parameter Tuning
# ================
# To demonstrate how parameter tuning works in a Pipeline model, we find the number of
# pre-selected assets `k` that maximizes the out-of-sample Sharpe Ratio using
# `GridSearchCV` with `WalkForward` cross-validation on the training set. The
# `WalkForward` is chosen to simulate a three months (60 business days) rolling
# portfolio fitted on the previous year (252 business days):
cv = WalkForward(train_size=252, test_size=60)

scorer = make_scorer(RatioMeasure.ANNUALIZED_SHARPE_RATIO)
# %%
# Note that we can also create a custom scorer this way:
# `scorer=make_scorer(lambda pred: pred.mean - 0.5 * pred.variance)`

grid_search = GridSearchCV(
    estimator=model,
    cv=cv,
    n_jobs=-1,
    param_grid={"pre_selection__k": list(range(5, 66, 3))},
    scoring=scorer,
    return_train_score=True,
)
grid_search.fit(X_train)
model = grid_search.best_estimator_
print(model)

# %%
# Let's plot the train and test scores as a function of the number of pre-selected
# assets. The vertical line represents the best test score and the selected model:
cv_results = grid_search.cv_results_
fig = go.Figure([
    go.Scatter(
        x=cv_results["param_pre_selection__k"],
        y=cv_results["mean_train_score"],
        name="Train",
        mode="lines",
        line=dict(color="rgb(31, 119, 180)"),
    ),
    go.Scatter(
        x=cv_results["param_pre_selection__k"],
        y=cv_results["mean_train_score"] + cv_results["std_train_score"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ),
    go.Scatter(
        x=cv_results["param_pre_selection__k"],
        y=cv_results["mean_train_score"] - cv_results["std_train_score"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        fillcolor="rgba(31, 119, 180,0.15)",
        fill="tonexty",
    ),
    go.Scatter(
        x=cv_results["param_pre_selection__k"],
        y=cv_results["mean_test_score"],
        name="Test",
        mode="lines",
        line=dict(color="rgb(255,165,0)"),
    ),
    go.Scatter(
        x=cv_results["param_pre_selection__k"],
        y=cv_results["mean_test_score"] + cv_results["std_test_score"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ),
    go.Scatter(
        x=cv_results["param_pre_selection__k"],
        y=cv_results["mean_test_score"] - cv_results["std_test_score"],
        line=dict(width=0),
        mode="lines",
        fillcolor="rgba(255,165,0, 0.15)",
        fill="tonexty",
        showlegend=False,
    ),
])
fig.add_vline(
    x=grid_search.best_params_["pre_selection__k"],
    line_width=2,
    line_dash="dash",
    line_color="green",
)
fig.update_layout(
    title="Train/Test score",
    xaxis_title="Number of pre-selected best performers",
    yaxis_title="Annualized Sharpe Ratio",
)
fig.update_yaxes(tickformat=".2f")
show(fig)

# %%
# |
#
# The mean test Sharpe Ratio increases from 1.17 (for k=5) to its maximum 1.91
# (for k=50) then decreases to 1.81 (for k=65).
# The selected model is a pre-selection of the top 50 performers based on their Sharpe
# Ratio, followed by a Minimum Variance optimization.

# %%
# Prediction
# ==========
# Now we evaluate the two models using the same `WalkForward` object on the test set:
pred_bench = cross_val_predict(
    benchmark,
    X_test,
    cv=cv,
    portfolio_params=dict(name="Benchmark"),
)

pred_model = cross_val_predict(
    model,
    X_test,
    cv=cv,
    n_jobs=-1,
    portfolio_params=dict(name="Pre-selection"),
)

# %%
# Each predicted object is a `MultiPeriodPortfolio`.
# For improved analysis, we can add them to a `Population`:
population = Population([pred_bench, pred_model])

# %%
# Let's plot the rolling portfolios cumulative returns on the test set:
population.plot_cumulative_returns()

# %%
# Let's plot the rolling portfolios compositions:
population.plot_composition(display_sub_ptf_name=False)

# %%
# Let's display the full summary:
population.summary()
