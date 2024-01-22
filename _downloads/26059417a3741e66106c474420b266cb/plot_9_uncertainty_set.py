r"""
===============
Uncertainty Set
===============

This tutorial shows how to incorporate expected returns uncertainty sets into the
:class:`~skfolio.optimization.MeanRisk` optimization.

By using the :ref:`Mu Uncertainty set estimator <uncertainty_set_estimator>`,
the assets expected returns are modelled with an ellipsoidal uncertainty set.
This approach, known as worst-case optimization, falls under the umbrella of robust
optimization. It reduces the instability that arises from the estimation errors of the
expected returns.

The worst case portfolio expect return is:

    .. math:: w^T\hat{\mu} - \kappa_{\mu}\lVert S_{\mu}^\frac{1}{2}w\rVert_{2}

with :math:`\kappa` the size of the ellipsoid (confidence region) and :math:`S` its
shape.

In this example, we will use a Mean-CVaR model with an
:class:`~skfolio.uncertainty_set.EmpiricalMuUncertaintySet` estimator.

Note that other uncertainty set can be used, for example:
:class:`~skfolio.uncertainty_set.BootstrapMuUncertaintySet`.
"""

# %%
# Data
# ====
# We load the FTSE 100 :ref:`dataset <datasets>` composed of the daily prices of 64
# assets from the FTSE 100 Index composition starting from 2000-01-04 up to 2023-05-31:
import numpy as np
import plotly.graph_objects as go
from plotly.io import show
from scipy.stats import uniform
from sklearn import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

from skfolio import PerfMeasure, Population, RatioMeasure, RiskMeasure
from skfolio.datasets import load_ftse100_dataset
from skfolio.metrics import make_scorer
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.uncertainty_set import EmpiricalMuUncertaintySet

prices = load_ftse100_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Efficient Frontier
# ==================
# First, we create a Mean-CVaR model to estimate the efficient frontier without
# uncertainty set. We constrain the CVaR at 95% to be below 2% (representing the
# average loss of the worst 5% daily returns over the period):
model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    min_weights=-1,
    max_cvar=0.02,
    efficient_frontier_size=20,
    portfolio_params=dict(name="Mean-CVaR", tag="No Uncertainty Set"),
)
model.fit(X_train)
model.weights_.shape

# %%
# Now, we create a robust (worst case) Mean-CVaR model with an uncertainty set on the
# expected returns:
model_uncertainty = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    min_weights=-1,
    max_cvar=0.02,
    efficient_frontier_size=20,
    mu_uncertainty_set_estimator=EmpiricalMuUncertaintySet(confidence_level=0.60),
    portfolio_params=dict(name="Mean-CVaR", tag="Mu Uncertainty Set - 60%"),
)
model_uncertainty.fit(X_train)
model_uncertainty.weights_.shape

# %%
# Let's plot both efficient frontiers on the training set:
population_train = model.predict(X_train) + model_uncertainty.predict(X_train)

population_train.plot_measures(
    x=RiskMeasure.CVAR,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
)

# %%
# Hyper-Parameter Tuning
# ======================
# In this section, we consider a 3 months rolling (60 business days) long-short
# allocation fitted on the preceding year of data (252 business days) that maximizes the
# portfolio return under a CVaR constraint.
# We will use `GridSearchCV` to select the below model parameters on the training set
# using walk forward analysis with a Mean/CVaR ratio scoring.
#
# The model parameters to tune are:
#
#   * `max_cvar`: CVaR target (upper constraint)
#   * `cvar_beta`: CVaR confidence level
#   * `confidence_level`: Mu uncertainty set confidence level of the :class:`~skfolio.uncertainty_set.EmpiricalMuUncertaintySet`
#
# For embedded parameters in the `GridSearchCV`, you need to use a double underscore:
# `mu_uncertainty_set_estimator__confidence_level`

model_no_uncertainty = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
    max_cvar=0.02,
    cvar_beta=0.9,
    min_weights=-1,
)

model_uncertainty = clone(model_no_uncertainty)
model_uncertainty.set_params(mu_uncertainty_set_estimator=EmpiricalMuUncertaintySet())

cv = WalkForward(train_size=252, test_size=60)

grid_search = GridSearchCV(
    estimator=model_uncertainty,
    cv=cv,
    n_jobs=-1,
    param_grid={
        "mu_uncertainty_set_estimator__confidence_level": [0.80, 0.90],
        "max_cvar": [0.03, 0.04, 0.05],
        "cvar_beta": [0.8, 0.9, 0.95],
    },
    scoring=make_scorer(RatioMeasure.CVAR_RATIO),
)
grid_search.fit(X_train)
best_model = grid_search.best_estimator_
print(best_model)

# %%
# The optimal parameters among the above 2x3x3 grid are the `max_cvar=3%`,
# `cvar_beta=90%` and :class:`~skfolio.uncertainty_set.EmpiricalMuUncertaintySet`
# `confidence_level=80%`. These parameters are the ones that achieved the highest mean
# out-of-sample Mean/CVaR ratio.
#
# For continuous parameters, such as `confidence_level`, a better approach is to use
# `RandomizedSearchCV` and specify a continuous distribution to take full advantage of
# the randomization. We specify a continuous random variable that is uniformly
# distributed between 0 and 1:

randomized_search = RandomizedSearchCV(
    estimator=model_uncertainty,
    cv=cv,
    n_jobs=-1,
    param_distributions={
        "mu_uncertainty_set_estimator__confidence_level": uniform(loc=0, scale=1),
    },
    n_iter=50,
    scoring=make_scorer(RatioMeasure.CVAR_RATIO),
)
randomized_search.fit(X_train)
best_model_rs = randomized_search.best_estimator_

# %%
# The selected confidence level is 58%.
#
# Let's plot the average out-of-sample score (CVaR ratio) as a function of the
# uncertainty set confidence level:
cv_results = randomized_search.cv_results_
x = np.asarray(
    cv_results["param_mu_uncertainty_set_estimator__confidence_level"]
).astype(float)
sort_idx = np.argsort(x)
y_test_mean = cv_results["mean_test_score"][sort_idx]
x = x[sort_idx]

fig = go.Figure([
    go.Scatter(
        x=x,
        y=y_test_mean,
        name="Test",
        mode="lines",
        line=dict(color="rgb(255,165,0)"),
    ),
])
fig.add_vline(
    x=randomized_search.best_params_["mu_uncertainty_set_estimator__confidence_level"],
    line_width=2,
    line_dash="dash",
    line_color="green",
)
fig.update_layout(
    title="Test score",
    xaxis_title="Uncertainty Set Confidence Level",
    yaxis_title="CVaR Ratio",
)
fig.update_yaxes(tickformat=".3f")
fig.update_xaxes(tickformat=".0%")
show(fig)

# %%
# |
#
# Now, we analyze all three models on the test set.
# By using `cross_val_predict` with `WalkForward`, we are able to compute efficiently
# the `MultiPeriodPortfolio` composed of 60 days rolling portfolios fitted on the
# preceding 252 days:
pred_no_uncertainty = cross_val_predict(model_no_uncertainty, X_test, cv=cv)
pred_no_uncertainty.name = "No Uncertainty set"

pred_uncertainty = cross_val_predict(best_model, X_test, cv=cv, n_jobs=-1)
pred_uncertainty.name = "Uncertainty set - Grid Search"

pred_uncertainty_rs = cross_val_predict(best_model_rs, X_test, cv=cv, n_jobs=-1)
pred_uncertainty_rs.name = "Uncertainty set - Randomized Search"

population = Population([pred_no_uncertainty, pred_uncertainty, pred_uncertainty_rs])
population.plot_cumulative_returns()

# %%
# From the plot and the below summary, we can see that the model without uncertainty set
# is overfitted and perform poorly on the test set. Its CVaR at 95% is 10% and its
# Mean/CVaR ratio is 0.006 which is the lowest of all models.
population.summary()

# %%
# Finally, let's plot the composition of the regularized multi-period portfolio:
pred_uncertainty.plot_composition()
