r"""
========================
L1 and L2 Regularization
========================

This tutorial shows how to incorporate regularization into the
:class:`~skfolio.optimization.MeanRisk` optimization.

Regularization tends to increase robustness and out-of-sample stability.

The `l1_coef` parameter is used to penalize the objective function by the L1 norm:

.. math:: l1\_coef \times \Vert w \Vert_{1} = l1\_coef \times \sum_{i=1}^{N} |w_{i}|

and the `l2_coef` parameter is used to penalize the objective function by the L2 norm:

.. math:: l2\_coef \times \Vert w \Vert_{2}^{2} = l2\_coef \times \sum_{i=1}^{N} w_{i}^2

.. warning ::

    Increasing the L1 coefficient may reduce the number of non-zero weights
    (cardinality), which can reduce diversification. However, a reduction in
    diversification does not necessarily equate to a reduction in robustness.

.. note ::

    Increasing the L1 coefficient has no impact if the portfolio is long only.

In this example we will use a dataset with a large number of assets and long-short
allocation to exacerbate overfitting.

First, we will analyze the impact of regularization on the entire Mean-Variance efficient
frontier and its stability from the training set to the test set. Then, we will show how
to tune the regularization coefficients using cross-validation with `GridSearchCV`.
"""

# %%
# Data
# ====
# We load the FTSE 100 :ref:`dataset <datasets>` composed of the daily prices of 64
# assets from the FTSE 100 Index composition starting from 2000-01-04 up to 2023-05-31.
import numpy as np
import plotly.graph_objects as go
from plotly.io import show
from scipy.stats import loguniform
from sklearn import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split

from skfolio import PerfMeasure, Population, RatioMeasure, RiskMeasure
from skfolio.datasets import load_ftse100_dataset
from skfolio.metrics import make_scorer
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import EqualWeighted, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

prices = load_ftse100_dataset()
X = prices_to_returns(prices)

X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Efficient Frontier
# ==================
# First, we create a Mean-Variance model to estimate the efficient frontier without
# regularization. We constrain the volatility to be below 30% p.a.
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    min_weights=-1,
    max_variance=0.3**2 / 252,
    efficient_frontier_size=30,
    portfolio_params=dict(name="Mean-Variance", tag="No Regularization"),
)
model.fit(X_train)
model.weights_.shape

# %%
# Now we create the two regularized models:
model_l1 = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    min_weights=-1,
    max_variance=0.3**2 / 252,
    efficient_frontier_size=30,
    l1_coef=0.001,
    portfolio_params=dict(name="Mean-Variance", tag="L1 Regularization"),
)
model_l1.fit(X_train)

model_l2 = clone(model_l1)
model_l2.set_params(
    l1_coef=0,
    l2_coef=0.001,
    portfolio_params=dict(name="Mean-Variance", tag="L2 Regularization"),
)
model_l2.fit(X_train)
model_l2.weights_.shape

# %%
# Let's plot the efficient frontiers on the training set:
population_train = (
    model.predict(X_train) + model_l1.predict(X_train) + model_l2.predict(X_train)
)

population_train.plot_measures(
    x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
)

# %%
# Prediction
# ==========
# The parameter `efficient_frontier_size=30` means that when we called the `fit` method,
# each model ran 30 optimizations along the efficient frontier. Therefore, the `predict`
# method will return a :class:`~skfolio.population.Population` composed of 30
# :class:`~skfolio.portfolio.Portfolio`:
population_test = (
    model.predict(X_test) + model_l1.predict(X_test) + model_l2.predict(X_test)
)

for tag in ["No Regularization", "L1 Regularization"]:
    print("=================")
    print(tag)
    print("=================")
    print(
        "Avg Sharpe Ratio Train:"
        f" {population_train.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO, tags=tag):0.2f}"
    )
    print(
        "Avg Sharpe Ratio Test:"
        f" {population_test.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO, tags=tag):0.2f}"
    )
    print(
        "Avg non-zeros assets:"
        f" {np.mean([len(ptf.nonzero_assets) for ptf in population_train.filter(tags=tag)]):0.2f}"
    )
    print("\n")

population_test.plot_measures(
    x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
)

# %%
# In this example we can clearly see that L1 regularization reduced the number of assets
# (from 64 down to 14) and made the model more robust: the portfolios without
# regularization have a higher Sharpe on the train set and a lower Sharpe on the test
# set compared to the portfolios with regularization.

# %%
# Hyper-parameter Tuning
# ======================
# In this section, we consider a 3 months rolling (60 business days) long-short
# allocation fitted on the preceding year of data (252 business days) that maximizes the
# return under a volatility constraint of 30% p.a.
#
# We use `GridSearchCV` to select the optimal L1 and L2 regularization coefficients on
# the training set using cross-validation that achieve the highest
# mean test score. We use the default score, which is the Sharpe ratio.
# Finally, we evaluate the model on the test set and compare it with the equal-weighted
# benchmark and a reference model without regularization:

ref_model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
    max_variance=0.3**2 / 252,
    min_weights=-1,
)

cv = WalkForward(train_size=252, test_size=60)

grid_search = GridSearchCV(
    estimator=ref_model,
    cv=cv,
    n_jobs=-1,
    param_grid={
        "l1_coef": [0.001, 0.01, 0.1],
        "l2_coef": [0.001, 0.01, 0.1],
    },
)
grid_search.fit(X_train)
best_model = grid_search.best_estimator_
print(best_model)

# %%
# The optimal parameters among the above 3x3 grid are 0.01 for the L1 coefficient
# and the L2 coefficient.
# These parameters are the ones that achieved the highest mean out-of-sample Sharpe
# Ratio. Note that the score can be changed to another measure or function using the
# `scoring` parameter.
#
# For continuous parameters, such as L1 and L2 above, a better approach is to use
# `RandomizedSearchCV` and specify a continuous distribution to take full advantage of
# the randomization.
#
# A continuous log-uniform random variable is the continuous version of a log-spaced
# parameter. For example, to specify the equivalent of the L1 parameter from above,
# `loguniform(1e-3, 1e-1)` can be used instead of `[0.001, 0.01, 0.1]`.
#
# Mirroring the example above in grid search, we can specify a continuous random
# variable that is log-uniformly distributed between 1e-3 and 1e-1:

randomized_search = RandomizedSearchCV(
    estimator=ref_model,
    cv=cv,
    n_jobs=-1,
    param_distributions={
        "l2_coef": loguniform(1e-3, 1e-1),
    },
    n_iter=100,
    return_train_score=True,
    scoring=make_scorer(RatioMeasure.ANNUALIZED_SHARPE_RATIO),
)
randomized_search.fit(X_train)
best_model_rd = randomized_search.best_estimator_
print(best_model_rd)

# %%
# Let's plot both the average in-sample and out-of-sample scores (annualized Sharpe
# ratio) as a function of `l2_coef`:

cv_results = randomized_search.cv_results_
x = np.asarray(cv_results["param_l2_coef"]).astype(float)
sort_idx = np.argsort(x)
y_train_mean = cv_results["mean_train_score"][sort_idx]
y_train_std = cv_results["std_train_score"][sort_idx]
y_test_mean = cv_results["mean_test_score"][sort_idx]
y_test_std = cv_results["std_test_score"][sort_idx]
x = x[sort_idx]

fig = go.Figure([
    go.Scatter(
        x=x,
        y=y_train_mean,
        name="Train",
        mode="lines",
        line=dict(color="rgb(31, 119, 180)"),
    ),
    go.Scatter(
        x=x,
        y=y_train_mean + y_train_std,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ),
    go.Scatter(
        x=x,
        y=y_train_mean - y_train_std,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        fillcolor="rgba(31, 119, 180,0.15)",
        fill="tonexty",
    ),
    go.Scatter(
        x=x,
        y=y_test_mean,
        name="Test",
        mode="lines",
        line=dict(color="rgb(255,165,0)"),
    ),
    go.Scatter(
        x=x,
        y=y_test_mean + y_test_std,
        mode="lines",
        line=dict(width=0),
        showlegend=False,
    ),
    go.Scatter(
        x=x,
        y=y_test_mean - y_test_std,
        line=dict(width=0),
        mode="lines",
        fillcolor="rgba(255,165,0, 0.15)",
        fill="tonexty",
        showlegend=False,
    ),
])
fig.add_vline(
    x=randomized_search.best_params_["l2_coef"],
    line_width=2,
    line_dash="dash",
    line_color="green",
)
fig.update_layout(
    title="Train/Test score",
    xaxis_title="L2 Coef",
    yaxis_title="Annualized Sharpe Ratio",
)
fig.update_yaxes(tickformat=".2f")
show(fig)

# %%
# |
#
# The highest mean out-of-sample Sharpe Ratio is 1.55 and is achieved for a L2 coef of
# 0.023.
# Also note that without regularization, the mean train Sharpe Ratio is around
# six time higher than the mean test Sharpe Ratio. That would be a clear indiction of
# overfitting.
#
# Now, we analyze all three models on the test set. By using `cross_val_predict` with
# `WalkForward`, we are able to compute efficiently the `MultiPeriodPortfolio`
# composed of 60 days rolling portfolios fitted on the preceding 252 days:

benchmark = EqualWeighted()
pred_bench = cross_val_predict(benchmark, X_test, cv=cv)
pred_bench.name = "Benchmark"

pred_no_reg = cross_val_predict(ref_model, X_test, cv=cv)
pred_no_reg.name = "No Regularization"

pred_reg = cross_val_predict(best_model, X_test, cv=cv, n_jobs=-1)
pred_reg.name = "Regularization"

population = Population([pred_no_reg, pred_reg, pred_bench])
population.plot_cumulative_returns()

# %%
# From the plot and the below summary, we can see that the un-regularized model is
# overfitted and perform poorly on the test set. Its annualized volatility is 54%, which
# is significantly above the model upper-bound of 30% and its Sharpe Ratio is 0.32 which
# is the lowest of all models.

population.summary()

# %%
# Finally, we plot the composition of the regularized multi-period portfolio:
pred_reg.plot_composition()
