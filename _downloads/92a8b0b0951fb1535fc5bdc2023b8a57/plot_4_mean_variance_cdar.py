"""
==========================
Mean-Variance-CDaR Surface
==========================

This tutorial uses the :class:`~skfolio.optimization.MeanRisk` optimization to find an
ensemble of portfolios belonging to the Mean-Variance-CDaR efficient frontier.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2015-01-05 up to 2022-12-28:

import numpy as np
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices["2015":]

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# First, we create a Maximum Sharpe Ratio model that we fit on the training set:
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
)
portfolio = model.fit_predict(X_train)
print(portfolio.cdar)

# %%
# Let's assume that we are not satisfied with the CDaR (Conditional Drawdown at Risk)
# of 17% corresponding to the maximum Sharpe portfolio. We want to analyze alternative
# portfolios that maximize the Sharpe under CDaR constraints.
# To have an idea of the feasible CDaR constraints, we analyze the Minimum CDaR
# portfolio:
model = MeanRisk(risk_measure=RiskMeasure.CDAR)
portfolio = model.fit_predict(X_train)
print(portfolio.cdar)

# %%
# The minimum CDaR is 9.72%.
# Now we find the pareto optimal portfolios that maximizes the Sharpe under CDaR
# constraint ranging from 9.72% to 17%:
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    max_cdar=np.linspace(start=0.0972, stop=0.17, num=10),
)
model.fit(X_train)
print(model.weights_.shape)

# %%
# Analysis
# ==========
# We predict this model on both the training set and the test set to analyze the
# deformation of the efficient frontier:
population_train = model.predict(X_train)
population_test = model.predict(X_test)

population_train.set_portfolio_params(tag="Train")
population_test.set_portfolio_params(tag="Test")

population = population_train + population_test

population.plot_measures(
    x=RiskMeasure.CDAR,
    y=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
)

# %%
# Pareto Optimal Surface
# ======================
# Instead of analyzing the Sharpe-CDaR efficient frontier, we can analyze the
# mean-Variance-CDaR pareto optimal surface:
variance_upper = population_train.max_measure(PerfMeasure.MEAN).variance
x = np.linspace(start=0.00012, stop=variance_upper, num=10)
y = np.linspace(start=0.10, stop=0.17, num=10)
x, y = map(np.ravel, np.meshgrid(x, y))

model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
    max_variance=x,
    max_cdar=y,
    raise_on_failure=False,
)
model.fit(X_train)

population_train = model.predict(X_train)

fig = population_train.plot_measures(
    x=RiskMeasure.ANNUALIZED_VARIANCE,
    y=RiskMeasure.CDAR,
    z=PerfMeasure.ANNUALIZED_MEAN,
    to_surface=True,
)
fig.update_layout(scene_camera=dict(eye=dict(x=-2, y=-0.5, z=1)))
show(fig)

# %%
# |
#
# Let's plot the composition of the portfolios:
population_train.plot_composition()

# %%
# Let's compare the average and standard-deviation of the Sharpe Ratio and CDaR Ratio of
# the portfolios on the training set versus the test set:
#
# Train:
print(population_train.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO))
print(population_train.measures_std(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO))

# %%
# Test:
print(population_test.measures_mean(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO))
print(population_test.measures_std(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO))
