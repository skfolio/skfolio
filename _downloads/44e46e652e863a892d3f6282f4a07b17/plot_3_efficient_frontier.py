"""
==================
Efficient Frontier
==================

This tutorial uses the :class:`~skfolio.optimization.MeanRisk` optimization to find an
ensemble of portfolios belonging to the Mean-Variance efficient frontier (pareto font).
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28:

import numpy as np
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create the Mean-Variance model and then fit it on the training set.
# The parameter `efficient_frontier_size=30` is used to find 30 portfolios on the entire
# efficient frontier:
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    efficient_frontier_size=30,
    portfolio_params=dict(name="Variance"),
)
model.fit(X_train)
print(model.weights_.shape)

# %%
# Prediction
# ==========
# We predict this model on both the training set and the test set.
# The `predict` method returns the :class:`~skfolio.population.Population` of
# 30 :class:`~skfolio.portfolio.Portfolio`:
population_train = model.predict(X_train)
population_test = model.predict(X_test)

# %%
# Analysis
# ========
# For improved analysis, we add a "Train" and "Test" tag to the portfolios and
# concatenate the training and the test populations:
population_train.set_portfolio_params(tag="Train")
population_test.set_portfolio_params(tag="Test")

population = population_train + population_test

fig = population.plot_measures(
    x=RiskMeasure.ANNUALIZED_VARIANCE,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
)
show(fig)

# %%
# |
#
# Let's plot the composition of the 30 portfolios:
population_train.plot_composition()

# %%
# Let's print the Sharpe Ratio of the 30 portfolios on the test set:
population_test.measures(measure=RatioMeasure.ANNUALIZED_SHARPE_RATIO)

# %%
# Finally, we can show a full summary of the 30 portfolios evaluated on the test set:
population.summary()

# %%
# Instead of providing `efficient_frontier_size=30`, you can also provide an array of
# lower bounds for the expected returns using `min_return`. In the below example, we
# find the 5 portfolios that minimize the variance under a minimum return constraint of
# 15%, 20%, 25%, 30% and 35% (annualized):
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    min_return=np.array([0.15, 0.20, 0.25, 0.30, 0.35]) / 252,
    portfolio_params=dict(name="Variance"),
)

population = model.fit_predict(X_train)

population.plot_measures(
    x=RiskMeasure.ANNUALIZED_VARIANCE,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
)
