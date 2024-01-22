r"""
=================
Transaction Costs
=================

This tutorial shows how to incorporate transaction costs (TC) into the
:class:`~skfolio.optimization.MeanRisk` optimization.

TC are fixed costs incurred when buying or selling an asset.

By using the `transaction_costs` parameter, you can add linear TC to the optimization
problem:

.. math:: total\_cost = \sum_{i=1}^{N} c_{i} \times |w_{i} - w\_prev_{i}|

with :math:`c_{i}` the TC of asset i, :math:`w_{i}` its weight and :math:`w\_prev_{i}`
its previous weight (defined in `previous_weights`).
The float :math:`total\_cost` is impacting the portfolio expected return in the
optimization:

.. math:: expected\_return = \mu^{T} \cdot w - total\_cost

with :math:`\mu` the vector af assets expected returns and :math:`w` the vector of
assets weights.

the `transaction_costs` parameter can be a float, a dictionary or an array-like of
shape `(n_assets, )`. If a float is provided, it is applied to each asset.
If a dictionary is provided, its (key/value) pair must be the (asset name/asset TC) and
the input `X` of the `fit` method must be a DataFrame with the assets names in columns.
The default is 0.0 (no transaction costs).

.. warning::

    According to the above formula, the periodicity of the transaction costs
    needs to be homogenous to the periodicity of :math:`\mu`. For example, if
    the input `X` is composed of **daily** returns, the `transaction_costs` need
    to be expressed as **daily** costs.

This means that you need to convert this fixed transaction costs into daily costs. To
achieve this, you need the notion of expected investment duration. This is crucial since
the optimization problem has no notion of investment duration.

For example, let's assume that asset A has an expected daily return of 0.01%
with a TC of 1% and asset B has an expected daily return of 0.005% with no TC.
Let's assume both assets have the same volatility and a correlation of 1.0.
If the investment duration is only one month, we should allocate all the weights to
asset B. However, if the investment duration is one year, we should allocate all the
weights to asset A.

Example:
    * Duration = 1 months (21 business days):
        * 1 month expected return A ≈ -0.8%
        * 1 month expected return B ≈ 0.1%
    * Duration = 1 year (252 business days):
        * 1 year expected return A ≈ 1.5%
        * 1 year expected return B ≈ 1.3%

So in order to take that duration into account, you should divide the fix TC by the
expected investment duration.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28.
# We select only 3 assets to make the example more readable, which are Apple (AAPL),
# General Electric (GE) and JPMorgan (JPM):

import numpy as np
from plotly.io import show

from skfolio import MultiPeriodPortfolio, Population, Portfolio
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices[["AAPL", "GE", "JPM"]]

X = prices_to_returns(prices)

# %%
# Model
# =====
# In this tutorial, we will use the Maximum Mean-Variance Utility model with a risk
# aversion of 1.0:
model = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY)
model.fit(X)
model.weights_

# %%
# Transaction Cost
# ================
# Let's assume we have the below TC:
#   * Apple: 1%
#   * General Electric: 0.50%
#   * JPMorgan: 0.20%
#
# and an investment duration of one month (21 business days):
transaction_costs = {"AAPL": 0.01 / 21, "GE": 0.005 / 21, "JPM": 0.002 / 21}
# Same as transaction_costs = np.array([0.01, 0.005, 0.002]) / 21

# %%
# First, we assume that there is no previous position:
model_tc = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    transaction_costs=transaction_costs,
)
model_tc.fit(X)
model_tc.weights_

# %%
# The higher TC of Apple induced a change of weights toward JPMorgan:
model_tc.weights_ - model.weights_

# %%
# Now, let's assume that the previous position was equal-weighted:
model_tc2 = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    transaction_costs=transaction_costs,
    previous_weights=np.ones(3) / 3,
)
model_tc2.fit(X)
model_tc2.weights_

# %%
# Notice that the weight of General Electric becomes non-negligible due to the cost of
# rebalancing the position:
model_tc2.weights_ - model.weights_

# %%
# Multi-period portfolio
# ======================
# Let's assume that we want to rebalance our portfolio every 60 days by re-fitting the
# model on the latest 60 days. We test the impact of TC using Walk Forward Analysis:
holding_period = 60
fitting_period = 60
cv = WalkForward(train_size=fitting_period, test_size=holding_period)


# %%
# As explained above, we transform the fix TC into a daily cost by dividing the TC by
# the expected investment duration:
transaction_costs = np.array([0.01, 0.005, 0.002]) / holding_period

# %%
# First, we train and test the model without TC:
model = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY)
# pred1 is a MultiPeriodPortfolio
pred1 = cross_val_predict(model, X, cv=cv, n_jobs=-1)
pred1.name = "pred1"

# %%
# Then, we train the model without TC and test it with TC. The model trained without TC
# is the same as above so we can retrieve the results and simply update the prediction
# with the TC:
pred2 = MultiPeriodPortfolio(name="pred2")
previous_weights = None
for portfolio in pred1:
    new_portfolio = Portfolio(
        X=portfolio.X,
        weights=portfolio.weights,
        previous_weights=previous_weights,
        transaction_costs=transaction_costs,
    )
    previous_weights = portfolio.weights
    pred2.append(new_portfolio)

# %%
# Finally, we train and test the model with TC. Note that we cannot use the
# `cross_val_predict` function anymore because it uses parallelization and cannot handle
# the `previous_weights` dependency between folds:
pred3 = MultiPeriodPortfolio(name="pred3")

model.set_params(transaction_costs=transaction_costs)
previous_weights = None
for train, test in cv.split(X):
    X_train = X.take(train)
    X_test = X.take(test)
    model.set_params(previous_weights=previous_weights)
    model.fit(X_train)
    portfolio = model.predict(X_test)
    pred3.append(portfolio)
    previous_weights = model.weights_

# %%
# We visualize the results by plotting the cumulative returns of the successive test
# periods:
population = Population([pred1, pred2, pred3])
fig = population.plot_cumulative_returns()
show(fig)

# %%
# |
#
# If we exclude the unrealistic prediction without TC, we notice that the model
# **fitted with TC** outperforms the model **fitted without TC**.
