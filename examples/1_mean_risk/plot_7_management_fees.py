r"""
===============
Management Fees
===============

This tutorial shows how to incorporate management fees (MF) into the 
:class:`~skfolio.optimization.MeanRisk` optimization.

By using The `management_fees` parameter, you can add linear MF to the optimization 
problem:

.. math:: total\_fee = \sum_{i=1}^{N} f_{i} \times w_{i}

with :math:`f_{i}` the management fee of asset i and :math:`w_{i}` its weight.
The float :math:`total\_fee` is impacting the portfolio expected return in the optimization:

.. math:: expected\_return = \mu^{T} \cdot w - total\_fee

with :math:`\mu` the vector af assets expected returns and :math:`w` the vector of 
assets weights.

The `management_fees` parameter can be a float, a dictionary or an array-like of
shape `(n_assets, )`. If a float is provided, it is applied to each asset.
If a dictionary is provided, its (key/value) pair must be the (asset name/asset MF) and
the input `X` of the `fit` method must be a DataFrame with the assets names in
columns. The default is 0.0 (no management fees).

.. note::

    Another approach is to direcly impact the MF to the input `X` in order to express 
    the returns net of fee. However, when estimating the :math:`\mu` parameter using,
    for example, Shrinkage estimators, this approach would mix a deterministic amount
    with an uncertain one leading to unwanted bias in the management fees.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28.
# We select only 3 assets to make the example more readable, which are Apple (AAPL),
# General Electric (GE) and JPMorgan (JPM).

import numpy as np
from plotly.io import show

from skfolio import Population
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
# Management Fees
# ===============
# Management fees are usually used in assets under management but for this example we
# will assume that it also applies for the below stocks:
#
#   * Apple: 3% p.a.
#   * General Electric: 6% p.a.
#   * JPMorgan: 1% p.a.
#
# The MF are expressed in per annum, so we need to convert them in daily MF.
# We suppose 252 trading days in a year:
management_fees = {"AAPL": 0.03 / 252, "GE": 0.06 / 252, "JPM": 0.01 / 252}
# Same as management_fees = np.array([0.03, 0.06, 0.01]) / 252

model_mf = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    management_fees=management_fees,
)
model_mf.fit(X)
model_mf.weights_

# %%
# The higher MF of Apple induced a change of weights toward JPMorgan:
model_mf.weights_ - model.weights_

# %%
# Multi-period portfolio
# ======================
# Let's assume that we want to rebalance our portfolio every 60 days by re-fitting the
# model on the latest 60 days. We test the impact of MF using Walk Forward Analysis:
holding_period = 60
fitting_period = 60
cv = WalkForward(train_size=fitting_period, test_size=holding_period)

# %%
# As explained above, we transform the yearly MF into a daily MF:
management_fees = np.array([0.03, 0.06, 0.01]) / 252

# %%
# First, we train the model without MF and test it with MF.
# Note that `portfolio_params` are parameters passed to the Portfolio during `predict`
# and **not** during `fit`:
model = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    portfolio_params=dict(management_fees=management_fees),
)
# pred1 is a MultiPeriodPortfolio
pred1 = cross_val_predict(model, X, cv=cv, n_jobs=-1)
pred1.name = "pred1"

# %%
# Then, we train and test the model with MF:
model.set_params(management_fees=management_fees)
pred2 = cross_val_predict(model, X, cv=cv, n_jobs=-1)
pred2.name = "pred2"

# %%
# We visualize the results by plotting the cumulative returns of the successive test
# periods:
population = Population([pred1, pred2])
fig = population.plot_cumulative_returns()
show(fig)

# %%
# |
#
# We notice that the model **fitted with MF** outperform the model **fitted without
# MF**.
