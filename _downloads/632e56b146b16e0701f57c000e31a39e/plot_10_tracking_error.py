r"""
==============
Tracking Error
==============

This tutorial shows how to incorporate a tracking error constraint into the
:class:`~skfolio.optimization.MeanRisk` optimization.

The tracking error is defined as the RMSE (root-mean-square error) of the portfolio
returns compared to a target returns.

In this example we will create a long-short portfolio of 20 stocks that tracks the
SPX Index with a tracking error constraint of 0.30% while minimizing the CVaR
(Conditional Value at Risk) at 95%.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition and the prices of the S&P 500 Index itself:

import numpy as np
from plotly.io import show
from sklearn import clone
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset, load_sp500_index
from skfolio.optimization import EqualWeighted, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices["2014":]
spx_prices = load_sp500_index()
spx_prices = spx_prices["2014":]

X, y = prices_to_returns(prices, spx_prices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create two long-short models: a Minimum CVaR without tracking error and a
# Minimum CVaR with a 0.30% tracking error constraint versus the SPX Index.
# A 0.30% tracking error constraint is a constraint on the RMSE of the difference
# between the daily portfolio returns and the SPX Index returns.
# We first create the Minimum CVaR model without tracking error:
model_no_tracking = MeanRisk(
    objective_function=ObjectiveFunction.MINIMIZE_RISK,
    risk_measure=RiskMeasure.CVAR,
    min_weights=-1,
    portfolio_params=dict(name="Minimum-CVaR", tag="No Tracking"),
)
model_no_tracking.fit(X_train, y_train)
model_no_tracking.weights_

# %%
# Then we create the Minimum CVaR model with a 0.30% tracking error constraint
# versus the SPX Index:
model_tracking = clone(model_no_tracking)
model_tracking.set_params(
    max_tracking_error=0.003,
    portfolio_params=dict(name="Minimum-CVaR", tag="Tracking 0.30%"),
)
model_tracking.fit(X_train, y_train)
model_no_tracking.weights_

# %%
# For comparison, we create a single asset Portfolio model containing the SPX Index.
model_spx = EqualWeighted(portfolio_params=dict(name="SPX Index"))
model_spx.fit(y_train)
model_spx.weights_

# %%
# Now we plot both models and the SPX Index on the training set:
ptf_no_tracking_train = model_no_tracking.predict(X_train)
ptf_tracking_train = model_tracking.predict(X_train)
spx_train = model_spx.predict(y_train)
# Note that we coule have directly used:
# train_spx = Portfolio(y_train, weights=[1], name="SPX Index")

population_train = Population([ptf_no_tracking_train, ptf_tracking_train, spx_train])

fig = population_train.plot_cumulative_returns()
show(fig)

# %%
# |
#
# Let's print the tracking error and the CVaR:
for portfolio in [ptf_no_tracking_train, ptf_tracking_train]:
    tracking_rmse = np.sqrt(np.mean((portfolio.returns - spx_train.returns) ** 2))
    print("========================")
    print(portfolio.tag)
    print("========================")
    print(f"Tracking RMSE: {tracking_rmse:0.2%}")
    print(f"CVaR at 95%: {portfolio.cvar:0.2%}")
    print(f"CVaR ratio: {portfolio.cvar_ratio:0.2f}")
    print("\n")

# %%
# The model with tracking error achieved the required RMSE of 0.30% versus the SPX on
# the training set. The tradeoff of this constraint is the higher CVaR value versus
# the model without tracking error.

# %%
# Prediction
# ==========
# Finally, we predict both models on the test set:
ptf_no_tracking_test = model_no_tracking.predict(X_test)
ptf_tracking_test = model_tracking.predict(X_test)
spx_test = model_spx.predict(y_test)

for portfolio in [ptf_no_tracking_test, ptf_tracking_test]:
    tracking_rmse = np.sqrt(np.mean((portfolio.returns - spx_test.returns) ** 2))
    print("========================")
    print(portfolio.tag)
    print("========================")
    print(f"Tracking RMSE: {tracking_rmse:0.2%}")
    print(f"CVaR at 95%: {portfolio.cvar:0.2%}")
    print(f"CVaR ratio: {portfolio.cvar_ratio:0.2f}")
    print("\n")

# %%
# As expected, the model with tracking error also achieved a lower RMSE on the test set
# compared to the model without tracking error.
