"""
=====================
Risk Budgeting - CVaR
=====================

This tutorial uses the :class:`~skfolio.optimization.RiskBudgeting` optimization to
build a risk budgeting portfolio by specifying a risk budget on each asset with CVaR as
the risk measure.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28:

from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import InverseVolatility, RiskBudgeting
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Risk Budget
# ===========
# We chose the following risk budget: 1.5 on Apples, 0.2 on General Electric and
# JPMorgan and 1.0 on the remaining assets:
risk_budget = {asset_name: 1 for asset_name in X.columns}
risk_budget["AAPL"] = 1.5
risk_budget["GE"] = 0.2
risk_budget["JPM"] = 0.2

# %%
# Model
# =====
# We create the risk budgeting model and then fit it on the training set:
model = RiskBudgeting(
    risk_measure=RiskMeasure.CVAR,
    risk_budget=risk_budget,
    portfolio_params=dict(name="Risk Budgeting - CVaR"),
)
model.fit(X_train)
model.weights_

# %%
# To compare this model, we use an inverse volatility benchmark using
# the :class:`~skfolio.optimization.InverseVolatility` estimator:
bench = InverseVolatility(portfolio_params=dict(name="Inverse Vol"))
bench.fit(X_train)
bench.weights_

# %%
# Risk Contribution Analysis
# ==========================
# Let's analyze the risk contribution of both models on the training set.
# As expected, the risk budgeting model has 50% more CVaR contribution to Apple and 80%
# less to General Electric and JPMorgan compared to the other assets:
ptf_model_train = model.predict(X_train)
fig = ptf_model_train.plot_contribution(measure=RiskMeasure.CVAR)
show(fig)

# %%
# |
#
# And the inverse volatility model has different CVaR contribution for each asset:
ptf_bench_train = bench.predict(X_train)
ptf_bench_train.plot_contribution(measure=RiskMeasure.CVAR)

# %%
# Prediction
# ==========
# We predict the model and the benchmark on the test set:
ptf_model_test = model.predict(X_test)
ptf_bench_test = bench.predict(X_test)

# %%
# Analysis
# ========
# For improved analysis, it's possible to load both predicted portfolios into a
# :class:`~skfolio.population.Population`:
population = Population([ptf_model_test, ptf_bench_test])

# %%
# Let's plot each portfolio composition:
population.plot_composition()

# %%
# Let's plot each portfolio cumulative returns:
fig = population.plot_cumulative_returns()
show(fig)

# %%
# |
#
# Finally, we print a full summary of both strategies evaluated on the test set:
population.summary()
