"""
======================
Risk Parity - Variance
======================

This tutorial uses the :class:`~skfolio.optimization.RiskBudgeting` optimization to
find the risk parity portfolio with variance as the risk measure.
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
# Model
# =====
# We create the risk parity model and then fit it on the training set:
model = RiskBudgeting(
    risk_measure=RiskMeasure.VARIANCE,
    portfolio_params=dict(name="Risk Parity - Variance"),
)
model.fit(X_train)
model.weights_

# %%
# To compare this model, we use an inverse volatility benchmark using
# the :class:`~skfolio.optimization.InverseVolatility` estimator.
bench = InverseVolatility(portfolio_params=dict(name="Inverse Vol"))
bench.fit(X_train)
bench.weights_

# %%
# Risk Contribution Analysis
# ==========================
# Let's analyze the risk contribution of both models on the training set.
# As expected, the risk parity model has the same variance contribution for each asset:
ptf_model_train = model.predict(X_train)
ptf_model_train.plot_contribution(measure=RiskMeasure.ANNUALIZED_VARIANCE)

# %%
# And the inverse volatility model has non-equal variance contribution. This is because
# the correlation is not taken into account in an inverse volatility model:
ptf_bench_train = bench.predict(X_train)
ptf_bench_train.plot_contribution(measure=RiskMeasure.ANNUALIZED_VARIANCE)

# %%
# Prediction
# ==========
# We predict the model and the benchmark on the test set:
ptf_model_test = model.predict(X_test)
ptf_bench_test = bench.predict(X_test)

# %%
# The `predict` method returns a :class:`~skfolio.portfolio.Portfolio` object.


# %%
# Analysis
# ========
# For improved analysis, we load both predicted portfolios into a
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
