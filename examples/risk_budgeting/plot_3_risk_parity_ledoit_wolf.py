"""
==================================
Risk Parity - Covariance shrinkage
==================================

This tutorial shows how to incorporate covariance shrinkage in the
:class:`~skfolio.optimization.RiskBudgeting` optimization.
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
from skfolio.moments import ShrunkCovariance
from skfolio.optimization import RiskBudgeting
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_sp500_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create a risk parity model by using :class:`~skfolio.moments.ShrunkCovariance` as
# the covariance estimator then fit it on the training set:
model = RiskBudgeting(
    risk_measure=RiskMeasure.VARIANCE,
    prior_estimator=EmpiricalPrior(
        covariance_estimator=ShrunkCovariance(shrinkage=0.9)
    ),
    portfolio_params=dict(name="Risk Parity - Covariance Shrinkage"),
)
model.fit(X_train)
model.weights_

# %%
# To compare this model, we use a basic risk parity without covariance shrinkage:
bench = RiskBudgeting(
    risk_measure=RiskMeasure.VARIANCE,
    portfolio_params=dict(name="Risk Parity - Basic"),
)
bench.fit(X_train)
bench.weights_


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
# Let's plot each portfolio cumulative returns:
fig = population.plot_cumulative_returns()
show(fig)

# %%
# |
#
# Finally, we print a full summary of both strategies evaluated on the test set:
population.summary()
