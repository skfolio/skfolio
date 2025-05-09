"""
============================
Distributionally Robust CVaR
============================

This tutorial introduces the :class:`~skfolio.optimization.DistributionallyRobustCVaR`
model.

The Distributionally Robust CVaR model constructs a Wasserstein ball in the space of
multivariate and non-discrete probability distributions centered at the uniform
distribution on the training samples, and find the allocation that minimize the CVaR of
the worst-case distribution within this Wasserstein ball.

Mohajerin Esfahani and Kuhn (2018) proved that for piecewise linear objective functions,
which is the case of CVaR (Rockafellar and Uryasev), the distributionally robust
optimization problem over Wasserstein ball can be reformulated as finite convex
programs.

It's advised to use a solver that handles a high number of constraints like `Mosek`.
For accessibility, this example uses the default open source solver `CLARABEL`, so to
increase convergence speed, we only use 3 years of data.

The radius of the Wasserstein ball is controlled with the `wasserstein_ball_radius`
parameter. Increasing the radius will increase the uncertainty about the
distribution, bringing the weights closer to the equal weighted portfolio.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2020-01-02 up to 2022-12-28:
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import DistributionallyRobustCVaR, EqualWeighted
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices["2020":]

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.5, shuffle=False)

# %%
# Model
# =====
# We create four distributionally robust CVaR models with different radius then fit them
# on the training set:
model1 = DistributionallyRobustCVaR(
    wasserstein_ball_radius=0.1,
    portfolio_params=dict(name="Distributionally Robust CVaR - 0.1"),
)
model1.fit(X_train)

model2 = DistributionallyRobustCVaR(
    wasserstein_ball_radius=0.01,
    portfolio_params=dict(name="Distributionally Robust CVaR - 0.01"),
)
model2.fit(X_train)

model3 = DistributionallyRobustCVaR(
    wasserstein_ball_radius=0.001,
    portfolio_params=dict(name="Distributionally Robust CVaR - 0.001"),
)
model3.fit(X_train)

model4 = DistributionallyRobustCVaR(
    wasserstein_ball_radius=0.0001,
    portfolio_params=dict(name="Distributionally Robust CVaR - 0.0001"),
)
model4.fit(X_train)
model4.weights_

# %%
# To compare the models, we use an equal weighted benchmark using
# the :class:`~skfolio.optimization.EqualWeighted` estimator:
bench = EqualWeighted()
bench.fit(X_train)
bench.weights_

# %%
# Prediction
# ==========
# We predict the models and the benchmark on the test set:
ptf_model1_test = model1.predict(X_test)
ptf_model2_test = model2.predict(X_test)
ptf_model3_test = model3.predict(X_test)
ptf_model4_test = model4.predict(X_test)
ptf_bench_test = bench.predict(X_test)

# %%
# Analysis
# ========
# We load all predicted portfolios into a :class:`~skfolio.population.Population` and
# plot their compositions:
population = Population(
    [ptf_model1_test, ptf_model2_test, ptf_model3_test, ptf_model4_test, ptf_bench_test]
)
population.plot_composition()

# %%
# We can see that by increasing the radius of the Wasserstein ball, the weights get
# closer to the equal weighted portfolio.
#
# Let's plot the portfolios cumulative returns:
fig = population.plot_cumulative_returns()
show(fig)
