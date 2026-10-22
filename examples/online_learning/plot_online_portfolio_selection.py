"""
Online portfolio selection on stock returns
==========================================

Compare online learners with a daily equal-weight allocation using the
skfolio S&P 500 dataset. The first 60 observations initialize the learners;
performance is measured only over the following observations. Every decision
uses returns available before its holding period.

This short historical comparison illustrates the API, not a claim that one
strategy consistently outperforms another. Costs are zero for all strategies.
"""

from plotly.io import show

from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import online_predict
from skfolio.optimization import EqualWeighted
from skfolio.optimization.online import (
    AdaGrad,
    ExponentiatedGradient,
    OnlineMovingAverageReversion,
    PassiveAggressiveMeanReversion,
)
from skfolio.population import Population
from skfolio.preprocessing import prices_to_returns

# %%
# Keep a small, fixed universe and a bounded period so the example runs quickly.
# Columns are chosen by dataset order, without using future strategy performance.
prices = load_sp500_dataset().iloc[:, :5]
X = prices_to_returns(prices).loc["2016":"2017"].iloc[:312]
warmup = 60
models = [
    ExponentiatedGradient(update_mode="ftrl"),
    AdaGrad(learning_rate=0.05),
    OnlineMovingAverageReversion(window=5),
    PassiveAggressiveMeanReversion(),
]

# %%
# Portfolio evaluation carries drifted holdings between daily rebalances.
# The comparator uses equal weights every day over exactly the same test dates.
portfolios = []
for model in models:
    model.set_params(portfolio_params={"weight_drift": True})
    portfolio = online_predict(model, X, warmup_size=warmup, test_size=1)
    portfolio.name = type(model).__name__
    portfolios.append(portfolio)

portfolios.append(
    EqualWeighted(
        portfolio_params={"name": "Daily equal weight", "weight_drift": False}
    )
    .fit(X.iloc[:warmup])
    .predict(X.iloc[warmup:])
)
population = Population(portfolios)
print(population.summary())
show(population.plot_cumulative_returns())

# %%
# Adaptive geometry and coupled constraints may require a convex solve at every
# observation. Their runtime can exceed the closed-form entropy update used by
# unconstrained EG. Compare runtimes on the same universe and constraints.
