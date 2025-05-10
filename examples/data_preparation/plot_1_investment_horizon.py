"""
==================
Investment Horizon
==================

This tutorial explores the difference between the general
procedure using different investment horizon and the simplified procedure as explained
in :ref:`data preparation <data_preparation>`.
"""

# %%
# Prices
# ======
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28:
from plotly.io import show

from skfolio import PerfMeasure, Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_sp500_dataset()
prices.head()

# %%
# Linear Returns
# ==============
# We transform the daily prices into daily linear returns:
X = prices_to_returns(prices)

# %%
# Model
# =====
# We first create a Mean-Variance model using the simplified procedure:
population = Population([])

model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    efficient_frontier_size=30,
    portfolio_params=dict(name="Simplified", tag="Simplified"),
)
population.extend(model.fit_predict(X))

for tag, investment_horizon in [
    ("3M", 252 / 4),
    ("1Y", 252),
    ("10Y", 10 * 252),
]:
    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=30,
        prior_estimator=EmpiricalPrior(
            is_log_normal=True, investment_horizon=investment_horizon
        ),
        portfolio_params=dict(name=f"General - {tag}", tag=f"General - {tag}"),
    )
    population.extend(model.fit_predict(X))


# %%
# Let's plot the efficient frontier:
fig = population.plot_measures(
    x=RiskMeasure.ANNUALIZED_VARIANCE,
    y=PerfMeasure.ANNUALIZED_MEAN,
)
show(fig)

# %%
# |
#
# Let's plot the portfolios compositions:
population.plot_composition()

# %%
# We can see that the simplified procedure only start to diverge from the general one
# for investment horizons longer than one year.
