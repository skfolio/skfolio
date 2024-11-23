"""
=======================
Cardinality Constraints
=======================

This tutorial shows how to use cardinality constraints with the
:class:`~skfolio.optimization.MeanRisk` optimization.

Cardinality constraint controls the total number of invested assets (non-zero weights)
in the portfolio. Cardinality constraints can also be specified for asset groups
(e.g., tech, healthcare).

In a previous tutorial, we showed how to reduce the number of assets using L1
regularization. However, asset managers sometimes require more granularity and
precision regarding the exact number of assets allowed, both in total and per group.

Cardinality constraints require a mixed-integer solver. For an open-source option,
we recommend using SCIP by setting `solver="SCIP"`. To install it, use:
`pip install cvxpy[SCIP]`. For commercial solvers, supported options include
MOSEK, GUROBI, or CPLEX.

Mixed-Integer Programming (MIP) involves optimization with both continuous and integer
variables and is inherently non-convex due to the discrete nature of integer variables.
Over recent decades, MIP solvers have significantly advanced, utilizing methods
like Branch and Bound and cutting planes to improve efficiency.

By leveraging specialized techniques such as homogenization and the Big M method,
combined with problem-specific calibration, Skfolio can reformulate these complex
problems into a Mixed-Integer Program that can be efficiently solved using these
solvers.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2018-01-02 up to 2022-12-28.

import numpy as np
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import PerfMeasure, RiskMeasure, RatioMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices["2018":]
X = prices_to_returns(prices)

# %%
# Cardinality Constraint
# ======================
# let's use a Minimum CVaR model and limit the total number of assets to 5:
model = MeanRisk(risk_measure=RiskMeasure.CVAR, cardinality=5, solver="SCIP")
model.fit(X)
model.weights_

# %%
# We notice that the number of non-zero weights is indeed equal to 5.
# You can change the default solver parameters using `solver_params`.
# For more details about solver arguments, check the CVXPY
# documentation: https://www.cvxpy.org/tutorial/solvers

# %%
# Cardinality Constraint per Group
# ================================
# First, let's assign two groups to each asset: sector and capitalization.

groups = {
    "AAPL": ["Technology", "Mega Cap"],
    "AMD": ["Technology", "Large Cap"],
    "BAC": ["Financials", "Mega Cap"],
    "BBY": ["Consumer", "Large Cap"],
    "CVX": ["Energy", "Mega Cap"],
    "GE": ["Industrials", "Large Cap"],
    "HD": ["Consumer", "Mega Cap"],
    "JNJ": ["Healthcare", "Mega Cap"],
    "JPM": ["Financials", "Mega Cap"],
    "KO": ["Consumer", "Mega Cap"],
    "LLY": ["Healthcare", "Mega Cap"],
    "MRK": ["Healthcare", "Mega Cap"],
    "MSFT": ["Technology", "Mega Cap"],
    "PEP": ["Consumer", "Mega Cap"],
    "PFE": ["Healthcare", "Mega Cap"],
    "PG": ["Consumer", "Mega Cap"],
    "RRC": ["Energy", "Small Cap"],
    "UNH": ["Healthcare", "Mega Cap"],
    "WMT": ["Consumer", "Mega Cap"],
    "XOM": ["Energy", "Mega Cap"],
}

# %%
# Let's restrict the maximum number of assets in the following groups:
#
#  * Healthcare: 2
#  * Small Cap: 1
#  * Mega Cap: 4

model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    groups=groups,
    group_cardinalities={"Healthcare": 2, "Small Cap": 1, "Mega Cap": 4},
    solver="SCIP",
)
model.fit(X)
model.weights_

# %%
# We can see that the maximum number of assets per group has been respected:
portfolio = model.predict(X)
print({k: groups[k] for k in portfolio.composition.index})

# %%
# Efficient Frontier
# ==================
# Let's plot the efficient frontiers of cardinality-constrained and unconstrained
# mean-CVaR portfolios on the training set and analyze the results on the test set.
# We will focus only on the portfolios on the frontier that have a CVaR at 95% below 5%:

X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

model = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    efficient_frontier_size=20,
    max_cvar=0.05,
    # Name and tag are used to improve plot readability
    portfolio_params=dict(name="Unconstrained", tag="Unconstrained"),
)
model.fit(X_train)

model_constrained = MeanRisk(
    risk_measure=RiskMeasure.CVAR,
    efficient_frontier_size=20,
    max_cvar=0.05,
    cardinality=3,
    solver="SCIP",
    portfolio_params=dict(name="Constrained", tag="Constrained"),
)
model_constrained.fit(X_train)

population_train = model.predict(X_train) + model_constrained.predict(X_train)
population_test = model.predict(X_test) + model_constrained.predict(X_test)

fig = population_train.plot_measures(
    x=RiskMeasure.CVAR,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.CVAR_RATIO,
    hover_measures=[RatioMeasure.ANNUALIZED_SHARPE_RATIO],
)
show(fig)


# %%
# |
#
# Let's plot the compositions:
population_train.plot_composition()

# %%
# Finlay, we can analyse the test population using methods such as:
#
#  * `population_test.summary()`
#  * `population_test.plot_cumulative_returns()`
#  * `population_test.plot_distribution(measure_list=[RatioMeasure.CVAR_RATIO, RatioMeasure.ANNUALIZED_SHARPE_RATIO])`
#  * `population_test.plot_rolling_measure(measure=RatioMeasure.CVAR_RATIO)`
