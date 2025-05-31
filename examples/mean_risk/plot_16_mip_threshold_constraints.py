"""
=====================
Threshold Constraints
=====================

This tutorial shows how to use threshold constraints with the
:class:`~skfolio.optimization.MeanRisk` optimization.

Threshold constraints ensure that invested assets have sufficiently large weights.
This can help eliminate insignificant allocations.

Both long and short position thresholds can be controlled using `threshold_long` and
`threshold_short`.

Threshold constraints require a mixed-integer solver. For an open-source option,
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

from plotly.io import show
from skfolio import RiskMeasure, Population
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices["2018":]
X = prices_to_returns(prices)

# %%
# Model
# =====
# let's use a long-short Minimum CVaR model:
model = MeanRisk(
    min_weights=-1,
    risk_measure=RiskMeasure.CVAR,
)
model.fit(X)
model.weights_

# %%
# Now, let's assume we don't want weights that are too small.
# This means that, let’s say, if an asset is invested (non-zero weight), it needs to be
# between -100% to -10% **or** +15% to +100%:
model_threshold = MeanRisk(
    min_weights=-1,
    risk_measure=RiskMeasure.CVAR,
    threshold_long=0.15,
    threshold_short=-0.10,
    solver="SCIP",
)
model_threshold.fit(X)
model_threshold.weights_

# %%
# We notice that the long and short threshold constraints have been respected.
# You can change the default solver parameters using `solver_params`.
# For more details about solver arguments, check the CVXPY
# documentation: https://www.cvxpy.org/tutorial/solvers
#
# To visualize both portfolio compositions, let's plot them:
ptf = model.predict(X)
ptf.name = "Min CVaR"
ptf_threshold = model_threshold.predict(X)
ptf_threshold.name = "Min CVaR with Threshold Constraints"
population = Population([ptf, ptf_threshold])
fig = population.plot_composition()
show(fig)
