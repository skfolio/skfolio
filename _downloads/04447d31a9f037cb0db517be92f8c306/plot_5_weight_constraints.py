"""
==================
Weight Constraints
==================

This tutorial shows how to incorporate weight constraints into the
:class:`~skfolio.optimization.MeanRisk` optimization.

We will show how to use the below parameters:
    * min_weights
    * max_weights
    * budget
    * min_budget
    * max_budget
    * max_short
    * max_long
    * linear_constraints
    * groups
    * left_inequality
    * right_inequality
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28.
# We select only 3 assets to make the example more readable, which are Apple (AAPL),
# General Electric (GE) and JPMorgan (JPM):

import numpy as np
from plotly.io import show

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
prices = prices[["AAPL", "GE", "JPM"]]

X = prices_to_returns(prices)

# %%
# Model
# =====
# In this tutorial, we will use a Minimum Variance model.
# By default,  :class:`~skfolio.optimization.MeanRisk` is long only (`min_weights=0`)
# and fully invested (`budget=1`). In other terms, all weights are positive and sum to
# one.
model = MeanRisk()
model.fit(X)
print(sum(model.weights_))
model.weights_

# %%
# Budget
# ======
# The budget is the sum of long positions and short positions (sum of all weights).
# It can be `None` or a float. `None` means that there are no budget constraints.
# The default is `1.0` (fully invested).
#
# Examples:
#
#   * budget = 1    –> fully invested portfolio
#   * budget = 0    –> market neutral portfolio
#   * budget = None –> no constraints on the sum of weights

model = MeanRisk(budget=0.5)
model.fit(X)
print(sum(model.weights_))
model.weights_

# %%
# You can also set a constraint on the minimum and maximum budget using `min_budget`
# and `max_budget`, which are the lower and upper bounds of the sum of long and short
# positions (sum of all weights). The default is `None`. If provided, you must set
# `budget=None`.
model = MeanRisk(budget=None, min_budget=0.3, max_budget=0.5)
model.fit(X)
print(sum(model.weights_))
model.weights_

# %%
# Lower and Upper Bounds on Weights
# =================================
# The weights lower and upper bounds are controlled by the parameters `min_weights` and
# `max_weights` respectively.
# You can provide `None`, a float, an array-like or a dictionary.
# `None` is equivalent to `-np.Inf` (no lower bounds).
# If a float is provided, it is applied to each asset.
# If a dictionary is provided, its (key/value) pair must be the (asset name/asset
# weight bound) and the input `X` of the `fit` method must be a DataFrame with the
# assets names in columns.
# The default values are `min_weights=0.0` (no short selling) and `max_weights=1.0`
# (each asset is below 100%). When using a dictionary, you don't have to provide
# constraints for all assets. If not provided, the default values (0.0 for min_weights
# and 1.0 for max_weights) will be assigned to the assets not specified in the
# dictionary.
#
# .. note ::
#
#   When incorporating a pre-selection transformer into a Pipeline, using a list for
#   weight constraints is not feasible, as we don't know in advance which assets will
#   be selected by the pre-selection process. This is where the dictionary proves
#   useful.
#
# Example:
#   * min_weights = 0                     –> long only portfolio (no short selling).
#   * min_weights = None                  –> no lower bound (same as -np.Inf).
#   * min_weights = -2                    –> each weight must be above -200%.
#   * min_weights = [0, -2, 0.5]          –> "AAPL", "GE" and "JPM" must be above 0%, -200% and 50% respectively.
#   * min_weights = {"AAPL": 0, "GE": -2} -> "AAPL", "GE" and "JPM"  must be above 0%, -200% and 0% (default) respectively.
#   * max_weights = 0                     –> no long position (short only portfolio).
#   * max_weights = None                  –> no upper bound (same as +np.Inf).
#   * max_weights = 2                     –> each weight must be below 200%.
#   * max_weights = [1, 2, -0.5]          -> "AAPL", "GE" and "JPM"  must be below 100%, 200% and -50% respectively.
#   * max_weights = {"AAPL": 1, "GE": 2}  -> "AAPL", "GE" and "JPM"  must be below 100%, 200% and 100% (default).

# %%
# Let's create a model that allows short positions with a budget of -100%:
model = MeanRisk(budget=-1, min_weights=-1)
model.fit(X)
print(sum(model.weights_))
model.weights_

# %%
# Let's add weight constraints on "AAPL", "GE" and "JPM" to be above 0%, 50% and 10%
# respectively:
model = MeanRisk(min_weights=[0, 0.5, 0.1])
model.fit(X)
print(sum(model.weights_))
model.weights_

# %%
# Let's plot the composition:
portfolio = model.predict(X)
fig = portfolio.plot_composition()
show(fig)

# %%
# |
#
# Let's create the same model as above but using partial dictionary:
model = MeanRisk(min_weights={"GE": 0.5, "JPM": 0.1})
model.fit(X)
print(sum(model.weights_))
model.weights_

# %%
# Let's create a model with a leverage of 3 and every weights below 150%:
model = MeanRisk(budget=3, max_weights=1.5)
model.fit(X)
print(sum(model.weights_))
model.weights_

# %%
# Short and Long Position Constraints
# ===================================
# Constraints on the upper bound for short and long positions can be set using
# `max_short` and `max_long`. The short position is defined as the sum of negative
# weights (in absolute term) and the long position as the sum of positive weights.

# %%
# Let's create a fully invested long-short portfolio model with a total short position
# less than 50%:
model = MeanRisk(min_weights=-1, max_short=0.5)
model.fit(X)
print(sum(model.weights_))
model.weights_

# %%
# Group and Linear Constraints
# ============================
# We can assign groups to each asset using the `groups` parameter and set
# constraints on these groups using the `linear_constraint` parameter.
# The `groups` parameter can be a 2D array-like or a dictionary. If a dictionary is
# provided, its (key/value)  pair must be the (asset name/asset groups).
# You can reference these groups and/or the asset names in `linear_constraint`, which
# is a list if strings following the below patterns:
#
#   * "2.5 * ref1 + 0.10 * ref2 + 0.0013 <= 2.5 * ref3"
#   * "ref1 >= 2.9 * ref2"
#   * "ref1 <= ref2"
#   * "ref1 >= ref1"
#
# Let's create a model with groups constraints on "industry sector" and
# "capitalization":
groups = {
    "AAPL": ["Technology", "Mega Cap"],
    "GE": ["Industrial", "Big Cap"],
    "JPM": ["Financial", "Big Cap"],
}
# You can also provide a 2D array-like:
# groups = [["Technology", "Industrial", "Financial"], ["Mega Cap", "Big Cap", "Big Cap"]]
linear_constraints = [
    "Technology + 1.5 * Industrial <= 2 * Financial",  # First group
    "Mega Cap >= 0.75 * Big Cap",  # Second group
    "Technology >= Big Cap",  # Mix of first and second groups
    "Mega Cap >= 2 * JPM",  # Mix of groups and assets
]
# Note that only the first constraint would be sufficient in that case.

model = MeanRisk(groups=groups, linear_constraints=linear_constraints)
model.fit(X)
model.weights_

# %%
# Left and Right Inequalities
# ===========================
# Finally, you can also directly provide the matrix :math:`A` and the vector
# :math:`b` of the linear constraint :math:`A \cdot w \leq b`:
left_inequality = np.array(
    [[1.0, 1.5, -2.0], [-1.0, 0.75, 0.75], [-1.0, 1.0, 1.0], [-1.0, -0.0, 2.0]]
)
right_inequality = np.array([0.0, 0.0, 0.0, 0.0])

model = MeanRisk(left_inequality=left_inequality, right_inequality=right_inequality)
model.fit(X)
model.weights_
