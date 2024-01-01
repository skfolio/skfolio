"""
====================
Maximum Sharpe Ratio
====================

This tutorial uses the :class:`~skfolio.optimization.MeanRisk` optimization to find the
maximum Sharpe Ratio portfolio.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28.
# Prices are transformed into linear returns (see :ref:`data preparation
# <data_preparation>`) and split into a training set and a test set without shuffling to
# avoid :ref:`data leakage <data_leakage>`.

import numpy as np
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import InverseVolatility, MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

print(X_train.head())

# %%
# Model
# =====
# We create a Maximum Sharpe Ratio model and then fit it on the training set.
# `portfolio_params` are parameters passed to the :class:`~skfolio.portfolio.Portfolio`
# returned by the `predict` method. It can be
# omitted, here we use it to give a name to our maximum Sharpe Ration portfolio:
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Max Sharpe"),
)
model.fit(X_train)
model.weights_

# %%
# To compare this model, we use an inverse volatility benchmark using
# the :class:`~skfolio.optimization.InverseVolatility` estimator:
benchmark = InverseVolatility(portfolio_params=dict(name="Inverse Vol"))
benchmark.fit(X_train)
benchmark.weights_

# %%
# Prediction
# ==========
# We predict the model and the benchmark on the test set:
pred_model = model.predict(X_test)
pred_bench = benchmark.predict(X_test)

# %%
# The `predict` method returns a :class:`~skfolio.portfolio.Portfolio` object.
#
# :class:`~skfolio.portfolio.Portfolio` is an array-container making it compatible
# with `scikit-learn` tools: calling `np.asarray(pred_model)` gives the portfolio
# returns (same as `pred_model.returns`):
np.asarray(pred_model)

# %%
# The :class:`~skfolio.portfolio.Portfolio` class contains a vast number of properties
# and methods used for analysis.
#
# | For example:
#
# * pred_model.plot_cumulative_returns()
# * pred_model.plot_composition()
# * pred_model.summary()
print(pred_model.annualized_sharpe_ratio)
print(pred_bench.annualized_sharpe_ratio)

# %%
# Analysis
# ========
# For improved analysis, we load both predicted portfolios into a
# :class:`~skfolio.population.Population`:
population = Population([pred_model, pred_bench])

# %%
# The :class:`~skfolio.population.Population` class also contains a
# vast number of properties and methods used for analysis.
# Let's plot each portfolio composition:
population.plot_composition()

# %%
# .. note::
#       Every `plot` methods in `skfolio` returns a `plotly` figure.
#       To display a plotly figure, you may need to call `show()` and change the
#       default renderer: https://plotly.com/python/renderers/
#
# Let's plot each portfolio cumulative returns:
fig = population.plot_cumulative_returns()
# show(fig) is only used for the documentation sticker.
show(fig)

# %%
# |
#
# Finally, let's display the full summary of both strategies evaluated on the test
# set:
population.summary()

# %%
# Conclusion
# ==========
# From the analysis on the test set, we see that the Maximum Sharpe Ratio portfolio
# outperform the inverse-volatility benchmark for the mean and the ratio
# measures including the Sharpe Ratio, and underperforms for the deviation and
# shortfall measures.
#
# .. seealso::
#       This was a toy example, for more advanced concepts check the :ref:`user guide
#       <user_guide>` or the :ref:`other examples <general_examples>`.
#
