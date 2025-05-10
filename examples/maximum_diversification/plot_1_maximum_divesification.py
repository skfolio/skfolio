"""
=======================
Maximum Diversification
=======================

This tutorial uses the :class:`~skfolio.optimization.MaximumDiversification`
optimization to find the portfolio that maximizes the diversification ratio, which is
the ratio of the weighted volatilities over the total volatility.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28:
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import EqualWeighted, MaximumDiversification
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)

X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create the maximum diversification model and then fit it on the training set:
model = MaximumDiversification()
model.fit(X_train)
model.weights_

# %%
# To compare this model, we use an equal weighted benchmark using
# the :class:`~skfolio.optimization.EqualWeighted` estimator:
bench = EqualWeighted()
bench.fit(X_train)
bench.weights_

# %%
# Diversification Analysis
# ========================
# Let's analyze the diversification ratio of both models on the training set.
# As expected, the maximum diversification model has the highest diversification ratio:
ptf_model_train = model.predict(X_train)
ptf_bench_train = bench.predict(X_train)
print("Diversification Ratio:")
print(f"    Maximum Diversification model: {ptf_model_train.diversification:0.2f}")
print(f"    Equal Weighted model: {ptf_bench_train.diversification:0.2f}")

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
fig = population.plot_composition()
show(fig)

# %%
# |
#
# Finally we can show a full summary of both strategies evaluated on the test set:
population.plot_cumulative_returns()

# %%
# |
#
# Finally, we print a full summary of both strategies evaluated on the test set:
population.summary()
