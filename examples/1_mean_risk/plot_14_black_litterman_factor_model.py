r"""
==============================
Black & Litterman Factor Model
==============================

This tutorial shows how to use the :class:`~skfolio.prior.FactorModel` estimator coupled
with the :class:`~skfolio.prior.BlackLitterman` estimator in the
:class:`~skfolio.optimization.MeanRisk` optimization.

The Black & Litterman Factor Model is a Factor Model in which we incorporate views on
factors using the Black & Litterman Model.

In the previous two tutorials, we introduced the Factor Model and the Black & Litterman
separately. In this tutorial we show how we can merge them together by building a
Maximum Sharpe Ratio portfolio using the `FactorModel` estimator.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the SPX Index composition and the Factors dataset composed of the daily
# prices of 5 ETF representing common factors:
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import BlackLitterman, FactorModel

prices = load_sp500_dataset()
factor_prices = load_factors_dataset()

prices = prices["2014":]
factor_prices = factor_prices["2014":]

X, y = prices_to_returns(prices, factor_prices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

# %%
# Analyst views
# =============
# Let's assume we are able to accurately estimate views about future realization of the
# factors. We estimate that the factor Size will have an expected return of 10% p.a.
# (absolute view) and will outperform the factor Value by 3% p.a. (relative view). We
# also estimate the factor Momentum will outperform the factor Quality by 2% p.a
# (relative view). By converting these annualized estimates into daily estimates to be
# homogenous with the input `X`, we get:
factor_views = [
    "SIZE == 0.00039",
    "SIZE - VLUE == 0.00011 ",
    "MTUM - QUAL == 0.00007",
]

# %%
# Black & Litterman Factor Model
# ==============================
# We create a Maximum Sharpe Ratio model using the Black & Litterman Factor Model that
# we fit on the training set:
model_bl_factor = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(
        factor_prior_estimator=BlackLitterman(views=factor_views),
    ),
    portfolio_params=dict(name="Black & Litterman Factor Model"),
)
model_bl_factor.fit(X_train, y_train)
model_bl_factor.weights_

# %%
# For comparison, we also create a Maximum Sharpe Ratio model using a simple Factor
# Model:
model_factor = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(),
    portfolio_params=dict(name="Factor Model"),
)
model_factor.fit(X_train, y_train)
model_factor.weights_

# %%
# Prediction
# ==========
# We predict both models on the test set:
ptf_bl_factor_test = model_bl_factor.predict(X_test)
ptf_factor_test = model_factor.predict(X_test)

population = Population([ptf_bl_factor_test, ptf_factor_test])

population.plot_cumulative_returns()

# %%
# Because our factor views were accurate, the Black & Litterman Factor Model
# outperformed the simple Factor Model on the test set.
#
# Let's plot the portfolios compositions:
fig = population.plot_composition()
show(fig)
