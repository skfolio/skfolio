r"""
=================
Black & Litterman
=================

This tutorial shows how to use the :class:`~skfolio.prior.BlackLitterman` estimator in
the :class:`~skfolio.optimization.MeanRisk` optimization.

A :ref:`prior estimator <prior>` fits a :class:`~skfolio.prior.PriorModel` containing
the distribution estimate of asset returns. It represents the investor's prior beliefs
about the model used to estimate such distribution.

The `PriorModel` is a dataclass containing:

    * `mu`: Expected returns estimation
    * `covariance`: Covariance matrix estimation
    * `returns`: assets returns estimation
    * `cholesky` : Lower-triangular Cholesky factor of the covariance estimation (optional)

The `BlackLitterman` estimator estimates the `PriorModel` using the Black & Litterman
model. It takes as input a prior estimator used to compute the prior expected returns
and prior covariance matrix, which are updated using the analyst's views to get the
posterior expected returns and posterior covariance matrix.

In this tutorial we will build a Maximum Sharpe Ratio portfolio using the
`BlackLitterman` estimator.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the SPX Index composition starting from 1990-01-02 up to 2022-12-28:
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import BlackLitterman

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Analyst views
# =============
# Let's assume we are able to accurately estimate views about future realization of the
# market. We estimate that Apple will have an expected return of 25% p.a. (absolute
# view) and will outperform General Electric by 22% p.a. (relative view). We also
# estimate that JPMorgan will outperform General Electric by 15% p.a (relative view).
# By converting these annualized estimates into daily estimates to be homogenous with
# the input `X`, we get:
analyst_views = [
    "AAPL == 0.00098",
    "AAPL - GE == 0.00086",
    "JPM - GE == 0.00059",
]

# %%
# Black & Litterman Model
# =======================
# We create a Maximum Sharpe Ratio model using the Black & Litterman estimator that we
# fit on the training set:
model_bl = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=BlackLitterman(views=analyst_views),
    portfolio_params=dict(name="Black & Litterman"),
)
model_bl.fit(X_train)
model_bl.weights_

# %%
# Empirical Model
# ===============
# For comparison, we also create a Maximum Sharpe Ratio model using the default
# Empirical estimator:
model_empirical = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Empirical"),
)
model_empirical.fit(X_train)
model_empirical.weights_

# %%
# Prediction
# ==========
# We predict both models on the test set:
pred_bl = model_bl.predict(X_test)
pred_empirical = model_empirical.predict(X_test)

population = Population([pred_bl, pred_empirical])

population.plot_cumulative_returns()

# %%
# Because our views were accurate, the Black & Litterman model outperformed the
# Empirical model on the test set. From the below composition, we can see that Apple
# and JPMorgan were allocated more weights:

fig = population.plot_composition()
show(fig)
