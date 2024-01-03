r"""
===============
Empirical Prior
===============

This tutorial shows how to use the :class:`~skfolio.prior.EmpiricalPrior` estimator in
the :class:`~skfolio.optimization.MeanRisk` optimization.

A :ref:`prior estimator <prior>` fits a :class:`~skfolio.prior.PriorModel` containing
the distribution estimate of asset returns. It represents the investor's prior beliefs
about the model used to estimate such distribution.

The `PriorModel` is a dataclass containing:

    * `mu`: Expected returns estimation
    * `covariance`: Covariance matrix estimation
    * `returns`: assets returns estimation
    * `cholesky` : Lower-triangular Cholesky factor of the covariance estimation (optional)

The `EmpiricalPrior` estimator simply estimates the `PriorModel` from a `mu_estimator`
and a `covariance_estimator`.

In this tutorial we will build a Maximum Sharpe Ratio portfolio using the
`EmpiricalPrior` estimator with James-Stein shrinkage for the estimation of expected
returns and Denoising for the estimation of the covariance matrix.
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
from skfolio.moments import DenoiseCovariance, ShrunkMu
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create a Maximum Sharpe Ratio model with shrinkage for the estimation of the
# expected returns and denoising for the estimation of the covariance matrix:
model = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=EmpiricalPrior(
        mu_estimator=ShrunkMu(), covariance_estimator=DenoiseCovariance()
    ),
    portfolio_params=dict(name="Max Sharpe - ShrunkMu & DenoiseCovariance"),
)
model.fit(X_train)
model.weights_

# %%
# Benchmark
# =========
# For comparison, we also create a Maximum Sharpe Ratio model using the default
# moments estimators:
bench = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    portfolio_params=dict(name="Max Sharpe"),
)
bench.fit(X_train)
bench.weights_

# %%
# Prediction
# ==========
# We predict both models on the test set:
pred_model = model.predict(X_test)
pred_bench = bench.predict(X_test)

population = Population([pred_model, pred_bench])

fig = population.plot_cumulative_returns()
show(fig)
