r"""
===============
Empirical Prior
===============

This tutorial shows how to use the :class:`~skfolio.prior.EmpiricalPrior` estimator in
the :class:`~skfolio.optimization.MeanRisk` optimization.

A :ref:`Prior Estimator <prior>` in `skfolio` fits a :class:`ReturnDistribution`
containing your pre-optimization inputs (:math:`\mu`, :math:`\Sigma`, returns, sample
weight, Cholesky decomposition).

The term "prior" is used in a general optimization sense, not confined to Bayesian
priors. It denotes any **a priori** assumption or estimation method for the return
distribution before optimization, unifying both **Frequentist**, **Bayesian** and
**Information-theoretic** approaches into a single cohesive framework:

1. Frequentist:
    * :class:`~skfolio.prior.EmpiricalPrior`
    * :class:`~skfolio.prior.FactorModel`
    * :class:`~skfolio.prior.SyntheticData`

2. Bayesian:
    * :class:`~skfolio.prior.BlackLitterman`

3. Information-theoretic:
    * :class:`~skfolio.prior.EntropyPooling`
    * :class:`~skfolio.prior.OpinionPooling`

In skfolio's API, all such methods share the same interface and adhere to scikit-learn's
estimator API: the `fit` method accepts `X` (the asset returns) and stores the
resulting :class:`~skfolio.prior.ReturnDistribution` in its `return_distribution_`
attribute.

The :class:`~skfolio.prior.ReturnDistribution` is a dataclass containing:

    * `mu`: Estimated expected returns of shape (n_assets,)
    * `covariance`: Estimated covariance matrix of shape (n_assets, n_assets)
    * `returns`: (Estimated) asset returns of shape (n_observations, n_assets)
    * `sample_weight` : Sample weight for each observation of shape (n_observations,) (optional)
    * `cholesky` : Lower-triangular Cholesky factor of the covariance (optional)

The `EmpiricalPrior` estimator estimates the `ReturnDistribution` by fitting its
`mu_estimator` and `covariance_estimator` independently.

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
