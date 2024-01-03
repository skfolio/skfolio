r"""
============
Factor Model
============

This tutorial shows how to use the :class:`~skfolio.prior.FactorModel` estimator in
the :class:`~skfolio.optimization.MeanRisk` optimization.

A :ref:`prior estimator <prior>` fits a :class:`~skfolio.prior.PriorModel` containing
the distribution estimate of asset returns. It represents the investor's prior beliefs
about the model used to estimate such distribution.

The `PriorModel` is a dataclass containing:

    * `mu`: Expected returns estimation
    * `covariance`: Covariance matrix estimation
    * `returns`: assets returns estimation
    * `cholesky` : Lower-triangular Cholesky factor of the covariance estimation (optional)

The `FactorModel` estimator estimates the :class:`PriorModel` using a factor model and a
:ref:`prior estimator <prior>` of the factor's returns. The purpose of factor models is
to impose a structure on financial variables and their covariance matrix by explaining
them through a small number of common factors. This can help overcome estimation
error by reducing the number of parameters, i.e., the dimensionality of the estimation
problem, making portfolio optimization more robust against noise in the data. Factor
models also provide a decomposition of financial risk to systematic and security
specific components.

To be compatible with `scikit-learn`, the `fit` method takes `X` as the assets returns
and `y` as the factors returns. Note that `y` is in lowercase even for a 2D array
(more than one factor). This is for consistency with the scikit-learn API.

In this tutorial we will build a Maximum Sharpe Ratio portfolio using the `FactorModel`
estimator.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the SPX Index composition and the Factors dataset composed of the daily
# prices of 5 ETF representing common factors:
from plotly.io import show
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.moments import GerberCovariance, ShrunkMu
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior, FactorModel, LoadingMatrixRegression

prices = load_sp500_dataset()
factor_prices = load_factors_dataset()

prices = prices["2014":]
factor_prices = factor_prices["2014":]

X, y = prices_to_returns(prices, factor_prices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

# %%
# Factor Model
# =============
# We create a Maximum Sharpe Ratio model using the Factor Model that we fit on the
# training set:
model_factor_1 = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(),
    portfolio_params=dict(name="Factor Model 1"),
)
model_factor_1.fit(X_train, y_train)
model_factor_1.weights_

# %%
# We can change the :class:`~skfolio.prior.BaseLoadingMatrix` that estimates the loading
# matrix (betas) of the factors.
#
# The default is the :class:`LoadingMatrixRegression`, which fit the factors using a
# `LassoCV` on each asset separately.
#
# For example, let's change the `LassoCV` into a `RidgeCV` without intercept and use
# parallelization:
model_factor_2 = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(
        loading_matrix_estimator=LoadingMatrixRegression(
            linear_regressor=RidgeCV(fit_intercept=False), n_jobs=-1
        )
    ),
    portfolio_params=dict(name="Factor Model 2"),
)
model_factor_2.fit(X_train, y_train)
model_factor_2.weights_

# %%
# We can also change the :ref:`prior estimator <prior>` of the factors.
# It is used to estimate the :class:`~skfolio.prior.PriorModel` containing the factors
# expected returns and covariance matrix.
#
# For example, let's estimate the factors expected returns with James-Stein shrinkage
# and the factors covariance matrix with the Gerber covariance estimator:
model_factor_3 = MeanRisk(
    risk_measure=RiskMeasure.VARIANCE,
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    prior_estimator=FactorModel(
        factor_prior_estimator=EmpiricalPrior(
            mu_estimator=ShrunkMu(), covariance_estimator=GerberCovariance()
        )
    ),
    portfolio_params=dict(name="Factor Model 3"),
)
model_factor_3.fit(X_train, y_train)
model_factor_3.weights_

# %%
# Factor Analysis
# ===============
# Each fitted estimator is saved with a trailing underscore.
# For example, we can access the fitted prior estimator with:
prior_estimator = model_factor_3.prior_estimator_

# %%
# We can access the prior model with:
prior_model = prior_estimator.prior_model_

# %%
# We can access the loading matrix with:
loading_matrix = prior_estimator.loading_matrix_estimator_.loading_matrix_

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
# We predict all models on the test set:
ptf_factor_1_test = model_factor_1.predict(X_test)
ptf_factor_2_test = model_factor_2.predict(X_test)
ptf_factor_3_test = model_factor_3.predict(X_test)
ptf_empirical_test = model_empirical.predict(X_test)

population = Population(
    [ptf_factor_1_test, ptf_factor_2_test, ptf_factor_3_test, ptf_empirical_test]
)

fig = population.plot_cumulative_returns()
show(fig)

# %%
# |
#
# Let's plot the portfolios' composition:
population.plot_composition()
