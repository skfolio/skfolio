r"""
=================
Black & Litterman
=================

This tutorial shows how to use the :class:`~skfolio.prior.BlackLitterman` estimator in
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

The `BlackLitterman` estimator estimates the `ReturnDistribution` using
the Black & Litterman model. It takes a Bayesian approach by starting from a prior
estimate of the assets' expected returns and covariance matrix, then updating them with
the analyst's views to obtain the posterior estimates.

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
