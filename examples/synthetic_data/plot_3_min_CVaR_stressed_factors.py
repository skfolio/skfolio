r"""
=================================
Minimize CVaR on Stressed Factors
=================================

This tutorial shows how to bridge scenario generation, factor models and portfolio
optimization.

In :ref:`the previous tutorial <sphx_glr_auto_examples_synthetic_data_plot_2_vine_copula.py>`,
we demonstrated how to generate conditional (stressed) synthetic returns using the
:class:`~skfolio.distribution.VineCopula` estimator.

Using the :class:`~skfolio.optimization.MeanRisk` optimization, you could directly
minimize the CVaR of your portfolio based on synthetic returns sampled from a given
model (Vine Copula, GAN, VAE, etc.).
However, in practice, we often need to perform cross-validation, portfolio rebalancing,
and hyperparameter tuning. To facilitate this, we require a unified model that
integrates synthetic data generation and optimization. This is exactly the role of
the :class:`~skfolio.prior.SyntheticData` estimator, which bridges scenario generation,
factor models and portfolio optimization.

There are several reasons why you might choose to run optimization on (factor) synthetic
data rather than (factor) historical data:

* Historical data is often limited, especially in the tails, which can make it
  challenging to model extreme events accurately. Using parametric copulas to explicitly
  capture tail dependencies allows for better extrapolation of joint extreme events.
  By generating a larger sample of returns from Vine Copulas, you improve the accuracy
  of capturing tail co-dependencies during the optimization process.

* Build portfolios optimized for specific stressed scenarios.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the SPX Index composition and the Factors dataset composed of the daily
# prices of 5 ETFs representing common factors.
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.distribution import VineCopula
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import FactorModel, SyntheticData

prices = load_sp500_dataset()
factor_prices = load_factors_dataset()

X, factors = prices_to_returns(prices, factor_prices)
X_train, X_test, factors_train, factors_test = train_test_split(
    X, factors, test_size=0.33, shuffle=False
)
print(factors_train.tail())

# %%
print("Shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"factors_train: {factors_train.shape}")
print(f"factors_test: {factors_test.shape}")


# %%
# Minimize CVaR on Synthetic Data
# ===============================
# Let's find the minimum CVaR portfolio on 10,000 synthetic returns generated from
# Vine Copula fitted on the historical training set and evaluate it on the historical
# test set.
vine = VineCopula(log_transform=True, n_jobs=-1, random_state=0)
prior = SyntheticData(distribution_estimator=vine, n_samples=10_000)
model = MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=prior)

model.fit(X_train)
print(model.weights_)
ptf = model.predict(X_test)
# You can then perform a full analysis using the portfolio methods.

# %%
# Multi-period Portfolio
# ======================
# Now let's run a walk-forward analysis where we optimize the minimum CVaR portfolio on
# synthetic data generated from a Vine Copula fitted on one year (252 business days) of
# historical data and evaluate it on the following 3 months (60 business days) of data,
# repeating over the full history.
cv = WalkForward(train_size=252, test_size=60)
ptf = cross_val_predict(model, X_train, cv=cv)
ptf.summary()

# %%
# Combining Synthetic Data with Factor Model
# ==========================================
# Now, let's add another layer of complexity by incorporating a Factor Model while
# stressing the quality factor (QUAL) by -20%.
# The model fits a Factor Model on historical data, then fits a Vine Copula on the
# factor data, samples 10,000 stressed scenarios from the Vine, and finally projects
# these scenarios back to the asset universe using the Factor Model.
vine = VineCopula(
    log_transform=True, central_assets=["QUAL"], n_jobs=-1, random_state=0
)
factor_prior = SyntheticData(
    distribution_estimator=vine,
    n_samples=10_000,
    sample_args=dict(conditioning={"QUAL": -0.2}),
)
factor_model = FactorModel(factor_prior_estimator=factor_prior)

model = MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=factor_model)
model.fit(X_train, factors_train)
print(model.weights_)

ptf = model.predict(X_test)

# %%
# Let's show how to drill down into the model to retrieve the fitted Vine Copula and
# plot the marginal distributions of the stressed factors alongside the historical data.
# The stressed Momentum (MTUM), Size (SIZE), Low Volatility (USMV), and Value (VLUE)
# factors deviate significantly from their unstressed distributions, reflecting the
# impact of stressing the Quality (QUAL) factor.
# Note that the stressed distribution of the Quality factor is a Dirac, since only -20%
# was sampled.
fitted_vine = model.prior_estimator_.factor_prior_estimator_.distribution_estimator_
fig = fitted_vine.plot_marginal_distributions(factors, conditioning={"QUAL": -0.2})
show(fig)

# %%
# Factor Stress Test
# ==================
# Finally, let's stress-test the portfolio by further stressing the quality factor by
# -50%.
factor_model.set_params(
    factor_prior_estimator__sample_args=dict(conditioning={"QUAL": -0.5})
)
# Refit the factor model on the full dataset to update the stressed scenarios
factor_model.fit(X, factors)
stressed_dist = factor_model.return_distribution_

stressed_ptf = model.predict(stressed_dist)

ptf.name = "Unstressed Ptf"
stressed_ptf.name = "Stressed Ptf"
population = Population([ptf, stressed_ptf])
summary = population.summary()
summary.loc[
    ["Mean", "Standard Deviation", "CVaR at 95%", "EVaR at 95%", "Worst Realization"]
]

# %%
population.plot_returns_distribution(percentile_cutoff=0.1)

# %%
# Conclusion
# ==========
# In this tutorial, we demonstrated how to bridge scenario generation, factor models,
# and portfolio optimization.
