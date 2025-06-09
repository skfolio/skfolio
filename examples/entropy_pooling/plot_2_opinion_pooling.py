r"""
===============
Opinion Pooling
===============

This tutorial introduces the :class:`~skfolio.prior.OpinionPooling` estimator.

Introduction
============

Opinion Pooling (also called Belief Aggregation or Risk Aggregation) is a process
in which different probability distributions (opinions), produced by different
experts, are combined to yield a single probability distribution (consensus).

Expert opinions (also called individual prior distributions) can be
**elicited** from domain experts or **derived** from quantitative analyses.

The `OpinionPooling` estimator takes a list of prior estimators, each of which
produces scenario probabilities (`sample_weight`), and pools them into a single
consensus probability .

You can choose between linear (arithmetic) pooling or logarithmic (geometric)
pooling, and optionally apply robust pooling using a Kullback-Leibler divergence
penalty to down-weight experts whose views deviate strongly from the group.

Linear Opinion Pooling
----------------------
* Retains all nonzero support: no "zero-forcing"
* Produces an averaging that is more evenly spread across all expert opinions.

Logarithmic Opinion Pooling
---------------------------
* Zero-Preservation: any scenario assigned zero probability by any expert
  remains zero in the aggregate.
* Information-Theoretic Optimality: yields the distribution that minimizes the
  weighted sum of KL-divergences from each expert's distribution.
* Robust to Extremes: down-weight extreme or contrarian views more severely.

Robust Pooling with Divergence Penalty
--------------------------------------
By specifying a `divergence_penalty`, you can penalize each opinion’s
divergence from the group consensus, yielding a more robust aggregate distribution.

In this tutorial, we will:
    1. Apply Opinion Pooling to historical return data.
    2. Construct portfolios based on the adjusted distribution.
    3. Demonstrate factor-based and synthetic-data-enhanced Opinion Pooling.
    4. Perform stress tests using Opinion Pooling.
"""


# %%
# Data Loading and Preparation
# ============================
# We load the S&P 500 :ref:`dataset <datasets>` and select seven stocks
# (for demonstration purposes). We also load the factors dataset, composed of
# daily prices for five ETFs representing common factors.

import numpy as np
import pandas as pd
from plotly.io import show

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.distribution import VineCopula
from skfolio.measures import (
    cvar,
    kurtosis,
    mean,
    skew,
    standard_deviation,
    value_at_risk,
)
from skfolio.optimization import HierarchicalRiskParity, RiskBudgeting
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EntropyPooling, FactorModel, OpinionPooling, SyntheticData
from skfolio.utils.figure import plot_kde_distributions

# Load stock price and factor data
prices = load_sp500_dataset()
prices = prices[["AMD", "BAC", "GE", "JNJ", "JPM", "LLY", "PG"]]
factor_prices = load_factors_dataset()

# Convert to daily returns
X, factors = prices_to_returns(prices, factor_prices)

print("Shapes:")
print(f"X: {X.shape}")
print(f"factors: {factors.shape}")

print(X.tail())
print(factors.tail())

# %%
# Summary Statistics
# ------------------
# We create a helper function to compute key return statistics, optionally weighted by
# sample probabilities:


def summary(X: pd.DataFrame, sample_weight: np.ndarray | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Mean": mean(X, sample_weight=sample_weight),
            "Volatility": standard_deviation(X, sample_weight=sample_weight),
            "Skew": skew(X, sample_weight=sample_weight),
            "Kurtosis": kurtosis(X, sample_weight=sample_weight),
            "VaR at 95%": value_at_risk(X, beta=0.95, sample_weight=sample_weight),
            "CVaR at 95%": cvar(X, beta=0.95, sample_weight=sample_weight),
        }
    )


summary(X)

# %%
# Expert Opinions
# ===============
# We consider two expert opinions, each generated via Entropy Pooling with user-defined
# views.
# We assign probabilities of 40% to Expert 1, 50% to Expert 2, and by default the
# remaining 10% is allocated to the prior distribution:

opinion_1 = EntropyPooling(cvar_views=["AMD == 0.10"])

opinion_2 = EntropyPooling(
    mean_views=["AMD >= BAC", "JPM <= prior(JPM) * 0.8"],
    cvar_views=["GE == 0.12"],
)

opinion_pooling = OpinionPooling(
    estimators=[("opinion_1", opinion_1), ("opinion_2", opinion_2)],
    opinion_probabilities=[0.4, 0.5],
)

opinion_pooling.fit(X)

sample_weight = opinion_pooling.return_distribution_.sample_weight
summary(X, sample_weight=sample_weight)

# %%
# Let's plot the prior versus the posterior returns distributions for each asset:
plot_kde_distributions(
    X,
    sample_weight=sample_weight,
    percentile_cutoff=0.05,
    title="Distribution of Asset Returns (Prior vs. Posterior)",
    unweighted_suffix="Prior",
    weighted_suffix="Posterior",
)

# %%
# Building a Portfolio based on Opinion Pooling
# =============================================
# Now that we've shown how the Opinion Pooling estimator works in isolation, let's
# see how to implement a risk parity portfolio with CVaR-90% as the risk measure based
# on Opinion Pooling:
model = RiskBudgeting(
    risk_measure=RiskMeasure.CVAR, cvar_beta=0.9, prior_estimator=opinion_pooling
)

model.fit(X)

print(model.weights_)

# %%
# Factor Opinion Pooling
# ======================
# Instead of applying Opinion Pooling directly to asset returns, we can embed it
# within a factor model so that expert views are expressed on the factors.

factor_opinion_1 = EntropyPooling(
    mean_views=["QUAL == -0.0005"], cvar_views=["SIZE == 0.08"]
)
factor_opinion_2 = EntropyPooling(cvar_views=["SIZE == 0.09"])

factor_opinion_pooling = OpinionPooling(
    estimators=[("opinion_1", factor_opinion_1), ("opinion_2", factor_opinion_2)],
    opinion_probabilities=[0.6, 0.4],
)

factor_model = FactorModel(factor_prior_estimator=factor_opinion_pooling)

model = RiskBudgeting(risk_measure=RiskMeasure.CVAR, prior_estimator=factor_model)


model.fit(X, factors)
print(model.weights_)

sample_weight = model.prior_estimator_.return_distribution_.sample_weight
summary(factors, sample_weight)

# %%
# Factor Opinion Pooling on Synthetic Data
# ========================================
# Rather than applying Option Pooling directly to a limited historical factor prior,
# we generate 100,000 synthetic factor returns using a Vine Copula. This synthetic
# dataset extrapolate the tail dependencies and allows more extreme EP views that were
# infeasible with sparse historical data:

vine = VineCopula(log_transform=True, n_jobs=-1, random_state=0)

factor_synth = SyntheticData(n_samples=100_000, distribution_estimator=vine)

factor_opinion_1 = EntropyPooling(cvar_views=["SIZE == 0.15"])
factor_opinion_2 = EntropyPooling(cvar_views=["SIZE == 0.20"])

factor_opinion_pooling = OpinionPooling(
    prior_estimator=factor_synth,
    estimators=[("opinion_1", factor_opinion_1), ("opinion_2", factor_opinion_2)],
    opinion_probabilities=[0.6, 0.4],
)

factor_model = FactorModel(factor_prior_estimator=factor_opinion_pooling)

model = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR, prior_estimator=factor_model
)

model.fit(X, factors)
print(model.weights_)

# %%
# Following scikit-learn conventions, all fitted attributes end with a trailing
# underscore. You can inspect each model step-by-step by drilling into these attributes:
fitted_vine = model.prior_estimator_.factor_prior_estimator_.prior_estimator_.distribution_estimator_

# %%
# Stress Test
# ===========
# Having demonstrated ex-ante Opinion Pooling (optimizing a portfolio based on specific
# views), we now apply ex-post Opinion Pooling to stress-test an existing portfolio.
# We start with a Hierarchical Risk Parity (HRP) portfolio using CVaR as the risk
# measure, optimized on historical data without Opinion Pooling:
model = HierarchicalRiskParity(risk_measure=RiskMeasure.CVAR)

model.fit(X)
print(model.weights_)

portfolio = model.predict(X)
portfolio.name = "HRP Unstressed"

# Add to a Population for better comparison with the stressed portfolios.
population = Population([portfolio])

# %%
# Create a Stressed Distribution
# ------------------------------
# Let's use Opinion Pooling on synthetic data by pooling two distinct expert views on
# AMD's CVaR:
vine = VineCopula(log_transform=True, n_jobs=-1, random_state=0)

synth = SyntheticData(n_samples=100_000, distribution_estimator=vine)

opinion_1 = EntropyPooling(cvar_beta=0.90, cvar_views=["AMD == 0.08"])
opinion_2 = EntropyPooling(cvar_views=["AMD == 0.10"])

opinion_pooling = OpinionPooling(
    prior_estimator=synth,
    estimators=[("opinion_1", opinion_1), ("opinion_2", opinion_2)],
    opinion_probabilities=[0.6, 0.4],
)

opinion_pooling.fit(X)

# We retrieve the stressed distribution:
stressed_dist = opinion_pooling.return_distribution_

# We stress-test our portfolio:
stressed_ptf = model.predict(stressed_dist)

# Add the stressed portfolio to the population
stressed_ptf.name = "HRP Stressed"
population.append(stressed_ptf)

# %%
# Now let's apply Factor Opinion Pooling to synthetic factor data by specifying two
# expert views on the CVaR of the quality factor (QUAL):
factor_synth = SyntheticData(n_samples=100_000, distribution_estimator=vine)

factor_opinion_1 = EntropyPooling(cvar_beta=0.90, cvar_views=["QUAL == 0.10"])
factor_opinion_2 = EntropyPooling(cvar_views=["QUAL == 0.12"])

factor_opinion_pooling = OpinionPooling(
    prior_estimator=factor_synth,
    estimators=[("opinion_1", factor_opinion_1), ("opinion_2", factor_opinion_2)],
    opinion_probabilities=[0.6, 0.4],
)

factor_model = FactorModel(factor_prior_estimator=factor_opinion_pooling)

factor_model.fit(X, factors)

# We retrieve the stressed distribution:
stressed_dist = factor_model.return_distribution_

# We stress-test our portfolio:
stressed_ptf = model.predict(stressed_dist)

# Add the stressed portfolio to the population
stressed_ptf.name = "HRP Factor Stressed"
population.append(stressed_ptf)

# %%
# Analysis of Unstressed vs Stressed Portfolios
# ---------------------------------------------
pop_summary = population.summary()
pop_summary.loc[
    [
        "Mean",
        "Standard Deviation",
        "CVaR at 95%",
        "Annualized Sharpe Ratio",
        "Worst Realization",
    ]
]

# %%
fig = population.plot_returns_distribution(percentile_cutoff=0.05)
show(fig)

# %%
# Conclusion
# ==========
# In this tutorial, we demonstrated how to leverage Opinion Pooling to aggregate
# multiple expert views into every stage of portfolio management, from ex-ante
# optimization to ex-post stress testing.

# %%
# References
# ==========
# [1] "Probabilistic opinion pooling generalized",
#      Social Choice and Welfare, Dietrich & List (2017)
#
# [2] "Opinion Aggregation and Individual Expertise",
#      Oxford University Press, Martini & Sprenger (2017)
#
# [3] "Rational Decisions",
#      Journal of the Royal Statistical Society, Good  (1952)
