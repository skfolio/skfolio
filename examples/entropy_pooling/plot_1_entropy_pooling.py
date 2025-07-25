r"""
===============
Entropy Pooling
===============

This tutorial introduces the :class:`~skfolio.prior.EntropyPooling` estimator.

Introduction
============

Entropy Pooling, introduced by Attilio Meucci in 2008 as a generalization of the
Black-Litterman framework, is a nonparametric method for adjusting a baseline ("prior")
probability distribution to incorporate user-defined views by finding the posterior
distribution closest to the prior while satisfying those views.

User-defined views can be **elicited** from domain experts or **derived** from
quantitative analyses.

Grounded in information theory, it updates the distribution in the least-informative
way by minimizing the Kullback-Leibler divergence (relative entropy) under the
specified view constraints.

Mathematically, the problem is formulated as:

.. math::

   \begin{aligned}
   \min_{\mathbf{q}} \quad & \sum_{i=1}^T q_i \log\left(\frac{q_i}{p_i}\right) \\
   \text{subject to} \quad & \sum_{i=1}^T q_i = 1 \quad \text{(normalization constraint)} \\
                           & \mathbb{E}_q[f_j(X)] = v_j \quad(\text{or } \le v_j, \text{ or } \ge v_j), \quad j = 1,\dots,k, \text{(view constraints)} \\
                           & q_i \ge 0, \quad i = 1, \dots, T
   \end{aligned}

Where:

- :math:`T` is the number of observations (number of scenarios).
- :math:`p_i` is the prior probability of scenario :math:`x_i`.
- :math:`q_i` is the posterior probability of scenario :math:`x_i`.
- :math:`X` is the scenario matrix of shape (n_observations, n_assets).
- :math:`f_j` is the j :sup:`th` view function.
- :math:`v_j` is the target value imposed by the j :sup:`th` view.
- :math:`k` is the total number of views.

The `skfolio` implementation supports the following views:
    * Equalities
    * Inequalities
    * Ranking
    * Linear combinations (e.g. relative views)
    * Views on groups of assets

On the following measures:
    * Mean
    * Variance
    * Skew
    * Kurtosis
    * Correlation
    * Value-at-Risk (VaR)
    * Conditional Value-at-Risk (CVaR)

Entropy Pooling re-weights the sample probabilities of the prior distribution and is
therefore constrained by the support (completeness) of that distribution. For example,
if the historical distribution contains no returns below -10% for a given asset, we
cannot impose a CVaR view of 15%: no matter how we adjust the sample probabilities,
such tail data do not exist.

Therefore, to impose extreme views on a sparse historical distribution, one must
generate synthetic data. In that case, the EP posterior is only as reliable as the
synthetic scenarios. It is thus essential to use a generator capable of extrapolating
tail dependencies, such as :class:`~skfolio.distribution.VineCopula`, to model joint
extreme events accurately.

In general, for extreme stress tests, it is recommended to use conditional sampling from
:class:`~skfolio.distribution.VineCopula` (see the previous tutorial
:ref:`sphx_glr_auto_examples_synthetic_data_plot_2_vine_copula.py`). However, when
conditional sampling does not provide sufficient granularity, one can combine Entropy
Pooling with Vine Copula, as demonstrated at the end of this tutorial.

In this tutorial, we will:
    1. Apply Entropy Pooling to historical return data.
    2. Construct portfolios based on the adjusted distribution.
    3. Demonstrate factor-based and synthetic-data-enhanced Entropy Pooling.
    4. Perform stress tests using Entropy Pooling.
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
    correlation,
    cvar,
    kurtosis,
    mean,
    skew,
    standard_deviation,
    value_at_risk,
)
from skfolio.optimization import HierarchicalRiskParity, RiskBudgeting
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EntropyPooling, FactorModel, SyntheticData
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
            "VaR at 90%": value_at_risk(X, beta=0.90, sample_weight=sample_weight),
            "CVaR at 90%": cvar(X, beta=0.90, sample_weight=sample_weight),
        }
    )


print(f"Corr(BAC, JPM): {correlation(X[['BAC', 'JPM']])[0][1]:.2%}")
summary(X)

# %%
# Specifying Views for Entropy Pooling
# =====================================
# Let's add the following views to demonstrate the API capabilities (they are not
# based on realistic economic assumptions):
#
# Mean Views
# ----------
# * The daily mean return of JPM equals -0.20%
# * The mean return of PG is greater than that of LLY (ranking view)
# * The mean return of BAC increases by at least 20% (relative to its prior)
# * The sum of mean returns for Financials assets equals twice the sum for Growth assets (group views)
#
# Variance Views
# --------------
# * The volatility of BAC doubles (relative to its prior)
#
# Correlation Views
# -----------------
# * The correlation between BAC and JPM equals 80%
# * The correlation between BAC and JNJ decreases by at least 50% (versus its prior)
#
# Skew Views
# ----------
# * The skew of BAC equals -0.05
#
# CVaR Views
# ----------
# * The CVaR at 90% of GE equals 7%
#
# Finally we specify asset groupings by sector and style.

groups = {
    "AMD": ["Technology", "Growth"],
    "BAC": ["Financials", "Value"],
    "GE": ["Industrials", "Value"],
    "JNJ": ["Healthcare", "Defensive"],
    "JPM": ["Financials", "Income"],
    "LLY": ["Healthcare", "Defensive"],
    "PG": ["Consumer", "Defensive"],
}

entropy_pooling = EntropyPooling(
    mean_views=[
        "JPM == -0.002",
        "PG >= LLY",
        "BAC >= prior(BAC) * 1.2",
        "Financials == 2 * Growth",
    ],
    variance_views=[
        "BAC == prior(BAC) * 4",
    ],
    correlation_views=[
        "(BAC,JPM) == 0.80",
        "(BAC,JNJ) <= prior(BAC,JNJ) * 0.5",
    ],
    skew_views=[
        "BAC == -0.05",
    ],
    cvar_views=[
        "GE == 0.07",
    ],
    cvar_beta=0.90,
    groups=groups,
)
entropy_pooling.fit(X)

print(f"Relative Entropy : {entropy_pooling.relative_entropy_:.2f}")
print(
    f"Effective Number of Scenarios : {entropy_pooling.effective_number_of_scenarios_:.0f}"
)


# %%
# The Effective Number of Scenarios quantifies how concentrated or diverse the
# posterior distribution (`sample_weight`) is after imposing views.
# It reflects how many scenarios are meaningfully contributing to the distribution
# and is defined as the exponential of the Shannon entropy of the posterior
# probabilities. As opposed to the relative entropy which measures
# how much the posterior deviated from the prior, the Effective Number of Scenarios
# only measures the diversity (entropy) of the posterior distribution.
#
# In Entropy Pooling, what are commonly named "posterior probabilities" are
# saved in `sample_weight` in the `ReturnDistribution` DataClass to be used by the
# optimization estimators.
# Let's now analyze the Entropy Pooling results:
sample_weight = entropy_pooling.return_distribution_.sample_weight

print(f"sample_weight Shape: {sample_weight.shape}")
print(
    f"Corr(BAC, JPM): {correlation(X[['BAC', 'JPM']], sample_weight=sample_weight)[0][1]:.2%}"
)
summary(X, sample_weight=sample_weight)

# %%
# We note that all views have been respected.
#
# Let's plot the prior versus the posterior returns distributions for each asset:
fig = plot_kde_distributions(
    X,
    sample_weight=sample_weight,
    percentile_cutoff=0.1,
    title="Distribution of Asset Returns (Prior vs. Posterior)",
    unweighted_suffix="Prior",
    weighted_suffix="Posterior",
)
show(fig)

# %%
# Building a Portfolio based on Entropy Pooling
# =============================================
# Now that we've shown how the Entropy Pooling estimator works in isolation, let's
# see how to implement a risk parity portfolio with CVaR-90% as the risk measure based
# on EP:
bench = RiskBudgeting(risk_measure=RiskMeasure.CVAR, cvar_beta=0.9)
model = RiskBudgeting(
    risk_measure=RiskMeasure.CVAR, cvar_beta=0.9, prior_estimator=entropy_pooling
)

bench.fit(X)
model.fit(X)

print(bench.weights_)
print(model.weights_)

# %%
# We notice that the weight on GE is lower in the portfolio based on EP versus the
# benchmark, reflecting that GE's tail risk was the most impacted by our views.
#
# Note that instead of :class:`~skfolio.optimization.RiskBudgeting`, Entropy Pooling is
# also compatible with the other :ref:`portfolio optimization <optimization>` methods
# such as  :class:`~skfolio.optimization.MeanRisk`,
# :class:`~skfolio.optimization.HierarchicalRiskParity` etc.

# %%
# Comparing Risk Contributions
# ----------------------------
# A CVaR risk-parity portfolio assigns weights so that each asset contributes the same
# amount to the portfolio's CVaR.
#
# Therefore, as shown in the contribution graphs below:
#
# * The benchmark has equal CVaR contributions under the **prior** distribution
# * The EP portfolio has equal CVaR contributions under the **posterior** distribution
#
sample_weight = model.prior_estimator_.return_distribution_.sample_weight

portfolio_bench = bench.predict(X)
portfolio_bench.name = "Benchmark (Optimized on prior)"
portfolio_ep = model.predict(X)
portfolio_ep.name = "Optimized on EP posterior"

# %%
# Backtest using the Prior Distribution
# -------------------------------------
population = Population([portfolio_bench, portfolio_ep])
population.plot_contribution(measure=RiskMeasure.CVAR)

# %%
# Backtest using the EP Posterior Distribution
# --------------------------------------------
population.set_portfolio_params(sample_weight=sample_weight)
population.plot_contribution(measure=RiskMeasure.CVAR)

# %%
# Factor Entropy Pooling
# ======================
# Instead of applying Entropy Pooling directly to asset returns, we can embed it
# within a Factor Model.
# This allows us to impose views on factor data such at the quality factor "QUAL":

factor_entropy_pooling = EntropyPooling(mean_views=["QUAL == 0.0005"])

factor_model = FactorModel(factor_prior_estimator=factor_entropy_pooling)

model = RiskBudgeting(risk_measure=RiskMeasure.CVAR, prior_estimator=factor_model)

model.fit(X, factors)
print(model.weights_)

sample_weight = model.prior_estimator_.return_distribution_.sample_weight
summary(factors, sample_weight)

# %%
# Factor Entropy Pooling on Synthetic Data
# ========================================
# Rather than applying Entropy Pooling directly to a limited historical factor prior,
# we generate 100,000 synthetic factor returns using a Vine Copula. This synthetic
# dataset extrapolate the tail dependencies and allows more extreme EP views that were
# infeasible with sparse historical data:

vine = VineCopula(log_transform=True, n_jobs=-1, random_state=0)

factor_synth = SyntheticData(n_samples=100_000, distribution_estimator=vine)

factor_entropy_pooling = EntropyPooling(
    prior_estimator=factor_synth,
    cvar_beta=0.9,
    cvar_views=["QUAL == 0.10"],
)

factor_model = FactorModel(factor_prior_estimator=factor_entropy_pooling)

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
# Having demonstrated ex-ante Entropy Pooling (optimizing a portfolio based on specific
# views), we now apply ex-post Entropy Pooling to stress-test an existing portfolio.
# We start with a Hierarchical Risk Parity (HRP) portfolio using CVaR as the risk
# measure, optimized on historical data without EP:
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
# Let's use Entropy Pooling on Synthetic Data by imposing a CVaR-90% view on AMD of
# 10%:
vine = VineCopula(log_transform=True, n_jobs=-1, random_state=0)

synth = SyntheticData(n_samples=100_000, distribution_estimator=vine)

entropy_pooling = EntropyPooling(
    prior_estimator=synth, cvar_beta=0.90, cvar_views=["AMD == 0.10"]
)

entropy_pooling.fit(X)

# We retrieve the stressed distribution:
stressed_dist = entropy_pooling.return_distribution_

# We stress-test our portfolio:
stressed_ptf = model.predict(stressed_dist)

# Add the stressed portfolio to the population
stressed_ptf.name = "HRP Stressed"
population.append(stressed_ptf)

# %%
# Now Let's use Factor Entropy Pooling on Factor Synthetic Data by imposing a
# CVaR-90% view on the quality factor "QUAL" of 10%:
factor_synth = SyntheticData(n_samples=100_000, distribution_estimator=vine)

factor_entropy_pooling = EntropyPooling(
    prior_estimator=factor_synth,
    cvar_beta=0.90,
    cvar_views=["QUAL == 0.10"],
)

factor_model = FactorModel(factor_prior_estimator=factor_entropy_pooling)

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
population.plot_returns_distribution(percentile_cutoff=0.0500)


# %%
# Conclusion
# ==========
# In this tutorial, we demonstrated how to leverage Entropy Pooling to integrate  views
# into every stage of portfolio management, from ex-ante optimization to ex-post
# stress testing.

# %%
# References
# ==========
#  [1] "Fully Flexible Extreme Views",
#      Journal of Risk, Meucci, Ardia & Keel (2011)
#
#  [2] "Fully Flexible Views: Theory and Practice",
#      Risk, Meucci (2013).
#
#  [3] "Effective Number of Scenarios in Fully Flexible Probabilities",
#      GARP Risk Professional, Meucci (2012)
#
#  [4] "I-Divergence Geometry of Probability Distributions and Minimization
#      Problems", The Annals of Probability, Csiszar (1975)
