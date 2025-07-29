"""
==============================
Schur Complementary Allocation
==============================

This tutorial introduces the :class:`~skfolio.optimization.SchurComplementary`
optimization.

Schur Complementary Allocation is a portfolio allocation method developed by Peter 4
Cotton [1]_.

It uses Schur-complement-inspired augmentation of sub-covariance matrices,
revealing a link between Hierarchical Risk Parity (HRP) and minimum-variance (MVO)
portfolios.

By tuning the regularization factor `gamma`, which governs how much off-diagonal
information is incorporated into the augmented covariance blocks, the method
smoothly interpolates from the heuristic divide-and-conquer allocation of HRP
(`gamma = 0`) to the exact MVO solution (`gamma -> 1`).

.. note ::
    A poorly conditioned covariance matrix can produce unstable or extreme portfolio
    weights and prevent convergence to the true MVO solution as gamma approaches one.
    To improve numerical stability and estimation accuracy, apply shrinkage or other
    conditioning techniques via the `prior_estimator`.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the SPX Index composition and the Factors dataset composed of the daily
# prices of 5 ETF representing common factors:
from plotly.io import show
import numpy as np
from sklearn.model_selection import train_test_split

from skfolio import Population, PerfMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_sp500_dataset, load_ftse100_dataset
from skfolio.distance import KendallDistance
from skfolio.prior import EmpiricalPrior
from skfolio.moments import ShrunkCovariance, LedoitWolf, GerberCovariance, GraphicalLassoCV
from skfolio.optimization import (
    MeanRisk,
    EqualWeighted,
    HierarchicalRiskParity,
    SchurComplementary,
)
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior, FactorModel

prices = load_sp500_dataset()
prices=prices["2010":]
prices = prices[["AMD", "BAC", "GE", "JNJ", "JPM", "LLY"]]

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.3, shuffle=False)


model = SchurComplementary()
model.fit(X)
print(model.weights_)

prior = EmpiricalPrior(covariance_estimator=LedoitWolf())

population = Population([])

for gamma in np.linspace(0.0,1.0,20):
    schur = SchurComplementary(gamma=gamma,prior_estimator=prior)
    # Train
    ptf = schur.fit_predict(X_train)
    ptf.tag = f"Schur"
    ptf.name = f"Schur {gamma:0.8f}"
    population.append(ptf)


hrp = HierarchicalRiskParity(prior_estimator=prior)
# Train
ptf = hrp.fit_predict(X_train)
ptf.tag = "HRP"
population.append(ptf)


mean_variance = MeanRisk(prior_estimator=prior,   efficient_frontier_size=30, max_standard_deviation=0.01,
               portfolio_params={"tag": "Mean-Variance"})
# Train
mv_population = mean_variance.fit_predict(X_train)
mv_population[0].tag = "MVO"
population +=  mv_population


fig = population.plot_measures(
    x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    y=PerfMeasure.ANNUALIZED_MEAN,
    title="Mean Variance Efficient Frontier - HRP - Schur"
)

fig.show()


