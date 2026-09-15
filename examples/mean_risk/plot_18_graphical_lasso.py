r"""Graphical Lasso Covariance Estimation.
==========================================

This tutorial shows how to use
:class:`~skfolio.moments.GraphicalLassoCV` as the covariance estimator of a
:class:`~skfolio.optimization.MeanRisk` optimization.

The empirical covariance matrix can be unstable when the number of assets is large
relative to the number of observations. Graphical Lasso [1]_ regularizes the inverse
covariance, also called the precision matrix, by solving an optimization problem of
the form:

.. math::

    \underset{\Theta \succ 0}{\operatorname{minimize}}\;
    \operatorname{tr}(S\Theta) - \log\det(\Theta)
    + \alpha \lVert\Theta\rVert_1,

where :math:`S` is the empirical covariance and :math:`\Theta` is the precision
matrix. The L1 penalty encourages zeros in :math:`\Theta`. Under a multivariate
Gaussian model, a zero off-diagonal entry means that two assets are conditionally
independent given all the other assets.

:class:`~skfolio.moments.GraphicalLassoCV` selects the regularization parameter
:math:`\alpha` by cross-validation and exposes both the regularized covariance in
`covariance_` and the sparse inverse covariance in `precision_`.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>`, convert prices to linear returns, and
# split the observations chronologically to avoid :ref:`data leakage <data_leakage>`.

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.io import show
from sklearn.model_selection import TimeSeriesSplit, train_test_split

from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.moments import EmpiricalCovariance, GraphicalLassoCV, LedoitWolf
from skfolio.optimization import MeanRisk
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Sparse conditional-dependence structure
# =======================================
# We use a chronological inner cross-validation to select `alpha`. An explicit,
# compact grid keeps the tutorial fast and reproducible. Larger research workflows
# can use a wider grid or the estimator's iterative grid refinement.

graphical_lasso = GraphicalLassoCV(
    alphas=[3e-6, 1e-5, 3e-5],
    cv=TimeSeriesSplit(n_splits=5),
    max_iter=300,
)
graphical_lasso.fit(X_train)

precision = graphical_lasso.precision_
off_diagonal = ~np.eye(precision.shape[0], dtype=bool)
sparsity = np.mean(np.isclose(precision[off_diagonal], 0.0))

print(f"Selected alpha: {graphical_lasso.alpha_:.2e}")
print(f"Zero off-diagonal precision entries: {sparsity:.1%}")

# %%
# The partial correlation between assets :math:`i` and :math:`j`, conditional on all
# other assets, is obtained from the precision matrix as
#
# .. math::
#
#     \rho_{ij \mid -ij} =
#     -\frac{\Theta_{ij}}{\sqrt{\Theta_{ii}\Theta_{jj}}}.
#
# The heatmap makes the sparse conditional-dependence structure easier to interpret
# than the raw precision matrix.

precision_scale = np.sqrt(np.outer(np.diag(precision), np.diag(precision)))
partial_correlation = -precision / precision_scale
np.fill_diagonal(partial_correlation, 1.0)
partial_correlation = pd.DataFrame(
    partial_correlation,
    index=X_train.columns,
    columns=X_train.columns,
)

fig = px.imshow(
    partial_correlation,
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu_r",
    title="Graphical Lasso Partial Correlations",
)
show(fig)

# %%
# Sparse precision versus conic optimization
# ===========================================
# :class:`~skfolio.optimization.MeanRisk` uses the estimator's `covariance_` to define
# portfolio variance. At the exact graphical-lasso optimum, the covariance and
# precision are inverses. The fitted numerical pair is nevertheless not an API
# guarantee of exact inversion, so skfolio does not silently replace `covariance_` with
# an inverse reconstructed from `precision_`. The precision is useful for statistical
# interpretation, but its zeros do not generally make `covariance_` sparse. Moreover,
# a Cholesky factor can introduce fill-in even when the precision itself is sparse.
# skfolio therefore preserves the covariance-estimation benefit of Graphical Lasso,
# while representing the risk with a Cholesky factor of `covariance_` in its
# second-order cone program.
#
# The following diagnostic compares lower-triangular non-zero counts. It is only a
# matrix-structure diagnostic: it does not imply a reduction in CVXPY cone dimensions,
# solver variables, or solve time.

covariance_cholesky = np.linalg.cholesky(graphical_lasso.covariance_)
precision_cholesky = np.linalg.cholesky(precision)
print(
    "Lower-triangular non-zeros "
    f"(precision / precision Cholesky / covariance Cholesky): "
    f"{np.count_nonzero(np.tril(precision))} / "
    f"{np.count_nonzero(precision_cholesky)} / "
    f"{np.count_nonzero(covariance_cholesky)}"
)

# We compare Graphical Lasso with empirical covariance and Ledoit-Wolf shrinkage. All
# three models solve the same long-only minimum-variance problem.

models = {
    "Empirical": MeanRisk(
        prior_estimator=EmpiricalPrior(covariance_estimator=EmpiricalCovariance()),
        portfolio_params=dict(name="Empirical"),
    ),
    "Ledoit-Wolf": MeanRisk(
        prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf()),
        portfolio_params=dict(name="Ledoit-Wolf"),
    ),
    "Graphical Lasso CV": MeanRisk(
        prior_estimator=EmpiricalPrior(covariance_estimator=graphical_lasso),
        portfolio_params=dict(name="Graphical Lasso CV"),
    ),
}

portfolios = []
rows = []
for name, model in models.items():
    model.fit(X_train)
    portfolio = model.predict(X_test)
    portfolios.append(portfolio)

    fitted_covariance = model.prior_estimator_.covariance_estimator_
    rows.append(
        {
            "Estimator": name,
            "Covariance test score": fitted_covariance.score(X_test),
            "Annualized variance": portfolio.annualized_variance,
            "Annualized Sharpe ratio": portfolio.annualized_sharpe_ratio,
            "Effective number of assets": portfolio.effective_number_assets,
        }
    )

comparison = pd.DataFrame(rows).set_index("Estimator")
print(comparison)

# %%
# `BaseCovariance.score` is the held-out Gaussian log-likelihood, for which higher is
# better. Portfolio measures evaluate a different downstream objective. Consequently,
# the covariance estimator with the best likelihood need not produce the portfolio
# with the lowest realized variance on one historical split.
#
# We inspect the allocations and cumulative returns without treating this single split
# as evidence that one estimator is universally superior.

population = Population(portfolios)
population.plot_composition()

# %%
fig = population.plot_cumulative_returns()
show(fig)

# %%
# References
# ==========
# .. [1] "Sparse inverse covariance estimation with the graphical lasso",
#         Jerome Friedman, Trevor Hastie, and Robert Tibshirani (2008)
