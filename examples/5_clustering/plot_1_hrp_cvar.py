"""
===============================
Hierarchical Risk Parity - CVaR
===============================

This tutorial introduces the :class:`~skfolio.optimization.HierarchicalRiskParity`
optimization.

Hierarchical Risk Parity (HRP) is a portfolio optimization method developed by Marcos
Lopez de Prado.

This algorithm uses a distance matrix to compute hierarchical clusters using the
Hierarchical Tree Clustering algorithm. It then employs seriation to rearrange the
assets in the dendrogram, minimizing the distance between leafs.

The final step is the recursive bisection where each cluster is split between two
sub-clusters by starting with the topmost cluster and traversing in a top-down
manner. For each sub-cluster, we compute the total cluster risk of an inverse-risk
allocation. A weighting factor is then computed from these two sub-cluster risks,
which is used to update the cluster weight.

.. note ::
    The original paper uses the variance as the risk measure and the single-linkage
    method for the Hierarchical Tree Clustering algorithm. Here we generalize it to
    multiple risk measures and linkage methods.
    The default linkage method is set to the Ward
    variance minimization algorithm, which is more stable and has better properties
    than the single-linkage method.

In this example, we will use the CVaR risk measure.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the SPX Index composition and the Factors dataset composed of the daily
# prices of 5 ETF representing common factors:
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.distance import KendallDistance
from skfolio.optimization import EqualWeighted, HierarchicalRiskParity
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import FactorModel

prices = load_sp500_dataset()
factor_prices = load_factors_dataset()

prices = prices["2014":]
factor_prices = factor_prices["2014":]

X, y = prices_to_returns(prices, factor_prices)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create the CVaR Hierarchical Risk Parity model and then fit it on the training set:
model1 = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR, portfolio_params=dict(name="HRP-CVaR-Ward-Pearson")
)
model1.fit(X_train)
model1.weights_

# %%
# Risk Contribution
# =================
# Let's analyze the risk contribution of the model on the training set:
ptf1 = model1.predict(X_train)
ptf1.plot_contribution(measure=RiskMeasure.CVAR)

# %%
# Dendrogram
# ==========
# To analyze the clusters structure, we plot the dendrogram.
# The blue lines represent distinct clusters composed of a single asset.
# The remaining colors represent clusters of more than one asset:
model1.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=False)

# %%
# The horizontal axis represents the assets. The links between clusters are represented
# as upside-down U-shaped lines. The height of the U indicates the distance between the
# clusters. For example, the link representing the cluster containing assets HD and WMT
# has a distance of 0.5 (called cophenetic distance).

# %%
#  When `heatmap` is set to True, the heatmap of the reordered distance matrix is
#  displayed below the dendrogram and clusters are outlined with yellow squares:
fig = model1.hierarchical_clustering_estimator_.plot_dendrogram()
show(fig)

# %%
# Linkage Methods
# ===============
# The clustering can be greatly affected by the choice of the linkage method.
# The original HRP is based on the single-linkage (equivalent to the minimum spanning
# tree), which suffers from the chaining effect.
# In the :class:`~skfolio.optimization.HierarchicalRiskParity` estimator, the default
# linkage method is set to the Ward variance minimization algorithm, which is more
# stable and has better properties than the single-linkage method.
#
# However, since the HRP optimization doesn’t utilize the full cluster structure but
# only their orders, the allocation remains relatively stable regardless of the chosen
# linkage method.

# To show this effect, let's create a second model with the single-linkage method:
model2 = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR,
    hierarchical_clustering_estimator=HierarchicalClustering(
        linkage_method=LinkageMethod.SINGLE,
    ),
    portfolio_params=dict(name="HRP-CVaR-Single-Pearson"),
)
model2.fit(X_train)

model2.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=True)

# %%
# We can see that the clustering has been greatly affected by the change of the linkage
# method. However, you will see bellow that the weights remain relatively stable for the
# reason explained earlier.

# %%
# Distance Estimator
# ==================
# The choice of distance metric has also an important effect on the clustering.
# The default is to use the distance from the pearson correlation matrix.
# This can be changed using the :ref:`distance estimators <distance>`.
#
# For example, let's create a third model with a distance computed from the absolute
# value of the Kendal correlation matrix:
model3 = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR,
    distance_estimator=KendallDistance(absolute=True),
    portfolio_params=dict(name="HRP-CVaR-Ward-Kendal"),
)
model3.fit(X_train)

model3.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=True)

# %%
# Prior Estimator
# ===============
# Finally, HRP like the other portfolio optimization, uses a
# :ref:`prior estimator <prior>` that fits a :class:`~skfolio.prior.PriorModel`
# containing the distribution estimate of asset returns. It represents the investor's
# prior beliefs about the model used to estimate such distribution.
# The default is the :class:`~skfolio.prior.EmpiricalPrior` estimator.
#
# Let's create new model with the :class:`~skfolio.prior.FactorModel` estimator:
model4 = HierarchicalRiskParity(
    risk_measure=RiskMeasure.CVAR,
    prior_estimator=FactorModel(),
    portfolio_params=dict(name="HRP-CVaR-Factor-Model"),
)
model4.fit(X_train, y_train)

model4.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=True)


# %%
# To compare the models, we use an equal weighted benchmark using
# the :class:`~skfolio.optimization.EqualWeighted` estimator:
bench = EqualWeighted()
bench.fit(X_train)
bench.weights_

# %%
# Prediction
# ==========
# We predict the models and the benchmark on the test set:
population_test = Population([])
for model in [model1, model2, model3, model4, bench]:
    population_test.append(model.predict(X_test))

population_test.plot_cumulative_returns()

# %%
# Composition
# ===========
# From the below composition, we notice that all models are relatively close to each
# others as explain earlier:
population_test.plot_composition()

# %%
# Summary
# =======
# Finally, let's print the summary statistics:
summary = population_test.summary()
summary.loc["Annualized Sharpe Ratio"]

# %% Full summary:
summary
