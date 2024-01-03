"""
===========================================
Hierarchical Equal Risk Contribution - CDaR
===========================================

This tutorial introduces the
:class:`~skfolio.optimization.HierarchicalEqualRiskContribution` optimization.

The Hierarchical Equal Risk Contribution (HERC) is a portfolio optimization method
developed by Thomas Raffinot.

This algorithm uses a distance matrix to compute hierarchical clusters using the
Hierarchical Tree Clustering algorithm. It then computes, for each cluster, the
total cluster risk of an inverse-risk allocation.

The final step is the top-down recursive division of the dendrogram, where the
assets weights are updated using a naive risk parity within clusters.

It differs from the Hierarchical Risk Parity by exploiting the dendrogram shape
during the top-down recursive division instead of bisecting it.

.. note ::
    The default linkage method is set to the Ward variance minimization algorithm,
    which is more stable and has better properties than the single-linkage method

In this example, we will use the CDaR risk measure.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2020-01-02 up to 2022-12-28:
from plotly.io import show
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_sp500_dataset
from skfolio.distance import KendallDistance
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalEqualRiskContribution,
)
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.5, shuffle=False)

# %%
# Model
# =====
# We create a CVaR Hierarchical Equal Risk Contribution model and then fit it on the
# training set:
model1 = HierarchicalEqualRiskContribution(
    risk_measure=RiskMeasure.CDAR, portfolio_params=dict(name="HERC-CDaR-Ward-Pearson")
)
model1.fit(X_train)
model1.weights_

# %%
# Risk Contribution
# =================
# Let's analyze the risk contribution of the model on the training set:
ptf1 = model1.predict(X_train)
ptf1.plot_contribution(measure=RiskMeasure.CDAR)

# %%
# Dendrogram
# ==========
# To analyze the clusters structure, we plot the dendrogram.
# The blue lines represent distinct clusters composed of a single asset.
# The remaining colors represent clusters of more than one asset:
fig = model1.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=False)
show(fig)

# %%
# |
#
# The horizontal axis represents the assets. The links between clusters are represented
# as upside-down U-shaped lines. The height of the U indicates the distance between the
# clusters. For example, the link representing the cluster containing Assets HD and WMT
# has a distance of 0.5 (called cophenetic distance).


# %%
# When `heatmap` is set to True, the heatmap of the reordered distance matrix is
# displayed below the dendrogram and clusters are outlined with yellow squares:
model1.hierarchical_clustering_estimator_.plot_dendrogram()

# %%
# Linkage Methods
# ===============
# The clustering can be greatly affected by the choice of the linkage method.
# In the :class:`~skfolio.optimization.HierarchicalEqualRiskContribution` estimator, the
# default linkage method is set to the Ward variance minimization algorithm which is
# more stable and has better properties than the single-linkage method, which suffers
# from the chaining effect.
#
# And because HERC rely on the dendrogram structure as opposed
# to HRP, the choice of the linkage method will have a greater impact on the allocation.
#
# To show this effect, let's create a second model with the single-linkage method:
model2 = HierarchicalEqualRiskContribution(
    risk_measure=RiskMeasure.CDAR,
    hierarchical_clustering_estimator=HierarchicalClustering(
        linkage_method=LinkageMethod.SINGLE,
    ),
    portfolio_params=dict(name="HERC-CDaR-Single-Pearson"),
)
model2.fit(X_train)
model2.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=True)

# %%
# We can see that the clustering has been greatly affected by the change of the linkage
# method. Let's analyze the risk contribution of this model on the training set:
ptf2 = model2.predict(X_train)
ptf2.plot_contribution(measure=RiskMeasure.CDAR)

# %%
# The risk of that second model is very concentrated. We can already conclude that the
# single-linkage method is not appropriate for this dataset. This will be confirmed
# below on the test set.

# %%
# Distance Estimator
# ==================
# The distance metric used has also an important effect on the clustering.
# The default is to use the distance of the pearson correlation matrix.
# This can be changed using the :ref:`distance estimators <distance>`.
# For example, let's create a third model with a distance computed from the absolute
# value of the Kendal correlation matrix:
model3 = HierarchicalEqualRiskContribution(
    risk_measure=RiskMeasure.CDAR,
    distance_estimator=KendallDistance(absolute=True),
    portfolio_params=dict(name="HERC-CDaR-Ward-Kendal"),
)
model3.fit(X_train)
model3.hierarchical_clustering_estimator_.plot_dendrogram(heatmap=True)

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
for model in [model1, model2, model3, bench]:
    population_test.append(model.predict(X_test))

population_test.plot_cumulative_returns()

# %%
# Composition
# ===========
# From the below composition, we notice that the model with single-linkage method is
# highly concentrated:
population_test.plot_composition()
