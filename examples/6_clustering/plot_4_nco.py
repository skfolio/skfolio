"""
============================
Nested Clusters Optimization
============================

This tutorial introduces the :class:`~skfolio.optimization.NestedClustersOptimization`
optimization.

Nested Clusters Optimization (NCO) is a portfolio optimization method developed by
Marcos Lopez de Prado.

It uses a distance matrix to compute clusters using a clustering algorithm (
Hierarchical Tree Clustering, KMeans, etc..). For each cluster, the inner-cluster
weights are computed by fitting the inner-estimator on each cluster using the whole
training data. Then the outer-cluster weights are computed by training the
outer-estimator using out-of-sample estimates of the inner-estimators with
cross-validation. Finally, the final assets weights are the dot-product of the
inner-weights and outer-weights.

.. note ::

    The original paper uses KMeans as the clustering algorithm, minimum Variance for
    the inner-estimator and equal-weight for the outer-estimator. Here we generalize
    it to all `sklearn` and `skfolio` clustering algorithm (Hierarchical Tree
    Clustering, KMeans, etc.), all portfolio optimizations (Mean-Variance, HRP, etc.)
    and risk measures (variance, CVaR, etc.).
    To avoid data leakage at the outer-estimator, we use out-of-sample estimates to
    fit the outer estimator.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 1990-01-02 up to 2022-12-28:
from plotly.io import show
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_sp500_dataset
from skfolio.distance import KendallDistance
from skfolio.optimization import (
    EqualWeighted,
    MeanRisk,
    NestedClustersOptimization,
    ObjectiveFunction,
    RiskBudgeting,
)
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

# %%
# Model
# =====
# We create a NCO model that maximizes the Sharpe Ratio intra-cluster and uses a CVaR
# Risk Parity inter-cluster. By default, the inter-cluster optimization
# uses `KFolds` out-of-sample estimates of the inner-estimator to avoid data leakage.
# and the :class:`~skfolio.cluster.HierarchicalClustering` estimator
# to form the clusters:
inner_estimator = MeanRisk(
    objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
    risk_measure=RiskMeasure.VARIANCE,
)
outer_estimator = RiskBudgeting(risk_measure=RiskMeasure.CVAR)

model1 = NestedClustersOptimization(
    inner_estimator=inner_estimator,
    outer_estimator=outer_estimator,
    n_jobs=-1,
    portfolio_params=dict(name="NCO-1"),
)
model1.fit(X_train)
model1.weights_

# %%
# Dendrogram
# ==========
# To analyze the clusters structure, we can plot the dendrogram.
# The blue lines represent distinct clusters composed of a single asset.
# The remaining colors represent clusters of more than one asset:
model1.clustering_estimator_.plot_dendrogram(heatmap=False)

# %%
# The horizontal axis represent the assets. The links between clusters are represented
# as upside-down U-shaped lines. The height of the U indicates the distance between the
# clusters. For example, the link representing the cluster containing Assets HD and WMT
# has a distance of 0.5 (called cophenetic distance).

# %%
#  When `heatmap` is set to True, the heatmap of the reordered distance matrix is
#  displayed below the dendrogram and clusters are outlined with yellow squares:
model1.clustering_estimator_.plot_dendrogram()

# %%
# Linkage Methods
# ===============
# The hierarchical clustering can be greatly affected by the choice of the linkage
# method. In the :class:`~skfolio.cluster.HierarchicalClustering` estimator, the default
# linkage method is set to the Ward variance minimization algorithm, which is more
# stable and has better properties than the single-linkage method which suffers from the
# chaining effect.
#
# To show this effect, let's create a second model with the
# single-linkage method:
model2 = NestedClustersOptimization(
    inner_estimator=inner_estimator,
    outer_estimator=outer_estimator,
    clustering_estimator=HierarchicalClustering(
        linkage_method=LinkageMethod.SINGLE,
    ),
    n_jobs=-1,
    portfolio_params=dict(name="NCO-2"),
)
model2.fit(X_train)
model2.clustering_estimator_.plot_dendrogram(heatmap=True)

# %%
# Distance Estimator
# ==================
# The distance metric used has also an important effect on the clustering.
# The default is to use the distance of the pearson correlation matrix.
# This can be changed using the :ref:`distance estimators <distance>`.
#
# For example, let's create a third model with a distance computed from the absolute
# value of the Kendal correlation matrix:
model3 = NestedClustersOptimization(
    inner_estimator=inner_estimator,
    outer_estimator=outer_estimator,
    distance_estimator=KendallDistance(absolute=True),
    n_jobs=-1,
    portfolio_params=dict(name="NCO-3"),
)
model3.fit(X_train)
model3.clustering_estimator_.plot_dendrogram(heatmap=True)

# %%
# Clustering Estimator
# ====================
# The above models used the default :class:`~skfolio.cluster.HierarchicalClustering`
# estimator. This can be replaced by any `sklearn` or `skfolio` clustering estimators.
#
# For example, let's create a new model with `sklearn.cluster.KMeans`:
model4 = NestedClustersOptimization(
    inner_estimator=inner_estimator,
    outer_estimator=outer_estimator,
    clustering_estimator=KMeans(n_init="auto"),
    n_jobs=-1,
    portfolio_params=dict(name="NCO-4"),
)
model4.fit(X_train)
model4.weights_

# %%
# To compare the NCO models, we use an equal weighted benchmark using
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
# Let's plot each portfolio composition:
fig = population_test.plot_composition()
show(fig)
