"""
========================================================
Hierarchical Equal Risk Contribution with Graph Distance
========================================================

:class:`~skfolio.distance.GraphDistance` measures similarity between assets based on
their dependencies. This supports diversification across different underlying
vulnerabilities, even when returns are correlated.

This tutorial compares two
:class:`~skfolio.optimization.HierarchicalEqualRiskContribution` (HERC) models for a
portfolio of six electric car manufacturers. The first uses return correlations to
cluster the stocks. The second uses :class:`~skfolio.distance.GraphDistance`, based on
shared dependencies on battery, lithium and chip suppliers. Both models then allocate
weights using risks estimated from returns.
"""

# %%
# Returns
# =======
# We'll simulate returns for six stocks, Stock1 through Stock6, representing fictional
# electric car manufacturers, in 2024 and 2025. Each return combines a shared sector
# component with independent company noise:
from itertools import repeat

import numpy as np
import pandas as pd
from plotly.io import show
from sklearn import set_config
from sklearn.model_selection import train_test_split

from skfolio.containers import Graph, GraphSeries
from skfolio.distance import GraphDistance
from skfolio.optimization import HierarchicalEqualRiskContribution

set_config(enable_metadata_routing=True)

asset_names = ["Stock1", "Stock2", "Stock3", "Stock4", "Stock5", "Stock6"]
dates = pd.date_range("2024-01-01", "2025-12-31", freq="B")

rng = np.random.default_rng(0)
sector_returns = rng.normal(0.0002, 0.01, size=(len(dates), 1))
stock_returns = sector_returns + rng.normal(0, 0.005, size=(len(dates), 6))

X = pd.DataFrame(stock_returns, index=dates, columns=asset_names)
X.head()

# %%
# Dependency Graph
# ================
# Let's assume each company buys batteries from a different supplier. Several
# battery suppliers depend on the same lithium producer, while the manufacturers
# also share chip suppliers.
#
# A disruption at a lithium producer can therefore affect manufacturers that buy
# batteries from different companies. The graph records these direct and indirect
# dependencies.
#
# Edge weights express relative dependency strengths. We'll use illustrative weights
# of 3 for each manufacturer's battery dependency and 1 for its chip dependency.
# Each battery supplier's link to its lithium producer has weight 1:
graph = Graph.from_edges(
    edges=[
        ("Stock1", "Battery1", 3),
        ("Stock1", "ChipA", 1),
        ("Stock2", "Battery2", 3),
        ("Stock2", "ChipA", 1),
        ("Stock3", "Battery3", 3),
        ("Stock3", "ChipA", 1),
        ("Stock4", "Battery4", 3),
        ("Stock4", "ChipB", 1),
        ("Stock5", "Battery5", 3),
        ("Stock5", "ChipA", 1),
        ("Stock6", "Battery6", 3),
        ("Stock6", "ChipB", 1),
        ("Battery1", "LithiumA", 1),
        ("Battery2", "LithiumA", 1),
        ("Battery3", "LithiumA", 1),
        ("Battery4", "LithiumA", 1),
        ("Battery5", "LithiumB", 1),
        ("Battery6", "LithiumB", 1),
    ],
    nodes=[
        *asset_names,
        "Battery1",
        "Battery2",
        "Battery3",
        "Battery4",
        "Battery5",
        "Battery6",
        "LithiumA",
        "LithiumB",
        "ChipA",
        "ChipB",
    ],
)

# %%
# Let's plot the graph with a color for each type of node. Hovering over an edge shows
# its weight:
fig = graph.plot(
    groups={
        "Stock1": "Stocks",
        "Stock2": "Stocks",
        "Stock3": "Stocks",
        "Stock4": "Stocks",
        "Stock5": "Stocks",
        "Stock6": "Stocks",
        "Battery1": "Battery suppliers",
        "Battery2": "Battery suppliers",
        "Battery3": "Battery suppliers",
        "Battery4": "Battery suppliers",
        "Battery5": "Battery suppliers",
        "Battery6": "Battery suppliers",
        "LithiumA": "Lithium producers",
        "LithiumB": "Lithium producers",
        "ChipA": "Chip suppliers",
        "ChipB": "Chip suppliers",
    },
    title="Automotive Supply Chain",
)
show(fig)

# %%
# Relationships can change over time. A point-in-time (PIT) history records the graph
# known at each return date. We'll use a :class:`~skfolio.containers.GraphSeries` to
# keep this history aligned with `X`.
#
# Here, we'll assume the relationships were known throughout the sample and remained
# unchanged. Each observation uses the same graph, which `GraphSeries` stores only
# once:
graphs = GraphSeries(repeat(graph, len(X)), observations=X.index)

# %%
# Models
# ======
# We'll use the first 75% of observations for training and the remaining 25% for
# testing. Splitting returns and graphs together keeps them aligned. Setting
# `shuffle=False` preserves their chronological order:
X_train, X_test, graphs_train, graphs_test = train_test_split(
    X, graphs, test_size=0.25, shuffle=False
)

# %%
# The first model uses HERC's default Pearson distance:
default_model = HierarchicalEqualRiskContribution()
default_model.fit(X_train)
print(default_model.weights_)

# %%
# Stock2 receives 46.2% of the portfolio. Together, Stock1, Stock2 and Stock3 account for
# 68.0%. Although they buy from different battery suppliers, all three depend on
# LithiumA and ChipA.
#
# The second model uses graph distances. By default, dependencies up to two edges
# away are included, such as `Stock1 -> Battery1 -> LithiumA`.
# Metadata routing passes the training graphs to the distance estimator:
graph_model = HierarchicalEqualRiskContribution(distance_estimator=GraphDistance())
graph_model.fit(X_train, graphs=graphs_train)
print(graph_model.weights_)

# %%
# With graph distances, the combined weight of Stock1, Stock2 and Stock3 falls from
# 68.0% to 28.1%. Stock4, Stock5 and Stock6 each receive about 23% to 25%.
#
# The portfolio is less concentrated in companies that depend on both LithiumA and
# ChipA. Both models use the same returns to estimate risk, so this change comes from
# clustering stocks by their dependencies.
#
# Clustering
# ==========
# Let's compare the two dendrograms and distance heatmaps. The yellow boxes mark the
# clusters used by HERC:
fig = default_model.hierarchical_clustering_estimator_.plot_dendrogram()
fig.update_layout(title="HERC with Return Distances")

# %%
fig = graph_model.hierarchical_clustering_estimator_.plot_dendrogram()
fig.update_layout(title="HERC with Graph Distances")

# %%
# With return distances, Stock2 forms a cluster on its own, while the other five
# stocks form the second cluster.
#
# With graph distances, Stock1, Stock2 and Stock3 form a cluster because they share
# LithiumA and ChipA. Following the battery suppliers' dependencies identifies their
# common lithium producer. Stock4, Stock5 and Stock6 form separate clusters, each with
# a different combination of lithium and chip suppliers.
#
# Dependency Exposure
# ===================
# Let's compare supplier exposures for both portfolios using
# :meth:`~skfolio.portfolio.Portfolio.dependency_exposure`. A supplier's exposure is
# the combined initial weight of manufacturers that depend on it, directly or
# indirectly. Each stock is counted once per supplier, regardless of edge weights.
#
# For the model using return distances:
default_portfolio = default_model.predict(X_test)
default_portfolio.dependency_exposure(graph)

# %%
# For the model using graph distances:
graph_portfolio = graph_model.predict(X_test)
graph_portfolio.dependency_exposure(graph)

# %%
# LithiumA exposure falls from 79.0% to 52.9%, and ChipA exposure from 78.7% to 52.2%.
# Both lithium producers and both chip suppliers now have exposures close to 50%.
#
# The graph shows that companies with different battery suppliers can still depend
# on the same lithium producer. Using these dependencies in HERC reduces the
# portfolio's concentration in companies exposed to the same supplier disruption,
# even though their returns are correlated.
