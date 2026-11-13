.. _distance:

.. currentmodule:: skfolio.distance

******************
Distance Estimator
******************

A :ref:`distance estimator <distance_ref>` estimates codependence and distance
matrices between assets for clustering and portfolio optimization.

It follows scikit-learn's estimator API: the `fit` method stores these matrices in
the `codependence_` and `distance_` attributes.


Return Dependence
=================

For return-based distances, `X` contains asset returns and can be any array-like
structure (NumPy array, pandas DataFrame, etc.). These estimators measure how
returns move together:

* :class:`PearsonDistance` measures linear dependence.
* :class:`KendallDistance` and :class:`SpearmanDistance` measure dependence through
  the ranks of returns.
* :class:`CovarianceDistance` derives correlations from a chosen covariance estimator
  or a precomputed covariance matrix.
* :class:`DistanceCorrelation` and :class:`MutualInformation` capture nonlinear
  dependence.

**Example:**

.. code-block:: python

    from skfolio.datasets import load_sp500_dataset
    from skfolio.distance import PearsonDistance
    from skfolio.optimization import HierarchicalRiskParity
    from skfolio.preprocessing import prices_to_returns

    # Load prices and convert them to asset returns.
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    # Estimate and inspect the codependence and distance matrices.
    model = PearsonDistance()
    model.fit(X)
    print(model.codependence_)
    print(model.distance_)

    # Use Pearson distances to construct an HRP portfolio.
    model = HierarchicalRiskParity(distance_estimator=PearsonDistance())
    model.fit(X)
    print(model.weights_)


.. _graph_distances:

Graph Dependencies
==================

:class:`GraphDistance` measures similarity between assets based on their dependencies.
This supports diversification across underlying vulnerabilities, including risks
that have not materialized historically and are not captured by return correlations.

For example:

* **Stocks.** Two car manufacturers buying batteries from the same supplier are
  both exposed to disruptions at that supplier.
* **Crypto.** Two tokens supported by the same protocol can both be affected by a
  vulnerability in its shared contracts.

These relationships can be represented as `manufacturer -> supplier` or
`token -> protocol` in a directed graph.

The estimator compares direct and indirect dependencies, including their relative
weights. Small distances indicate similar dependencies. In the stock example,
the manufacturers have distance zero if each depends only on the shared supplier.
Two assets with no shared dependencies have distance one. The resulting matrix can
be used for clustering and portfolio optimization. See :ref:`graph_based_allocation`
for how HRP, HERC, Schur and NCO use these distances.

For this estimator, `X` is a pandas DataFrame of asset returns with unique string
column names and at least one asset. Only its column names are used to identify
assets. The `graphs` parameter accepts a
:class:`~skfolio.containers.GraphSeries`. Estimation uses the graph at the final
training observation.

The example below uses a static graph with relationships known throughout the sample:

.. code-block:: python

    from itertools import repeat

    import numpy as np
    import pandas as pd
    from sklearn import set_config

    from skfolio.containers import Graph, GraphSeries
    from skfolio.distance import GraphDistance
    from skfolio.optimization import HierarchicalRiskParity

    set_config(enable_metadata_routing=True)

    X = pd.DataFrame(
        np.random.default_rng(0).normal(0, 0.01, (100, 3)),
        columns=["A", "B", "C"],
    )
    graph = Graph.from_edges(
        [
            ("A", "supplier", 1),
            ("B", "supplier", 1),
            ("C", "material", 1),
            ("supplier", "material", 1),
        ],
        nodes=["A", "B", "C", "supplier", "material"],
    )
    graphs = GraphSeries(repeat(graph, len(X)), observations=X.index)

    model = HierarchicalRiskParity(distance_estimator=GraphDistance())
    model.fit(X, graphs=graphs)
    print(model.weights_)

For changing relationships, supply one graph per observation in the same order as
`X`, using only information available at that date. Reuse the last known graph until
an update is available. Cross-validation slices the series alongside `X`, so each
fold uses its final training graph. Any graph construction or weighting learned
from observations must be fitted within the training fold.

See :ref:`sphx_glr_auto_examples_clustering_plot_graph_distance.py` for a comparison
of return and graph distances in HERC, including clustering and portfolio weights.

Online updates through HRP or Schur use `model.partial_fit(X_batch, graphs=graphs_batch)`.
The graph series describes the new observations, and distances use its final graph.
The prior must support `partial_fit`.

With a :class:`~skfolio.prior.CharacteristicsFactorModel` prior, pass both inputs
to the optimizer: `model.fit(X, characteristics=panel, graphs=graphs)`.
