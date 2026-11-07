"""Portfolio analytics and shared helpers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sps
import sklearn.utils.validation as skv
from scipy.sparse.csgraph import dijkstra

from skfolio.containers import Graph, GraphSeries
from skfolio.containers._utils import _as_string_labels
from skfolio.typing import AnyArray, ArrayLike, FloatArray, StrArray
from skfolio.utils.tools import _validate_positive_integer

if TYPE_CHECKING:
    from skfolio.portfolio._portfolio import Portfolio

__all__ = ["dependency_exposure"]

_MAX_ARRAY_ELEMENTS = 1_000_000


def dependency_exposure(
    graph: Graph | GraphSeries,
    *,
    asset_names: ArrayLike,
    weights: ArrayLike,
    max_hops: int | None = None,
    nodes: ArrayLike | None = None,
) -> pd.DataFrame:
    r"""Compute portfolio exposure to direct and indirect dependencies.

    The exposure to a node is the sum of portfolio weights of assets that depend on
    it, directly or indirectly, plus the portfolio weight of the node itself. An edge
    from A to B represents a dependency of A on B. Each asset is counted once per node.
    Any positive edge counts as a dependency, regardless of its weight.

    Parameters
    ----------
    graph : Graph or GraphSeries
        Dependency graph with nonnegative edge weights. For a `GraphSeries`, exposure
        is computed separately for each observation using its graph and weights.

    asset_names : array-like of shape (n_assets,)
        Unique asset names corresponding to `weights`. Names must be strings. Assets
        with nonzero weights must appear in the corresponding graph.

    weights : array-like of shape (n_assets,) or (n_observations, n_assets)
        Portfolio weights in the order of `asset_names`. A `Graph` requires one weight
        per asset. For a `GraphSeries`, a 1D array applies the same weights to every
        observation. A 2D array provides weights for each observation, in the order of
        `graph.observations`. Pandas index and column labels are not used. Weights are
        not normalized. Negative weights reduce exposure. Each row must contain only
        finite values or only NaN.

    max_hops : int, optional
        Maximum number of edges traversed when computing dependencies. Must be at
        least 1. If None, all direct and indirect dependencies are included.

    nodes : array-like of shape (n_nodes,), optional
        Unique string node names for which exposure is returned. If None, all graph
        nodes are included in order of first appearance. Paths can pass through other
        nodes. Required when every graph in a `GraphSeries` is None.

    Returns
    -------
    exposure : DataFrame
        Exposure to each node, with columns:

        - `weight`: portfolio weight of the node itself.
        - `direct`: sum of weights of other assets with an edge to the node.
        - `indirect`: sum of weights of other assets connected to the node only through
          intermediate nodes.
        - `total`: sum of `weight`, `direct` and `indirect`.

        Rows are indexed by `dependency` for a `Graph`, or by (`observation`,
        `dependency`) for a `GraphSeries`. Missing graphs or all-NaN weights give NaN
        exposure. A node absent from the graph at an observation has zero exposure.

    Raises
    ------
    ValueError
        If an asset with a nonzero weight is missing from its graph, or a requested
        node is missing from all graphs.

    Notes
    -----
    Let :math:`w_i` denote the weight of asset :math:`i`. The exposure to node
    :math:`j` is

    .. math::

        e_j = \sum_i w_i \mathbf{1}\{i = j \text{ or } i \text{ reaches } j\}.

    An asset's own weight is included only in `weight`, even when a path returns to
    that asset. The same asset can contribute to several nodes, so exposures across
    nodes need not sum to one. Absolute portfolio weights give gross exposure.

    Examples
    --------
    Token B depends on Token A, which depends on a protocol:

    >>> from skfolio.containers import Graph
    >>> from skfolio.portfolio import dependency_exposure
    >>> graph = Graph.from_edges(
    ...     [("Token A", "Protocol", 1), ("Token B", "Token A", 3)],
    ...     nodes=["Token A", "Token B", "Protocol"],
    ... )
    >>> exposure = dependency_exposure(
    ...     graph, asset_names=["Token A", "Token B"], weights=[0.6, 0.4]
    ... )
    >>> exposure
                weight  direct  indirect  total
    dependency
    Token A        0.6     0.4       0.0    1.0
    Token B        0.4     0.0       0.0    0.4
    Protocol       0.0     0.6       0.4    1.0
    """
    if not isinstance(graph, (Graph, GraphSeries)):
        raise TypeError("graph must be a Graph or GraphSeries.")

    if max_hops is not None:
        _validate_positive_integer(max_hops, "max_hops")

    asset_names = _as_string_labels(values=asset_names, name="asset_names")
    is_history = isinstance(graph, GraphSeries)
    n_observations = len(graph) if is_history else 1
    weights = skv.check_array(
        weights,
        dtype=np.float64,
        ensure_2d=False,
        ensure_min_samples=0,
        ensure_min_features=0,
        ensure_all_finite="allow-nan",
        input_name="weights",
    )

    if weights.shape != (len(asset_names),) and (
        not is_history or weights.shape != (n_observations, len(asset_names))
    ):
        raise ValueError(
            "weights must have shape (n_assets,), or "
            "(n_observations, n_assets) for a GraphSeries."
        )

    missing_weights = np.isnan(weights)
    missing_weight_rows = missing_weights.any(axis=-1)

    if np.any(missing_weight_rows & ~missing_weights.all(axis=-1)):
        raise ValueError("weights must be finite or contain an entirely NaN row.")

    missing_weight_rows = np.broadcast_to(missing_weight_rows, (n_observations,))
    weights = np.broadcast_to(weights, (n_observations, len(asset_names)))

    graph_observations: dict[Graph, list[int]] = {}
    for position, snapshot in enumerate(graph if is_history else [graph]):
        if snapshot is not None:
            graph_observations.setdefault(snapshot, []).append(position)

    all_nodes = np.asarray(
        list(
            dict.fromkeys(
                node for snapshot in graph_observations for node in snapshot.nodes
            )
        ),
        dtype=str,
    )

    if nodes is None:
        if n_observations and not graph_observations:
            raise ValueError("nodes is required when every graph is missing.")
        nodes = all_nodes
    else:
        nodes = _as_string_labels(values=nodes, name="nodes")
        if graph_observations:
            unknown = nodes[pd.Index(all_nodes).get_indexer(nodes) < 0]
            if len(unknown):
                raise ValueError(
                    f"Nodes absent from graphs ({len(unknown)}): "
                    f"{unknown[:5].tolist()}{'...' if len(unknown) > 5 else ''}."
                )

    contributions = np.full((n_observations, len(nodes), 3), np.nan)
    for snapshot, positions in graph_observations.items():
        positions = np.asarray(positions, dtype=np.intp)
        positions = positions[~missing_weight_rows[positions]]
        contributions[positions] = _graph_dependency_exposure(
            graph=snapshot,
            asset_names=asset_names,
            weights=weights[positions],
            nodes=nodes,
            max_hops=max_hops,
            observations=graph.observations[positions] if is_history else None,
        )

    index = (
        pd.MultiIndex.from_product(
            [graph.observations, nodes], names=["observation", "dependency"]
        )
        if is_history
        else pd.Index(nodes, name="dependency")
    )
    exposure = pd.DataFrame(
        contributions.reshape(-1, 3),
        index=index,
        columns=["weight", "direct", "indirect"],
    )
    exposure["total"] = contributions.sum(axis=2).ravel()

    return exposure


def _graph_dependency_exposure(
    *,
    graph: Graph,
    asset_names: StrArray,
    weights: FloatArray,
    nodes: StrArray,
    max_hops: int | None,
    observations: AnyArray | None,
) -> FloatArray:
    """Dependency exposure for observations sharing a graph.

    Parameters
    ----------
    graph : Graph
        Graph used for all weight rows.

    asset_names : StrArray of shape (n_assets,)
        Asset identifiers corresponding to the weights.

    weights : FloatArray of shape (n_observations, n_assets)
        Finite portfolio weights for the selected observations.

    nodes : StrArray of shape (n_nodes,)
        Nodes for which exposure is computed, including nodes absent from this graph.

    max_hops : int or None
        Maximum number of edges traversed. If None, all direct and indirect
        dependencies are included.

    observations : AnyArray of shape (n_observations,) or None
        Observation labels included in errors for missing assets. If None, errors do
        not include an observation label.

    Returns
    -------
    contributions : FloatArray of shape (n_observations, n_nodes, 3)
        Portfolio weight, direct and indirect contributions for each node.
    """
    adjacency = graph.adjacency
    edge_weights = adjacency.data if isinstance(adjacency, sps.csr_array) else adjacency
    if np.any(edge_weights < 0):
        raise ValueError("dependency_exposure requires nonnegative graph edge weights.")

    node_index = pd.Index(graph.nodes)
    asset_node_positions = node_index.get_indexer(asset_names)
    missing_assets = asset_node_positions < 0
    missing_holdings = weights[:, missing_assets] != 0
    if missing_holdings.any():
        row = np.flatnonzero(missing_holdings.any(axis=1))[0]
        missing_asset_names = asset_names[missing_assets][missing_holdings[row]]
        when = "" if observations is None else f" at observation {observations[row]!r}"
        raise ValueError(
            f"Assets absent from graph{when} ({len(missing_asset_names)}): "
            f"{missing_asset_names[:5].tolist()}"
            f"{'...' if len(missing_asset_names) > 5 else ''}."
        )

    contributions = np.zeros((len(weights), len(nodes), 3))
    held_assets = np.flatnonzero(np.any(weights != 0, axis=0))
    node_positions = node_index.get_indexer(nodes)
    report_positions = np.flatnonzero(node_positions >= 0)
    if not len(held_assets) or not len(report_positions):
        return contributions

    # CSR conversion preserves tiny positive edges that SciPy's dense input treats
    # as zero when constructing its graph.
    adjacency = sps.csr_array(adjacency)
    source_positions = asset_node_positions[held_assets]
    target_positions = node_positions[report_positions]

    # Starting at selected nodes in the reversed graph gives the same hop counts
    # to assets, with fewer traversals when few nodes are selected.
    reverse = len(target_positions) < len(source_positions)
    if reverse:
        adjacency = adjacency.T.tocsr()
        source_positions, target_positions = target_positions, source_positions

    chunk_size = max(1, _MAX_ARRAY_ELEMENTS // graph.n_nodes)
    for start in range(0, len(source_positions), chunk_size):
        chunk = slice(start, start + chunk_size)
        hop_counts = dijkstra(
            csgraph=adjacency,
            directed=True,
            indices=source_positions[chunk],
            return_predecessors=False,
            unweighted=True,
            limit=np.inf if max_hops is None else max_hops,
        )[:, target_positions]

        if reverse:
            hop_counts = hop_counts.T
            asset_indices = held_assets
            report_indices = report_positions[chunk]
        else:
            asset_indices = held_assets[chunk]
            report_indices = report_positions

        chunk_weights = weights[:, asset_indices]
        contributions[:, report_indices, 0] += chunk_weights @ (hop_counts == 0)
        contributions[:, report_indices, 1] += chunk_weights @ (hop_counts == 1)
        contributions[:, report_indices, 2] += chunk_weights @ (
            (hop_counts > 1) & np.isfinite(hop_counts)
        )

    return contributions


def _select_graph_observations(
    *, graphs: GraphSeries, observations: AnyArray
) -> GraphSeries:
    """Graphs selected by portfolio observation labels.

    Parameters
    ----------
    graphs : GraphSeries
        Graph history covering the requested observations.

    observations : AnyArray of shape (n_observations,)
        Portfolio observation labels in the order to return.

    Returns
    -------
    selected : GraphSeries
        Graph series matching the requested observations.

    Raises
    ------
    ValueError
        If an observation is missing, or `graphs.observations` contains duplicates and
        does not match `observations` in both values and order.
    """
    if np.array_equal(graphs.observations, observations):
        return graphs

    graph_index = pd.Index(graphs.observations)
    if not graph_index.is_unique:
        raise ValueError(
            "GraphSeries observations must be unique unless they match the "
            "portfolio observations exactly."
        )

    positions = graph_index.get_indexer(observations)
    if np.any(positions < 0):
        missing = observations[positions < 0]
        raise ValueError(
            f"Portfolio observations absent from GraphSeries ({len(missing)}): "
            f"{missing[:5].tolist()}{'...' if len(missing) > 5 else ''}."
        )

    return graphs[positions]  # ty: ignore[invalid-return-type]  # array selectors return GraphSeries


def _get_weights_per_observation(
    *,
    portfolios: Sequence[Portfolio],
    asset_names: StrArray,
    source: str = "target assets",
) -> FloatArray:
    """Asset weights at the start of each observation across portfolios.

    Parameters
    ----------
    portfolios : sequence of Portfolio
        Portfolios in the order to concatenate.

    asset_names : StrArray of shape (n_assets,)
        Target asset names defining the output columns.

    source : str, default="target assets"
        Label identifying the target assets in error messages.

    Returns
    -------
    weights : FloatArray of shape (n_observations, n_assets)
        Weights following each portfolio's `weight_drift` setting, including small and
        zero positions. Assets absent from a portfolio receive zero weight. Missing
        allocations give NaN rows.

    Raises
    ------
    ValueError
        If a portfolio with known weights contains an asset absent from `asset_names`.
    """
    n_observations = sum(portfolio.n_observations for portfolio in portfolios)
    weights = np.full((n_observations, len(asset_names)), np.nan)
    start = 0
    for portfolio in portfolios:
        stop = start + portfolio.n_observations
        if not np.isnan(portfolio.weights).any():
            weights[start:stop] = _align_weights(
                weights=portfolio._get_weights_per_observation(),
                asset_names=portfolio.assets,
                target_asset_names=asset_names,
                source=source,
            )
        start = stop
    return weights


def _align_weights(
    *,
    weights: FloatArray,
    asset_names: StrArray,
    target_asset_names: StrArray,
    source: str = "target assets",
) -> FloatArray:
    """Align portfolio weights with target asset names.

    Parameters
    ----------
    weights : FloatArray of shape (..., n_assets)
        Portfolio weights. The last axis corresponds to `asset_names`.

    asset_names : StrArray of shape (n_assets,)
        Asset identifiers corresponding to the weights.

    target_asset_names : StrArray of shape (n_target_assets,)
        Target asset identifiers in the desired order.

    source : str, default="target assets"
        Label identifying the target in error messages.

    Returns
    -------
    aligned : FloatArray of shape (..., n_target_assets)
        Weights in target asset order. Additional target assets receive zero.

    Raises
    ------
    ValueError
        If an input asset is absent from `target_asset_names`.
    """
    if np.array_equal(asset_names, target_asset_names):
        return weights
    positions = pd.Index(target_asset_names).get_indexer(asset_names)
    missing = asset_names[positions < 0]
    if len(missing):
        raise ValueError(
            f"Portfolio contains {len(missing)} asset(s) not in {source}: "
            f"{missing[:5].tolist()}{'...' if len(missing) > 5 else ''}."
        )
    aligned = np.zeros(
        (*weights.shape[:-1], len(target_asset_names)), dtype=weights.dtype
    )
    aligned[..., positions] = weights
    return aligned
