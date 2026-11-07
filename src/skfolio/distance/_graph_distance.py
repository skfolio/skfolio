"""Graph distance estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# Credits: Matteo Manzi, Ojas Arora
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import ClassVar, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sps
import sklearn.utils as sku
import sklearn.utils.validation as skv
from sklearn.utils.extmath import row_norms

from skfolio.containers import GraphSeries
from skfolio.distance._base import BaseDistance
from skfolio.distance._utils import _corr_to_distance
from skfolio.typing import CSRArray, FloatArray, IntArray
from skfolio.utils.tools import (
    _validate_bool,
    _validate_positive_integer,
    _validate_unit_interval,
)
from skfolio.utils.validation import _asset_positions

_FITTED_ATTR = "distance_"


class GraphDistance(BaseDistance):
    r"""Graph Distance estimator.

    The codependence is computed from cosine similarities between the assets'
    dependency profiles. The distance is then computed as :math:`\sqrt{1-S}`, where
    :math:`S` is the codependence matrix.

    A dependency profile contains the weights of an asset's direct and indirect
    dependencies. An edge from :math:`i` to :math:`j` represents a dependency of
    :math:`i` on :math:`j`. For adjacency matrix :math:`M`, after optional row
    normalization, the profiles are the asset rows of:

    .. math::

        \sum_{h=1}^{H} \alpha^{h-1} M^h

    where :math:`H` = `max_hops` and :math:`\alpha` = `decay`.

    Parameters
    ----------
    max_hops : int, default=2
        Maximum number of edges traversed when computing dependencies. A value of 1
        includes only direct dependencies and 2 also includes their dependencies. Must
        be at least 1. Nodes can be revisited through cycles.

    decay : float, default=0.5
        Weight multiplier for each additional hop, in :math:`[0, 1]`. A value of zero
        includes only direct dependencies.

    normalize_adjacency : bool, default=True
        If True, the outgoing weights of each node are normalized to sum to one before
        propagation. If False, the original edge weights are used.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix of cosine similarities between dependency profiles, with
        values in :math:`[0, 1]`.

    distance_ : ndarray of shape (n_assets, n_assets)
        Euclidean distances between dependency profiles normalized to unit length,
        divided by :math:`\sqrt{2}`. Values are in :math:`[0, 1]`, with a zero diagonal
        for assets with positive outgoing edge weights.

    n_features_in_ : int
        Number of assets seen during fitting.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during fitting.

    Notes
    -----
    Estimation uses the graph at the final observation. Graph observations are assumed
    to be chronological, with each graph containing only information available at that
    observation.

    Assets absent from the graph or without positive outgoing edges have NaN rows and
    columns in both fitted matrices, including their diagonals. The prior determines
    which assets are investable. Negative edge weights are not supported.

    For `max_hops=1` or `decay=0`, only direct dependencies are included and row
    normalization has no effect on the distances. Indirect dependencies can include
    graph nodes outside the investment universe.

    See Also
    --------
    skfolio.containers.Graph : Weighted directed graph with named nodes.
    skfolio.containers.GraphSeries : Graphs indexed by observation.

    Examples
    --------
    Asset B depends on two suppliers, with weights 3 and 1. It shares one supplier
    with A and the other with C:

    >>> import pandas as pd
    >>> from skfolio.containers import Graph, GraphSeries
    >>> from skfolio.distance import GraphDistance
    >>> X = pd.DataFrame({"A": [0.01], "B": [-0.02], "C": [0.03]})
    >>> graph = Graph.from_edges(
    ...     [
    ...         ("A", "Supplier 1", 1),
    ...         ("B", "Supplier 1", 3),
    ...         ("B", "Supplier 2", 1),
    ...         ("C", "Supplier 2", 1),
    ...     ],
    ...     nodes=["A", "B", "C", "Supplier 1", "Supplier 2"],
    ... )
    >>> graphs = GraphSeries([graph], observations=X.index)
    >>> model = GraphDistance().fit(X, graphs=graphs)
    >>> model.distance_.round(3)
    array([[0.   , 0.227, 1.   ],
           [0.227, 0.   , 0.827],
           [1.   , 0.827, 0.   ]])
    """

    __metadata_request__fit: ClassVar[dict[str, bool]] = {"graphs": True}
    __metadata_request__partial_fit: ClassVar[dict[str, bool]] = {"graphs": True}

    def __init__(
        self,
        *,
        max_hops: int = 2,
        decay: float = 0.5,
        normalize_adjacency: bool = True,
    ) -> None:
        self.max_hops = max_hops
        self.decay = decay
        self.normalize_adjacency = normalize_adjacency

    def fit(
        self, X: pd.DataFrame, y: None = None, *, graphs: GraphSeries | None = None
    ) -> GraphDistance:
        """Fit the Graph Distance estimator.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_observations, n_assets)
            Asset returns with unique string column names identifying assets in the
            graph. At least one asset is required. Return values are not used and can
            contain NaN.

        y : Ignored
            Not used, present for API consistency by convention.

        graphs : GraphSeries, required
            Graphs for the training observations. The graph at the final observation
            must be present. Passing `graphs` through an optimizer or cross-validation
            requires scikit-learn metadata routing to be enabled.

        Returns
        -------
        self : GraphDistance
            Fitted estimator.
        """
        self._reset()
        return self.partial_fit(X, y, graphs=graphs)

    def partial_fit(
        self, X: pd.DataFrame, y: None = None, *, graphs: GraphSeries | None = None
    ) -> GraphDistance:
        """Update distances using the latest graph.

        Parameters
        ----------
        X : pandas.DataFrame of shape (n_observations, n_assets)
            New asset returns with unique string column names identifying assets in the
            graph. Columns must match previous calls. Return values are not used and
            can contain NaN.

        y : Ignored
            Not used, present for API consistency by convention.

        graphs : GraphSeries, required
            Graphs for the new observations. The graph at the final observation must
            be present. Passing `graphs` through an optimizer requires scikit-learn
            metadata routing to be enabled.

        Returns
        -------
        self : GraphDistance
            Updated estimator.
        """
        _validate_positive_integer(self.max_hops, "max_hops")
        _validate_unit_interval(self.decay, "decay")
        _validate_bool(self.normalize_adjacency, "normalize_adjacency")

        if graphs is None:
            raise ValueError(
                "GraphDistance requires graphs=. When fitting through an optimizer "
                "or cross-validation, enable metadata routing with "
                "sklearn.set_config(enable_metadata_routing=True) and pass graphs=."
            )
        if not isinstance(graphs, GraphSeries):
            raise TypeError("graphs must be a GraphSeries.")
        if not len(graphs):
            raise ValueError("graphs must contain at least one observation.")
        graph = graphs[-1]
        if graph is None:
            raise ValueError("graphs has no graph at the last observation.")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame.")

        first_call = not hasattr(self, _FITTED_ATTR)
        skv.validate_data(self, X, reset=first_call, skip_check_array=True)
        if not hasattr(self, "feature_names_in_"):
            raise ValueError("X must have string column names.")

        asset_node_positions = _asset_positions(
            names=self.feature_names_in_,
            coverage=graph.nodes,
            source="graph",
            allow_missing=True,
        )

        adjacency = graph.adjacency
        edge_weights = (
            adjacency.data if isinstance(adjacency, sps.csr_array) else adjacency
        )
        if np.any(edge_weights < 0):
            raise ValueError(
                "GraphDistance requires non-negative weights throughout the graph."
            )

        n_assets = self.n_features_in_
        in_graph = asset_node_positions >= 0
        matched_node_positions = asset_node_positions[in_graph]
        has_dependencies = np.zeros(n_assets, dtype=bool)

        if isinstance(adjacency, sps.csr_array):
            # Graph omits zero-weight edges, so nonempty rows have positive weights.
            has_dependencies[in_graph] = (
                adjacency.indptr[matched_node_positions + 1]
                > adjacency.indptr[matched_node_positions]
            )
        else:
            has_dependencies[in_graph] = adjacency[matched_node_positions].any(axis=1)

        if not has_dependencies.any():
            self.codependence_ = np.full((n_assets, n_assets), np.nan)
            self.distance_ = np.full_like(self.codependence_, np.nan)
            return self

        similarity = _graph_similarity(
            adjacency=adjacency,
            asset_node_positions=asset_node_positions[has_dependencies],
            max_hops=self.max_hops,
            decay=self.decay,
            normalize_adjacency=self.normalize_adjacency,
        )
        codependence, distance = _corr_to_distance(similarity, absolute=True, power=1)
        if has_dependencies.all():
            self.codependence_ = codependence
            self.distance_ = distance
        else:
            asset_indices = np.ix_(has_dependencies, has_dependencies)
            self.codependence_ = np.full((n_assets, n_assets), np.nan)
            self.distance_ = np.full_like(self.codependence_, np.nan)
            self.codependence_[asset_indices] = codependence
            self.distance_[asset_indices] = distance

        return self

    def _reset(self) -> None:
        """Reset fitted state for a new learning run."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def __sklearn_tags__(self) -> sku.Tags:
        """Allow missing return values because only asset names are used."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags


def _normalize_graph_rows(
    matrix: FloatArray | CSRArray, *, norm: Literal["l1", "l2"]
) -> None:
    """Normalize nonnegative rows in place.

    Parameters
    ----------
    matrix : ndarray or scipy.sparse.csr_array of shape (n_rows, n_nodes)
        Writable float64 matrix of finite nonnegative weights. Zero rows remain zero.

    norm : {"l1", "l2"}
        Normalize rows to sum to one with "l1", or to unit Euclidean length with "l2".

    Notes
    -----
    Rows are first divided by their largest entry to avoid overflow or underflow when
    computing the norms.
    """
    if isinstance(matrix, sps.csr_array):
        row_lengths = np.diff(matrix.indptr)
        maxima = matrix.max(axis=1).toarray().ravel()
        matrix.data /= np.repeat(np.where(maxima > 0, maxima, 1), row_lengths)
        scales = (
            row_norms(matrix)
            if norm == "l2"
            else np.asarray(matrix.sum(axis=1)).ravel()
        )
        matrix.data /= np.repeat(np.where(scales > 0, scales, 1), row_lengths)
    else:
        maxima = matrix.max(axis=1, keepdims=True)
        matrix /= np.where(maxima > 0, maxima, 1)
        scales = (
            row_norms(matrix)[:, None]
            if norm == "l2"
            else matrix.sum(axis=1, keepdims=True)
        )
        matrix /= np.where(scales > 0, scales, 1)


def _graph_similarity(
    *,
    adjacency: FloatArray | CSRArray,
    asset_node_positions: IntArray,
    max_hops: int,
    decay: float,
    normalize_adjacency: bool,
) -> FloatArray:
    """Compute cosine similarities between propagated dependency profiles.

    Parameters
    ----------
    adjacency : ndarray or scipy.sparse.csr_array of shape (n_nodes, n_nodes)
        Adjacency matrix with finite nonnegative weights. The input is not modified.

    asset_node_positions : ndarray of shape (n_assets,)
        Positions of assets with positive outgoing weights in the adjacency matrix.

    max_hops : int
        Maximum number of edges traversed, at least one.

    decay : float
        Weight multiplier for each additional hop, between zero and one inclusive.

    normalize_adjacency : bool
        Whether to normalize outgoing weights to sum to one before propagation.

    Returns
    -------
    similarity : ndarray of shape (n_assets, n_assets)
        Cosine similarities of dependency profiles, with a unit diagonal.

    Raises
    ------
    ValueError
        If propagation overflows float64 arithmetic.
    """
    if max_hops == 1 or decay == 0:
        # Row scaling does not change cosine similarity of direct dependencies, so
        # only asset rows are converted.
        dependency_weights = adjacency[asset_node_positions].astype(
            np.float64, copy=False
        )
    else:
        adjacency = adjacency.astype(np.float64, copy=normalize_adjacency)
        if normalize_adjacency:
            _normalize_graph_rows(adjacency, norm="l1")

        dependency_weights = adjacency[asset_node_positions]
        hop_weights = dependency_weights.copy()
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(1, max_hops):
                hop_weights = hop_weights @ adjacency
                hop_weights *= decay
                dependency_weights += hop_weights
                dependency_values = (
                    dependency_weights.data
                    if isinstance(dependency_weights, sps.csr_array)
                    else dependency_weights
                )
                if not np.isfinite(dependency_values).all():
                    raise ValueError(
                        "Graph propagation overflowed. Reduce weights, max_hops or decay, "
                        "or enable normalize_adjacency."
                    )

    _normalize_graph_rows(dependency_weights, norm="l2")
    similarity = dependency_weights @ dependency_weights.T

    if isinstance(similarity, sps.csr_array):
        similarity = similarity.toarray()

    np.clip(similarity, 0, 1, out=similarity)
    np.fill_diagonal(similarity, 1.0)

    return similarity
