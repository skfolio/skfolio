"""Weighted directed graph container."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# Credits: Matteo Manzi, Ojas Arora
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.sparse as sps

from skfolio.containers._graph._plot import _plot_graph
from skfolio.containers._utils import _as_string_labels
from skfolio.typing import (
    ArrayLike,
    CSRArray,
    FloatArray,
    IntArray,
    SparseArray,
    StrArray,
)

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["Graph"]


class Graph:
    """Weighted directed graph with named nodes.

    Adjacency rows represent source nodes and columns represent target nodes. Zero
    indicates no edge. Positive and negative weights, cycles, and self-loops are
    supported. Estimators using the graph can impose additional restrictions.

    Input arrays are copied and stored as read-only arrays.

    Parameters
    ----------
    adjacency : array-like or SciPy sparse array of shape (n_nodes, n_nodes)
        Adjacency matrix of finite real edge weights. Dense input is stored as a NumPy
        array. Sparse input is converted to compressed sparse row (CSR) format. In
        sparse input, the same edge cannot be specified more than once. For a pandas
        DataFrame, the index and columns must match `nodes` in order.

    nodes : array-like of shape (n_nodes,), optional
        Unique string identifiers, including nodes with no incoming or outgoing edges.
        Their order defines the rows and columns of the adjacency matrix. Missing values
        are not allowed. If None, `adjacency` must be a pandas DataFrame and its index
        is used.

    Attributes
    ----------
    nodes : ndarray of shape (n_nodes,)
        Read-only node identifiers.

    adjacency : ndarray or scipy.sparse.csr_array of shape (n_nodes, n_nodes)
        Read-only adjacency matrix of edge weights.

    n_nodes : int
        Number of nodes, including isolated nodes.

    Examples
    --------
    >>> from skfolio.containers import Graph
    >>> graph = Graph.from_edges(
    ...     [("A", "Supplier", 1.0), ("B", "Supplier", 0.5)],
    ...     nodes=["A", "B", "Supplier"],
    ... )
    >>> graph.n_nodes
    3
    """

    __slots__ = ("_adjacency", "_nodes")

    _nodes: StrArray
    _adjacency: FloatArray | CSRArray

    def __init__(
        self, adjacency: ArrayLike | SparseArray, *, nodes: ArrayLike | None = None
    ) -> None:
        nodes_source = "nodes"
        if nodes is None:
            if not isinstance(adjacency, pd.DataFrame):
                raise ValueError("nodes is required when adjacency is not a DataFrame.")
            nodes = adjacency.index
            nodes_source = "adjacency index"
        nodes = _as_string_labels(values=nodes, name=nodes_source).copy()
        if isinstance(adjacency, pd.DataFrame):
            node_index = pd.Index(nodes, dtype=object)
            if any(
                not node_index.equals(pd.Index(labels, dtype=object))
                for labels in (adjacency.index, adjacency.columns)
            ):
                raise ValueError(
                    "DataFrame adjacency index and columns must match nodes in order."
                )
        adjacency = _normalize_adjacency(adjacency, n_nodes=len(nodes))
        self._nodes = nodes
        self._adjacency = adjacency
        self._set_readonly()

    @property
    def nodes(self) -> StrArray:
        """Read-only node identifiers ordered as the adjacency rows and columns."""
        return self._nodes.view()

    @property
    def adjacency(self) -> FloatArray | CSRArray:
        """Read-only adjacency matrix of edge weights."""
        adjacency = self._adjacency
        if isinstance(adjacency, sps.csr_array):
            return sps.csr_array(
                (
                    adjacency.data.view(),
                    adjacency.indices.view(),
                    adjacency.indptr.view(),
                ),
                shape=adjacency.shape,
                copy=False,
            )
        return adjacency.view()

    @property
    def n_nodes(self) -> int:
        """Number of nodes, including isolated nodes."""
        return len(self._nodes)

    def to_dataframe(
        self, *, output_format: Literal["edges", "adjacency"] = "edges"
    ) -> pd.DataFrame:
        """Convert the graph to a pandas DataFrame.

        Parameters
        ----------
        output_format : {"edges", "adjacency"}, default="edges"
            With "edges", return nonzero edges as `source`, `target` and `weight`
            columns, ordered by source and target positions in `nodes`. With
            "adjacency", return the adjacency matrix with node identifiers as both
            index and columns. Sparse adjacency produces a pandas sparse DataFrame.

        Returns
        -------
        dataframe : DataFrame
            Edge table or adjacency matrix, independent of the graph's arrays.

        Examples
        --------
        >>> from skfolio.containers import Graph
        >>> graph = Graph.from_edges([("A", "Supplier", 2)], nodes=["A", "Supplier"])
        >>> graph.to_dataframe()
          source    target  weight
        0      A  Supplier     2.0
        """
        if output_format == "edges":
            sources, targets, weights = self._edge_arrays()
            return pd.DataFrame(
                {
                    "source": self._nodes[sources],
                    "target": self._nodes[targets],
                    "weight": weights,
                }
            )
        if output_format == "adjacency":
            if isinstance(self._adjacency, sps.csr_array):
                return pd.DataFrame.sparse.from_spmatrix(
                    self._adjacency,
                    index=self._nodes.copy(),
                    columns=self._nodes.copy(),
                )
            return pd.DataFrame(
                self._adjacency,
                index=self._nodes.copy(),
                columns=self._nodes.copy(),
                copy=True,
            )
        raise ValueError("output_format must be 'edges' or 'adjacency'.")

    def plot(
        self, *, groups: Mapping[str, str] | None = None, title: str | None = None
    ) -> go.Figure:
        """Plot the directed graph with an automatic layout.

        Dependencies are arranged from left to right, with node ordering chosen to
        reduce edge crossings. Graphs with many edges spanning several layers use a
        spring layout. Edge weights do not affect node positions.

        Parameters
        ----------
        groups : mapping of str to str, optional
            Group labels keyed by node name. Each node has one string label, used for
            its color and legend entry. Nodes omitted from `groups` belong to the gray
            "Ungrouped" category. Entries for nodes outside the graph are ignored.
            Groups are ordered alphabetically for color assignment and the legend.
            If None, all nodes have the same color and no legend is displayed.

        title : str, optional
            Figure title. If None, no title is displayed.

        Returns
        -------
        figure : plotly.graph_objects.Figure
            Interactive graph figure.

        Notes
        -----
        Plotting supports up to 500 nodes and 5,000 edges. Larger graphs can be exported
        with `to_dataframe` for visualization with other tools. Positions can change
        between snapshots, so their coordinates do not represent changes in edge weights.

        Examples
        --------
        Assets and suppliers shown in different colors:

        >>> from skfolio.containers import Graph
        >>> graph = Graph.from_edges(
        ...     [("A", "Supplier", 1), ("B", "Supplier", 1)],
        ...     nodes=["A", "B", "Supplier"],
        ... )
        >>> groups = {"A": "Assets", "B": "Assets", "Supplier": "Suppliers"}
        >>> figure = graph.plot(groups=groups)
        """
        return _plot_graph(graph=self, groups=groups, title=title)

    def __repr__(self) -> str:
        adjacency = self._adjacency
        n_edges = (
            adjacency.nnz
            if isinstance(adjacency, sps.csr_array)
            else np.count_nonzero(adjacency)
        )
        return f"Graph(n_nodes={self.n_nodes}, n_edges={n_edges})"

    def __getstate__(self) -> tuple[StrArray, FloatArray | CSRArray]:
        return self._nodes, self._adjacency

    def __setstate__(self, state: tuple[StrArray, FloatArray | CSRArray]) -> None:
        self._nodes, self._adjacency = state
        self._set_readonly()

    def __copy__(self) -> Self:
        return self

    def _edge_arrays(self) -> tuple[IntArray, IntArray, FloatArray]:
        """Return edge endpoints and weights ordered by source and target position.

        Returns
        -------
        sources : ndarray of shape (n_edges,)
            Source positions in `nodes`.

        targets : ndarray of shape (n_edges,)
            Target positions in `nodes`.

        weights : ndarray of shape (n_edges,)
            Nonzero edge weights.
        """
        sources, targets = self._adjacency.nonzero()
        weights = (
            self._adjacency.data
            if isinstance(self._adjacency, sps.csr_array)
            else self._adjacency[sources, targets]
        )
        return sources, targets, weights

    def _set_readonly(self) -> None:
        """Make node and adjacency arrays read-only."""
        if isinstance(self._adjacency, sps.csr_array):
            buffers = (
                self._nodes,
                self._adjacency.data,
                self._adjacency.indices,
                self._adjacency.indptr,
            )
        else:
            buffers = (self._nodes, self._adjacency)

        for buffer in buffers:
            # Restored arrays and CSR buffers can be views onto writable arrays.
            while isinstance(buffer, np.ndarray):
                buffer.flags.writeable = False
                buffer = buffer.base

    @classmethod
    def from_edges(
        cls, edges: Iterable[tuple[str, str, float]], *, nodes: ArrayLike
    ) -> Self:
        """Create a graph from weighted edges.

        Parameters
        ----------
        edges : iterable of (source, target, weight)
            Directed edges with finite real weights. Source and target must belong to
            `nodes`. The same edge cannot be specified more than once.

        nodes : array-like of shape (n_nodes,)
            Unique string identifiers for all nodes. Nodes absent from `edges` are
            included as isolated nodes. Their order defines the rows and columns of the
            adjacency matrix. Missing values are not allowed.

        Returns
        -------
        graph : Graph
            Weighted directed graph.

        Examples
        --------
        Node B has no incoming or outgoing edges:

        >>> from skfolio.containers import Graph
        >>> graph = Graph.from_edges(
        ...     [("A", "Supplier", 1.0)],
        ...     nodes=["A", "B", "Supplier"],
        ... )
        >>> graph.adjacency.toarray()
        array([[0., 0., 1.],
               [0., 0., 0.],
               [0., 0., 0.]])
        """
        nodes = _as_string_labels(values=nodes, name="nodes")
        node_positions = {node: position for position, node in enumerate(nodes)}
        rows, columns, weights = [], [], []
        for source, target, weight in edges:
            try:
                rows.append(node_positions[source])
                columns.append(node_positions[target])
            except KeyError as exc:
                raise ValueError(
                    f"Edge endpoint {exc.args[0]!r} is absent from nodes."
                ) from exc
            weights.append(weight)
        edge_weights = np.asarray(weights)
        adjacency = sps.coo_array(
            (edge_weights, (rows, columns)),
            shape=(len(nodes), len(nodes)),
            dtype=_weight_dtype(edge_weights.dtype),
        )
        return cls(adjacency, nodes=nodes)


def _weight_dtype(dtype: np.dtype) -> np.dtype:
    """Return the floating-point dtype used for edge weights.

    Float32 and float64 are preserved. Other real numeric dtypes are converted to
    float64. Non-numeric and complex dtypes are not supported.
    """
    if dtype.kind not in "biuf":
        raise ValueError("adjacency must contain finite real numerical weights.")
    if dtype in (np.dtype("float32"), np.dtype("float64")):
        return dtype
    return np.dtype("float64")


def _normalize_adjacency(
    adjacency: ArrayLike | SparseArray,
    *,
    n_nodes: int,
) -> FloatArray | CSRArray:
    """Validate and copy an adjacency matrix.

    The matrix must have shape (n_nodes, n_nodes) and contain finite real weights.
    Sparse input is converted to CSR with sorted column indices. Explicit zero entries
    are removed. Duplicate source-target pairs are not allowed.
    """
    if (
        isinstance(adjacency, pd.DataFrame)
        and any(
            isinstance(column_dtype, pd.api.extensions.ExtensionDtype)
            for column_dtype in adjacency.dtypes
        )
        and all(
            pd.api.types.is_numeric_dtype(column_dtype)
            for column_dtype in adjacency.dtypes
        )
    ):
        # Convert nullable columns directly, avoiding an intermediate object array.
        weight_dtype = _weight_dtype(
            np.result_type(
                *(
                    getattr(column_dtype, "numpy_dtype", column_dtype.type)
                    for column_dtype in adjacency.dtypes
                )
            )
        )
        adjacency = adjacency.to_numpy(dtype=weight_dtype, na_value=np.nan)
    else:
        adjacency = (
            sps.coo_array(adjacency)
            if sps.issparse(adjacency)
            else np.asarray(adjacency)
        )
    if adjacency.shape != (n_nodes, n_nodes):
        raise ValueError("adjacency must be square and match nodes.")
    weight_dtype = _weight_dtype(adjacency.dtype)
    if isinstance(adjacency, sps.coo_array):
        result = sps.csr_array(adjacency, dtype=weight_dtype, copy=True)
        # CSR conversion combines duplicate coordinates, including explicit zeros.
        if result.nnz != adjacency.nnz:
            raise ValueError("Duplicate source-target pairs are not allowed.")
        result.eliminate_zeros()
        index_dtype = (
            np.int32
            if max(*result.shape, result.nnz) <= np.iinfo(np.int32).max
            else np.int64
        )
        result.indices = result.indices.astype(index_dtype, copy=False)
        result.indptr = result.indptr.astype(index_dtype, copy=False)
        edge_weights = result.data
    else:
        result = np.array(adjacency, dtype=weight_dtype, copy=True)
        edge_weights = result
    if not np.isfinite(edge_weights).all():
        raise ValueError("adjacency must contain finite real weights.")
    return result
