"""Observation-aligned graph sequences."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# Credits: Matteo Manzi, Ojas Arora
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Iterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING, overload

import numpy as np
import scipy.sparse as sps

from skfolio.containers._graph._graph import Graph
from skfolio.containers._io import (
    _prepare_save_directory,
    _read_manifest,
    _validate_save_directory,
    _write_manifest,
)
from skfolio.containers._utils import (
    _as_observation_array,
    _normalize_positional_selector,
    _observation_key,
)
from skfolio.typing import AnyArray, ArrayLike, BoolArray, IntArray
from skfolio.utils.tools import _is_integer_number

if TYPE_CHECKING:
    from typing_extensions import Self


__all__ = ["GraphSeries"]

_SAVE_VERSION = 1


class GraphSeries:
    """Graphs indexed by observation.

    Each observation has a `Graph` or `None`. Repeated uses of the same graph share
    storage. Graphs can contain different nodes.

    Parameters
    ----------
    graphs : finite iterable of Graph or None
        Graph for each observation. Use `None` when no graph is available.

    observations : array-like of shape (n_observations,), optional
        Observation labels in the same order as `graphs`. If None, use integer labels
        from 0 to `n_observations - 1`.

    Attributes
    ----------
    n_observations : int
        Number of observations.

    missing_mask : ndarray of bool of shape (n_observations,)
        `True` where no graph is available.

    Notes
    -----
    For point-in-time use, supply only information available at each observation. Reuse
    the last known graph while it remains applicable. Missing graphs are not filled
    automatically.

    When used as metadata, the series must have the same observation order and length
    as `X`. Selecting observations preserves the graphs and their complete node sets.

    For a static graph, repeat the same `Graph` object for every observation.

    Examples
    --------
    A supplier relationship is known from January 2, and a second is added on
    January 4. No graph is available before January 2:

    >>> import pandas as pd
    >>> from skfolio.containers import Graph, GraphSeries
    >>> dates = pd.date_range("2025-01-01", periods=4)
    >>> nodes = ["A", "Supplier 1", "Supplier 2"]
    >>> initial_graph = Graph.from_edges([("A", "Supplier 1", 1)], nodes=nodes)
    >>> updated_graph = Graph.from_edges(
    ...     [("A", "Supplier 1", 1), ("A", "Supplier 2", 1)], nodes=nodes
    ... )
    >>> graphs = GraphSeries(
    ...     [None, initial_graph, initial_graph, updated_graph], observations=dates
    ... )
    >>> graphs
    GraphSeries(n_observations=4, n_graphs=2)

    The graph at the final observation contains both relationships:

    >>> graphs[-1]
    Graph(n_nodes=3, n_edges=2)

    Select the last two observations:

    >>> recent_graphs = graphs[2:]
    >>> recent_graphs
    GraphSeries(n_observations=2, n_graphs=2)

    Iterate over the selected graphs:

    >>> list(recent_graphs)
    [Graph(n_nodes=3, n_edges=1), Graph(n_nodes=3, n_edges=2)]

    For a static graph known throughout the sample:

    >>> from itertools import repeat
    >>> static_graphs = GraphSeries(repeat(initial_graph, len(dates)), observations=dates)
    """

    __slots__ = ("_graphs", "_references", "observations")

    observations: AnyArray
    _graphs: tuple[Graph, ...]
    _references: IntArray

    def __init__(
        self, graphs: Iterable[Graph | None], *, observations: ArrayLike | None = None
    ) -> None:
        if observations is not None:
            observations = _as_observation_array(observations)
            if observations.ndim != 1:
                raise ValueError("observations must be a 1D array.")

        unique_graphs: list[Graph] = []
        graph_positions: dict[int, int] = {}
        references = []
        for position, graph in enumerate(graphs):
            if graph is None:
                references.append(-1)
            elif isinstance(graph, Graph):
                graph_id = id(graph)
                if graph_id not in graph_positions:
                    graph_positions[graph_id] = len(unique_graphs)
                    unique_graphs.append(graph)
                references.append(graph_positions[graph_id])
            else:
                raise TypeError(
                    f"GraphSeries observation {position} must be a Graph or None."
                )

        if observations is None:
            observations = np.arange(len(references), dtype=np.intp)
        elif len(observations) != len(references):
            raise ValueError("graphs and observations must have the same length.")
        else:
            observations = observations.copy()

        self.observations = observations
        self._graphs = tuple(unique_graphs)
        self._references = np.asarray(references, dtype=np.intp)

    def __len__(self) -> int:
        return len(self._references)

    def __iter__(self) -> Iterator[Graph | None]:
        """Iterate over graphs in observation order."""
        for reference in self._references:
            yield None if reference == -1 else self._graphs[reference]

    def __repr__(self) -> str:
        n_graphs = np.unique(self._references[self._references >= 0]).size
        return f"GraphSeries(n_observations={len(self)}, n_graphs={n_graphs})"

    def __reduce__(self) -> tuple:
        """Copy or serialize the selected observations and graphs."""
        return partial(type(self), observations=self.observations), (list(self),)

    @overload
    def __getitem__(
        self, rows: int | np.integer | tuple[int | np.integer, EllipsisType]
    ) -> Graph | None: ...

    @overload
    def __getitem__(self, rows: slice | tuple[slice, EllipsisType]) -> Self: ...

    @overload
    def __getitem__(self, rows: ArrayLike | tuple) -> Graph | Self | None: ...

    def __getitem__(self, rows: int | slice | ArrayLike | tuple) -> Graph | Self | None:
        """Return a graph or select observations by position.

        Parameters
        ----------
        rows : int, slice, array-like or tuple
            Integer position, slice, integer array or boolean mask. Negative positions
            count from the end. `series[rows, ...]` is equivalent to `series[rows]`.

        Returns
        -------
        selected : Graph, None or GraphSeries
            Graph or None for an integer position. Other selectors return a series
            containing the selected observations and sharing the original graphs.

        Raises
        ------
        IndexError
            If a position is outside the series.

        TypeError
            If the selector type is unsupported.
        """
        rows = _observation_key(rows)
        if _is_integer_number(rows):
            if not -len(self) <= int(rows) < len(self):
                raise IndexError("GraphSeries index out of range.")
            reference = self._references[int(rows)]
            return None if reference == -1 else self._graphs[reference]
        if np.isscalar(rows):
            raise TypeError(
                "GraphSeries supports positional indexing with integers, slices, "
                "integer arrays or boolean masks."
            )
        selector = _normalize_positional_selector(length=len(self), selector=rows)
        series = object.__new__(type(self))
        series.observations = self.observations[selector]
        series._graphs = self._graphs
        series._references = self._references[selector]
        return series

    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return len(self)

    @property
    def shape(self) -> tuple[int]:
        """Shape of the series, `(n_observations,)`."""
        return (len(self),)

    @property
    def ndim(self) -> int:
        """Number of dimensions, always 1."""
        return 1

    @property
    def missing_mask(self) -> BoolArray:
        """Boolean mask indicating observations with no graph."""
        return self._references == -1

    def items(self) -> Iterator[tuple[object, Graph | None]]:
        """Iterate over observation labels and graphs.

        Returns
        -------
        items : iterator of (observation, graph) tuples
            Observation labels paired with their `Graph` or `None`, in series order.
        """
        return zip(self.observations, self, strict=True)

    def copy(self, *, deep: bool = False) -> Self:
        """Return a copy of the series.

        The copy retains only graphs used by its observations.

        Parameters
        ----------
        deep : bool, default=False
            If True, copy each distinct graph. If False, share the original graphs.

        Returns
        -------
        series : GraphSeries
            Copy with independent observation labels.
        """
        graphs = iter(self)
        if deep:
            memo: dict = {}
            graphs = (deepcopy(graph, memo) for graph in graphs)
        return type(self)(graphs, observations=self.observations)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the series to a directory.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination directory.

        overwrite : bool, default=False
            If True, replace an existing saved `GraphSeries` at `path`. Other
            directories cannot be overwritten.
        """
        path = Path(path)
        _validate_save_directory(path, overwrite=overwrite, container="GraphSeries")

        # Save only graphs used by the selected observations.
        series = self.copy()
        _prepare_save_directory(path)

        np.save(path / "observations.npy", series.observations, allow_pickle=False)
        np.save(path / "references.npy", series._references, allow_pickle=False)

        graphs_path = path / "graphs"
        graphs_path.mkdir()
        for graph_position, graph in enumerate(series._graphs):
            adjacency = graph.adjacency
            if isinstance(adjacency, sps.csr_array):
                np.savez(
                    graphs_path / f"{graph_position}.npz",
                    nodes=graph.nodes,
                    data=adjacency.data,
                    indices=adjacency.indices,
                    indptr=adjacency.indptr,
                )
            else:
                np.savez(
                    graphs_path / f"{graph_position}.npz",
                    nodes=graph.nodes,
                    adjacency=adjacency,
                )

        _write_manifest(
            path,
            container="GraphSeries",
            version=_SAVE_VERSION,
            n_observations=len(series),
            n_graphs=len(series._graphs),
        )

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a saved graph series.

        Parameters
        ----------
        path : str or pathlib.Path
            Directory containing a saved `GraphSeries`.

        Returns
        -------
        series : GraphSeries
            Loaded series. Repeated graphs remain shared.
        """
        path = Path(path)
        metadata = _read_manifest(
            path, container="GraphSeries", max_version=_SAVE_VERSION
        )
        references = np.load(path / "references.npy", allow_pickle=False)

        if (
            references.shape != (metadata["n_observations"],)
            or references.dtype.kind not in "iu"
            or np.any(references < -1)
            or np.any(references >= metadata["n_graphs"])
        ):
            raise ValueError("Invalid GraphSeries observation references.")

        observations = np.load(path / "observations.npy", allow_pickle=False)

        graphs = []
        for graph_position in range(metadata["n_graphs"]):
            graph_path = path / "graphs" / f"{graph_position}.npz"
            with np.load(graph_path, allow_pickle=False) as graph_data:
                nodes = graph_data["nodes"]
                adjacency = (
                    graph_data["adjacency"]
                    if "adjacency" in graph_data
                    else sps.csr_array(
                        (
                            graph_data["data"],
                            graph_data["indices"],
                            graph_data["indptr"],
                        ),
                        shape=(len(nodes), len(nodes)),
                    )
                )
                graphs.append(Graph(adjacency, nodes=nodes))

        return cls(
            (
                None if reference == -1 else graphs[reference]
                for reference in references
            ),
            observations=observations,
        )
