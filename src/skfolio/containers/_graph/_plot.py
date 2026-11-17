"""Graph layout and Plotly rendering."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Mapping
from html import escape
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.optimize as sco
from scipy.sparse.csgraph import connected_components

from skfolio.typing import BoolArray, FloatArray, IntArray

if TYPE_CHECKING:
    from skfolio.containers._graph._graph import Graph

_MAX_PLOT_NODES = 500
_MAX_PLOT_EDGES = 5_000
_MAX_LAYERED_NODES = 5_000
_MAX_LABEL_LENGTH = 16
_FONT_SIZE = 12
_CHAR_WIDTH_RATIO = 0.6
_NODE_SIZE = 12
_NODE_HEIGHT = 36
_NODE_GAP = 12
_LAYER_GAP = 80
_COMPONENT_GAP = 40
_PORT_GAP = 10
_ARROW_GAP = 12
_LOOP_RADIUS = 14
_RECIPROCAL_OFFSET = 8
_DEFAULT_FIGURE_WIDTH = 700
_FIGURE_MARGIN = dict(l=20, r=20, b=20, t=60)
_MIN_FIGURE_HEIGHT = 220
_MAX_FIGURE_HEIGHT = 1_600
_VERTICAL_PADDING = 30
_LEGEND_WIDTH = 180


def _plot_graph(
    *, graph: Graph, groups: Mapping[str, str] | None, title: str | None
) -> go.Figure:
    """Draw a graph with node labels, directed edges and weight hover text.

    Parameters
    ----------
    graph : Graph
        Weighted directed graph with at most 500 nodes and 5,000 edges.

    groups : mapping of str to str or None
        One group label per node, keyed by node name. Omitted nodes are grouped under
        "Ungrouped". If None, all nodes have the same color and no legend is displayed.

    title : str or None
        Figure title.

    Returns
    -------
    figure : plotly.graph_objects.Figure
        Graph figure with reproducible node positions.
    """
    if graph.n_nodes > _MAX_PLOT_NODES:
        raise ValueError(
            f"plot supports at most {_MAX_PLOT_NODES} nodes; got {graph.n_nodes}. "
            "Use to_dataframe() to inspect or export larger graphs."
        )

    sources, targets, weights = graph._edge_arrays()
    if len(weights) > _MAX_PLOT_EDGES:
        raise ValueError(
            f"plot supports at most {_MAX_PLOT_EDGES} edges; got {len(weights)}. "
            "Use to_dataframe() to inspect or export larger graphs."
        )

    if groups is None:
        group_positions = {"Nodes": list(range(graph.n_nodes))}
    else:
        if not isinstance(groups, Mapping):
            raise TypeError("groups must be a mapping keyed by node.")
        group_positions = {}
        for position, node in enumerate(graph.nodes):
            group = groups.get(node, "Ungrouped")
            if not isinstance(group, str):
                raise TypeError(
                    f"groups must assign one string label per node; got {group!r} "
                    f"for node {node!r}."
                )
            group_positions.setdefault(group, []).append(position)

    node_labels = [escape(node) for node in graph.nodes]
    connectivity = np.zeros((graph.n_nodes, graph.n_nodes), dtype=bool)
    connectivity[sources, targets] = True
    display_labels = [
        node if len(node) <= _MAX_LABEL_LENGTH else node[: _MAX_LABEL_LENGTH - 1] + "…"
        for node in graph.nodes
    ]
    label_widths = (
        np.array([len(label) for label in display_labels])
        * _FONT_SIZE
        * _CHAR_WIDTH_RATIO
    )
    node_width = float(np.max(label_widths, initial=_FONT_SIZE))
    positions, paths, preserve_aspect = _layered_layout(
        connectivity=connectivity, node_width=node_width
    )
    lower, upper = _layout_bounds(
        positions=positions, paths=paths, self_loops=connectivity.diagonal()
    )
    span = upper - lower
    padding = np.array([node_width / 2 + _NODE_GAP, _VERTICAL_PADDING])
    plot_width = _DEFAULT_FIGURE_WIDTH - _FIGURE_MARGIN["l"] - _FIGURE_MARGIN["r"]
    if groups is not None:
        plot_width -= _LEGEND_WIDTH
    drawing_height = span[1]
    if preserve_aspect:
        drawing_height *= (plot_width - 2 * padding[0]) / max(span[0], 1)
    vertical_margin = _FIGURE_MARGIN["t"] + _FIGURE_MARGIN["b"]
    height = int(
        np.clip(
            drawing_height + 2 * padding[1] + vertical_margin,
            _MIN_FIGURE_HEIGHT,
            _MAX_FIGURE_HEIGHT,
        )
    )
    plot_size = np.array([plot_width, height - vertical_margin])
    # Fit the layout before placing labels and edge decorations in pixel coordinates.
    pixel_scale = (plot_size - 2 * padding) / np.maximum(span, 1)
    if preserve_aspect:
        pixel_scale[:] = pixel_scale.min()
    else:
        # Keep compact layouts at their natural width, allowing tall layouts more room.
        pixel_scale[0] = min(pixel_scale[0], span.max() / max(span[0], 1))
    offset = (plot_size - span * pixel_scale) / 2 - lower * pixel_scale
    positions = positions * pixel_scale + offset
    paths = {edge: path * pixel_scale + offset for edge, path in paths.items()}
    curves, arrow_coordinates = _edge_geometry(
        positions=positions, paths=paths, sources=sources, targets=targets
    )
    label_positions = _label_positions(
        positions=positions,
        label_widths=label_widths,
        lower=np.zeros(2),
        upper=plot_size,
        curves=curves,
    )
    edge_coordinates, edge_labels, arrow_labels = [], [], []
    for source, target, weight in zip(sources, targets, weights, strict=True):
        curve = curves[source, target]
        label = f"{node_labels[source]} → {node_labels[target]}<br>Weight: {weight:g}"
        edge_coordinates.extend(curve.tolist())
        edge_coordinates.append([np.nan, np.nan])
        edge_labels.extend([label] * len(curve) + [""])
        arrow_labels.extend([label, label])
    edge_coordinates = np.asarray(edge_coordinates).reshape(-1, 2)
    figure = go.Figure()
    figure.add_scatter(
        x=edge_coordinates[:, 0],
        y=edge_coordinates[:, 1],
        mode="lines",
        line=dict(color="#aaa", width=1),
        text=edge_labels,
        hovertemplate="%{text}<extra></extra>",
        name="Edges",
        showlegend=False,
    )
    figure.add_scatter(
        x=arrow_coordinates[:, 0],
        y=arrow_coordinates[:, 1],
        mode="markers",
        # Each invisible point sets the direction of the following arrow.
        marker=dict(
            symbol="arrow",
            size=np.tile([0, 11], len(weights)),
            color="#666",
            angleref="previous",
        ),
        text=arrow_labels,
        hovertemplate="%{text}<extra></extra>",
        name="Directions",
        showlegend=False,
    )
    colors = px.colors.qualitative.Plotly
    for group_index, (group, node_positions) in enumerate(
        sorted(group_positions.items())
    ):
        color = "#999" if group == "Ungrouped" else colors[group_index % len(colors)]
        figure.add_scatter(
            x=positions[node_positions, 0],
            y=positions[node_positions, 1],
            mode="markers+text",
            marker=dict(size=_NODE_SIZE, color=color),
            text=[escape(display_labels[position]) for position in node_positions],
            hovertext=[node_labels[position] for position in node_positions],
            textposition=[label_positions[position] for position in node_positions],
            textfont=dict(size=_FONT_SIZE, shadow="auto"),
            meta=escape(group),
            hovertemplate=(
                "%{hovertext}<extra></extra>"
                if groups is None
                else "%{hovertext}<br>Group: %{meta}<extra></extra>"
            ),
            cliponaxis=False,
            name=escape(group),
            showlegend=groups is not None,
        )
    figure.update_layout(
        title=title,
        showlegend=groups is not None,
        # Hiding a node trace would leave its edges visible
        legend=dict(itemclick=False, itemdoubleclick=False),
        height=height,
        margin=_FIGURE_MARGIN,
        xaxis=dict(visible=False, range=[0, plot_size[0]], constrain="domain"),
        yaxis=dict(
            visible=False,
            range=[0, plot_size[1]],
            scaleanchor="x" if preserve_aspect else None,
            scaleratio=1,
            constrain="domain",
        ),
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", bordercolor="#bbb", font_color="#333"),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return figure


def _edge_geometry(
    *,
    positions: FloatArray,
    paths: dict[tuple[int, int], FloatArray],
    sources: IntArray,
    targets: IntArray,
) -> tuple[dict[tuple[int, int], FloatArray], FloatArray]:
    """Create edge curves and arrow positions after fitting the layout to the figure.

    Parameters
    ----------
    positions : ndarray of shape (n_nodes, 2)
        Node coordinates in estimated pixels.

    paths : dict of (int, int) to ndarray
        Edge routes keyed by source and target, excluding self-loops.

    sources : ndarray of shape (n_edges,)
        Indices of source nodes.

    targets : ndarray of shape (n_edges,)
        Indices of target nodes.

    Returns
    -------
    curves : dict of (int, int) to ndarray
        Edge curves, including self-loops and offsets for reciprocal edges.

    arrows : ndarray of shape (2 * n_edges, 2)
        Pairs of points defining the direction and position of each arrow. Plotly
        orients the arrow from the preceding point when drawing or resizing the figure.
    """
    curves, arrows = {}, []
    for source, target in zip(sources, targets, strict=True):
        start, end = positions[source], positions[target]
        if source == target:
            angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, 25)
            curve = start + _LOOP_RADIUS * np.column_stack(
                (np.cos(angles), 1 + np.sin(angles))
            )
        else:
            curve = paths[source, target]
            if (target, source) in paths:
                curve = curve.copy()
                delta = end - start
                curve[1:-1] += (
                    _RECIPROCAL_OFFSET
                    * np.array([-delta[1], delta[0]])
                    / np.linalg.norm(delta)
                )
        curves[source, target] = curve
        lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        cumulative = lengths.cumsum()
        # Place arrows before the final segment, where incoming edges can overlap.
        distance = min(
            0.8 * cumulative[-1], cumulative[-2] - min(_ARROW_GAP, lengths[-2] / 2)
        )
        segment = int(np.searchsorted(cumulative, distance))
        fraction = 1 - (cumulative[segment] - distance) / lengths[segment]
        direction = curve[segment + 1] - curve[segment]
        arrows.extend([curve[segment], curve[segment] + fraction * direction])
    return curves, np.asarray(arrows).reshape(-1, 2)


def _label_positions(
    *,
    positions: FloatArray,
    label_widths: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    curves: dict[tuple[int, int], FloatArray],
) -> list[str]:
    """Reduce label overlap with nodes, nearby edges and previously placed labels.

    Parameters
    ----------
    positions : ndarray of shape (n_nodes, 2)
        Estimated node coordinates in pixels.

    label_widths : ndarray of shape (n_nodes,)
        Estimated label widths in pixels.

    lower : ndarray of shape (2,)
        Lower bounds of the plotting area in pixels.

    upper : ndarray of shape (2,)
        Upper bounds of the plotting area in pixels.

    curves : dict of (int, int) to ndarray
        Edge curves in estimated pixels. Labels avoid self-loops and the edge segments
        adjacent to nodes.

    Returns
    -------
    text_positions : list of str
        Plotly text positions, one per node.
    """
    text_positions = [
        "bottom center",
        "top center",
        "middle right",
        "middle left",
        "top right",
        "top left",
        "bottom right",
        "bottom left",
    ]
    directions = np.array(
        [[0, -1], [0, 1], [1, 0], [-1, 0], [1, 1], [-1, 1], [1, -1], [-1, -1]]
    )
    obstacles_lower = [positions - _NODE_SIZE / 2]
    obstacles_upper = [positions + _NODE_SIZE / 2]
    for (source, target), curve in curves.items():
        segments = [curve] if source == target else (curve[:2], curve[-2:])
        for segment in segments:
            obstacles_lower.append(segment.min(axis=0) - 2)
            obstacles_upper.append(segment.max(axis=0) + 2)
    occupied_lower = np.vstack(obstacles_lower)
    occupied_upper = np.vstack(obstacles_upper)
    result = []
    for position, width in zip(positions, label_widths, strict=True):
        half_size = np.array([width, 1.5 * _FONT_SIZE]) / 2
        centers = position + directions * (half_size + _NODE_SIZE / 2 + 2)
        label_lower, label_upper = centers - half_size, centers + half_size
        # Only obstacles within the candidate labels' bounds can contribute overlap.
        nearby = (
            (occupied_upper > label_lower.min(axis=0))
            & (occupied_lower < label_upper.max(axis=0))
        ).all(axis=1)
        intersection = np.minimum(
            label_upper[:, None], occupied_upper[nearby]
        ) - np.maximum(label_lower[:, None], occupied_lower[nearby])
        overlap = np.maximum(intersection, 0).prod(axis=2).sum(axis=1)
        outside = ((label_lower < lower) | (label_upper > upper)).any(axis=1)
        overlap[outside] = np.inf
        choice = int(np.argmin(overlap))
        result.append(text_positions[choice])
        occupied_lower = np.vstack((occupied_lower, label_lower[choice]))
        occupied_upper = np.vstack((occupied_upper, label_upper[choice]))
    return result


def _spring_layout(connectivity: BoolArray) -> FloatArray:
    """Place densely connected components with Fruchterman-Reingold forces.

    Parameters
    ----------
    connectivity : ndarray of shape (n_nodes, n_nodes)
        Symmetric Boolean adjacency without self-loops.

    Returns
    -------
    positions : ndarray of shape (n_nodes, 2)
        Node coordinates.
    """
    n_nodes = len(connectivity)
    positions = np.random.default_rng(0).uniform(-0.5, 0.5, size=(n_nodes, 2))
    spacing = np.sqrt(1.0 / n_nodes)
    for temperature in np.linspace(0.1, 0, 100, endpoint=False):
        differences = positions[:, None, :] - positions[None, :, :]
        distances = np.maximum(np.linalg.norm(differences, axis=2), 0.01)
        forces = spacing**2 / distances**2 - connectivity * distances / spacing
        displacement = np.einsum("ijk,ij->ik", differences, forces) - positions
        lengths = np.linalg.norm(displacement, axis=1, keepdims=True)
        positions += temperature * displacement / np.maximum(lengths, 0.01)
    positions -= positions.mean(axis=0)
    return positions * np.sqrt(n_nodes) * _NODE_HEIGHT / np.abs(positions).max()


def _node_layers(connectivity: BoolArray) -> IntArray:
    """Assign layers while ordering nodes to break cycles for layout purposes.

    Parameters
    ----------
    connectivity : ndarray of shape (n_nodes, n_nodes)
        Directed Boolean adjacency.

    Returns
    -------
    layers : ndarray of shape (n_nodes,)
        Layer indices. Edges point to higher layers except those reversed to break
        cycles. Their original direction is preserved when drawing the graph.
    """
    n_nodes = len(connectivity)
    incoming = connectivity.sum(axis=0) - connectivity.diagonal()
    outgoing = connectivity.sum(axis=1) - connectivity.diagonal()
    remaining = np.ones(n_nodes, dtype=bool)
    layers = np.zeros(n_nodes, dtype=int)
    for _ in range(n_nodes):
        # Prefer sources. Within a cycle, favor nodes with more outgoing edges.
        priority = np.where(incoming == 0, np.inf, outgoing - incoming)
        priority[~remaining] = -np.inf
        node = int(np.argmax(priority))
        remaining[node] = False
        neighbors = remaining & (connectivity[node] | connectivity[:, node])
        layers[neighbors] = np.maximum(layers[neighbors], layers[node] + 1)
        incoming -= connectivity[node]
        outgoing -= connectivity[:, node]
    return layers


def _count_crossings(
    *, layers: list[list[int]], successors: list[list[int]], order: IntArray
) -> int:
    """Count crossings between adjacent layers.

    Parameters
    ----------
    layers : list of lists of int
        Nodes in each layer, including intermediate edge-routing points.

    successors : list of lists of int
        Neighbors in the following layer, indexed by node.

    order : ndarray of shape (n_layout_nodes,)
        Position of each node within its layer.

    Returns
    -------
    crossings : int
        Number of crossings, excluding edges with a common endpoint.
    """
    crossings = 0
    for left, right in pairwise(layers):
        counts = np.zeros(len(right), dtype=int)
        for node in left:
            targets = order[successors[node]]
            # Count earlier edges whose targets come after these targets.
            crossings += int((counts.sum() - counts.cumsum()[targets]).sum())
            np.add.at(counts, targets, 1)
    return crossings


def _order_layers(
    *,
    layers: list[list[int]],
    predecessors: list[list[int]],
    successors: list[list[int]],
) -> list[list[int]]:
    """Reduce crossings with alternating barycenter sweeps.

    Parameters
    ----------
    layers : list of lists of int
        Initial node order in each layer, including edge-routing points. Reordered
        in place during the search.

    predecessors : list of lists of int
        Neighbors in the preceding layer, indexed by node.

    successors : list of lists of int
        Neighbors in the following layer, indexed by node.

    Returns
    -------
    layers : list of lists of int
        Best ordering found, measured by the number of edge crossings.
    """
    order = np.empty(len(predecessors), dtype=int)
    for layer in layers:
        order[layer] = np.arange(len(layer))
    best = [layer.copy() for layer in layers]
    best_crossings = _count_crossings(layers=layers, successors=successors, order=order)
    for sweep in range(8):
        if best_crossings == 0:
            break
        forward = sweep % 2 == 0
        sweep_neighbors = predecessors if forward else successors
        sweep_layers = layers[1:] if forward else layers[-2::-1]
        for layer in sweep_layers:
            layer.sort(
                key=lambda node: (
                    np.mean(order[sweep_neighbors[node]])
                    if sweep_neighbors[node]
                    else order[node]
                )
            )
            order[layer] = np.arange(len(layer))
        # Adjacent swaps improve ties and local minima left by the barycenter sweep.
        for layer in layers:
            for index in range(len(layer) - 1):
                first, second = layer[index : index + 2]
                improvement = 0
                for neighbors in (predecessors, successors):
                    difference = (
                        order[neighbors[first]][:, None] - order[neighbors[second]]
                    )
                    improvement += np.sign(difference).sum()
                if improvement > 0:
                    layer[index : index + 2] = second, first
                    order[first], order[second] = order[second], order[first]
        crossings = _count_crossings(layers=layers, successors=successors, order=order)
        if crossings < best_crossings:
            best_crossings = crossings
            best = [layer.copy() for layer in layers]
    return best


def _layout_component(
    *, connectivity: BoolArray, node_width: float
) -> tuple[FloatArray, dict[tuple[int, int], FloatArray], bool]:
    """Place one connected component and route edges through its layers.

    Parameters
    ----------
    connectivity : ndarray of shape (n_nodes, n_nodes)
        Directed Boolean adjacency.

    node_width : float
        Width reserved for each node and its label.

    Returns
    -------
    positions : ndarray of shape (n_nodes, 2)
        Node coordinates, with dependencies running from left to right.

    paths : dict of (int, int) to ndarray
        Edge coordinates keyed by source and target. Intermediate routing points are
        included for edges spanning several layers. Self-loops are excluded.

    preserve_aspect : bool
        Whether the component requires equal axis scales, as spring layouts do.
    """
    n_nodes = len(connectivity)
    ranks = _node_layers(connectivity)
    undirected_connectivity = connectivity | connectivity.T
    np.fill_diagonal(undirected_connectivity, False)
    # Route each connection from its lower layer to its higher layer.
    sources, targets = np.nonzero(undirected_connectivity & (ranks[:, None] < ranks))
    n_layout_nodes = n_nodes + (ranks[targets] - ranks[sources] - 1).sum()
    if n_layout_nodes > _MAX_LAYERED_NODES:
        # Long edges need a routing point in every crossed layer. Bound that expansion
        # so dense graphs and long chains with many shortcuts remain quick to plot.
        positions = _spring_layout(undirected_connectivity)
        paths = {
            (source, target): np.linspace(positions[source], positions[target], 4)
            for source, target in zip(*np.nonzero(connectivity), strict=True)
            if source != target
        }
        return positions, paths, True
    layers = [
        np.flatnonzero(ranks == rank).tolist() for rank in range(np.max(ranks) + 1)
    ]
    predecessors = [[] for _ in range(n_nodes)]
    successors = [[] for _ in range(n_nodes)]
    routes = {}
    for source, target in zip(sources, targets, strict=True):
        route = [int(source)]
        # Routing points allow long edges to participate in crossing reduction.
        for rank in range(ranks[source] + 1, ranks[target]):
            node = len(predecessors)
            layers[rank].append(node)
            predecessors.append([])
            successors.append([])
            route.append(node)
        route.append(int(target))
        for start, end in pairwise(route):
            successors[start].append(end)
            predecessors[end].append(start)
        routes[source, target] = route

    layers = _order_layers(
        layers=layers, predecessors=predecessors, successors=successors
    )
    positions = np.zeros((len(predecessors), 2))
    top_space = np.zeros(len(predecessors))
    bottom_space = np.zeros(len(predecessors))
    # Reserve loop and route clearance in layout units. These approximate pixels
    # before figure fitting, which can rescale the spacing at the height limits.
    top_space[:n_nodes] = np.where(
        connectivity.diagonal(),
        max(_NODE_HEIGHT / 2, 2 * _LOOP_RADIUS + _RECIPROCAL_OFFSET + _NODE_GAP),
        _NODE_HEIGHT / 2,
    )
    bottom_space[:n_nodes] = _NODE_HEIGHT / 2
    offsets = []
    for rank, layer in enumerate(layers):
        separation = np.r_[
            0, bottom_space[layer[:-1]] + top_space[layer[1:]] + _NODE_GAP
        ]
        offset = separation.cumsum()
        offsets.append(offset)
        positions[layer, 0] = rank * (node_width + _LAYER_GAP)
        positions[layer, 1] = offset - offset[-1] / 2

    # Isotonic regression aligns neighbors while preserving order and label spacing.
    neighbors = [
        left + right for left, right in zip(predecessors, successors, strict=True)
    ]
    for sweep in range(8):
        indices = range(len(layers)) if sweep % 2 == 0 else reversed(range(len(layers)))
        for index in indices:
            layer = layers[index]
            desired = np.array(
                [
                    positions[neighbors[node], 1].mean()
                    if neighbors[node]
                    else positions[node, 1]
                    for node in layer
                ]
            )
            positions[layer, 1] = (
                sco.isotonic_regression(desired - offsets[index]).x + offsets[index]
            )
    positions[:, 1] *= -1

    paths = {}
    port_offset = np.array([node_width / 2 + _PORT_GAP, 0])
    for (source, target), route in routes.items():
        start, end = positions[[source, target]]
        path = np.vstack(
            (
                start,
                start + port_offset,
                positions[route[1:-1]],
                end - port_offset,
                end,
            )
        )
        if connectivity[source, target]:
            paths[source, target] = path
        if connectivity[target, source]:
            paths[target, source] = path[::-1]
    return positions[:n_nodes], paths, False


def _layered_layout(
    *, connectivity: BoolArray, node_width: float
) -> tuple[FloatArray, dict[tuple[int, int], FloatArray], bool]:
    """Draw connected components separately using a Sugiyama-style layered layout.

    Parameters
    ----------
    connectivity : ndarray of shape (n_nodes, n_nodes)
        Directed Boolean adjacency.

    node_width : float
        Width reserved for each node and its label.

    Returns
    -------
    positions : ndarray of shape (n_nodes, 2)
        Node coordinates. Disconnected components are packed into separate rows.

    paths : dict of (int, int) to ndarray
        Edge coordinates keyed by source and target, excluding self-loops.

    preserve_aspect : bool
        Whether all components use a spring layout. Figures containing layered
        components use independent axis scales.

    Notes
    -----
    The layout follows the cycle breaking, layering, crossing reduction and placement
    stages of the Sugiyama method. Layers are ordered with barycenter sweeps, as in
    ELK Layered. Components requiring more than 5,000 nodes and intermediate routing
    points use a spring layout. Edge magnitudes and node groups do not affect the layout.
    """
    n_components, components = connected_components(connectivity, directed=False)
    node_size = np.array([node_width, _NODE_HEIGHT])
    layouts = []
    preserve_aspect = True
    for component in range(n_components):
        nodes = np.flatnonzero(components == component)
        component_positions, component_paths, component_preserve_aspect = (
            _layout_component(
                connectivity=connectivity[np.ix_(nodes, nodes)],
                node_width=node_width,
            )
        )
        preserve_aspect &= component_preserve_aspect
        lower, upper = _layout_bounds(
            positions=component_positions,
            paths=component_paths,
            self_loops=connectivity.diagonal()[nodes],
        )
        origin = np.array([lower[0], upper[1]])
        size = upper - lower + node_size
        layouts.append((nodes, component_positions, component_paths, origin, size))

    # Pack taller components first, with a target width based on their total area.
    layouts.sort(key=lambda layout: -layout[-1][1])
    total_area = sum(np.prod(size + _COMPONENT_GAP) for *_, size in layouts)
    row_width = max(
        np.sqrt(1.5 * total_area), max((size[0] for *_, size in layouts), default=0)
    )
    positions = np.empty((len(connectivity), 2))
    paths = {}
    left = top = row_height = 0.0
    for nodes, component_positions, component_paths, origin, size in layouts:
        if left and left + size[0] > row_width:
            left = 0.0
            top -= row_height + _COMPONENT_GAP
            row_height = 0.0
        offset = np.array([left, top]) - origin
        positions[nodes] = component_positions + offset
        for (source, target), path in component_paths.items():
            paths[nodes[source], nodes[target]] = path + offset
        left += size[0] + _COMPONENT_GAP
        row_height = max(row_height, size[1])
    return positions, paths, preserve_aspect


def _layout_bounds(
    *,
    positions: FloatArray,
    paths: dict[tuple[int, int], FloatArray],
    self_loops: BoolArray,
) -> FloatArray:
    """Compute bounds containing nodes, edge routes and self-loops.

    Parameters
    ----------
    positions : ndarray of shape (n_nodes, 2)
        Node coordinates.

    paths : dict of (int, int) to ndarray
        Edge routes keyed by source and target, excluding self-loops.

    self_loops : ndarray of shape (n_nodes,)
        Boolean mask identifying nodes with self-loops.

    Returns
    -------
    bounds : ndarray of shape (2, 2)
        Lower and upper coordinates. Empty layouts have zero bounds.
    """
    loop_offset = np.array([0, 2 * _LOOP_RADIUS])
    coordinates = np.vstack(
        (positions, *paths.values(), positions[self_loops] + loop_offset)
    )
    if len(coordinates) == 0:
        return np.zeros((2, 2))
    return np.array([coordinates.min(axis=0), coordinates.max(axis=0)])
