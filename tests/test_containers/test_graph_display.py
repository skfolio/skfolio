"""Graph table exports and interactive figures."""

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from skfolio.containers import Graph
from skfolio.containers._graph._plot import (
    _LOOP_RADIUS,
    _MAX_PLOT_EDGES,
    _MAX_PLOT_NODES,
    _NODE_HEIGHT,
    _RECIPROCAL_OFFSET,
    _count_crossings,
    _edge_geometry,
    _label_positions,
    _layered_layout,
    _order_layers,
)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_graph_to_dataframe(sparse, dtype):
    nodes = ["Z", "A", "Isolated"]
    adjacency = np.array([[1, -2, 0], [3, 0, 0], [0, 0, 0]], dtype=dtype)
    graph = Graph(sps.csr_array(adjacency) if sparse else adjacency, nodes=nodes)

    edges = graph.to_dataframe()
    pd.testing.assert_frame_equal(
        edges,
        pd.DataFrame(
            {
                "source": ["Z", "Z", "A"],
                "target": ["Z", "A", "Z"],
                "weight": np.array([1, -2, 3], dtype=dtype),
            }
        ),
    )
    edges.loc[0, "source"] = "Changed"
    edges.loc[0, "weight"] = 99
    frame = graph.to_dataframe(output_format="adjacency")
    expected = pd.DataFrame(adjacency, index=nodes, columns=nodes)
    pd.testing.assert_frame_equal(
        frame.sparse.to_dense() if sparse else frame, expected
    )
    if sparse:
        assert all(isinstance(dtype, pd.SparseDtype) for dtype in frame.dtypes)
        frame["Z"].array.sp_values[:] = 99
    else:
        frame.iloc[0, 0] = 99
    frame.index = ["Changed", "A", "Isolated"]
    frame.columns = ["Z", "Changed", "Isolated"]
    np.testing.assert_array_equal(graph.nodes, nodes)
    np.testing.assert_array_equal(
        graph.adjacency.toarray() if sparse else graph.adjacency, adjacency
    )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n_nodes", [0, 1, 3])
def test_graph_empty_exports(sparse, n_nodes):
    adjacency = np.zeros((n_nodes, n_nodes))
    graph = Graph(
        sps.csr_array(adjacency) if sparse else adjacency,
        nodes=[f"Node {i}" for i in range(n_nodes)],
    )
    edges = graph.to_dataframe()
    assert edges.shape == (0, 3)
    assert edges.columns.tolist() == ["source", "target", "weight"]
    assert graph.to_dataframe(output_format="adjacency").shape == (n_nodes, n_nodes)


def test_graph_invalid_output_format():
    graph = Graph([[0]], nodes=["A"])
    with pytest.raises(ValueError, match="output_format"):
        graph.to_dataframe(output_format="unknown")


@pytest.mark.parametrize("sparse", [False, True])
def test_graph_plot_edges_and_isolated_nodes(sparse):
    adjacency = np.array([[1, 2, 0, 0], [-3, 0, 4, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    nodes = ["A", "B", "Supplier", "Isolated"]
    graph = Graph(sps.csr_array(adjacency) if sparse else adjacency, nodes=nodes)
    figure = graph.plot(title="Dependencies")
    assert figure.layout.title.text == "Dependencies"
    edges, arrows, node_trace = figure.data
    assert list(node_trace.text) == nodes
    assert len(arrows.x[1::2]) == 4
    assert list(arrows.text[1::2]) == [
        "A → A<br>Weight: 1",
        "A → B<br>Weight: 2",
        "B → A<br>Weight: -3",
        "B → Supplier<br>Weight: 4",
    ]
    assert arrows.text[::2] == arrows.text[1::2]
    assert np.isfinite(node_trace.x).all()
    assert np.isfinite(node_trace.y).all()
    assert arrows.marker.angleref == "previous"

    # The self-loop has visible extent, and reciprocal edges follow distinct curves.
    separators = np.flatnonzero(np.isnan(edges.x))
    loops_and_edges = np.split(np.column_stack((edges.x, edges.y)), separators + 1)
    loop = loops_and_edges[0][:-1]
    forward = loops_and_edges[1][:-1]
    backward = loops_and_edges[2][:-1]
    assert np.ptp(loop[:, 0]) > 0
    assert np.ptp(loop[:, 1]) > 0
    np.testing.assert_allclose(loop[0], loop[-1])
    np.testing.assert_allclose(forward[0], backward[-1])
    assert not np.allclose(forward, backward[::-1])
    for trace in (edges, arrows):
        assert trace.hovertemplate == "%{text}<extra></extra>"


def test_graph_plot_reproducible_and_independent_of_weights():
    adjacency = np.array([[0, 2, 0], [0, 0, 3], [0, 0, 0]])
    nodes = ["A", "B", "C"]
    graph = Graph(adjacency, nodes=nodes)
    figure = graph.plot()
    assert figure.to_json() == graph.plot().to_json()
    for equivalent in [
        Graph(sps.csr_array(adjacency), nodes=nodes),
        Graph(-10 * adjacency, nodes=nodes),
    ]:
        other = equivalent.plot()
        np.testing.assert_array_equal(figure.data[-1].x, other.data[-1].x)
        np.testing.assert_array_equal(figure.data[-1].y, other.data[-1].y)


def test_graph_plot_dependency_direction_and_bounds():
    adjacency = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    nodes = ["Stock", "Battery", "Lithium"]
    figure = Graph(adjacency, nodes=nodes).plot()
    reverse = Graph(adjacency.T, nodes=nodes).plot()
    assert np.all(np.diff(figure.data[-1].x) > 0)
    assert np.all(np.diff(reverse.data[-1].x) < 0)
    assert figure.layout.yaxis.scaleanchor is None
    for trace in figure.data:
        x, y = np.asarray(trace.x), np.asarray(trace.y)
        finite = np.isfinite(x) & np.isfinite(y)
        assert np.all(x[finite] > figure.layout.xaxis.range[0])
        assert np.all(x[finite] < figure.layout.xaxis.range[1])
        assert np.all(y[finite] > figure.layout.yaxis.range[0])
        assert np.all(y[finite] < figure.layout.yaxis.range[1])


def test_graph_plot_long_chain_labels():
    graph = Graph(
        np.diag(np.ones(9), 1), nodes=[f"SupplierNumber{i}" for i in range(10)]
    )
    figure = graph.plot()
    nodes = figure.data[-1]
    assert len(set(nodes.textposition)) > 1

    # Even after fitting the long chain to the default 700-pixel figure, the outer
    # nodes leave at least half a label's width before the plotting-area boundary.
    x_min, x_max = figure.layout.xaxis.range
    plot_width = 700 - figure.layout.margin.l - figure.layout.margin.r
    pixel_scale = plot_width / (x_max - x_min)
    assert (nodes.x[0] - x_min) * pixel_scale >= 60
    assert (x_max - nodes.x[-1]) * pixel_scale >= 60


def test_graph_plot_tall_layer_uses_width_and_preserves_label_spacing():
    nodes = [f"Fund{i}" for i in range(30)] + ["Hub"]
    graph = Graph.from_edges([(node, "Hub", 1) for node in nodes[:-1]], nodes=nodes)
    figure = graph.plot()
    trace = figure.data[-1]
    assert figure.layout.yaxis.scaleanchor is None
    assert np.ptp(trace.x) > 0.8 * np.ptp(figure.layout.xaxis.range)

    plot_height = figure.layout.height - figure.layout.margin.t - figure.layout.margin.b
    pixel_scale = plot_height / np.ptp(figure.layout.yaxis.range)
    spacing = np.diff(np.sort(trace.y[:-1])) * pixel_scale
    assert np.all(spacing >= trace.marker.size + 1.5 * trace.textfont.size)


@pytest.mark.parametrize("grouped", [False, True])
def test_graph_plot_two_nodes_stay_compact_and_centered(grouped):
    graph = Graph.from_edges([("Stock", "Supplier", 1)], nodes=["Stock", "Supplier"])
    groups = {"Stock": "Stocks", "Supplier": "Suppliers"} if grouped else None
    figure = graph.plot(groups=groups)
    positions = np.concatenate([trace.x for trace in figure.data[2:]])
    x_min, x_max = figure.layout.xaxis.range
    assert 100 < np.ptp(positions) < (x_max - x_min) / 2
    assert positions.min() + positions.max() == pytest.approx(x_min + x_max)


def test_graph_plot_arrows_follow_edge_segments_after_resizing():
    graph = Graph.from_edges(
        [("A", "A", 1), ("A", "B", 1), ("B", "A", 1), ("B", "C", 1)],
        nodes=["A", "B", "C"],
    )
    figure = graph.plot()
    figure.update_layout(width=1200, height=400)
    edges, arrows, _ = figure.data
    assert arrows.marker.angleref == "previous"
    assert np.all(np.asarray(arrows.marker.size)[::2] == 0)
    assert np.all(np.asarray(arrows.marker.size)[1::2] > 0)
    separators = np.flatnonzero(np.isnan(edges.x))
    curves = np.split(np.column_stack((edges.x, edges.y)), separators + 1)[:-1]
    arrow_pairs = np.column_stack((arrows.x, arrows.y)).reshape(-1, 2, 2)
    for curve, (anchor, tip) in zip(curves, arrow_pairs, strict=True):
        segment = int(np.flatnonzero(np.all(curve[:-1] == anchor, axis=1))[0])
        direction = curve[segment + 1] - anchor
        fraction = (tip - anchor) @ direction / (direction @ direction)
        assert 0 < fraction < 1
        np.testing.assert_allclose(tip, anchor + fraction * direction)


@pytest.mark.parametrize("n_nodes", [0, 1, 3])
def test_graph_plot_no_edges(n_nodes):
    graph = Graph(
        np.zeros((n_nodes, n_nodes)), nodes=[f"Node {i}" for i in range(n_nodes)]
    )
    figure = graph.plot()
    assert len(figure.data[-1].x) == n_nodes
    assert len(figure.data[0].x) == 0
    assert len(figure.data[1].x) == 0
    assert np.isfinite(figure.data[-1].x).all()


def test_graph_plot_single_loop_and_escaped_labels():
    graph = Graph([[2]], nodes=["<Supplier> & Co"])
    figure = graph.plot()
    assert list(figure.data[-1].text) == ["&lt;Supplier&gt; &amp; Co"]
    assert len(figure.data[1].x[1::2]) == 1
    assert figure.data[1].marker.angleref == "previous"


def test_graph_plot_node_limit():
    n_nodes = _MAX_PLOT_NODES + 1
    graph = Graph(
        sps.csr_array((n_nodes, n_nodes)), nodes=[str(i) for i in range(n_nodes)]
    )
    with pytest.raises(ValueError, match="to_dataframe"):
        graph.plot()


@pytest.mark.parametrize("sparse", [False, True])
def test_graph_plot_edge_limit(sparse):
    n_nodes = int(np.sqrt(_MAX_PLOT_EDGES)) + 1
    adjacency = np.zeros((n_nodes, n_nodes))
    adjacency.flat[:_MAX_PLOT_EDGES] = 1
    nodes = [str(i) for i in range(n_nodes)]
    graph = Graph(sps.csr_array(adjacency) if sparse else adjacency, nodes=nodes)
    assert len(graph.plot().data[1].x[1::2]) == _MAX_PLOT_EDGES

    adjacency.flat[_MAX_PLOT_EDGES] = 1
    graph = Graph(sps.csr_array(adjacency) if sparse else adjacency, nodes=nodes)
    with pytest.raises(ValueError, match=r"edges.*to_dataframe"):
        graph.plot()


def test_graph_plot_groups_match_nodes_and_preserve_layout_shape():
    graph = Graph.from_edges(
        [("A", "Supplier", 2), ("B", "Supplier", 1)],
        nodes=["A", "B", "Supplier", "Isolated"],
    )
    groups = {
        "Supplier": "Foundries",
        "B": "Assets",
        "Isolated": "Equipment",
        "A": "Assets",
        "Outside graph": "Unused group",
    }
    figure = graph.plot(groups=groups)
    ungrouped = graph.plot()

    assert figure.layout.showlegend
    assert not ungrouped.layout.showlegend
    assert not figure.data[0].showlegend
    assert not figure.data[1].showlegend
    node_traces = {trace.name: trace for trace in figure.data[2:]}
    assert set(node_traces) == {"Assets", "Foundries", "Equipment"}
    assert len({trace.marker.color for trace in node_traces.values()}) == 3
    assert list(node_traces["Assets"].text) == ["A", "B"]
    assert list(node_traces["Foundries"].text) == ["Supplier"]
    assert list(node_traces["Equipment"].text) == ["Isolated"]
    grouped_positions = np.empty((graph.n_nodes, 2))
    for trace in node_traces.values():
        assert trace.showlegend
        assert trace.meta == trace.name
        assert "Group: %{meta}" in trace.hovertemplate
        positions = pd.Index(graph.nodes).get_indexer(trace.text)
        grouped_positions[positions] = np.column_stack((trace.x, trace.y))
    original_positions = np.column_stack((ungrouped.data[-1].x, ungrouped.data[-1].y))
    # The legend reduces the drawing width, while relative node positions stay fixed.
    np.testing.assert_allclose(
        (grouped_positions - grouped_positions.min(axis=0))
        / np.ptp(grouped_positions, axis=0),
        (original_positions - original_positions.min(axis=0))
        / np.ptp(original_positions, axis=0),
    )
    for edge_trace, original in zip(figure.data[:2], ungrouped.data[:2], strict=True):
        assert edge_trace.text == original.text


def test_graph_plot_group_colors_independent_of_node_order():
    groups = {"A": "Assets", "Supplier": "Suppliers"}
    graph = Graph.from_edges([("A", "Supplier", 1)], nodes=list(groups))
    reordered = Graph.from_edges([("A", "Supplier", 1)], nodes=list(reversed(groups)))
    expected = {
        trace.name: trace.marker.color for trace in graph.plot(groups=groups).data[2:]
    }
    actual = {
        trace.name: trace.marker.color
        for trace in reordered.plot(groups=groups).data[2:]
    }
    assert actual == expected


def test_graph_plot_partial_groups():
    graph = Graph.from_edges(
        [("A", "Supplier", 2)], nodes=["A", "Supplier", "Isolated"]
    )
    figure = graph.plot(groups={"A": "Assets", "Isolated": "Ungrouped"})
    assets, ungrouped = figure.data[2:]
    assert assets.name == "Assets"
    assert list(assets.text) == ["A"]
    assert ungrouped.name == "Ungrouped"
    assert list(ungrouped.text) == ["Supplier", "Isolated"]
    assert ungrouped.marker.color == "#999"


def test_graph_plot_empty_groups():
    graph = Graph([[0]], nodes=["A"])
    figure = graph.plot(groups={})
    assert figure.data[-1].name == "Ungrouped"
    assert list(figure.data[-1].text) == ["A"]
    empty = Graph(np.empty((0, 0)), nodes=[]).plot(groups={})
    assert all(len(trace.x) == 0 for trace in empty.data)


@pytest.mark.parametrize(
    "groups",
    [["Assets"], "Assets", pd.DataFrame({"A": ["Assets"]}), pd.Series({"A": "Assets"})],
)
def test_graph_plot_requires_groups_keyed_by_node(groups):
    graph = Graph([[0]], nodes=["A"])
    with pytest.raises(TypeError, match="mapping keyed by node"):
        graph.plot(groups=groups)


@pytest.mark.parametrize("group", [["Assets", "US"], 1, None])
def test_graph_plot_requires_one_string_group_per_node(group):
    graph = Graph([[0]], nodes=["A"])
    with pytest.raises(TypeError, match="one string label per node"):
        graph.plot(groups={"A": group})


def test_graph_plot_escapes_group_labels():
    graph = Graph([[0]], nodes=["A"])
    trace = graph.plot(groups={"A": "<Assets> & Companies"}).data[-1]
    assert trace.name == "&lt;Assets&gt; &amp; Companies"
    assert trace.meta == trace.name


@pytest.mark.parametrize("grouped", [False, True])
def test_graph_plot_short_labels_preserve_full_node_names(grouped):
    nodes = ["Very long supplier name A", "Very long supplier name B", "Short"]
    graph = Graph.from_edges([(nodes[0], nodes[1], 2)], nodes=nodes)
    groups = {node: "Suppliers" for node in nodes} if grouped else None
    figure = graph.plot(groups=groups)
    node_trace = figure.data[-1]
    assert list(node_trace.text) == ["Very long suppl…", "Very long suppl…", "Short"]
    assert list(node_trace.hovertext) == nodes
    assert "%{hovertext}" in node_trace.hovertemplate
    assert figure.data[1].text[1] == f"{nodes[0]} → {nodes[1]}<br>Weight: 2"
    np.testing.assert_array_equal(graph.nodes, nodes)


def test_graph_plot_shortens_before_html_escaping():
    graph = Graph([[0]], nodes=["Supplier & <long name>"])
    trace = graph.plot().data[-1]
    assert trace.text[0] == "Supplier &amp; &lt;lon…"
    assert trace.hovertext[0] == "Supplier &amp; &lt;long name&gt;"


def test_labels_keep_uniform_positions_when_space_allows():
    positions = np.array([[0, 0], [0, 50], [120, 0], [120, 50]])
    result = _label_positions(
        positions=positions,
        label_widths=np.full(4, 100),
        lower=np.array([-60, -40]),
        upper=np.array([180, 90]),
        curves={},
    )
    assert result == ["bottom center"] * 4


def test_labels_avoid_other_nodes():
    result = _label_positions(
        positions=np.array([[0, 0], [0, -20]]),
        label_widths=np.full(2, 100),
        lower=np.array([-60, -50]),
        upper=np.array([60, 40]),
        curves={},
    )
    assert result == ["top center", "bottom center"]


def test_labels_avoid_previously_placed_labels():
    result = _label_positions(
        positions=np.array([[0, 0], [80, 0]]),
        label_widths=np.full(2, 100),
        lower=np.array([-100, -100]),
        upper=np.array([180, 100]),
        curves={},
    )
    assert result == ["bottom center", "top center"]


def test_labels_avoid_plot_boundary():
    result = _label_positions(
        positions=np.array([[0, 0]]),
        label_widths=np.array([100]),
        lower=np.array([-60, -10]),
        upper=np.array([60, 40]),
        curves={},
    )
    assert result == ["top center"]


def test_labels_stay_inside_bounds_when_edges_are_crowded():
    positions = np.array([[50, 0], [300, 100], [300, 120], [300, -100], [300, -120]])
    curves = {(0, i): np.linspace(positions[0], positions[i], 4) for i in range(1, 5)}
    result = _label_positions(
        positions=positions,
        label_widths=np.full(5, 100),
        lower=np.array([0, -200]),
        upper=np.array([400, 200]),
        curves=curves,
    )
    # Only left-hand placements would extend beyond the figure for the first node.
    assert "left" not in result[0]


@pytest.mark.parametrize("reverse", [False, True])
def test_labels_avoid_edge_stubs_in_either_direction(reverse):
    curve = np.array([[0, 0], [60, 0], [80, 0], [140, 0]])
    curves = {(3, 0): curve[::-1]} if reverse else {(0, 3): curve}
    result = _label_positions(
        positions=np.array([[0, 0], [0, 20], [0, -20], [140, 0]]),
        label_widths=np.full(4, 100),
        lower=np.array([-200, -100]),
        upper=np.array([300, 100]),
        curves=curves,
    )
    assert result[0] == "middle left"


def test_edge_geometry_self_loop_has_equal_pixel_radii():
    curves, _ = _edge_geometry(
        positions=np.array([[100, 200]]),
        paths={},
        sources=np.array([0]),
        targets=np.array([0]),
    )
    loop = curves[0, 0]
    center = loop[0] + np.array([0, _LOOP_RADIUS])
    np.testing.assert_allclose(np.linalg.norm(loop - center, axis=1), _LOOP_RADIUS)


def _assert_separate(first, second):
    """Check that component bounds leave room for node labels."""
    node_size = np.array([60, _NODE_HEIGHT])
    assert np.any(first.max(axis=0) + node_size <= second.min(axis=0)) or np.any(
        second.max(axis=0) + node_size <= first.min(axis=0)
    )


def test_crossing_count_matches_pairwise_oracle():
    rng = np.random.default_rng(0)
    layers = [list(range(5)), list(range(5, 9))]
    for _ in range(20):
        connections = rng.random((5, 4)) < 0.5
        successors = [[] for _ in range(9)]
        for source, target in zip(*np.nonzero(connections), strict=True):
            successors[source].append(int(target + 5))
        rng.shuffle(layers[0])
        rng.shuffle(layers[1])
        order = np.empty(9, dtype=int)
        for layer in layers:
            order[layer] = np.arange(len(layer))
        edges = [
            (source, target) for source in range(5) for target in successors[source]
        ]
        expected = sum(
            (order[a] - order[c]) * (order[b] - order[d]) < 0
            for (a, b), (c, d) in combinations(edges, 2)
        )
        assert (
            _count_crossings(layers=layers, successors=successors, order=order)
            == expected
        )


def test_order_layers_removes_avoidable_crossing():
    layers = [[0, 1], [2, 3]]
    predecessors = [[], [], [0, 1], [0]]
    successors = [[2, 3], [2], [], []]
    original_order = np.array([0, 1, 0, 1])
    assert (
        _count_crossings(layers=layers, successors=successors, order=original_order)
        == 1
    )
    result = _order_layers(
        layers=layers, predecessors=predecessors, successors=successors
    )
    order = np.empty(4, dtype=int)
    for layer in result:
        order[layer] = np.arange(len(layer))
    assert _count_crossings(layers=result, successors=successors, order=order) == 0


@pytest.mark.parametrize(
    "edges",
    [
        [(0, 1), (1, 2), (2, 3)],
        [(0, 1), (1, 2), (0, 2), (2, 3)],
        [(0, 1), (1, 2), (2, 0), (0, 2), (2, 3)],
        [(0, 1), (2, 3)],
        [],
    ],
)
def test_layout_preserves_edges_and_separates_nodes(edges):
    connectivity = np.zeros((5, 5), dtype=bool)
    for source, target in edges:
        connectivity[source, target] = True
    original = connectivity.copy()
    positions, paths, _ = _layered_layout(connectivity=connectivity, node_width=60)
    np.testing.assert_array_equal(connectivity, original)
    assert positions.shape == (5, 2)
    assert np.isfinite(positions).all()
    assert len(np.unique(positions, axis=0)) == 5
    assert set(paths) == set(edges)
    for (source, target), path in paths.items():
        np.testing.assert_array_equal(path[0], positions[source])
        np.testing.assert_array_equal(path[-1], positions[target])
        assert np.isfinite(path).all()
    for first, second in combinations(positions, 2):
        assert (
            abs(first[0] - second[0]) >= 60 or abs(first[1] - second[1]) >= _NODE_HEIGHT
        )


def test_layout_routes_long_edge_around_intermediate_node():
    connectivity = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=bool)
    positions, paths, _ = _layered_layout(connectivity=connectivity, node_width=60)
    assert np.all(np.diff(positions[:, 0]) > 0)
    intermediate = paths[0, 2][paths[0, 2][:, 0] == positions[1, 0]]
    assert len(intermediate) == 1
    assert abs(intermediate[0, 1] - positions[1, 1]) >= 30


def test_layout_keeps_disconnected_components_separate():
    connectivity = np.zeros((7, 7), dtype=bool)
    connectivity[0, 1] = connectivity[1, 2] = connectivity[0, 2] = True
    connectivity[3, 4] = connectivity[4, 5] = connectivity[3, 5] = True
    positions, paths, _ = _layered_layout(connectivity=connectivity, node_width=60)
    first = np.vstack([path for (source, _), path in paths.items() if source < 3])
    second = np.vstack([path for (source, _), path in paths.items() if source >= 3])
    _assert_separate(first, second)
    _assert_separate(first, positions[[6]])
    _assert_separate(second, positions[[6]])


def test_layout_with_many_long_edges():
    n_nodes = 100
    connectivity = np.diag(np.ones(n_nodes - 1, dtype=bool), k=1)
    connectivity[:10, -10:] = True
    positions, paths, preserve_aspect = _layered_layout(
        connectivity=connectivity, node_width=60
    )
    assert preserve_aspect
    assert np.isfinite(positions).all()
    assert len(paths) == np.count_nonzero(connectivity)
    for (source, target), path in paths.items():
        np.testing.assert_array_equal(path[0], positions[source])
        np.testing.assert_array_equal(path[-1], positions[target])


def test_mixed_layout_uses_independent_axes():
    connectivity = np.zeros((101, 101), dtype=bool)
    connectivity[:100, :100] = np.diag(np.ones(99, dtype=bool), 1)
    connectivity[:10, 90:100] = True
    _, _, preserve_aspect = _layered_layout(connectivity=connectivity, node_width=60)
    assert not preserve_aspect


def test_layout_reserves_space_above_self_loops():
    edges = [
        (0, 1),
        (0, 2),
        (0, 4),
        (0, 5),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 4),
        (4, 5),
        (2, 2),
    ]
    connectivity = np.zeros((6, 6), dtype=bool)
    for source, target in edges:
        connectivity[source, target] = True
    positions, paths, _ = _layered_layout(connectivity=connectivity, node_width=60)
    assert (2, 2) not in paths
    route = paths[1, 3]
    above_loop = route[route[:, 0] == positions[2, 0]]
    assert len(above_loop) == 1
    clearance = above_loop[0, 1] - positions[2, 1]
    assert clearance >= 2 * _LOOP_RADIUS + 2 * _RECIPROCAL_OFFSET


def test_component_spacing_includes_edge_routes():
    rng = np.random.default_rng(0)
    connectivity = np.zeros((20, 20), dtype=bool)
    for start in (0, 10):
        component = np.triu(rng.random((10, 10)) < 0.25, 1)
        component |= np.diag(np.ones(9, dtype=bool), 1)
        connectivity[start : start + 10, start : start + 10] = component
    _, paths, _ = _layered_layout(connectivity=connectivity, node_width=60)
    first = np.vstack([path for (source, _), path in paths.items() if source < 10])
    second = np.vstack([path for (source, _), path in paths.items() if source >= 10])
    _assert_separate(first, second)


def test_isolated_nodes_use_both_dimensions():
    connectivity = np.zeros((40, 40), dtype=bool)
    positions, paths, _ = _layered_layout(connectivity=connectivity, node_width=60)
    assert not paths
    assert np.ptp(positions[:, 0]) > 0
    assert np.ptp(positions[:, 1]) > 0
    for first, second in combinations(positions, 2):
        _assert_separate(first[None, :], second[None, :])
