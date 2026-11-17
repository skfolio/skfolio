import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps

from skfolio.containers import Graph, GraphSeries
from skfolio.portfolio import _utils as portfolio_utils
from skfolio.portfolio import dependency_exposure


@pytest.fixture(params=["dense", "sparse"])
def graph(request):
    graph = Graph.from_edges(
        [
            ("Token1", "ProtocolA", 1),
            ("Token2", "ProtocolA", 3),
            ("Token2", "BridgeA", 1),
            ("Token3", "BridgeA", 1),
            ("ProtocolA", "BridgeA", 1),
        ],
        nodes=["Token1", "Token2", "Token3", "ProtocolA", "BridgeA"],
    )
    if request.param == "dense":
        return Graph(graph.adjacency.toarray(), nodes=graph.nodes)
    return graph


def test_dependency_exposure_example(graph):
    weights = np.array([0.3, 0.2, 0.5])
    weights.flags.writeable = False
    actual = dependency_exposure(graph, asset_names=graph.nodes[:3], weights=weights)
    expected = pd.DataFrame(
        [
            [0.3, 0, 0, 0.3],
            [0.2, 0, 0, 0.2],
            [0.5, 0, 0, 0.5],
            [0, 0.5, 0, 0.5],
            [0, 0.7, 0.3, 1],
        ],
        columns=["weight", "direct", "indirect", "total"],
        index=pd.Index(graph.nodes, name="dependency"),
        dtype=float,
    )
    pd.testing.assert_frame_equal(actual, expected)
    np.testing.assert_array_equal(weights, [0.3, 0.2, 0.5])
    adjacency = graph.adjacency
    buffer = adjacency.data if sps.issparse(adjacency) else adjacency
    assert not buffer.flags.writeable


@pytest.mark.parametrize("max_hops", [1, 2, 3, None, np.int64(2)])
def test_dependency_exposure_depth(graph, max_hops):
    result = dependency_exposure(
        graph, asset_names=graph.nodes[:3], weights=[0.3, 0.2, 0.5], max_hops=max_hops
    )
    expected_indirect = 0 if max_hops == 1 else 0.3
    assert result.loc["BridgeA", "indirect"] == expected_indirect
    assert result.loc["BridgeA", "total"] == pytest.approx(0.7 + expected_indirect)


@pytest.mark.parametrize("max_hops", [1, 2, 4, None])
def test_dependency_exposure_cycles_and_isolated_assets(max_hops):
    graph = Graph.from_edges(
        [("B", "A", 1), ("A", "B", 1), ("A", "A", 2), ("B", "S", 1)],
        nodes=["S", "B", "Isolated", "A"],
    )
    result = dependency_exposure(
        graph,
        asset_names=["A", "B", "Isolated"],
        weights=[0.3, 0.2, 0.5],
        max_hops=max_hops,
    )
    np.testing.assert_allclose(result.loc["A"], [0.3, 0.2, 0, 0.5])
    np.testing.assert_allclose(result.loc["B"], [0.2, 0.3, 0, 0.5])
    np.testing.assert_allclose(result.loc["Isolated"], [0.5, 0, 0, 0.5])
    assert result.loc["S", "total"] == pytest.approx(0.2 if max_hops == 1 else 0.5)


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_dependency_exposure_ignores_positive_edge_magnitudes(graph, scale):
    scaled_graph = Graph(graph.adjacency * scale, nodes=graph.nodes)
    kwargs = dict(asset_names=graph.nodes[:3], weights=[1.2, -0.4, 0.5])
    expected = dependency_exposure(graph, **kwargs)
    actual = dependency_exposure(scaled_graph, **kwargs)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc["ProtocolA", "total"] == pytest.approx(0.8)
    gross = dependency_exposure(
        graph, asset_names=graph.nodes[:3], weights=[1.2, 0.4, 0.5]
    )
    assert gross.loc["ProtocolA", "total"] == pytest.approx(1.6)


def test_dependency_exposure_changing_graphs_and_missing_data():
    first = Graph.from_edges([("A", "S", 1)], nodes=["A", "S"])
    second = Graph.from_edges([("B", "T", 1)], nodes=["B", "T"])
    graphs = GraphSeries(
        [first, None, second, first, first], observations=[2, 2, 4, 5, 6]
    )
    weights = [[0.8, 0], [0.5, 0.5], [0, 1], [np.nan, np.nan], [0.25, 0]]
    result = dependency_exposure(graphs, asset_names=["A", "B"], weights=weights)
    expected = np.zeros((5, 4, 4))
    expected[0, 0] = [0.8, 0, 0, 0.8]
    expected[0, 1] = [0, 0.8, 0, 0.8]
    expected[1] = np.nan
    expected[2, 2] = [1, 0, 0, 1]
    expected[2, 3] = [0, 1, 0, 1]
    expected[3] = np.nan
    expected[4, 0] = [0.25, 0, 0, 0.25]
    expected[4, 1] = [0, 0.25, 0, 0.25]
    np.testing.assert_allclose(result.to_numpy().reshape(5, 4, 4), expected)
    expected_index = pd.MultiIndex.from_product(
        [[2, 2, 4, 5, 6], ["A", "S", "B", "T"]], names=["observation", "dependency"]
    )
    pd.testing.assert_index_equal(result.index, expected_index)


def test_dependency_exposure_constant_weights_and_selection(graph):
    graphs = GraphSeries([graph, graph], observations=[1, 2])
    kwargs = dict(asset_names=graph.nodes[:3], nodes=["BridgeA", "Token2"])
    constant = dependency_exposure(graphs, weights=[0.3, 0.2, 0.5], **kwargs)
    repeated = dependency_exposure(graphs, weights=[[0.3, 0.2, 0.5]] * 2, **kwargs)
    pd.testing.assert_frame_equal(constant, repeated)
    full = dependency_exposure(
        graph, asset_names=graph.nodes[:3], weights=[0.3, 0.2, 0.5]
    )
    pd.testing.assert_frame_equal(constant.loc[1], full.loc[["BridgeA", "Token2"]])


def test_dependency_exposure_selected_node_absent_from_snapshot():
    first = Graph.from_edges([("A", "S", 1)], nodes=["A", "B", "C", "S"])
    second = Graph.from_edges([("A", "T", 1)], nodes=["A", "B", "C", "T"])
    result = dependency_exposure(
        GraphSeries([first, second]),
        asset_names=["A", "B", "C"],
        weights=[0.4, 0.3, 0.3],
        nodes=["T", "S"],
    )
    np.testing.assert_array_equal(result["total"], [0, 0.4, 0.4, 0])


@pytest.mark.parametrize(
    "nodes, expected_sources",
    [(["BridgeA", "ProtocolA"], [0, 1]), (["BridgeA"], [4])],
)
def test_dependency_exposure_reuses_graph_and_chunks_sources(
    graph, monkeypatch, nodes, expected_sources
):
    original_dijkstra = portfolio_utils.dijkstra
    calls = []

    def record_call(**kwargs):
        calls.append(kwargs["indices"].copy())
        return original_dijkstra(**kwargs)

    monkeypatch.setattr(portfolio_utils, "dijkstra", record_call)
    monkeypatch.setattr(portfolio_utils, "_MAX_ARRAY_ELEMENTS", graph.n_nodes)
    result = dependency_exposure(
        GraphSeries([graph, None, graph, graph]),
        asset_names=graph.nodes[:3],
        weights=[[0.3, 0.2, 0], [0, 0, 0], [0.4, 0.1, 0], [0.8, 0.1, 0]],
        nodes=nodes,
    )
    assert len(calls) == len(expected_sources)
    np.testing.assert_array_equal(np.concatenate(calls), expected_sources)
    np.testing.assert_allclose(
        result.xs("BridgeA", level="dependency")["total"], [0.5, np.nan, 0.5, 0.9]
    )


@pytest.mark.parametrize("max_hops", [1, 2, None])
def test_dependency_exposure_chunks_reverse_traversal(graph, monkeypatch, max_hops):
    graphs = GraphSeries([graph, None, graph])
    kwargs = dict(
        asset_names=graph.nodes[:3],
        weights=[[0.3, -0.2, 0.5], [0, 0, 0], [0.2, 0.3, 0.5]],
        max_hops=max_hops,
    )
    full = dependency_exposure(graphs, **kwargs)
    original_dijkstra = portfolio_utils.dijkstra
    calls = []

    def record_call(**kwargs):
        calls.append(kwargs["indices"].copy())
        return original_dijkstra(**kwargs)

    monkeypatch.setattr(portfolio_utils, "dijkstra", record_call)
    monkeypatch.setattr(portfolio_utils, "_MAX_ARRAY_ELEMENTS", graph.n_nodes)
    selected = dependency_exposure(graphs, nodes=["BridgeA", "Token2"], **kwargs)
    pd.testing.assert_frame_equal(selected, full.reindex(selected.index))
    assert len(calls) == 2
    np.testing.assert_array_equal(np.concatenate(calls), [4, 1])


@pytest.mark.parametrize("max_hops", [1, 2, 3, None])
@pytest.mark.parametrize("report_indices", [None, [5, 7]])
def test_dependency_exposure_matches_boolean_reachability(max_hops, report_indices):
    rng = np.random.default_rng(12)
    adjacency = rng.random((9, 9)) < 0.2
    nodes = [f"N{i}" for i in range(9)]
    source_indices = [7, 0, 5]
    weights = np.array([0.5, -0.2, 0.7])
    reachability = adjacency.copy()
    power = adjacency.copy()
    for _ in range(1, 9 if max_hops is None else max_hops):
        power = power @ adjacency
        reachability |= power
    np.fill_diagonal(reachability, False)
    direct = adjacency.copy()
    np.fill_diagonal(direct, False)
    own = np.eye(9)[source_indices]
    expected = np.column_stack(
        [
            weights @ own,
            weights @ direct[source_indices],
            weights @ (reachability & ~direct)[source_indices],
            weights @ (own + reachability[source_indices]),
        ]
    )
    result = dependency_exposure(
        Graph(sps.csr_array(adjacency), nodes=nodes),
        asset_names=np.asarray(nodes)[source_indices],
        weights=weights,
        max_hops=max_hops,
        nodes=None if report_indices is None else np.asarray(nodes)[report_indices],
    )
    if report_indices is not None:
        expected = expected[report_indices]
    np.testing.assert_allclose(result, expected, atol=1e-15)


@pytest.mark.parametrize("max_hops", [0, -1, True, False, 1.5, 2.0, "2"])
def test_dependency_exposure_invalid_depth(graph, max_hops):
    with pytest.raises(ValueError, match="max_hops"):
        dependency_exposure(
            graph, asset_names=["Token1"], weights=[1], max_hops=max_hops
        )


@pytest.mark.parametrize(
    "weights", [[1, 0], [[1]], [np.inf], [-np.inf], [1 + 2j], [[[1]]]]
)
def test_dependency_exposure_invalid_snapshot_weights(graph, weights):
    with pytest.raises((ValueError, TypeError)):
        dependency_exposure(graph, asset_names=["Token1"], weights=weights)


@pytest.mark.parametrize("weights", [[1, np.nan], [[np.nan, 0]], [[1, 0]]])
def test_dependency_exposure_invalid_history_weights(graph, weights):
    with pytest.raises(ValueError):
        dependency_exposure(
            GraphSeries([graph, graph]), asset_names=graph.nodes[:2], weights=weights
        )


@pytest.mark.parametrize(
    "asset_names", [["Token1", "Token1"], [1, 2], ["Token1", None]]
)
def test_dependency_exposure_invalid_asset_names(graph, asset_names):
    with pytest.raises(ValueError, match="asset_names"):
        dependency_exposure(graph, asset_names=asset_names, weights=[0.5, 0.5])


@pytest.mark.parametrize("nodes", [["unknown"], ["Token1", "Token1"], [1], "Token1"])
def test_dependency_exposure_invalid_nodes(graph, nodes):
    with pytest.raises(ValueError, match=r"[Nn]odes"):
        dependency_exposure(graph, asset_names=["Token1"], weights=[1], nodes=nodes)


def test_dependency_exposure_missing_assets(graph):
    result = dependency_exposure(
        graph, asset_names=["Token1", "missing"], weights=[1, 0]
    )
    assert result.loc["BridgeA", "total"] == 1
    with pytest.raises(ValueError, match="missing"):
        dependency_exposure(
            graph,
            asset_names=["Token1", "missing"],
            weights=[1, 1e-12],
            nodes=["BridgeA"],
        )
    with pytest.raises(ValueError, match=r"observation.*42.*missing"):
        dependency_exposure(
            GraphSeries([graph, graph], observations=[41, 42]),
            asset_names=["Token1", "missing"],
            weights=[[1, 0], [0.9, 0.1]],
        )


@pytest.mark.parametrize("argument", ["asset_names", "nodes"])
def test_dependency_exposure_truncates_missing_names(argument):
    graph = Graph(np.zeros((1, 1)), nodes=["A"])
    names = [f"missing_{i}" for i in range(100)]
    kwargs = dict(asset_names=["A"], weights=[1])
    kwargs[argument] = names
    if argument == "asset_names":
        kwargs["weights"] = np.ones(len(names))
    with pytest.raises(ValueError) as error:
        dependency_exposure(graph, **kwargs)
    message = str(error.value)
    assert "(100)" in message
    assert repr(names[0]) in message and repr(names[4]) in message
    assert repr(names[5]) not in message and repr(names[-1]) not in message
    assert "..." in message


@pytest.mark.parametrize("dense", [False, True])
def test_dependency_exposure_rejects_negative_edges_outside_held_assets(dense):
    adjacency = np.array([[0, 1, 0], [0, 0, 0], [0, -1, 0]])
    graph = Graph(
        adjacency if dense else sps.csr_array(adjacency), nodes=["A", "S", "B"]
    )
    with pytest.raises(ValueError, match="nonnegative"):
        dependency_exposure(graph, asset_names=["A"], weights=[1], nodes=["A"])


def test_dependency_exposure_missing_graphs_and_weights(graph):
    failed = dependency_exposure(graph, asset_names=["Token1"], weights=[np.nan])
    assert failed.isna().all().all()
    missing = GraphSeries([None, None], observations=[10, 20])
    with pytest.raises(ValueError, match="nodes is required"):
        dependency_exposure(missing, asset_names=["Token1"], weights=[1])
    result = dependency_exposure(
        missing, asset_names=["Token1"], weights=[1], nodes=["S"]
    )
    assert result.isna().all().all()
    assert result.index.tolist() == [(10, "S"), (20, "S")]


def test_dependency_exposure_empty_inputs(graph):
    for weights in ([1], np.empty((0, 1))):
        result = dependency_exposure(
            GraphSeries([]), asset_names=["Token1"], weights=weights
        )
        assert result.empty
        assert result.index.names == ["observation", "dependency"]
        assert (result.dtypes == np.float64).all()
    empty_graph = Graph(np.empty((0, 0)), nodes=[])
    result = dependency_exposure(empty_graph, asset_names=["A"], weights=[0])
    assert result.empty
    with pytest.raises(ValueError, match="absent from graph"):
        dependency_exposure(empty_graph, asset_names=["A"], weights=[1])
    result = dependency_exposure(graph, asset_names=[], weights=[])
    assert (result == 0).all().all()
    assert dependency_exposure(
        graph, asset_names=["Token1"], weights=[1], nodes=[]
    ).empty


@pytest.mark.parametrize("weights", [[np.inf, 0], [0, np.nan]])
def test_dependency_exposure_empty_history_still_validates_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        dependency_exposure(GraphSeries([]), asset_names=["A", "B"], weights=weights)


def test_dependency_exposure_invalid_graph():
    with pytest.raises(TypeError, match="Graph or GraphSeries"):
        dependency_exposure(None, asset_names=["A"], weights=[1])
