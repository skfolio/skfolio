"""Graph storage and observation-indexed histories."""

import copy
import json
import pickle
from itertools import repeat

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from sklearn.utils import _safe_indexing

from skfolio.containers import AssetPanel, Graph, GraphSeries, concat


def assert_graph_buffers_readonly(graph):
    adjacency = graph.adjacency
    buffers = (
        [adjacency.data, adjacency.indices, adjacency.indptr]
        if sps.issparse(adjacency)
        else [adjacency]
    )
    for buffer in [graph.nodes, *buffers]:
        assert not buffer.flags.writeable
        with pytest.raises(ValueError):
            buffer.flags.writeable = True


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_graph_ownership_and_roundtrip(sparse, dtype):
    names = np.array(["A", "B"])
    matrix = np.array([[0, 2], [-1, 0]], dtype=dtype)
    source = sps.csr_array(matrix) if sparse else matrix
    graph = Graph(source, nodes=names)
    assert copy.copy(graph) is graph
    names[0] = "C"
    if sparse:
        source.data[:] = 9
    else:
        source[:] = 9
    for restored in [
        graph,
        copy.deepcopy(graph),
        pickle.loads(pickle.dumps(graph, protocol=4)),
        pickle.loads(pickle.dumps(graph, protocol=5)),
    ]:
        assert restored.nodes.tolist() == ["A", "B"]
        assert_graph_buffers_readonly(restored)
        adjacency = restored.adjacency
        if restored is not graph:
            assert not np.shares_memory(restored.nodes, graph.nodes)
            original = graph.adjacency
            if sparse:
                for name in ["data", "indices", "indptr"]:
                    assert not np.shares_memory(
                        getattr(adjacency, name), getattr(original, name)
                    )
            else:
                assert not np.shares_memory(adjacency, original)
        np.testing.assert_array_equal(
            adjacency.toarray() if sparse else adjacency, [[0, 2], [-1, 0]]
        )
        assert adjacency.dtype == (np.float32 if dtype == np.float32 else np.float64)
        with pytest.raises(ValueError):
            restored.nodes[0] = "Z"
        with pytest.raises(AttributeError):
            restored.nodes = np.array(["Z", "B"])
        with pytest.raises(AttributeError):
            restored.adjacency = np.eye(2)
        with pytest.raises(ValueError):
            (adjacency.data if sparse else adjacency).flat[0] = 10
        if sparse:
            assert adjacency.indices.dtype == np.int32
            assert np.shares_memory(adjacency.data, restored.adjacency.data)
            adjacency.data = np.ones_like(adjacency.data)
            np.testing.assert_array_equal(
                restored.adjacency.toarray(), [[0, 2], [-1, 0]]
            )
    assert graph != copy.deepcopy(graph)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("mmap_mode", [None, "r"])
def test_graph_joblib_roundtrip(sparse, mmap_mode, tmp_path):
    adjacency = np.array([[0.0, 2.0], [1.0, 0.0]])
    graph = Graph(sps.csr_array(adjacency) if sparse else adjacency, nodes=["A", "B"])
    path = tmp_path / "graphs.pkl"
    joblib.dump(GraphSeries([graph, None, graph]), path)
    restored = joblib.load(path, mmap_mode=mmap_mode)
    assert restored[0] is restored[2]
    assert restored[1] is None
    assert_graph_buffers_readonly(restored[0])
    np.testing.assert_array_equal(restored[0].nodes, graph.nodes)
    result = restored[0].adjacency
    np.testing.assert_array_equal(result.toarray() if sparse else result, adjacency)


@pytest.mark.parametrize("weights", [[0, 1], [0, 0], [1, -1], [1e308, 1e308]])
@pytest.mark.parametrize("sparse_format", ["coo", "csr"])
def test_duplicate_sparse_coordinates(weights, sparse_format):
    matrix = (
        sps.coo_array((weights, ([0, 0], [1, 1])), shape=(2, 2))
        if sparse_format == "coo"
        else sps.csr_array((weights, [1, 1], [0, 2, 2]), shape=(2, 2))
    )
    with pytest.raises(ValueError, match="Duplicate"):
        Graph(matrix, nodes=["A", "B"])
    np.testing.assert_array_equal(matrix.data, weights)
    assert matrix.nnz == 2


def test_unsorted_sparse_input_with_explicit_zeros():
    matrix = sps.coo_array(([0, 3, 2], ([0, 1, 0], [0, 0, 1])), shape=(2, 2))
    graph = Graph(matrix, nodes=["A", "B"])
    np.testing.assert_array_equal(graph.adjacency.toarray(), [[0, 2], [3, 0]])
    assert graph.adjacency.has_canonical_format
    assert graph.adjacency.nnz == 2
    np.testing.assert_array_equal(matrix.data, [0, 3, 2])


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int64])
def test_edge_weights_follow_dense_dtype_conversion(dtype):
    graph = Graph.from_edges(
        iter([("A", "B", dtype(2)), ("B", "A", dtype(1))]), nodes=["A", "B"]
    )
    dense = Graph(np.array([[0, 2], [1, 0]], dtype=dtype), nodes=["A", "B"])
    assert graph.adjacency.dtype == dense.adjacency.dtype
    assert graph.adjacency.dtype == (np.float32 if dtype == np.float32 else np.float64)
    np.testing.assert_array_equal(graph.adjacency.toarray(), dense.adjacency)


@pytest.mark.parametrize(
    "nodes",
    [
        ["001", "1", "supplier"],
        np.array(["001", "1", "supplier"]),
        np.array(["001", "1", "supplier"], dtype=object),
        pd.Index(["001", "1", "supplier"], dtype="string"),
        np.array([np.str_("001"), np.str_("1"), np.str_("supplier")], dtype=object),
    ],
)
def test_string_node_identity_and_persistence(nodes, tmp_path):
    graph = Graph.from_edges([("1", "supplier", 7)], nodes=nodes)
    assert graph.nodes.tolist() == ["001", "1", "supplier"]
    assert graph.nodes.dtype.kind == "U"
    np.testing.assert_array_equal(
        graph.adjacency.toarray(), [[0, 0, 0], [0, 0, 7], [0, 0, 0]]
    )
    series = GraphSeries([graph], observations=[0])
    series.save(tmp_path / "graphs")
    restored = GraphSeries.load(tmp_path / "graphs")[0]
    np.testing.assert_array_equal(restored.nodes, graph.nodes)
    np.testing.assert_array_equal(
        restored.adjacency.toarray(), graph.adjacency.toarray()
    )
    assert_graph_buffers_readonly(restored)


@pytest.mark.parametrize(
    "nodes",
    [
        [1, 2],
        np.array([1, 2]),
        np.array([2**63, 2**63 + 1], dtype=np.uint64),
        pd.Index([1, 2], dtype="Int64"),
        [1.0, 2.0],
        [b"A", b"B"],
        np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
        [1, "supplier"],
    ],
)
def test_non_string_node_identifiers_are_rejected(nodes):
    with pytest.raises(ValueError, match="Convert numeric identifiers to strings"):
        Graph(np.eye(len(nodes)), nodes=nodes)
    with pytest.raises(ValueError, match="Convert numeric identifiers to strings"):
        Graph.from_edges([], nodes=nodes)


@pytest.mark.parametrize("nodes,match", [([["A"]], "1D"), (["A", "A"], "unique")])
def test_invalid_node_labels(nodes, match):
    with pytest.raises(ValueError, match=match):
        Graph(np.eye(len(nodes)), nodes=nodes)
    with pytest.raises(ValueError, match=match):
        Graph.from_edges([], nodes=nodes)


@pytest.mark.parametrize(
    "nodes",
    [
        [np.nan],
        [None],
        [pd.NA],
        [pd.NaT],
        np.array(["NaT"], dtype="datetime64[ns]"),
    ],
)
def test_missing_node_identifiers_are_rejected(nodes):
    with pytest.raises(ValueError, match="no missing values"):
        Graph([[1]], nodes=nodes)
    with pytest.raises(ValueError, match="no missing values"):
        Graph.from_edges([], nodes=nodes)


def test_missing_value_names_are_valid_strings():
    nodes = ["None", "nan", "NaT"]
    graph = Graph.from_edges([("None", "nan", 1)], nodes=nodes)
    assert graph.nodes.tolist() == nodes
    assert graph.adjacency[0, 1] == 1


@pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.0.0", reason="StringDType requires NumPy 2"
)
@pytest.mark.parametrize("from_edges", [False, True])
def test_variable_width_string_nodes_roundtrip(from_edges, tmp_path):
    nodes = np.array(["A", "Long supplier name"], dtype=np.dtypes.StringDType())
    graph = (
        Graph.from_edges([("A", "Long supplier name", 2)], nodes=nodes)
        if from_edges
        else Graph([[0, 2], [0, 0]], nodes=nodes)
    )
    nodes[0] = "Changed"
    assert graph.nodes.tolist() == ["A", "Long supplier name"]
    path = tmp_path / "graphs"
    GraphSeries([graph, graph], observations=[0, 1]).save(path)
    with np.load(path / "graphs" / "0.npz", allow_pickle=False) as buffers:
        np.testing.assert_array_equal(buffers["nodes"], graph.nodes)
    restored = GraphSeries.load(path)
    assert restored[0] is restored[1]
    np.testing.assert_array_equal(restored[0].nodes, graph.nodes)
    np.testing.assert_array_equal(
        restored[0].adjacency.toarray() if from_edges else restored[0].adjacency,
        [[0, 2], [0, 0]],
    )
    assert_graph_buffers_readonly(restored[0])


@pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.0.0", reason="StringDType requires NumPy 2"
)
@pytest.mark.parametrize("missing", [None, np.nan, pd.NA])
def test_variable_width_string_nodes_reject_missing_identifiers(missing):
    nodes = np.array(["A", missing], dtype=np.dtypes.StringDType(na_object=missing))
    with pytest.raises(ValueError, match="no missing values"):
        Graph(np.eye(2), nodes=nodes)
    with pytest.raises(ValueError, match="no missing values"):
        Graph.from_edges([], nodes=nodes)


def test_edges_and_validation():
    graph = Graph.from_edges([("A", "B", 2), ("B", "C", 0)], nodes=["A", "B", "C"])
    assert isinstance(graph.adjacency, sps.csr_array)
    assert graph.adjacency.nnz == 1
    assert Graph.from_edges([], nodes=["A"]).adjacency.nnz == 0
    with pytest.raises(ValueError, match="Duplicate"):
        Graph.from_edges([("A", "B", 0), ("A", "B", 1)], nodes=["A", "B"])
    with pytest.raises(ValueError, match="endpoint"):
        Graph.from_edges([("A", "B", 1)], nodes=["A"])
    with pytest.raises(ValueError, match="endpoint"):
        Graph.from_edges([(1, 2, 1)], nodes=["1", "2"])
    for names, adjacency in [
        (["A", "A"], np.eye(2)),
        (["A"], np.eye(2)),
        (["A"], [[np.inf]]),
        (["A"], [[1j]]),
    ]:
        with pytest.raises(ValueError):
            Graph(adjacency, nodes=names)


@pytest.mark.parametrize(
    "index,columns",
    [(["B", "A"], ["B", "A"]), (["A", "B"], ["B", "A"]), (["B", "A"], ["A", "B"])],
)
def test_dataframe_adjacency_rejects_misordered_labels(index, columns):
    frame = pd.DataFrame([[0.0, 7.0], [0.0, 0.0]], index=index, columns=columns)
    with pytest.raises(ValueError, match="match nodes in order"):
        Graph(frame, nodes=["A", "B"])


@pytest.mark.parametrize(
    "names",
    [
        ["A", "B"],
        pd.Index(["A", "B"], dtype=object),
        pd.Index(["A", "B"], dtype="string"),
        pd.CategoricalIndex(["A", "B"]),
    ],
)
@pytest.mark.parametrize("nodes", [None, ["A", "B"]])
def test_dataframe_adjacency_preserves_labelled_edges(names, nodes):
    frame = pd.DataFrame([[0.0, 7.0], [0.0, 0.0]], index=names, columns=names)
    graph = Graph(frame, nodes=nodes)
    np.testing.assert_array_equal(graph.nodes, ["A", "B"])
    np.testing.assert_array_equal(graph.adjacency, frame.to_numpy())


def test_dataframe_inferred_nodes_include_isolated_nodes():
    frame = pd.DataFrame(
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        index=["A", "Supplier", "Isolated"],
        columns=["A", "Supplier", "Isolated"],
    )
    graph = Graph(frame)
    np.testing.assert_array_equal(graph.nodes, frame.index)
    np.testing.assert_array_equal(graph.adjacency, frame.to_numpy())
    assert_graph_buffers_readonly(graph)


@pytest.mark.parametrize(
    "index, columns, match",
    [
        (["A", "B"], ["B", "A"], "match nodes in order"),
        (["A", "B"], ["A", "C"], "match nodes in order"),
        (["A", "A"], ["A", "A"], "adjacency index must be unique"),
        ([1, 2], [1, 2], "adjacency index must contain only strings"),
        (["A", None], ["A", None], "adjacency index must contain only strings"),
    ],
)
def test_dataframe_inferred_nodes_validation(index, columns, match):
    frame = pd.DataFrame(np.eye(2), index=index, columns=columns)
    with pytest.raises(ValueError, match=match):
        Graph(frame)


@pytest.mark.parametrize("adjacency", [[[0]], np.zeros((1, 1)), sps.csr_array((1, 1))])
def test_non_dataframe_adjacency_requires_nodes(adjacency):
    with pytest.raises(ValueError, match="nodes is required"):
        Graph(adjacency)


@pytest.mark.parametrize("axis", ["index", "columns"])
@pytest.mark.parametrize(
    "labels",
    [
        [1, 2],
        [b"1", b"2"],
        ["1", None],
        pd.Index(["1", pd.NA], dtype="string"),
    ],
)
def test_dataframe_labels_must_match_string_nodes(labels, axis):
    nodes = ["1", "2"]
    frame = pd.DataFrame(np.eye(2), index=nodes, columns=nodes)
    setattr(frame, axis, labels)
    with pytest.raises(ValueError, match="match nodes in order"):
        Graph(frame, nodes=nodes)


@pytest.mark.parametrize("dtype", ["Float32", "Float64", "Int64", "UInt64", "boolean"])
def test_nullable_numeric_dataframe_adjacency(dtype):
    frame = pd.DataFrame(
        [[0, 1], [0, 0]], index=["A", "B"], columns=["A", "B"], dtype=dtype
    )
    graph = Graph(frame, nodes=["A", "B"])
    assert graph.adjacency.dtype == (np.float32 if dtype == "Float32" else np.float64)
    np.testing.assert_array_equal(graph.adjacency, [[0, 1], [0, 0]])
    frame.iloc[0, 1] = pd.NA
    assert graph.adjacency[0, 1] == 1


@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_nullable_dataframe_mixed_numeric_columns(dtype):
    frame = pd.DataFrame(
        {
            "A": pd.Series([0, 1], index=["A", "B"], dtype="Float32"),
            "B": pd.Series([2, 0], index=["A", "B"], dtype=dtype),
        }
    )
    graph = Graph(frame, nodes=["A", "B"])
    assert graph.adjacency.dtype == (np.float32 if dtype == "float32" else np.float64)
    np.testing.assert_array_equal(graph.adjacency, [[0, 2], [1, 0]])


def test_sparse_dataframe_adjacency_stays_dense():
    frame = pd.DataFrame(
        [[0, 2], [1, 0]], index=["A", "B"], columns=["A", "B"], dtype=np.float32
    ).astype(pd.SparseDtype("float32", 0))
    graph = Graph(frame, nodes=["A", "B"])
    assert isinstance(graph.adjacency, np.ndarray)
    assert graph.adjacency.dtype == np.float32
    np.testing.assert_array_equal(graph.adjacency, [[0, 2], [1, 0]])


@pytest.mark.parametrize("value", [pd.NA, np.inf, -np.inf])
@pytest.mark.parametrize("dtype", ["Float32", "Float64"])
def test_nullable_numeric_dataframe_rejects_nonfinite_weights(value, dtype):
    frame = pd.DataFrame([[value]], index=["A"], columns=["A"], dtype=dtype)
    with pytest.raises(ValueError, match="finite"):
        Graph(frame, nodes=["A"])


@pytest.mark.parametrize("dtype", [object, "string"])
def test_dataframe_rejects_numeric_strings(dtype):
    frame = pd.DataFrame([["1"]], index=["A"], columns=["A"], dtype=dtype)
    with pytest.raises(ValueError, match="real numerical"):
        Graph(frame, nodes=["A"])


def test_series_default_observations_and_iteration():
    first = Graph([[1]], nodes=["A"])
    second = Graph([[2]], nodes=["B"])
    entries = [first, None, first, second]
    graphs = GraphSeries(iter(entries))
    np.testing.assert_array_equal(graphs.observations, [0, 1, 2, 3])
    assert list(graphs) == entries
    assert list(reversed(graphs)) == entries[::-1]
    reconstructed = GraphSeries(graphs, observations=None)
    np.testing.assert_array_equal(reconstructed.observations, graphs.observations)
    assert list(reconstructed) == entries
    empty = GraphSeries([])
    assert empty.observations.shape == (0,)
    assert list(empty) == []
    assert list(reversed(empty)) == []


def test_series_references_and_copying():
    graph = Graph(np.eye(2), nodes=["A", "supplier"])
    entries = [graph, None, graph]
    observations = np.arange(3)
    graphs = GraphSeries(iter(entries), observations=observations)
    entries[0] = None
    observations[0] = 99
    assert graphs[0] is graphs[-1] is graph
    np.testing.assert_array_equal(graphs.observations, [0, 1, 2])
    np.testing.assert_array_equal(graphs.missing_mask, [False, True, False])
    assert graphs.shape == (3,) and graphs.ndim == 1 and graphs.n_observations == 3
    assert repr(graphs) == "GraphSeries(n_observations=3, n_graphs=1)"
    assert GraphSeries(repeat(graph, 20))[-1] is graph
    shallow = graphs.copy()
    assert shallow[0] is graph
    shallow.observations[0] = 10
    assert graphs.observations[0] == 0
    for copied in [
        graphs.copy(deep=True),
        copy.deepcopy(graphs),
        pickle.loads(pickle.dumps(graphs)),
    ]:
        assert copied[0] is copied[2]
        assert copied[0] is not graph
        assert copied[1] is None
        assert not copied[0].adjacency.flags.writeable
        np.testing.assert_array_equal(copied.observations, graphs.observations)
        assert not np.shares_memory(copied.observations, graphs.observations)


@pytest.mark.parametrize("rows", [slice(1, None), slice(0, 0), [3]])
def test_series_copy_and_pickle_omit_unreferenced_graphs(rows):
    class UnreferencedGraph(Graph):
        def __reduce__(self):
            raise AssertionError("An unreferenced graph must not be serialized.")

        def __deepcopy__(self, memo):
            raise AssertionError("An unreferenced graph must not be copied.")

    unused = UnreferencedGraph([[1]], nodes=["unused"])
    first = Graph(np.eye(2), nodes=["A", "B"])
    second = Graph.from_edges([("A", "S", 1)], nodes=["A", "S"])
    selected = GraphSeries([unused, first, second, None, first])[rows]
    # References also remain shared across separately selected series and graphs.
    payload = (selected, selected[:1], first)
    restored_payloads = [copy.deepcopy(payload)]
    restored_payloads.extend(
        pickle.loads(pickle.dumps(payload, protocol=protocol)) for protocol in [4, 5]
    )
    for restored, prefix, copied_first in restored_payloads:
        np.testing.assert_array_equal(restored.observations, selected.observations)
        np.testing.assert_array_equal(restored.missing_mask, selected.missing_mask)
        assert copied_first is not first
        for i in range(len(selected)):
            graph = restored[i]
            if selected[i] is first:
                assert graph is copied_first
            if graph is not None:
                assert graph is not selected[i]
                assert_graph_buffers_readonly(graph)
        if len(prefix):
            assert prefix[0] is restored[0]


@pytest.mark.parametrize(
    "rows,expected",
    [
        (slice(1, 3), [1, 2]),
        (slice(None, None, -1), [3, 2, 1, 0]),
        ([2], [2]),
        ([2, 0, 2], [2, 0, 2]),
        (np.array([3, 0], dtype=np.uint64), [3, 0]),
        ([True, False, True, False], [0, 2]),
        (np.array([], dtype=int), []),
    ],
)
def test_series_selection(rows, expected):
    first, second = Graph([[1]], nodes=["A"]), Graph([[2]], nodes=["B"])
    entries = [first, None, first, second]
    graphs = GraphSeries(entries)
    for selected in [graphs[rows], graphs[rows, ...], _safe_indexing(graphs, rows)]:
        assert isinstance(selected, GraphSeries)
        np.testing.assert_array_equal(selected.observations, expected)
        assert list(selected) == [entries[position] for position in expected]
        for i, position in enumerate(expected):
            assert selected[i] is entries[position]
    assert graphs[1:][[2, 1]][0] is second


@pytest.mark.parametrize("position", [0, 1, 2, -1, -4, np.int64(1), np.uint64(2)])
def test_series_scalar_indexing(position):
    first, second = Graph([[1]], nodes=["A"]), Graph([[2]], nodes=["B"])
    entries = [first, None, first, second]
    graphs = GraphSeries(entries, observations=[10, 20, 30, 40])
    expected = entries[position]
    assert graphs[position] is expected
    assert graphs[position, ...] is expected
    assert _safe_indexing(graphs, position) is expected


@pytest.mark.parametrize(
    "rows", ["A", (slice(None), slice(None)), (..., [0]), ([0], ..., ...)]
)
def test_series_rejects_nonpositional_selection(rows):
    with pytest.raises(TypeError):
        GraphSeries([None])[rows]


@pytest.mark.parametrize(
    "rows",
    [
        np.array([2**64 - 1], dtype=np.uint64),
        np.array([2**64 - 2], dtype=np.uint64),
        np.array([2**63 - 1], dtype=np.int64),
        np.array([-3], dtype=np.int64),
    ],
)
def test_containers_reject_out_of_bounds_observation_indices(rows):
    graphs = GraphSeries([None, None], observations=[0, 1])
    panel = AssetPanel(
        fields={"returns": [[0.01], [0.02]]},
        observations=[0, 1],
        asset_names=["A"],
    )
    for container in [graphs, panel, panel[:]]:
        with pytest.raises(IndexError, match="out of bounds"):
            container[rows]


@pytest.mark.parametrize("key", [None, (None, ...)])
def test_containers_reject_none_indexing(key):
    graphs = GraphSeries([None, None])
    panel = AssetPanel(
        fields={"returns": [[0.01], [0.02]]},
        observations=[0, 1],
        asset_names=["A"],
    )
    for container in [graphs, panel, panel[:]]:
        with pytest.raises(TypeError, match="None is not an observation selector"):
            container[key]
        np.testing.assert_array_equal(container[:].observations, [0, 1])
    np.testing.assert_array_equal(panel.isel(observations=None).observations, [0, 1])


def test_series_construction_and_positions():
    with pytest.raises(ValueError, match="same length"):
        GraphSeries([None], observations=[])
    with pytest.raises(ValueError, match="1D"):
        GraphSeries([None], observations=[[0]])
    with pytest.raises(TypeError, match="Graph or None"):
        GraphSeries([object()])
    graphs = GraphSeries([None])
    for position in [True, np.bool_(False), 0.0, "0"]:
        with pytest.raises(TypeError, match="positional indexing"):
            graphs[position]
    for position in [1, -2, np.uint64(2**64 - 1), 2**100]:
        with pytest.raises(IndexError):
            graphs[position]
    dates = pd.date_range("2024-01-01", periods=2)
    dated = GraphSeries([None, None], observations=dates)
    assert np.issubdtype(dated.observations.dtype, np.datetime64)
    assert len(GraphSeries([], observations=[])) == 0


@pytest.mark.skipif(
    np.lib.NumpyVersion(np.__version__) < "2.0.0", reason="StringDType requires NumPy 2"
)
@pytest.mark.parametrize("container_type", [GraphSeries, AssetPanel])
@pytest.mark.parametrize(
    "labels,expected",
    [
        (["t0", "t1"], np.array(["t0", "t1"])),
        (
            ["2024-01-01", "2024-01-02"],
            np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
        ),
    ],
)
def test_containers_variable_width_observations_roundtrip(
    container_type, labels, expected, tmp_path
):
    observations = np.array(labels, dtype=np.dtypes.StringDType())
    if container_type is GraphSeries:
        container = GraphSeries([None, None], observations=observations)
    else:
        container = AssetPanel(
            fields={"returns": [[0.01], [0.02]]},
            observations=observations,
            asset_names=["A"],
        )
    observations[0] = "changed"
    np.testing.assert_array_equal(container.observations, expected)
    path = tmp_path / "container"
    container.save(path)
    restored = container_type.load(path)
    np.testing.assert_array_equal(restored.observations, expected)


def test_series_save_roundtrip_and_compaction(tmp_path):
    dense = Graph(np.eye(2, dtype=np.float32), nodes=["A", "B"])
    sparse = Graph.from_edges([("A", "S", 2)], nodes=["A", "S"])
    unused = Graph([[1]], nodes=["unused"])
    graphs = GraphSeries([unused, dense, None, sparse, dense])[1:]
    path = tmp_path / "graphs"
    graphs.save(path)
    metadata = json.loads((path / "_metadata.json").read_text())
    assert metadata == {
        "container": "GraphSeries",
        "version": 1,
        "n_observations": 4,
        "n_graphs": 2,
    }
    assert len(list((path / "graphs").glob("*.npz"))) == 2
    restored = GraphSeries.load(path)
    assert_graph_buffers_readonly(restored[0])
    assert_graph_buffers_readonly(restored[2])
    assert restored[0] is restored[3]
    assert restored[1] is None
    assert restored[0].adjacency.dtype == np.float32
    assert not restored[0].adjacency.flags.writeable
    assert isinstance(restored[2].adjacency, sps.csr_array)
    assert restored[2].adjacency.indices.dtype == np.int32
    assert not restored[2].adjacency.data.flags.writeable
    np.testing.assert_array_equal(restored.observations, graphs.observations)
    np.testing.assert_array_equal(
        restored[2].adjacency.toarray(), sparse.adjacency.toarray()
    )
    with pytest.raises(FileExistsError, match="overwrite=True"):
        graphs.save(path)
    graphs[[3, 3]].save(path, overwrite=True)
    restored = GraphSeries.load(path)
    np.testing.assert_array_equal(restored.observations, [4, 4])
    assert restored[0] is restored[1]
    assert repr(restored) == "GraphSeries(n_observations=2, n_graphs=1)"


@pytest.mark.parametrize("n", [0, 2])
def test_series_save_missing_graphs(tmp_path, n):
    graphs = GraphSeries([None] * n)
    graphs.save(tmp_path / "graphs")
    restored = GraphSeries.load(tmp_path / "graphs")
    assert len(restored) == n
    assert restored.missing_mask.all()
    np.testing.assert_array_equal(restored.observations, graphs.observations)
    assert not list((tmp_path / "graphs" / "graphs").glob("*.npz"))


def test_container_persistence_identity_and_legacy_panel(tmp_path):
    panel = AssetPanel(fields={"x": [[1.0]]}, observations=[0], asset_names=["A"])
    graphs = GraphSeries([None])
    panel_path, graph_path = tmp_path / "panel", tmp_path / "graphs"
    panel.save(panel_path)
    graphs.save(graph_path)
    for saved_path, wrong_type, wrong_container in [
        (panel_path, GraphSeries, graphs),
        (graph_path, AssetPanel, panel),
    ]:
        with pytest.raises(
            ValueError, match=r"AssetPanel.*GraphSeries|GraphSeries.*AssetPanel"
        ):
            wrong_type.load(saved_path)
        with pytest.raises(
            ValueError, match=r"AssetPanel.*GraphSeries|GraphSeries.*AssetPanel"
        ):
            wrong_container.save(saved_path, overwrite=True)
        assert (saved_path / "_metadata.json").is_file()
    metadata_path = panel_path / "_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    assert metadata.pop("container") == "AssetPanel"
    assert metadata["version"] == 1
    metadata_path.write_text(json.dumps(metadata))
    np.testing.assert_array_equal(AssetPanel.load(panel_path)["x"], panel["x"])
    with pytest.raises(ValueError, match=r"GraphSeries.*AssetPanel"):
        GraphSeries.load(panel_path)


def test_series_save_preserves_unrelated_paths(tmp_path):
    graphs = GraphSeries([None])
    path = tmp_path / "unrelated"
    path.mkdir()
    sentinel = path / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(ValueError, match="not a saved"):
        graphs.save(path, overwrite=True)
    assert sentinel.read_text() == "keep"
    with pytest.raises(ValueError, match="not a directory"):
        graphs.save(sentinel, overwrite=True)
    with pytest.raises(FileNotFoundError, match="Not a saved GraphSeries"):
        GraphSeries.load(path)


@pytest.mark.parametrize(
    "references", [np.array([0]), np.array([-2]), np.array([[0]]), np.array([0.0])]
)
def test_series_rejects_invalid_saved_references(tmp_path, references):
    GraphSeries([None]).save(tmp_path / "graphs")
    np.save(tmp_path / "graphs" / "references.npy", references, allow_pickle=False)
    with pytest.raises(ValueError, match="references"):
        GraphSeries.load(tmp_path / "graphs")


def test_series_concat():
    first = Graph.from_edges([("A", "S", 1)], nodes=["A", "S"])
    second = Graph(np.eye(1, dtype=np.float32), nodes=["B"])
    left = GraphSeries([first, None])
    right = GraphSeries([second, first], observations=[2, 3])
    combined = concat(iter([left, right]), verify_observations=True)
    assert combined[0] is combined[3] is first
    assert combined[2] is second
    assert combined[1] is None
    np.testing.assert_array_equal(combined.observations, [0, 1, 2, 3])
    np.testing.assert_array_equal(concat([left, left]).observations, [0, 1, 0, 1])
    with pytest.raises(ValueError, match="unique"):
        concat([left, left], verify_observations=True)
    with pytest.raises(TypeError, match="single GraphSeries"):
        concat(left)
    panel = AssetPanel(fields={"x": [[1.0]]}, observations=[0], asset_names=["A"])
    for items in [[left, panel], [panel, left]]:
        with pytest.raises(TypeError, match="GraphSeries"):
            concat(items)
