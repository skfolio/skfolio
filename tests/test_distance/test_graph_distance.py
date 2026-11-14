"""Graph distances, numerical behavior and metadata routing."""

from itertools import repeat

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from sklearn import clone, config_context
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.utils.estimator_checks import estimator_checks_generator

from skfolio.cluster import HierarchicalClustering
from skfolio.containers import Graph, GraphSeries
from skfolio.distance import GraphDistance
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    NestedClustersOptimization,
    SchurComplementary,
)
from skfolio.prior import BasePrior, EmpiricalPrior, ReturnDistribution


def series_for(graph, n=40):
    return GraphSeries(repeat(graph, n))


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("hops,decay", [(1, 0.5), (2, 0.5), (5, 0), (4, 1)])
def test_matches_explicit_powers(sparse, normalize, dtype, hops, decay):
    matrix = np.array(
        [[0, 1, 2, 0], [0, 0, 1, 3], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=dtype
    )
    graph = Graph(
        sps.csr_array(matrix) if sparse else matrix, nodes=["A", "B", "U", "V"]
    )
    model = GraphDistance(
        max_hops=hops, decay=decay, normalize_adjacency=normalize
    ).fit(pd.DataFrame(columns=["A", "B"]), graphs=series_for(graph))
    transition = matrix.astype(np.float64)
    if normalize:
        sums = transition.sum(axis=1, keepdims=True)
        transition /= np.where(sums > 0, sums, 1)
    profiles = sum(
        decay ** (h - 1) * np.linalg.matrix_power(transition, h)[:2]
        for h in range(1, hops + 1)
    )
    profiles /= np.linalg.norm(profiles, axis=1, keepdims=True)
    expected = profiles @ profiles.T
    np.testing.assert_allclose(model.codependence_, expected, atol=1e-14)
    np.testing.assert_allclose(model.distance_**2, 1 - expected, atol=1e-14)
    assert np.linalg.eigvalsh(model.codependence_).min() >= -1e-14
    np.testing.assert_array_equal(
        graph.adjacency.toarray() if sparse else graph.adjacency, matrix
    )


def test_latest_graph_targets_and_intermediates():
    graph = Graph.from_edges(
        [("A", "C", 1), ("C", "S", 1), ("B", "S", 1)], nodes=["A", "B", "C", "S"]
    )
    graphs = series_for(graph)
    X = pd.DataFrame(np.nan, index=range(100), columns=["B", "A"])
    model = GraphDistance().fit(X, graphs=graphs)
    np.testing.assert_allclose(
        model.codependence_, [[1, 1 / np.sqrt(5)], [1 / np.sqrt(5), 1]]
    )
    assert np.isfinite(model.distance_).all()
    assert model.n_features_in_ == 2
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    model.fit(pd.DataFrame(columns=["D"]), graphs=graphs)
    assert np.isnan(model.distance_).all()
    with pytest.raises(ValueError, match="unique"):
        model.fit(X[["A", "A"]], graphs=graphs)
    graphs = GraphSeries([graph] * 39 + [None], observations=np.arange(40))
    with pytest.raises(ValueError, match="last observation"):
        model.fit(X, graphs=graphs)
    model.fit(X, graphs=graphs[:39])


@pytest.mark.parametrize("prefit", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_partial_fit_uses_latest_graph_and_fit_resets(prefit, sparse):
    X = pd.DataFrame(np.nan, index=range(6), columns=["A", "B", "C"])
    first = Graph(np.eye(3), nodes=X.columns)
    adjacency = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0]], dtype=float
    )
    latest = Graph(
        sps.csr_array(adjacency) if sparse else adjacency, nodes=["A", "B", "C", "S"]
    )
    graphs = GraphSeries([first] * 5 + [latest])
    model = GraphDistance()
    first_method = "fit" if prefit else "partial_fit"
    assert getattr(model, first_method)(X.iloc[:3], graphs=graphs[:3]) is model
    original_distance = model.distance_.copy()
    assert model.partial_fit(X.iloc[3:], graphs=graphs[3:]) is model
    reference = GraphDistance().fit(X, graphs=graphs)
    np.testing.assert_allclose(model.distance_, reference.distance_)
    np.testing.assert_allclose(model.codependence_, reference.codependence_)
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    assert model.n_features_in_ == 3
    assert not np.allclose(model.distance_, original_distance)

    model.fit(X[["C", "A"]], graphs=graphs)
    reference = GraphDistance().fit(X[["C", "A"]], graphs=graphs)
    np.testing.assert_allclose(model.distance_, reference.distance_)
    np.testing.assert_array_equal(model.feature_names_in_, ["C", "A"])
    assert model.n_features_in_ == 2
    assert not hasattr(clone(model), "distance_")


@pytest.mark.parametrize("columns", [["B", "A"], ["A", "C"], ["A"], ["A", "B", "C"]])
def test_partial_fit_requires_consistent_asset_columns(columns):
    graphs = GraphSeries([Graph(np.eye(3), nodes=["A", "B", "C"])])
    model = GraphDistance().partial_fit(pd.DataFrame(columns=["A", "B"]), graphs=graphs)
    original_distance = model.distance_.copy()
    with pytest.raises(ValueError, match="feature names"):
        model.partial_fit(pd.DataFrame(columns=columns), graphs=graphs)
    np.testing.assert_array_equal(model.distance_, original_distance)


@pytest.mark.parametrize("graphs", [None, GraphSeries([]), GraphSeries([None])])
def test_partial_fit_does_not_reuse_a_previous_graph(graphs):
    X = pd.DataFrame(columns=["A", "B"])
    initial = GraphSeries([Graph(np.eye(2), nodes=X.columns)])
    model = GraphDistance().fit(X, graphs=initial)
    original_distance = model.distance_.copy()
    with pytest.raises(ValueError, match="graphs"):
        model.partial_fit(X, graphs=graphs)
    np.testing.assert_array_equal(model.distance_, original_distance)


@pytest.mark.parametrize("sparse", [False, True])
def test_extreme_weights(sparse):
    for normalize in [False, True]:
        matrix = np.array([[0, 0, 1e-300], [0, 0, 1e300], [0, 0, 0]])
        graph = Graph(
            sps.csr_array(matrix) if sparse else matrix, nodes=["A", "B", "S"]
        )
        model = GraphDistance(max_hops=1, normalize_adjacency=normalize).fit(
            pd.DataFrame(columns=["A", "B"]), graphs=series_for(graph)
        )
        np.testing.assert_array_equal(model.distance_, 0)
    huge = Graph(
        sps.csr_array(np.full((2, 2), 1e300)) if sparse else np.full((2, 2), 1e300),
        nodes=["A", "B"],
    )
    with pytest.raises(ValueError, match="overflow"):
        GraphDistance(normalize_adjacency=False).fit(
            pd.DataFrame(columns=["A", "B"]), graphs=series_for(huge)
        )
    GraphDistance().fit(pd.DataFrame(columns=["A", "B"]), graphs=series_for(huge))


@pytest.mark.parametrize(
    "params",
    [
        {"max_hops": True},
        {"max_hops": 0},
        {"max_hops": 1.5},
        {"decay": -1},
        {"decay": np.inf},
        {"decay": True},
        {"normalize_adjacency": 1},
    ],
)
def test_parameter_validation(params):
    estimator = GraphDistance(**params)
    clone(estimator)
    with pytest.raises(ValueError):
        estimator.fit(
            pd.DataFrame(columns=["A", "B"]),
            graphs=series_for(Graph(np.eye(2), nodes=["A", "B"])),
        )


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_unavailable_profiles_preserve_asset_schema(sparse, normalize):
    names = ["A", "B", "supplier", "isolated"]
    missing = "missing"
    matrix = np.array([[0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    graph = Graph(sps.csr_array(matrix) if sparse else matrix, nodes=names)
    requested = [missing, names[1], names[3], names[0]]
    X = pd.DataFrame(np.nan, index=range(5), columns=requested)
    model = GraphDistance(normalize_adjacency=normalize).fit(
        X, graphs=series_for(graph)
    )
    reference = GraphDistance(normalize_adjacency=normalize).fit(
        X.iloc[:, [1, 3]], graphs=series_for(graph)
    )
    assert model.n_features_in_ == 4
    np.testing.assert_array_equal(model.feature_names_in_, requested)
    for actual, expected in [
        (model.codependence_, reference.codependence_),
        (model.distance_, reference.distance_),
    ]:
        assert actual.shape == (4, 4)
        assert np.isnan(actual[[0, 2], :]).all()
        assert np.isnan(actual[:, [0, 2]]).all()
        np.testing.assert_allclose(actual[np.ix_([1, 3], [1, 3])], expected)
    np.testing.assert_array_equal(np.diag(model.distance_)[[1, 3]], 0)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("n_nodes", [0, 2])
def test_no_available_profiles_and_refitting(sparse, n_nodes):
    X = pd.DataFrame(columns=["A", "B"])
    model = GraphDistance().fit(X, graphs=series_for(Graph(np.eye(2), nodes=X.columns)))
    matrix = np.zeros((n_nodes, n_nodes))
    graph = Graph(
        sps.csr_array(matrix) if sparse else matrix, nodes=X.columns[:n_nodes]
    )
    model.fit(X, graphs=series_for(graph))
    assert model.n_features_in_ == 2
    assert np.isnan(model.codependence_).all()
    assert np.isnan(model.distance_).all()


def test_negative_weights_are_rejected_outside_requested_assets():
    graph = Graph([[1, 0], [0, -1]], nodes=["A", "B"])
    for names in [["A"], ["missing"]]:
        with pytest.raises(ValueError, match="non-negative"):
            GraphDistance().fit(pd.DataFrame(columns=names), graphs=series_for(graph))


def test_values_of_X_are_not_read():
    class NamesOnlyDataFrame(pd.DataFrame):
        def __array__(self, *args, **kwargs):
            raise AssertionError("X values must not be converted")

        def copy(self, *args, **kwargs):
            raise AssertionError("X must not be copied")

    X = NamesOnlyDataFrame({"B": [object()], "A": [object()]})
    graph = Graph([[0, 1], [1, 0]], nodes=["A", "B"])
    model = GraphDistance().fit(X, graphs=series_for(graph))
    np.testing.assert_array_equal(model.feature_names_in_, ["B", "A"])
    assert model.distance_.shape == (2, 2)


def test_standalone_fit_resets_feature_names():
    model = GraphDistance()
    for names in [["A", "B"], ["20", "10"], ["B", "A"]]:
        model.fit(
            pd.DataFrame(columns=names),
            graphs=series_for(Graph([[0, 1], [1, 0]], nodes=names)),
        )
        assert model.n_features_in_ == 2
        np.testing.assert_array_equal(model.feature_names_in_, names)


@pytest.mark.parametrize(
    "columns",
    [
        np.array(["A", "B"]),
        np.array(["A", "B"], dtype=object),
        pd.Index(["A", "B"], dtype="string"),
    ],
)
def test_string_column_types_without_return_observations(columns):
    X = pd.DataFrame(columns=columns)
    original_columns = X.columns
    graph = Graph([[0, 1], [1, 0]], nodes=["A", "B"])
    model = GraphDistance(max_hops=1).fit(X, graphs=series_for(graph))
    assert X.columns is original_columns
    np.testing.assert_array_equal(model.feature_names_in_, ["A", "B"])
    np.testing.assert_array_equal(model.codependence_, np.eye(2))
    np.testing.assert_array_equal(model.distance_, [[0, 1], [1, 0]])


@pytest.mark.parametrize(
    "columns",
    [
        [1, 2],
        np.array([1, 2], dtype=object),
        pd.Index([1, 2], dtype="Int64"),
        [1.0, 2.0],
        ["A", 2],
        ["A", None],
        ["A", np.nan],
        pd.Index(["A", pd.NA], dtype="string"),
    ],
)
def test_invalid_asset_names(columns):
    graph = Graph([[0, 1], [1, 0]], nodes=["A", "B"])
    with pytest.raises((TypeError, ValueError), match="string"):
        GraphDistance().fit(pd.DataFrame(columns=columns), graphs=series_for(graph))


def test_explicitly_converted_identifiers_match_assets():
    asset_names = np.array([1, 2]).astype(str)
    graph = Graph([[0, 1], [1, 0]], nodes=asset_names)
    model = GraphDistance(max_hops=1).fit(
        pd.DataFrame(columns=asset_names), graphs=series_for(graph)
    )
    np.testing.assert_array_equal(model.codependence_, np.eye(2))
    np.testing.assert_array_equal(model.distance_, [[0, 1], [1, 0]])


def test_empty_requested_universe():
    graph = Graph([[0, 1], [1, 0]], nodes=["A", "B"])
    with pytest.raises(ValueError, match="string column names"):
        GraphDistance().fit(pd.DataFrame(), graphs=series_for(graph))


def test_missing_metadata_explains_routing():
    with pytest.raises(ValueError, match=r"graphs.*enable_metadata_routing"):
        GraphDistance().fit(pd.DataFrame({"A": [1]}))


def test_required_graph_inputs():
    X = pd.DataFrame(columns=["A"])
    graphs = series_for(Graph([[1]], nodes=["A"]))
    with pytest.raises(TypeError, match="X"):
        GraphDistance().fit(graphs=graphs)
    for invalid in [None, [[1.0]]]:
        with pytest.raises(ValueError, match="DataFrame"):
            GraphDistance().fit(invalid, graphs=graphs)
    with pytest.raises(TypeError, match="GraphSeries"):
        GraphDistance().fit(X, graphs=[graphs[0]])
    with pytest.raises(ValueError, match="at least one observation"):
        GraphDistance().fit(X, graphs=GraphSeries([]))
    with pytest.raises(ValueError, match="last observation"):
        GraphDistance().fit(X, graphs=GraphSeries([None]))


def test_applicable_sklearn_checks():
    # Generic fitting checks omit required graph metadata and use unlabeled arrays.
    # These input contracts are covered explicitly above.
    applicable = {
        "check_estimator_cloneable",
        "check_parameters_default_constructible",
        "check_estimator_tags_renamed",
        "check_valid_tag_types",
        "check_estimator_repr",
        "check_no_attributes_set_in_init",
        "check_get_params_invariance",
        "check_set_params",
        "check_mixin_order",
    }
    for estimator, check in estimator_checks_generator(GraphDistance()):
        if check.func.__name__ in applicable:
            check(estimator)


@pytest.mark.parametrize(
    "optimizer,params",
    [
        (HierarchicalRiskParity, {"distance_from_prior": False}),
        (HierarchicalRiskParity, {"distance_from_prior": True}),
        (SchurComplementary, {"distance_from_prior": False}),
        (SchurComplementary, {"distance_from_prior": True}),
        (HierarchicalEqualRiskContribution, {}),
    ],
)
def test_one_investable_asset(optimizer, params):
    graph = Graph.from_edges([], nodes=["S"])
    graphs = series_for(graph)
    X = pd.DataFrame(np.ones((40, 3)), columns=["A", "B", "D"])
    with config_context(enable_metadata_routing=True):
        model = optimizer(
            prior_estimator=SubsetPrior(mask=(True, False, False)),
            distance_estimator=GraphDistance(),
            **params,
        )
        model.fit(X, graphs=graphs)
        np.testing.assert_array_equal(model.weights_, [1, 0, 0])
        if params.get("distance_from_prior", True):
            assert model.distance_estimator_ is None
            # Prior-based paths skip distance fitting when only one asset is needed.
            model.fit(X)
            np.testing.assert_array_equal(model.weights_, [1, 0, 0])
        else:
            assert np.isnan(model.distance_estimator_.distance_).all()
            with pytest.raises(ValueError, match="requires graphs"):
                model.fit(X)


def test_nested_clusters_with_graph_distance():
    X = pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, (40, 3)), columns=["A", "B", "C"]
    )
    graph = Graph.from_edges(
        [("A", "Supplier1", 1), ("B", "Supplier1", 1), ("C", "Supplier2", 1)],
        nodes=["A", "B", "C", "Supplier1", "Supplier2"],
    )
    with config_context(enable_metadata_routing=True):
        model = NestedClustersOptimization(
            distance_estimator=GraphDistance(),
            clustering_estimator=HierarchicalClustering(max_clusters=2),
            inner_estimator=EqualWeighted(),
            outer_estimator=EqualWeighted(),
            cv=2,
        ).fit(X, graphs=series_for(graph, len(X)))
    np.testing.assert_allclose(model.weights_, [0.25, 0.25, 0.5])
    np.testing.assert_allclose(
        model.distance_estimator_.distance_, [[0, 0, 1], [0, 0, 1], [1, 1, 0]]
    )


def test_large_sparse_node_universe():
    graph = Graph.from_edges(
        [("0", "100", 1), ("1", "19999", 1), ("100", "19999", 1)],
        nodes=np.arange(20000).astype(str),
    )
    model = GraphDistance().fit(
        pd.DataFrame(columns=["0", "1"]), graphs=series_for(graph)
    )
    np.testing.assert_allclose(model.codependence_[0, 1], 1 / np.sqrt(5))


class SubsetPrior(BasePrior):
    def __init__(self, mask=(True, True, False), n_scenarios=70):
        self.mask = mask
        self.n_scenarios = n_scenarios

    def fit(self, X, y=None):
        values = np.random.default_rng(13).normal(
            0, 0.01, (self.n_scenarios, len(self.mask))
        )
        covariance = np.eye(len(self.mask)) * 0.0001
        covariance[~np.array(self.mask), :] = np.nan
        covariance[:, ~np.array(self.mask)] = np.nan
        self.return_distribution_ = ReturnDistribution(
            mu=np.zeros(len(self.mask)), covariance=covariance, returns=values
        )
        return self


@pytest.mark.parametrize(
    "optimizer,params",
    [
        (HierarchicalRiskParity, {"distance_from_prior": False}),
        (HierarchicalRiskParity, {"distance_from_prior": True}),
        (SchurComplementary, {"distance_from_prior": False}),
        (SchurComplementary, {"distance_from_prior": True}),
        (HierarchicalEqualRiskContribution, {}),
    ],
)
@pytest.mark.parametrize("numeric_strings", [False, True])
@pytest.mark.parametrize("isolated", [False, True])
def test_prior_selection_and_scenario_counts(
    optimizer, params, numeric_strings, isolated
):
    names = ["10", "20", "30"] if numeric_strings else ["A", "B", "D"]
    X = pd.DataFrame(np.random.default_rng(12).normal(0, 0.01, (40, 3)), columns=names)
    graph = (
        Graph([[0, 1, 0], [1, 0, 0], [0, 0, 0]], nodes=names)
        if isolated
        else Graph([[0, 1], [1, 0]], nodes=names[:2])
    )
    graphs = series_for(graph)
    with config_context(enable_metadata_routing=True):
        estimator = optimizer(
            prior_estimator=SubsetPrior(),
            distance_estimator=GraphDistance(),
            **params,
        )
        estimator.fit(X, graphs=graphs)
        np.testing.assert_allclose(estimator.weights_, [0.5, 0.5, 0])
        distance = estimator.distance_estimator_.distance_
        if params.get("distance_from_prior", True):
            assert distance.shape == (2, 2)
        else:
            assert distance.shape == (3, 3)
            assert np.isnan(distance[2]).all()
            assert np.isnan(distance[:, 2]).all()
        with pytest.raises(ValueError, match="same number of observations"):
            estimator.fit(X, graphs=graphs[:20])
        missing = series_for(
            Graph([[0, 0], [0, 1]], nodes=names[:2])
            if isolated
            else Graph([[1]], nodes=[names[1]])
        )
        with pytest.raises(
            ValueError, match="unavailable for prior-investable"
        ) as error:
            estimator.fit(X, graphs=missing)
        assert str([names[0]]) in str(error.value)


@pytest.mark.parametrize("optimizer", [HierarchicalRiskParity, SchurComplementary])
@pytest.mark.parametrize("distance_from_prior", [False, True])
@pytest.mark.parametrize("prefit", [False, True])
@pytest.mark.parametrize("metadata_name", ["graphs", "dependencies"])
def test_online_graph_distance(optimizer, distance_from_prior, prefit, metadata_name):
    X = pd.DataFrame(
        np.random.default_rng(42).normal(0, 0.01, (40, 3)), columns=["A", "B", "C"]
    )
    X.iloc[:20, 2] = np.nan
    first = Graph(np.eye(3), nodes=X.columns)
    latest = Graph.from_edges(
        [("A", "S", 1), ("S", "C", 1), ("B", "C", 1), ("C", "A", 1)],
        nodes=["A", "B", "C", "S"],
    )
    graphs = GraphSeries([first] * 20 + [latest] * 20)
    with config_context(enable_metadata_routing=True):
        distance = GraphDistance()
        if metadata_name != "graphs":
            distance.set_fit_request(graphs=metadata_name)
            distance.set_partial_fit_request(graphs=metadata_name)
        model = optimizer(
            prior_estimator=EmpiricalPrior(
                mu_estimator=EWMu(half_life=5, min_observations=2),
                covariance_estimator=EWCovariance(half_life=5, min_observations=2),
            ),
            distance_estimator=distance,
            distance_from_prior=distance_from_prior,
        )
        first_method = "fit" if prefit else "partial_fit"
        getattr(model, first_method)(X.iloc[:20], **{metadata_name: graphs[:20]})
        assert model.weights_[2] == 0
        fitted_distance = model.distance_estimator_
        distribution = model.prior_estimator_.return_distribution_
        with pytest.raises(ValueError, match="same number of observations"):
            model.partial_fit(X.iloc[20:], **{metadata_name: graphs[21:]})
        assert model.prior_estimator_.return_distribution_ is distribution

        model.partial_fit(X.iloc[20:], **{metadata_name: graphs[20:]})
        reference = GraphDistance().fit(X, graphs=graphs)
        np.testing.assert_allclose(
            model.distance_estimator_.distance_, reference.distance_
        )
        np.testing.assert_array_equal(
            model.distance_estimator_.feature_names_in_, X.columns
        )
        assert np.isfinite(model.weights_).all()
        assert model.weights_.sum() == pytest.approx(1)
        assert model.weights_[2] > 0
        if distance_from_prior:
            assert model.distance_estimator_ is not fitted_distance
        else:
            assert model.distance_estimator_ is fitted_distance


@pytest.mark.parametrize("distance_from_prior", [False, True])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_cross_validation_uses_training_graph(distance_from_prior, n_jobs):
    X = pd.DataFrame(
        np.random.default_rng(1).normal(0, 0.01, (40, 2)), columns=["A", "B"]
    )
    good = Graph([[0, 1], [1, 0]], nodes=["A", "B"])
    # No training fold sees the missing final graph. Preselecting it would fail.
    graphs = GraphSeries([good] * 39 + [None], observations=X.index)
    with config_context(enable_metadata_routing=True):
        search = GridSearchCV(
            HierarchicalRiskParity(
                distance_estimator=GraphDistance(),
                distance_from_prior=distance_from_prior,
            ),
            {"distance_estimator__decay": [0.25, 0.5]},
            cv=TimeSeriesSplit(2),
            refit=False,
            error_score="raise",
            n_jobs=n_jobs,
        )
        search.fit(X, graphs=graphs)
    assert np.isfinite(search.cv_results_["mean_test_score"]).all()


@pytest.mark.parametrize(
    "optimizer,params",
    [
        (HierarchicalEqualRiskContribution, {}),
        (HierarchicalRiskParity, {"distance_from_prior": True}),
        (HierarchicalRiskParity, {"distance_from_prior": False}),
        (SchurComplementary, {"distance_from_prior": True}),
        (SchurComplementary, {"distance_from_prior": False}),
    ],
)
def test_excluded_assets_remain_dependency_intermediaries(optimizer, params):
    X = pd.DataFrame(
        np.random.default_rng(12).normal(size=(40, 3)), columns=list("ABC")
    )
    graph = Graph.from_edges(
        [("A", "C", 1), ("C", "Supplier", 1), ("B", "Supplier", 1)],
        nodes=["A", "B", "C", "Supplier"],
    )
    graphs = series_for(graph)
    with config_context(enable_metadata_routing=True):
        model = optimizer(
            prior_estimator=SubsetPrior(),
            distance_estimator=GraphDistance(max_hops=2, decay=1),
            **params,
        ).fit(X, graphs=graphs)
    assert model.weights_[2] == 0
    # A reaches the supplier through C even though C cannot receive an allocation.
    assert model.distance_estimator_.codependence_[0, 1] == pytest.approx(
        1 / np.sqrt(2)
    )
    assert graphs[-1] is graph
