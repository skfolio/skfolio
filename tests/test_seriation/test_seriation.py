"""Test pairwise identity, hierarchical compatibility and spectral geometry."""

import pickle

import numpy as np
import pandas as pd
import pytest
import scipy.cluster.hierarchy as sch
import scipy.linalg as scl
import scipy.spatial.distance as scd
from sklearn.base import clone
from sklearn.utils import get_tags
from sklearn.utils.estimator_checks import estimator_checks_generator

from skfolio.cluster import HierarchicalClustering
from skfolio.distance import PearsonDistance
from skfolio.seriation import HierarchicalSeriation, SpectralSeriation
from skfolio.seriation._spectral import _order_coordinates


def snapshot(distance, available, names=None):
    result = np.full(distance.shape, np.nan)
    result[np.ix_(available, available)] = distance[np.ix_(available, available)]
    if names is not None:
        return pd.DataFrame(result, index=names, columns=names)
    return result


@pytest.fixture
def distance():
    returns = np.random.default_rng(45).normal(size=(80, 7))
    correlation = np.corrcoef(returns.T)
    return np.sqrt(np.maximum(0, (1 - correlation) / 2))


@pytest.mark.parametrize("estimator", [HierarchicalSeriation(), SpectralSeriation()])
def test_full_schema_and_trivial_universes(estimator, distance):
    assert get_tags(estimator).input_tags.pairwise
    assert get_tags(estimator).input_tags.allow_nan
    names = np.array(list("gfedcba"))
    for eligible in [[], [3], [0, 2, 4, 6], list(range(7))]:
        mask = np.isin(np.arange(7), eligible)
        estimator.fit(snapshot(distance, mask, names))
        np.testing.assert_array_equal(np.sort(estimator.ordering_), eligible)
        np.testing.assert_array_equal(estimator.investable_mask_, mask)
        np.testing.assert_array_equal(estimator.feature_names_in_, names)
        assert estimator.n_features_in_ == 7
        assert not hasattr(clone(estimator), "ordering_")
        if isinstance(estimator, HierarchicalSeriation) and len(eligible) < 2:
            assert estimator.hierarchical_clustering_estimator_ is None
            assert estimator.ordered_linkage_matrix_.shape == (0, 4)
        if isinstance(estimator, SpectralSeriation):
            assert np.isnan(estimator.coordinates_[~mask]).all()
            if len(eligible) < 2:
                assert estimator.eigenvalue_multiplicity_ == 0
                assert np.isnan(estimator.spectral_gap_)


@pytest.mark.parametrize("optimal", [True, False])
def test_hierarchical_matches_existing_tree(distance, optimal):
    expected = HierarchicalClustering().fit(distance)
    linkage = expected.linkage_matrix_
    if optimal:
        linkage = sch.optimal_leaf_ordering(linkage, expected.condensed_distance_)
    model = HierarchicalSeriation(optimal_ordering=optimal).fit(distance)
    np.testing.assert_array_equal(model.ordering_, sch.leaves_list(linkage))
    np.testing.assert_allclose(model.ordered_linkage_matrix_, linkage)


@pytest.mark.parametrize("n_available", [0, 1, 3])
@pytest.mark.parametrize("optimal_ordering", [None, "False", 0, 1])
def test_hierarchical_invalid_optimal_ordering(distance, n_available, optimal_ordering):
    matrix = snapshot(distance, np.arange(len(distance)) < n_available)
    model = HierarchicalSeriation(optimal_ordering=optimal_ordering)
    with pytest.raises(ValueError, match="optimal_ordering must be a boolean"):
        model.fit(matrix)


@pytest.mark.parametrize("n_available", [0, 1, 3])
def test_hierarchical_invalid_clustering_estimator(distance, n_available):
    matrix = snapshot(distance, np.arange(len(distance)) < n_available)
    model = HierarchicalSeriation(hierarchical_clustering_estimator=PearsonDistance())
    with pytest.raises(TypeError, match=r"Expected type.*HierarchicalClustering"):
        model.fit(matrix)


@pytest.mark.parametrize("size", [2, 3, 6])
@pytest.mark.parametrize("equal_distance", [0.0, 1.0])
def test_hierarchical_small_and_equal_distances(size, equal_distance):
    distance = equal_distance * (np.ones((size, size)) - np.eye(size))
    model = HierarchicalSeriation().fit(distance)
    np.testing.assert_array_equal(np.sort(model.ordering_), np.arange(size))
    assert model.ordered_linkage_matrix_.shape == (size - 1, 4)


@pytest.mark.parametrize("size", [2, 3, 6, 7, 12])
def test_cotton_dense_affinity_equivalence(size):
    returns = np.random.default_rng(size).normal(size=(90, size))
    correlation = np.corrcoef(returns.T)
    distance = np.sqrt(np.maximum(0, (1 - correlation) / 2))
    affinity = (1 + correlation) / 2
    laplacian = np.diag(affinity.sum(axis=1)) - affinity
    values, vectors = scl.eigh(laplacian)
    model = SpectralSeriation().fit(distance)
    vector = model.coordinates_
    assert abs(vector @ vectors[:, 1]) == pytest.approx(1)
    assert vector.sum() == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(laplacian @ vector, values[1] * vector, atol=1e-12)
    assert model.eigenvalue_multiplicity_ == 1


def test_general_distances_and_scale(distance):
    distance = scd.squareform(scd.pdist(np.random.default_rng(1).normal(size=(7, 3))))
    base = SpectralSeriation().fit(distance)
    for scale in [1e-100, 0.25, 1e100, 1e308 / distance.max()]:
        model = SpectralSeriation().fit(scale * distance)
        np.testing.assert_array_equal(model.ordering_, base.ordering_)
        np.testing.assert_allclose(model.coordinates_, base.coordinates_, atol=1e-12)
        assert model.spectral_gap_ == pytest.approx(base.spectral_gap_)


@pytest.mark.parametrize("size", [2, 3, 6])
@pytest.mark.parametrize("equal_distance", [0.0, 1.0])
def test_spectral_small_and_equal_distances(size, equal_distance):
    distance = equal_distance * (np.ones((size, size)) - np.eye(size))
    model = SpectralSeriation().fit(distance)
    expected = np.full(size, -1.0)
    expected[0] = size - 1
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(model.coordinates_, expected, atol=1e-12)
    np.testing.assert_array_equal(model.ordering_, [*range(1, size), 0])
    assert model.eigenvalue_multiplicity_ == size - 1
    if size == 2 or equal_distance == 0:
        assert np.isnan(model.spectral_gap_)
    else:
        assert model.spectral_gap_ == pytest.approx(0, abs=1e-12)


def test_spectral_disconnected_laplacian():
    distance = np.zeros((4, 4))
    distance[0, 1] = distance[1, 0] = 1
    model = SpectralSeriation().fit(distance)
    np.testing.assert_allclose(
        model.coordinates_, np.array([1, -1, 0, 0]) / np.sqrt(2), atol=1e-12
    )
    np.testing.assert_array_equal(model.ordering_, [1, 2, 3, 0])
    assert model.eigenvalue_multiplicity_ == 1
    assert model.spectral_gap_ == pytest.approx(1)


def test_spectral_relative_gap():
    # The squared-distance Laplacian has eigenvalues 0, 1 and 3.
    distance = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    model = SpectralSeriation().fit(distance)
    assert model.spectral_gap_ == pytest.approx(2 / 3)


@pytest.mark.parametrize("equal_distance", [0, 0.7])
def test_repeated_eigenspace_is_basis_invariant(monkeypatch, equal_distance):
    distance = np.full((6, 6), equal_distance)
    np.fill_diagonal(distance, 0)
    baseline = SpectralSeriation().fit(distance)
    original_eigh = scl.eigh
    original_helmert = scl.helmert
    rotation, _ = np.linalg.qr(np.random.default_rng(4).normal(size=(5, 5)))

    def rotated_eigh(matrix):
        values, vectors = original_eigh(matrix)
        vectors[:, -5:] = vectors[:, -5:] @ rotation
        return values, vectors

    def rotated_helmert(n, full=False):
        return rotation @ original_helmert(n, full=full)

    monkeypatch.setattr("skfolio.seriation._spectral.scl.eigh", rotated_eigh)
    monkeypatch.setattr("skfolio.seriation._spectral.scl.helmert", rotated_helmert)
    actual = SpectralSeriation().fit(distance)
    np.testing.assert_allclose(actual.coordinates_, baseline.coordinates_, atol=1e-12)
    np.testing.assert_array_equal(actual.ordering_, baseline.ordering_)
    assert actual.eigenvalue_multiplicity_ == 5
    if equal_distance == 0:
        assert np.isnan(actual.spectral_gap_)
    else:
        assert actual.spectral_gap_ == 0


@pytest.mark.parametrize("online", [False, True])
def test_repeated_eigenspace_with_distinct_lower_eigenvalue(monkeypatch, online):
    distance = np.full((4, 4), np.sqrt(0.5))
    distance[:3, :3] = np.sqrt(0.75)
    np.fill_diagonal(distance, 0)
    baseline = SpectralSeriation()
    actual = SpectralSeriation()
    if online:
        returns = np.random.default_rng(42).normal(size=(4, 50))
        initial = np.sqrt(np.maximum(0, (1 - np.corrcoef(returns)) / 2))
        baseline.fit(initial)
        actual.fit(initial)
        reference = baseline.coordinates_[:3]
        expected = np.append(reference - reference.mean(), 0)
        baseline.partial_fit(distance)
    else:
        expected = np.array([2.0, -1.0, -1.0, 0.0])
        baseline.fit(distance)
    # The leading eigenspace is the zero-sum subspace of the first three assets.
    expected /= np.linalg.norm(expected)
    original = scl.eigh
    rotation, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(2, 2)))

    def rotated_eigh(matrix):
        values, vectors = original(matrix)
        vectors[:, -2:] = vectors[:, -2:] @ rotation
        return values, vectors

    monkeypatch.setattr("skfolio.seriation._spectral.scl.eigh", rotated_eigh)
    if online:
        actual.partial_fit(distance)
    else:
        actual.fit(distance)
    np.testing.assert_allclose(actual.coordinates_, expected, atol=1e-12)
    np.testing.assert_allclose(actual.coordinates_, baseline.coordinates_, atol=1e-12)
    np.testing.assert_array_equal(actual.ordering_, baseline.ordering_)
    assert actual.eigenvalue_multiplicity_ == 2
    assert actual.spectral_gap_ == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("names", [list("bcafed"), list("fedcba")])
@pytest.mark.parametrize("geometry", ["zero", "equal", "disconnected"])
def test_spectral_names_do_not_affect_ordering(names, geometry):
    distance = np.zeros((6, 6))
    if geometry == "equal":
        distance = np.ones((6, 6)) - np.eye(6)
    elif geometry == "disconnected":
        distance[0, 1] = distance[1, 0] = 1
    baseline = SpectralSeriation().fit(distance)
    named = SpectralSeriation().fit(pd.DataFrame(distance, index=names, columns=names))
    np.testing.assert_array_equal(named.ordering_, baseline.ordering_)
    np.testing.assert_allclose(named.coordinates_, baseline.coordinates_, atol=1e-12)

    # Names must not influence online alignment or ties after exits and reentry.
    for mask in [
        np.array([False, True, True, True, False, True]),
        np.ones(6, dtype=bool),
    ]:
        baseline.partial_fit(snapshot(distance, mask))
        named.partial_fit(snapshot(distance, mask, names))
        np.testing.assert_array_equal(named.ordering_, baseline.ordering_)
        np.testing.assert_allclose(
            named.coordinates_, baseline.coordinates_, atol=1e-12
        )


@pytest.mark.parametrize("equal_distance", [0.0, 1.0])
def test_online_alignment_and_reentry(distance, equal_distance):
    model = SpectralSeriation().fit(distance)
    previous = model.coordinates_.copy()
    model.partial_fit(distance)
    np.testing.assert_allclose(model.coordinates_, previous, atol=1e-12)
    # A repeated eigenspace contains every centered previous coordinate.
    equal = equal_distance * (np.ones_like(distance) - np.eye(len(distance)))
    model.partial_fit(equal)
    np.testing.assert_allclose(model.coordinates_, previous, atol=1e-12)
    mask = np.array([True, False, True, False, True, True, False])
    model.partial_fit(snapshot(equal, mask))
    expected = previous[mask] - previous[mask].mean()
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(model.coordinates_[mask], expected, atol=1e-12)
    model.partial_fit(equal)
    reference = np.zeros(7)
    reference[mask] = expected
    np.testing.assert_allclose(model.coordinates_, reference, atol=1e-12)
    model.partial_fit(np.full((7, 7), np.nan))
    model.partial_fit(distance)
    np.testing.assert_allclose(
        model.coordinates_, SpectralSeriation().fit(distance).coordinates_
    )


def test_tie_groups_use_first_coordinate_not_adjacent_gaps():
    coordinate = np.array([0, 0.75, 1.5, 2.25])
    order = _order_coordinates(coordinate, np.arange(4), np.array([3, 2, 1, 0]), 1)
    np.testing.assert_array_equal(order, [1, 0, 3, 2])


@pytest.mark.parametrize("estimator", [HierarchicalSeriation(), SpectralSeriation()])
@pytest.mark.parametrize(
    "invalid",
    [
        "asymmetric",
        "negative",
        "diagonal",
        "infinity",
        "nan_block",
        "names",
        "nonsquare",
    ],
)
def test_invalid_matrices(estimator, distance, invalid):
    matrix = distance.copy()
    if invalid == "asymmetric":
        matrix[0, 1] += 0.1
    elif invalid == "negative":
        matrix[0, 1] = matrix[1, 0] = -0.1
    elif invalid == "diagonal":
        matrix[0, 0] = 0.1
    elif invalid == "infinity":
        matrix[0, 0] = np.inf
    elif invalid == "nan_block":
        matrix[0, 1] = np.nan
    elif invalid == "names":
        matrix = pd.DataFrame(matrix, index=list("abcdefg"), columns=list("bacdefg"))
    else:
        matrix = matrix[:-1]
    with pytest.raises(ValueError):
        estimator.fit(matrix)


def test_online_schema_and_explicit_restart(distance):
    names = list("abcdefg")
    matrix = pd.DataFrame(distance, index=names, columns=names)
    model = SpectralSeriation().fit(matrix)
    with pytest.raises(ValueError, match="feature names"):
        model.partial_fit(matrix.iloc[::-1, ::-1])
    with pytest.raises(ValueError, match="feature"):
        model.partial_fit(matrix.iloc[:-1, :-1])
    model.fit(matrix.iloc[:-1, :-1])
    assert model.n_features_in_ == 6


@pytest.mark.parametrize("estimator", [HierarchicalSeriation(), SpectralSeriation()])
def test_applicable_sklearn_checks(estimator):
    # Other generic checks manufacture linear kernels or arbitrary missing entries.
    # Those are not distance snapshots. Their fit invariants are tested below with
    # valid distance fixtures, without changing the public estimator's input tags.
    applicable = {
        "check_estimator_cloneable",
        "check_estimator_tags_renamed",
        "check_valid_tag_types",
        "check_estimator_repr",
        "check_no_attributes_set_in_init",
        "check_estimators_unfitted",
        "check_do_not_raise_errors_in_init_or_set_params",
        "check_mixin_order",
        "check_complex_data",
        "check_estimators_empty_data_messages",
        "check_nonsquare_error",
        "check_estimator_sparse_tag",
        "check_estimator_sparse_array",
        "check_estimator_sparse_matrix",
        "check_parameters_default_constructible",
        "check_get_params_invariance",
        "check_set_params",
        "check_fit1d",
    }
    for instance, check in estimator_checks_generator(estimator):
        if check.func.__name__ in applicable:
            check(instance)


@pytest.mark.parametrize("estimator", [HierarchicalSeriation(), SpectralSeriation()])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, object])
def test_semantic_pairwise_fit_invariants(estimator, distance, dtype, tmp_path):
    matrix = np.asfortranarray(distance.astype(dtype))
    matrix.setflags(write=False)
    parameters = estimator.get_params()
    assert estimator.fit(matrix) is estimator
    ordering = estimator.ordering_.copy()
    assert estimator.get_params() == parameters
    estimator.fit(matrix)
    np.testing.assert_array_equal(estimator.ordering_, ordering)
    restored = pickle.loads(pickle.dumps(estimator))
    np.testing.assert_array_equal(restored.ordering_, ordering)
    if hasattr(restored, "partial_fit"):
        restored.partial_fit(matrix)
        np.testing.assert_array_equal(restored.ordering_, ordering)
    if dtype is not object:
        filename = tmp_path / "distance.npy"
        np.save(filename, matrix)
        estimator.fit(np.load(filename, mmap_mode="r"))
        np.testing.assert_array_equal(estimator.ordering_, ordering)
    np.testing.assert_array_equal(matrix, distance.astype(dtype))
