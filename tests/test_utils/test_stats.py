import itertools
import math

import cvxpy as cp
import numpy as np
import pytest
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd
from numpy.testing import assert_allclose

from skfolio.cluster import LinkageMethod
from skfolio.datasets import load_nasdaq_dataset, load_sp500_dataset
from skfolio.distance import PearsonDistance
from skfolio.preprocessing import prices_to_returns
from skfolio.utils.stats import (
    assert_is_distance,
    assert_is_square,
    assert_is_symmetric,
    combination_by_index,
    commutation_matrix,
    compute_optimal_n_clusters,
    corr_to_cov,
    cov_nearest,
    cov_to_corr,
    cs_rank,
    cs_rank_correlation,
    cs_weighted_correlation,
    inverse_multiply,
    is_cholesky_dec,
    minimize_relative_weight_deviation,
    multiply_by_inverse,
    n_bins_freedman,
    n_bins_knuth,
    rand_weights,
    rand_weights_dirichlet,
    safe_cholesky,
    sample_unique_subsets,
    squared_mahalanobis_dist,
    squared_standardized_euclidean_dist,
    symmetric_step_up_matrix,
    symmetrize,
)


def _norm_frobenious(x, y):
    return np.sqrt(((x - y) ** 2).sum())


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="module")
def returns():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices[["AAPL"]], log_returns=False)
    returns = X.to_numpy().reshape(-1)
    return returns


@pytest.fixture(scope="module")
def nasdaq_X():
    prices = load_nasdaq_dataset()
    nasdaq_X = prices_to_returns(prices)
    return nasdaq_X


@pytest.fixture(scope="module")
def distance(X):
    distance_estimator = PearsonDistance()
    distance_estimator.fit(X)
    distance = distance_estimator.distance_
    return distance


@pytest.fixture(scope="module")
def linkage_matrix(distance):
    condensed_distance = scd.squareform(distance, checks=False)
    linkage_matrix = sch.linkage(
        condensed_distance,
        method=LinkageMethod.SINGLE.value,
        optimal_ordering=False,
    )
    return linkage_matrix


@pytest.fixture()
def weights():
    weights = np.array(
        [
            0.15686274156862742,
            0.09803922098039221,
            0.07843137078431371,
            0.12500000125000002,
            0.20833333208333335,
            0.3333333333333333,
        ]
    )
    return weights


def test_n_bins_freedman(returns):
    n_bins = n_bins_freedman(returns)
    assert n_bins == 329


def test_n_bins_knuth(returns):
    n_bins = n_bins_knuth(returns)
    assert n_bins == 346


def test_cov_nearest(nasdaq_X):
    cov = np.cov(np.array(nasdaq_X).T)
    corr, _ = cov_to_corr(cov)
    _, _ = np.linalg.eigh(corr)
    assert not is_cholesky_dec(cov)
    cov2 = cov_nearest(cov, higham=False)
    assert is_cholesky_dec(cov2)


def test_corr_nearest_psd():
    x = np.array([[1, -0.2, -0.9], [-0.2, 1, -0.2], [-0.9, -0.2, 1]])
    y = cov_nearest(x, higham=True)
    np.testing.assert_almost_equal(x, y)
    y = cov_nearest(x, higham=False)
    np.testing.assert_almost_equal(x, y)


def test_corr_nearest_non_psd():
    x = np.array(
        [
            1,
            0.477,
            0.644,
            0.478,
            0.651,
            0.826,
            0.477,
            1,
            0.516,
            0.233,
            0.682,
            0.75,
            0.644,
            0.516,
            1,
            0.599,
            0.581,
            0.742,
            0.478,
            0.233,
            0.599,
            1,
            0.741,
            0.8,
            0.651,
            0.682,
            0.581,
            0.741,
            1,
            0.798,
            0.826,
            0.75,
            0.742,
            0.8,
            0.798,
            1,
        ]
    ).reshape(6, 6)
    assert not is_cholesky_dec(x)

    y = cov_nearest(x, higham=False)
    assert is_cholesky_dec(y)
    np.testing.assert_almost_equal(
        y,
        np.array(
            [
                1.0,
                0.4808738,
                0.64110485,
                0.48219267,
                0.64263258,
                0.80093596,
                0.4808738,
                1.0,
                0.51168908,
                0.2425915,
                0.66965194,
                0.71938778,
                0.64110485,
                0.51168908,
                1.0,
                0.59295412,
                0.58054676,
                0.73448752,
                0.48219267,
                0.2425915,
                0.59295412,
                1.0,
                0.72583221,
                0.76455881,
                0.64263258,
                0.66965194,
                0.58054676,
                0.72583221,
                1.0,
                0.79668556,
                0.80093596,
                0.71938778,
                0.73448752,
                0.76455881,
                0.79668556,
                1.0,
            ]
        ).reshape(6, 6),
    )
    np.testing.assert_almost_equal(_norm_frobenious(x, y), 0.08390962832371579)

    y = cov_nearest(x, higham=True)
    assert is_cholesky_dec(y)
    np.testing.assert_almost_equal(
        y,
        np.array(
            [
                1.0,
                0.48778612,
                0.64293091,
                0.49045543,
                0.64471508,
                0.80821008,
                0.48778612,
                1.0,
                0.51451154,
                0.25034126,
                0.67324973,
                0.72523171,
                0.64293091,
                0.51451154,
                1.0,
                0.59728118,
                0.5818673,
                0.74445497,
                0.49045543,
                0.25034126,
                0.59728118,
                1.0,
                0.7308955,
                0.77139846,
                0.64471508,
                0.67324973,
                0.5818673,
                0.7308955,
                1.0,
                0.81243213,
                0.80821008,
                0.72523171,
                0.74445497,
                0.77139846,
                0.81243213,
                1.0,
            ]
        ).reshape(6, 6),
    )
    np.testing.assert_almost_equal(_norm_frobenious(x, y), 0.07429322106703319)


def test_commutation_matrix():
    def vec(y):
        m, n = y.shape
        return y.reshape(m * n, order="F")

    x = np.random.rand(500, 500)
    k = commutation_matrix(x)

    assert np.all(k @ vec(x) == vec(x.T))


def test_compute_optimal_n_clusters(distance, linkage_matrix):
    n_clusters = compute_optimal_n_clusters(
        distance=distance, linkage_matrix=linkage_matrix
    )
    assert n_clusters == 4


def test_inverse_multiply(X):
    cov = np.cov(X.T)
    a = cov[12:, :][:, :-12]
    b = cov[:8, :][:, -12:]

    r1 = inverse_multiply(a, b)
    r2 = np.linalg.inv(a) @ b
    np.testing.assert_array_almost_equal(r1, r2)


def test_multiply_by_inverse(X):
    cov = np.cov(X.T)
    a = cov[:12, :][:, -8:]
    b = cov[12:, :][:, :-12]

    r1 = multiply_by_inverse(a, b)
    r2 = a @ np.linalg.inv(b)
    np.testing.assert_array_almost_equal(r1, r2)


@pytest.mark.parametrize("n1", [5])
@pytest.mark.parametrize("n2", [4, 5, 6])
def test_symmetric_step_up_matrix(n1, n2):
    m = symmetric_step_up_matrix(n1, n2)
    np.testing.assert_array_almost_equal(m @ np.ones(n2), np.ones(n1))


class TestSymmetrize:
    """Tests for the symmetrize utility."""

    def test_full_matrix(self):
        m = np.array([[1.0, 2.0], [4.0, 5.0]])
        symmetrize(m)
        np.testing.assert_array_equal(m, np.array([[1.0, 3.0], [3.0, 5.0]]))

    def test_with_where_mask(self):
        m = np.full((3, 3), np.nan)
        m[:2, :2] = [[1.0, 2.0], [4.0, 5.0]]
        where = np.array([True, True, False])
        symmetrize(m, where=where)
        np.testing.assert_array_equal(m[:2, :2], [[1.0, 3.0], [3.0, 5.0]])
        assert np.all(np.isnan(m[2, :]))

    def test_all_nan_is_noop(self):
        m = np.full((2, 2), np.nan)
        original = m.copy()
        symmetrize(m, where=np.array([False, False]))
        np.testing.assert_array_equal(np.isnan(m), np.isnan(original))

    def test_inplace(self):
        m = np.array([[1.0, 0.0], [2.0, 1.0]])
        original_id = id(m)
        result = symmetrize(m)
        assert result is None
        assert id(m) == original_id


class TestSafeCholesky:
    """Tests for the safe_cholesky utility."""

    def test_zero_matrix_fallback_adds_positive_ridge(self):
        cov = np.zeros((2, 2))
        chol = safe_cholesky(cov)
        reconstructed = chol @ chol.T
        np.testing.assert_allclose(reconstructed, np.diag(np.diag(reconstructed)))
        assert np.all(np.diag(reconstructed) > 0.0)

    def test_fallback_symmetrizes_near_spd_matrix(self):
        cov = np.array([[1.0, 1.0 + 1e-14], [1.0 - 1e-14, 1.0]])
        chol = safe_cholesky(cov)
        reconstructed = chol @ chol.T
        np.testing.assert_allclose(reconstructed, np.array([[1.0, 1.0], [1.0, 1.0]]))

    def test_does_not_mutate_input(self):
        cov = np.zeros((2, 2))
        cov_copy = cov.copy()
        _ = safe_cholesky(cov)
        np.testing.assert_array_equal(cov, cov_copy)

    def test_raises_on_strongly_indefinite_matrix(self):
        cov = np.array([[1.0, 2.0], [2.0, 1.0]])
        with pytest.raises(ValueError, match="Cholesky failed"):
            safe_cholesky(cov, max_tries=3)


# Generated by CodiumAI
class TestRandWeightsDirichlet:
    #  The function returns an array of n weights that sum to one.
    def test_weights_sum_to_one(self):
        weights = rand_weights_dirichlet(5)
        assert np.isclose(np.sum(weights), 1.0)

    #  The function returns an array of length n.
    def test_array_length(self):
        n = 10
        weights = rand_weights_dirichlet(n)
        assert len(weights) == n

    #  The function returns an array of floats.
    def test_array_type(self):
        weights = rand_weights_dirichlet(3)
        assert all(isinstance(w, float) for w in weights)


# Generated by CodiumAI
class TestRandWeights:
    #  Returns an array of n random weights that sum to 1.
    def test_weights_sum_to_one(self):
        weights = rand_weights(5)
        assert np.isclose(np.sum(weights), 1.0)

    #  Returns an array of n random weights that sum to 1, when n is 1.
    def test_weights_sum_to_one_n_1(self):
        weights = rand_weights(1)
        assert np.isclose(np.sum(weights), 1.0)

    #  Returns an array of n random weights that sum to 1, when n is 2.
    def test_weights_sum_to_one_n_2(self):
        weights = rand_weights(2)
        assert np.isclose(np.sum(weights), 1.0)


# Generated by CodiumAI
class TestIsDefPos:
    #  Returns True for a 2x2 definite-positive matrix
    def test_definite_positive_2x2(self):
        matrix = np.array([[2, 1], [1, 2]])
        assert is_cholesky_dec(matrix) is True

    #  Returns False for a 2x2 negative definite matrix
    def test_negative_definite_2x2(self):
        matrix = np.array([[-2, -1], [-1, -2]])
        assert is_cholesky_dec(matrix) is False

    #  Returns False for a 3x3 negative definite matrix
    def test_negative_definite_3x3(self):
        matrix = np.array([[-2, -1, 0], [-1, -2, -1], [0, -1, -2]])
        assert is_cholesky_dec(matrix) is False


# Generated by CodiumAI
class TestAssertIsSquare:
    #  The function receives a square matrix and does not raise any error.
    def test_square_matrix_no_error(self):
        # Arrange
        x = np.array([[1, 2], [3, 4]])

        # Act and Assert
        assert_is_square(x)

    #  The function receives a non-square matrix with shape (n,m)
    #  where n != m and raises a ValueError.
    def test_non_square_matrix_value_error(self):
        # Arrange
        x = np.array([[1, 2, 3], [4, 5, 6]])

        # Act and Assert
        with pytest.raises(ValueError):
            assert_is_square(x)

    #  The function receives a non-square matrix with shape (n,1)
    #  where n > 1 and raises a ValueError.
    def test_non_square_matrix_value_error_2(self):
        # Arrange
        x = np.array([[1], [2], [3]])

        # Act and Assert
        with pytest.raises(ValueError):
            assert_is_square(x)


# Generated by CodiumAI
class TestAssertIsSymmetric:
    #  The function should not raise an error when given a symmetric matrix.
    def test_symmetric_matrix(self):
        matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
        try:
            assert_is_symmetric(matrix)
        except ValueError:
            pytest.fail("assert_is_symmetric raised ValueError unexpectedly")

    #  The function should raise a ValueError when given a non-square matrix.
    def test_non_square_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            assert_is_symmetric(matrix)

    #  The function should raise a ValueError when given a non-symmetric matrix.
    def test_non_symmetric_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            assert_is_symmetric(matrix)


# Generated by CodiumAI
class TestAssertIsDistance:
    #  The function receives a valid distance matrix and does not raise any errors.
    def test_valid_distance_matrix(self):
        # Arrange
        x = np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]])

        # Act and Assert
        assert_is_distance(x)

    #  The function receives a non-square matrix and raises a ValueError.
    def test_non_square_matrix(self):
        # Arrange
        x = np.array([[0, 1, 2], [1, 0, 3]])

        # Act and Assert
        with pytest.raises(ValueError):
            assert_is_distance(x)

    #  The function receives a non-symmetric matrix and raises a ValueError.
    def test_non_symmetric_matrix(self):
        # Arrange
        x = np.array([[0, 1, 2], [1, 0, 3], [2, 4, 0]])

        # Act and Assert
        with pytest.raises(ValueError):
            assert_is_distance(x)


# Generated by CodiumAI
class TestCovToCorr:
    #  Should return a tuple with two ndarrays when given a valid 2D ndarray
    #  as input
    def test_valid_input(self):
        # Arrange
        cov = np.array([[1, 2], [2, 4]])

        # Act
        corr, std = cov_to_corr(cov)

        # Assert
        assert isinstance(corr, np.ndarray)
        assert isinstance(std, np.ndarray)

    #  Should raise a ValueError when given a 1D ndarray as input
    def test_1d_input(self):
        # Arrange
        cov = np.array([1, 2, 3])

        # Act and Assert
        with pytest.raises(ValueError):
            cov_to_corr(cov)

    #  Should raise a ValueError when given a 3D ndarray as input
    def test_3d_input(self):
        # Arrange
        cov = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]], [[3, 0], [0, 3]]])

        # Act and Assert
        with pytest.raises(ValueError):
            cov_to_corr(cov)


# Generated by CodiumAI
class TestCorrToCov:
    #  Should return a covariance matrix with the same shape as the input
    #  correlation matrix and standard deviation vector
    def test_same_shape(self):
        corr = np.array([[1, 0.5], [0.5, 1]])
        std = np.array([1, 2])
        expected_cov = np.array([[1, 1], [1, 4]])

        cov = corr_to_cov(corr, std)

        assert cov.shape == corr.shape == expected_cov.shape

    #  Should raise a ValueError when the input standard deviation vector
    #  is not a 1D array
    def test_invalid_std(self):
        corr = np.array([[1, 0.5], [0.5, 1]])
        std = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError):
            corr_to_cov(corr, std)

    #  Should raise a ValueError when the input correlation matrix is not a
    #  2D array
    def test_invalid_corr(self):
        corr = np.array([1, 0.5, 0.5, 1])
        std = np.array([1, 2])

        with pytest.raises(ValueError):
            corr_to_cov(corr, std)


# Generated by CodiumAI
class TestCovNearest:
    #  Should return the input covariance matrix if it is already
    #  positive semi-definite.
    def test_return_input_covariance_matrix_if_positive_semi_definite(self):
        cov = np.array([[1, 0], [0, 1]])
        result = cov_nearest(cov)
        np.testing.assert_array_equal(result, cov)

    #  Should raise a ValueError if the input covariance matrix is not
    #  square.
    def test_raise_value_error_if_input_covariance_matrix_not_square(self):
        cov = np.array([[1, 0, 0], [0, 1, 0]])
        with pytest.raises(ValueError):
            cov_nearest(cov)

    #  Should raise a ValueError if the input covariance matrix is not
    #  symmetric.
    def test_raise_value_error_if_input_covariance_matrix_not_symmetric(self):
        cov = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            cov_nearest(cov)


class TestMinimizeRelativeWeightDeviation:
    def tests_no_constraints(self, weights):
        new_weights = minimize_relative_weight_deviation(
            weights=weights, min_weights=np.zeros(6), max_weights=np.ones(6)
        )

        np.testing.assert_array_almost_equal(weights, new_weights)

    def tests_constraints(self, weights):
        new_weights = minimize_relative_weight_deviation(
            weights=weights, min_weights=np.ones(6) * 0.1, max_weights=np.ones(6) * 0.18
        )

        assert np.isclose(sum(new_weights), 1.0)
        np.testing.assert_array_almost_equal(
            new_weights, np.array([0.18, 0.16116994, 0.11883005, 0.18, 0.18, 0.18])
        )

    def test_non_feasible(self, weights):
        with pytest.raises(cp.SolverError):
            _ = minimize_relative_weight_deviation(
                weights=weights, min_weights=np.zeros(6), max_weights=np.ones(6) * 0.1
            )


# Helper to generate all combinations for small N, k
def generate_all_combinations(N: int, k: int):
    return list(itertools.combinations(range(N), k))


@pytest.mark.parametrize(
    "N,k",
    [
        (5, 1),
        (5, 2),
        (5, 3),
        (5, 5),
        (6, 3),
    ],
)
def test_unrank_all_positions(N, k):
    """Test that it maps every possible index correctly for small N,k."""
    all_combos = generate_all_combinations(N, k)
    M = math.comb(N, k)
    for idx in range(M):
        expected = list(all_combos[idx])
        result = combination_by_index(idx, N, k)
        np.testing.assert_array_equal(expected, result)


def test_unrank_invalid_index():
    """Out-of-range indices should raise ValueError."""
    with pytest.raises(ValueError):
        combination_by_index(-1, 5, 2)
    with pytest.raises(ValueError):
        combination_by_index(math.comb(5, 2), 5, 2)


def test_unrank_combination_full_cycle():
    """Unranking all indices yields the full set of combinations exactly once."""
    N, k = 7, 3
    all_combos = generate_all_combinations(N, k)
    M = math.comb(N, k)
    results = [tuple(combination_by_index(i, N, k)) for i in range(M)]
    assert set(results) == set(all_combos)
    assert len(results) == len(all_combos)


def test_unrank_edge_cases():
    """Handle k=0 and k=N edge cases correctly."""
    # k = 0: only one empty combination
    np.testing.assert_array_equal(combination_by_index(0, 5, 0), [])
    with pytest.raises(ValueError):
        combination_by_index(1, 5, 0)

    # k = N: only one full combination
    np.testing.assert_array_equal(combination_by_index(0, 5, 5), [0, 1, 2, 3, 4])
    with pytest.raises(ValueError):
        combination_by_index(1, 5, 5)


def test_sample_unique_subsets_properties():
    n, k, n_subsets = 10, 4, 20
    subs = sample_unique_subsets(n, k, n_subsets, random_state=1)
    assert isinstance(subs, np.ndarray)
    assert subs.shape == (n_subsets, k)
    seen = set()
    for row in subs:
        lst = row.tolist()
        assert len(lst) == k
        assert lst == sorted(lst)
        assert all(0 <= x < n for x in lst)
        seen.add(tuple(lst))
    assert len(seen) == n_subsets


def test_sample_unique_subsets_full_cycle():
    n, k = 5, 3
    total = math.comb(n, k)
    subs = sample_unique_subsets(n, k, total, random_state=42)
    expected = set(generate_all_combinations(n, k))
    actual = set(tuple(row.tolist()) for row in subs)
    assert actual == expected
    assert subs.shape[0] == total


def test_sample_unique_subsets_big_comb():
    n, k = 100, 10
    total = math.comb(n, k)
    assert total > 1e9
    subs = sample_unique_subsets(n, k, 5, random_state=42)
    assert subs.shape == (5, 10)


def test_sample_unique_subsets_errors():
    with pytest.raises(ValueError):
        sample_unique_subsets(-1, 2, 1)
    with pytest.raises(ValueError):
        sample_unique_subsets(5, 6, 1)
    with pytest.raises(ValueError):
        sample_unique_subsets(5, 2, math.comb(5, 2) + 1)


def test_edge_cases():
    # k=0
    arr0 = sample_unique_subsets(5, 0, 1, random_state=1)
    assert isinstance(arr0, np.ndarray)
    assert arr0.shape == (1, 0)
    with pytest.raises(ValueError):
        sample_unique_subsets(5, 0, 2)
    # k=n
    arr1 = sample_unique_subsets(4, 4, 1, random_state=2)
    assert arr1.shape == (1, 4)
    assert arr1.tolist() == [list(range(4))]
    with pytest.raises(ValueError):
        sample_unique_subsets(4, 4, 2)


class TestSquaredStandardizedEuclideanDist:
    """Tests for squared_standardized_euclidean_dist function."""

    def test_identity_covariance(self):
        """With identity covariance, result equals sum of squared returns."""
        returns = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        result = squared_standardized_euclidean_dist(returns, cov)
        expected = np.sum(returns**2)
        assert np.isclose(result, expected)

    def test_diagonal_covariance(self):
        """With diagonal covariance, result is sum of (r_i / sigma_i)^2."""
        returns = np.array([1.0, 2.0])
        cov = np.array([[4.0, 0.0], [0.0, 9.0]])  # std = [2, 3]
        result = squared_standardized_euclidean_dist(returns, cov)
        expected = (1.0 / 2.0) ** 2 + (2.0 / 3.0) ** 2
        assert np.isclose(result, expected)

    def test_zero_returns(self):
        """Zero returns should give zero distance."""
        returns = np.zeros(3)
        cov = np.eye(3)
        result = squared_standardized_euclidean_dist(returns, cov)
        assert result == 0.0

    def test_non_negative(self):
        """Result should always be non-negative."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(5)
        cov = rng.random((5, 5))
        cov = cov @ cov.T + np.eye(5)  # Make positive definite
        result = squared_standardized_euclidean_dist(returns, cov)
        assert result >= 0.0


class TestSquaredMahalanobisDist:
    """Tests for squared_mahalanobis_dist function."""

    def test_identity_covariance(self):
        """With identity covariance, equals sum of squared returns."""
        returns = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        result = squared_mahalanobis_dist(returns, cov)
        expected = np.sum(returns**2)
        assert np.isclose(result, expected)

    def test_diagonal_covariance(self):
        """With diagonal covariance, equals sum of (r_i / sigma_i)^2."""
        returns = np.array([1.0, 2.0])
        cov = np.array([[4.0, 0.0], [0.0, 9.0]])
        result = squared_mahalanobis_dist(returns, cov)
        expected = (1.0 / 2.0) ** 2 + (2.0 / 3.0) ** 2
        assert np.isclose(result, expected)

    def test_zero_returns(self):
        """Zero returns should give zero distance."""
        returns = np.zeros(3)
        cov = np.eye(3)
        result = squared_mahalanobis_dist(returns, cov)
        assert result == 0.0

    def test_non_negative(self):
        """Result should always be non-negative."""
        rng = np.random.default_rng(42)
        returns = rng.standard_normal(5)
        cov = rng.random((5, 5))
        cov = cov @ cov.T + np.eye(5)
        result = squared_mahalanobis_dist(returns, cov)
        assert result >= 0.0

    def test_correlated_covariance(self):
        """Test with a correlated covariance matrix."""
        returns = np.array([1.0, 1.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = squared_mahalanobis_dist(returns, cov)
        # Analytical: r^T @ inv(cov) @ r
        cov_inv = np.linalg.inv(cov)
        expected = returns @ cov_inv @ returns
        assert np.isclose(result, expected)

    def test_ridge_regularization(self):
        """Test that ridge regularization handles near-singular matrices."""
        # Create a nearly singular covariance matrix
        cov = np.array([[1.0, 0.9999], [0.9999, 1.0]])
        returns = np.array([1.0, 1.0])
        # Should not raise with ridge regularization
        result = squared_mahalanobis_dist(returns, cov, ridge_scale=1e-6)
        assert result >= 0.0

    def test_does_not_mutate_input(self):
        """Function should not mutate input covariance."""
        returns = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        cov_copy = cov.copy()
        _ = squared_mahalanobis_dist(returns, cov)
        np.testing.assert_array_equal(cov, cov_copy)

    def test_2d_returns_multiple_observations(self):
        """Test with 2D returns (multiple observations)."""
        returns = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, 1.5]])
        cov = np.eye(2)
        result = squared_mahalanobis_dist(returns, cov)
        # With identity covariance, d^2 = sum of squared returns per row
        expected = np.array([1**2 + 2**2, 3**2 + 4**2, 0.5**2 + 1.5**2])
        np.testing.assert_allclose(result, expected)
        assert result.shape == (3,)

    def test_2d_returns_single_observation(self):
        """Test with 2D returns (single observation, shape (1, n))."""
        returns = np.array([[1.0, 2.0, 3.0]])
        cov = np.eye(3)
        result = squared_mahalanobis_dist(returns, cov)
        expected = np.array([1**2 + 2**2 + 3**2])
        np.testing.assert_allclose(result, expected)
        assert result.shape == (1,)

    def test_1d_returns_scalar_output(self):
        """1D input should return a scalar float for backward compatibility."""
        returns = np.array([1.0, 2.0, 3.0])
        cov = np.eye(3)
        result = squared_mahalanobis_dist(returns, cov)
        assert isinstance(result, float)

    def test_2d_returns_array_output(self):
        """2D input should return an ndarray."""
        returns = np.array([[1.0, 2.0], [3.0, 4.0]])
        cov = np.eye(2)
        result = squared_mahalanobis_dist(returns, cov)
        assert isinstance(result, np.ndarray)

    def test_mean_subtraction(self):
        """Test mean parameter subtracts location from returns."""
        returns = np.array([3.0, 5.0])
        mean = np.array([1.0, 2.0])
        cov = np.eye(2)
        result = squared_mahalanobis_dist(returns, cov, mean=mean)
        # (r - mu) = [2.0, 3.0], d^2 = 4 + 9 = 13
        expected = (3.0 - 1.0) ** 2 + (5.0 - 2.0) ** 2
        assert np.isclose(result, expected)

    def test_mean_with_2d_returns(self):
        """Test mean parameter with matrix returns."""
        returns = np.array([[2.0, 4.0], [3.0, 5.0]])
        mean = np.array([1.0, 2.0])
        cov = np.eye(2)
        result = squared_mahalanobis_dist(returns, cov, mean=mean)
        # Row 0: (2-1)^2 + (4-2)^2 = 1 + 4 = 5
        # Row 1: (3-1)^2 + (5-2)^2 = 4 + 9 = 13
        expected = np.array([5.0, 13.0])
        np.testing.assert_allclose(result, expected)

    def test_mean_none_is_zero_centered(self):
        """mean=None should be equivalent to mean=zeros."""
        returns = np.array([1.0, 2.0])
        cov = np.eye(2)
        result_none = squared_mahalanobis_dist(returns, cov, mean=None)
        result_zero = squared_mahalanobis_dist(returns, cov, mean=np.zeros(2))
        assert np.isclose(result_none, result_zero)

    def test_incompatible_covariance_shape_raises(self):
        """Incompatible covariance shape should raise ValueError."""
        returns = np.array([[1.0, 2.0, 3.0]])
        cov = np.eye(2)  # Wrong size
        with pytest.raises(ValueError, match="cholesky shape"):
            squared_mahalanobis_dist(returns, cov)

    def test_incompatible_mean_shape_raises(self):
        """Incompatible mean shape should raise ValueError."""
        returns = np.array([1.0, 2.0])
        cov = np.eye(2)
        mean = np.array([1.0, 2.0, 3.0])  # Wrong size
        with pytest.raises(ValueError, match="mean must be 1D"):
            squared_mahalanobis_dist(returns, cov, mean=mean)

    def test_2d_mean_raises(self):
        """2D mean should raise ValueError."""
        returns = np.array([1.0, 2.0])
        cov = np.eye(2)
        mean = np.array([[1.0, 2.0]])  # 2D instead of 1D
        with pytest.raises(ValueError, match="mean must be 1D"):
            squared_mahalanobis_dist(returns, cov, mean=mean)


class TestCsWeightedCorrelation:
    """Tests for cs_weighted_correlation."""

    def test_perfect_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert_allclose(cs_weighted_correlation(a, b), 1.0)

    def test_perfect_negative_correlation(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 8.0, 6.0, 4.0, 2.0])
        assert_allclose(cs_weighted_correlation(a, b), -1.0)

    def test_zero_correlation(self):
        a = np.array([1.0, -1.0, 1.0, -1.0])
        b = np.array([1.0, 1.0, -1.0, -1.0])
        assert_allclose(cs_weighted_correlation(a, b), 0.0, atol=1e-12)

    def test_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)
        expected = np.corrcoef(a, b)[0, 1]
        assert_allclose(cs_weighted_correlation(a, b), expected)

    def test_equal_weights_matches_unweighted(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        w = np.ones(50) * 3.0
        assert_allclose(
            cs_weighted_correlation(a, b, weights=w),
            cs_weighted_correlation(a, b),
        )

    def test_weighted_shifts_result(self):
        a = np.array([1.0, 2.0, 3.0, 10.0])
        b = np.array([1.0, 2.0, 3.0, -5.0])
        corr_equal = cs_weighted_correlation(a, b)
        w_down_outlier = np.array([1.0, 1.0, 1.0, 0.01])
        corr_weighted = cs_weighted_correlation(a, b, weights=w_down_outlier)
        assert corr_weighted > corr_equal

    def test_nan_handling(self):
        a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        b = np.array([2.0, np.nan, 6.0, 8.0, 10.0])
        result = cs_weighted_correlation(a, b)
        valid = np.array([True, False, False, True, True])
        expected = np.corrcoef(a[valid], b[valid])[0, 1]
        assert_allclose(result, expected)

    def test_min_count(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert np.isnan(cs_weighted_correlation(a, b, min_count=3))
        assert np.isfinite(cs_weighted_correlation(a, b, min_count=2))

    def test_constant_vector_returns_nan(self):
        a = np.array([5.0, 5.0, 5.0, 5.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(cs_weighted_correlation(a, b))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="broadcastable"):
            cs_weighted_correlation(np.ones(3), np.ones(4))

    def test_2d_vectorized_over_factors(self):
        rng = np.random.default_rng(42)
        n_assets, n_factors = 50, 4
        a = rng.standard_normal((n_assets, n_factors))
        b = rng.standard_normal((n_assets, n_factors))
        result = cs_weighted_correlation(a, b, axis=0)
        assert result.shape == (n_factors,)
        for k in range(n_factors):
            expected = np.corrcoef(a[:, k], b[:, k])[0, 1]
            assert_allclose(result[k], expected, rtol=1e-10)

    def test_3d_vectorized_over_time_and_factors(self):
        rng = np.random.default_rng(42)
        n_time, n_assets, n_factors = 10, 50, 3
        a = rng.standard_normal((n_time, n_assets, n_factors))
        b = rng.standard_normal((n_time, n_assets, n_factors))
        result = cs_weighted_correlation(a, b, axis=1)
        assert result.shape == (n_time, n_factors)
        for t in range(n_time):
            for k in range(n_factors):
                expected = np.corrcoef(a[t, :, k], b[t, :, k])[0, 1]
                assert_allclose(result[t, k], expected, rtol=1e-10)

    def test_3d_with_weights(self):
        rng = np.random.default_rng(42)
        n_time, n_assets, n_factors = 5, 30, 2
        a = rng.standard_normal((n_time, n_assets, n_factors))
        b = rng.standard_normal((n_time, n_assets, n_factors))
        w = rng.uniform(0.1, 1.0, size=(n_time, n_assets))
        result = cs_weighted_correlation(a, b, weights=w, axis=1)
        assert result.shape == (n_time, n_factors)
        for val in result.ravel():
            assert -1.0 <= val <= 1.0 or np.isnan(val)

    def test_2d_axis1_with_1d_weights(self):
        a = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        b = np.array([[1.0, 4.0, 9.0], [9.0, 4.0, 1.0]])
        w = np.array([1.0, 2.0, 3.0])

        result = cs_weighted_correlation(a, b, weights=w, axis=1)

        expected = np.array(
            [
                cs_weighted_correlation(a[0], b[0], weights=w),
                cs_weighted_correlation(a[1], b[1], weights=w),
            ]
        )
        assert_allclose(result, expected)

    def test_scalar_output_for_1d(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = cs_weighted_correlation(a, b)
        assert isinstance(result, float)

    def test_nan_row_in_batch(self):
        a = np.array([[1.0, 2.0], [np.nan, np.nan], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [np.nan, np.nan], [7.0, 8.0]])
        result = cs_weighted_correlation(a, b, axis=0, min_count=3)
        assert result.shape == (2,)
        assert all(np.isnan(result))


class TestCsRank:
    """Tests for cs_rank."""

    def test_basic_1d(self):
        a = np.array([30.0, 10.0, 20.0])
        expected = np.array([3.0, 1.0, 2.0])
        assert_allclose(cs_rank(a), expected)

    def test_nan_excluded(self):
        a = np.array([30.0, np.nan, 10.0])
        result = cs_rank(a)
        assert np.isnan(result[1])
        assert_allclose(result[0], 2.0)
        assert_allclose(result[2], 1.0)

    def test_2d_axis0(self):
        a = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        result = cs_rank(a, axis=0)
        assert result.shape == (3, 2)
        assert_allclose(result[:, 0], [3.0, 1.0, 2.0])
        assert_allclose(result[:, 1], [1.0, 3.0, 2.0])

    def test_2d_axis1(self):
        a = np.array([[30.0, 10.0, 20.0], [1.0, 3.0, 2.0]])
        result = cs_rank(a, axis=1)
        assert result.shape == (2, 3)
        assert_allclose(result[0], [3.0, 1.0, 2.0])
        assert_allclose(result[1], [1.0, 3.0, 2.0])

    def test_3d(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((5, 10, 3))
        result = cs_rank(a, axis=1)
        assert result.shape == (5, 10, 3)
        assert np.all(np.isfinite(result))

    def test_preserves_shape(self):
        a = np.ones((2, 3, 4))
        assert cs_rank(a, axis=1).shape == (2, 3, 4)

    def test_all_nan_row(self):
        a = np.array([np.nan, np.nan, np.nan])
        result = cs_rank(a)
        assert all(np.isnan(result))


class TestCsRankCorrelation:
    """Tests for cs_rank_correlation."""

    def test_matches_scipy_spearmanr_1d(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        a = rng.standard_normal(100)
        b = rng.standard_normal(100)
        expected = spearmanr(a, b).statistic
        assert_allclose(cs_rank_correlation(a, b), expected, rtol=1e-10)

    def test_perfect_monotonic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        assert_allclose(cs_rank_correlation(a, b), 1.0)

    def test_perfect_negative_monotonic(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
        assert_allclose(cs_rank_correlation(a, b), -1.0)

    def test_2d_vectorized(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        n_assets, n_factors = 50, 3
        a = rng.standard_normal((n_assets, n_factors))
        b = rng.standard_normal((n_assets, n_factors))
        result = cs_rank_correlation(a, b, axis=0)
        assert result.shape == (n_factors,)
        for k in range(n_factors):
            expected = spearmanr(a[:, k], b[:, k]).statistic
            assert_allclose(result[k], expected, rtol=1e-10)

    def test_3d_vectorized(self):
        from scipy.stats import spearmanr

        rng = np.random.default_rng(42)
        n_time, n_assets, n_factors = 5, 30, 2
        a = rng.standard_normal((n_time, n_assets, n_factors))
        b = rng.standard_normal((n_time, n_assets, n_factors))
        result = cs_rank_correlation(a, b, axis=1)
        assert result.shape == (n_time, n_factors)
        for t in range(n_time):
            for k in range(n_factors):
                expected = spearmanr(a[t, :, k], b[t, :, k]).statistic
                assert_allclose(result[t, k], expected, rtol=1e-10)

    def test_scalar_output_for_1d(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        result = cs_rank_correlation(a, b)
        assert isinstance(result, float)

    def test_pairwise_nan_matches_spearmanr(self):
        from scipy.stats import spearmanr

        a = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        b = np.array([5.0, np.nan, 3.0, 2.0, 1.0])

        valid = np.isfinite(a) & np.isfinite(b)
        expected = spearmanr(a[valid], b[valid]).statistic

        assert_allclose(cs_rank_correlation(a, b), expected)
