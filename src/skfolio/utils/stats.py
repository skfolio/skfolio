"""Tools module."""


# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# Precise, Copyright (c) 2021, Peter Cotton.
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# Statsmodels, Copyright (C) 2006, Jonathan E. Taylor, Licensed under BSD 3 clause.

import math
import random
import warnings
from enum import auto

import cvxpy as cp
import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.optimize as sco
import scipy.sparse.linalg as scl
import scipy.spatial.distance as scd
import scipy.special as scs
from scipy.sparse import csr_matrix

from skfolio.utils.tools import AutoEnum

__all__ = [
    "NBinsMethod",
    "assert_is_distance",
    "assert_is_square",
    "assert_is_symmetric",
    "combination_by_index",
    "commutation_matrix",
    "compute_optimal_n_clusters",
    "corr_to_cov",
    "cov_nearest",
    "cov_to_corr",
    "inverse_multiply",
    "is_cholesky_dec",
    "minimize_relative_weight_deviation",
    "multiply_by_inverse",
    "n_bins_freedman",
    "n_bins_knuth",
    "rand_weights",
    "rand_weights_dirichlet",
    "sample_unique_subsets",
    "symmetric_step_up_matrix",
]


class NBinsMethod(AutoEnum):
    """Enumeration of the Number of Bins Methods.

    Parameters
    ----------
    FREEDMAN : str
        Freedman method

    KNUTH : str
        Knuth method
    """

    FREEDMAN = auto()
    KNUTH = auto()


def n_bins_freedman(x: np.ndarray) -> int:
    """Compute the optimal histogram bin size using the Freedman-Diaconis rule [1]_.

    Parameters
    ----------
    x : ndarray of shape (n_observations,)
        The input array.

    Returns
    -------
    n_bins : int
        The optimal bin size.

    References
    ----------
    .. [1] "On the histogram as a density estimator: L2 theory".
        Freedman & Diaconis (1981).
    """
    if x.ndim != 1:
        raise ValueError("`x` must be a 1d-array")
    n = len(x)
    p_25, p_75 = np.percentile(x, [25, 75])
    d = 2 * (p_75 - p_25) / (n ** (1 / 3))
    if d == 0:
        return 5
    n_bins = max(1, np.ceil((np.max(x) - np.min(x)) / d))
    return round(n_bins)


def n_bins_knuth(x: np.ndarray) -> int:
    """Compute the optimal histogram bin size using Knuth's rule [1]_.

    Parameters
    ----------
    x : ndarray of shape (n_observations,)
        The input array.

    Returns
    -------
    n_bins : int
        The optimal bin size.

    References
    ----------
    .. [1] "Optimal Data-Based Binning for Histograms".
        Knuth.
    """
    x = np.sort(x)
    n = len(x)

    def func(y: np.ndarray) -> float:
        y = y[0]
        if y <= 0:
            return np.inf
        bin_edges = np.linspace(x[0], x[-1], int(y) + 1)
        hist, _ = np.histogram(x, bin_edges)
        return -(
            n * np.log(y)
            + scs.gammaln(0.5 * y)
            - y * scs.gammaln(0.5)
            - scs.gammaln(n + 0.5 * y)
            + np.sum(scs.gammaln(hist + 0.5))
        )

    n_bins_init = n_bins_freedman(x)
    n_bins = sco.fmin(func, n_bins_init, disp=0)[0]
    return round(n_bins)


def rand_weights_dirichlet(n: int) -> np.array:
    """Produces n random weights that sum to one from a dirichlet distribution
    (uniform distribution over a simplex).

    Parameters
    ----------
    n : int
        Number of weights.

    Returns
    -------
    weights : ndarray of shape (n, )
        The vector of weights.
    """
    return np.random.dirichlet(np.ones(n))


def rand_weights(n: int, zeros: int = 0, seed: int | None = None) -> np.ndarray:
    """Produces n random weights that sum to one from a uniform distribution
    (non-uniform distribution over a simplex).

    Parameters
    ----------
    n : int
        Number of weights.

    zeros : int, default=0
        The number of weights to randomly set to zeros.

    seed : int or None, default=None
        Seed for reproducibility. If None, use an unseeded generator.

    Returns
    -------
    weights : ndarray of shape (n, )
        The vector of weights.
    """
    rng = np.random.default_rng(seed)

    k = rng.random(n)
    if zeros > 0:
        zeros_idx = rng.choice(n, zeros, replace=False)
        k[zeros_idx] = 0
    return k / k.sum()


def is_cholesky_dec(x: np.ndarray) -> bool:
    """Returns True if Cholesky decomposition can be computed.
    The matrix must be Hermitian (symmetric if real-valued) and positive-definite.
    No checking is performed to verify whether the matrix is Hermitian or not.

    Parameters
    ----------
    x : ndarray of shape (n, m)
       The matrix.

    Returns
    -------
    value : bool
        True if Cholesky decomposition can be applied to the matrix, False otherwise.
    """
    # Around 100 times faster than checking for positive eigenvalues with np.linalg.eigh
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError:
        return False


def is_positive_definite(x: np.ndarray) -> bool:
    """Returns True if the matrix is positive definite.

    Parameters
    ----------
    x : ndarray of shape (n, m)
       The matrix.

    Returns
    -------
    value : bool
        True if the matrix is positive definite, False otherwise.
    """
    return np.all(np.linalg.eigvals(x) > 0)


def assert_is_square(x: np.ndarray) -> None:
    """Raises an error if the matrix is not square.

    Parameters
    ----------
    x : ndarray of shape (n, n)
       The matrix.

    Raises
    ------
    ValueError: if the matrix is not square.
    """
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("The matrix must be square")


def assert_is_symmetric(x: np.ndarray) -> None:
    """Raises an error if the matrix is not symmetric.

    Parameters
    ----------
    x : ndarray of shape (n, m)
       The matrix.

    Raises
    ------
    ValueError: if the matrix is not symmetric.
    """
    assert_is_square(x)
    if not np.allclose(x, x.T):
        raise ValueError("The matrix must be symmetric")


def assert_is_distance(x: np.ndarray) -> None:
    """Raises an error if the matrix is not a distance matrix.

    Parameters
    ----------
    x : ndarray of shape (n, n)
       The matrix.

    Raises
    ------
    ValueError: if the matrix is a distance matrix.
    """
    assert_is_symmetric(x)
    if not np.allclose(np.diag(x), np.zeros(x.shape[0]), atol=1e-5):
        raise ValueError(
            "The distance matrix must have diagonal elements close to zeros"
        )


def cov_to_corr(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a covariance matrix to a correlation matrix.

    Parameters
    ----------
    cov : ndarray of shape (n, n)
        Covariance matrix.

    Returns
    -------
    corr, std : tuple[ndarray of shape (n, n), ndarray of shape (n, )]
        Correlation matrix and standard-deviation vector
    """
    if cov.ndim != 2:
        raise ValueError(f"`cov` must be a 2D array, got a {cov.ndim}D array")
    std = np.sqrt(np.diag(cov))
    corr = cov / std / std[:, None]
    return corr, std


def corr_to_cov(corr: np.ndarray, std: np.ndarray):
    """Convert a correlation matrix to a covariance matrix given its
    standard-deviation vector.

    Parameters
    ----------
    corr : ndarray of shape (n, n)
        Correlation matrix.

    std : ndarray of shape (n, )
        Standard-deviation vector.

    Returns
    -------
    cov : ndarray of shape (n, n)
        Covariance matrix
    """
    if std.ndim != 1:
        raise ValueError(f"`std` must be a 1D array, got a {std.ndim}D array")
    if corr.ndim != 2:
        raise ValueError(f"`corr` must be a 2D array, got a {corr.ndim}D array")
    cov = corr * std * std[:, None]
    return cov


_CLIPPING_VALUE = 1e-13


def cov_nearest(
    cov: np.ndarray,
    higham: bool = False,
    higham_max_iteration: int = 100,
    warn: bool = False,
):
    """Compute the nearest covariance matrix that is positive definite and with a
    cholesky decomposition than can be computed. The variance is left unchanged.
    A covariance matrix that is not positive definite often occurs in high
    dimensional problems. It can be due to multicollinearity, floating-point
    inaccuracies, or when the number of observations is smaller than the number of
    assets.

    First, it converts the covariance matrix to a correlation matrix.
    Then, it finds the nearest correlation matrix and converts it back to a covariance
    matrix using the initial standard deviation.

    Cholesky decomposition can fail for symmetric positive definite (SPD) matrix due
    to floating point error and inversely, Cholesky decomposition can succeed for
    non-SPD matrix. Therefore, we need to test for both. We always start by testing
    for Cholesky decomposition which is significantly faster than checking for positive
    eigenvalues.

    Parameters
    ----------
    cov : ndarray of shape (n, n)
        Covariance matrix.

    higham : bool, default=False
        If this is set to True, the Higham (2002) algorithm [1]_ is used,
        otherwise the eigenvalues are clipped to threshold above zeros (1e-13).
        The default (`False`) is to use the clipping method as the Higham
        algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iterations of the Higham (2002) algorithm.
        The default value is `100`.

    warn : bool, default=False
        If this is set to True, a user warning is emitted when the covariance matrix
        is not positive definite and replaced by the nearest. The default is False.

    Returns
    -------
    cov : ndarray
        The nearest covariance matrix.

    References
    ----------
    .. [1] "Computing the nearest correlation matrix - a problem from finance"
        IMA Journal of Numerical Analysis
        Higham (2002)
    """
    assert_is_square(cov)
    assert_is_symmetric(cov)

    # Around 100 times faster than checking eigenvalues with np.linalg.eigh
    if is_cholesky_dec(cov) and is_positive_definite(cov):
        return cov

    if warn:
        warnings.warn(
            "The covariance matrix is not positive definite. "
            f"The {'Higham' if higham else 'Clipping'} algorithm will be used to find "
            "the nearest positive definite covariance.",
            stacklevel=2,
        )
    corr, std = cov_to_corr(cov)

    if higham:
        eps = np.finfo(np.float64).eps * 5
        diff = np.zeros(corr.shape)
        x = corr.copy()
        for _ in range(higham_max_iteration):
            x_adj = x - diff
            eig_vals, eig_vecs = np.linalg.eigh(x_adj)
            x = eig_vecs * np.maximum(eig_vals, eps) @ eig_vecs.T
            diff = x - x_adj
            np.fill_diagonal(x, 1)
            cov = corr_to_cov(x, std)
            if is_cholesky_dec(cov) and is_positive_definite(cov):
                break
        else:
            raise ValueError("Unable to find the nearest positive definite matrix")
    else:
        eig_vals, eig_vecs = np.linalg.eigh(corr)
        # Clipping the eigenvalues with a value smaller than 1e-13 can cause scipy to
        # consider the matrix non-psd is some corner cases (see test/test_stats.py)
        x = eig_vecs * np.maximum(eig_vals, _CLIPPING_VALUE) @ eig_vecs.T
        x, _ = cov_to_corr(x)
        cov = corr_to_cov(x, std)

    return cov


def commutation_matrix(x):
    """Compute the commutation matrix.

    Parameters
    ----------
    x : ndarray of shape (n,  m)
        The matrix.

    Returns
    -------
    K : ndarray of shape (m * n, m * n)
        The commutation matrix.
    """
    (m, n) = x.shape
    row = np.arange(m * n)
    col = row.reshape((m, n), order="F").ravel()
    data = np.ones(m * n, dtype=np.int8)
    k = csr_matrix((data, (row, col)), shape=(m * n, m * n))
    return k


def compute_optimal_n_clusters(distance: np.ndarray, linkage_matrix: np.ndarray) -> int:
    r"""Compute the optimal number of clusters based on Two-Order Difference to Gap
    Statistic [1]_.

    The Two-Order Difference to Gap Statistic has been developed to improve the
    performance and stability of the Tibshiranis Gap statistic.
    It applies the two-order difference of the within-cluster dispersion to replace the
    reference null distribution in the Gap statistic.

    The number of cluster :math:`k` is determined by:

    .. math::  \begin{cases}
                \begin{aligned}
                &\max_{k} & & W_{k+2} + W_{k} - 2 W_{k+1} \\
                &\text{s.t.} & & 1 \ge c \ge max\bigl(8, \sqrt{n}\bigr) \\
                \end{aligned}
                \end{cases}

    with :math:`n` the sample size and :math:`W_{k}` the within-cluster dispersions
    defined as:

    .. math:: W_{k} = \sum_{i=1}^{k} \frac{D_{i}}{2|C_{i}|}

    where :math:`|C_{i}|` is the cardinality of cluster :math:`i` and :math:`D_{i}` its
    density defined as:

    .. math:: D_{i} = \sum_{u \in C_{i}} \sum_{v \in C_{i}} d(u,v)

    with :math:`d(u,v)` the distance between u and v.

    Parameters
    ----------
    distance : ndarray of shape (n, n)
        Distance matrix.

    linkage_matrix : ndarray of shape (n - 1, 4)
        Linkage matrix.

    Returns
    -------
    value : int
        Optimal number of clusters.

    References
    ----------
    .. [1] "Application of two-order difference to gap statistic".
        Yue, Wang & Wei (2009)
    """
    cut_tree = sch.cut_tree(linkage_matrix)
    n = cut_tree.shape[1]
    max_clusters = min(n, max(8, round(np.sqrt(n))))
    dispersion = []
    for k in range(max_clusters):
        level = cut_tree[:, n - k - 1]
        cluster_density = []
        for i in range(np.max(level) + 1):
            cluster_idx = np.argwhere(level == i).flatten()
            cluster_dists = scd.squareform(
                distance[cluster_idx, :][:, cluster_idx], checks=False
            )
            if cluster_dists.shape[0] != 0:
                cluster_density.append(np.nan_to_num(cluster_dists.mean()))
        dispersion.append(np.sum(cluster_density))
    dispersion = np.array(dispersion)
    gaps = np.roll(dispersion, -2) + dispersion - 2 * np.roll(dispersion, -1)
    gaps = gaps[:-2]
    # k=0 represents one cluster
    k = np.argmax(gaps) + 2
    return k


def minimize_relative_weight_deviation(
    weights: np.ndarray,
    min_weights: np.ndarray,
    max_weights: np.ndarray,
    solver: str = "CLARABEL",
    solver_params: dict | None = None,
) -> np.ndarray:
    r"""
    Apply weight constraints to an initial array of weights by minimizing the relative
    weight deviation of the final weights from the initial weights.

    .. math::
            \begin{cases}
            \begin{aligned}
            &\min_{w} & & \Vert \frac{w - w_{init}}{w_{init}} \Vert_{2}^{2} \\
            &\text{s.t.} & & \sum_{i=1}^{N} w_{i} = 1 \\
            & & & w_{min} \leq w_i \leq w_{max}, \quad \forall i
            \end{aligned}
            \end{cases}

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Initial weights.

    min_weights : ndarray of shape (n_assets,)
        Minimum assets weights (weights lower bounds).

    max_weights : ndarray of shape (n_assets,)
        Maximum assets weights (weights upper bounds).

    solver : str, default="CLARABEL"
        The solver to use. The default is "CLARABEL" which is written in Rust and has
        better numerical stability and performance than ECOS and SCS.
        For more details about available solvers, check the CVXPY documentation:
        https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver

    solver_params : dict, optional
        Solver parameters. For example, `solver_params=dict(verbose=True)`.
        The default (`None`) is to use the CVXPY default.
        For more details about solver arguments, check the CVXPY documentation:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options
    """
    if not (weights.shape == min_weights.shape == max_weights.shape):
        raise ValueError("`min_weights` and `max_weights` must have same size")

    if np.any(weights < 0):
        raise ValueError("Initial weights must be strictly positive")

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Initial weights must sum to one")

    if np.any(max_weights < min_weights):
        raise ValueError("`min_weights` must be lower or equal to `max_weights`")

    if np.all((weights >= min_weights) & (weights <= max_weights)):
        return weights

    if solver_params is None:
        solver_params = {}

    n = len(weights)
    w = cp.Variable(n)

    objective = cp.Minimize(cp.norm(w / weights - 1))
    constraints = [cp.sum(w) == 1, w >= min_weights, w <= max_weights]
    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=solver, **solver_params)

        if w.value is None:
            raise cp.SolverError("No solution found")

    except (cp.SolverError, scl.ArpackNoConvergence):
        raise cp.SolverError(
            f"Solver '{solver}' failed. Try another"
            " solver, or solve with solver_params=dict(verbose=True) for more"
            " information"
        ) from None

    return w.value


def combination_by_index(idx: int, n: int, k: int) -> np.ndarray:
    """
    Retrieve the k-combination at a given lexicographic position without enumerating
    all combinations.

    This function implements the *unranking* algorithm (also known as the combinatorial
    number system or "combinadic") to retrieve the specific k-combination corresponding
    to a given lexicographic `idx` without generating all C(n, k) possible subsets.

    Given a universe of size `n`, there are M = C(n, k) possible subsets of size k.
    This function returns the subset corresponding to the `idx` in lex order.

    This approach is crucial when M = C(n, k) is too large to generate or store all
    combinations, and you need to draw random subsets uniformly (sampling k=5 from n=100
    gives M ≈ 7.5e7).

    Time complexity: O(k)
    Space complexity: O(k)

    Parameters
    ----------
    idx : int
        Index (rank) of the desired combination in lex order. Must satisfy
        0 <= idx < C(n, k).

    n : int
        Size of the universe.

    k : int
        Size of each combination (0 <= k <= n).

    Returns
    -------
    combination : ndarray of shape (k,)
        1D integer array of length k containing the sorted k-combination.

    Raises
    ------
    ValueError
        If parameters are out of valid range.

    References
    ----------
    ..[1] "The Art of Computer Programming", Vol. 4A: Combinatorial Algorithms,
      Section 7.2.1.3. Knuth, D. E. (1998).
    """
    total = math.comb(n, k)
    if idx < 0 or idx >= total:
        raise ValueError(
            f"Index {idx} out of range for C({n},{k})={total} combinations."
        )

    combination = np.empty(k, dtype=int)
    remaining_rank = idx
    next_element = 0

    for pos in range(k):
        remaining_slots = k - pos
        x = next_element
        block_size = math.comb(n - x - 1, remaining_slots - 1)
        while block_size <= remaining_rank:
            remaining_rank -= block_size
            x += 1
            block_size = math.comb(n - x - 1, remaining_slots - 1)
        combination[pos] = x
        next_element = x + 1

    return combination


def sample_unique_subsets(
    n: int, k: int, n_subsets: int, random_state: int | None = None
) -> np.ndarray:
    """
    Generate unique k-element subsets from a universe of size n using combinatorial
    unranking.

    Each subset is drawn without replacement (elements within subset are distinct) and
    no subset is repeated across draws. Ranks are sampled uniformly without replacement
    over [0, C(n, k)).

    Time complexity: O(n_subsets * k)
    Space complexity: O(n_subsets * k)

    Parameters
    ----------
    n : int
        Universe size.

    k : int
        Subset size (0 <= k <= n).

    n_subsets : int
        Number of distinct subsets to generate (0 <= n_subsets <= C(n, k)).

    random_state : int, RandomState instance or None, default=None
        Seed or random state to ensure reproducibility.

    Returns
    -------
    subsets : ndarray of shape (n_subsets, k)
        2D integer array of shape (n_subsets, k) where each row is a sorted
        k-combination.

    Raises
    ------
    ValueError
        If any parameters are out of valid ranges.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}.")
    if k < 0 or k > n:
        raise ValueError(f"k={k} must satisfy 0 <= k <= n={n}.")

    total = math.comb(n, k)
    if n_subsets < 0 or n_subsets > total:
        raise ValueError(
            f"n_subsets={n_subsets} must satisfy 0 <= n_subsets <= C({n},{k})={total}."
        )

    rng = random.Random(random_state)
    ranks = rng.sample(range(total), k=n_subsets)
    # random.sample has a special-case for range objects that avoids building a list of
    # length M=C(n,k) and runs in O(n_subsets) time and space as opposed to
    # `choice(total, size=n_subsets, replace=False)` which run in O(M) space and raises
    # ArrayMemoryError for very big M.
    subsets = np.empty((n_subsets, k), dtype=int)
    for i, rank in enumerate(ranks):
        subsets[i, :] = combination_by_index(rank, n, k)

    return subsets


def inverse_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply the inverse of matrix a by matrix b.
    We use np.linalg.solve as it tends to produce more accurate results than
    np.linalg.inv.

    Parameters
    ----------
    a : ndarray of shape (n, n)
        Square matrix.

    b : ndarray of shape (n, m)
        Matrix.

    Returns
    -------
    m : ndarray of shape (n, m)
        The inverse of matrix a multiplied by matrix b.
    """
    assert_is_square(a)
    if a.shape[1] != b.shape[0]:
        raise ValueError("Wrong dimension")
    return np.linalg.solve(a, b)


def multiply_by_inverse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply matrix a by the inverse of matrix b.
    We use np.linalg.solve as it tends to produce more accurate results than
    np.linalg.inv.

    Parameters
    ----------
    a : ndarray of shape (n, m)
        Matrix.

    b : ndarray of shape (n, n)
        Square matrix.

    Returns
    -------
    m : ndarray of shape (n, m)
        The matrix a multiplied by the inverse of matrix b.
    """
    return inverse_multiply(b.T, a.T).T


def symmetric_step_up_matrix(n1: int, n2: int) -> np.ndarray:
    """Compute the Symmetric step-up matrix M such that `M @ np.ones(n2) = np.ones(n1)`.

    Parameters
    ----------
    n1 : int
        First dimension.

    n2 : int
        Second dimension.

    Returns
    -------
    m : ndarray of shape (n1, n2)
        The Symmetric step-up matrix.
    """
    assert abs(n1 - n2) <= 1

    if n1 == n2:
        return np.eye(n1)

    if n1 < n2:
        return symmetric_step_up_matrix(n2, n1).T * n1 / n2

    m = np.zeros((n1, n2))
    j_row = np.ones((1, n2)) / n2
    e = np.eye(n2)
    for j in range(n1):
        mj = np.concatenate([e[:j], j_row, e[j:]], axis=0)
        m += mj / n1

    return m
