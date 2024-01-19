"""Tools module"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# Statsmodels, Copyright (C) 2006, Jonathan E. Taylor, Licensed under BSD 3 clause.

from enum import auto

import numpy as np
import scipy.cluster.hierarchy as sch
import scipy.optimize as sco
import scipy.spatial.distance as scd
import scipy.special as scs
from scipy.sparse import csr_matrix

from skfolio.utils.tools import AutoEnum

__all__ = [
    "NBinsMethod",
    "n_bins_freedman",
    "n_bins_knuth",
    "is_cholesky_dec",
    "assert_is_square",
    "assert_is_symmetric",
    "assert_is_distance",
    "cov_nearest",
    "cov_to_corr",
    "corr_to_cov",
    "commutation_matrix",
    "compute_optimal_n_clusters",
    "rand_weights",
    "rand_weights_dirichlet",
]


class NBinsMethod(AutoEnum):
    """Enumeration of the Number of Bins Methods

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
    return int(round(n_bins))


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

    def func(y: float):
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
    return int(round(n_bins))


def rand_weights_dirichlet(n: int) -> np.array:
    """Produces n random weights that sum to one from a dirichlet distribution
    (uniform distribution over a simplex)

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


def rand_weights(n: int, zeros: int = 0) -> np.array:
    """Produces n random weights that sum to one from an uniform distribution
    (non-uniform distribution over a simplex)

    Parameters
    ----------
    n : int
        Number of weights.

    zeros : int, default=0
        The number of weights to randomly set to zeros.

    Returns
    -------
    weights : ndarray of shape (n, )
        The vector of weights.
    """
    k = np.random.rand(n)
    if zeros > 0:
        zeros_idx = np.random.choice(n, zeros, replace=False)
        k[zeros_idx] = 0
    return k / sum(k)


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
    except np.linalg.linalg.LinAlgError:
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
        True if if the matrix is positive definite, False otherwise.
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


def cov_nearest(cov: np.ndarray, higham: bool = False, higham_max_iteration: int = 100):
    """Compute the nearest covariance matrix that is positive definite and with a
    cholesky decomposition than can be computed. The variance is left unchanged.

    First, it converts the covariance matrix to a correlation matrix.
    Then, it finds the nearest correlation matrix and converts it back to a covariance
    matrix using the initial standard deviation.

    Cholesky decomposition can fail for symmetric positive definite (SPD) matrix due
    to floating point error and inversely, Cholesky decomposition can success for
    non-SPD matrix. Therefore, we need to test for both. We always start by testing
    for Cholesky decomposition which is significantly faster than checking for positive
    eigenvalues.

    Parameters
    ----------
    cov : ndarray of shape (n, n)
        Covariance matrix.

    higham : bool, default=False
        If this is set to True, the Higham & Nick (2002) algorithm [1]_ is used,
        otherwise the eigenvalues are clipped to threshold above zeros (1e-13).
        The default (`False`) is to use the clipping method as the Higham & Nick
        algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iteration of the Higham & Nick (2002) algorithm.
        The default value is `100`.

    Returns
    -------
    cov : ndarray
        The nearest covariance matrix.

    References
    ----------
    .. [1] "Computing the nearest correlation matrix - a problem from finance"
        IMA Journal of Numerical Analysis
        Higham & Nick (2002)
    """
    assert_is_square(cov)
    assert_is_symmetric(cov)

    # Around 100 times faster than checking eigenvalues with np.linalg.eigh
    if is_cholesky_dec(cov) and is_positive_definite(cov):
        return cov

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
    max_clusters = max(8, round(np.sqrt(n)))
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
