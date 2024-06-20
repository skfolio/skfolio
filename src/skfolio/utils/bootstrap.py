"""Bootstrap module."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

import numpy as np

__all__ = ["stationary_bootstrap"]


def optimal_block_size(x: np.ndarray) -> float:
    """Compute the optimal block size for a single series using Politis & White
    algorithm [1]_.

    Parameters
    ----------
    x : ndarray
        The input 1D-array.

    Returns
    -------
    value : float
        The optimal block size.

    References
    ----------
    .. [1] "Automatic Block-Length Selection for the Dependent Bootstrap".
        Politis & White (2004).

    .. [2] "Correction to Automatic Block-Length Selection for the Dependent Bootstrap".
        Patton, Politis & White (2009).
    """
    n = x.shape[0]
    eps = x - x.mean(0)
    b_max = np.ceil(min(3 * np.sqrt(n), n / 3))
    kn = max(5, int(np.log10(n)))
    m_max = int(np.ceil(np.sqrt(n))) + kn
    cv = 2 * np.sqrt(np.log10(n) / n)
    acv = np.zeros(m_max + 1)
    abs_acorr = np.zeros(m_max + 1)
    opt_m = None
    for i in range(m_max + 1):
        v1 = eps[i + 1 :] @ eps[i + 1 :]
        v2 = eps[: -(i + 1)] @ eps[: -(i + 1)]
        cross_prod = eps[i:] @ eps[: n - i]
        acv[i] = cross_prod / n
        abs_acorr[i] = np.abs(cross_prod) / np.sqrt(v1 * v2)
        if i >= kn:
            if np.all(abs_acorr[i - kn : i] < cv) and opt_m is None:
                opt_m = i - kn
    m = 2 * max(opt_m, 1) if opt_m is not None else m_max
    m = min(m, m_max)
    g = 0.0
    lr_acv = acv[0]
    for k in range(1, m + 1):
        lam = 1 if k / m <= 1 / 2 else 2 * (1 - k / m)
        g += 2 * lam * k * acv[k]
        lr_acv += 2 * lam * acv[k]
    d = 2 * lr_acv**2
    b = ((2 * g**2) / d) ** (1 / 3) * n ** (1 / 3)
    b = min(b, b_max)
    return b


def stationary_bootstrap(
    returns: np.ndarray,
    n_bootstrap_samples: int,
    block_size: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Creates `n_bootstrap_samples` samples from a multivariate return series via
    stationary bootstrapping.

    Parameters
    ----------
    returns: ndarray of shape (n_observations, n_assets)
        The returns array.

    n_bootstrap_samples: int
        The number of bootstrap samples to generate.

    block_size: float, optional
        The block size.
        If this is set to None, we estimate the optimal block size using Politis &
        White algorithm for all individual asset and the median.

    seed: int, optional
        Random seed used to initialize the pseudo-random number generator

    Returns
    -------
    value: ndarray
           The sample returns of shape (reps, nb observations, nb assets)

    """
    np.random.seed(seed=seed)
    n_observations, n_assets = returns.shape
    x = np.vstack((returns, returns))
    # Loop over reps bootstraps
    if block_size is None:
        block_size = np.median(
            [optimal_block_size(returns[:, i]) for i in range(n_assets)]
        )

    indices = np.random.randint(
        n_observations, size=(n_bootstrap_samples, n_observations)
    )
    cond = np.random.rand(n_bootstrap_samples, n_observations) >= 1.0 / block_size
    # TODO: don't use loop
    for i in range(n_bootstrap_samples):
        for j in range(1, n_observations):
            if cond[i, j]:
                indices[i, j] = indices[i, j - 1] + 1
    indices[indices > 2 * n_observations] = 0
    return x[indices, :]
