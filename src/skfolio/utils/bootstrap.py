"""Bootstrap module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Some implementations derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

from __future__ import annotations

import numpy as np

from skfolio.typing import ArrayLike, FloatArray, IntArray

__all__ = ["stationary_bootstrap"]


def optimal_block_size(x: FloatArray) -> float:
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
    returns: ArrayLike,
    n_bootstrap_samples: int,
    block_size: float | None = None,
    seed: int | None = None,
) -> FloatArray:
    """Create `n_bootstrap_samples` samples from a multivariate return series via
    stationary bootstrapping.

    Parameters
    ----------
    returns: array-like of shape (n_observations, n_assets)
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
    samples : ndarray of shape (n_bootstrap_samples, n_observations, n_assets)
        The resampled returns.

    Notes
    -----
    All samples are allocated in memory. The output contains
    `n_bootstrap_samples * n_observations * n_assets` values.
    """
    returns = np.asarray(returns)
    indices = _stationary_bootstrap_indices(
        returns, n_bootstrap_samples, block_size, seed
    )
    return returns[indices]


def _stationary_bootstrap_indices(
    returns: FloatArray,
    n_bootstrap_samples: int,
    block_size: float | None,
    seed: int | None,
) -> IntArray:
    """Generate stationary bootstrap indices into the original returns array.

    Parameters
    ----------
    returns : ndarray of shape (n_observations, n_assets)
        Asset returns. Used to determine the number of observations and, when
        `block_size` is None, to estimate the block size.

    n_bootstrap_samples : int
        Number of bootstrap samples to generate. Each sample contains `n_observations`
        observations.

    block_size : float or None
        Expected number of consecutive observations in a block. If None, use the median
        of the optimal block sizes estimated for each asset with the Politis & White
        algorithm.

    seed : int or None
        Seed for the local pseudo-random number generator. If None, initialize the
        generator without a fixed seed.

    Returns
    -------
    indices : ndarray of shape (n_bootstrap_samples, n_observations)
        Integer row indices in the range `[0, n_observations)`. Each row defines one
        bootstrap sample, with the same observation indices used for all assets. The
        corresponding sample is `returns[indices[i]]`.
    """
    rng = np.random.RandomState(seed=seed)
    n_observations, n_assets = returns.shape
    if block_size is None:
        block_size = np.median(
            [optimal_block_size(returns[:, i]) for i in range(n_assets)]
        )

    indices = rng.randint(n_observations, size=(n_bootstrap_samples, n_observations))
    continue_block = rng.rand(n_bootstrap_samples, n_observations) >= 1.0 / block_size
    # Continue the block at the next observation, or restart at a random one.
    for observation in range(1, n_observations):
        indices[:, observation] = np.where(
            continue_block[:, observation],
            indices[:, observation - 1] + 1,
            indices[:, observation],
        )
    indices %= n_observations
    return indices
