"""Sparse sorting-network formulation of the empirical Gini mean difference."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import functools

import numpy as np
import scipy.sparse as sp

from skfolio.measures import owa_gmd_weights
from skfolio.typing import FloatArray


def _odd_even_merge_sort_comparators(n: int) -> list[tuple[int, int]]:
    """Batcher odd-even merge sort comparators in execution order.

    The network also supports observation counts that are not powers of two:
    comparisons involving missing wires are omitted.
    """
    comparators = []
    p = 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        comparators.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return comparators


@functools.lru_cache(maxsize=8)
def _gmd_sorting_network(
    n_observations: int,
) -> tuple[sp.csr_matrix, sp.csr_matrix, FloatArray]:
    r"""Exact GMD epigraph using a sparse Batcher sorting network.

    Each comparator with inputs ``u`` and ``v`` has outputs ``lo`` and ``hi``
    constrained by ``lo + hi = u + v``, ``hi >= u`` and ``hi >= v``. Minimizing
    the ascending OWA weights on the final outputs yields the exact GMD.
    Feasible outputs need not themselves be sorted.

    The formulation uses :math:`O(T\log^2 T)` variables and constraints and is
    the LP dual of Goemans' (2015) permutahedron formulation:
    https://doi.org/10.1007/s10107-014-0757-1
    """
    if n_observations < 2:
        raise ValueError("GMD requires at least two observations")

    comparators = _odd_even_merge_sort_comparators(n_observations)
    n_comparators = len(comparators)
    n_variables = n_observations + 2 * n_comparators
    source = np.arange(n_observations)
    lo = n_observations + 2 * np.arange(n_comparators)
    hi = lo + 1
    u = np.empty(n_comparators, dtype=int)
    v = np.empty(n_comparators, dtype=int)
    for k, (i, j) in enumerate(comparators):
        u[k], v[k] = source[i], source[j]
        source[i], source[j] = lo[k], hi[k]

    rows = np.repeat(np.arange(n_comparators), 4)
    cols = np.column_stack([lo, hi, u, v]).ravel()
    vals = np.tile([1.0, 1.0, -1.0, -1.0], n_comparators)
    equality = sp.csr_matrix((vals, (rows, cols)), shape=(n_comparators, n_variables))

    rows = np.repeat(np.arange(2 * n_comparators), 2)
    cols = np.column_stack([u, hi, v, hi]).ravel()
    vals = np.tile([1.0, -1.0], 2 * n_comparators)
    inequality = sp.csr_matrix(
        (vals, (rows, cols)), shape=(2 * n_comparators, n_variables)
    )

    risk_coefficients = np.zeros(n_variables)
    risk_coefficients[source] = owa_gmd_weights(n_observations)
    # Cached arrays are shared by independent fits. Keep them immutable.
    for matrix in (equality, inequality):
        for values in (matrix.data, matrix.indices, matrix.indptr):
            values.flags.writeable = False
    risk_coefficients.flags.writeable = False
    return equality, inequality, risk_coefficients
