"""Utility functions for online portfolio selection."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

CLIP_EPSILON = 1e-12


def relatives_to_net(X: ArrayLike) -> np.ndarray:
    """Convert finite nonnegative price relatives to net returns.

    Accepts one observation or a matrix and returns shape (T, n_assets).
    Zero relatives represent a total loss and convert to -1 without clipping.
    Algorithms may impose stricter positivity requirements.
    """
    relatives = np.asarray(X, dtype=float)
    if relatives.ndim not in (1, 2):
        raise ValueError("Price relatives must be a 1D or 2D array")
    if not np.all(np.isfinite(relatives)) or np.any(relatives < 0):
        raise ValueError("Price relatives must be finite and nonnegative")
    return np.atleast_2d(relatives) - 1.0


def evaluate_learning_rate(
    schedule: float | np.ndarray | Callable[[int], float], t: int
) -> float:
    """Evaluate a schedule, holding its final array value after exhaustion."""
    if callable(schedule):
        return float(np.asarray(schedule(t)))
    if isinstance(schedule, np.ndarray):
        if t < len(schedule):
            return float(schedule[t])
        return float(schedule[-1])
    return float(schedule)
