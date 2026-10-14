"""Utility functions for online portfolio selection."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

CLIP_EPSILON = 1e-12


def net_to_relatives(X: ArrayLike) -> np.ndarray:
    """Convert net returns to price relatives robustly.

    Accepts 1D or 2D inputs and always returns a 2D array of shape (T, n).
    """
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    return np.maximum(1.0 + X_arr, CLIP_EPSILON)


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
