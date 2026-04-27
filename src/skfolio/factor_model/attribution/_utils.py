"""Utilities for attribution module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.typing import FloatArray


def _format_percent(x: float) -> str:
    """Format a float as percentage string."""
    if np.isnan(x):
        return "NaN"
    return f"{x * 100:.2f}%"


def _format_decimal(x: float, decimals: int = 4) -> str:
    """Format a float with fixed decimal places."""
    if np.isnan(x):
        return "NaN"
    return f"{x:.{decimals}f}"


def _cov_with_centered(x: FloatArray, y_centered: FloatArray) -> FloatArray | float:
    """Compute covariance of x (or each column of x) with pre-centered y."""
    n = len(y_centered)
    x_centered = x - np.mean(x, axis=0)
    return np.dot(y_centered, x_centered) / (n - 1)


def _format_ci(lower: float, upper: float) -> str:
    """Format a confidence interval as a bracketed percentage string."""
    if np.isnan(lower) or np.isnan(upper):
        return ""
    return f"[{lower * 100:.2f}%, {upper * 100:.2f}%]"


def _format_contrib_with_ci_margin(mu: float, se: float, z: float) -> str:
    r"""Format mean return contribution with optional "±" z*SE margin (percent).

    Used for formatted DataFrames: `\"12.34% ± 2.00%\"` when `se` is finite; otherwise
    the contribution only.
    """
    out = _format_percent(mu)
    if not np.isfinite(se) or se < 0:
        return out
    return f"{out} ± {_format_percent(z * se)}"


def _validate_no_nan(arr: np.ndarray, name: str) -> None:
    """Raise if the array contains any NaN."""
    nan_mask = np.isnan(arr)
    if nan_mask.any():
        n = int(nan_mask.sum())
        raise ValueError(f"`{name}` contains {n:,} NaN value(s).")
