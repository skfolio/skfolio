"""Shared helpers of the factor model diagnostics."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np

from skfolio.typing import (
    ArrayLike,
    FloatArray,
    IntArray,
    StrArray,
)
from skfolio.utils.stats import (
    CorrelationMethod,
    safe_divide,
)


class _GramDiagnostics(NamedTuple):
    """Per-observation Gram-matrix diagnostics."""

    t_stats: FloatArray  # (n_observations - exposure_lag, n_reduced_factors)
    vif: FloatArray  # (n_observations - exposure_lag, n_reduced_factors)
    condition_number: FloatArray  # (n_observations - exposure_lag,)


class _RegressionData(NamedTuple):
    """Lag-aligned data used by cross-sectional regression diagnostics."""

    exposures: FloatArray
    factor_returns: FloatArray
    idio_returns: FloatArray
    regression_weights: FloatArray | None
    factor_names: StrArray


def _cs_kurtosis(z: FloatArray) -> FloatArray:
    """Cross-sectional excess kurtosis per observation.

    Uses the bias-corrected (Fisher) estimator matching scipy
    defaults. Observations with fewer than 4 valid assets return
    NaN.

    Parameters
    ----------
    z : ndarray of shape (n_observations, n_assets)

    Returns
    -------
    kurtosis : ndarray of shape (n_observations,)
    """
    n = np.sum(np.isfinite(z), axis=1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = z - np.nanmean(z, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = np.nansum(d**2, axis=1) / n
        m4 = np.nansum(d**4, axis=1) / n
        raw_kurt = m4 / m2**2 - 3.0

    kurt = np.full(z.shape[0], np.nan)
    ok = n >= 4
    if np.any(ok):
        adj = (n[ok] - 1) / ((n[ok] - 2) * (n[ok] - 3))
        kurt[ok] = ((n[ok] + 1) * raw_kurt[ok] + 6) * adj
    return kurt


def _cs_skewness(z: FloatArray) -> FloatArray:
    """Cross-sectional skewness per observation.

    Uses the bias-corrected (Fisher) estimator matching scipy
    defaults. Observations with fewer than 3 valid assets return
    NaN.

    Parameters
    ----------
    z : ndarray of shape (n_observations, n_assets)

    Returns
    -------
    skewness : ndarray of shape (n_observations,)
    """
    n = np.sum(np.isfinite(z), axis=1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = z - np.nanmean(z, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = np.nansum(d**2, axis=1) / n
        m3 = np.nansum(d**3, axis=1) / n
        raw_skew = m3 / m2**1.5

    skew = np.full(z.shape[0], np.nan)
    ok = n >= 3
    if np.any(ok):
        skew[ok] = raw_skew[ok] * np.sqrt(n[ok] * (n[ok] - 1)) / (n[ok] - 2)
    return skew


def _check_correlation_method(correlation_method: CorrelationMethod) -> None:
    """Check that `correlation_method` is a `CorrelationMethod`."""
    if not isinstance(correlation_method, CorrelationMethod):
        raise TypeError("correlation_method must be a `CorrelationMethod`.")


def _exceedance_agg(threshold: float):
    """Return an aggregation function for t-stat exceedance rate."""

    def _agg(raw_t: FloatArray) -> FloatArray:
        significant = np.abs(raw_t) > threshold
        n_valid = np.sum(np.isfinite(raw_t), axis=0)
        return safe_divide(np.nansum(significant, axis=0), n_valid, fill_value=0.0)

    return _agg


def _lag1_autocorr(x: FloatArray) -> FloatArray:
    """Column-wise lag-1 Pearson autocorrelation."""
    a, b = x[:-1], x[1:]
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def _selector_to_positions(
    selector: ArrayLike | slice | None, labels: StrArray, *, axis_name: str
) -> IntArray:
    """Resolve an axis selector to positional indices.

    Parameters
    ----------
    selector : array-like, slice or None
        Axis selector. Boolean arrays are interpreted as masks, integer arrays
        as positional selectors, slices as positional slices, and other arrays
        as labels matched against `labels`. Negative integer positions follow
        NumPy indexing rules. If `None`, all positions are kept.

    labels : ndarray of shape (n_labels,)
        Labels available on the selected axis.

    axis_name : str
        Axis name used in validation error messages.

    Returns
    -------
    positions : ndarray of shape (n_selected,)
        Positional indices corresponding to `selector`.

    Raises
    ------
    ValueError
        If `selector` is not one-dimensional, if a boolean mask has the wrong
        length, if an integer selector is out of bounds, or if a label is not
        present in `labels`.
    """
    n_labels = len(labels)
    if selector is None:
        return np.arange(n_labels, dtype=np.intp)

    if isinstance(selector, slice):
        start, stop, step = selector.indices(n_labels)
        return np.arange(start, stop, step, dtype=np.intp)

    arr = np.asarray(selector)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"`{axis_name}` must be a 1D selector.")

    if np.issubdtype(arr.dtype, np.bool_):
        if arr.shape[0] != n_labels:
            raise ValueError(
                f"Boolean `{axis_name}` selector must have length {n_labels}, "
                f"got {arr.shape[0]}."
            )
        return np.flatnonzero(arr).astype(np.intp, copy=False)

    if np.issubdtype(arr.dtype, np.integer):
        positions = arr.astype(np.intp, copy=False)
        if np.any((positions < -n_labels) | (positions >= n_labels)):
            raise ValueError(
                f"Integer `{axis_name}` selector contains out-of-bounds positions."
            )
        if np.any(positions < 0):
            positions = positions.copy()
            positions[positions < 0] += n_labels
        return positions

    missing_sentinel = object()
    label_lookup = {label: idx for idx, label in enumerate(labels)}
    positions = np.empty(arr.shape[0], dtype=np.intp)
    missing = []
    for target_index, label in enumerate(arr):
        position = label_lookup.get(label, missing_sentinel)
        if position is missing_sentinel:
            missing.append(label)
        else:
            positions[target_index] = position

    if missing:
        raise ValueError(
            f"{len(missing)} {axis_name} label(s) not found in FactorModel. "
            f"First five: {missing[:5]}"
        )
    return positions


def _positions_to_indexer(positions: IntArray) -> IntArray | slice:
    """Convert contiguous positional indices to an indexer.

    Parameters
    ----------
    positions : ndarray of shape (n_selected,)
        Positional indices on a single axis.

    Returns
    -------
    indexer : ndarray or slice
        A `slice` when `positions` is contiguous, otherwise the input integer
        positions. The slice form lets NumPy return views for contiguous
        selections.
    """
    if len(positions) > 0 and (len(positions) == 1 or np.all(np.diff(positions) == 1)):
        return slice(int(positions[0]), int(positions[-1]) + 1)
    return positions
