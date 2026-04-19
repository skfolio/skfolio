"""Utility functions for cross-sectional transformers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers

import numpy as np
import numpy.typing as npt
from sklearn.utils import check_array

# Scaling constant for MAD under normality (MAD multiplied by this value is consistent
# with the STD).
_MAD_CONSISTENCY = 1.4826022185056018


def _safe_divide(
    numerator: np.ndarray | float, denominator: np.ndarray | float
) -> np.ndarray:
    """Return an element-wise division with safe zero handling.

    Division is performed only where the denominator is non-zero. All other ntries are
    set to zero. Any non-finite intermediate is coerced to zero so he returned array is
    always finite.
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    quotient = np.zeros(
        np.broadcast_shapes(numerator.shape, denominator.shape), dtype=np.float64
    )
    np.divide(numerator, denominator, out=quotient, where=denominator != 0)
    np.nan_to_num(quotient, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return quotient


def _cs_weighted_mean(
    X: np.ndarray, cs_weights: np.ndarray | None, estimation_mask: np.ndarray
) -> np.ndarray:
    """Return the cross-sectional weighted mean for each observation.

    The weighted product is computed in place only at estimation entries, avoiding
    both the full `cs_weights * X` temporary and the `0 * NaN = NaN` artifact at
    non-estimation cells.
    """
    if cs_weights is None:
        return np.nanmean(X, axis=1, keepdims=True)

    weighted_X = np.zeros_like(X)
    np.multiply(cs_weights, X, out=weighted_X, where=estimation_mask)
    weighted_sum = weighted_X.sum(axis=1, keepdims=True)
    total_weight = np.sum(cs_weights, axis=1, keepdims=True)
    return _safe_divide(weighted_sum, total_weight)


def _cs_equal_weighted_std(
    X: np.ndarray, mean: np.ndarray | float, estimation_mask: np.ndarray
) -> np.ndarray:
    """Return the cross-sectional equal-weighted standard deviation around `mean`."""
    # `np.subtract(out=, where=)` skips the subtraction where estimation_mask is False,
    # which also avoids `NaN` propagation when X is non-finite outside the mask.
    residuals = np.zeros(X.shape, dtype=np.float64)
    np.subtract(X, mean, out=residuals, where=estimation_mask)
    sample_size = np.sum(estimation_mask, axis=1, keepdims=True)
    degrees_of_freedom = np.maximum(sample_size - 1.0, 0.0)
    return np.sqrt(
        _safe_divide(
            np.sum(residuals * residuals, axis=1, keepdims=True),
            degrees_of_freedom,
        )
    )


def _cs_recenter_rescale(
    X: np.ndarray,
    finite_mask: np.ndarray,
    cs_weights: np.ndarray | None,
    estimation_mask: np.ndarray,
    atol: float,
    scale: bool,
) -> np.ndarray:
    """Return `X` recentered and optionally rescaled on the estimation universe.

    The output is recentered to weighted mean zero and, when `scale` is True,
    rescaled to unit equal-weighted standard deviation over the estimation universe.

    - `X` must be NaN-masked at non-finite entries (i.e. `X[~finite_mask]` is NaN).
    - When `cs_weights` is provided, weights must already be zero outside the
      estimation universe (typically via :func:`_prepare_cs_estimation_inputs`).
    - Each observation must contain at least one estimation asset.
    """
    centered_X = X - _cs_weighted_mean(X, cs_weights, estimation_mask)
    if not scale:
        return centered_X

    equal_weighted_std = _cs_equal_weighted_std(centered_X, 0.0, estimation_mask)

    # Constant cross-sections (std below atol) get a zero output rather than NaN
    # propagation from a 0/0 division; the mask is reapplied afterwards.
    scaled_X = np.zeros_like(X)
    np.divide(
        centered_X,
        equal_weighted_std,
        out=scaled_X,
        where=finite_mask & (equal_weighted_std > atol),
    )
    return np.where(finite_mask, scaled_X, np.nan)


def _validate_cs_weights(
    X: np.ndarray, cs_weights: npt.ArrayLike | None
) -> np.ndarray | None:
    """Validate `cs_weights` against `X`."""
    if cs_weights is None:
        return None
    cs_weights = check_array(cs_weights, dtype=np.float64, input_name="cs_weights")
    if cs_weights.shape != X.shape:
        raise ValueError("`cs_weights` must have the same shape as `X`.")
    if np.any(cs_weights < 0):
        raise ValueError("`cs_weights` must be non-negative.")
    return cs_weights


def _prepare_cs_estimation_inputs(
    X: np.ndarray, cs_weights: np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    """Return the effective weights, the finite mask, and the estimation mask.

    When `cs_weights` is provided, non-finite entries of `X` receive zero weight.
    """
    finite_mask = np.isfinite(X)

    if cs_weights is None:
        estimation_mask = finite_mask
    else:
        cs_weights = np.where(finite_mask, cs_weights, 0.0)
        estimation_mask = finite_mask & (cs_weights > 0)

    if not estimation_mask.any(axis=1).all():
        raise ValueError("Each observation must contain at least one estimation asset.")

    return cs_weights, finite_mask, estimation_mask


def _mask_non_estimation_values(
    X: np.ndarray, cs_weights: np.ndarray | None
) -> np.ndarray:
    """Mask values outside the estimation universe with NaN.

    When `cs_weights` is `None`, the validated `X` is returned unchanged.
    Otherwise, values outside the estimation universe are replaced with NaN so
    functions such as `numpy.nanmedian` and `numpy.nanpercentile` ignore them.
    """
    if cs_weights is None:
        masked_X = X
    else:
        masked_X = np.where((cs_weights > 0) & np.isfinite(X), X, np.nan)

    if not np.isfinite(masked_X).any(axis=1).all():
        raise ValueError("Each observation must contain at least one estimation asset.")
    return masked_X


def _global_group_keys(n_observations: int, n_assets: int) -> np.ndarray:
    """Return the global cross-sectional group key for each entry.

    The output is a contiguous 2D array. Downstream consumers ravel it once and a
    contiguous source avoids the implicit copy that broadcast views would incur.
    """
    return np.repeat(np.arange(n_observations, dtype=np.int64), n_assets).reshape(
        n_observations, n_assets
    )


def _validate_and_normalize_groups(
    X: np.ndarray, cs_groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    """Validate `cs_groups` and map labels to contiguous non-negative ids.

    Missing entries, encoded as `-1`, are routed to a dedicated dummy id placed after
    all valid ids so valid-group aggregates are never contaminated by missing entries.

    Returns
    -------
    group_ids : ndarray of shape `X.shape`
        Contiguous integer ids in `[0, n_groups)`. The dummy id, when present, is
        `n_groups - 1`.

    missing_group_mask : ndarray of bool of shape `X.shape`
        Entries originally labeled `-1`.

    n_groups : int
        Total number of groups, including the dummy one when present.
    """
    cs_groups = np.asarray(cs_groups)
    if cs_groups.shape != X.shape:
        raise ValueError("`cs_groups` must have the same shape as `X`.")
    if np.issubdtype(cs_groups.dtype, np.integer):
        cs_groups = cs_groups.astype(np.int64, copy=False)
    elif cs_groups.dtype == object:
        flat = cs_groups.ravel()
        if not all(
            isinstance(value, numbers.Integral) and not isinstance(value, bool)
            for value in flat
        ):
            raise ValueError("`cs_groups` must be an integer array.")
        cs_groups = cs_groups.astype(np.int64)
    else:
        raise ValueError("`cs_groups` must be an integer array.")
    if np.any(cs_groups < -1):
        raise ValueError("`cs_groups` must contain integers >= -1.")

    missing_group_mask = cs_groups == -1
    valid_group_labels = cs_groups[~missing_group_mask]

    if valid_group_labels.size == 0:
        group_ids = np.zeros(cs_groups.shape, dtype=np.int64)
        return group_ids, missing_group_mask, 1

    # Build a contiguous remap proportional to the number of unique ids rather than to
    # their max value, so a sparse label set (e.g. {0, 10**9}) does not allocate a giant
    # lookup table.
    unique_group_labels, inverse_group_ids = np.unique(
        valid_group_labels, return_inverse=True
    )
    inverse_group_ids = inverse_group_ids.astype(np.int64, copy=False)
    n_valid_groups = unique_group_labels.size

    group_ids = np.empty(cs_groups.shape, dtype=np.int64)
    group_ids[~missing_group_mask] = inverse_group_ids
    if np.any(missing_group_mask):
        group_ids[missing_group_mask] = n_valid_groups
        n_groups = n_valid_groups + 1
    else:
        n_groups = n_valid_groups
    return group_ids, missing_group_mask, n_groups


def _cs_group_keys(group_ids: np.ndarray, n_groups: int) -> np.ndarray:
    """Return the composite group key for each observation-group pair."""
    n_observations = group_ids.shape[0]
    observation_ids = np.arange(n_observations, dtype=np.int64)[:, None]
    return observation_ids * n_groups + group_ids


def _group_key_midrank_percentile(
    X: np.ndarray,
    estimation_mask: np.ndarray,
    group_keys: np.ndarray,
    n_group_keys: int,
    finite_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return percentile ranks by composite group key with midrank ties.

    For each finite value, the percentile is computed within its composite group key
    using average-rank tie handling: values strictly below count fully, tied values
    count half, and the result is divided by the group size. The output is then
    clipped to :math:`[0.5 / n,\, 1 - 0.5 / n]` for numerical stability.

    Notes
    -----
    The implementation avoids per-group sorting by combining two ideas:

    1. A single dense ranking of all values (estimation and queried) maps floats to
       small non-negative integers where ties share the same rank.
    2. A composite integer key `group_key * stride + rank` encodes both the group
       membership and the rank. Sorting estimation samples by this composite key
       once produces an array that is correctly ordered within each group, so a
       single global :func:`numpy.searchsorted` resolves the per-group `count_lt`
       and `count_le` for every queried value at once.
    """
    if finite_mask is None:
        finite_mask = np.isfinite(X)
    percentile_ranks = np.full(X.shape, np.nan, dtype=np.float64)
    estimation_counts = np.zeros(n_group_keys, dtype=np.int64)
    if not np.any(finite_mask):
        return percentile_ranks, estimation_counts

    group_keys_flat = group_keys.ravel().astype(np.int64, copy=False)
    finite_mask_flat = finite_mask.ravel()
    estimation_mask_flat = estimation_mask.ravel()

    X_flat = X.ravel()
    finite_values = X_flat[finite_mask_flat]
    finite_group_keys = group_keys_flat[finite_mask_flat]
    sample_values = X_flat[estimation_mask_flat]
    sample_group_keys = group_keys_flat[estimation_mask_flat]

    if sample_values.size == 0:
        return percentile_ranks, estimation_counts

    combined = np.concatenate((sample_values, finite_values))
    _, dense_rank = np.unique(combined, return_inverse=True)
    dense_rank = dense_rank.astype(np.int64, copy=False)
    sample_rank = dense_rank[: sample_values.size]
    finite_rank = dense_rank[sample_values.size :]

    sample_order = np.lexsort((sample_rank, sample_group_keys))
    sorted_group_keys = sample_group_keys[sample_order]
    sorted_rank = sample_rank[sample_order]

    estimation_counts = np.bincount(sample_group_keys, minlength=n_group_keys)
    group_key_start = np.concatenate(([0], np.cumsum(estimation_counts))).astype(
        np.int64
    )

    stride = int(dense_rank.max()) + 2
    sorted_composite_keys = sorted_group_keys * stride + sorted_rank
    finite_composite_keys = finite_group_keys.astype(np.int64) * stride + finite_rank

    count_lt = (
        np.searchsorted(sorted_composite_keys, finite_composite_keys, side="left")
        - group_key_start[finite_group_keys]
    )
    count_le = (
        np.searchsorted(sorted_composite_keys, finite_composite_keys, side="right")
        - group_key_start[finite_group_keys]
    )
    finite_group_sizes = estimation_counts[finite_group_keys]
    count_lt = np.clip(count_lt, 0, finite_group_sizes)
    count_le = np.clip(count_le, 0, finite_group_sizes)

    # `safe_size` defaults empty groups to 1 so the division never produces NaN; the
    # explicit `np.where` below restores NaN for those entries.
    positive_size = finite_group_sizes > 0
    safe_size = np.where(positive_size, finite_group_sizes, 1).astype(np.float64)
    midranks = 0.5 * (count_lt + count_le) / safe_size
    lower_bound = 0.5 / safe_size
    upper_bound = 1.0 - lower_bound
    percentile_values = np.where(
        positive_size,
        np.clip(midranks, lower_bound, upper_bound),
        np.nan,
    )

    percentile_ranks[finite_mask] = percentile_values
    return percentile_ranks, estimation_counts


def _cs_percentile_rank(
    X: np.ndarray,
    finite_mask: np.ndarray,
    estimation_mask: np.ndarray,
    cs_groups: npt.ArrayLike | None,
    min_group_size: int,
) -> np.ndarray:
    """Return cross-sectional percentile ranks with optional grouped fallback.

    When `cs_groups` is `None`, ranks are computed over the full cross-section.
    Otherwise, ranks are computed within each group, with fallback to the full
    cross-section for small or missing groups.
    """
    n_observations, n_assets = X.shape

    if cs_groups is None:
        observation_group_keys = _global_group_keys(n_observations, n_assets)
        percentile, _ = _group_key_midrank_percentile(
            X,
            estimation_mask,
            observation_group_keys,
            n_observations,
            finite_mask=finite_mask,
        )
        return percentile

    group_ids, missing_group_mask, n_groups = _validate_and_normalize_groups(
        X=X,
        cs_groups=cs_groups,
    )
    group_keys = _cs_group_keys(group_ids, n_groups)
    n_group_keys = n_observations * n_groups

    estimation_group_mask = estimation_mask & ~missing_group_mask
    grouped_percentile, group_sizes = _group_key_midrank_percentile(
        X,
        estimation_group_mask,
        group_keys,
        n_group_keys,
        finite_mask=finite_mask,
    )
    # Decide fallback at the (observation, group) level and broadcast once.
    small_group = group_sizes < min_group_size
    fallback_mask = small_group[group_keys] | missing_group_mask

    if not np.any(fallback_mask):
        return grouped_percentile

    # Only observations that contain at least one fallback entry need the
    # observation-level rank. Restricting the sort to those observations is much faster
    # when the fallback is sparse.
    observations_with_fallback = fallback_mask.any(axis=1)
    X_fallback = X[observations_with_fallback]
    estimation_fallback = estimation_mask[observations_with_fallback]
    finite_fallback = finite_mask[observations_with_fallback]
    fallback_group_keys = _global_group_keys(X_fallback.shape[0], n_assets)
    fallback_percentile, _ = _group_key_midrank_percentile(
        X_fallback,
        estimation_fallback,
        fallback_group_keys,
        X_fallback.shape[0],
        finite_mask=finite_fallback,
    )

    fallback_observation_mask = fallback_mask[observations_with_fallback]
    grouped_percentile[observations_with_fallback] = np.where(
        fallback_observation_mask,
        fallback_percentile,
        grouped_percentile[observations_with_fallback],
    )
    return grouped_percentile
