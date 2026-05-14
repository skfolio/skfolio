"""Shared utilities for factor-model estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import defaultdict

import numpy as np

from skfolio.linear_model import BaseCSLinearModel, CSLinearRegression
from skfolio.preprocessing import CSStandardScaler
from skfolio.typing import AnyArray, ArrayLike, FloatArray, ObjArray
from skfolio.utils.stats import safe_divide


def _factor_name_maps(
    factor_names: ObjArray, factor_families: ObjArray | None = None
) -> tuple[dict[str, int], dict[str, list[int]]]:
    """Build lookup maps for factor names and factor families.

    Parameters
    ----------
    factor_names : ndarray of shape (n_factors,)
        Factor names.

    factor_families : ndarray of shape (n_factors,), optional
        Family label for each factor. If `None`, the family map is empty.

    Returns
    -------
    factor_to_idx : dict of {str: int}
        Mapping from factor name to factor index.

    family_to_idx : dict of {str: list[int]}
        Mapping from family name to the factor indices in that family.
    """
    factor_to_idx = {v: i for i, v in enumerate(factor_names)}
    family_to_idx = defaultdict(list)
    if factor_families is not None:
        for i, v in enumerate(factor_families):
            family_to_idx[v].append(i)
    return factor_to_idx, dict(family_to_idx)


def _resolve_factor_name(
    name: str, factor_to_idx: dict[str, int], family_to_idx: dict[str, list[int]]
) -> set[int]:
    """Resolve one factor or family name to factor indices.

    Factor names take precedence over family names when the same label appears in both
    maps.

    Parameters
    ----------
    name : str
        Factor name or family name to resolve.

    factor_to_idx : dict of {str: int}
        Mapping from factor name to factor index.

    family_to_idx : dict of {str: list[int]}
        Mapping from family name to factor indices.

    Returns
    -------
    indices : set[int]
        Resolved factor indices.

    Raises
    ------
    ValueError
        If `name` is neither a factor name nor a family name.
    """
    if name in factor_to_idx:
        return {factor_to_idx[name]}
    if name in family_to_idx:
        return set(family_to_idx[name])
    raise ValueError(
        f"'{name}' is neither a factor name nor a family name. "
        f"Available factors: {list(factor_to_idx)}. "
        f"Available families: {list(family_to_idx)}"
    )


def _expand_factor_names(
    names: list[str], factor_to_idx: dict[str, int], family_to_idx: dict[str, list[int]]
) -> list[int]:
    """Expand factor and family names to ordered unique factor indices.

    Each input name may resolve to one factor or to all members of a family.
    Repeated factors are kept only at their first occurrence in the expanded
    sequence.

    Parameters
    ----------
    names : list of str
        Factor names or family names to expand.

    factor_to_idx : dict of {str: int}
        Mapping from factor name to factor index.

    family_to_idx : dict of {str: list[int]}
        Mapping from family name to factor indices.

    Returns
    -------
    indices : list[int]
        Ordered deduplicated factor indices.

    Raises
    ------
    ValueError
        If any name is neither a factor name nor a family name.
    """
    indices = []
    seen = set()
    for name in names:
        for idx in sorted(_resolve_factor_name(name, factor_to_idx, family_to_idx)):
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
    return indices


def _cross_sectional_neutralize(
    y: FloatArray,
    x: FloatArray,
    cs_weights: FloatArray,
    cs_regressor: BaseCSLinearModel | None = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Neutralize a panel against cross-sectional explanatory variables.

    For each observation :math:`t`, regress :math:`y_t` on :math:`X_t` across
    assets with cross-sectional weights and return the residuals:

    .. math::

        \tilde{y}_t = y_t - X_t \hat{\beta}_t

    Missing `y` values or rows of `x` with missing features are excluded by
    setting their effective regression weights to zero.

    Parameters
    ----------
    y : ndarray of shape (n_observations, n_assets)
        Cross-sectional response values to neutralize.

    x : ndarray of shape (n_observations, n_assets, n_features)
        Cross-sectional explanatory variables.

    cs_weights : ndarray of shape (n_observations, n_assets)
        Base cross-sectional weights. Entries whose `y` or `x` values are
        missing receive zero effective weight.

    cs_regressor : BaseCSLinearModel, optional
        Cross-sectional linear regressor used for residualization. If `None`,
        `CSLinearRegression(fit_intercept=False)` is used.

    Returns
    -------
    neutralized : ndarray of shape (n_observations, n_assets)
        Cross-sectional residuals. Missing values in `y` or `x` propagate to
        the corresponding residual entries.

    weights : ndarray of shape (n_observations, n_assets)
        Effective regression weights after missing-value exclusion.
    """
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=2)
    weights = np.where(valid, cs_weights, 0.0)
    regressor = (
        CSLinearRegression(fit_intercept=False)
        if cs_regressor is None
        else cs_regressor
    )
    neutralized = y - regressor.fit(x, y, cs_weights=weights).predict(x)
    return neutralized, weights


def _neutralize_scores(
    neutralize_against: list[str],
    scores: FloatArray,
    exposures: FloatArray,
    cs_weights: FloatArray,
    factor_names: ObjArray,
    factor_families: ObjArray | None = None,
) -> FloatArray:
    """Neutralize descriptor scores against selected factor exposures.

    Parameters
    ----------
    neutralize_against : list of str
        Factor names or family names to neutralize each descriptor score against.

    scores : ndarray of shape (n_descriptors, n_observations, n_assets)
        Descriptor score panels. The array is modified in-place.

    exposures : ndarray of shape (n_observations, n_assets, n_factors)
        Factor exposures used as neutralization variables.

    cs_weights : ndarray of shape (n_observations, n_assets)
        Cross-sectional weights for the neutralization regressions.

    factor_names : ndarray of shape (n_factors,)
        Factor names.

    factor_families : ndarray of shape (n_factors,), optional
        Family label for each factor. If provided, `neutralize_against` may contain
        family names.

    Returns
    -------
    scores : ndarray of shape (n_descriptors, n_observations, n_assets)
        The input score array with each descriptor replaced by its neutralized residuals.

    Raises
    ------
    ValueError
        If a neutralization target is neither a factor name nor a family name.
    """
    factor_to_idx, family_to_idx = _factor_name_maps(factor_names, factor_families)
    targets_idx = _expand_factor_names(neutralize_against, factor_to_idx, family_to_idx)
    if len(targets_idx) == 0:
        return scores

    x = exposures[:, :, targets_idx]
    for i, score in enumerate(scores):
        scores[i], _ = _cross_sectional_neutralize(y=score, x=x, cs_weights=cs_weights)

    return scores


def _neutralize_exposures(
    cs_regressor: BaseCSLinearModel,
    neutralize_against: dict[str, list[str]],
    exposures: FloatArray,
    benchmark_weights: FloatArray,
    factor_names: ObjArray,
    factor_families: ObjArray,
) -> None:
    """Neutralize factor exposures against specified factors or families.

    For each entry in `neutralize_against`, the key's exposures are regressed against
    the target factors and replaced in-place by standardized residuals. Keys and targets
    accept factor names or family names. Entries are processed in insertion order, so
    later entries see exposures already modified by earlier entries.

    Parameters
    ----------
    cs_regressor : BaseCSLinearModel
        Cross-sectional linear regressor used for residualization.

    neutralize_against : dict of {str: list[str]}
        Mapping from factor name or family name to the factor names or family names it
        must be neutralized against. A family key neutralizes each factor in that family
        independently against the same targets.

    exposures : ndarray of shape (n_observations, n_assets, n_factors)
        Factor exposures. Neutralized columns are overwritten in-place.

    benchmark_weights : ndarray of shape (n_observations, n_assets)
        Cross-sectional weights for neutralization and residual scaling.

    factor_names : ndarray of shape (n_factors,)
        Factor names.

    factor_families : ndarray of shape (n_factors,)
        Family label for each factor.

    Raises
    ------
    ValueError
        If a key or target is neither a factor name nor a family name, or if a key
        resolves to factors that overlap with its neutralization targets.
    """
    factor_to_idx, family_to_idx = _factor_name_maps(factor_names, factor_families)

    for key, targets in neutralize_against.items():
        neutralize_idx = _resolve_factor_name(key, factor_to_idx, family_to_idx)
        targets_list = _expand_factor_names(targets, factor_to_idx, family_to_idx)
        targets_idx = set(targets_list)

        overlap = neutralize_idx & targets_idx
        if overlap:
            overlap_names = sorted(factor_names[i] for i in overlap)
            raise ValueError(
                f"`neutralize_against` key '{key}' resolves to factors "
                f"that overlap with its targets: {overlap_names}. "
                f"A factor cannot be neutralized against itself."
            )

        x = exposures[:, :, targets_list]
        for factor_idx in sorted(neutralize_idx):
            neutralized, weights = _cross_sectional_neutralize(
                y=exposures[:, :, factor_idx],
                x=x,
                cs_weights=benchmark_weights,
                cs_regressor=cs_regressor,
            )
            exposures[:, :, factor_idx] = CSStandardScaler().fit_transform(
                neutralized, cs_weights=weights
            )


def _powered_market_cap(market_caps: ArrayLike, power: float = 1.0) -> np.ndarray:
    r"""Compute :math:`\text{mcap}^{\text{power}}` as an unnormalized cap signal.

    Returns the raw powered market capitalizations **without** normalizing
    them to sum to one.  Downstream consumers (cross-sectional regression,
    cross-sectional scalers, basis transforms, etc.) are responsible for
    masking invalid entries and normalizing to their own validity context.

    This follows the scikit-learn `sample_weight` convention: weights
    express relative importance and need not sum to one.

    Parameters
    ----------
    market_caps : array-like of shape (n_observations, n_assets)
        Market capitalizations.  May contain NaN for delisted /
        out-of-universe assets.

    power : float, default=1.0
        Exponent applied to market caps.  Typical choices:

        * 0.0 -- equal weight (caller should handle separately)
        * 0.5 -- sqrt-cap weight
        * 1.0 -- cap weight (default)

    Returns
    -------
    cap_signal : ndarray of shape (n_observations, n_assets)
        :math:`\text{mcap}^{\text{power}}`.  NaN where `market_caps`
        is NaN.
    """
    market_caps = np.asarray(market_caps, dtype=float)
    if power != 1.0:
        market_caps = np.power(market_caps, power)
    return market_caps


def _market_returns(
    asset_returns: ArrayLike,
    weights: ArrayLike,
    estimation_mask: ArrayLike | None = None,
) -> FloatArray:
    """Compute market returns on the estimation universe.

    Parameters
    ----------
    asset_returns : array-like of shape (n_observations, n_assets)
        Asset returns at each observation.

    weights : array-like of shape (n_observations, n_assets)
        Asset weights per observation, typically market capitalizations.

    estimation_mask : array-like of shape (n_observations, n_assets) or None, default=None
        Boolean mask indicating which entries are eligible for market-return
        construction. If `None`, all entries are eligible.

    Returns
    -------
    market_ret : ndarray of shape (n_observations,)
        Market (cap-weighted) return at each date.

    Raises
    ------
    ValueError
        If inputs do not have matching 2D shapes.

    ValueError
        If no eligible asset has finite returns and finite positive total
        weight at any observation.
    """
    asset_returns = np.asarray(asset_returns, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if asset_returns.ndim != 2:
        raise ValueError(
            "asset_returns must be a 2D array of shape (n_observations, n_assets)."
        )
    if weights.shape != asset_returns.shape:
        raise ValueError(
            "weights must have the same shape as asset_returns; "
            f"got weights.shape={weights.shape} and "
            f"asset_returns.shape={asset_returns.shape}."
        )

    if estimation_mask is None:
        estimation_mask = np.ones(asset_returns.shape, dtype=bool)
    else:
        estimation_mask = np.asarray(estimation_mask, dtype=bool)
        if estimation_mask.shape != asset_returns.shape:
            raise ValueError(
                "estimation_mask must have the same shape as asset_returns; "
                f"got estimation_mask.shape={estimation_mask.shape} and "
                f"asset_returns.shape={asset_returns.shape}."
            )

    valid = estimation_mask & np.isfinite(asset_returns) & np.isfinite(weights)
    weights = np.where(valid, weights, 0.0)
    asset_returns = np.where(valid, asset_returns, 0.0)

    w_sum = weights.sum(axis=1, keepdims=True)
    valid_rows = w_sum[:, 0] > 0
    if not np.all(valid_rows):
        bad_obs = int(np.where(~valid_rows)[0][0])
        raise ValueError(
            "Market return is undefined because no estimable asset has finite "
            f"returns and finite positive total weight at observation index {bad_obs}."
        )

    norm_w = np.divide(
        weights, w_sum, out=np.zeros_like(weights, dtype=float), where=w_sum > 0
    )
    return np.sum(norm_w * asset_returns, axis=1)


def _forward_mean_return(X: ArrayLike, horizon: int = 5, lag: int = 1) -> FloatArray:
    r"""Compute a forward H-period mean return target.

    Computes the mean of the next H returns for each observation. This is used
    as the regression target in alpha estimation models.

    Under the as-of indexing convention, the target for observation :math:`t` is:

    .. math::

        y_t = \frac{1}{H}\sum_{s=\ell}^{\ell + H - 1} X_{t+s}

    where :math:`\ell` is the signal lag. The default `lag=1` means scores at
    :math:`t` predict returns starting at :math:`t{+}1`. The last
    `lag + horizon - 1` rows are NaN due to incomplete forward windows.

    Parameters
    ----------
    horizon : int, default=5
        Number of forward periods to average. Must be >= 1.

        * `horizon=1`: Returns next-period values.
        * `horizon>1`: Returns mean of next H periods.

    lag : int, default=1
        Number of periods between the score date and the first return in the target
        window. Must be >= 1 to respect the as-of convention.

    Returns
    -------
    y : ndarray of shape (T, N)
        Forward mean returns. Last `lag + horizon - 1` rows are NaN.

    Notes
    -----
    The computation uses an O(n) cumsum algorithm and handles NaN values by
    excluding them from both sum and count (i.e., `nanmean` semantics).
    """
    horizon = int(horizon)
    lag = int(lag)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, N), got {X.shape}")

    n_observations, n_assets = X.shape

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")

    if n_observations == 0:
        return np.empty((0, n_assets), dtype=np.float64)

    # NaN-safe sums: replace NaN with 0, and count valid observations
    X0 = np.nan_to_num(X, nan=0.0)
    V0 = np.isfinite(X).astype(np.float64)

    target_gap = lag + horizon - 1

    # Pad target_gap rows at the end so incomplete forward windows are treated as missing.

    # (n_obs + target_gap, n_assets)
    X0 = np.vstack([X0, np.zeros((target_gap, n_assets))])
    V0 = np.vstack([V0, np.zeros((target_gap, n_assets))])

    # Prefix a zero row so csum has length (n_obs + target_gap + 1, n_assets)
    csum = np.vstack([np.zeros((1, n_assets)), np.cumsum(X0, axis=0)])
    ccnt = np.vstack([np.zeros((1, n_assets)), np.cumsum(V0, axis=0)])

    # For each t, window is [t + lag, t + lag + horizon) in the padded array.
    start = np.arange(n_observations) + lag
    end = np.arange(n_observations) + lag + horizon

    # (n_obs, n_assets)
    sum_fwd = csum[end] - csum[start]
    cnt_fwd = ccnt[end] - ccnt[start]

    y = safe_divide(sum_fwd, cnt_fwd, fill_value=np.nan)
    y[-target_gap:] = np.nan
    return y


def _update_buffer(buffer: AnyArray, values: AnyArray, lag: int) -> None:
    """Update the lag buffer in-place with the last `lag` rows of `values`.

    All operations copy data into the pre-allocated `buffer`.  No view into `values` is
    retained, so the caller's input array can be freed.

    Parameters
    ----------
    buffer : ndarray of shape (lag, n_assets)
        Pre-allocated buffer to update.

    values : ndarray of shape (n_observations, n_assets)
        New observations (may be a view into a larger array).

    lag : int
        Buffer size (number of lagged rows to retain).
    """
    n_observations = values.shape[0]
    if n_observations == 0:
        return
    if n_observations >= lag:
        np.copyto(buffer, values[-lag:])
    else:
        buffer[:-n_observations] = buffer[n_observations:]
        buffer[-n_observations:] = values
