"""Powered market-cap utilities for factor-model weighting."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.typing import AnyArray, ArrayLike, FloatArray
from skfolio.utils.stats import safe_divide


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


def _forward_mean_return(X: ArrayLike, horizon: int = 5) -> FloatArray:
    r"""Compute a forward H-period mean return target.

    Computes the mean of the next H returns for each observation. This is used
    as the regression target in alpha estimation models.

    Under the as-of indexing convention, the target for observation :math:`t` is:

    .. math::

        y_t = \frac{1}{H}\sum_{s=1}^{H} X_{t+s}

    This implies a fixed 1-period lag: scores at :math:`t` predict returns
    starting at :math:`t{+}1`. The last H rows are NaN due to incomplete
    forward windows.

    Parameters
    ----------
    horizon : int, default=5
        Number of forward periods to average. Must be >= 1.

        * `horizon=1`: Returns next-period values.
        * `horizon>1`: Returns mean of next H periods.

    Returns
    -------
    y : ndarray of shape (T, N)
        Forward mean returns. Last H rows are NaN.

    Notes
    -----
    The computation uses an O(n) cumsum algorithm and handles NaN values by
    excluding them from both sum and count (i.e., `nanmean` semantics).
    """
    horizon = int(horizon)
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (T, N), got {X.shape}")

    n_observations, n_assets = X.shape

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    if n_observations == 0:
        return np.empty((0, n_assets), dtype=np.float64)

    # NaN-safe sums: replace NaN with 0, and count valid observations
    X0 = np.nan_to_num(X, nan=0.0)
    V0 = np.isfinite(X).astype(np.float64)

    # Pad horizon rows at the end (so forward windows beyond n_obs are treated as missing)

    # (n_obs + horizon, n_assets)
    X0 = np.vstack([X0, np.zeros((horizon, n_assets))])
    V0 = np.vstack([V0, np.zeros((horizon, n_assets))])

    # Prefix a zero row so csum has length (n_obs + horizon + 1, n_assets)
    csum = np.vstack([np.zeros((1, n_assets)), np.cumsum(X0, axis=0)])
    ccnt = np.vstack([np.zeros((1, n_assets)), np.cumsum(V0, axis=0)])

    # For each t, window is [t + 1, t + horizon + ) in the padded array
    start = np.arange(n_observations) + 1
    end = np.arange(n_observations) + horizon + 1

    # (n_obs, n_assets)
    sum_fwd = csum[end] - csum[start]
    cnt_fwd = ccnt[end] - ccnt[start]

    y = safe_divide(sum_fwd, cnt_fwd, fill_value=np.nan)
    y[-horizon:] = np.nan  # last H rows have incomplete forward horizon
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
