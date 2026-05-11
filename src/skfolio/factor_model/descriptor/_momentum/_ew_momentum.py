"""Exponentially weighted momentum descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.factor_model.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.tools import half_life_to_decay_factor
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "momentum_"


class EWMomentum(BaseDescriptor):
    r"""Exponentially weighted momentum descriptor.

    Computes an EWMA of log returns with an optional skip period to exclude the most
    recent observations:

    The skip period separates medium-term momentum from short-term reversal. The classic
    "12-1" momentum signal uses a skip of approximately one month [1]_.

    .. math::

        x(t) = \log(1 + r(t))

        S(t) = \lambda \cdot S(t-1)
             + (1 - \lambda) \cdot x(t - \text{skip})

        \text{momentum}(t) =
        \begin{cases}
            \exp(S(t)) - 1 & \text{if exponentiate} \\
            S(t) & \text{otherwise}
        \end{cases}

    where :math:`\lambda = \exp(-\ln(2) / \text{half\_life})` is the EWMA decay factor.
    At observation :math:`t`, the EWMA input is the log return from :math:`t - \text{skip}`.
    Therefore, :math:`\text{half\_life}` is measured on the delayed input series. An
    EWMA with this half-life is comparable to a fixed window of about
    :math:`2 \cdot \text{half\_life} / \ln 2` delayed observations, with the most recent
    :math:`\text{skip}` observations excluded.

    Parameters
    ----------
    half_life : float, default=87.0
        Controls how fast old returns decay in the EWMA of
        :math:`\log(1 + r(t - \text{skip}))`. The default of 87 approximately matches a
        :class:`RollingMomentum` window of 252 observations (~1 year of daily data). To
        match a different window :math:`W`, use
        :math:`\text{half\_life} \approx W \cdot \ln(2) / 2 \approx 0.35 \, W`. Adjust
        for other frequencies (e.g., `half_life=6` for monthly data).

    skip : int, default=21
        Number of most recent observations to exclude before the EWMA window starts.
        Used to separate medium-term momentum from short-term reversal. The default
        assumes daily data and skips approximately one month. Set to `0` for short-term
        momentum.

    min_periods : int or None, default=None
        Minimum number of valid delayed returns required for each asset. Until an asset
        reaches this count, its output is NaN. This warm-up period avoids exposing early
        EWMA values before the estimate has sufficiently converged from its zero
        initialization. If `None`, defaults to :math:`\lceil\text{half\_life}\rceil`,
        with a minimum of 1.

    exponentiate : bool, default=False
        If True, output is :math:`\exp(S(t)) - 1` (return units). If False, output is
        :math:`S(t)` (EWMA of log returns; log space). Cross-sectional ranking is
        unchanged.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    momentum_ : ndarray of shape (n_assets,)
        Last exponentially weighted momentum value for each asset.

    Notes
    -----
    The EWMA is initialized to zero (no bias correction).

    NaNs are allowed as missing observations. Non-missing `returns` values must be
    finite and greater than `-1`, so :math:`\log(1 + r)` is finite. The EWMA state is
    updated only for finite delayed log returns, and each asset's valid-observation
    count controls when its output starts. The `active_mask` property of the input
    :class:`AssetPanel` distinguishes holidays from delistings.

    See Also
    --------
    RollingMomentum : Fixed-window (equal-weighted) momentum.

    References
    ----------
    .. [1] "Returns to buying winners and selling losers: Implications for stock market
        efficiency" The Journal of Finance. Jegadeesh, N., & Titman, S. (1993).

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import EWMomentum
    >>> # 12-1 momentum with daily data (default)
    >>> descriptor = EWMomentum()
    >>> mom = descriptor.fit_transform(X)
    >>>
    >>> # Short-term momentum (no skip)
    >>> descriptor = EWMomentum(half_life=10, skip=0)
    >>> stm = descriptor.fit_transform(X)
    >>>
    >>> # Monthly data
    >>> descriptor = EWMomentum(half_life=6, skip=1)
    >>> mom = descriptor.fit_transform(X)
    >>>
    >>> # Log-space output
    >>> descriptor = EWMomentum(exponentiate=False)
    >>> mom_log = descriptor.fit_transform(X)
    """

    momentum_: FloatArray

    def __init__(
        self,
        half_life: float = 87.0,
        skip: int = 21,
        min_periods: int | None = None,
        exponentiate: bool = False,
    ):
        self.half_life = half_life
        self.skip = skip
        self.min_periods = min_periods
        self.exponentiate = exponentiate

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute exponentially weighted momentum from returns.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        momentum : ndarray of shape (n_observations, n_assets)
            EWMA momentum signal for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute exponentially weighted momentum from returns.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        momentum : ndarray of shape (n_observations, n_assets)
            EWMA momentum signal for each observation and asset.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=["returns"],
            finite_or_nan=["returns"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets

        returns = X["returns"]

        non_missing = ~np.isnan(returns)
        if np.any(returns[non_missing] <= -1):
            raise ValueError(
                'Field "returns" contains values less than or equal to -1. '
                "EWMomentum requires returns greater than -1 because it uses "
                "log returns."
            )

        # Pre-compute finite log returns for the batch.
        log_returns = np.log1p(returns)

        result = np.empty((n_observations, n_assets), dtype=float)

        for t in range(n_observations):
            # Delay buffer: read oldest then overwrite with newest
            if self.skip > 0:
                idx = self._skip_write_idx
                delayed_log_ret = self._skip_buffer[idx].copy()
                self._skip_buffer[idx] = log_returns[t]
                self._skip_write_idx = (idx + 1) % self.skip
            else:
                delayed_log_ret = log_returns[t]

            self._n_seen += 1

            # Only update EWMA once skip buffer has been filled.
            if self._n_seen > self.skip:
                valid = np.isfinite(delayed_log_ret)
                if np.any(valid):
                    self._n_valid[valid] += 1
                    self._ewma[valid] = (
                        self.decay_ * self._ewma[valid]
                        + (1 - self.decay_) * delayed_log_ret[valid]
                    )

            if self.exponentiate:
                momentum = np.expm1(self._ewma)
            else:
                momentum = self._ewma

            ready = self._n_valid >= self.min_periods_
            result[t] = np.where(ready, momentum, np.nan)

        # Mask for inactive assets.
        result = np.where(X.active_mask, result, np.nan)

        self.momentum_ = result[-1].copy() if n_observations > 1 else result[-1]
        return result

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        if self.half_life <= 0:
            raise ValueError(f"half_life must be positive, got {self.half_life}")
        if self.skip < 0:
            raise ValueError(f"skip must be non-negative, got {self.skip}")
        if self.min_periods is not None and self.min_periods < 1:
            raise ValueError(f"min_periods must be >= 1, got {self.min_periods}")

    def _initialize(self) -> None:
        """Initialize EWMA state and delay buffer."""
        n_assets = self.n_assets_

        self.decay_ = half_life_to_decay_factor(self.half_life)

        # Minimum valid delayed returns before output.
        if self.min_periods is None:
            self.min_periods_ = max(1, int(np.ceil(self.half_life)))
        else:
            self.min_periods_ = int(self.min_periods)

        # EWMA accumulator (one per asset, initialized to zero)
        self._ewma = np.zeros(n_assets, dtype=float)
        self._n_valid = np.zeros(n_assets, dtype=int)

        # Delay ring buffer for skip (filled with NaN for warm-up)
        if self.skip > 0:
            self._skip_buffer = np.full((self.skip, n_assets), np.nan, dtype=float)
            self._skip_write_idx = 0

        # Counter of total observations processed
        self._n_seen = 0
