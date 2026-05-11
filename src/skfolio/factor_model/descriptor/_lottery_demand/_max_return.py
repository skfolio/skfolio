"""Maximum return descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from skfolio.containers import AssetPanel
from skfolio.factor_model._utils import _update_buffer
from skfolio.factor_model.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "max_return_"


class MaxReturn(BaseDescriptor):
    r"""Maximum return over a trailing window.

    Computes the maximum return over the last `window` observations:

    .. math::

        \text{MAX}(t) = \max_{k \in [t-w+1,\, t]} \; r_k

    where :math:`w` is the `window` size. High values identify assets with recent
    extreme positive returns, capturing lottery-like payoff that may attract speculative
    demand.

    The output is NaN until an asset has a full trailing window of active observations.
    NaN returns are allowed as missing observations and ignored when computing the
    maximum. If all returns in an active trailing window are missing, the output is NaN.
    Non-missing `returns` values must be finite.

    Stocks with high MAX are found to earn lower subsequent returns, consistent with
    investor overpricing of lottery-like payoffs [1]_.

    Parameters
    ----------
    window : int, default=21
        Number of trailing observations for the rolling maximum. Must be
        greater than 1. The default of 21 corresponds to approximately one
        trading month, matching the original definition in [1]_.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    max_return_ : ndarray of shape (n_assets,)
        Last maximum return value for each asset.

    References
    ----------
    .. [1] "Maxing out: stocks as lotteries and the cross-section of expected returns"
        Journal of Financial Economics. Bali, Cakici & Whitelaw (2011).

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import MaxReturn
    >>> # 1-month rolling max (default)
    >>> descriptor = MaxReturn()
    >>> max_ret = descriptor.fit_transform(X)
    >>>
    >>> # 1-week rolling max
    >>> descriptor = MaxReturn(window=5)
    >>> max_ret_5d = descriptor.fit_transform(X)
    """

    max_return_: FloatArray

    def __init__(self, window: int = 21):
        self.window = window

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute rolling maximum returns over the configured window.

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
        max_return : ndarray of shape (n_observations, n_assets)
            Rolling maximum return for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update state and return rolling max return for this batch.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `"returns"`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        max_return : ndarray of shape (n_observations, n_assets)
            Rolling maximum return over the trailing window.
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

        returns = X["returns"]
        n_observations = X.n_observations

        # Replace NaN with -inf so they don't affect the max
        clean_returns = np.where(np.isnan(returns), -np.inf, returns)
        combined = np.concatenate([self._returns_buffer, clean_returns], axis=0)

        windowed = sliding_window_view(combined, self.window, axis=0)

        result = np.max(windowed, axis=-1)

        active_combined = np.concatenate(
            [self._active_mask_buffer, X.active_mask], axis=0
        )
        active_windowed = sliding_window_view(active_combined, self.window, axis=0)

        has_valid_return = np.any(np.isfinite(windowed), axis=-1)
        has_full_active_window = np.all(active_windowed, axis=-1)

        # A full active window handles warm-up, late listings and inactive gaps.
        # A valid return is still required because all-missing windows map to -inf.
        result[~(has_valid_return & has_full_active_window)] = np.nan

        _update_buffer(self._returns_buffer, clean_returns, self.window - 1)
        _update_buffer(self._active_mask_buffer, X.active_mask, self.window - 1)

        # Mask for inactive assets.
        result = np.where(X.active_mask, result, np.nan)

        self.max_return_ = result[-1].copy() if n_observations > 1 else result[-1]
        return result

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)
        if hasattr(self, "_returns_buffer"):
            delattr(self, "_returns_buffer")
        if hasattr(self, "_active_mask_buffer"):
            delattr(self, "_active_mask_buffer")

    def _validate_params(self) -> None:
        """Validate parameters."""
        if self.window <= 1:
            raise ValueError(f"window must be > 1, got {self.window}")

    def _initialize(self) -> None:
        """Initialize states."""
        self._returns_buffer = np.full((self.window - 1, self.n_assets_), -np.inf)
        self._active_mask_buffer = np.zeros(
            (self.window - 1, self.n_assets_), dtype=bool
        )
