"""Base Exponentially weighted volatility descriptors."""

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

_FITTED_ATTR = "volatility_"


class _BaseEWVolatility(BaseDescriptor):
    r"""Private base for EWMA volatility descriptors.

    Computes volatility from an exponentially weighted sum of squared returns, with a
    per-asset correction for the weights missing at the start of each asset history:

    .. math::

        S_i(t) = \lambda \cdot S_i(t-1)
            + (1 - \lambda) \cdot f(r_i(t))^2

        \text{output}_i(t) =
            \sqrt{\frac{S_i(t)}{1 - \lambda^{n_i(t)}}}

    where :math:`\lambda = \exp(-\ln(2)/\text{half\_life})` and :math:`n_i(t)` is the
    number of valid observations for asset :math:`i`.

    When `min_acceptable_return` is `None`, :math:`f(r) = r`. When set to a float,
    :math:`f(r) = \min(r - \text{mar}, 0)` (downside volatility).

    NaNs are allowed as missing observations. Non-missing `returns` values must be
    finite. The EWMA state is updated only for active assets with valid returns.
    Missing active returns freeze state, while inactive observations reset state and
    output NaN. The zero-initialized variance accumulator is bias-corrected at output
    time using each asset's valid observation count.
    """

    def __init__(
        self,
        half_life: float = 40.0,
        min_acceptable_return: float | None = None,
        min_periods: int | None = None,
    ):
        self.half_life = half_life
        self.min_acceptable_return = min_acceptable_return
        self.min_periods = min_periods

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute exponentially weighted return volatility.

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
        volatility : ndarray of shape (n_observations, n_assets)
            EWMA volatility (or downside volatility) for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update EWMA state and return volatility for this batch.

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
        volatility : ndarray of shape (n_observations, n_assets)
            Total (or downside) return volatility at each observation.
            Outputs are NaN until each asset reaches `min_periods` valid returns.
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

        result = np.full((n_observations, n_assets), np.nan, dtype=float)

        for t in range(n_observations):
            returns_t = returns[t]
            active_assets = X.active_mask[t]
            valid_returns = active_assets & np.isfinite(returns_t)

            newly_inactive = self._is_active & ~active_assets
            if np.any(newly_inactive):
                self._var[newly_inactive] = 0.0
                self._n_valid_assets[newly_inactive] = 0
            self._is_active[:] = active_assets

            if self._downside:
                contribution = np.minimum(returns_t - self.min_acceptable_return, 0.0)
            else:
                contribution = returns_t

            self._n_valid_assets[valid_returns] += 1
            self._var[valid_returns] = (
                self._decay * self._var[valid_returns]
                + (1 - self._decay) * contribution[valid_returns] ** 2
            )

            ready = self._n_valid_assets >= self._min_periods
            result_t = result[t]

            if np.any(ready):
                weight_sum = 1.0 - self._decay ** self._n_valid_assets[ready]
                result_t[ready] = np.sqrt(self._var[ready] / weight_sum)

        self.volatility_ = result[-1].copy() if n_observations > 1 else result[-1]

        return result

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        if self.half_life <= 0:
            raise ValueError(f"half_life must be positive, got {self.half_life}")
        if self.min_periods is not None and self.min_periods < 1:
            raise ValueError(f"min_periods must be >= 1, got {self.min_periods}")

    def _initialize(self) -> None:
        """Initialize states."""
        n_assets = self.n_assets_

        self._decay = half_life_to_decay_factor(self.half_life)
        self._downside = self.min_acceptable_return is not None

        if self.min_periods is None:
            self._min_periods = max(1, int(np.ceil(self.half_life)))
        else:
            self._min_periods = int(self.min_periods)

        self._var = np.zeros(n_assets, dtype=float)
        self._n_valid_assets = np.zeros(n_assets, dtype=int)
        self._is_active = np.ones(n_assets, dtype=bool)
