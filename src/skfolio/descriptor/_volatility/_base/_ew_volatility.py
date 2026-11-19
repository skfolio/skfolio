"""Base Exponentially weighted volatility descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.descriptor._utils import _EWMean
from skfolio.typing import FloatArray
from skfolio.utils.tools import (
    _validate_positive_integer,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "volatility_"


class _BaseEWVolatility(BaseDescriptor):
    r"""Private base for EWMA volatility descriptors.

    Computes volatility from an exponentially weighted sum of squared returns, with a
    per-asset correction for the weights missing at the start of each asset history:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        S_i(t)
            &= \lambda \cdot S_i(t-1)
               + (1 - \lambda) \cdot f(r_i(t))^2 \\[0.75em]
        \text{output}_i(t)
            &= \sqrt{\frac{S_i(t)}{1 - \lambda^{n_i(t)}}}
        \end{aligned}
        \]

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
    ) -> None:
        self.half_life = half_life
        self.min_acceptable_return = min_acceptable_return
        self.min_periods = min_periods

    def fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
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

    def partial_fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
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
            EWMA volatility (or downside volatility) for each observation and asset.
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

        result = np.empty((n_observations, n_assets), dtype=float)

        for t in range(n_observations):
            returns_t = returns[t]
            active_mask = X.active_mask[t]
            self._variance.reset(inactive_mask=~active_mask)

            if self.min_acceptable_return is not None:
                contribution = np.minimum(returns_t - self.min_acceptable_return, 0.0)
            else:
                contribution = returns_t

            self._variance.update(
                values=contribution**2,
                valid_mask=active_mask & np.isfinite(returns_t),
            )
            result[t] = np.sqrt(self._variance.mean(min_periods=self._min_periods))

        self.volatility_ = result[-1].copy() if n_observations > 1 else result[-1]

        return result

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        _validate_positive_real(self.half_life, "half_life")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")

    def _initialize(self) -> None:
        """Initialize states."""
        if self.min_periods is None:
            self._min_periods = max(1, int(np.ceil(self.half_life)))
        else:
            self._min_periods = int(self.min_periods)

        self._variance = _EWMean(
            n_assets=self.n_assets_, decay=half_life_to_decay_factor(self.half_life)
        )
