"""Base Exponentially weighted CAPM residual volatility descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.stats import _market_returns
from skfolio.utils.tools import (
    _validate_positive_integer,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "residual_volatility_"


class _BaseEWResidualVolatility(BaseDescriptor):
    r"""Private base for EWMA CAPM residual volatility descriptors.

    Computes volatility from an exponentially weighted sum of squared CAPM residuals,
    with a per-asset correction for the weights missing at the start of each asset
    history:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        \epsilon_i(t)
            &= r_i(t) - \hat\beta_i(t) \cdot r_m(t) \\[0.75em]
        S_{\epsilon,i}(t)
            &= \lambda_v \cdot S_{\epsilon,i}(t-1)
               + (1 - \lambda_v) \cdot f(\epsilon_i(t))^2 \\[0.75em]
        \text{output}_i(t)
            &= \sqrt{\frac{S_{\epsilon,i}(t)}
                         {1 - \lambda_v^{n_i(t)}}}
        \end{aligned}
        \]

    where :math:`\hat\beta_i(t)` is the EWMA beta estimated with decay
    :math:`\lambda_\beta = \exp(-\ln(2)/\text{beta\_half\_life})` and the residual
    variance uses decay :math:`\lambda_v = \exp(-\ln(2)/\text{half\_life})`.

    When `min_acceptable_return` is `None`, :math:`f(\epsilon) = \epsilon` (total
    residual volatility). When set to a float,
    :math:`f(\epsilon) = \min(\epsilon - \text{mar}, 0)` (downside residual volatility).

    NaNs are allowed as missing observations. Non-missing `returns` values must be
    finite. Asset-specific EWMA states are updated only for active assets with valid
    returns. Missing active returns freeze asset-specific state, while inactive
    observations reset asset-specific state and output NaN. The zero-initialized
    residual variance accumulator is bias-corrected at output time using each asset's
    valid observation count.

    Market returns are computed from the estimation universe. If no estimable asset has
    finite returns and finite positive total `market_cap` for a date, the market return
    is undefined and a `ValueError` is raised.
    """

    def __init__(
        self,
        half_life: float = 40.0,
        beta_half_life: float = 60.0,
        min_acceptable_return: float | None = None,
        min_periods: int | None = None,
        eps: float = 1e-12,
    ):
        self.half_life = half_life
        self.beta_half_life = beta_half_life
        self.min_acceptable_return = min_acceptable_return
        self.min_periods = min_periods
        self.eps = eps

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute exponentially weighted CAPM residual volatility.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns` and `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        residual_volatility : ndarray of shape (n_observations, n_assets)
            Residual return volatility for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update EWMA state and return residual volatility for this batch.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns` and `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        residual_volatility : ndarray of shape (n_observations, n_assets)
            Residual return volatility for each observation and asset.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=["returns", "market_cap"],
            finite_or_nan=["returns"],
            finite_when_active=["market_cap"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets

        asset_returns = X["returns"]
        market_caps = X["market_cap"]

        market_returns = _market_returns(
            asset_returns=asset_returns,
            weights=market_caps,
            estimation_mask=X.estimation_mask,
        )

        result = np.full((n_observations, n_assets), np.nan, dtype=float)

        for t in range(n_observations):
            returns_t = asset_returns[t]
            market_return_t = market_returns[t]
            active_assets = X.active_mask[t]
            valid_returns = active_assets & np.isfinite(returns_t)

            newly_inactive = self._is_active & ~active_assets
            if np.any(newly_inactive):
                self._mu_assets[newly_inactive] = 0.0
                self._cov_assets[newly_inactive] = 0.0
                self._var_residual[newly_inactive] = 0.0
                self._n_valid_assets[newly_inactive] = 0
            self._is_active[:] = active_assets

            # Deviations from lagged means
            market_deviation = market_return_t - self._mu_market
            valid_asset_returns = returns_t[valid_returns]
            asset_deviations = valid_asset_returns - self._mu_assets[valid_returns]

            # Update EWMA means
            self._mu_market = (
                self._beta_decay * self._mu_market
                + (1 - self._beta_decay) * market_return_t
            )
            self._mu_assets[valid_returns] = (
                self._beta_decay * self._mu_assets[valid_returns]
                + (1 - self._beta_decay) * valid_asset_returns
            )

            # Update EWMA market variance and asset-market covariances
            self._var_market = (
                self._beta_decay * self._var_market
                + (1 - self._beta_decay) * market_deviation * market_deviation
            )
            self._cov_assets[valid_returns] = (
                self._beta_decay * self._cov_assets[valid_returns]
                + (1 - self._beta_decay) * asset_deviations * market_deviation
            )

            # Compute residuals using the current beta estimate
            beta = self._cov_assets / (self._var_market + self.eps)
            residual = returns_t - beta * market_return_t

            if self._downside:
                contribution = np.minimum(residual - self.min_acceptable_return, 0.0)
            else:
                contribution = residual

            self._n_valid_assets[valid_returns] += 1
            self._var_residual[valid_returns] = (
                self._vol_decay * self._var_residual[valid_returns]
                + (1 - self._vol_decay) * contribution[valid_returns] ** 2
            )

            ready = self._n_valid_assets >= self._min_periods
            result_t = result[t]
            if np.any(ready):
                weight_sum = 1.0 - self._vol_decay ** self._n_valid_assets[ready]
                result_t[ready] = np.sqrt(self._var_residual[ready] / weight_sum)

        self.residual_volatility_ = (
            result[-1].copy() if n_observations > 1 else result[-1]
        )

        return result

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        _validate_positive_real(self.half_life, "half_life")
        _validate_positive_real(self.beta_half_life, "beta_half_life")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")
        _validate_positive_real(self.eps, "eps")

    def _initialize(self) -> None:
        """Initialize states."""
        n_assets = self.n_assets_

        # Separate decay factors for beta and volatility
        self._beta_decay = half_life_to_decay_factor(self.beta_half_life)
        self._vol_decay = half_life_to_decay_factor(self.half_life)

        # Downside mode flag avoids repeated None checks in the hot loop
        self._downside = self.min_acceptable_return is not None

        if self.min_periods is None:
            self._min_periods = max(
                1, int(np.ceil(max(self.half_life, self.beta_half_life)))
            )
        else:
            self._min_periods = int(self.min_periods)

        self._mu_market = 0.0
        self._var_market = 0.0
        self._mu_assets = np.zeros(n_assets, dtype=float)
        self._cov_assets = np.zeros(n_assets, dtype=float)

        self._var_residual = np.zeros(n_assets, dtype=float)
        self._n_valid_assets = np.zeros(n_assets, dtype=int)
        self._is_active = np.ones(n_assets, dtype=bool)
