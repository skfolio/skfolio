"""Base Exponentially weighted CAPM residual volatility descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.descriptor._utils import (
    _EWRegression,
    _warn_deprecated_eps,
)
from skfolio.typing import FloatArray
from skfolio.utils.stats import _market_returns
from skfolio.utils.tools import (
    _validate_positive_integer,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "residual_volatility_"
# Too few observations can produce an extreme beta and inflate residual volatility.
_MIN_BETA_OBSERVATIONS = 10


class _BaseEWResidualVolatility(BaseDescriptor):
    r"""Private base for EWMA CAPM residual volatility descriptors.

    Computes volatility from an exponentially weighted sum of squared CAPM residuals,
    with a per-asset correction for the weights missing at the start of each asset
    history:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        \epsilon_i(t)             &= r_i(t) - \hat\beta_i(t) \cdot r_m(t) \\[0.75em]
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
    finite. Each asset's beta and residual variance are updated only where
    `X.active_mask=True` and returns are non-missing. Missing returns leave previous
    estimates unchanged. Setting `X.active_mask=False` resets the estimates and sets
    output to NaN.

    Residuals are included from the tenth observation used to estimate beta, provided
    beta can be computed. The zero-initialized residual variance accumulator is
    bias-corrected at output time using each asset's residual count, :math:`n_i(t)`.

    Market returns are the `X["market_cap"]`-weighted average of `X["returns"]` where
    `X.estimation_mask=True` and both fields are finite. A `ValueError` is raised if
    these assets have no positive total market capitalization at any observation.
    """

    def __init__(
        self,
        half_life: float = 40.0,
        beta_half_life: float = 60.0,
        min_acceptable_return: float | None = None,
        min_periods: int | None = None,
        eps: float | None = None,
    ) -> None:
        self.half_life = half_life
        self.beta_half_life = beta_half_life
        self.min_acceptable_return = min_acceptable_return
        self.min_periods = min_periods
        self.eps = eps

    def fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
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

    def partial_fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
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
            active_mask = X.active_mask[t]
            inactive_mask = ~active_mask
            self._regression.reset(inactive_mask)
            self._var_residual[inactive_mask] = 0
            self._n_residuals[inactive_mask] = 0

            valid_mask = active_mask & np.isfinite(asset_returns[t])
            returns = np.vstack(
                (np.full(n_assets, market_returns[t]), asset_returns[t])
            )
            self._regression.update(returns=returns, valid_mask=valid_mask)
            beta = self._regression.beta(min_periods=_MIN_BETA_OBSERVATIONS)
            has_beta = np.isfinite(beta)
            valid_residuals = valid_mask & has_beta

            # Use the updated beta and retain the intercept in the residual.
            residual = (
                asset_returns[t, valid_residuals]
                - beta[valid_residuals] * market_returns[t]
            )
            if self.min_acceptable_return is not None:
                residual = np.minimum(residual - self.min_acceptable_return, 0.0)

            self._var_residual[valid_residuals] = (
                self._vol_decay * self._var_residual[valid_residuals]
                + (1 - self._vol_decay) * residual**2
            )
            self._n_residuals[valid_residuals] += 1

            # Squaring finite residuals can overflow and invalidate later volatility.
            if not np.isfinite(self._var_residual).all():
                raise ValueError(
                    "Non-finite EW residual moments. Rescale the input returns."
                )

            # An unresolved beta suppresses output but preserves residual history.
            ready = has_beta & (self._n_residuals >= self._min_periods)
            result[t, ready] = np.sqrt(
                self._var_residual[ready]
                / (1 - self._vol_decay ** self._n_residuals[ready])
            )

        self.residual_volatility_ = (
            result[-1].copy() if n_observations > 1 else result[-1]
        )

        return result

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        _validate_positive_real(self.half_life, "half_life")
        _validate_positive_real(self.beta_half_life, "beta_half_life")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")
        _warn_deprecated_eps(self.eps)

    def _initialize(self) -> None:
        """Initialize states."""
        n_assets = self.n_assets_

        # Separate decay factors for beta and volatility
        beta_decay = half_life_to_decay_factor(self.beta_half_life)
        self._vol_decay = half_life_to_decay_factor(self.half_life)
        self._regression = _EWRegression(n_assets=n_assets, decay=beta_decay)

        if self.min_periods is None:
            self._min_periods = max(
                1, int(np.ceil(max(self.half_life, self.beta_half_life)))
            )
        else:
            self._min_periods = int(self.min_periods)

        self._var_residual = np.zeros(n_assets)
        self._n_residuals = np.zeros(n_assets, dtype=int)
