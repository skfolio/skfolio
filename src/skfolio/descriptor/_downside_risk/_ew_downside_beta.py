"""Exponentially weighted downside beta descriptor."""

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

_FITTED_ATTR = "downside_beta_"


class EWDownsideBeta(BaseDescriptor):
    r"""Exponentially weighted downside beta descriptor.

    Measures the sensitivity of each asset to market downturns using lower partial
    moment co-moments. Unlike standard beta, which treats up-moves and down-moves
    symmetrically, downside beta captures how much an asset tends to drop when the
    market drops.

    The lower partial moment co-moment formulation is:

    .. math::

        D_i(t) = \min(r_i(t) - \text{mar},\; 0)

        D_m(t) = \min(r_m(t) - \text{mar},\; 0)

        \beta^{\text{down}}_i(t)
          = \frac{\text{EWMA}(D_i \cdot D_m)}
                 {\text{EWMA}(D_m^2)}

    where :math:`\text{mar}` is the minimum acceptable return threshold and the EWMA
    uses decay :math:`\lambda = \exp(-\ln(2) / \text{half_life})`.

    The EWMA is updated at every observation. Returns above `mar` add zero downside
    co-moment for that observation, while previous downside co-moments still decay.
    This avoids freezing the estimator during calm periods, unlike a conditional
    estimator that updates only on down-market days.

    Parameters
    ----------
    half_life : float, default=60.0
        EWMA half-life in observations. Controls how fast old observations decay. The
        default of 60 trading days (~3 months) balances responsiveness and stability.
        Adjust for other frequencies (e.g., `half_life=12` for weekly data).

    min_acceptable_return : float, default=0.0
        Threshold below which returns are considered "downside". The default of `0.0`
        defines downside as negative returns (losses).

    min_periods : int, optional
        Minimum number of market observations and valid asset returns required before
        computing downside betas. Until both counts reach this value, the asset's output
        is NaN. This warm-up period avoids exposing early EWMA values before the
        downside beta estimate has sufficiently converged from its zero initialization.
        If `None`, defaults to :math:`\lceil\text{half_life}\rceil`, with a minimum
        of 1.

    eps : float, default=1e-12
        Small constant for numerical stability in :math:`1 / \text{EWMA}(D_m^2)`.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    downside_beta_ : ndarray of shape (n_assets,)
        Last fitted downside beta value for each asset.

    Notes
    -----
    The EWMA is initialized to zero (no bias correction). Since the initialization bias
    is identical across all assets at each time step, cross-sectional rankings are
    unaffected.

    The market downside variance is updated at every observation. Asset co-moments are
    updated only for assets with valid (non-NaN) returns and each asset's
    valid-observation count controls when its output starts. This avoids emitting
    initialized values for late-listed or sparsely observed assets. The `active_mask`
    property of :class:`AssetPanel` distinguishes holidays from delistings.

    Market returns are computed from the estimation universe (`estimation_mask` of
    :class:`AssetPanel`). If no estimable asset has both finite returns and finite
    `market_cap` at an observation, the market return is undefined and a `ValueError` is
    raised.

    References
    ----------
    .. [1] "Downside risk" The Review of Financial Studies.
       Ang, A., Chen, J., & Xing, Y. (2006).

    .. [2] "Systematic risk in emerging markets: the D-CAPM". Emerging Markets Review.
       Estrada, J. (2002).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EWDownsideBeta
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> # Standard downside beta (losses only)
    >>> descriptor = EWDownsideBeta()
    >>> downside_beta = descriptor.fit_transform(X)
    >>>
    >>> # Custom threshold
    >>> descriptor = EWDownsideBeta(min_acceptable_return=-0.01)
    >>> downside_beta = descriptor.fit_transform(X)
    """

    downside_beta_: FloatArray

    def __init__(
        self,
        half_life: float = 60.0,
        min_acceptable_return: float = 0.0,
        min_periods: int | None = None,
        eps: float = 1e-12,
    ):
        self.half_life = half_life
        self.min_acceptable_return = min_acceptable_return
        self.min_periods = min_periods
        self.eps = eps

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute exponentially weighted downside betas.

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
        downside_beta : ndarray of shape (n_observations, n_assets)
            Downside beta for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update EWMA state and return downside betas for this batch.

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
        downside_beta : ndarray of shape (n_observations, n_assets)
            Downside beta at each observation. Values are NaN until the global market
            state and the asset-specific valid return count both reach `min_periods`.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=["returns", "market_cap"],
            finite_when_active=["market_cap"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets
        asset_rets = X["returns"]
        market_rets = _market_returns(
            asset_returns=asset_rets,
            weights=X["market_cap"],
            estimation_mask=X.estimation_mask,
        )

        down_beta = np.empty((n_observations, n_assets), dtype=float)
        for t in range(n_observations):
            valid = ~np.isnan(asset_rets[t])

            self._t += 1

            # Downside deviations
            down_market = min(market_rets[t] - self.min_acceptable_return, 0.0)
            down_assets = np.minimum(asset_rets[t] - self.min_acceptable_return, 0.0)

            # Update EWMA of market downside variance
            self._var_down_market = (
                self._decay * self._var_down_market
                + (1 - self._decay) * down_market * down_market
            )

            # Update EWMA of co-moment, valid assets only
            self._cov_down[valid] = (
                self._decay * self._cov_down[valid]
                + (1 - self._decay) * down_assets[valid] * down_market
            )
            self._n_valid_assets[valid] += 1

            if self._t >= self._min_periods:
                beta = self._cov_down / (self._var_down_market + self.eps)
                asset_ready = self._n_valid_assets >= self._min_periods
                down_beta[t] = np.where(asset_ready, beta, np.nan)
            else:
                down_beta[t] = np.nan

        # Mask for inactive assets.
        down_beta = np.where(X.active_mask, down_beta, np.nan)

        self.downside_beta_ = (
            down_beta[-1].copy() if n_observations > 1 else down_beta[-1]
        )
        return down_beta

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        _validate_positive_real(self.half_life, "half_life")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")
        _validate_positive_real(self.eps, "eps")

    def _initialize(self) -> None:
        """Initialize EWMA state."""
        # EWMA decay factor
        self._decay = half_life_to_decay_factor(self.half_life)

        # Min periods before output
        if self.min_periods is None:
            self._min_periods = max(1, int(np.ceil(self.half_life)))
        else:
            self._min_periods = int(self.min_periods)

        # EWMA state for LPM co-moments
        self._var_down_market = 0.0
        self._cov_down = np.zeros(self.n_assets_, dtype=float)
        self._n_valid_assets = np.zeros(self.n_assets_, dtype=int)

        # Observation counter
        self._t = 0
