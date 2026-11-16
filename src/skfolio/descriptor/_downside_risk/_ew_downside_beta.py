"""Exponentially weighted downside beta descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.descriptor._utils import _warn_deprecated_eps
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
    co-moments. Unlike standard beta, which treats up-moves and down-moves
    symmetrically, downside beta captures how much an asset tends to drop when the
    market drops [1]_ [2]_.

    The lower partial co-moment formulation is:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        D_i(t)
            &= \min(r_i(t) - \text{mar},\; 0) \\[0.75em]
        D_m(t)
            &= \min(r_m(t) - \text{mar},\; 0) \\[0.75em]
        \beta^{\text{down}}_i(t)
            &= \frac{\text{EWMA}(D_i \cdot D_m)}
                    {\text{EWMA}(D_m^2)}
        \end{aligned}
        \]

    where :math:`\text{mar}` is the minimum acceptable return threshold and the EWMA
    uses decay :math:`\lambda = \exp(-\ln(2) / \text{half\_life})`.

    The EWMA is updated where `X.active_mask=True` and asset and market returns are
    finite. Returns above `mar` add zero downside co-moment for that observation, while
    previous downside co-moments still decay. This avoids freezing the estimator during
    calm periods.

    Parameters
    ----------
    half_life : float, default=60.0
        EWMA half-life in observations. Controls how fast old observations decay. The
        default of 60 trading days (~3 months) balances responsiveness and stability.
        Adjust for other frequencies (e.g. `half_life=12` for weekly data).

    min_acceptable_return : float, default=0.0
        Threshold below which returns are considered "downside". The default of `0.0`
        defines downside as negative returns (losses).

    min_periods : int, optional
        Minimum number of observations required before computing each asset's downside
        beta. Until this count is reached, the asset's output is NaN. If `None`,
        defaults to :math:`\lceil\text{half\_life}\rceil`, with a minimum of 1.

    eps : float, optional
        Deprecated and ignored. It will be removed in version 2.0.
        Supplying a positive value emits a `FutureWarning`.

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
    NaNs are allowed as missing observations. Non-missing `returns` values must be
    finite. Both EWMA accumulators use the same observations for each asset, with
    `X.active_mask=True` and non-missing returns. A missing return leaves the previous
    estimate unchanged. Setting `X.active_mask=False` resets estimates and sets output
    to NaN. Output is also NaN if the market has no returns below
    `min_acceptable_return` among those observations.

    Market returns are the `X["market_cap"]`-weighted average of `X["returns"]` where
    `X.estimation_mask=True` and both fields are finite. A `ValueError` is raised if
    these assets have no positive total market capitalization at any observation.

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
        eps: float | None = None,
    ) -> None:
        self.half_life = half_life
        self.min_acceptable_return = min_acceptable_return
        self.min_periods = min_periods
        self.eps = eps

    def fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
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

    def partial_fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
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
            Downside beta for each observation and asset.
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
        asset_rets = X["returns"]
        market_rets = _market_returns(
            asset_returns=asset_rets,
            weights=X["market_cap"],
            estimation_mask=X.estimation_mask,
        )

        down_beta = np.full((n_observations, n_assets), np.nan)

        for t in range(n_observations):
            active = X.active_mask[t]
            inactive = ~active
            self._var_down_market[inactive] = 0
            self._cov_down[inactive] = 0
            self._n_valid_assets[inactive] = 0

            valid = active & np.isfinite(asset_rets[t])

            # Update the EWMA even when returns are above min_acceptable_return.
            down_market = min(market_rets[t] - self.min_acceptable_return, 0.0)
            down_assets = np.minimum(
                asset_rets[t, valid] - self.min_acceptable_return, 0.0
            )

            # Both sums use the same dates, so their observation weights cancel.
            self._var_down_market[valid] = (
                self._decay * self._var_down_market[valid]
                + (1 - self._decay) * down_market**2
            )
            self._cov_down[valid] = (
                self._decay * self._cov_down[valid]
                + (1 - self._decay) * down_assets * down_market
            )
            self._n_valid_assets[valid] += 1

            # Products of finite returns can overflow and invalidate later updates.
            if not (
                np.isfinite(self._var_down_market).all()
                and np.isfinite(self._cov_down).all()
            ):
                raise ValueError(
                    "Non-finite EW downside moments. Rescale the input returns."
                )

            ready = (self._n_valid_assets >= self._min_periods) & (
                self._var_down_market > 0
            )
            np.divide(
                self._cov_down, self._var_down_market, out=down_beta[t], where=ready
            )

            # Finite moments can still produce an infinite ratio for a tiny denominator.
            down_beta[t, ~np.isfinite(down_beta[t])] = np.nan

        self.downside_beta_ = (
            down_beta[-1].copy() if n_observations > 1 else down_beta[-1]
        )
        return down_beta

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        _validate_positive_real(self.half_life, "half_life")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")
        _warn_deprecated_eps(self.eps)

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
        self._var_down_market = np.zeros(self.n_assets_, dtype=float)
        self._cov_down = np.zeros(self.n_assets_, dtype=float)
        self._n_valid_assets = np.zeros(self.n_assets_, dtype=int)
