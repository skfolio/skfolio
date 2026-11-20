"""Exponentially weighted momentum descriptor."""

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
    _validate_non_negative_integer,
    _validate_positive_integer,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "momentum_"


class EWMomentum(BaseDescriptor):
    r"""Exponentially weighted momentum descriptor.

    Computes an EWMA of log returns with an optional skip period to exclude the most
    recent observations.

    The skip period separates medium-term momentum from short-term reversal. The classic
    "12-1" momentum signal uses a skip of approximately one month [1]_.

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        x(t)
            &= \log(1 + r(t)) \\[0.75em]
        S(t)
            &= \lambda \cdot S(t-1)
               + (1 - \lambda) \cdot x(t - \text{skip}) \\[0.75em]
        \text{momentum}(t)
            &=
            \begin{cases}
            \exp(S(t)) - 1 & \text{if } \texttt{exponentiate=True} \\
            S(t) & \text{otherwise}
            \end{cases}
        \end{aligned}
        \]

    where :math:`\lambda = \exp(-\ln(2) / \text{half\_life})` is the EWMA decay factor
    and :math:`S(0) = 0`. At observation :math:`t`, the EWMA adds the log return from
    :math:`t - \text{skip}`. Only non-missing returns count toward the half-life. An
    EWMA with this half-life is comparable to a fixed window of about
    :math:`2 \cdot \text{half\_life} / \ln 2` observations, excluding the most recent
    :math:`\text{skip}` observations.

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
        Number of most recent observations excluded from the EWMA, including those with
        missing returns. The default assumes daily data and skips about one month to
        separate medium-term momentum from short-term reversal. Set to `0` to include
        the current return.

    min_periods : int, optional
        Minimum number of non-missing returns included in the EWMA for each asset, after
        excluding the most recent `skip` observations. Until an asset reaches this
        count, its output is NaN. The count restarts when the asset becomes active
        again. If `None`, defaults to :math:`\lceil\text{half\_life}\rceil`, with a
        minimum of 1.

    exponentiate : bool, default=False
        If True, output is :math:`\exp(S(t)) - 1` (return units). If False, output is
        :math:`S(t)` (EWMA of log returns). Cross-sectional ranking is unchanged.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    momentum_ : ndarray of shape (n_assets,)
        Last exponentially weighted momentum value for each asset.

    decay_ : float
        Exponential decay factor computed from `half_life`.

    min_periods_ : int
        Minimum number of non-missing returns required per asset, resolved from
        `min_periods` and `half_life` during fitting.

    Notes
    -----
    The EWMA starts at zero and is not bias-corrected. After :math:`n` returns have
    contributed, :math:`S(t)` equals their normalized exponentially weighted average
    multiplied by :math:`1 - \lambda^n`. This factor is 50% after one half-life and 75%
    after two.

    This scaling provides useful shrinkage toward zero for momentum signals based on
    short, potentially noisy return histories. It applies at the start of fitting, for
    late listings and after an asset becomes active again. In risk factor models, it
    can improve risk forecasts by reducing the influence of estimation noise. Bias
    correction would remove this shrinkage.

    NaNs are allowed as missing observations. Non-missing `returns` values must be
    finite and greater than `-1`, so :math:`\log(1 + r)` is finite. If the return at
    :math:`t - \text{skip}` is missing, the EWMA and observation count remain unchanged,
    without decay. When `skip > 0`, a missing current return does not prevent an update
    using the earlier return.

    Setting `X.active_mask=False` clears the asset's history and sets its output to NaN.
    After the asset becomes active again, only returns from that point onward are used.
    The most recent `skip` observations remain excluded.

    See Also
    --------
    RollingMomentum : Fixed-window (equal-weighted) momentum.

    References
    ----------
    .. [1] "Returns to buying winners and selling losers: Implications for stock market
        efficiency" The Journal of Finance. Jegadeesh, N., & Titman, S. (1993).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EWMomentum
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> # 12-1 momentum with daily data (default)
    >>> descriptor = EWMomentum()
    >>> momentum = descriptor.fit_transform(X)
    >>>
    >>> # Short-term momentum (no skip)
    >>> descriptor = EWMomentum(half_life=10, skip=0)
    >>> short_term_momentum = descriptor.fit_transform(X)
    >>>
    >>> # Output in return units
    >>> descriptor = EWMomentum(exponentiate=True)
    >>> momentum_returns = descriptor.fit_transform(X)
    """

    momentum_: FloatArray

    def __init__(
        self,
        half_life: float = 87.0,
        skip: int = 21,
        min_periods: int | None = None,
        exponentiate: bool = False,
    ) -> None:
        self.half_life = half_life
        self.skip = skip
        self.min_periods = min_periods
        self.exponentiate = exponentiate

    def fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
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

    def partial_fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
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

        # Compute log returns once for all observations.
        log_returns = np.log1p(returns)

        result = np.empty((n_observations, n_assets), dtype=float)

        for t in range(n_observations):
            active_mask = X.active_mask[t]
            inactive_mask = ~active_mask
            self._mean.reset(inactive_mask=inactive_mask)

            if self.skip > 0:
                # Clear pending returns only when an asset becomes inactive.
                newly_inactive = self._previous_active_mask & inactive_mask
                if np.any(newly_inactive):
                    self._skip_buffer[:, newly_inactive] = np.nan
                self._previous_active_mask[:] = active_mask

                # Read returns from t - skip before storing the current returns.
                idx = self._skip_write_idx
                delayed_log_returns = self._skip_buffer[idx].copy()
                self._skip_buffer[idx] = np.where(active_mask, log_returns[t], np.nan)
                self._skip_write_idx = (idx + 1) % self.skip
            else:
                delayed_log_returns = log_returns[t]

            self._mean.update(
                values=delayed_log_returns,
                valid_mask=active_mask & np.isfinite(delayed_log_returns),
            )
            # Retain zero initialization's shrinkage for assets with short histories.
            result[t] = self._mean.mean(
                min_periods=self.min_periods_, bias_correction=False
            )

        if self.exponentiate:
            np.expm1(result, out=result)

        self.momentum_ = result[-1].copy() if n_observations > 1 else result[-1]
        return result

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        _validate_positive_real(self.half_life, "half_life")
        _validate_non_negative_integer(self.skip, "skip")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")

    def _initialize(self) -> None:
        """Initialize EWMA state and storage for returns excluded by skip."""
        n_assets = self.n_assets_

        self.decay_ = half_life_to_decay_factor(self.half_life)

        # Minimum non-missing returns included in the EWMA before output.
        if self.min_periods is None:
            self.min_periods_ = max(1, int(np.ceil(self.half_life)))
        else:
            self.min_periods_ = int(self.min_periods)

        self._mean = _EWMean(n_assets=n_assets, decay=self.decay_)

        # Store the most recent skip observations until they can enter the EWMA.
        if self.skip > 0:
            self._skip_buffer = np.full((self.skip, n_assets), np.nan, dtype=float)
            self._skip_write_idx = 0
            self._previous_active_mask = np.zeros(n_assets, dtype=bool)
