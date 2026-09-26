"""Exponentially weighted share turnover descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.stats import safe_divide
from skfolio.utils.tools import (
    _validate_positive_integer,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "turnover_"


class EWShareTurnover(BaseDescriptor):
    r"""Exponentially weighted share turnover descriptor.

    Computes an EWMA of per-observation share turnover:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        \text{turnover\_raw}(t)
            &= \frac{\text{adj\_volume}(t)}
                    {\text{adj\_shares\_outstanding}(t)} \\[0.75em]
        \text{turnover}(t)
            &= \lambda \cdot \text{turnover}(t-1)
               + (1 - \lambda) \cdot \text{turnover\_raw}(t)
        \end{aligned}
        \]

    where :math:`\lambda = \exp(-\ln(2) / \text{half\_life})` is the EWMA decay factor.

    Share turnover measures trading intensity as the fraction of shares outstanding that
    changes hands over each observation period. Lower turnover indicates weaker trading
    activity and lower liquidity, making trades more likely to incur price impact.
    Low-turnover stocks are often associated with higher expected returns, commonly
    interpreted as an illiquidity premium [1]_.

    EWMA smoothing is preferred over a fixed rolling average because turnover can spike
    around earnings, index rebalances or news events. EWMA dampens these spikes
    gradually, producing more stable factor exposures.

    Parameters
    ----------
    half_life : float, default=21.0
        EWMA half-life in observations. Controls how fast old turnover values decay.
        With daily data, common choices are:

        - `half_life=21`: ~1 month
        - `half_life=63`: ~3 months
        - `half_life=252`: ~1 year

    min_periods : int, optional
        Minimum number of valid turnover observations required for each asset. Until an
        asset reaches this count, its output is NaN. This warm-up period avoids
        exposing early EWMA values before the turnover estimate has sufficiently
        converged from its zero initialization. If `None`, defaults to
        :math:`\lceil\text{half\_life}\rceil`, with a minimum of 1.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    turnover_ : ndarray of shape (n_assets,)
        Last EWMA-smoothed share turnover value for each asset.

    Notes
    -----
    `adj_shares_outstanding` is common shares outstanding. Both `adj_volume` and
    `adj_shares_outstanding` must use the same split-adjustment basis.

    NaNs are allowed as missing observations. Non-missing `adj_volume` values must be
    finite and non-negative. Non-missing `adj_shares_outstanding` values must be finite
    and strictly positive.

    The EWMA state is updated only for valid observations. NaN in `adj_volume` or
    `adj_shares_outstanding` holds the EWMA state and does not increment the
    valid-observation count. Zero `adj_volume` is valid and produces zero turnover.

    The `active_mask` property of the :class:`~skfolio.containers.AssetPanel`
    distinguishes holidays from delistings.

    References
    ----------
    .. [1] "Liquidity and stock returns: an alternative test"
        Journal of Financial Markets. Datar, V. T., Naik, N. Y., & Radcliffe, R. (1998).

    See Also
    --------
    EWAmihudIlliquidity : EWMA price-impact illiquidity measure.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EWShareTurnover
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> # 1-month effective window (default)
    >>> descriptor = EWShareTurnover()
    >>> turnover = descriptor.fit_transform(X)
    >>>
    >>> # 3-month effective window
    >>> descriptor = EWShareTurnover(half_life=63)
    >>> turnover_3m = descriptor.fit_transform(X)
    """

    turnover_: FloatArray

    def __init__(self, half_life: float = 21.0, min_periods: int | None = None):
        self.half_life = half_life
        self.min_periods = min_periods

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute exponentially weighted share turnover.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `adj_volume` and `adj_shares_outstanding`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        turnover : ndarray of shape (n_observations, n_assets)
            EWMA-smoothed share turnover for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update state and return smoothed turnover for this batch.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `"adj_volume"` and
            `"adj_shares_outstanding"`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        turnover : ndarray of shape (n_observations, n_assets)
            EWMA-smoothed share turnover for each observation and asset.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=["adj_volume", "adj_shares_outstanding"],
            non_negative_or_nan=["adj_volume"],
            strictly_positive_or_nan=["adj_shares_outstanding"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets

        adj_volume = X["adj_volume"]
        adj_shares_outstanding = X["adj_shares_outstanding"]

        raw_turnover = safe_divide(
            adj_volume, adj_shares_outstanding, fill_value=np.nan
        )

        result = np.empty((n_observations, n_assets), dtype=float)

        for t in range(n_observations):
            valid = np.isfinite(raw_turnover[t])
            if np.any(valid):
                self._n_valid[valid] += 1
                self._ewma[valid] = (
                    self._decay * self._ewma[valid]
                    + (1 - self._decay) * raw_turnover[t][valid]
                )

            result[t] = np.where(self._n_valid >= self._min_periods, self._ewma, np.nan)

        # Mask for inactive assets
        result = np.where(X.active_mask, result, np.nan)

        self.turnover_ = result[-1].copy() if n_observations > 1 else result[-1]

        return result

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate parameters."""
        _validate_positive_real(self.half_life, "half_life")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")

    def _initialize(self) -> None:
        """Initialize state."""
        n_assets = self.n_assets_

        self._decay = half_life_to_decay_factor(self.half_life)

        if self.min_periods is None:
            self._min_periods = max(1, int(np.ceil(self.half_life)))
        else:
            self._min_periods = int(self.min_periods)

        self._ewma = np.zeros(n_assets, dtype=float)
        self._n_valid = np.zeros(n_assets, dtype=int)
