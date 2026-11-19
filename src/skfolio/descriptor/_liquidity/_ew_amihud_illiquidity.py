"""Exponentially weighted Amihud illiquidity descriptor."""

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
from skfolio.utils.stats import safe_divide
from skfolio.utils.tools import (
    _validate_positive_integer,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "illiquidity_"


class EWAmihudIlliquidity(BaseDescriptor):
    r"""Exponentially weighted Amihud illiquidity descriptor.

    Computes an EWMA of the per-observation Amihud illiquidity ratio:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        \text{ILLIQ\_raw}(t)
            &= \frac{|r(t)|}
                    {\text{adj\_close}(t) \times \text{adj\_volume}(t)} \\[0.75em]
        S(t)
            &= \lambda \cdot S(t-1)
               + (1 - \lambda) \cdot \text{ILLIQ\_raw}(t) \\[0.75em]
        \text{ILLIQ}(t)
            &= \frac{S(t)}{1 - \lambda^{n(t)}}
        \end{aligned}
        \]

    where :math:`\lambda = \exp(-\ln(2) / \text{half\_life})` is the EWMA decay factor,
    :math:`S(0) = 0` and :math:`n(t)` is the asset's number of valid observations since
    fitting started or the asset became active again. The denominator of the raw ratio
    is the dollar trading volume (traded amount).

    The Amihud illiquidity ratio is a proxy for price impact, defined as the absolute
    return per unit of dollar volume traded. Higher values imply larger price moves for
    a given dollar amount traded, reflecting lower liquidity. Higher illiquidity is
    often associated with higher expected returns, commonly interpreted as an
    illiquidity premium for bearing higher trading costs and exit risk [1]_.

    EWMA smoothing is preferred over a fixed rolling average because the raw ratio is
    very noisy (it can spike when volume is low or returns are large). EWMA dampens
    transient spikes gradually, producing more stable factor exposures.

    Parameters
    ----------
    half_life : float, default=63.0
        EWMA half-life in observations. Controls how fast old illiquidity values decay.
        With daily data, common choices are:

        - `half_life=21`: ~1 month
        - `half_life=63`: ~3 months (default)
        - `half_life=252`: ~1 year

    min_periods : int, optional
        Minimum number of valid illiquidity observations required for each asset. Until
        an asset reaches this count, its output is NaN. This warm-up period avoids
        exposing illiquidity estimates based on too few observations. If `None`,
        defaults to :math:`\lceil\text{half\_life}\rceil`, with a minimum of 1.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    illiquidity_ : ndarray of shape (n_assets,)
        Last EWMA-smoothed Amihud illiquidity value for each asset.

    Notes
    -----
    Observation weights are normalized for each asset so that starting the EWMA at zero
    does not understate illiquidity for assets with short histories.

    Dollar trading volume (`traded_amount`) is computed internally as
    `adj_close * adj_volume`. Both fields must use the same split-adjustment basis.

    NaNs are allowed as missing observations. Non-missing `returns` values must be
    finite. Non-missing `adj_close` values must be finite and strictly positive.
    Non-missing `adj_volume` values must be finite and non-negative.

    Zero `adj_volume` means the stock did not trade, so the per-observation ratio is
    undefined. With `X.active_mask=True`, zero volume or NaN in `returns`, `adj_close`
    or `adj_volume` leaves the estimate unchanged without decay and does not count
    toward `min_periods`. Setting `X.active_mask=False` clears the asset's history and
    sets output to NaN. Estimation restarts when the asset becomes active again.

    References
    ----------
    .. [1] "Illiquidity and stock returns: cross-section and time-series effects"
        Journal of Financial Markets. Amihud, Y. (2002).

    See Also
    --------
    EWShareTurnover : EWMA share turnover (volume-based liquidity).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EWAmihudIlliquidity
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> # 3-month effective window (default)
    >>> descriptor = EWAmihudIlliquidity()
    >>> illiq = descriptor.fit_transform(X)
    >>>
    >>> # 1-month effective window
    >>> descriptor = EWAmihudIlliquidity(half_life=21)
    >>> illiq_1m = descriptor.fit_transform(X)
    """

    illiquidity_: FloatArray

    def __init__(self, half_life: float = 63.0, min_periods: int | None = None) -> None:
        self.half_life = half_life
        self.min_periods = min_periods

    def fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
        """Compute exponentially weighted Amihud illiquidity.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns`, `adj_close`, and `adj_volume`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        illiquidity : ndarray of shape (n_observations, n_assets)
            EWMA-smoothed Amihud illiquidity for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
        """Update state and return smoothed illiquidity for this batch.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `"returns"`, `"adj_close"`, and
            `"adj_volume"`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        illiquidity : ndarray of shape (n_observations, n_assets)
            EWMA-smoothed Amihud illiquidity for each observation and asset.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=["returns", "adj_close", "adj_volume"],
            finite_or_nan=["returns"],
            strictly_positive_or_nan=["adj_close"],
            non_negative_or_nan=["adj_volume"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets

        returns = X["returns"]
        adj_close = X["adj_close"]
        adj_volume = X["adj_volume"]

        # Dollar volume (traded amount)
        traded_amount = adj_close * adj_volume
        raw_illiquidity = safe_divide(np.abs(returns), traded_amount, fill_value=np.nan)

        result = np.empty((n_observations, n_assets), dtype=float)

        for t in range(n_observations):
            active_mask = X.active_mask[t]
            self._mean.reset(inactive_mask=~active_mask)
            self._mean.update(
                values=raw_illiquidity[t],
                valid_mask=active_mask & np.isfinite(raw_illiquidity[t]),
            )
            result[t] = self._mean.mean(min_periods=self._min_periods)

        self.illiquidity_ = result[-1].copy() if n_observations > 1 else result[-1]

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

        self._mean = _EWMean(
            n_assets=self.n_assets_, decay=half_life_to_decay_factor(self.half_life)
        )
