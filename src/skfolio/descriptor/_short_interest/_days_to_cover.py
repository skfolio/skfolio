"""EWMA-smoothed days-to-cover descriptor."""

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

_FITTED_ATTR = "days_to_cover_"


class DaysToCover(BaseDescriptor):
    r"""Exponentially weighted days-to-cover descriptor.

    Computes the ratio of shares sold short to exponentially weighted average daily
    volume:

    .. math::
        :nowrap:

        \[
        \begin{aligned}
        S(t)
            &= \lambda \cdot S(t-1)
               + (1 - \lambda) \cdot \text{adj\_volume}(t) \\[0.75em]
        \text{EWMA\_volume}(t)
            &= \frac{S(t)}{1 - \lambda^{n(t)}} \\[0.75em]
        \text{days\_to\_cover}(t)
            &= \frac{\text{short\_interest}(t)}{\text{EWMA\_volume}(t)}
        \end{aligned}
        \]

    where :math:`\lambda = \exp(-\ln(2) / \text{half\_life})` is the EWMA decay factor,
    :math:`S(0) = 0` and :math:`n(t)` is the asset's number of valid positive-volume
    observations since fitting started or the asset became active again.

    Days to cover measures how many trading days it would take short sellers to buy back
    their positions at the current trading rate. High values indicate crowded short
    positions relative to liquidity [1]_ [2]_.

    EWMA smoothing is preferred over a fixed rolling average because daily volume can
    spike around earnings, index rebalances or news events. EWMA dampens these spikes
    gradually, producing more stable factor exposures.

    Parameters
    ----------
    half_life : float, default=21.0
        EWMA half-life in observations for volume smoothing. With daily data, common
        choices are:

        - `half_life=21`: ~1 month
        - `half_life=63`: ~3 months
        - `half_life=252`: ~1 year

    min_periods : int, optional
        Minimum number of valid positive-volume observations required for each asset.
        Until an asset reaches this count, its output is NaN. This warm-up period avoids
        exposing volume estimates based on too few observations. If `None`, defaults to
        :math:`\lceil\text{half\_life}\rceil`, with a minimum of 1.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    days_to_cover_ : ndarray of shape (n_assets,)
        Last days-to-cover value for each asset.

    Notes
    -----
    Observation weights are normalized for each asset so that starting the volume EWMA
    at zero does not understate average volume and overstate days to cover for assets
    with short histories.

    `short_interest` is the number of shares held short. Non-missing values must be
    finite and non-negative.

    `adj_volume` is split-adjusted trading volume. Non-missing values must be finite and
    non-negative.

    With `X.active_mask=True`, NaN or zero `adj_volume` leaves the average volume
    unchanged without decay and does not count toward `min_periods`. Missing
    `short_interest` propagates to the output but does not prevent volume updates.
    Setting `X.active_mask=False` clears the asset's history and sets output to NaN.
    Estimation restarts when the asset becomes active again.

    References
    ----------
    .. [1] "An investigation of the informational role of short interest in the Nasdaq
        market" The Journal of Finance.
        Desai, H., Ramesh, K., Thiagarajan, S. R., & Balachandran, B. V. (2002).

    .. [2] "Short interest, institutional ownership, and stock returns"
        Journal of Financial Economics.
        Asquith, P., Pathak, P. A., & Ritter, J. R. (2005).

    See Also
    --------
    ShortInterest : Short interest as fraction of shares outstanding.
    EWShareTurnover : EWMA share turnover (volume / shares outstanding).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import DaysToCover
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> # 1-month EWMA volume smoothing (default)
    >>> descriptor = DaysToCover()
    >>> days_to_cover = descriptor.fit_transform(X)
    >>>
    >>> # 3-month EWMA volume smoothing
    >>> descriptor = DaysToCover(half_life=63)
    >>> days_to_cover = descriptor.fit_transform(X)
    """

    days_to_cover_: FloatArray

    def __init__(self, half_life: float = 21.0, min_periods: int | None = None) -> None:
        self.half_life = half_life
        self.min_periods = min_periods

    def fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
        """Compute exponentially weighted days to cover.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `short_interest` and `adj_volume`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        days_to_cover : ndarray of shape (n_observations, n_assets)
            Short interest divided by EWMA-smoothed daily volume for each observation
            and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(
        self, X: AssetPanel, y: None = None, **fit_params: Any
    ) -> FloatArray:
        """Update state and return days to cover for this batch.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `short_interest` and `adj_volume`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        days_to_cover : ndarray of shape (n_observations, n_assets)
            Short interest divided by EWMA-smoothed daily volume for each observation
            and asset.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=["short_interest", "adj_volume"],
            non_negative_or_nan=["short_interest", "adj_volume"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        short_interest = X["short_interest"]
        adj_volume = X["adj_volume"]

        n_observations, n_assets = X.n_observations, X.n_assets

        result = np.empty((n_observations, n_assets), dtype=float)

        for t in range(n_observations):
            active_mask = X.active_mask[t]
            self._mean.reset(inactive_mask=~active_mask)
            self._mean.update(
                values=adj_volume[t],
                valid_mask=active_mask
                & np.isfinite(adj_volume[t])
                & (adj_volume[t] > 0),
            )
            result[t] = safe_divide(
                short_interest[t],
                self._mean.mean(min_periods=self._min_periods),
                fill_value=np.nan,
            )

        self.days_to_cover_ = result[-1].copy() if n_observations > 1 else result[-1]
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
