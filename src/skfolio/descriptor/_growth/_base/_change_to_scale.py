"""Change-to-scale descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils._array_buffer import _update_buffer
from skfolio.utils.stats import safe_divide
from skfolio.utils.tools import _validate_positive_integer
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "change_to_scale_"


class ChangeToScale(BaseDescriptor):
    r"""Lagged change normalized by a positive scale.

    Computes the change in `field` over a fixed lag, divided by the current value of
    `scale_field`:

    .. math::

        \text{ChangeToScale}_\ell(t) = \frac{A(t) - A(t - \ell)}{S(t)}

    where :math:`A` is the `field` value and :math:`S` is the `scale_field` value.

    The first `lag` observations are NaN because no lagged history is available.

    This descriptor is appropriate when the numerator field can be negative or cross
    zero, such as earnings, capex or cash flows, and the change should be scaled by the
    firm's current size or valuation. NaNs are allowed as missing observations and
    propagate when the current, lagged or scale value is missing.

    Non-missing numerator values must be finite. Non-missing scale values must be finite
    and strictly positive. A `ValueError` is raised otherwise.

    Parameters
    ----------
    field : str
        Field name in the :class:`~skfolio.containers.AssetPanel` to compute the
        change for. Non-missing values must be finite.

    scale_field : str
        Field name in the :class:`~skfolio.containers.AssetPanel` used as the current
        positive denominator. Non-missing values must be finite and strictly positive.

    lag : int
        Number of observations to look back. The interpretation depends on the data
        frequency: `lag=12` means 1 year for monthly data, `lag=252` for daily data,
        `lag=4` for quarterly data.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    change_to_scale_ : ndarray of shape (n_assets,)
        Last change-to-scale value for each asset.

    See Also
    --------
    ChangeInIntensity : Change in the ratio :math:`A/S` (intensity change).
    GrowthRate : Simple growth rate for positive-definite characteristics.

    Examples
    --------
    >>> from skfolio.descriptor import ChangeToScale
    >>>
    >>> # Earnings change to price (equivalent to EarningsChangeToPrice)
    >>> earnings_chg = ChangeToScale("net_income_ttm", "market_cap", lag=12)
    >>>
    >>> # Capex change to total assets
    >>> capex_chg = ChangeToScale("capex_ttm", "total_assets", lag=12)
    """

    change_to_scale_: FloatArray

    def __init__(self, field: str, scale_field: str, lag: int):
        self.field = field
        self.scale_field = scale_field
        self.lag = lag

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute changes in level normalized by current scale.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing the `field` and `scale_field` characteristics
            configured at construction.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        change_to_scale : ndarray of shape (n_observations, n_assets)
            Change in `field` over the lag window, divided by current
            `scale_field`.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute changes in level normalized by current scale.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing the `field` and `scale_field` fields configured at
            construction.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        change_to_scale : ndarray of shape (n_observations, n_assets)
            Change in `field` over the lag window, divided by current `scale_field`.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=[self.field, self.scale_field],
            finite_or_nan=[self.field],
            strictly_positive_or_nan=[self.scale_field],
            reset=first_call,
        )
        _validate_positive_integer(self.lag, "lag")

        values = X[self.field]
        scale = X[self.scale_field]
        n_observations, n_assets = X.n_observations, X.n_assets

        if first_call:
            self._buffer = np.full((self.lag, n_assets), np.nan, dtype=float)

        result = np.full((n_observations, n_assets), np.nan, dtype=float)

        # Lagged values from the existing buffer.
        n_from_buffer = min(self.lag, n_observations)
        result[:n_from_buffer] = safe_divide(
            values[:n_from_buffer] - self._buffer[:n_from_buffer],
            scale[:n_from_buffer],
            fill_value=np.nan,
        )

        # Lagged values from the current batch.
        if n_observations > self.lag:
            result[self.lag :] = safe_divide(
                values[self.lag :] - values[: n_observations - self.lag],
                scale[self.lag :],
                fill_value=np.nan,
            )

        # Update the buffer in-place.
        _update_buffer(self._buffer, values, self.lag)

        # Mask output for inactive assets.
        result = np.where(X.active_mask, result, np.nan)

        self.change_to_scale_ = result[-1].copy() if n_observations > 1 else result[-1]

        return result

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)
        if hasattr(self, "_buffer"):
            delattr(self, "_buffer")
