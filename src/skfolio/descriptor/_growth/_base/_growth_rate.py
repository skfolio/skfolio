"""Period-over-period growth rate descriptor."""

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

_FITTED_ATTR = "growth_rate_"


class GrowthRate(BaseDescriptor):
    r"""Period-over-period growth rate descriptor.

    Computes the growth rate of a characteristic over a fixed lag:

    .. math::

        \text{growth}(t) = \frac{x(t)}{x(t - \text{lag})} - 1

    The first `lag` observations are NaN because no lagged history is available.

    This descriptor is intended for non-negative fields such as sales, total assets,
    capital expenditure or shares outstanding. NaNs are allowed as missing observations
    and propagate when either the current or lagged value is missing. It raises a
    `ValueError` for negative or infinite values. Zero values are allowed and a zero
    lagged value makes the growth rate undefined and produces NaN.

    For fields that can be negative, such as net income or EPS, use
    :class:`~skfolio.descriptor.EarningsChangeToPrice` instead. It
    normalizes the level change by market capitalization and does not rely on a positive
    base value.

    This is the standard period-over-period growth rate used in anomaly and
    investment-style factors. For trailing-twelve-month (TTM) fields with a one-year
    lag, the two observations cover non-overlapping fiscal content, so intermediate
    quarterly filings contribute to the comparison.

    Other growth definitions exist, including regression-based multi-year trend growth
    and compound annual growth rate (CAGR). For positive values, CAGR is monotonic in
    simple growth and gives the same cross-sectional ranks.

    Common investment-factor descriptors:

    Asset growth (`field="total_assets"`):
        Year-over-year balance-sheet expansion. Firms with rapid asset growth tend to
        earn lower future returns [1]_.

    Issuance growth (`field="adj_shares_outstanding"`):
        Year-over-year change in split-adjusted shares outstanding. Net share issuance
        is a negative predictor of future returns, independent of size, value and
        momentum [2]_.

    Capital expenditure growth (`field="capex_ttm"`):
        Year-over-year change in trailing capital expenditure. Firms with large capex
        increases subsequently underperform, consistent with investor under-reaction to
         overinvestment [3]_.

    Parameters
    ----------
    field : str
        Field name in the :class:`AssetPanel` to compute growth for. Non-missing values
        must be finite and non-negative.

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

    growth_rate_ : ndarray of shape (n_assets,)
        Last growth rate value for each asset.

    References
    ----------
    .. [1] "Asset growth and the cross-section of stock returns"
        The Journal of Finance. Cooper, M. J., Gulen, H., & Schill, M. J. (2008).

    .. [2] "Share issuance and cross-sectional returns"
        The Journal of Finance. Pontiff, J., & Woodgate, A. (2008).

    .. [3] "Capital investments and stock returns"
        Journal of Financial and Quantitative Analysis. Titman, Wei & Xie (2004).

    Examples
    --------
    >>> from skfolio.descriptor import GrowthRate
    >>>
    >>> # 1-year sales growth
    >>> sales_growth = GrowthRate("sales_ttm", lag=252)
    >>>
    >>> # 1-year asset growth
    >>> asset_growth = GrowthRate("total_assets", lag=252)
    >>>
    >>> # 1-year share issuance growth
    >>> issuance_growth = GrowthRate("adj_shares_outstanding", lag=252)
    """

    growth_rate_: FloatArray

    def __init__(self, field: str, lag: int):
        self.field = field
        self.lag = lag

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute simple growth rates of the configured field.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing the `field` characteristic configured at
            construction.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        growth_rate : ndarray of shape (n_observations, n_assets)
            Period-over-period growth rate for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute simple growth rates of the configured field.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing the `field` characteristic configured at construction.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        growth_rate : ndarray of shape (n_observations, n_assets)
            Period-over-period growth rate for each observation and asset.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        validate_asset_panel(
            self,
            X,
            required_fields=[self.field],
            finite_or_nan=[self.field],
            reset=first_call,
        )

        _validate_positive_integer(self.lag, "lag")

        values = X[self.field]
        n_observations, n_assets = X.n_observations, X.n_assets

        if np.any(values[~np.isnan(values)] < 0):
            raise ValueError(
                f'Field "{self.field}" contains negative values. GrowthRate requires a '
                f"non-negative characteristic (e.g., sales, total assets, capex). "
                f"For characteristics that can be negative (e.g., net income, EPS), "
                f"use EarningsChangeToPrice instead. If negatives are data quality "
                f"artifacts, clean the input field before using GrowthRate."
            )

        if first_call:
            # Pre-allocate the lag buffer, warm-up output stays NaN.
            self._buffer = np.full((self.lag, n_assets), np.nan, dtype=float)

        result = np.full((n_observations, n_assets), np.nan, dtype=float)

        # Lagged values from the existing buffer.
        n_from_buffer = min(self.lag, n_observations)
        result[:n_from_buffer] = (
            safe_divide(
                values[:n_from_buffer], self._buffer[:n_from_buffer], fill_value=np.nan
            )
            - 1
        )

        # Lagged values from the current batch.
        if n_observations > self.lag:
            result[self.lag :] = (
                safe_divide(
                    values[self.lag :],
                    values[: n_observations - self.lag],
                    fill_value=np.nan,
                )
                - 1
            )

        # Update the buffer in-place.
        _update_buffer(self._buffer, values, self.lag)

        # Mask output for inactive assets.
        result = np.where(X.active_mask, result, np.nan)

        self.growth_rate_ = result[-1].copy() if n_observations > 1 else result[-1]

        return result

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)
        if hasattr(self, "_buffer"):
            delattr(self, "_buffer")
