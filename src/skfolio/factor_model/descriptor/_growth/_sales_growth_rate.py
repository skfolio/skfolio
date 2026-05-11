"""Sales growth rate descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.factor_model.descriptor._growth._base._growth_rate import GrowthRate


class SalesGrowthRate(GrowthRate):
    r"""Sales growth rate descriptor.

    Computes period-over-period growth in trailing twelve-month sales:

    .. math::

        \text{sales\_growth}(t)
        = \frac{\text{sales\_ttm}(t)}{\text{sales\_ttm}(t - \text{lag})} - 1

    The first `lag` observations are NaN because no lagged history is available.

    `sales_ttm` must contain non-missing finite non-negative values. NaNs are allowed
    as missing observations and propagate when either the current or lagged value is
    missing. Zero values are allowed and a zero lagged value makes the growth rate
    undefined and produces NaN.

    Sales growth measures top-line expansion over the lag window. For TTM sales with a
    one-year lag, the two observations cover non-overlapping fiscal content.

    This is a convenience subclass of :class:`GrowthRate` with `field="sales_ttm"`.

    Parameters
    ----------
    lag : int, default=252
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
        Last sales growth value for each asset.

    See Also
    --------
    GrowthRate : Generic period-over-period growth rate descriptor.

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import SalesGrowthRate
    >>> descriptor = SalesGrowthRate(lag=252)
    >>> sales_growth = descriptor.fit_transform(X)
    """

    def __init__(self, lag: int = 252):
        super().__init__(field="sales_ttm", lag=lag)
