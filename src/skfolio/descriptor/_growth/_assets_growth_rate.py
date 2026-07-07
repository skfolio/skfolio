"""Assets growth rate descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.descriptor._growth._base._growth_rate import GrowthRate


class AssetsGrowthRate(GrowthRate):
    r"""Asset growth rate descriptor.

    Computes period-over-period growth in total assets:

    .. math::

        \text{assets_growth}(t)
        = \frac{\text{total_assets}(t)}
               {\text{total_assets}(t - \text{lag})} - 1

    The first `lag` observations are NaN because no lagged history is available.

    `total_assets` must contain non-missing finite non-negative values. NaNs are
    allowed as missing observations and propagate when either the current or lagged
    value is missing. Zero values are allowed and a zero lagged value makes the growth
    rate undefined and produces NaN.

    Asset growth is commonly used as an investment or balance-sheet expansion signal.
    Firms with rapid asset growth tend to earn lower future returns [1]_.

    This is a convenience subclass of :class:`GrowthRate` with `field="total_assets"`.

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
        Last asset growth value for each asset.

    References
    ----------
    .. [1] "Asset growth and the cross-section of stock returns"
        The Journal of Finance. Cooper, M. J., Gulen, H., & Schill, M. J. (2008).

    See Also
    --------
    GrowthRate : Generic period-over-period growth rate descriptor.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import AssetsGrowthRate
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = AssetsGrowthRate(lag=252)
    >>> assets_growth_rate = descriptor.fit_transform(X)
    """

    def __init__(self, lag: int = 252):
        super().__init__(field="total_assets", lag=lag)
