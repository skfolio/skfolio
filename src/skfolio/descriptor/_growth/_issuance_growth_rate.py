"""Issuance growth rate descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.descriptor._growth._base._growth_rate import GrowthRate


class IssuanceGrowthRate(GrowthRate):
    r"""Issuance growth rate descriptor.

    Computes period-over-period growth in split-adjusted shares outstanding:

    .. math::

        \text{issuance_growth}(t)
        = \frac{\text{adj_shares_outstanding}(t)}
               {\text{adj_shares_outstanding}(t - \text{lag})} - 1

    The first `lag` observations are NaN because no lagged history is available.

    `adj_shares_outstanding` must contain non-missing finite non-negative values. NaNs
    are allowed as missing observations and propagate when either the current or lagged
    value is missing. Zero values are allowed and a zero lagged value makes the growth
    rate undefined and produces NaN.

    Positive issuance growth indicates an increase in split-adjusted shares outstanding.
    Net share issuance is a negative predictor of future returns, independent of size,
    value and momentum [1]_.

    This is a convenience subclass of :class:`GrowthRate` with
    `field="adj_shares_outstanding"`.

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
        Last issuance growth value for each asset.

    References
    ----------
    .. [1] "Share issuance and cross-sectional returns"
        The Journal of Finance. Pontiff, J., & Woodgate, A. (2008).

    See Also
    --------
    GrowthRate : Generic period-over-period growth rate descriptor.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import IssuanceGrowthRate
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = IssuanceGrowthRate(lag=252)
    >>> issuance_growth_rate = descriptor.fit_transform(X)
    """

    def __init__(self, lag: int = 252):
        super().__init__(field="adj_shares_outstanding", lag=lag)
