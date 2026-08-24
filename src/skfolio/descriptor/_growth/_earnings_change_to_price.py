"""Earnings change to price descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.descriptor._growth._base._change_to_scale import ChangeToScale


class EarningsChangeToPrice(ChangeToScale):
    r"""Lagged earnings change divided by current market capitalization.

    Computes the change in trailing twelve-month net income over a fixed lag, divided by
    current market capitalization:

    .. math::

        \text{earnings\_change\_to\_price}(t) =
        \frac{\text{net\_income\_ttm}(t) - \text{net\_income\_ttm}(t - \text{lag})}
             {\text{market\_cap}(t)}

    The first `lag` observations are NaN because no lagged history is available.

    NaNs are allowed as missing observations and propagate when the current, lagged or
    market-cap value is missing. Non-missing `net_income_ttm` values must be finite.
    Non-missing `market_cap` values must be finite and strictly positive.

    This descriptor captures earnings momentum: whether a firm's profitability is
    improving or deteriorating relative to its market value [1]_. A positive value
    indicates earnings improvement and a negative value indicates deterioration.

    Unlike :class:`GrowthRate`, which computes `x(t) / x(t-lag) - 1`, this formulation
    is well-defined when earnings are negative. A standard growth rate with a negative
    base produces sign-inverted rankings, making it unsuitable for earnings. By
    normalizing the level change with market capitalization, the sign of the output
    reflects the direction of change.

    This is a convenience subclass of :class:`ChangeToScale` with
    `field="net_income_ttm"` and `scale_field="market_cap"`.

    This descriptor uses aggregate quantities (net income and market capitalization).
    The per-share equivalent is:

    .. math::

        \frac{\text{eps\_ttm}(t) - \text{eps\_ttm}(t - \text{lag})}
             {\text{adj\_close}(t)}

    The aggregate form is preferred for consistency with the other value descriptors
    and to avoid split-adjustment mismatches.

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

    change_to_scale_ : ndarray of shape (n_assets,)
        Last earnings change to price value for each asset.

    References
    ----------
    .. [1] Fama, E. F., & French, K. R. (2006). "Profitability,
       investment and average returns." *Journal of Financial Economics*,
       82(3), 491-518.

    See Also
    --------
    ChangeToScale : Generic change-to-scale descriptor.
    GrowthRate : Simple growth rate for positive-definite characteristics.
    EarningsToPrice : Level of trailing earnings to price (value signal).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EarningsChangeToPrice
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = EarningsChangeToPrice(lag=252)
    >>> earnings_change_to_price = descriptor.fit_transform(X)
    """

    def __init__(self, lag: int = 252):
        super().__init__(field="net_income_ttm", scale_field="market_cap", lag=lag)
