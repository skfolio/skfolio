"""Shareholder yield descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.stats import safe_divide
from skfolio.utils.validation import validate_asset_panel


class ShareholderYield(BaseDescriptor, stateless=True):
    r"""Shareholder yield descriptor.

    Computes net cash returned to common shareholders through dividends and share
    repurchases as a fraction of market capitalization:

    .. math::

        \text{shareholder\_yield}(t) =
        \frac{\text{dividends\_ttm}(t) + \text{net\_buybacks\_ttm}(t)}
             {\text{market\_cap}(t)}

    Dividend yield alone misses a large share of corporate payout. Since the 1990s,
    share repurchases have overtaken dividends as the dominant mechanism for returning
    cash to shareholders. Shareholder yield captures the total payout: a company
    paying 0% dividends but buying back 5% of its equity annually has a positive payout
    yield that pure dividend yield scores as zero [1]_.

    High shareholder yield identifies firms that return substantial capital.
    Empirically, shareholder yield subsumes much of the stand-alone dividend yield
    premium and provides a stronger value/payout signal [2]_.

    Parameters
    ----------
    None

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    Notes
    -----
    `dividends_ttm` should contain positive cash dividends paid on common shares only,
    excluding preferred dividends.

    `net_buybacks_ttm` should equal net share repurchases, defined as repurchases minus
    issuances, over the trailing twelve months. Positive values increase shareholder
    yield and negative values represent net issuance. Some data vendors provide net
    equity issuance from the cash flow statement instead, with the opposite sign
    convention. In that case, `net_buybacks_ttm = -net_equity_issuance_ttm`.

    This descriptor uses aggregate quantities divided by `market_cap`, consistent with
    `DividendToPrice` and other value descriptors.

    References
    ----------
    .. [1] "On the importance of measuring payout yield: implications for empirical
       asset pricing" The Journal of Finance. Boudoukh, Michaely, Richardson
       & Roberts (2007).

    .. [2] "Dividends, share repurchases, and the substitution hypothesis"
        The Journal of Finance. Grullon & Michaely (2002).

    See Also
    --------
    DividendToPrice : Dividend-only yield (trailing).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import ShareholderYield
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = ShareholderYield()
    >>> shareholder_yield = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute shareholder yield ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `dividends_ttm`, `net_buybacks_ttm` and
            `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        shareholder_yield : ndarray of shape (n_observations, n_assets)
            Shareholder yield ratio for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["dividends_ttm", "net_buybacks_ttm", "market_cap"],
            non_negative_or_nan=["dividends_ttm"],
            finite_or_nan=["net_buybacks_ttm"],
            strictly_positive_or_nan=["market_cap"],
        )
        return safe_divide(
            X["dividends_ttm"] + X["net_buybacks_ttm"],
            X["market_cap"],
            fill_value=np.nan,
        )
