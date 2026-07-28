"""Earnings-to-price ratio descriptor."""

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


class EarningsToPrice(BaseDescriptor, stateless=True):
    r"""Earnings-to-price ratio descriptor.

    Computes the ratio of trailing twelve-month net income to market capitalization:

    .. math::

        \text{earnings\_to\_price}(t) =
        \frac{\text{net\_income\_ttm}(t)}{\text{market\_cap}(t)}

    This is the inverse of the price-to-earnings (P/E) ratio and measures how much
    profit a firm generates per unit of market value. A high ratio identifies firms with
    strong current profitability relative to their price. Unlike :class:`BookToPrice`,
    which is based on the balance sheet, this descriptor is based on the income
    statement, capturing a distinct dimension of value.

    This descriptor can be negative for loss-making firms, which is economically
    meaningful (unlike P/E, which becomes uninterpretable for negative earnings).

    `net_income_ttm` should represent net income available to common shareholders when
    the data source distinguishes common and preferred claims. This is consistent with
    `market_cap`, which reflects common equity.

    This descriptor uses aggregate quantities (total net income divided by total market
    capitalization) rather than per-share quantities (earnings per share divided by
    price). The two are mathematically equivalent when EPS and price use the same
    split-adjustment basis:

    .. math::

        \frac{\text{net\_income\_ttm}}{\text{market\_cap}}
        = \frac{\text{eps\_ttm}}{\text{price}}

    The aggregate form is preferred because it avoids subtle split-adjustment mismatches
    between numerator and denominator. Aggregate fundamentals are the primary form from
    data providers. Per-share quantities are derived from them.

    Parameters
    ----------
    None

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    References
    ----------
    .. [1] "Investment performance of common stocks in relation to their price-earnings
       ratios: A test of the efficient market hypothesis" The Journal of Finance.
       Basu, S. (1977).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EarningsToPrice
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = EarningsToPrice()
    >>> earnings_to_price = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute trailing earnings-to-price ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `net_income_ttm` and `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        earnings_to_price : ndarray of shape (n_observations, n_assets)
            Earnings-to-price ratio for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["net_income_ttm", "market_cap"],
            finite_or_nan=["net_income_ttm"],
            strictly_positive_or_nan=["market_cap"],
        )
        return safe_divide(X["net_income_ttm"], X["market_cap"], fill_value=np.nan)
