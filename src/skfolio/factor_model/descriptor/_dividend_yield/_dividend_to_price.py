"""Dividend-to-price ratio descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.factor_model.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.stats import safe_divide
from skfolio.utils.validation import validate_asset_panel


class DividendToPrice(BaseDescriptor, stateless=True):
    r"""Dividend-to-price ratio descriptor.

    Computes the ratio of trailing twelve-month common dividends to market
     capitalization:

    .. math::

        \text{dividend\_to\_price}(t) =
        \frac{\text{dividends\_ttm}(t)}{\text{market\_cap}(t)}

    Dividend-to-price measures the income yield that shareholders receive relative to
    the current market price. High-yield stocks tend to be mature, cash-generative
    businesses, while low-yield stocks are typically growth-oriented or retain earnings
    for reinvestment [1]_. The dividend yield factor captures a distinct dimension of
    value beyond book or earnings ratios because dividends reflect management's
    confidence in sustainable cash flows.

    `dividends_ttm` should contain positive cash dividends paid on common shares only,
    excluding preferred dividends. This is consistent with `market_cap`, which reflects
    common equity.

    This descriptor uses aggregate quantities (dividends paid divided by market
    capitalization) rather than per-share quantities (dividends per share divided by
    split-adjusted close price). The two are mathematically equivalent when the price
    and per-share dividend use the same split-adjustment basis:

    .. math::

        \frac{\text{dividends\_ttm}}{\text{market\_cap}}
        = \frac{\text{dividends\_ttm} / \text{shares\_out}}{\text{adj\_close}}
        = \frac{\text{dps\_ttm}}{\text{adj\_close}}

    The aggregate form is preferred because it avoids subtle split-adjustment mismatches
    between numerator and denominator. Aggregate fundamentals are the primary form from
    data providers and per-share quantities are derived from them.

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
    .. [1] "Common risk factors in the returns on stocks and bonds"
        Journal of Financial Economics. Fama, E. F., & French, K. R. (1993).

    See Also
    --------
    ForwardDividendToPrice : Forward (analyst-predicted) dividend yield.
    ShareholderYield : Dividend yield plus net buybacks.

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import DividendToPrice
    >>> descriptor = DividendToPrice()
    >>> div_yield = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute trailing dividend-to-price ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `dividends_ttm` and `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        dividend_to_price : ndarray of shape (n_observations, n_assets)
            Dividend-to-price ratio for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["dividends_ttm", "market_cap"],
            non_negative_or_nan=["dividends_ttm"],
            strictly_positive_or_nan=["market_cap"],
        )
        return safe_divide(X["dividends_ttm"], X["market_cap"], fill_value=np.nan)
