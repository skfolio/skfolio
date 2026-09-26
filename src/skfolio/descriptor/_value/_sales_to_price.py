"""Sales-to-price ratio descriptor."""

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


class SalesToPrice(BaseDescriptor, stateless=True):
    r"""Sales-to-price ratio descriptor.

    Computes the ratio of trailing twelve-month sales to market capitalization:

    .. math::

        \text{sales\_to\_price}(t) = \frac{\text{sales\_ttm}(t)}{\text{market\_cap}(t)}

    Sales are less directly affected by accounting choices than earnings, providing a
    stable value signal. Firms with high sales relative to market capitalization are
    cheap on a sales basis [1]_. This ratio remains available for firms with negative
    earnings or book equity, complementing :class:`BookToPrice`.

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
    Non-missing `market_cap` values must be finite and strictly positive.

    This descriptor uses aggregate quantities (total sales divided by total market
    capitalization) rather than per-share quantities (sales per share divided by price).
    The two are mathematically equivalent:

    .. math::

        \frac{\text{sales\_ttm}}{\text{market\_cap}}
        = \frac{\text{sales\_per\_share}}{\text{price}}

    The aggregate form is preferred because it avoids subtle split-adjustment
    mismatches between numerator and denominator. Aggregate fundamentals are the primary
    form from data providers; per-share quantities are derived from them.

    See Also
    --------
    BookToPrice : Common equity normalized by market capitalization.
    CashFlowToPrice : Operating cash flow normalized by market capitalization.

    References
    ----------
    .. [1] "Do sales-price and debt-equity explain stock returns better than book-market
        and firm size?" Financial Analysts Journal. Barbee, Mukherji & Raines (1996).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import SalesToPrice
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = SalesToPrice()
    >>> sales_to_price = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute sales to price.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `sales_ttm` and `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        sales_to_price : ndarray of shape (n_observations, n_assets)
            Sales-to-price ratio for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["sales_ttm", "market_cap"],
            finite_or_nan=["sales_ttm"],
            strictly_positive_or_nan=["market_cap"],
        )
        return safe_divide(X["sales_ttm"], X["market_cap"], fill_value=np.nan)
