"""Book-to-price ratio descriptor."""

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


class BookToPrice(BaseDescriptor, stateless=True):
    r"""Book-to-price ratio descriptor.

    Computes the ratio of common shareholders' equity (book equity) to market
    capitalization:

    .. math::

        \text{book\_to\_price}(t) = \frac{\text{book\_equity}(t)}{\text{market\_cap}(t)}

    A high book-to-price ratio identifies stocks trading at a discount relative to their
    common equity. Historically, cheap stocks (those with high book-to-price ratios) have
    earned higher average returns than expensive stocks (those with low book-to-price
    ratios) [1]_.

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
    Non-missing `market_cap` values must be finite and strictly positive. Negative
    `book_equity` values are preserved because they carry information about the firm's
    balance sheet.

    This descriptor uses aggregate quantities (common equity divided by total market
    capitalization) rather than per-share quantities (book value per share divided by
    price). The two are mathematically equivalent:

    .. math::

        \frac{\text{book\_equity}}{\text{price} \times \text{shares\_out}}
        = \frac{\text{book\_value\_per\_share}}{\text{price}}

    The aggregate form is preferred because it avoids subtle split-adjustment mismatches
    between numerator and denominator. Aggregate fundamentals are the primary form from
    data providers; per-share quantities are derived from them.

    See Also
    --------
    SalesToPrice : Sales normalized by market capitalization.
    CashFlowToPrice : Operating cash flow normalized by market capitalization.

    References
    ----------
    .. [1] "The cross-section of expected stock returns"
        The Journal of Finance. Fama, E. F., & French, K. R. (1992).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import BookToPrice
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = BookToPrice()
    >>> book_to_price = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute book to price.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `book_equity` and `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        book_to_price : ndarray of shape (n_observations, n_assets)
            Book-to-price ratio for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["book_equity", "market_cap"],
            finite_or_nan=["book_equity"],
            strictly_positive_or_nan=["market_cap"],
        )
        return safe_divide(X["book_equity"], X["market_cap"], fill_value=np.nan)
