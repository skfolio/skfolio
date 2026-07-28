"""Market leverage descriptor."""

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


class MarketLeverage(BaseDescriptor, stateless=True):
    r"""Market leverage descriptor.

    Computes the proportion of total capital (at market value) financed by debt:

    .. math::

        \text{market\_leverage}(t) =
        \frac{\text{total\_debt}(t)}
             {\text{total\_debt}(t) + \text{market\_cap}(t)}

    Market leverage blends accounting data (total debt) with market data (market
    capitalization). Unlike :class:`BookLeverage`, the denominator updates daily with
    the stock price, making it more responsive to changes in the firm's risk profile.
    When a stock drops sharply, market leverage rises immediately, capturing the
    increased financial risk before any accounting restatement [1]_.

    NaNs are allowed as missing observations and propagate to the output. Non-missing
    `total_debt` values must be finite. Non-missing `market_cap` values must be finite
    and strictly positive.

    When `total_debt` is non-negative and `market_cap` is positive, the ratio is bounded
    in :math:`[0, 1)`. This makes it the most numerically well-behaved of the leverage
    descriptors, requiring no special treatment for negative-equity firms.

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
    .. [1] "Capital structure decisions: which factors are reliably important?"
        Financial Management. Frank, M. Z., & Goyal, V. K. (2009).

    See Also
    --------
    DebtToAssets : Leverage relative to total assets.
    BookLeverage : Leverage as a fraction of total book capital.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import MarketLeverage
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = MarketLeverage()
    >>> market_leverage = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute market leverage ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `total_debt` and `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        market_leverage : ndarray of shape (n_observations, n_assets)
            Market leverage ratio, with NaN where market capitalization or total
            capital is not positive.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["total_debt", "market_cap"],
            finite_or_nan=["total_debt", "market_cap"],
        )

        total_debt = X["total_debt"]
        market_cap = X["market_cap"]

        denominator = total_debt + market_cap
        market_leverage = safe_divide(total_debt, denominator, fill_value=np.nan)
        market_leverage = np.where(
            (market_cap > 0) & (denominator > 0), market_leverage, np.nan
        )

        return market_leverage
