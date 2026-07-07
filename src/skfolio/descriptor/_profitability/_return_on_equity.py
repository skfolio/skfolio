"""Return on equity descriptor."""

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


class ReturnOnEquity(BaseDescriptor, stateless=True):
    r"""Return on equity (ROE) descriptor.

    Computes the ratio of trailing twelve-month net income to common shareholders'
    equity:

    .. math::

        \text{ROE}(t) =  \frac{\text{net_income_ttm}(t)}{\text{book_equity}(t)}

    Return on equity measures profitability from the common shareholders' perspective:
    how much profit a firm generates per unit of common equity capital. Stocks with
    high :math:`ROE` tend to earn higher average returns [1]_.

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
    When `book_equity <= 0` (e.g., firms with accumulated deficits or heavy share
    buybacks), the ratio does not represent an interpretable return on equity. These
    observations are masked to NaN.

    See Also
    --------
    ReturnOnAssets : Profitability per unit of total assets.

    References
    ----------
    .. [1] "A five-factor asset pricing model"
        Journal of Financial Economics. Fama, E. F., & French, K. R. (2015).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import ReturnOnEquity
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = ReturnOnEquity()
    >>> return_on_equity = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute return on equity.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `net_income_ttm` and `book_equity`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        return_on_equity : ndarray of shape (n_observations, n_assets)
            Net income divided by book equity for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["net_income_ttm", "book_equity"],
            finite_or_nan=["net_income_ttm", "book_equity"],
        )
        net_income = X["net_income_ttm"]
        book_equity = X["book_equity"]
        return_on_equity = safe_divide(net_income, book_equity, fill_value=np.nan)
        return np.where(book_equity > 0, return_on_equity, np.nan)
