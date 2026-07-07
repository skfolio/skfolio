"""Gross profitability descriptor."""

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


class GrossProfitability(BaseDescriptor, stateless=True):
    r"""Gross profitability descriptor.

    Computes the ratio of gross profit to total assets:

    .. math::

        \text{gross_profitability}(t) =
        \frac{\text{sales_ttm}(t) - \text{cost_of_revenue_ttm}(t)}
             {\text{total_assets}(t)}

    Gross profitability captures a firm's ability to generate profit from its asset base
    before operating expenses, interest and taxes. It is less affected by financing,
    tax and accrual accounting choices than net income-based ratios.

    Novy-Marx (2013) shows that profitable firms earn significantly higher returns than
    unprofitable ones [1]_.

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
    `cost_of_revenue_ttm` (trailing twelve months) should be reported as a positive
    number representing the cost. The descriptor computes
    `sales_ttm - cost_of_revenue_ttm` to obtain gross profit.

    See Also
    --------
    GrossMargin : Gross profit normalized by sales (pricing power).
    ReturnOnAssets : Net income normalized by total assets.

    References
    ----------
    .. [1] "The other side of value: The gross profitability premium"
        Journal of Financial Economics. Novy-Marx, R. (2013).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import GrossProfitability
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = GrossProfitability()
    >>> gross_profitability = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute gross profitability.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `sales_ttm`, `cost_of_revenue_ttm`, and
            `total_assets`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        gross_profitability : ndarray of shape (n_observations, n_assets)
            Gross profit divided by total assets for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["sales_ttm", "cost_of_revenue_ttm", "total_assets"],
            finite_or_nan=["sales_ttm", "cost_of_revenue_ttm", "total_assets"],
        )
        gross_profitability = safe_divide(
            X["sales_ttm"] - X["cost_of_revenue_ttm"],
            X["total_assets"],
            fill_value=np.nan,
        )
        return np.where(X["total_assets"] > 0, gross_profitability, np.nan)
