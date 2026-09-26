"""Gross margin descriptor."""

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


class GrossMargin(BaseDescriptor, stateless=True):
    r"""Gross margin descriptor.

    Computes the ratio of gross profit to sales:

    .. math::

        \text{gross\_margin}(t) =
        \frac{\text{sales\_ttm}(t) - \text{cost\_of\_revenue\_ttm}(t)}
             {\text{sales\_ttm}(t)}

    Gross margin captures pricing power and unit economics: the fraction of each dollar
    of revenue retained after direct production costs. A high and stable gross margin
    may reflect strong competitive positioning, brand value or cost advantages.

    While :class:`GrossProfitability` normalizes by total assets [1]_, gross margin
    normalizes by sales. The two descriptors capture related but distinct aspects of
    firm quality.

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
    `sales_ttm - cost_of_revenue_ttm` to obtain gross profit. Observations with
    `sales_ttm <= 0` are masked to NaN because the margin is not economically
    interpretable.

    See Also
    --------
    GrossProfitability : Gross profit normalized by total assets.

    References
    ----------
    .. [1] "The other side of value: The gross profitability premium"
        Journal of Financial Economics. Novy-Marx, R. (2013).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import GrossMargin
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = GrossMargin()
    >>> gross_margin = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute gross margin.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `sales_ttm` and `cost_of_revenue_ttm`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        gross_margin : ndarray of shape (n_observations, n_assets)
            Gross profit divided by sales for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["sales_ttm", "cost_of_revenue_ttm"],
            finite_or_nan=["sales_ttm", "cost_of_revenue_ttm"],
        )

        sales = X["sales_ttm"]
        gross_profit = sales - X["cost_of_revenue_ttm"]
        gross_margin = safe_divide(gross_profit, sales, fill_value=np.nan)
        return np.where(sales > 0, gross_margin, np.nan)
