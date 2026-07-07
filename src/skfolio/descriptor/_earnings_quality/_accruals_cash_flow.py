"""Cash-flow accruals descriptor."""

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


class AccrualsCashFlow(BaseDescriptor, stateless=True):
    r"""Cash-flow statement accruals descriptor.

    Computes the non-cash component of earnings, scaled by total assets:

    .. math::

        \text{accruals_cash_flow}(t) =
        \frac{\text{net_income_ttm}(t) - \text{operating_cash_flow_ttm}(t)}
             {\text{total_assets}(t)}

    High accruals indicate that reported earnings substantially exceed cash generated
    from operations. Empirically, firms with high accruals tend to have less persistent
    earnings and lower future returns, a pattern known as the accrual anomaly [1]_.

    This cash-flow statement version is preferred over the balance-sheet version because
    it requires fewer line items and is less sensitive to data-provider mapping
    differences. The balance-sheet version (which computes accruals from changes in
    working capital items) can be pre-computed and fed via :class:`Passthrough` if
    needed.

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
    The sign convention follows the academic literature: a positive value means earnings
    exceed cash flow (high accruals, lower quality), while a negative value means cash
    flow exceeds earnings (low accruals, higher quality).

    See Also
    --------
    ReturnOnAssets : Net income-based profitability per unit of assets.
    CashFlowToAssets : Cash flow-based profitability per unit of assets.

    References
    ----------
    .. [1] "Do stock prices fully reflect information in accruals and cash flows about
        future earnings?" The Accounting Review. Sloan, R. G. (1996).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import AccrualsCashFlow
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = AccrualsCashFlow()
    >>> accruals_cash_flow = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute accruals scaled by total assets.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `net_income_ttm`, `operating_cash_flow_ttm`, and
            `total_assets`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        accruals_cash_flow : ndarray of shape (n_observations, n_assets)
            Accruals (net income minus operating cash flow) divided by total assets.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=[
                "net_income_ttm",
                "operating_cash_flow_ttm",
                "total_assets",
            ],
            finite_or_nan=[
                "net_income_ttm",
                "operating_cash_flow_ttm",
                "total_assets",
            ],
        )
        accruals_cash_flow = safe_divide(
            X["net_income_ttm"] - X["operating_cash_flow_ttm"],
            X["total_assets"],
            fill_value=np.nan,
        )
        return np.where(X["total_assets"] > 0, accruals_cash_flow, np.nan)
