"""Cash flow to assets descriptor."""

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


class CashFlowToAssets(BaseDescriptor, stateless=True):
    r"""Cash flow to assets descriptor.

    Computes the ratio of trailing twelve-month operating cash flow to total assets:

    .. math::

        \text{cash\_flow\_to\_assets}(t) =
        \frac{\text{operating\_cash\_flow\_ttm}(t)}{\text{total\_assets}(t)}

    Cash flow to assets measures cash-based profitability: how much cash a firm
    generates from operations per unit of assets. Unlike net income-based measures
    (:class:`ReturnOnAssets`), operating cash flow is less directly affected by accrual
    accounting choices.

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
    Operating cash flow can be negative, so this descriptor can take negative
    values.

    See Also
    --------
    ReturnOnAssets : Net income-based profitability per unit of assets.
    CashFlowToPrice : Cash flow normalized by market cap (value signal).

    References
    ----------
    .. [1] "Accruals, cash flows, and operating profitability in the cross section of
       stock returns" Journal of Financial Economics.
       Ball, Gerakos, Linnainmaa & Nikolaev (2016).

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import CashFlowToAssets
    >>> descriptor = CashFlowToAssets()
    >>> cfoa = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute cash flow to assets.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `operating_cash_flow_ttm` and `total_assets`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        cash_flow_to_assets : ndarray of shape (n_observations, n_assets)
            Operating cash flow divided by total assets for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["operating_cash_flow_ttm", "total_assets"],
            finite_or_nan=["operating_cash_flow_ttm", "total_assets"],
        )
        cash_flow_to_assets = safe_divide(
            X["operating_cash_flow_ttm"], X["total_assets"], fill_value=np.nan
        )
        return np.where(X["total_assets"] > 0, cash_flow_to_assets, np.nan)
