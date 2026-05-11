"""Debt-to-assets ratio descriptor."""

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


class DebtToAssets(BaseDescriptor, stateless=True):
    r"""Debt-to-assets ratio descriptor.

    Computes the ratio of total debt to total assets:

    .. math::

        \text{debt\_to\_assets}(t) =
        \frac{\text{total\_debt}(t)}{\text{total\_assets}(t)}

    Debt-to-assets is the most widely used leverage descriptor in equity risk models.
    It measures the proportion of a firm's asset base financed by debt. Higher values
    indicate greater financial risk: the firm has less equity cushion to absorb losses,
    making it more vulnerable to earnings shocks, rising interest rates and credit
    deterioration [1]_.

    The ratio is naturally bounded between 0 (no debt) and approximately 1 (assets fully
    debt-financed), though it can exceed 1 when accumulated losses erode equity below
    zero, making total liabilities exceed total assets.

    NaNs are allowed as missing observations and propagate to the output. Non-missing
    `total_debt` values must be finite. Non-missing `total_assets`  values must be
    finite and strictly positive.

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
    .. [1] "Debt/equity ratio and expected common stock returns: empirical evidence"
        The Journal of Finance. Bhandari, L. C. (1988).

    See Also
    --------
    BookLeverage : Leverage as a fraction of total book capital.
    MarketLeverage : Leverage as a fraction of total market capital.

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import DebtToAssets
    >>> descriptor = DebtToAssets()
    >>> d_to_a = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute debt-to-assets ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `total_debt` and `total_assets`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        debt_to_assets : ndarray of shape (n_observations, n_assets)
            Debt-to-assets ratio, with NaN where total assets is not positive.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["total_debt", "total_assets"],
            finite_or_nan=["total_debt", "total_assets"],
        )

        total_debt = X["total_debt"]
        total_assets = X["total_assets"]

        debt_to_asset = safe_divide(total_debt, total_assets, fill_value=np.nan)
        debt_to_asset = np.where(total_assets > 0, debt_to_asset, np.nan)

        return debt_to_asset
