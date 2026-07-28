"""Asset turnover descriptor."""

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


class AssetTurnover(BaseDescriptor, stateless=True):
    r"""Asset turnover descriptor.

    Computes the ratio of trailing twelve-month sales to total assets:

    .. math::

        \text{asset\_turnover}(t) = \frac{\text{sales\_ttm}(t)}{\text{total\_assets}(t)}

    Asset turnover measures how efficiently a firm uses its assets to generate revenue.
    Higher values indicate greater capital efficiency.

    Asset-light business models tend to have high turnover, while capital-intensive
    industries tend to have low turnover.

    Parameters
    ----------
    None

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    See Also
    --------
    ReturnOnAssets : :math:`ROA`, which decomposes into margin and turnover.

    References
    ----------
    .. [1] "Using asset turnover and profit margin to forecast changes in profitability"
        Review of Accounting Studies. Fairfield, P. M., & Yohn, T. L. (2001).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import AssetTurnover
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = AssetTurnover()
    >>> asset_turnover = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute asset turnover.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `sales_ttm` and `total_assets`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        asset_turnover : ndarray of shape (n_observations, n_assets)
            Sales divided by total assets for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["sales_ttm", "total_assets"],
            finite_or_nan=["sales_ttm", "total_assets"],
        )
        asset_turnover = safe_divide(
            X["sales_ttm"], X["total_assets"], fill_value=np.nan
        )
        return np.where(X["total_assets"] > 0, asset_turnover, np.nan)
