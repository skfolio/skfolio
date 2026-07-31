"""Return on assets descriptor."""

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


class ReturnOnAssets(BaseDescriptor, stateless=True):
    r"""Return on assets (ROA) descriptor.

    Computes the ratio of trailing twelve-month net income to total assets:

    .. math::

        \text{ROA}(t) = \frac{\text{net\_income\_ttm}(t)}{\text{total\_assets}(t)}

    Return on assets measures how efficiently a firm converts its asset base into
    earnings. Higher values indicate greater profitability per unit of capital deployed.

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
    Net income can be negative, so :math:`ROA` can be negative. Negative values
    distinguish profitable from unprofitable firms.

    See Also
    --------
    ReturnOnEquity : Profitability per unit of equity.
    AssetTurnover : Efficiency component of the DuPont decomposition.

    References
    ----------
    .. [1] "A five-factor asset pricing model"
        Journal of Financial Economics. Fama, E. F., & French, K. R. (2015).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import ReturnOnAssets
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = ReturnOnAssets()
    >>> return_on_assets = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute return on assets.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `net_income_ttm` and `total_assets`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        return_on_assets : ndarray of shape (n_observations, n_assets)
            Net income divided by total assets for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["net_income_ttm", "total_assets"],
            finite_or_nan=["net_income_ttm", "total_assets"],
        )
        return_on_assets = safe_divide(
            X["net_income_ttm"], X["total_assets"], fill_value=np.nan
        )
        return np.where(X["total_assets"] > 0, return_on_assets, np.nan)
