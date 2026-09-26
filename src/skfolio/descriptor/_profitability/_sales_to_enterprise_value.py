"""Sales to enterprise value descriptor."""

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


class SalesToEnterpriseValue(BaseDescriptor, stateless=True):
    r"""Sales to enterprise value descriptor.

    Computes the ratio of trailing twelve-month sales to enterprise value:

    .. math::

        \text{sales\_to\_enterprise\_value}(t) =
        \frac{\text{sales\_ttm}(t)}{\text{enterprise\_value}(t)}

    This descriptor is a valuation and efficiency measure: it measures how much revenue
    a firm generates per unit of enterprise value. Unlike :class:`AssetTurnover`, which
    normalizes by book assets, enterprise value reflects the market's assessment of the
    entire capital structure [1]_.

    A high sales-to-enterprise-value ratio identifies firms that generate substantial
    revenue relative to their market valuation, combining elements of both value and
    operational efficiency.

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
    If `enterprise_value` is not available directly from your data provider, it can be
    computed as:

    .. math::

        \text{EV} = \text{market\_cap} + \text{total\_debt}
                   - \text{cash\_and\_equivalents}

    Non-missing `enterprise_value` values must be finite. Observations with
    `enterprise_value <= 0` are masked to NaN because the valuation yield is not
    economically interpretable.

    See Also
    --------
    AssetTurnover : Sales normalized by book assets (efficiency).
    EbitdaToEnterpriseValue : EBITDA normalized by enterprise value.

    References
    ----------
    .. [1] "New evidence on the relation between the enterprise multiple and average
       stock returns" Journal of Financial and Quantitative Analysis.
       Loughran, T., & Wellman, J. W. (2011).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import SalesToEnterpriseValue
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = SalesToEnterpriseValue()
    >>> sales_to_enterprise_value = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute sales to enterprise value.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `sales_ttm` and `enterprise_value`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        sales_to_enterprise_value : ndarray of shape (n_observations, n_assets)
            Sales divided by enterprise value for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["sales_ttm", "enterprise_value"],
            finite_or_nan=["sales_ttm", "enterprise_value"],
        )
        sales_to_enterprise_value = safe_divide(
            X["sales_ttm"], X["enterprise_value"], fill_value=np.nan
        )
        return np.where(X["enterprise_value"] > 0, sales_to_enterprise_value, np.nan)
