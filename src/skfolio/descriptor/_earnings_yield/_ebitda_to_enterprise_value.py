"""EBITDA-to-enterprise-value ratio descriptor."""

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


class EbitdaToEnterpriseValue(BaseDescriptor, stateless=True):
    r"""EBITDA-to-enterprise-value ratio descriptor.

    Computes the ratio of trailing twelve-month EBITDA to enterprise value:

    .. math::

        \text{ebitda\_to\_enterprise\_value}(t) =
        \frac{\text{ebitda\_ttm}(t)}{\text{enterprise\_value}(t)}

    Enterprise value adjusts for capital structure by adding debt and subtracting cash
    and equivalents from market capitalization:

    .. math::

        EV = \text{market\_cap} + \text{total\_debt} - \text{cash\_and\_equivalents}

    EBITDA measures operating profitability before financing, taxes and non-cash
    charges. A high ratio identifies firms generating strong operating income relative
    to their total firm value, regardless of how they are financed. The corresponding
    enterprise multiple has been studied as a predictor of average stock returns [1]_.

    This is the inverse of the conventional EV/EBITDA multiple. It provides a valuation
    measure that is comparable across firms with different leverage, unlike price-based
    ratios which only reflect equity value.

    If `enterprise_value` is not available directly from your data provider, it can be
    computed as:

    .. math::

        \text{EV} = \text{market\_cap} + \text{total\_debt}
                   - \text{cash\_and\_equivalents}

    Non-missing `enterprise_value` values must be finite. Observations with
    `enterprise_value <= 0` are masked to NaN because the valuation yield is not
    economically interpretable.

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
    .. [1] "New evidence on the relation between the enterprise multiple and average
       stock returns" Journal of Financial and Quantitative Analysis.
       Loughran, T., & Wellman, J. W. (2011).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import EbitdaToEnterpriseValue
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = EbitdaToEnterpriseValue()
    >>> ebitda_to_enterprise_value = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute EBITDA-to-enterprise-value ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `ebitda_ttm` and `enterprise_value`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        ebitda_to_enterprise_value : ndarray of shape (n_observations, n_assets)
            EBITDA divided by enterprise value for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["ebitda_ttm", "enterprise_value"],
            finite_or_nan=["ebitda_ttm", "enterprise_value"],
        )
        ebitda_to_enterprise_value = safe_divide(
            X["ebitda_ttm"], X["enterprise_value"], fill_value=np.nan
        )
        return np.where(X["enterprise_value"] > 0, ebitda_to_enterprise_value, np.nan)
