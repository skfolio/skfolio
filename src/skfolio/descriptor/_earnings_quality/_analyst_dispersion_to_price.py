"""Analyst forecast dispersion to price descriptor."""

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


class AnalystDispersionToPrice(BaseDescriptor, stateless=True):
    r"""Analyst forecast dispersion to price descriptor.

    Computes the ratio of analyst earnings forecast dispersion to the split-adjusted
    close price:

    .. math::

        \text{analyst_dispersion_to_price}(t) =
        \frac{\text{eps_ntm_std}(t)}{\text{adj_close}(t)}

    Higher values indicate greater disagreement among analysts about a firm's forward
    earnings relative to its price. Forecast dispersion is a proxy for earnings
    uncertainty and information asymmetry. Empirically, stocks with high analyst
    disagreement tend to be overpriced and earn lower future returns [1]_.

    This descriptor uses per-share quantities (standard deviation of per-share EPS
    forecasts divided by split-adjusted price) because analyst consensus data is
    natively reported on a per-share basis.

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
    `eps_ntm_std` is the cross-analyst standard deviation of NTM EPS estimates,
    typically provided by consensus data vendors. It should use the same
    split-adjustment basis as `adj_close`.

    See Also
    --------
    ForwardEarningsToPrice : Level of forward earnings to price.

    References
    ----------
    .. [1] "Differences of opinion and the cross section of stock returns"
        The Journal of Finance. Diether, K. B., Malloy, C. J., & Scherbina, A. (2002).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import AnalystDispersionToPrice
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = AnalystDispersionToPrice()
    >>> analyst_dispersion_to_price = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute analyst earnings dispersion relative to price.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `eps_ntm_std` and `adj_close`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        analyst_dispersion_to_price : ndarray of shape (n_observations, n_assets)
            Standard deviation of forward EPS estimates divided by split-adjusted close.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["eps_ntm_std", "adj_close"],
            non_negative_or_nan=["eps_ntm_std"],
            strictly_positive_or_nan=["adj_close"],
        )
        return safe_divide(X["eps_ntm_std"], X["adj_close"], fill_value=np.nan)
