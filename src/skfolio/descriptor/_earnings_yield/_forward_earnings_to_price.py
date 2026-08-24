"""Forward earnings-to-price ratio descriptor."""

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


class ForwardEarningsToPrice(BaseDescriptor, stateless=True):
    r"""Forward earnings-to-price ratio descriptor.

    Computes the ratio of consensus NTM earnings per share to split-adjusted close
    price:

    .. math::

        \text{forward\_earnings\_to\_price}(t) =
        \frac{\text{eps\_ntm}(t)}{\text{adj\_close}(t)}

    Forward earnings-to-price reflects consensus expectations of future profitability
    relative to the current price [1]_. Because it incorporates analyst forecasts rather
    than trailing accounting data, it captures forward-looking value and is less
    affected by stale or one-off items in historical earnings. A high ratio identifies
    firms expected to generate strong earnings relative to their price.

    Unlike the other value descriptors which use aggregate fundamentals divided by
    `market_cap`, this descriptor uses per-share quantities (`eps_ntm / adj_close`).
    Consensus estimates from data providers are delivered as per-share forecasts,
    making per-share the primary form. `eps_ntm` should use the same split-adjustment
    basis as `adj_close`.

    The aggregate equivalent is:

    .. math::

        \frac{\text{eps\_ntm}}{\text{adj\_close}}
        = \frac{\text{eps\_ntm} \times \text{shares\_out}}
              {\text{adj\_close} \times \text{shares\_out}}
        = \frac{\text{earnings\_ntm}}{\text{market\_cap}}


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
    .. [1] "Expectations and share prices"
        Management Science. Elton, E. J., Gruber, M. J., & Gultekin, M. (1981).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import ForwardEarningsToPrice
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = ForwardEarningsToPrice()
    >>> forward_earnings_to_price = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute forward earnings-to-price ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `eps_ntm` and `adj_close`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        forward_earnings_to_price : ndarray of shape (n_observations, n_assets)
            Forward earnings-to-price ratio for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["eps_ntm", "adj_close"],
            finite_or_nan=["eps_ntm"],
            strictly_positive_or_nan=["adj_close"],
        )
        return safe_divide(X["eps_ntm"], X["adj_close"], fill_value=np.nan)
