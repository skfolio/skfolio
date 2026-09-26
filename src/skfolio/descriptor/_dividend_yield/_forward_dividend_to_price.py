"""Forward dividend-to-price ratio descriptor."""

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


class ForwardDividendToPrice(BaseDescriptor, stateless=True):
    r"""Forward dividend-to-price ratio descriptor.

    Computes the ratio of consensus forward twelve-month dividend per share to
    split-adjusted close price:

    .. math::

        \text{forward\_dividend\_to\_price}(t) =
        \frac{\text{dps\_ntm}(t)}{\text{adj\_close}(t)}

    Forward dividend-to-price captures the expected income yield based on analyst
    consensus forecasts. Because it incorporates forward-looking estimates rather than
    trailing accounting data, it reacts more quickly to dividend initiations, cuts or
    policy changes. A high ratio identifies firms where analysts expect generous payouts
    relative to the current price.

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
    Unlike `DividendToPrice`, which uses aggregate fundamentals divided by `market_cap`,
    this descriptor uses per-share quantities (`dps_ntm / adj_close`). Consensus
    estimates from data providers are typically delivered as per-share forecasts, making
    per-share the primary form. `dps_ntm` should use the same split-adjustment basis
    as `adj_close`.

    The aggregate equivalent is:

    .. math::

        \frac{\text{dps\_ntm}}{\text{adj\_close}}
        = \frac{\text{dps\_ntm} \times \text{shares\_out}}
              {\text{adj\_close} \times \text{shares\_out}}
        = \frac{\text{forward\_dividends\_ntm}}{\text{market\_cap}}

    See Also
    --------
    DividendToPrice : Trailing (historical) dividend yield.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import ForwardDividendToPrice
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = ForwardDividendToPrice()
    >>> forward_dividend_to_price = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute forward dividend-to-price ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `dps_ntm` and `adj_close`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        forward_dividend_to_price : ndarray of shape (n_observations, n_assets)
            Forward dividend-to-price ratio for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["dps_ntm", "adj_close"],
            non_negative_or_nan=["dps_ntm"],
            strictly_positive_or_nan=["adj_close"],
        )
        return safe_divide(X["dps_ntm"], X["adj_close"], fill_value=np.nan)
