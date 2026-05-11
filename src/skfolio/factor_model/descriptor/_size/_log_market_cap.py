"""Log market capitalization descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers import AssetPanel
from skfolio.factor_model.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray
from skfolio.utils.validation import validate_asset_panel


class LogMarketCap(BaseDescriptor, stateless=True):
    r"""Log market capitalization descriptor.

    Computes the natural logarithm of market capitalization:

    .. math::

        \text{log\_market\_cap}(t) = \ln(\text{market\_cap}(t))

    Log market capitalization is the standard size descriptor in equity factor models.
    The logarithm reduces the right skew of market capitalization and produces a more
    stable cross-sectional scale.

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
    `market_cap` is the market value of common equity, computed as split-adjusted price
    times common shares outstanding.

    Non-missing `market_cap` values must be finite and strictly positive.

    References
    ----------
    .. [1] "Common risk factors in the returns on stocks and bonds"
        Journal of Financial Economics. Fama, E. F., & French, K. R. (1993).

    Examples
    --------
    >>> from skfolio.factor_model.descriptor import LogMarketCap
    >>> descriptor = LogMarketCap()
    >>> log_market_cap = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute log market capitalization.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `market_cap`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        log_market_cap : ndarray of shape (n_observations, n_assets)
            Log market capitalization for each observation and asset.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["market_cap"],
            strictly_positive_or_nan=["market_cap"],
        )
        return np.log(X["market_cap"])
