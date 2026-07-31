"""Short interest as fraction of shares outstanding."""

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


class ShortInterest(BaseDescriptor, stateless=True):
    r"""Short interest descriptor.

    Computes the ratio of shares sold short to common shares outstanding:

    .. math::

        \text{short\_interest}(t) =
        \frac{\text{short\_interest}(t)}
             {\text{adj\_shares\_outstanding}(t)}

    Short interest measures the fraction of common shares outstanding that has been
    borrowed and sold short. High values indicate stronger bearish positioning and may
    proxy for informed negative sentiment [1]_ [2]_.

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
    `short_interest` is the number of shares held short. Non-missing values must be
    finite and non-negative.

    `adj_shares_outstanding` is common shares outstanding. Non-missing values must be
    finite and strictly positive.

    Both fields must use the same split-adjustment basis.

    See Also
    --------
    DaysToCover : EWMA-smoothed days to cover (short interest / volume).

    References
    ----------
    .. [1] "An investigation of the informational role of short interest in the Nasdaq
        market" The Journal of Finance.
        Desai, H., Ramesh, K., Thiagarajan, S. R., & Balachandran, B. V. (2002).

    .. [2] "Short interest and aggregate stock returns"
        Journal of Financial Economics.
        Rapach, D. E., Ringgenberg, M. C., & Zhou, G. (2016).

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import ShortInterest
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = ShortInterest()
    >>> short_interest = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute short interest.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `short_interest` and `adj_shares_outstanding`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        short_interest : ndarray of shape (n_observations, n_assets)
            Short interest divided by adjusted shares outstanding.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["short_interest", "adj_shares_outstanding"],
            non_negative_or_nan=["short_interest"],
            strictly_positive_or_nan=["adj_shares_outstanding"],
        )
        return safe_divide(
            X["short_interest"], X["adj_shares_outstanding"], fill_value=np.nan
        )
