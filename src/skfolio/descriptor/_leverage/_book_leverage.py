"""Book leverage descriptor."""

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


class BookLeverage(BaseDescriptor, stateless=True):
    r"""Book leverage descriptor.

    Computes the proportion of total book capital financed by debt:

    .. math::

        \text{book_leverage}(t) =
        \frac{\text{total_debt}(t)}
             {\text{total_debt}(t) + \text{book_equity}(t)}

    Book leverage measures financial risk through the lens of the capital structure: the
    fraction of a firm's total invested capital (debt plus common equity) that comes
    from creditors rather than common shareholders [1]_.

    NaNs are allowed as missing observations and propagate to the output.
    Non-missing `total_debt` and `book_equity` values must be finite.

    This form is preferred over the debt-to-equity ratio (:math:`D / E`) because the two
    are monotonically related (:math:`D / E = \text{book_leverage} / (1 - \text{book_leverage})`)
    but book leverage is bounded in :math:`[0, 1]` for healthy firms, producing
    well-behaved cross-sectional distributions that do not require aggressive
    winsorization.


    `book_equity` is common shareholders' equity (excluding preferred stock and
    minority interest). When it is negative (e.g., firms with accumulated losses
    exceeding paid-in capital), the denominator `total_debt + book_equity` may remain
    positive, become zero or turn negative:

    - **Denominator > 0 and book_equity < 0**: the ratio exceeds 1. The firm is extremely
      leveraged, with debt exceeding total book capital. The value is a valid distress
      signal and is preserved in the output.
    - **Denominator <= 0**: the ratio is undefined or negative, which is economically
      meaningless. These observations are masked to NaN.

    This differs from :class:`ReturnOnEquity`, where any negative equity makes the
    concept meaningless. Here, a ratio above 1 carries real information about financial
    risk.

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
    .. [1] "Capital structure decisions: which factors are reliably important?"
        Financial Management. Frank, M. Z., & Goyal, V. K. (2009).

    See Also
    --------
    DebtToAssets : Leverage relative to total assets.
    MarketLeverage : Leverage as a fraction of total market capital.

    Examples
    --------
    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import BookLeverage
    >>>
    >>> X = make_synthetic_characteristics()
    >>>
    >>> descriptor = BookLeverage()
    >>> book_leverage = descriptor.fit_transform(X)
    """

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute book leverage ratios.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `total_debt` and `book_equity`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        book_leverage : ndarray of shape (n_observations, n_assets)
            Book leverage ratio, with NaN where total debt plus book equity is not
            positive.
        """
        validate_asset_panel(
            self,
            X,
            required_fields=["total_debt", "book_equity"],
            finite_or_nan=["total_debt", "book_equity"],
        )

        total_debt = X["total_debt"]
        book_equity = X["book_equity"]

        denominator = total_debt + book_equity

        book_leverage = safe_divide(total_debt, denominator, fill_value=np.nan)

        # Mask negative denominator to Nan: see docstring
        book_leverage = np.where(denominator > 0, book_leverage, np.nan)

        return book_leverage
