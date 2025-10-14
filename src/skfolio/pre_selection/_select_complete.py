"""pre-selection SelectComplete module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.utils.validation as skv


class SelectComplete(skf.SelectorMixin, skb.BaseEstimator):
    """
    Transformer to select assets with complete data across the entire observation
    period.

    This transformer removes assets (columns) that have missing values (NaNs) at the
    beginning or end of the period.

    This transformer is especially useful for financial datasets where assets
    (e.g., stocks, bonds) may have data gaps due to late inception (assets that started
    trading later), early expiry or default (assets that stopped trading before the
    end of the period).

    If missing values are not at the beginning or end but occur between non-missing
    values, the asset is not removed unless `drop_assets_with_internal_nan` is set to
    `True`.

    Parameters
    ----------
    drop_assets_with_internal_nan : bool, default=False
        If set to True, assets with missing values (NaNs) that appear between
        non-missing values (i.e., internal NaNs) will also be removed. By default,
        only assets with leading or trailing NaNs are removed.

    Attributes
    ----------
    to_keep_ : ndarray of shape (n_assets, )
       Boolean array indicating which assets are remaining.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skfolio.pre_selection import SelectComplete
    >>> X = pd.DataFrame({
    ...     'asset1': [np.nan, np.nan, 2, 3, 4],    # Starts late (inception)
    ...     'asset2': [1, 2, 3, 4, 5],         # Complete data
    ...     'asset3': [1, 2, 3, np.nan, 5], # Missing values within data
    ...     'asset4': [1, 2, 3, 4, np.nan]      # Ends early (expiration)
    ... })
    >>> selector = SelectComplete()
    >>> selector.fit_transform(X)
     array([[ 1.,  1.],
            [ 2.,  2.],
            [ 3.,  3.],
            [ 4., nan],
            [ 5.,  5.]])
    >>> selector = SelectComplete(drop_assets_with_internal_nan=True)
    >>> selector.fit_transform(X)
     array([[1.],
           [2.],
           [3.],
           [4.],
           [5.]])
    """

    to_keep_: np.ndarray

    def __init__(self, drop_assets_with_internal_nan: bool = False):
        self.drop_assets_with_internal_nan = drop_assets_with_internal_nan

    def fit(self, X: npt.ArrayLike, y=None) -> "SelectComplete":
        """Run the SelectComplete transformer and get the appropriate assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SelectComplete
            Fitted estimator.
        """
        # Validate by allowing NaNs
        X = skv.validate_data(self, X, ensure_all_finite="allow-nan")

        if self.drop_assets_with_internal_nan:
            # Identify columns with any NaNs
            self.to_keep_ = ~np.isnan(X).any(axis=0)
        else:
            # Identify columns with no leading or trailing NaNs
            self.to_keep_ = ~np.isnan(X[0, :]) & ~np.isnan(X[-1, :])

        return self

    def _get_support_mask(self) -> np.ndarray:
        skv.check_is_fitted(self)
        return self.to_keep_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
