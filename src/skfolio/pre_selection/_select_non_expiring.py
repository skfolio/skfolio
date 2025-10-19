"""pre-selection estimators module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Implementation derived from:
# Conway-Yu https://github.com/skfolio/skfolio/discussions/60
# SPDX-License-Identifier: BSD-3-Clause

import datetime as dt

import numpy as np
import pandas as pd
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.utils.validation as skv


class SelectNonExpiring(skf.SelectorMixin, skb.BaseEstimator):
    """
    Transformer to select assets that do not expire within a specified lookahead period
    after the end of the observation period.

    This transformer removes assets (columns) that have expiration dates within a
    given lookahead period from the end of the dataset, allowing only assets that
    remain active beyond this lookahead period to be selected.

    This is useful when an exit strategy is needed before asset expiration, such as
    for bonds or options with known end dates, or when applying WalkForward
    cross-validation. It ensures that assets expiring during the test period are
    excluded, so that only live assets are included in each training and test period.

    Parameters
    ----------
    expiration_dates : dict[str, dt.datetime | pd.Timestamp], optional
        Dictionary with asset names as keys and expiration dates as values.
        Used to check if each asset expires within the date offset.
        Assets with no expiration date will be retained by default.

    expiration_lookahead : pd.offsets.BaseOffset | dt.timedelta, optional
        The lookahead period after the end of the dataset within which assets with
        expiration dates will be removed.

    Attributes
    ----------
    to_keep_ : ndarray of shape (n_assets, )
       Boolean array indicating which assets are remaining.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Notes
    -----
    This transformer only supports DataFrames with a DateTime index.

    Examples
    --------
    >>> import pandas as pd
    >>> import datetime as dt
    >>> from sklearn import set_config
    >>> set_config(transform_output="pandas")
    >>> X = pd.DataFrame(
    ...    {
    ...        'asset1': [1, 2, 3, 4],
    ...        'asset2': [2, 3, 4, 5],
    ...        'asset3': [3, 4, 5, 6],
    ...        'asset4': [4, 5, 6, 7]
    ...    }, index=pd.date_range("2023-01-01", periods=4, freq="D")
    ...)
    >>> expiration_dates = {
    ...    'asset1': pd.Timestamp("2023-01-10"),
    ...    'asset2': pd.Timestamp("2023-01-02"),
    ...    'asset3': pd.Timestamp("2023-01-06"),
    ...    'asset4': dt.datetime(2023, 5, 1)
    ... }
    >>> selector = SelectNonExpiring(
    ...    expiration_dates=expiration_dates,
    ...    expiration_lookahead=pd.DateOffset(days=5)
    ...)
    >>> selector.fit_transform(X)
               asset1  asset4
    2023-01-01      1      4
    2023-01-02      2      5
    2023-01-03      3      6
    2023-01-04      4      7
    """

    to_keep_: np.ndarray

    def __init__(
        self,
        expiration_dates: dict[str, dt.datetime | pd.Timestamp] | None = None,
        expiration_lookahead: pd.offsets.BaseOffset | dt.timedelta | None = None,
    ):
        self.expiration_dates = expiration_dates
        self.expiration_lookahead = expiration_lookahead

    def fit(self, X: pd.DataFrame, y=None) -> "SelectNonExpiring":
        """Run the SelectNonExpiring transformer and get the appropriate assets.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_observations, n_assets)
            Returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SelectNonExpiring
            Fitted estimator.
        """
        _ = skv.validate_data(self, X, ensure_all_finite="allow-nan")

        # Validate by allowing NaNs
        if not hasattr(X, "index") or not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must be a DataFrame with an index of type DatetimeIndex"
            )

        if self.expiration_dates is None:
            raise ValueError("`expiration_dates` must be provided")

        if self.expiration_lookahead is None:
            raise ValueError("`expiration_lookahead` must be provided")

        # Calculate the cutoff date
        end_date = X.index[-1]
        cutoff_date = end_date + self.expiration_lookahead
        self.to_keep_ = np.array(
            [
                self.expiration_dates.get(asset, pd.Timestamp.max) > cutoff_date
                for asset in X.columns
            ]
        )

        return self

    def _get_support_mask(self) -> np.ndarray:
        skv.check_is_fitted(self)
        return self.to_keep_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags
