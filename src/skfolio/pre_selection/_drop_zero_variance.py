"""Pre-selection DropZeroVariance module."""

# Copyright (c) 2025
# Author: Vincent Maladiere <maladiere.vincent@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.utils.validation as skv


class DropZeroVariance(skf.SelectorMixin, skb.BaseEstimator):
    """Transformer for dropping assets with near-zero variance.

    On short windows, some assets can experience a near-zero variance, making
    the covariance matrix improper for optimization. This simple transformer drops
    assets whose variance is below some threshold.

    Parameters
    ----------
    threshold : float, default=1e-8
        Minimum variance threshold. The default value is 1e-8. For daily asset returns,
        this value filters out assets whose daily standard deviation is below 1e-4
        (0.01%), which corresponds to an annual standard deviation of approximately
        0.16%, assuming 252 trading days.

    Attributes
    ----------
    to_keep_ : ndarray of shape (n_assets, )
        Boolean array indicating which assets are remaining.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    to_keep_: np.ndarray

    def __init__(self, threshold: float = 1e-8):
        self.threshold = threshold

    def fit(self, X: npt.ArrayLike, y=None):
        """Fit the transformer on some assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : DropZeroVariance
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        if self.threshold < 0:
            raise ValueError(
                f"`threshold` must be higher than 0, got {self.threshold}."
            )

        self.to_keep_ = X.var(axis=0) > self.threshold

        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_
