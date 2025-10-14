"""Pre-selection DropCorrelated module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.utils.validation as skv


class DropCorrelated(skf.SelectorMixin, skb.BaseEstimator):
    """Transformer for dropping highly correlated assets.

    Simply removing all correlation pairs above the threshold will remove more assets
    than necessary and a naive sequential removal is suboptimal and depends on the
    initial assets ordering.

    Let's suppose X,Y,Z are three random variables with corr(X,Y) and corr(X,Z) above
    the threshold and corr(Y,Z) below.
    The first approach would remove X,Y,Z and the second approach would remove either
    Y and Z or X depending on the initial ordering.

    To avoid these shortcomings, we implement the below algorithm:

        * Step 1: select all correlation pairs above the threshold.
        * Step 2: sort all the selected correlation pairs from highest to lowest.
        * Step 3: for each pair, if none of the two assets has been removed, keep the
          asset with the lowest average correlation against the other assets.

    Parameters
    ----------
    threshold : float, default=0.95
        Correlation threshold. The default value is `0.95`.

    absolute : bool, default=False
        If this is set to True, we take the absolute value of the correlation. This has
        for effect to also include negatively correlated assets.

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

    def __init__(self, threshold: float = 0.95, absolute: bool = False):
        self.threshold = threshold
        self.absolute = absolute

    def fit(self, X: npt.ArrayLike, y=None):
        """Run the correlation transformer and get the appropriate assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : DropCorrelated
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        if not -1 <= self.threshold <= 1:
            raise ValueError("`threshold` must be between -1 and 1")

        n_assets = X.shape[1]
        corr = np.corrcoef(X.T)
        mean_corr = corr.mean(axis=0)

        triu_idx = np.triu_indices(n_assets, 1)

        # select all correlation pairs above the threshold
        selected_idx = np.argwhere(corr[triu_idx] > self.threshold).flatten()

        # sort all the selected correlation pairs from highest to lowest
        selected_idx = selected_idx[np.argsort(-corr[triu_idx][selected_idx])]

        # for each pair, if none of the two assets has been removed, keep the asset with
        # the lowest average correlation with other assets
        to_remove = set()
        for idx in selected_idx:
            i, j = triu_idx[0][idx], triu_idx[1][idx]
            if i not in to_remove and j not in to_remove:
                if mean_corr[i] > mean_corr[j]:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
        self.to_keep_ = ~np.isin(np.arange(n_assets), list(to_remove))
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_
