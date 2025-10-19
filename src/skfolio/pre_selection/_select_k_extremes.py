"""Pre-selection SelectKExtremes module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio.measures import RatioMeasure
from skfolio.population import Population
from skfolio.portfolio import Portfolio


class SelectKExtremes(skf.SelectorMixin, skb.BaseEstimator):
    """Transformer for selecting the `k` best or worst assets.

    Keep the `k` best or worst assets according to a given measure.

    Parameters
    ----------
    k : int, default=10
        Number of assets to select. If `k` is higher than the number of assets, all
        assets are selected.

    measure : Measure, default=RatioMeasure.SHARPE_RATIO
        The :ref:`measure <measures_ref>` used to sort the assets.
        The default is `RatioMeasure.SHARPE_RATIO`.

    highest : bool, default=True
        If this is set to True, the `k` assets with the highest `measure` are selected,
        otherwise it is the `k` lowest.

    Attributes
    ----------
    to_keep_ : ndarray of shape (n_assets, )
       Boolean array indicating which assets are remaining.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.
    """

    to_keep_: np.ndarray

    def __init__(
        self,
        k: int = 10,
        measure: skt.Measure = RatioMeasure.SHARPE_RATIO,
        highest: bool = True,
    ):
        self.k = k
        self.measure = measure
        self.highest = highest

    def fit(self, X: npt.ArrayLike, y=None) -> "SelectKExtremes":
        """Run the SelectKExtremes transformer and get the appropriate assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SelectKExtremes
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        k = int(self.k)
        if k <= 0:
            raise ValueError("`k` must be strictly positive")
        n_assets = X.shape[1]
        # Build a population of single assets portfolio
        population = Population([])
        for i in range(n_assets):
            weights = np.zeros(n_assets)
            weights[i] = 1
            population.append(Portfolio(X=X, weights=weights))

        selected = population.sort_measure(measure=self.measure, reverse=self.highest)[
            :k
        ]
        selected_idx = [x.nonzero_assets_index[0] for x in selected]
        self.to_keep_ = np.isin(np.arange(n_assets), selected_idx)
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_
