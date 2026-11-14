"""Mutual information estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipy.stats as sct
import sklearn.metrics as skmc
import sklearn.utils.validation as skv

from skfolio.distance._base import BaseDistance
from skfolio.typing import ArrayLike
from skfolio.utils.stats import NBinsMethod, n_bins_freedman, n_bins_knuth


class MutualInformation(BaseDistance):
    r"""Mutual Information estimator.

    In information theory, the mutual information is a measure of the mutual dependence
    between variables.
    The related distance metric is called the variation of information.

    For two random variables X and Y, the mutual information I(X,Y) is defined as:

    .. math:: I(X,Y) = H(X) + H(Y) - H(X,Y)

    with H(X) and H(Y) the marginal entropies and H(X,Y) the joint entropy.

    The related distance metric known as the  variation of information is defined as:

    .. math:: d(X,Y) = H(X,Y) - I(X,Y) =  H(X) + H(Y) - 2 \times I(X,Y)

    and its normalization as:

    .. math:: D(X,Y) = \frac{d(X,Y)}{H(X,Y)} = \frac{H(X) + H(Y) - 2 \times I(X,Y)}{H(X) + H(Y) - I(X,Y)}

    Parameters
    ----------
    n_bins_method : NBinsMethod, default=NBinsMethod.FREEDMAN
        Method to compute the number of bins for the contingency matrix estimation used
        for the computation of the mutual information.
        Possible values are:

            * FREEDMAN (`default`)
            * KNUTH

    n_bins : int, optional
        Instead of using `n_bins_method`, you can directly specify the number of bins
        with `n_bins`.

    normalize : bool, default=True
        If this is set to True, the variation of information is normalized.
        The default is `True`.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.
    """

    def __init__(
        self,
        n_bins_method: NBinsMethod = NBinsMethod.FREEDMAN,
        n_bins: int | None = None,
        normalize: bool = True,
    ) -> None:
        self.n_bins_method = n_bins_method
        self.n_bins = n_bins
        self.normalize = normalize

    def fit(self, X: ArrayLike, y: None = None) -> MutualInformation:
        """Fit the Mutual Information estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : MutualInformation
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        n_assets = X.shape[1]
        if self.n_bins is None:
            match self.n_bins_method:
                case NBinsMethod.FREEDMAN:
                    n_bins_func = n_bins_freedman
                case NBinsMethod.KNUTH:
                    n_bins_func = n_bins_knuth
                case _:
                    raise ValueError(f"n_bins_method {self.n_bins_method} is not valid")
            n_bins_list = [n_bins_func(x=X[:, i]) for i in range(n_assets)]
        else:
            n_bins_list = [self.n_bins] * n_assets

        corr = np.full((n_assets, n_assets), np.nan)
        dist = corr.copy()
        for i, j in zip(*np.triu_indices(n_assets), strict=True):
            n_bins = max(n_bins_list[i], n_bins_list[j])
            x_i = X[:, i]
            x_j = X[:, j]
            contingency = np.histogram2d(x_i, x_j, bins=n_bins)[0]
            mutual_information = skmc.mutual_info_score(
                None, None, contingency=contingency
            )
            entropy_x = sct.entropy(np.histogram(x_i, n_bins)[0])
            entropy_y = sct.entropy(np.histogram(x_j, n_bins)[0])
            if self.normalize:
                corr[i, j] = mutual_information / min(entropy_x, entropy_y)
                dist[i, j] = max(
                    0.0,
                    (entropy_x + entropy_y - 2 * mutual_information)
                    / (entropy_x + entropy_y - mutual_information),
                )
            else:
                corr[i, j] = mutual_information
                dist[i, j] = max(0.0, entropy_x + entropy_y - 2 * mutual_information)
            corr[j, i] = corr[i, j]
            dist[j, i] = dist[i, j]
        self.codependence_ = corr
        self.distance_ = dist
        return self
