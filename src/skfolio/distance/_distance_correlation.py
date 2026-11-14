"""Distance correlation estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipy.spatial.distance as scd
import sklearn.utils.validation as skv

from skfolio.distance._base import BaseDistance
from skfolio.typing import ArrayLike, FloatArray


class DistanceCorrelation(BaseDistance):
    """Distance Correlation estimator.

    Distance Correlation was introduced by Szekely [1]_ to capture non-linear
    dependencies.

    Parameters
    ----------
    threshold : float, default=0.5
        Distance correlation threshold.

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Measuring and testing independence by correlation of distances"
        Gábor J. Szekely , 2005
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    @staticmethod
    def _dcorr(x: FloatArray, y: FloatArray) -> float:
        """Calculate the distance correlation between two variables."""
        x = scd.squareform(scd.pdist(x.reshape(-1, 1)))
        y = scd.squareform(scd.pdist(y.reshape(-1, 1)))
        x = x - x.mean(axis=0)[np.newaxis, :] - x.mean(axis=1)[:, np.newaxis] + x.mean()
        y = y - y.mean(axis=0)[np.newaxis, :] - y.mean(axis=1)[:, np.newaxis] + y.mean()
        value = np.sqrt((x * y).sum()) / np.sqrt(
            np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())
        )
        return value

    def fit(self, X: ArrayLike, y: None = None) -> DistanceCorrelation:
        """Fit the Distance Correlation estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : DistanceCorrelation
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        n_assets = X.shape[1]
        corr = np.ones((n_assets, n_assets))
        # TODO: parallelize
        for i, j in zip(*np.triu_indices(n_assets, 1), strict=True):
            corr[i, j] = self._dcorr(x=X[:, i], y=X[:, j])
            corr[j, i] = corr[i, j]
        self.codependence_ = corr
        self.distance_ = np.sqrt(np.clip(1 - self.codependence_, a_min=0.0, a_max=1.0))
        return self
