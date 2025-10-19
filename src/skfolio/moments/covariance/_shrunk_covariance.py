"""Shrunk Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy.typing as npt
import sklearn.covariance as skc

from skfolio.moments.covariance._base import BaseCovariance


class ShrunkCovariance(BaseCovariance, skc.ShrunkCovariance):
    """Covariance estimator with shrinkage.

    Read more in `scikit-learn
    <https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html>`_.

    Parameters
    ----------
    store_precision : bool, default=True
        Specify if the estimated precision is stored.

    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False (default), data will be centered before computation.

    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_assets, n_assets)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Notes
    -----
    The regularized covariance is given by:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    """

    def __init__(
        self,
        store_precision=True,
        assume_centered=False,
        shrinkage=0.1,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        skc.ShrunkCovariance.__init__(
            self,
            store_precision=store_precision,
            assume_centered=assume_centered,
            shrinkage=shrinkage,
        )

    def fit(self, X: npt.ArrayLike, y=None) -> "ShrunkCovariance":
        """Fit the shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : ShrunkCovariance
          Fitted estimator.
        """
        skc.ShrunkCovariance.fit(self, X)
        self._set_covariance(self.covariance_)
        return self
