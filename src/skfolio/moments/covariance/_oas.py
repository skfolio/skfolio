"""Oracle Approximating Shrinkage Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy.typing as npt
import sklearn.covariance as skc

from skfolio.moments.covariance._base import BaseCovariance


class OAS(BaseCovariance, skc.OAS):
    """Oracle Approximating Shrinkage Estimator as proposed in [1]_.

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

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.

    precision_ : ndarray of shape (n_assets, n_assets)
        Estimated pseudo inverse matrix.
        (stored only if store_precision is True)

    shrinkage_ : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Notes
    -----
    The regularised covariance is:

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features),

    where mu = trace(cov) / n_features and shrinkage is given by the OAS formula
    (see [1]_).

    The shrinkage formulation implemented here differs from Eq. 23 in [1]_. In
    the original article, formula (23) states that 2/p (p being the number of
    features) is multiplied by Trace(cov*cov) in both the numerator and
    denominator, but this operation is omitted because for a large p, the value
    of 2/p is so small that it doesn't affect the value of the estimator.

    References
    ----------
    .. [1] "Shrinkage algorithms for MMSE covariance estimation".
        Chen, Y., Wiesel, A., Eldar, Y. C., & Hero, A. O.
        IEEE Transactions on Signal Processing, 58(10), 5016-5029, 2010.
    """

    def __init__(
        self,
        store_precision=True,
        assume_centered=False,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        skc.OAS.__init__(
            self,
            store_precision=store_precision,
            assume_centered=assume_centered,
        )

    def fit(self, X: npt.ArrayLike, y=None) -> "OAS":
        """Fit the Oracle Approximating Shrinkage covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : OAS
          Fitted estimator.
        """
        skc.OAS.fit(self, X)
        self._set_covariance(self.covariance_)
        return self
