"""LedoitWolf Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy.typing as npt
import sklearn.covariance as skc

from skfolio.moments.covariance._base import BaseCovariance


class LedoitWolf(BaseCovariance, skc.LedoitWolf):
    """LedoitWolf Covariance Estimator.

    Ledoit-Wolf is a particular form of shrinkage, where the shrinkage
    coefficient is computed using O. Ledoit and M. Wolf's formula as
    described in [1]_.

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

    block_size : int, default=1000
        Size of blocks into which the covariance matrix will be split
        during its Ledoit-Wolf estimation. This is purely a memory
        optimization and does not affect results.

    nearest : bool, default=True
        If this is set to True, the covariance is replaced by the nearest covariance
        matrix that is positive definite and with a Cholesky decomposition than can be
        computed. The variance is left unchanged.
        A covariance matrix that is not positive definite often occurs in high
        dimensional problems. It can be due to multicollinearity, floating-point
        inaccuracies, or when the number of observations is smaller than the number of
        assets. For more details, see :func:`~skfolio.utils.stats.cov_nearest`.
        The default is `True`.

    higham : bool, default=False
        If this is set to True, the Higham (2002) algorithm is used to find the
        nearest PD covariance, otherwise the eigenvalues are clipped to a threshold
        above zeros (1e-13). The default is `False` and uses the clipping method as the
        Higham algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iterations of the Higham (2002) algorithm.
        The default value is `100`.

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

    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)

    where mu = trace(cov) / n_features
    and shrinkage is given by the Ledoit and Wolf formula (see References)

    References
    ----------
    .. [1]  "A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices".
        Ledoit and Wolf, Journal of Multivariate Analysis, Volume 88, Issue 2.
        February 2004, pages 365-41.
    """

    def __init__(
        self,
        store_precision=True,
        assume_centered=False,
        block_size=1000,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        skc.LedoitWolf.__init__(
            self,
            store_precision=store_precision,
            assume_centered=assume_centered,
            block_size=block_size,
        )

    def fit(self, X: npt.ArrayLike, y=None) -> "LedoitWolf":
        """Fit the Ledoit-Wolf shrunk covariance model to X.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : LedoitWolf
          Fitted estimator.
        """
        skc.LedoitWolf.fit(self, X)
        self._set_covariance(self.covariance_)
        return self
