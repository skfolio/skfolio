"""Empirical Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance


class EmpiricalCovariance(BaseCovariance):
    """Empirical Covariance estimator.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    ddof : int, default=1
        Normalization is by `(n_observations - ddof)`.
        Note that `ddof=1` will return the unbiased estimate, and `ddof=0`
        will return the simple average. The default value is `1`.

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
        Estimated covariance matrix.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    def __init__(
        self,
        window_size: int | None = None,
        ddof: int = 1,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.ddof = ddof

    def fit(self, X: npt.ArrayLike, y=None) -> "EmpiricalCovariance":
        """Fit the empirical covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EmpiricalCovariance
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        if self.window_size is not None:
            X = X[-int(self.window_size) :]
        covariance = np.cov(X, ddof=self.ddof, rowvar=False)

        if covariance.ndim == 0:
            covariance = covariance.reshape(1, 1)

        self._set_covariance(covariance)
        return self
