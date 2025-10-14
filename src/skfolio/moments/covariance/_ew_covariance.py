"""Exponentially Weighted Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy.typing as npt
import pandas as pd
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance


class EWCovariance(BaseCovariance):
    r"""Exponentially Weighted Covariance estimator.

    Estimator of the covariance using the historical exponentially weighted returns.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    alpha : float, default=0.2
       Exponential smoothing factor. The default value is `0.2`.

       :math:`0 < \alpha \leq 1`.

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

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.
    """

    def __init__(
        self,
        window_size: int | None = None,
        alpha: float = 0.2,
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
        self.alpha = alpha

    def fit(self, X: npt.ArrayLike, y=None):
        """Fit the Exponentially Weighted Covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
         Price returns of the assets.

        y : Ignored
          Not used, present for API consistency by convention.

        Returns
        -------
        self : EWCovariance
          Fitted estimator.
        """
        X = skv.validate_data(self, X)
        if self.window_size is not None:
            X = X[-int(self.window_size) :]
        n_observations = X.shape[0]
        covariance = (
            pd.DataFrame(X)
            .ewm(alpha=self.alpha)
            .cov()
            .loc[(n_observations - 1, slice(None)), :]
            .to_numpy()
        )
        self._set_covariance(covariance)
        return self
