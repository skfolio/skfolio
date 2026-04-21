"""Empirical Covariance Estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import numbers

import numpy as np
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.typing import ArrayLike
from skfolio.utils.tools import apply_window_size


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

    assume_centered : bool, default=False
        If False (default), the data are mean-centered before computing the covariance.
        This is the standard behavior when working with raw returns where the mean is
        not guaranteed to be zero.
        If True, the estimator assumes the input data are already centered. Use this
        when you know the returns have zero mean, such as pre-demeaned data or
        regression residuals.

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

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.
        Use for compatibility with scikit-learn Covariance estimators and for
        mahalanobis and score methods.

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
        assume_centered: bool = False,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        super().__init__(
            assume_centered=assume_centered,
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.ddof = ddof

    def fit(self, X: ArrayLike, y=None) -> EmpiricalCovariance:
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
        X = apply_window_size(X, window_size=self.window_size)

        n_observations, _ = X.shape

        if not isinstance(self.ddof, numbers.Integral) or self.ddof < 0:
            raise ValueError(f"ddof must be a non-negative integer, got {self.ddof}")
        if self.ddof >= n_observations:
            raise ValueError(
                "ddof must be strictly less than the number of observations, "
                f"got ddof={self.ddof} and n_observations={n_observations}"
            )

        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
            covariance = (X.T @ X) / (n_observations - self.ddof)
        else:
            self.location_ = X.mean(axis=0)
            covariance = np.cov(X, rowvar=False, ddof=self.ddof)
            # np.cov returns a scalar when X has a single column (one asset).
            if covariance.ndim == 0:
                covariance = covariance.reshape(1, 1)

        self._set_covariance(covariance)
        return self
