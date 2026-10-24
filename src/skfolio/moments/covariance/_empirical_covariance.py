"""Empirical Covariance Estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import numbers
from typing import ClassVar

import numpy as np
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.typing import ArrayLike
from skfolio.utils.tools import (
    _normalize_sample_weight,
    _validate_sample_weight,
    apply_window_size,
)


class EmpiricalCovariance(BaseCovariance):
    r"""Empirical Covariance estimator.

    Parameters
    ----------
    window_size : int, optional
        Window size. The model is fitted on the last `window_size` observations.
        The default (`None`) is to use all the data.

    ddof : int, default=1
        Normalization is by `(n_observations - ddof)`.
        Note that `ddof=1` will return the unbiased estimate, and `ddof=0`
        will return the simple average. The default value is `1`.
        With `sample_weight`, the weighted second moment is divided by
        :math:`1 - \text{ddof} \sum_i w_i^2`, where :math:`w_i` are the weights
        rescaled to sum to one. This matches the unweighted result for uniform
        weights. Use `ddof=0` when the weights are probabilities defining the
        return distribution.

    assume_centered : bool, default=False
        If False (default), the data are mean-centered before computing the covariance.
        This is the standard behavior when working with raw returns where the mean is
        not guaranteed to be zero.
        If True, the estimator assumes the input data are already centered. Use this
        when you know the returns have zero mean, such as pre-demeaned data or
        regression residuals.

    nearest : bool, default=True
        If this is set to True, the covariance is replaced by the nearest covariance
        matrix that is positive definite and with a Cholesky decomposition that can be
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

    # Preserve unweighted results in existing metadata-routing pipelines.
    # TODO: Remove WARN after an announced transition to explicit weight requests.
    __metadata_request__fit: ClassVar[dict[str, str]] = {"sample_weight": skm.WARN}

    def __init__(
        self,
        window_size: int | None = None,
        ddof: int = 1,
        assume_centered: bool = False,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ) -> None:
        super().__init__(
            assume_centered=assume_centered,
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.ddof = ddof

    def fit(
        self,
        X: ArrayLike,
        y: None = None,
        sample_weight: ArrayLike | None = None,
    ) -> EmpiricalCovariance:
        """Fit the empirical covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        sample_weight : array-like of shape (n_observations,), optional
            Relative observation weights. Must be finite and nonnegative, and are
            rescaled to sum to one. When `window_size` is set, the weights of the
            last `window_size` observations are used. If None (default), all
            observations have equal weight.
            With metadata routing enabled, weights routed through a parent are
            ignored with a warning by default. Call
            `set_fit_request(sample_weight=True)` to use them, or `False` to
            ignore them silently.

        Returns
        -------
        self : EmpiricalCovariance
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        n_observations_full = X.shape[0]
        X = apply_window_size(X, window_size=self.window_size)

        n_observations, _ = X.shape

        if not isinstance(self.ddof, numbers.Integral) or self.ddof < 0:
            raise ValueError(f"ddof must be a non-negative integer, got {self.ddof}")
        if self.ddof >= n_observations:
            raise ValueError(
                "ddof must be strictly less than the number of observations, "
                f"got ddof={self.ddof} and n_observations={n_observations}"
            )

        if sample_weight is not None:
            sample_weight = _validate_sample_weight(
                sample_weight, n_observations=n_observations_full
            )
            sample_weight = apply_window_size(
                sample_weight, window_size=self.window_size
            )
            if not sample_weight.any():
                raise ValueError(
                    "sample_weight must contain at least one positive weight."
                )
            weights = _normalize_sample_weight(sample_weight)
            sum_squared_weights = float(weights @ weights)
            correction = 1.0 - int(self.ddof) * sum_squared_weights
            # Rounding can leave a correction of a few eps where it is exactly zero.
            if correction <= n_observations * np.finfo(float).eps:
                raise ValueError(
                    "sample_weight gives too few effective observations for "
                    f"ddof={self.ddof}: the effective number of observations "
                    f"1 / sum(w**2) = {1.0 / sum_squared_weights:.6g} must be "
                    "strictly greater than ddof."
                )
            if self.assume_centered:
                self.location_ = np.zeros(X.shape[1])
                deviations = X
            else:
                self.location_ = weights @ X
                deviations = X - self.location_
            # Square-root weighting lets NumPy use a symmetric matrix product.
            scaled = deviations * np.sqrt(weights)[:, np.newaxis]
            covariance = scaled.T @ scaled / correction
        elif self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
            covariance = (X.T @ X) / (n_observations - self.ddof)
        else:
            self.location_ = X.mean(axis=0)
            covariance = np.atleast_2d(np.cov(X, rowvar=False, ddof=self.ddof))

        self._set_covariance(covariance)
        return self
