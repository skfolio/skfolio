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
from skfolio.typing import ArrayLike, FloatArray
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
        weights. See the Notes section.

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

    Notes
    -----
    With `sample_weight`, the weights are rescaled to sum to one, so only their
    relative sizes matter. The location is the weighted mean (or zero when
    `assume_centered=True`) and the covariance is

    .. math::
        \Sigma = \frac{\sum_i w_i (x_i - \mu)(x_i - \mu)^T}{1 - \text{ddof} \sum_i w_i^2}.

    With `ddof=0` this is the covariance of the distribution that puts
    probability :math:`w_i` on observation :math:`x_i`. With `ddof=1` it applies
    the correction for reliability weights, which makes the estimate unbiased when
    the weights describe the relative reliability of i.i.d. observations. When the
    weights are arbitrary scenario probabilities, this correction does not by
    itself make the estimate unbiased, and `ddof=0` is usually the natural choice.
    """

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

        Returns
        -------
        self : EmpiricalCovariance
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        if sample_weight is not None:
            sample_weight = _validate_sample_weight(
                sample_weight, n_observations=X.shape[0]
            )
            sample_weight = apply_window_size(
                sample_weight, window_size=self.window_size
            )
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
            self._set_covariance(self._weighted_covariance(X, sample_weight))
            return self

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

    def _weighted_covariance(
        self, X: FloatArray, sample_weight: FloatArray
    ) -> FloatArray:
        """Compute the weighted covariance and set `location_`."""
        if not sample_weight.any():
            raise ValueError("sample_weight must contain at least one positive weight.")
        weights = _normalize_sample_weight(sample_weight)
        correction = 1.0 - self.ddof * float(weights @ weights)
        if correction <= 0:
            raise ValueError(
                "sample_weight gives too few effective observations for "
                f"ddof={self.ddof}: the effective number of observations "
                f"1 / sum(w**2) = {1.0 / float(weights @ weights):.6g} must be "
                "strictly greater than ddof."
            )
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
            deviations = X
        else:
            self.location_ = weights @ X
            deviations = X - self.location_
        return (deviations.T * weights) @ deviations / correction
