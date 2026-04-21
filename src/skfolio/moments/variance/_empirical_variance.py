"""Empirical Variance Estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers

import numpy as np
import sklearn.utils.validation as skv

from skfolio.moments.variance._base import BaseVariance
from skfolio.typing import ArrayLike
from skfolio.utils.tools import apply_window_size


class EmpiricalVariance(BaseVariance):
    """Empirical Variance estimator.

    This is the variance-only counterpart of
    :class:`~skfolio.moments.covariance.EmpiricalCovariance`, computing only
    the diagonal elements (variances) and assuming zero correlation. This is
    appropriate when:

    * Estimating **idiosyncratic (specific) risk** in factor models, where residual
      returns are uncorrelated by construction
    * Working with **orthogonalized** or **uncorrelated** return series
    * The full covariance structure is not needed or is constructed separately

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
        If False (default), the data are mean-centered before computing the variance.
        This is the standard behavior when working with raw returns where the mean is
        not guaranteed to be zero.
        If True, the estimator assumes the input data are already centered. Use this
        when you know the returns have zero mean, such as pre-demeaned data or
        regression residuals.

    Attributes
    ----------
    variance_ : ndarray of shape (n_assets,)
        Estimated variance vector.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has asset names that are all strings.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.moments import EmpiricalVariance
    >>> from skfolio.preprocessing import prices_to_returns
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> model = EmpiricalVariance()
    >>> model.fit(X)
    >>> print(model.variance_[:5])
    """

    def __init__(
        self,
        window_size: int | None = None,
        ddof: int = 1,
        assume_centered: bool = False,
    ):
        super().__init__(assume_centered=assume_centered)
        self.window_size = window_size
        self.ddof = ddof

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ) -> EmpiricalVariance:
        """Fit the empirical variance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : EmpiricalVariance
            Fitted estimator.
        """
        X = skv.validate_data(self, X)
        X = apply_window_size(X, window_size=self.window_size)

        n_observations, n_assets = X.shape

        if not isinstance(self.ddof, numbers.Integral) or self.ddof < 0:
            raise ValueError(f"ddof must be a non-negative integer, got {self.ddof}")
        if self.ddof >= n_observations:
            raise ValueError(
                "ddof must be strictly less than the number of observations, "
                f"got ddof={self.ddof} and n_observations={n_observations}"
            )

        if self.assume_centered:
            self.location_ = np.zeros(n_assets)
            self.variance_ = np.sum(X**2, axis=0) / (n_observations - self.ddof)
        else:
            self.location_ = X.mean(axis=0)
            self.variance_ = np.var(X, axis=0, ddof=self.ddof)

        return self
