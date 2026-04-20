"""Base Variance Estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import sklearn.base as skb

from skfolio.typing import ArrayLike, FloatArray


class BaseVariance(skb.BaseEstimator, ABC):
    r"""Base class for all variance estimators in `skfolio`.

    Variance estimators estimate the diagonal elements of a covariance matrix,
    assuming **zero correlation** between assets. This is appropriate when:

    * Estimating **idiosyncratic (specific) risk** in factor models, where residual
      returns are uncorrelated by construction
    * Working with **orthogonalized** or **uncorrelated** return series
    * The full covariance structure is not needed or is constructed separately

    Parameters
    ----------
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
        Estimated variance vector :math:`(\\sigma^2_1, ..., \\sigma^2_n)`.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.
        When `assume_centered=True`, this is zero.
        When `assume_centered=False`, this is the sample mean.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has asset names that are all strings.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    variance_: FloatArray
    location_: FloatArray

    def __init__(self, assume_centered: bool = False):
        self.assume_centered = assume_centered

    @abstractmethod
    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
    ):
        pass
