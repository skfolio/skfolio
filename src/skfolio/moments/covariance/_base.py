"""Base Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import sklearn.base as skb

from skfolio.exceptions import NonPositiveVarianceError
from skfolio.utils.stats import cov_nearest


class BaseCovariance(skb.BaseEstimator, ABC):
    """Base class for all covariance estimators in `skfolio`.

    Parameters
    ----------
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

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    covariance_: np.ndarray

    @abstractmethod
    def __init__(
        self,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ):
        self.nearest = nearest
        self.higham = higham
        self.higham_max_iteration = higham_max_iteration

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None):
        pass

    def _sanity_check(self, covariance: np.ndarray) -> None:
        """Perform a sanity check on the covariance matrix by verifying that all
        diagonal elements are strictly positive.
        The goal is to early detect corrupted asset data (with zero variance) that
        would lead to optimizations errors.
        """
        cond = np.diag(covariance) < 1e-15
        if np.any(cond):
            corrupted_assets = list(np.argwhere(cond).flatten())
            detail = "assets indices"
            if hasattr(self, "feature_names_in_"):
                corrupted_assets = list(self.feature_names_in_[corrupted_assets])
                detail = "assets"
            raise NonPositiveVarianceError(
                f"The following {detail} have a non positive variance:"
                f" {corrupted_assets}"
            )

    def _set_covariance(self, covariance: np.ndarray) -> None:
        """Perform checks, convert to nearest PSD if specified and saves the covariance.

        Parameters
        ----------
        covariance : array-like of shape (n_assets, n_assets)
            Estimated covariance matrix to be stored.
        """
        self._sanity_check(covariance)
        if self.nearest:
            covariance = cov_nearest(
                covariance,
                higham=self.higham,
                higham_max_iteration=self.higham_max_iteration,
                warn=True,
            )
        # set covariance
        self.covariance_ = covariance
