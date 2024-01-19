"""Base Covariance Estimators."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
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
    nearest : bool, default=False
        If this is set to True, the covariance is replaced by the nearest covariance
        matrix that is positive definite and with a Cholesky decomposition than can be
        computed. The variance is left unchanged. A covariance matrix is in theory PSD.
        However, due to floating-point inaccuracies, we can end up with a covariance
        matrix that is slightly non-PSD or where Cholesky decomposition is failing.
        This often occurs in high dimensional problems.
        For more details, see :func:`~skfolio.units.stats.cov_nearest`.
        The default is `False`.

    higham : bool, default=False
        If this is set to True, the Higham & Nick (2002) algorithm is used to find the
        nearest PSD covariance, otherwise the eigenvalues are clipped to a threshold
        above zeros (1e-13). The default is `False` and use the clipping method as the
        Higham & Nick algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iteration of the Higham & Nick (2002) algorithm.
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
        nearest: bool = False,
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
            )
        # set covariance
        self.covariance_ = covariance
