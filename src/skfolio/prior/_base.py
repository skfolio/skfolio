"""Base Prior estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import sklearn.base as skb


# frozen=True with eq=False will lead to an id-based hashing which is needed for
# caching CVX models in Optimization without impacting performance
@dataclass(frozen=True, eq=False)
class ReturnDistribution:
    """Return Distribution dataclass used by the optimization estimators.

    Attributes
    ----------
    mu : ndarray of shape (n_assets,)
        Estimation of the assets expected returns.

    covariance : ndarray of shape (n_assets, n_assets)
        Estimation of the assets covariance matrix.

    returns : ndarray of shape (n_observations, n_assets)
        Estimation of the assets returns.

    cholesky : ndarray, optional
        Lower-triangular Cholesky factor of the covariance. In some cases it is possible
        to obtain a cholesky factor with less dimension compared to the one obtained
        directly by applying the cholesky decomposition to the covariance estimation
        (for example in Factor Models). When provided, this cholesky factor is use in
        some optimizations (for example in mean-variance) to improve performance and
        convergence. The default is `None`.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.
    """

    mu: np.ndarray
    covariance: np.ndarray
    returns: np.ndarray
    cholesky: np.ndarray | None = None
    sample_weight: np.ndarray | None = None


class BasePrior(skb.BaseEstimator, ABC):
    """Base class for all prior estimators in skfolio.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    return_distribution_: ReturnDistribution

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None, **fit_params):
        pass
