"""Base Uncertainty estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.prior import BasePrior


# frozen=True with eq=False will lead to an id-based hashing which is needed for
# caching CVX models in Optimization without impacting performance
@dataclass(frozen=True, eq=False)
class UncertaintySet:
    r"""Ellipsoidal uncertainty set dataclass.

    An ellipsoidal uncertainty set is defined by its size :math:`\kappa` and
    shape :math:`S`. Ellipsoidal uncertainty set can be used with both expected returns
    and covariance:

    Expected returns ellipsoidal uncertainty set:

    .. math:: U_{\mu}=\left\{\mu\,|\left(\mu-\hat{\mu}\right)S^{-1}\left(\mu-\hat{\mu}\right)^{T}\leq\kappa^{2}\right\}

    Covariance ellipsoidal uncertainty set:

    .. math:: U_{\Sigma}=\left\{\Sigma\,|\left(\text{vec}(\Sigma)-\text{vec}(\hat{\Sigma})\right)S^{-1}\left(\text{vec}(\Sigma)-\text{vec}(\hat{\Sigma})\right)^{T}\leq k^{2}\,,\,\Sigma\succeq 0\right\}

    Attributes
    ----------
    k : float
        Size of the ellipsoid  :math:`\kappa` that defines the confidence region

    sigma : ndarray of shape (n_assets)
        Shape of the ellipsoid :math:`S`
    """

    k: float
    sigma: np.ndarray


class BaseMuUncertaintySet(skb.BaseEstimator, ABC):
    """Base class for all Mu Uncertainty Set estimators in `skfolio`.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    uncertainty_set_: UncertaintySet
    prior_estimator_: BasePrior

    @abstractmethod
    def __init__(self, prior_estimator: BasePrior | None = None):
        self.prior_estimator = prior_estimator

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            prior_estimator=self.prior_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None, **fit_params):
        pass


class BaseCovarianceUncertaintySet(skb.BaseEstimator, ABC):
    """Base class for all Covariance Uncertainty Set estimators in `skfolio`.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).
    """

    uncertainty_set_: UncertaintySet
    prior_estimator_: BasePrior

    @abstractmethod
    def __init__(self, prior_estimator: BasePrior | None = None):
        self.prior_estimator = prior_estimator

    def _validate_X_y(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        """Validate X and y if provided.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like of shape (n_observations, n_targets), optional
            Price returns of factors or a target benchmark.
            The default is `None`.

        Returns
        -------
        X : ndarray of shape (n_observations, n_assets)
            Validated price returns of the assets.
        y : ndarray of shape (n_observations, n_targets), optional
            Validated price returns of factors or a target benchmark if provided.
        """
        if y is None:
            X = skv.validate_data(self, X)
        else:
            X, y = skv.validate_data(self, X, y, multi_output=True)
        return X, y

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            prior_estimator=self.prior_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y=None, **fit_params):
        pass
