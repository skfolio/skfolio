"""Base Uncertainty estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import sklearn.base as skb
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

from skfolio.prior import BasePrior
from skfolio.typing import ArrayLike
from skfolio.uncertainty_set._model import (
    CompactCovarianceUncertaintySet,
    UncertaintySet,
)


class BaseMuUncertaintySet(skb.BaseEstimator, ABC):
    """Base class for all Mu Uncertainty Set estimators in `skfolio`.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their `__init__` as explicit keyword
    arguments (no `*args` or `**kwargs`).
    """

    uncertainty_set_: UncertaintySet
    prior_estimator_: BasePrior

    @abstractmethod
    def __init__(self, prior_estimator: BasePrior | None = None):
        self.prior_estimator = prior_estimator

    def get_metadata_routing(self):
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            prior_estimator=self.prior_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    @abstractmethod
    def fit(self, X: ArrayLike, y=None, **fit_params):
        pass


class BaseCovarianceUncertaintySet(skb.BaseEstimator, ABC):
    """Base class for all Covariance Uncertainty Set estimators in `skfolio`.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their `__init__` as explicit keyword
    arguments (no `*args` or `**kwargs`).
    """

    uncertainty_set_: UncertaintySet | CompactCovarianceUncertaintySet
    prior_estimator_: BasePrior

    @abstractmethod
    def __init__(self, prior_estimator: BasePrior | None = None):
        self.prior_estimator = prior_estimator

    def _validate_X_y(self, X: ArrayLike, y: ArrayLike | None = None):
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
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            prior_estimator=self.prior_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    @abstractmethod
    def fit(self, X: ArrayLike, y=None, **fit_params):
        pass
