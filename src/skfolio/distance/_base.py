"""Base Distance Estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import sklearn.base as skb

from skfolio.typing import ArrayLike, FloatArray


class BaseDistance(skb.BaseEstimator, ABC):
    """Base class for all distance estimators in skfolio.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword
    arguments (no ``*args`` or ``**kwargs``).

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix.
    """

    codependence_: FloatArray
    distance_: FloatArray

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, X: ArrayLike, y=None) -> BaseDistance:
        """Fit the Distance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BaseDistance
            Fitted estimator.
        """
        pass
