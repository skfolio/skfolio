"""Base Distance Estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import sklearn.base as skb
import sklearn.utils as sku

from skfolio.typing import ArrayLike, FloatArray, StrArray


class BaseDistance(skb.BaseEstimator, ABC):
    """Base class for all distance estimators in skfolio.

    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their `__init__` as explicit keyword
    arguments (no `*args` or `**kwargs`).

    Attributes
    ----------
    codependence_ : ndarray of shape (n_assets, n_assets)
        Codependence matrix.

    distance_ : ndarray of shape (n_assets, n_assets)
        Distance matrix with rows and columns in the input asset order.
    """

    codependence_: FloatArray
    distance_: FloatArray
    n_features_in_: int
    feature_names_in_: StrArray

    @abstractmethod
    def __init__(self) -> None: ...

    @abstractmethod
    def fit(self, X: ArrayLike, y: None = None) -> BaseDistance:
        """Fit the Distance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets) or (n_assets, n_assets)
            Price returns of the assets, or a covariance matrix when
            `requires_covariance_input` is `True`. Estimators may obtain their data from
            fit metadata and use X only to identify the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : BaseDistance
            Fitted estimator.
        """
        ...

    @property
    def requires_covariance_input(self) -> bool:
        """Whether `X` must contain a covariance matrix.

        Return-consuming distances use the default value of `False`. Subclasses
        consuming covariance override this property. Its value must be available
        before fitting and reflect parameter changes made through `set_params`.
        The `pairwise` input tag is derived from this property.
        """
        return False

    def __sklearn_tags__(self) -> sku.Tags:
        """Declare the input representation required by the distance estimator."""
        tags = super().__sklearn_tags__()
        tags.input_tags.pairwise = self.requires_covariance_input
        return tags
