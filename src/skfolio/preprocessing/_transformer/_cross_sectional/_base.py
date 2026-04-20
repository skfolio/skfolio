"""Base class for cross-sectional transformers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import sklearn.base as skb
import sklearn.utils.validation as skv
from sklearn.utils.validation import FLOAT_DTYPES

__all__ = ["BaseCSTransformer"]


class BaseCSTransformer(skb.OneToOneFeatureMixin, skb.BaseEstimator, ABC):
    """Base class for all cross-sectional transformers in skfolio.

    Cross-sectional transformers process each observation of a 2D input array using
    values from the same observation only.

    These transformers are stateless. The default `fit` method validates the estimator
    parameters, validates `X`, and records `n_features_in_` for scikit-learn
    compatibility.

    Notes
    -----
    All estimators should specify all the parameters that can be set at the class level
    in their `__init__` as explicit keyword arguments (no `*args` or `**kwargs`).
    """

    @abstractmethod
    def __init__(self):
        """Initialize the transformer."""
        pass

    def _validate_params(self) -> None:
        """Validate estimator-specific parameters."""
        return None

    def fit(
        self,
        X: npt.ArrayLike,
        y=None,
        cs_weights: npt.ArrayLike | None = None,
        cs_groups: npt.ArrayLike | None = None,
    ):
        """Fit the transformer.

        Cross-sectional transformers are stateless and do not learn data-dependent
        parameters. This method validates the estimator parameters, validates `X`, and
        records `n_features_in_` for scikit-learn compatibility.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Input matrix where each row is an observation and each column is an asset.

        y : Ignored
            Not used, present for API consistency by convention.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Optional cross-sectional weights accepted for API consistency with
            `transform`. They are ignored during fitting.

        cs_groups : array-like of shape (n_observations, n_assets), optional
            Optional cross-sectional group labels accepted for API consistency with
            `transform`. They are ignored during fitting.

        Returns
        -------
        self : BaseCSTransformer
            Fitted estimator.
        """
        self._validate_params()
        skv.validate_data(
            self, X, reset=True, dtype=FLOAT_DTYPES, ensure_all_finite="allow-nan"
        )
        return self

    @abstractmethod
    def transform(
        self,
        X: npt.ArrayLike,
        cs_weights: npt.ArrayLike | None = None,
        cs_groups: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """Transform `X` observation by observation.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Input matrix where each row is an observation and each column is an asset.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Optional cross-sectional weights used by the concrete transformer.

        cs_groups : array-like of shape (n_observations, n_assets), optional
            Optional cross-sectional group labels used by the concrete transformer.

        Returns
        -------
        X_transformed : ndarray of shape (n_observations, n_assets)
            Transformed values.
        """
        pass

    def fit_transform(
        self,
        X: npt.ArrayLike,
        y=None,
        cs_weights: npt.ArrayLike | None = None,
        cs_groups: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """Fit to `X` and return the transformed values.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Input matrix where each row is an observation and each column is an asset.

        y : Ignored
            Not used, present for API consistency by convention.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Optional cross-sectional weights forwarded to `transform`.

        cs_groups : array-like of shape (n_observations, n_assets), optional
            Optional cross-sectional group labels forwarded to `transform`.

        Returns
        -------
        X_new : ndarray of shape (n_observations, n_assets)
            Transformed array.
        """
        return self.fit(
            X,
            cs_weights=cs_weights,
            cs_groups=cs_groups,
        ).transform(
            X,
            cs_weights=cs_weights,
            cs_groups=cs_groups,
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.requires_fit = False
        return tags
