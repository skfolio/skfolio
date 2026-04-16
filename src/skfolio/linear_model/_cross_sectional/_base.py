"""Base class for cross-sectional linear model estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from skfolio.utils.validation import validate_cross_sectional_data


class BaseCSLinearModel(BaseEstimator, RegressorMixin, ABC):
    """Base class for all cross-sectional linear model estimators.

    This abstract base class defines the common interface for cross-sectional linear
    model estimators that fit one linear model per observation across a set of assets.
    Subclasses are responsible for implementing `fit` and for setting the fitted
    attributes used by `predict` and `score`.

    Parameters
    ----------
    fit_intercept : bool, default=False
        Whether to calculate the intercept for each observation. If set to False, no
        intercept will be used in calculations.

    Attributes
    ----------
    coef_ : ndarray of shape (n_observations, n_features)
        Estimated coefficients for each observation.

    intercept_ : ndarray of shape (n_observations,)
        intercept for each observation. Set to zeros if `fit_intercept=False`.

    n_features_in_ : int
        Number of features seen during `fit`.

    n_valid_assets_ : ndarray of shape (n_observations,)
        Number of assets that participated in estimation (those with positive
        weight) for each observation.
    """

    coef_: np.ndarray
    intercept_: np.ndarray
    n_features_in_: int
    n_valid_assets_: np.ndarray

    def __init__(self, fit_intercept: bool = False) -> None:
        self.fit_intercept = fit_intercept

    @abstractmethod
    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        cs_weights: npt.ArrayLike | None = None,
    ):
        """Fit one cross-sectional linear model per observation.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets, n_features)
            Feature tensor. The first axis indexes observations, the second
            axis indexes assets, and the third axis indexes features.

        y : array-like of shape (n_observations, n_assets)
            Target values aligned with `X`.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Cross-sectional weights for each `(observation, asset)` pair.

        Returns
        -------
        self : BaseCSLinearModel
            Fitted estimator.
        """
        pass

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        r"""Predict using the cross-sectional linear model.

        For each observation :math:`t` and asset :math:`i`, the prediction is
        the systematic part; realized outcomes satisfy
        :math:`y_{ti} = \hat{y}_{ti} + \epsilon_{ti}` with residual
        :math:`\epsilon_{ti}`. The prediction is

        .. math::

            \hat{y}_{ti} = X_{ti}^{T} \beta_t + \beta_{t,0}

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets, n_features)
            Feature tensor used for prediction.
            The observation and feature axes must match those seen during
            `fit`. The asset axis may differ.

        Returns
        -------
        y_pred : ndarray of shape (n_observations, n_assets)
            Predicted values.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_", "n_features_in_"])

        X = validate_cross_sectional_data(self, X=X, reset=False)

        n_observations = X.shape[0]
        expected_n_observations = self.coef_.shape[0]
        if n_observations != expected_n_observations:
            raise ValueError(
                f"X has {n_observations} observations but model was fitted with "
                f"{expected_n_observations} observations."
            )

        return (X @ self.coef_[..., None]).squeeze(-1) + self.intercept_[:, None]

    def score(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        cs_weights: npt.ArrayLike | None = None,
    ) -> float:
        r"""Return the mean coefficient of determination across observations.

        The coefficient of determination :math:`R^2` is computed independently
        for each observation and then averaged. For observation :math:`t`:

        .. math::

            R^2_t = 1 - \frac{\sum_i w_{ti}(y_{ti} - \hat{y}_{ti})^2}
                             {\sum_i w_{ti}(y_{ti} - \bar{y}_t)^2}

        where :math:`\bar{y}_t` is the weighted mean of :math:`y` for
        observation :math:`t`.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets, n_features)
            Feature tensor on which to evaluate the model.

        y : array-like of shape (n_observations, n_assets)
            Target values aligned with `X`.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Asset weights for computing weighted :math:`R^2` scores.
            If None, all assets are given equal weight. Pairs with zero weight
            are excluded from the score. Pairs with positive weight must have
            finite `X` and finite `y`.

        Returns
        -------
        score : float
            Mean :math:`R^2` across all observations with finite values.
            Returns NaN if no observations have valid :math:`R^2` values.
        """
        check_is_fitted(self, "coef_")

        X, y, cs_weights = validate_cross_sectional_data(
            self,
            X=X,
            y=y,
            cs_weights=cs_weights,
            reset=False,
        )

        positive_weight = _validate_positive_weight_pairs(X, y, cs_weights)
        y_pred = self.predict(X)
        is_valid = positive_weight & np.isfinite(y_pred)
        weights_masked = np.where(is_valid, cs_weights, 0.0)
        y_masked = np.where(is_valid, y, 0.0)
        y_pred_masked = np.where(is_valid, y_pred, 0.0)

        n_observations = y.shape[0]
        rss = (weights_masked * (y_masked - y_pred_masked) ** 2).sum(axis=1)

        y_mean = np.full(n_observations, np.nan, dtype=float)
        valid_obs = weights_masked.sum(axis=1) > 0
        if np.any(valid_obs):
            y_mean[valid_obs] = np.average(
                y_masked[valid_obs], axis=1, weights=weights_masked[valid_obs]
            )

        ss_tot = (weights_masked * (y_masked - y_mean[:, None]) ** 2).sum(axis=1)

        r2 = 1.0 - np.divide(
            rss,
            ss_tot,
            out=np.full(n_observations, np.nan, dtype=float),
            where=ss_tot > 0,
        )
        return float(np.nanmean(r2))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags


def _validate_positive_weight_pairs(
    X: np.ndarray,
    y: np.ndarray,
    cs_weights: np.ndarray,
) -> np.ndarray:
    """Validate positive-weight pairs and return the fit mask.

    Each `(observation, asset)` pair with positive `cs_weights` must have all
    features in `X` finite and `y` finite.
    """
    positive_weight = cs_weights > 0
    invalid_positive_weight = positive_weight & (
        ~np.all(np.isfinite(X), axis=2) | ~np.isfinite(y)
    )
    if np.any(invalid_positive_weight):
        raise ValueError(
            "Each `(observation, asset)` pair with positive `cs_weights` must "
            "have all features in `X` finite and `y` finite."
        )
    return positive_weight
