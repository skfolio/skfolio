"""Cross-sectional weighted least squares regression."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from skfolio.linear_model._cross_sectional._base import (
    BaseCSLinearModel,
    _validate_positive_weight_pairs,
)
from skfolio.utils.validation import validate_cross_sectional_data


class CSLinearRegression(BaseCSLinearModel):
    r"""Cross-sectional weighted least squares regression.

    This estimator fits one weighted least squares regression per
    observation across the asset cross-section. The implementation is fully
    vectorized and is designed for panel data whose asset universe may vary
    over time.

    The model solves the weighted least squares problem independently for each
    observation:

    .. math::

        \beta_t = \arg\min_{\beta} \sum_{i=1}^{n_{\text{assets}}}
                  w_{ti} (y_{ti} - X_{ti}^T \beta)^2

    where :math:`t` denotes the observation, :math:`i` denotes the asset,
    :math:`w_{ti}` are the cross-section weights, and :math:`X_{ti}` is the
    feature vector for asset :math:`i` at observation :math:`t`.

    The cross-sectional weights must be finite and non-negative. A pair with
    zero weight is excluded from estimation for that observation. This is the
    intended way to represent inactive pairs such as assets outside the
    estimation universe, listed or delisted assets, or pairs with missing (NaNs)
    data.

    For each `(observation, asset)` pair:

    - If `cs_weights > 0`, all features in `X` and `y` must be finite.
    - If `cs_weights == 0`, the pair is excluded from estimation and `X` and
      `y` may be finite or missing (NaNs).

    Parameters
    ----------
    fit_intercept : bool, default=False
        Whether to calculate the intercept for each observation.
        If set to False, no intercept will be used in calculations.

    Attributes
    ----------
    coef_ : ndarray of shape (n_observations, n_features)
        Estimated coefficients for each observation.

    intercept_ : ndarray of shape (n_observations,)
        Intercept for each observation. Set to zeros if `fit_intercept=False`.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    n_valid_assets_ : ndarray of shape (n_observations,)
        Number of assets that participated in estimation (those with positive weight)
        for each observation.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.linear_model import CSLinearRegression
    >>>
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(3, 5, 2)
    >>> y = rng.randn(3, 5)
    >>>
    >>> model = CSLinearRegression()
    >>> model.fit(X, y)
    CSLinearRegression()
    >>>
    >>> model.intercept_.shape
    (3,)
    >>> model.coef_.shape
    (3, 2)
    >>> model.predict(X).shape
    (3, 5)
    >>> model.score(X, y)
    0.6353...
    """

    def __init__(self, fit_intercept: bool = False) -> None:
        super().__init__(fit_intercept=fit_intercept)

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        cs_weights: npt.ArrayLike | None = None,
    ) -> CSLinearRegression:
        """Fit the cross-sectional regression model.

        Estimates regression coefficients independently for each observation by
        solving weighted least squares problems across assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets, n_features)
            Training data.  3D array where the first axis indexes
            observations, the second axis indexes assets, and the third
            axis indexes features.

        y : array-like of shape (n_observations, n_assets)
            Target values.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Cross-sectional weights for each `(observation, asset)` pair.

            - Must be finite and non-negative.
            - Pairs with zero weight are excluded from estimation.
            - If None, all pairs receive unit weight.

        Returns
        -------
        self : CSLinearRegression
            Fitted estimator.

        Notes
        -----
        Each `(observation, asset)` pair with positive `cs_weights` must have
        finite `X` and finite `y`. Pairs with zero weight are excluded from
        estimation and may contain missing values.
        """
        X, y, cs_weights = validate_cross_sectional_data(
            self,
            X=X,
            y=y,
            cs_weights=cs_weights,
            reset=True,
            copy=False,  # arrays are copied in np.where below
        )
        n_observations, _, n_features = X.shape

        is_valid = _validate_positive_weight_pairs(X, y, cs_weights)

        X = np.where(is_valid[..., None], X, 0.0)
        y = np.where(is_valid, y, 0.0)
        cs_weights = np.where(is_valid, cs_weights, 0.0)

        if self.fit_intercept:
            w_sums = cs_weights.sum(axis=1, keepdims=True)

            y_mean = np.zeros(n_observations, dtype=X.dtype)
            np.einsum("tn,tn->t", y, cs_weights, out=y_mean, optimize=True)
            np.divide(y_mean, w_sums[:, 0], out=y_mean, where=w_sums[:, 0] > 0)
            y -= y_mean[:, None]

            X_mean = np.zeros((n_observations, n_features), dtype=X.dtype)
            np.einsum("tni,tn->ti", X, cs_weights, out=X_mean, optimize=True)
            np.divide(X_mean, w_sums, out=X_mean, where=w_sums > 0)
            X -= X_mean[:, None, :]

        np.sqrt(cs_weights, out=cs_weights)
        X *= cs_weights[..., None]
        y *= cs_weights
        XtWX = X.transpose(0, 2, 1) @ X
        XtWy = (X.transpose(0, 2, 1) @ y[..., None]).squeeze(-1)

        try:
            coef = np.linalg.solve(XtWX, XtWy[:, :, None]).squeeze(-1)
        except np.linalg.LinAlgError:
            coef = np.einsum("tij,tj->ti", np.linalg.pinv(XtWX), XtWy)

        if self.fit_intercept:
            intercept = y_mean - np.einsum("tk,tk->t", coef, X_mean)
        else:
            intercept = np.zeros(n_observations, dtype=float)

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_valid_assets_ = is_valid.sum(axis=1).astype(int)

        return self
