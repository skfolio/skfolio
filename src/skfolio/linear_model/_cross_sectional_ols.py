"""Cross-sectional OLS regression."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from skfolio.utils.validation import validate_cross_sectional_data


class CrossSectionalOLS(BaseEstimator, RegressorMixin):
    r"""Cross-sectional weighted ordinary least squares regression.

    This estimator performs independent weighted OLS regressions for each observation
    across multiple assets and their factors. It is fully vectorized and handles missing
    data gracefully, making it suitable for financial applications where assets may be
    listed and delisted over time.

    The model solves the weighted least squares problem independently for each
    observation:

    .. math::

        \\beta_t = \\arg\\min_{\\beta} \\sum_{i=1}^{n_{assets}} w_{ti} (y_{ti} - X_{ti}^T \\beta)^2

    where :math:`t` denotes the observation, :math:`i` denotes the asset,
    :math:`w_{ti}` are the sample weights, and :math:`X_{ti}` is the feature
    vector (factors) for asset :math:`i` at observation :math:`t`.

    **NaN Policy**
    For a given (observation, asset):
      - If some but not all factors in X are NaN: error is raised
      - If all factors in X are NaN: the entire asset is dropped for that observation;
        y and sample_weight must also be NaN
      - If X has no NaNs: y and sample_weight must be finite

    This policy allows handling asset listing/delisting without introducing survival
    bias.

    Parameters
    ----------
    fit_intercept : bool, default=False
        Whether to calculate the intercept for each observation.
        If set to False, no intercept will be used in calculations.

    Attributes
    ----------
    coef_ : ndarray of shape (n_observations, n_factors)
        Estimated coefficients for each observation.

    intercept_ : ndarray of shape (n_observations,)
        Independent term (intercept) for each observation.
        Set to zeros if `fit_intercept=False`.

    n_features_in_ : int
        Number of features (factors) seen during :term:`fit`.

    n_used_ : ndarray of shape (n_observations,)
        Number of valid (non-NaN, positive weight) assets used in the
        regression for each observation.

    feature_names_in_ : ndarray of shape (`n_features_in_`,), optional
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    assets_ : ndarray of shape (n_assets,), optional
        Asset names extracted from DataFrame input with MultiIndex columns.
        Only present when `X` in `fit` was a DataFrame.

    factors_ : ndarray of shape (n_factors,), optional
        Factor names extracted from DataFrame input with MultiIndex columns.
        Only present when `X` in `fit` was a DataFrame.

    index_ : Index, optional
        Observation index (e.g., dates) extracted from DataFrame input.
        Only present when `X` in `fit` was a DataFrame.

    See Also
    --------
    sklearn.linear_model.LinearRegression : Ordinary least squares Linear Regression.

    Notes
    -----
    The estimator uses the pseudo-inverse (SVD-based) to solve the normal equations,
    which handles rank-deficient design matrices gracefully.

    Each observation is fitted independently, making this estimator suitable for
    panel data where the cross-sectional relationship may vary over time.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.linear_model import CrossSectionalOLS
    >>> # Generate sample data: 3 observations, 5 assets, 2 factors
    >>> rng = np.random.RandomState(42)
    >>> X = rng.randn(3, 5, 2)
    >>> y = rng.randn(3, 5)
    >>> # Fit the model
    >>> model = CrossSectionalOLS()
    >>> model.fit(X, y)
    CrossSectionalOLS()
    >>> # Coefficients shape: (n_observations, n_factors)
    >>> model.coef_.shape
    (3, 2)
    >>> # Predictions
    >>> predictions = model.predict(X)
    >>> predictions.shape
    (3, 5)
    """

    def __init__(self, fit_intercept: bool = False) -> None:
        self.fit_intercept = fit_intercept

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> CrossSectionalOLS:
        """Fit cross-sectional regression model.

        Estimates regression coefficients independently for each observation by
        solving weighted least squares problems across assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets, n_factors)
            Training data containing factor values.

            - If array-like: 3D array where the first axis indexes observations
              (e.g., time periods), the second axis indexes assets, and the third
              axis indexes factors
            - If DataFrame: Must have a MultiIndex with two levels representing
              (asset, factor) pairs in columns, and observations in the index

        y : array-like of shape (n_observations, n_assets)
            Target values (e.g., asset returns).

            - If array-like: 2D array where rows are observations and columns are assets
            - If DataFrame: Index must match `X` observations, columns must be assets

        sample_weight : array-like of shape (n_observations, n_assets), optional
            Individual weights for each (observation, asset) pair.

            - Must be non-negative.
            - Assets with zero weight are excluded from the regression for that
              observation.
            - If None (default), all samples are given equal weight.

        Returns
        -------
        self : object
            Fitted estimator.

        Notes
        -----
        The method handles missing data by excluding (observation, asset) pairs where:

        - Any factor in `X` is NaN (for that pair)
        - The target `y` is NaN
        - The weight is zero, NaN, or infinite

        If all factors for an asset are NaN at a given observation, that asset is
        simply excluded from the regression for that observation without raising an
        error.
        """
        # Validate and coerce inputs; stores n_features_in_, assets_, factors_, index_
        X, y = validate_cross_sectional_data(self, X=X, y=y, reset=True)
        n_observations, n_assets, n_factors = X.shape

        # Validate and prepare sample weights
        if sample_weight is None:
            weights = np.ones((n_observations, n_assets), dtype=float)
        else:
            weights = check_array(
                sample_weight,
                dtype="numeric",
                ensure_all_finite=False,
                ensure_2d=True,
                input_name="sample_weight",
            )
            if weights.shape != (n_observations, n_assets):
                raise ValueError(
                    f"sample_weight must have shape (n_observations, n_assets)="
                    f"{(n_observations, n_assets)}; got {weights.shape}."
                )
            # Check for negative weights (must be non-negative)
            if np.any(weights < 0):
                idx = np.argwhere(weights < 0)[0]
                raise ValueError(
                    f"sample_weight contains negative value at "
                    f"(observation={idx[0]}, asset={idx[1]})."
                )
            # Check for non-finite weights (NaN/Inf not allowed)
            if np.any(~np.isfinite(weights)):
                idx = np.argwhere(~np.isfinite(weights))[0]
                raise ValueError(
                    f"sample_weight contains non-finite value at "
                    f"(observation={idx[0]}, asset={idx[1]})."
                )

        # Build validity mask: shape (n_observations, n_assets)
        is_valid = np.all(np.isfinite(X), axis=2) & np.isfinite(y) & (weights > 0)

        # Zero-out invalid entries
        X = np.where(is_valid[..., None], X, 0.0)
        y = np.where(is_valid, y, 0.0)
        weights = np.where(is_valid, weights, 0.0)

        # Center data if fitting intercept
        if self.fit_intercept:
            # Compute sum of weights per observation
            weight_sums = weights.sum(axis=1, keepdims=True)  # (n_observations, 1)

            # Weighted mean of y: shape (n_observations,)
            y_mean = np.zeros(n_observations, dtype=X.dtype)
            np.einsum("tn,tn->t", y, weights, out=y_mean, optimize=True)
            np.divide(
                y_mean, weight_sums[:, 0], out=y_mean, where=weight_sums[:, 0] > 0
            )
            y -= y_mean[:, None]

            # Weighted mean of X: shape (n_observations, n_factors)
            X_mean = np.zeros((n_observations, n_factors), dtype=X.dtype)
            np.einsum("tni,tn->ti", X, weights, out=X_mean, optimize=True)
            np.divide(X_mean, weight_sums, out=X_mean, where=weight_sums > 0)
            X -= X_mean[:, None, :]

            del weight_sums

        # Pre-multiply by sqrt(weights) for numerical stability
        np.sqrt(weights, out=weights)
        X *= weights[..., None]
        y *= weights
        XtWX = X.transpose(0, 2, 1) @ X
        XtWy = (X.transpose(0, 2, 1) @ y[..., None]).squeeze(-1)

        # Try vectorized solve first (10-100x faster than pinv for full rank)
        try:
            # np.linalg.solve expects shape (..., n, n) @ (..., n, 1)
            coef = np.linalg.solve(XtWX, XtWy[:, :, None]).squeeze(-1)
        except np.linalg.LinAlgError:
            # Fall back to pseudo-inverse for rank-deficient case (matches sklearn)
            coef = np.einsum("tij,tj->ti", np.linalg.pinv(XtWX), XtWy)

        # Compute intercept
        if self.fit_intercept:
            intercept = y_mean - np.einsum("tk,tk->t", coef, X_mean)
        else:
            intercept = np.zeros(n_observations, dtype=float)

        self.coef_ = coef
        self.intercept_ = intercept
        self.n_used_ = is_valid.sum(axis=1).astype(int)

        return self

    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        r"""Predict using the cross-sectional regression model.

        Applies the per-observation linear models to compute predictions:
        :math:`\\hat{y}_{ti} = X_{ti}^T \\beta_t + \\alpha_t` where :math:`t` denotes
        the observations and :math:`i` denotes the assets.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets, n_factors)
            Samples to predict.

            - The number of observations must match the number used during `fit`.
            - The number of factors must equal `n_features_in_`.
            - The number of assets can vary but typically matches the fitted model.

        Returns
        -------
        y_pred : ndarray of shape (n_observations, n_assets)
            Predicted target values for each (observation, asset) pair.
        """
        check_is_fitted(self, attributes=["coef_", "intercept_", "n_features_in_"])

        X = validate_cross_sectional_data(self, X=X, reset=False)

        # Check compatibility with fitted coefficients
        n_observations = X.shape[0]
        expected_n_observations = self.coef_.shape[0]
        if n_observations != expected_n_observations:
            raise ValueError(
                f"X has {n_observations} observations but model was fitted with "
                f"{expected_n_observations} observations."
            )

        # Predict: y_pred = X @ beta + alpha
        return (X @ self.coef_[..., None]).squeeze(-1) + self.intercept_[:, None]

    def score(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> float:
        r"""Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is computed independently for each
        observation and then averaged. For observation :math:`t`:

        .. math::

            R^2_t = 1 - \\frac{\\sum_i w_{ti}(y_{ti} - \\hat{y}_{ti})^2}{\\sum_i w_{ti}(y_{ti} - \\bar{y}_t)^2}

        where :math:`\\bar{y}_t` is the weighted mean of :math:`y` for observation
        :math:`t`.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets, n_factors)
            Test samples. Must have the same shape structure as the training data.

        y : array-like of shape (n_observations, n_assets)
            True values for `X`.

        sample_weight : array-like of shape (n_observations, n_assets), optional
            Sample weights for computing weighted R² scores.
            If None, all samples are given equal weight.

        Returns
        -------
        score : float
            Mean :math:`R^2` coefficient across all observations with finite values.
            Returns NaN if no observations have valid :math:`R^2` values.

        Notes
        -----
        The :math:`R^2` score can be negative for individual observations if the model
        performs worse than a horizontal line (constant prediction). The best possible
        score is 1.0.

        Observations with undefined :math:`R^2` (e.g., where the denominator is zero)
        are excluded from the mean calculation.
        """
        check_is_fitted(self, attributes=["coef_", "n_features_in_"])

        X, y = validate_cross_sectional_data(self, X=X, y=y, reset=False)
        n_observations, n_assets = y.shape

        y_pred = self.predict(X)

        # Prepare weights for scoring
        if sample_weight is None:
            weights = np.ones_like(y, dtype=float)
        else:
            weights = np.asarray(sample_weight, dtype=float)
            if not (weights.ndim == 2 and weights.shape == (n_observations, n_assets)):
                raise ValueError(
                    f"sample_weight must be None or of shape (n_observations, n_assets)="
                    f"{(n_observations, n_assets)}; got {weights.shape}."
                )
            # Sanitize: set invalid/negative weights to zero
            weights = np.where(np.isfinite(weights) & (weights >= 0), weights, 0.0)

        # Identify valid cells for scoring
        is_valid = np.isfinite(y) & np.isfinite(y_pred) & (weights > 0)
        weights_masked = np.where(is_valid, weights, 0.0)
        y_masked = np.where(is_valid, y, 0.0)
        y_pred_masked = np.where(is_valid, y_pred, 0.0)

        # Compute weighted R2 per observation
        # R2 = 1 - SS_res / SS_tot

        # Weighted mean of y per observation (only over valid values)
        # Handle case where all weights are zero for an observation
        # Set those observations to NaN
        y_mean = np.full(n_observations, np.nan, dtype=float)
        valid_obs = weights_masked.sum(axis=1) > 0
        if np.any(valid_obs):
            y_mean[valid_obs] = np.average(
                y_masked[valid_obs], axis=1, weights=weights_masked[valid_obs]
            )

        # Weighted sum of squared residuals
        residuals_sq = (y_masked - y_pred_masked) ** 2
        ss_res = (weights_masked * residuals_sq).sum(axis=1)

        # Weighted total sum of squares
        y_deviations_sq = (y_masked - y_mean[:, None]) ** 2
        ss_tot = (weights_masked * y_deviations_sq).sum(axis=1)

        # R2 per observation (NaN where ss_tot == 0)
        r2_per_obs = 1.0 - np.divide(
            ss_res,
            ss_tot,
            out=np.full(n_observations, np.nan, dtype=float),
            where=ss_tot > 0,
        )

        return np.nanmean(r2_per_obs)
