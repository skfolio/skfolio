"""Cross-sectional regression wrapping a scikit-learn regressor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import sklearn as sk
import sklearn.base as skb
import sklearn.utils.parallel as skp

from skfolio.linear_model._cross_sectional._base import (
    BaseCSLinearModel,
    _validate_positive_weight_pairs,
)
from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.validation import validate_cross_sectional_data


class CSLinearRegressorWrapper(BaseCSLinearModel):
    r"""Cross-sectional regression based on a scikit-learn regressor.

    This estimator wraps a scikit-learn regressor and fits one independent regression
    across assets for each observation. These independent observation-level regressions
    can be fitted in parallel by setting `n_jobs`. The wrapped regressor must define
    `fit_intercept`, implement `fit`, accept a `sample_weight` argument, and expose
    fitted `coef_` and `intercept_` attributes.

    Missing-value handling is driven by `cs_weights` on each `(observation, asset)`
    pair:

    - If `cs_weights > 0`, all features in `X` and `y` must be finite.
    - If `cs_weights == 0`, the pair is excluded from estimation and `X` and `y` may be
      finite or missing.
    - Each observation must retain at least one valid asset after applying `cs_weights`.

    Parameters
    ----------
    regressor : BaseEstimator
        Scikit-learn regressor used at each observation.

    n_jobs : int, default=1
        Number of parallel jobs used to fit the observation-level regressions.

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
    >>> from sklearn.linear_model import HuberRegressor
    >>> from skfolio.linear_model import CSLinearRegressorWrapper
    >>>
    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(3, 5, 2))
    >>> y = rng.normal(size=(3, 5))
    >>> cs_weights = 1.0 + rng.random(size=(3, 5))
    >>>
    >>> model = CSLinearRegressorWrapper(
    ...     regressor=HuberRegressor(fit_intercept=True, max_iter=200)
    ... )
    >>> model.fit(X, y, cs_weights=cs_weights)
    CSLinearRegressorWrapper(...)
    >>>
    >>> model.intercept_.shape
    (3,)
    >>> model.coef_.shape
    (3, 2)
    >>> model.predict(X).shape
    (3, 5)
    >>> model.score(X, y)
    0.4901...

    See Also
    --------
    :class:`~skfolio.linear_model.CSLinearRegression`
    """

    def __init__(self, regressor: skb.BaseEstimator, n_jobs: int = 1) -> None:
        if not hasattr(regressor, "fit_intercept"):
            raise ValueError(
                "CSLinearRegressorWrapper requires the wrapped regressor to "
                "define `fit_intercept`."
            )
        super().__init__(fit_intercept=regressor.fit_intercept)
        self.regressor = regressor
        self.n_jobs = n_jobs

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        cs_weights: ArrayLike | None = None,
    ) -> CSLinearRegressorWrapper:
        """Fit one wrapped regressor per observation.

        Each observation must contain at least one asset with positive weight
        and finite `X` and `y` values.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets, n_features)
            Input feature tensor.

        y : array-like of shape (n_observations, n_assets)
            Target values.

        cs_weights : array-like of shape (n_observations, n_assets), optional
            Cross-sectional weights passed to the wrapped regressor as `sample_weight`.
            If None, all assets receive unit weight.

        Returns
        -------
        self : CSLinearRegressorWrapper
            Fitted estimator.
        """
        X, y, cs_weights = validate_cross_sectional_data(
            self,
            X=X,
            y=y,
            cs_weights=cs_weights,
            reset=True,
            copy=False,
        )

        n_observations = X.shape[0]

        is_valid = _validate_positive_weight_pairs(X, y, cs_weights)
        n_valid_assets = is_valid.sum(axis=1).astype(int)
        if np.any(n_valid_assets == 0):
            raise ValueError(
                "Each observation must contain at least one asset with positive "
                "weight and finite `X` and `y` values."
            )

        results = skp.Parallel(n_jobs=self.n_jobs)(
            skp.delayed(_fit_regressor_for_observation)(
                regressor=sk.clone(self.regressor),
                X=X,
                y=y,
                cs_weights=cs_weights,
                is_valid=is_valid,
                observation_idx=observation_idx,
            )
            for observation_idx in range(n_observations)
        )

        coefs, intercepts = zip(*results, strict=True)
        self.coef_ = np.array(coefs)
        self.intercept_ = np.array(intercepts)
        self.n_valid_assets_ = n_valid_assets

        return self


def _fit_regressor_for_observation(
    regressor: skb.BaseEstimator,
    X: FloatArray,
    y: FloatArray,
    cs_weights: FloatArray,
    is_valid: FloatArray,
    observation_idx: int,
) -> tuple[FloatArray, float]:
    """Fit one cloned regressor for a single observation.

    The selected observation defines one cross-section across assets. Only
    `(observation, asset)` pairs with positive weight and finite `X` and `y`
    values are passed to the wrapped regressor.

    Parameters
    ----------
    regressor : BaseEstimator
        Cloned regressor to fit.

    X : ndarray of shape (n_observations, n_assets, n_features)
        Feature tensor.

    y : ndarray of shape (n_observations, n_assets)
        Target values.

    cs_weights : ndarray of shape (n_observations, n_assets)
        Cross-sectional weights.

    is_valid : ndarray of shape (n_observations, n_assets)
        Boolean mask indicating which `(observation, asset)` pairs are used
        in the fit.

    observation_idx : int
        Observation index to fit.

    Returns
    -------
    coef : ndarray of shape (n_features,)
        Estimated coefficients for the selected observation.

    intercept : float
        Estimated intercept for the selected observation.
    """
    keep = is_valid[observation_idx]
    X_t = X[observation_idx, keep, :]
    y_t = y[observation_idx, keep]
    cs_weights_t = cs_weights[observation_idx, keep]
    regressor.fit(X_t, y_t, sample_weight=cs_weights_t)

    if not hasattr(regressor, "coef_"):
        raise ValueError(
            "CSLinearRegressorWrapper requires the wrapped regressor to expose "
            "`coef_` after `fit`."
        )
    if not hasattr(regressor, "intercept_"):
        raise ValueError(
            "CSLinearRegressorWrapper requires the wrapped regressor to expose "
            "`intercept_` after `fit`."
        )
    return np.asarray(regressor.coef_).copy(), float(regressor.intercept_)
