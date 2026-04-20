"""Validation utilities for cross-sectional data."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

import numpy as np
import sklearn.utils.validation as skv
from sklearn.utils._tags import get_tags

from skfolio.typing import ArrayLike, FloatArray

__all__ = ["validate_cross_sectional_data"]


def validate_cross_sectional_data(
    _estimator,
    /,
    X: ArrayLike,
    y: ArrayLike | Literal["no_validation"] | None = "no_validation",
    cs_weights: ArrayLike | None = None,
    *,
    reset: bool = True,
    copy: bool = False,
) -> FloatArray | tuple[FloatArray, FloatArray, FloatArray]:
    """Validate cross-sectional data.

    This helper follows the design of scikit-learn's `validate_data` and is specialized
    for cross-sectional arrays with shape `(n_observations, n_assets, n_features)`.
    The target `y` and the optional cross-sectional weights must have shape
    `(n_observations, n_assets)`.

    Missing values encoded as NaN are allowed in `X` and `y`. Infinite values
    are rejected. The weights must be finite and non-negative.

    Parameters
    ----------
    _estimator : estimator instance
        Estimator on which `n_features_in_` is set or checked.

    X : array-like of shape (n_observations, n_assets, n_features)
        Input feature tensor.

    y : array-like of shape (n_observations, n_assets), None, or "no_validation", default="no_validation"
        Target values.

        - `"no_validation"`: skip target validation and return only the validated `X`.
          This is the default and is used by methods like `predict` that only need `X`.
        - `None`: skip target validation, but check the estimator's
          `target_tags.required` tag. If the tag is `True`, a `ValueError` is raised.
        - array-like: validate as a numeric 2D array.

    cs_weights : array-like of shape (n_observations, n_assets) or None, default=None
        Cross-sectional weights for each (observation, asset) pair.

        - `None` with `y` provided: return a matrix of ones.
        - `None` with `y` skipped: weights are not validated.
        - array-like: validate as a finite, non-negative 2D array.

    reset : bool, default=True
        If `True`, set `n_features_in_` on the estimator. If `False`, check consistency
        with the stored number of features.

    copy : bool, default=False
        If `True`, force a copy of the validated arrays.

    Returns
    -------
    X_validated : ndarray of shape (n_observations, n_assets, n_features)
        Validated feature tensor, returned alone when `y` is `"no_validation"` or `None`.

    X_validated, y_validated, cs_weights_validated : tuple of ndarrays
        Validated `X`, `y`, and weights, returned when `y` is an array-like.

    Raises
    ------
    ValueError
        If `X` is not a 3D array.

    ValueError
        If `y` is `None` and the estimator's `target_tags.required` tag is `True`.

    ValueError
        If `y` is provided but its shape does not match the first two dimensions of `X`.

    ValueError
        If `cs_weights` is provided without `y`.

    ValueError
        If `cs_weights` contains negative or non-finite values.

    ValueError
        If `cs_weights` shape does not match the first two dimensions of `X`.

    ValueError
        If `reset` is `False` and the number of features in `X` differs from
        `n_features_in_`.
    """
    # X validation
    X_validated = skv.check_array(
        X,
        dtype="numeric",
        ensure_all_finite="allow-nan",
        ensure_2d=False,
        allow_nd=True,
        copy=copy,
        estimator=_estimator,
        input_name="X",
    )
    if X_validated.ndim != 3:
        raise ValueError(
            "X must be a 3D array of shape (n_observations, n_assets, n_features). "
            f"Got shape {X_validated.shape}."
        )

    n_observations, n_assets, n_features = X_validated.shape
    expected_shape = (n_observations, n_assets)

    # n_features_in_ management
    if reset:
        _estimator.n_features_in_ = n_features
    elif hasattr(_estimator, "n_features_in_"):
        if n_features != _estimator.n_features_in_:
            raise ValueError(
                f"X has {n_features} features, but "
                f"{_estimator.__class__.__name__} is expecting "
                f"{_estimator.n_features_in_} features as input."
            )

    skip_y = y is None or (isinstance(y, str) and y == "no_validation")

    # Check estimator tags when y is explicitly None
    if y is None:
        tags = get_tags(_estimator)
        if tags.target_tags.required:
            raise ValueError(
                f"{_estimator.__class__.__name__} requires y to be passed, "
                "but the target y is None."
            )

    # X-only path (predict or unsupervised)
    if skip_y:
        if cs_weights is not None:
            raise ValueError(
                "cs_weights cannot be provided without y. "
                "Pass both y and cs_weights, or neither."
            )
        return X_validated

    # y validation
    y_validated = skv.check_array(
        y,
        dtype="numeric",
        ensure_all_finite="allow-nan",
        ensure_2d=True,
        copy=copy,
        estimator=_estimator,
        input_name="y",
    )
    if y_validated.shape != expected_shape:
        raise ValueError(
            f"y must have shape {expected_shape} to match X with shape "
            f"{X_validated.shape}, got {y_validated.shape}."
        )

    # cs_weights validation
    if cs_weights is None:
        w_validated = np.ones(expected_shape, dtype=np.float64)
    else:
        w_validated = skv.check_array(
            cs_weights,
            dtype="numeric",
            ensure_all_finite=True,
            ensure_non_negative=True,
            ensure_2d=True,
            copy=copy,
            estimator=_estimator,
            input_name="cs_weights",
        )
        if w_validated.shape != expected_shape:
            raise ValueError(
                "cs_weights must have shape "
                f"(n_observations, n_assets)={expected_shape}; "
                f"got {w_validated.shape}."
            )

    return X_validated, y_validated, w_validated
