"""Validation utilities for cross-sectional data."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

import numpy as np
import sklearn.utils.validation as skv
from sklearn.utils._tags import get_tags

from skfolio.containers import AssetPanel, AssetPanelView
from skfolio.typing import AnyArray, ArrayLike, FloatArray

__all__ = ["validate_asset_panel", "validate_cross_sectional_data"]


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

    cs_weights : array-like of shape (n_observations, n_assets), optional
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


def validate_asset_panel(
    _estimator,
    /,
    asset_panel: AssetPanel | AssetPanelView,
    required_fields: list[str] | None = None,
    reserved_fields: list[str] | None = None,
    finite_or_nan: list[str] | None = None,
    finite_when_active: list[str] | None = None,
    strictly_positive_or_nan: list[str] | None = None,
    strictly_positive_when_active: list[str] | None = None,
    non_negative_or_nan: list[str] | None = None,
    reset: bool = True,
    copy: bool = False,
) -> AssetPanel | AssetPanelView:
    """Validate an AssetPanel and set estimator metadata attributes.

    This function validates that the panel contains required fields, doesn't contain
    reserved ones and sets standard metadata attributes on the estimator.

    Parameters
    ----------
    _estimator : estimator instance
        The estimator on which to set validation attributes.

    asset_panel : AssetPanel or AssetPanelView
        AssetPanel. AssetPanel already validates that all fields have consistent shapes
        and consistent masks, allowing this validation to be lightweight.

    required_fields : list of str, optional
        Fields that must be present in the AssetPanel. If any are missing, a ValueError
        is raised.

    reserved_fields : list of str, optional
        Fields that must NOT be present in the AssetPanel. These are typically names the
        estimator will create internally. If any are found, a ValueError is raised.

    finite_or_nan : list of str, optional
        Fields whose values must be finite or NaN.

    finite_when_active : list of str, optional
        Fields whose values must be finite wherever `active_mask` is True. This is
        typically used for fields like "market_cap" that must be forward-filled for
        holidays before constructing the AssetPanel.

    strictly_positive_or_nan : list of str, optional
        Fields whose values must be strictly positive and finite, or NaN.

    strictly_positive_when_active : list of str, optional
        Fields whose values must be strictly positive and finite wherever
        `active_mask` is True.

    non_negative_or_nan : list of str, optional
        Fields whose values must be non-negative and finite, or NaN.

    reset : bool, default=True
        If True, sets metadata attributes on the estimator:
        - `asset_names_`: array of asset identifiers
        - `n_assets_`: number of assets

        If False, validates that existing attributes match the panel.

    copy : bool, default=False
        If True, returns a shallow copy of the panel (new dict containers with shared
        arrays). Use this when the estimator needs to add fields without mutating the
        user's input.

    Returns
    -------
    AssetPanel or AssetPanelView
        The validated panel, or a shallow copy if `copy=True`.

    Raises
    ------
    TypeError
        If asset_panel is not an AssetPanel or AssetPanelView.

    ValueError
        If required fields are missing, reserved fields are present, a
        field validation rule is violated or (when reset=False) panel doesn't match
        stored metadata.
    """
    # Normalize empty lists to None.
    if required_fields is not None and len(required_fields) == 0:
        required_fields = None
    if reserved_fields is not None and len(reserved_fields) == 0:
        reserved_fields = None
    if finite_or_nan is not None and len(finite_or_nan) == 0:
        finite_or_nan = None
    if finite_when_active is not None and len(finite_when_active) == 0:
        finite_when_active = None
    if strictly_positive_or_nan is not None and len(strictly_positive_or_nan) == 0:
        strictly_positive_or_nan = None
    if (
        strictly_positive_when_active is not None
        and len(strictly_positive_when_active) == 0
    ):
        strictly_positive_when_active = None
    if non_negative_or_nan is not None and len(non_negative_or_nan) == 0:
        non_negative_or_nan = None

    if not isinstance(asset_panel, (AssetPanel, AssetPanelView)):
        raise TypeError(
            f"Must be an AssetPanel or AssetPanelView, got {type(asset_panel).__name__}"
        )

    field_names = list(asset_panel.keys())
    asset_names = asset_panel.asset_names

    # Check consistency with previous fit (if reset=False).
    if not reset:
        if not np.array_equal(_estimator.asset_names_, asset_names):
            raise ValueError(
                f"asset_names don't match. Expected {list(_estimator.asset_names_)}, "
                f"got {list(asset_names)}."
            )

    # Check reserved fields.
    if reserved_fields is not None:
        reserved = set(reserved_fields) & set(field_names)
        if reserved:
            raise ValueError(
                f"Reserved fields must be removed or renamed: {sorted(reserved)}."
            )

    # Check required fields.
    if required_fields is not None:
        missing = set(required_fields) - set(field_names)
        if missing:
            raise ValueError(
                f"Required fields are missing: {sorted(missing)}. "
                f"Available: {sorted(field_names)}."
            )

    # Check finite-or-NaN constraints.
    if finite_or_nan is not None:
        _check_fields_exist(finite_or_nan, field_names, "finite_or_nan")
        for field in finite_or_nan:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = np.isinf(values)
            else:
                bad = np.isinf(values).any(axis=2)
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains infinite values '
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must contain finite values or NaN.'
                )

    # Check finite-when-active constraints.
    if finite_when_active is not None:
        _check_fields_exist(finite_when_active, field_names, "finite_when_active")
        for field in finite_when_active:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = ~np.isfinite(values) & asset_panel.active_mask
            else:
                # 3-D field: check all components
                bad = (~np.isfinite(values)).any(axis=2) & asset_panel.active_mask
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains NaN/inf for active assets '
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must be finite wherever `active_mask` is '
                    f'True. Forward-fill "{field}" for holidays '
                    f'or set `active_mask` to False until the first finite "{field}".'
                )

    # Check strictly-positive-or-NaN constraints.
    if strictly_positive_or_nan is not None:
        _check_fields_exist(
            strictly_positive_or_nan, field_names, "strictly_positive_or_nan"
        )
        for field in strictly_positive_or_nan:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = ~np.isnan(values) & (~np.isfinite(values) | (values <= 0))
            else:
                bad = (~np.isnan(values) & (~np.isfinite(values) | (values <= 0))).any(
                    axis=2
                )
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains non-positive or infinite values '
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must contain strictly positive finite values or NaN.'
                )

    # Check strictly-positive-when-active constraints.
    if strictly_positive_when_active is not None:
        _check_fields_exist(
            strictly_positive_when_active,
            field_names,
            "strictly_positive_when_active",
        )
        for field in strictly_positive_when_active:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = (~np.isfinite(values) | (values <= 0)) & asset_panel.active_mask
            else:
                bad = (~np.isfinite(values) | (values <= 0)).any(
                    axis=2
                ) & asset_panel.active_mask
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains non-finite or non-positive values for active '
                    f"assets "
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must be strictly positive and finite wherever '
                    f"`active_mask` is True."
                )

    # Check non-negative-or-NaN constraints.
    if non_negative_or_nan is not None:
        _check_fields_exist(non_negative_or_nan, field_names, "non_negative_or_nan")
        for field in non_negative_or_nan:
            values = asset_panel[field]
            if values.ndim == 2:
                bad = ~np.isnan(values) & (~np.isfinite(values) | (values < 0))
            else:
                bad = (~np.isnan(values) & (~np.isfinite(values) | (values < 0))).any(
                    axis=2
                )
            if bad.any():
                bad_obs = _bad_observation(bad)
                raise ValueError(
                    f'Field "{field}" contains negative values or infinite values '
                    f"(first at observation index {bad_obs}). "
                    f'"{field}" must contain non-negative finite values or NaN.'
                )

    # Set estimator metadata attributes.
    if reset:
        _estimator.asset_names_ = np.asarray(asset_names)
        _estimator.n_assets_ = len(asset_names)

    return asset_panel.copy() if copy else asset_panel


def _check_fields_exist(
    fields: list[str], field_names: list[str], arg_name: str
) -> None:
    """Check that all validation rule fields exist in the panel."""
    for field in fields:
        if field not in field_names:
            raise ValueError(
                f'{arg_name} lists "{field}", but that field is not in the '
                f"AssetPanel. Available: {sorted(field_names)}."
            )


def _bad_observation(bad: AnyArray) -> int:
    """Return the first observation index containing a validation failure."""
    return int(np.where(bad.any(axis=1))[0][0])
