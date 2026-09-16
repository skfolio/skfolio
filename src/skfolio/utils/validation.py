"""Validation utilities for cross-sectional data."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, NamedTuple

import numpy as np
import sklearn.utils.validation as skv
from sklearn.utils._tags import get_tags

from skfolio.typing import AnyArray, ArrayLike, FloatArray

if TYPE_CHECKING:
    from skfolio.containers import AssetPanel, AssetPanelView

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
    # Normalize empty lists to None: an empty rule constrains nothing.
    required_fields = required_fields or None
    reserved_fields = reserved_fields or None
    rule_fields = {
        "finite_or_nan": finite_or_nan or None,
        "finite_when_active": finite_when_active or None,
        "strictly_positive_or_nan": strictly_positive_or_nan or None,
        "strictly_positive_when_active": strictly_positive_when_active or None,
        "non_negative_or_nan": non_negative_or_nan or None,
    }

    # Imported here rather than at module scope: `skfolio.containers` imports
    # `skfolio.utils.tools`, so a module-level import would close a package-level
    # cycle between `skfolio.utils` and `skfolio.containers`. Annotations are
    # postponed, so the `TYPE_CHECKING` import above covers the signature.
    from skfolio.containers import AssetPanel, AssetPanelView

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

    # Check the per-field value constraints.
    for rule in _VALUE_RULES:
        fields = rule_fields[rule.arg_name]
        if fields is None:
            continue
        _check_fields_exist(fields, field_names, rule.arg_name)
        for field in fields:
            _check_value_rule(asset_panel, field, rule)

    # Set estimator metadata attributes.
    if reset:
        _estimator.asset_names_ = np.asarray(asset_names)
        _estimator.n_assets_ = len(asset_names)

    return asset_panel.copy() if copy else asset_panel


class _ValueRule(NamedTuple):
    """One per-field value constraint of `validate_asset_panel`.

    Attributes
    ----------
    arg_name : str
        The `validate_asset_panel` argument that lists the fields to check.

    is_bad : Callable[[AnyArray], AnyArray]
        Returns, elementwise, whether a value violates the rule.

    when_active : bool
        Whether the rule applies only where `active_mask` is True. An inactive asset
        has no value to constrain, so those entries are exempt.

    problem : str
        What the offending values are, as it reads in the error.

    requirement : str
        What the field must contain instead, as it reads in the error. May reference
        `{field}`.
    """

    arg_name: str
    is_bad: Callable[[AnyArray], AnyArray]
    when_active: bool
    problem: str
    requirement: str


# The five value constraints, which differ only in their predicate, whether they are
# restricted to active assets, and their wording.
_VALUE_RULES = (
    _ValueRule(
        "finite_or_nan",
        np.isinf,
        False,
        "infinite values",
        "contain finite values or NaN.",
    ),
    _ValueRule(
        "finite_when_active",
        lambda values: ~np.isfinite(values),
        True,
        "NaN/inf for active assets",
        'be finite wherever `active_mask` is True. Forward-fill "{field}" for '
        'holidays or set `active_mask` to False until the first finite "{field}".',
    ),
    _ValueRule(
        "strictly_positive_or_nan",
        lambda values: ~np.isnan(values) & (~np.isfinite(values) | (values <= 0)),
        False,
        "non-positive or infinite values",
        "contain strictly positive finite values or NaN.",
    ),
    _ValueRule(
        "strictly_positive_when_active",
        lambda values: ~np.isfinite(values) | (values <= 0),
        True,
        "non-finite or non-positive values for active assets",
        "be strictly positive and finite wherever `active_mask` is True.",
    ),
    _ValueRule(
        "non_negative_or_nan",
        lambda values: ~np.isnan(values) & (~np.isfinite(values) | (values < 0)),
        False,
        "negative values or infinite values",
        "contain non-negative finite values or NaN.",
    ),
)


def _check_value_rule(
    asset_panel: AssetPanel | AssetPanelView, field: str, rule: _ValueRule
) -> None:
    """Check one field of `asset_panel` against one value rule.

    Parameters
    ----------
    asset_panel : AssetPanel | AssetPanelView
        The panel holding the field.

    field : str
        The field to check.

    rule : _ValueRule
        The rule to apply.

    Raises
    ------
    ValueError
        If any value of `field` violates `rule`, naming the first observation index
        where it does.
    """
    values = asset_panel[field]
    bad = rule.is_bad(values)
    if values.ndim != 2:
        # A 3-D field violates the rule as soon as one of its components does.
        bad = bad.any(axis=2)
    if rule.when_active:
        bad = bad & asset_panel.active_mask
    if not bad.any():
        return

    requirement = rule.requirement.format(field=field)
    raise ValueError(
        f'Field "{field}" contains {rule.problem} '
        f"(first at observation index {_bad_observation(bad)}). "
        f'"{field}" must {requirement}'
    )


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
