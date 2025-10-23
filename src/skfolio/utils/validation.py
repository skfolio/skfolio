"""Validation utilities for cross-sectional data."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.utils.validation as skv


def validate_cross_sectional_data(
    _estimator,
    /,
    X: npt.ArrayLike | Literal["no_validation"] = "no_validation",
    y: npt.ArrayLike | Literal["no_validation"] | None = "no_validation",
    *,
    reset: bool = True,
    copy: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Validate and convert cross-sectional 3D inputs for estimator fitting.

    Similar to sklearn's `validate_data` but specialized for cross-sectional
    data with 3D features (observations x assets x factors) and 2D targets
    (observations x assets). Handles both NumPy arrays and pandas DataFrames
    with MultiIndex columns.

    This function performs the following operations:

    1. Validates input shapes and dtypes
    2. Converts DataFrames to NumPy arrays with proper reshaping
    3. Stores metadata (assets, factors, index) on the estimator if reset=True
    4. Ensures compatibility between X and y shapes

    Parameters
    ----------
    _estimator : estimator instance
        The estimator object on which to set validation attributes.
        Sets `n_features_in_`, `assets_`, `factors_`, and `index_`
        attributes if `reset=True`.

    X : array-like of shape (n_observations, n_assets, n_factors) or 'no_validation', default='no_validation'
        The input feature array.

        - If array-like: Must be 3D with shape (n_observations, n_assets, n_factors)
        - If DataFrame: Must have a 2-level MultiIndex with levels (asset, factor)
          containing all (asset x factor) combinations, then reshaped to 3D.
        - If 'no_validation': No validation is performed on X.

    y : array-like of shape  (n_observations, n_assets) or 'no_validation', default='no_validation'
        The target values.

        - If ndarray or array-like: Must be 2D with shape (n_observations, n_assets)
        - If DataFrame: Will be reindexed to match X's index (observations) and
          columns (assets) if X is a DataFrame
        - If None or 'no_validation': Only X is validated and returned

    reset : bool, default=True
        Whether to set/reset the estimator's validation attributes:

        - `n_features_in_`: Number of factors (features)
        - `assets_`: Array of asset names (if X is DataFrame)
        - `factors_`: Array of factor names (if X is DataFrame)
        - `index_`: Index values (if X is DataFrame)

        If False, validation assumes these attributes were previously set
        and checks for consistency.

    copy : bool, default=False
        Whether to force a copy of X and y.
        If False, a copy will only be made if required for dtype conversion.
        If True, a copy is always made regardless of dtype.

    Returns
    -------
    X_validated : ndarray of shape (n_observations, n_assets, n_factors)
        Validated feature array. Returned when y is 'no_validation' or None.

    X_validated, y_validated : tuple of ndarrays
        Validated feature and target arrays. X has shape
        (n_observations, n_assets, n_factors), y has shape
        (n_observations, n_assets). Returned when both X and y are provided.

    Raises
    ------
    ValueError
        - If both X and y are 'no_validation'.
        - If X is 'no_validation' but y is provided.
        - If X DataFrame columns are not a 2-level MultiIndex.
        - If X DataFrame is missing some (asset, factor) combinations.
        - If X array is not 3D.
        - If y array is not 2D.
        - If X and y have incompatible shapes.

    Notes
    -----
    This function uses sklearn's :func:`~sklearn.utils.validation.check_array`
    internally for robust dtype conversion and basic validation. NaN values are
    allowed (`ensure_all_finite=False`) since cross-sectional financial data
    often contains missing values due to asset listing/delisting.

    The function follows scikit-learn conventions:

    - Uses keyword-only arguments after the positional estimator parameter.
    - Sets `n_features_in_` following sklearn's pattern (number of factors).
    - Stores feature metadata similar to sklearn's feature names tracking.

    See Also
    --------
    sklearn.utils.validation.check_array : Validate and convert input arrays.
    sklearn.base.BaseEstimator : Base class for all estimators.
    """
    skip_X_validation = isinstance(X, str) and X == "no_validation"
    skip_y_validation = y is None or (isinstance(y, str) and y == "no_validation")

    if skip_X_validation and skip_y_validation:
        raise ValueError("Validation should be done on X, y, or both.")

    if skip_X_validation and not skip_y_validation:
        raise ValueError("X must be provided when validating cross-sectional data.")

    index_ = None
    assets_ = None
    factors_ = None

    if isinstance(X, pd.DataFrame):
        if not isinstance(X.columns, pd.MultiIndex):
            raise ValueError(
                "X must have MultiIndex columns with 2 levels (asset, factor). "
                f"Got columns of type {type(X.columns).__name__}."
            )

        if X.columns.nlevels != 2:
            raise ValueError(
                "X columns must be a MultiIndex with exactly 2 levels (asset, factor). "
                f"Got {X.columns.nlevels} levels."
            )

        unique_assets = X.columns.get_level_values(0).unique()
        unique_factors = X.columns.get_level_values(1).unique()

        n_assets = len(unique_assets)
        n_factors = len(unique_factors)
        expected_n_columns = n_assets * n_factors

        if len(X.columns) != expected_n_columns:
            raise ValueError(
                f"X must have all (asset, factor) combinations. "
                f"Expected {expected_n_columns} columns "
                f"({n_assets} assets x {n_factors} factors), got {len(X.columns)}."
            )

        index_ = X.index
        assets_ = unique_assets.to_numpy()
        factors_ = unique_factors.to_numpy()

        X_2d = skv.check_array(
            X,
            dtype="numeric",
            ensure_all_finite=False,
            ensure_2d=True,
            copy=copy,
            input_name="X",
        )

        n_observations = X_2d.shape[0]
        X_validated = X_2d.reshape(n_observations, n_assets, n_factors)

    else:
        X_validated = skv.check_array(
            X,
            dtype="numeric",
            ensure_all_finite=False,
            ensure_2d=False,
            allow_nd=True,
            copy=copy,
            input_name="X",
        )

        if X_validated.ndim != 3:
            raise ValueError(
                f"X must be a 3D array (n_observations, n_assets, n_factors). "
                f"Got {X_validated.ndim}D array with shape {X_validated.shape}."
            )

    if reset:
        _estimator.n_features_in_ = X_validated.shape[2]
        if assets_ is not None:
            _estimator.assets_ = assets_
        if factors_ is not None:
            _estimator.factors_ = factors_
        if index_ is not None:
            _estimator.index_ = index_
    else:
        if hasattr(_estimator, "n_features_in_"):
            n_factors = X_validated.shape[2]
            if n_factors != _estimator.n_features_in_:
                raise ValueError(
                    f"Expected n_factors={_estimator.n_features_in_} "
                    f"(from fitted model), got {n_factors}."
                )

    if skip_y_validation:
        return X_validated

    if isinstance(y, pd.DataFrame):
        if index_ is not None:
            y = y.reindex(index=index_)
        if assets_ is not None:
            y = y.reindex(columns=assets_)

    y_validated = skv.check_array(
        y,
        dtype="numeric",
        ensure_all_finite=False,
        ensure_2d=True,
        copy=copy,
        input_name="y",
    )

    n_observations, n_assets, _ = X_validated.shape
    expected_shape = (n_observations, n_assets)

    if y_validated.shape != expected_shape:
        raise ValueError(
            f"Incompatible shapes: X has shape {X_validated.shape}, "
            f"so y must have shape {expected_shape}, got {y_validated.shape}."
        )

    return X_validated, y_validated
