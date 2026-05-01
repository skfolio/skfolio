"""Asset Panel Fields module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any

import numpy as np

from skfolio.typing import AnyArray, ArrayLike, BoolArray, StrArray

__all__ = [
    "MISSING_CATEGORY_CODE",
    "BaseField",
    "Field2D",
    "Field3D",
    "FieldCategorical",
]

MISSING_CATEGORY_CODE: int = -1


@dataclass(slots=True)
class BaseField(ABC):
    """Base class for fields stored in an `AssetPanel`.

    A field stores a NumPy array for one named variable. The first two axes are always
    observations and assets with shape (n_observations, n_assets).
    Subclasses define any additional metadata required by the array layout.

    Parameters
    ----------
    values : ndarray of shape (n_observations, n_assets)
        Field values. The first two axes must be observations and assets.
    """

    values: AnyArray

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        if self.values.ndim < 2:
            raise ValueError("Field values must have at least two dimensions.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying values."""
        return self.values.shape

    @property
    def dtype(self) -> np.dtype:
        """Dtype of the underlying values."""
        return self.values.dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions of the underlying values."""
        return self.values.ndim

    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return int(self.values.shape[0])

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return int(self.values.shape[1])

    @property
    def missing_mask(self) -> BoolArray:
        """Return a boolean mask indicating missing entries.

        Returns
        -------
        mask : ndarray of bool
            Boolean array with the same shape as `values`. Entries are `True`
            where `values` is NaN for floating-point fields and `False` elsewhere.
            Non-floating fields without a missing-value convention return an all-`False`
            mask; subclasses override this method when they define their own convention.
        """
        if np.issubdtype(self.values.dtype, np.floating):
            return np.isnan(self.values)
        return np.zeros(self.values.shape, dtype=np.bool_)

    def copy(self, *, deep: bool = False) -> BaseField:
        """Return a copy of the field.

        Parameters
        ----------
        deep : bool, default=False
            If `True`, copy NumPy arrays stored by the field. If `False`, reuse the same
            array objects.

        Returns
        -------
        field : BaseField
            Field instance of the same concrete class.
        """
        kwargs = {}
        for field in dataclass_fields(self):
            value = getattr(self, field.name)
            if deep and isinstance(value, np.ndarray):
                value = value.copy()
            kwargs[field.name] = value
        return type(self)(**kwargs)

    def with_values(self, values: ArrayLike) -> BaseField:
        """Return a field of the same type with replacement values.

        Metadata arrays are reused. The replacement values are validated by the
        concrete field class constructor.

        Parameters
        ----------
        values : array-like
            Replacement field values.

        Returns
        -------
        field : BaseField
            Field instance of the same concrete class.
        """
        kwargs = {
            field.name: getattr(self, field.name)
            for field in dataclass_fields(self)
            if field.name != "values"
        }
        return type(self)(values=np.asarray(values), **kwargs)


@dataclass(slots=True)
class Field2D(BaseField):
    """Numeric 2D field with axes (observations, assets).

    Parameters
    ----------
    values : ndarray of shape (n_observations, n_assets)
        Numeric field values. Object dtype is rejected. Use `FieldCategorical` for
        categorical data.
    """

    def __post_init__(self) -> None:
        BaseField.__post_init__(self)
        if self.values.ndim != 2:
            raise ValueError(f"Field2D values must be 2D, got ndim={self.values.ndim}.")
        if self.values.dtype == object:
            raise ValueError(
                "Field2D values cannot have dtype=object. Store categoricals as "
                "integer codes with FieldCategorical."
            )


@dataclass(slots=True)
class FieldCategorical(Field2D):
    """Integer-coded categorical 2D field.

    Codes are stored as integers in a 2D array with axes (observations, assets).
    Code -1 is reserved for missing values. Code 0 selects `levels[0]`, code 1 selects
    `levels[1]` and so on.

    Parameters
    ----------
    values : integers ndarray of shape (n_observations, n_assets)
        Integer category codes. Missing values must be encoded with
        `MISSING_CATEGORY_CODE`.

    levels : ndarray of shape (n_levels,)
        Category labels selected by codes 0, 1 and so on.
        Labels must be unique.
    """

    levels: StrArray | list[str]

    def __post_init__(self) -> None:
        Field2D.__post_init__(self)
        self.levels = _as_pickle_safe_array(self.levels)
        if not np.issubdtype(self.values.dtype, np.integer):
            raise ValueError(
                "FieldCategorical values must have an integer dtype; "
                f"got {self.values.dtype}."
            )
        if self.levels.ndim != 1:
            raise ValueError("FieldCategorical levels must be a 1D array.")
        _validate_unique_labels(self.levels, name="FieldCategorical levels")

        if self.values.size != 0:
            min_code = int(np.min(self.values))
            max_code = int(np.max(self.values))
            if min_code < MISSING_CATEGORY_CODE:
                raise ValueError(
                    f"Categorical codes must be >= {MISSING_CATEGORY_CODE}; "
                    f"got minimum code {min_code}."
                )
            if max_code >= len(self.levels):
                raise ValueError(
                    f"Categorical code {max_code} is out of bounds for "
                    f"{len(self.levels)} levels."
                )

    @property
    def missing_mask(self) -> BoolArray:
        """Return a boolean mask indicating missing categorical codes.

        Returns
        -------
        mask : ndarray of bool
            Boolean array with the same shape as `values`. Entries are `True`
            where `values` equals `MISSING_CATEGORY_CODE`.
        """
        return self.values == MISSING_CATEGORY_CODE

    def decode(self, *, missing_label: str = "MISSING") -> StrArray:
        """Decode integer codes to level labels.

        Parameters
        ----------
        missing_label : str, default="MISSING"
            Label assigned to missing or out-of-bound codes.

        Returns
        -------
        decoded : ndarray
            Decoded labels with the same shape as `values`.
        """
        codes = self.values
        levels = self.levels

        if codes.size == 0:
            return np.asarray([], dtype=levels.dtype).reshape(codes.shape)

        dtype = np.result_type(levels.dtype, np.asarray(missing_label).dtype)
        decoded = np.empty(codes.shape, dtype=dtype)
        valid = (codes >= 0) & (codes < len(levels))
        decoded[valid] = levels[codes[valid]]
        decoded[~valid] = str(missing_label)
        return decoded


@dataclass(slots=True)
class Field3D(BaseField):
    """Numeric 3D field with axes (observations, assets, third_axis).

    Use this field for homogeneous tensor (e.g. factor exposures). The third-axis
    metadata is stored on the field. The array is stored physically in 3D so that
    operations along any axis remain vectorized and avoid stacking many 2D arrays which
    is expensive for large panels. When you need repeated tensor operation, prefer a
    `Field3D` instead of multiple `Field2D`.

    Parameters
    ----------
    values : ndarray of shape `(n_observations, n_assets, n_third_axis)`
        Numeric 3D values. Object dtype is rejected.

    third_axis_name : str
        Name describing what the third axis represents (e.g. `factor`).

    third_axis_labels : ndarray of shape (n_third_axis,)
        Labels for entries along the third axis such as factor names (`size`,
        `momentum`). Labels must be unique.

    third_axis_groups : ndarray of shape (n_third_axis,), optional
        Optional group label for each third-axis entry such as factor families (e.g.
        `style`, `industry`, `country`).
    """

    third_axis_name: str
    third_axis_labels: StrArray | list[str]
    third_axis_groups: StrArray | list[str] | None = None

    def __post_init__(self) -> None:
        BaseField.__post_init__(self)
        self.third_axis_labels = _as_pickle_safe_array(self.third_axis_labels)
        if self.third_axis_groups is not None:
            self.third_axis_groups = _as_pickle_safe_array(self.third_axis_groups)

        if self.values.ndim != 3:
            raise ValueError(f"Field3D values must be 3D, got ndim={self.values.ndim}.")
        if self.values.dtype == object:
            raise ValueError("Field3D values cannot have dtype=object.")
        if not isinstance(self.third_axis_name, str) or not self.third_axis_name:
            raise ValueError("third_axis_name must be a non-empty string.")
        if self.third_axis_labels.ndim != 1:
            raise ValueError("third_axis_labels must be a 1D array.")
        _validate_unique_labels(self.third_axis_labels, name="third_axis_labels")
        if len(self.third_axis_labels) != self.values.shape[2]:
            raise ValueError(
                f"Field3D has {self.values.shape[2]} entries on the third axis but "
                f"third_axis_labels has length {len(self.third_axis_labels)}."
            )
        if self.third_axis_groups is not None:
            if self.third_axis_groups.ndim != 1:
                raise ValueError("third_axis_groups must be a 1D array.")
            if len(self.third_axis_groups) != self.values.shape[2]:
                raise ValueError(
                    "third_axis_groups length must match third_axis_labels length."
                )


def _as_observation_array(values: ArrayLike) -> AnyArray:
    """Return observation labels with pickle-safe and predictable semantics.

    - non-object NumPy dtypes are preserved as provided, including integer, string and
      `datetime64` dtypes with their existing units.
    - object arrays containing only strings are converted to a NumPy string dtype,
      unless NumPy can interpret all strings as datetime labels.
    - object arrays containing only integer-like labels are converted through NumPy
      asarray.
    - other object arrays are converted with NumPy datetime parsing when they contain
      no string labels, which supports Python `date`, `datetime` and compatible
      timestamp objects.
    - mixed string/non-string object arrays and custom object labels are rejected to
      avoid silently changing sample-axis semantics through string conversion.

    The returned array can be saved with `allow_pickle=False`.
    """
    arr = np.asarray(values)
    if arr.dtype != object:
        return arr

    flat = arr.ravel()
    if _all_string_like(flat):
        try:
            return np.asarray(arr, dtype="datetime64")
        except (TypeError, ValueError):
            return np.asarray(arr, dtype=str)

    if _all_integer_like(flat):
        return np.asarray(arr.tolist())

    if not any(_is_string_like(value) for value in flat):
        try:
            return np.asarray(arr, dtype="datetime64")
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "`observations` with object dtype must contain datetime-like, "
                "integer or string labels. Mixed or custom object labels are not "
                "supported because observations define the sample axis."
            ) from exc

    raise ValueError(
        "`observations` with object dtype must contain datetime-like, integer or "
        "string labels. Mixed or custom object labels are not supported because "
        "observations define the sample axis."
    )


def _as_pickle_safe_array(values: ArrayLike) -> AnyArray:
    """Return an array that can be loaded without pickle, converting object dtype to
    strings.
    """
    arr = np.asarray(values)
    if arr.dtype == object:
        return np.asarray(arr, dtype=str)
    return arr


def _all_string_like(values: AnyArray) -> bool:
    """Return True when all values are string-like."""
    return all(_is_string_like(value) for value in values)


def _all_integer_like(values: AnyArray) -> bool:
    """Return True when all values are integer-like and not booleans."""
    return all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        for value in values
    )


def _is_string_like(value: Any) -> bool:
    """Return True when a value is string-like."""
    return isinstance(value, (str, np.str_))


def _validate_unique_labels(labels: AnyArray, *, name: str) -> None:
    """Validate that a 1D label array contains no duplicates."""
    if len(np.unique(labels)) != len(labels):
        raise ValueError(f"{name} must be unique.")
