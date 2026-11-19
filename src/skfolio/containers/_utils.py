"""Shared label validation and positional selectors for containers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from skfolio.typing import AnyArray, ArrayLike, IntArray, StrArray
from skfolio.utils.tools import _is_integer_number


def _as_string_labels(values: ArrayLike, *, name: str) -> StrArray:
    """Validate unique string labels and return a NumPy string array.

    Parameters
    ----------
    values : array-like of shape (n_labels,)
        Unique string labels. Numeric and missing identifiers are not supported.

    name : str
        Input name used in error messages.

    Returns
    -------
    labels : StrArray of shape (n_labels,)
        Labels stored as a NumPy string array. Existing NumPy string arrays are returned
        without copying.

    Raises
    ------
    ValueError
        If labels are not one-dimensional or contain non-string or duplicate values.
    """
    labels = (
        values
        if isinstance(values, np.ndarray) and values.dtype.kind == "U"
        else np.asarray(values, dtype=object)
    )
    if labels.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if labels.dtype.kind != "U":
        if not _all_string_like(labels):
            raise ValueError(
                f"{name} must contain only strings with no missing values. "
                "Convert numeric identifiers to strings explicitly."
            )
        labels = labels.astype(str)
    _validate_unique_labels(labels, name=name)
    return labels


def _as_observation_array(values: ArrayLike) -> AnyArray:
    """Normalize observation labels while preserving timezone-aware timestamps.

    - NumPy dtypes without object storage are preserved as provided, including integer,
      fixed-width string and `datetime64` dtypes with their existing units.
    - object arrays containing only strings are converted to a NumPy string dtype,
      unless NumPy can interpret all strings as datetime labels.
    - object arrays containing only integer-like labels are converted through NumPy
      asarray.
    - timezone-aware dates retain their timezone as an object array of pandas timestamps.
      Dates with different timezones, or a mix of dates with and without a timezone,
      are rejected.
    - object-dtype dates without a timezone are converted with NumPy datetime parsing.
    - mixed string/non-string object arrays and custom object labels are rejected to
      avoid silently changing sample-axis semantics through string conversion.

    Empty and all-missing datetime inputs use a native datetime dtype with an explicit
    unit. Timezone-aware arrays require the shared observation encoder when saving.
    """
    if isinstance(getattr(values, "dtype", None), pd.DatetimeTZDtype):
        dates = pd.DatetimeIndex(values)
        if dates.isna().all():
            return dates.tz_localize(None).to_numpy()
        return dates.to_numpy()

    arr = np.asarray(values)
    if arr.dtype.hasobject:
        arr = np.asarray(arr, dtype=object)
    if arr.dtype != object:
        return (
            arr.astype("datetime64[ns]") if arr.dtype == np.dtype("datetime64") else arr
        )

    flat = arr.ravel()
    missing = pd.isna(arr)
    if missing.all():
        return np.full(arr.shape, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    if _all_string_like(flat):
        try:
            return np.asarray(arr, dtype="datetime64")
        except (TypeError, ValueError):
            return np.asarray(arr, dtype=str)

    if _all_integer_like(flat):
        return np.asarray(arr.tolist())

    if not any(_is_string_like(value) for value in flat):
        dates = np.where(missing, None, arr)
        if any(
            isinstance(value, datetime) and value.tzinfo is not None for value in flat
        ):
            try:
                return pd.DatetimeIndex(dates.ravel()).to_numpy().reshape(arr.shape)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "`observations` must contain either timezone-naive datetimes or "
                    "timezone-aware datetimes in a single timezone."
                ) from exc
        try:
            return np.asarray(dates, dtype="datetime64")
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
    return all(_is_integer_number(value) for value in values)


def _is_string_like(value: object) -> bool:
    """Return True when a value is string-like."""
    return isinstance(value, (str, np.str_))


def _validate_unique_labels(labels: AnyArray, *, name: str) -> None:
    """Validate that a 1D label array contains no duplicates."""
    # Hashing avoids the sorting performed by np.unique when only uniqueness is needed.
    if not pd.Index(labels).is_unique:
        raise ValueError(f"{name} must be unique.")


def _observation_key(key: Any) -> Any:  # noqa: ANN401  # indexing accepts heterogeneous selectors
    """Unwrap scikit-learn's observation-only tuple indexing convention."""
    if isinstance(key, tuple):
        if len(key) != 2 or key[1] is not Ellipsis or isinstance(key[0], str):
            raise TypeError(
                "Only one-dimensional observation selectors or (observations, ...) are supported."
            )
        key = key[0]
    if key is None:
        raise TypeError(
            "None is not an observation selector. Use [:] to select all observations."
        )
    return key


def _normalize_positional_selector(
    *, length: int, selector: slice | ArrayLike | None
) -> slice | IntArray:
    """Normalize a positional selector to a slice or integer positions."""
    if selector is None:
        return slice(None)
    if isinstance(selector, slice):
        return selector
    if isinstance(selector, (int, np.integer)):
        index = int(selector)
        if index < 0:
            index += length
        if index < 0 or index >= length:
            raise IndexError(
                f"Positional index {selector} is out of bounds for axis with "
                f"length {length}."
            )
        return slice(index, index + 1)

    selector_arr = np.asarray(selector)
    if selector_arr.ndim != 1:
        raise ValueError("Selector must be 1D.")
    if selector_arr.dtype == np.bool_:
        if selector_arr.shape != (length,):
            raise ValueError(f"Boolean selector must have shape ({length},).")
        return np.flatnonzero(selector_arr)
    if not np.issubdtype(selector_arr.dtype, np.integer):
        raise TypeError("Positional selector must contain integers or booleans.")
    # Check bounds before converting unsigned indices to signed integers.
    if ((selector_arr < -length) | (selector_arr >= length)).any():
        raise IndexError(
            "Positional selector contains index out of bounds for axis with "
            f"length {length}."
        )
    selector_arr = selector_arr.astype(np.intp, copy=False)
    selector_arr = np.where(selector_arr < 0, selector_arr + length, selector_arr)
    contiguous = _try_as_contiguous_slice(selector_arr)
    return contiguous if contiguous is not None else selector_arr


def _try_as_contiguous_slice(indices: IntArray) -> slice | None:
    """Return an equivalent slice when integer positions are contiguous."""
    if indices.ndim != 1:
        return None
    if indices.size == 0:
        return slice(0, 0)
    if indices.size == 1:
        start = int(indices[0])
        return slice(start, start + 1)
    if np.all(np.diff(indices) == 1):
        return slice(int(indices[0]), int(indices[-1]) + 1)
    return None


def _validate_unique_observations(observations: AnyArray) -> None:
    """Validate that concatenated observations contain no duplicate labels."""
    observation_index = pd.Index(observations)
    duplicate_positions = np.flatnonzero(observation_index.duplicated())
    if duplicate_positions.size:
        first_duplicate_position = int(duplicate_positions[0])
        first_duplicate = observations[first_duplicate_position]
        raise ValueError(
            "`observations` must be unique when verify_observations=True. "
            f"Found duplicate label {first_duplicate!r} at position "
            f"{first_duplicate_position}."
        )
