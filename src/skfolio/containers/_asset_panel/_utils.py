"""Asset Panel utils module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from skfolio.containers._asset_panel._fields import (
    BaseField,
    Field2D,
    Field3D,
    FieldCategorical,
)
from skfolio.typing import AnyArray, ArrayLike, BoolArray, IntArray, StrArray

if TYPE_CHECKING:
    from skfolio.containers._asset_panel._panel import AssetPanel
    from skfolio.containers._asset_panel._view import AssetPanelView


_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)
_FIELD_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.-]*$")


def _as_field(value: BaseField | ArrayLike) -> BaseField:
    """Return a field object, wrapping raw arrays as `Field2D`."""
    if isinstance(value, BaseField):
        return value
    return Field2D(np.asarray(value))


def _validate_field_name(name: str) -> None:
    """Validate a field name against syntax and filesystem constraints."""
    if not isinstance(name, str) or not name:
        raise ValueError("Field name must be a non-empty string.")
    if len(name) > 200:
        raise ValueError(f"Field name too long ({len(name)} chars, max 200): '{name}'")
    if not _FIELD_NAME_PATTERN.match(name):
        raise ValueError(
            f"Field name '{name}' contains invalid characters. "
            "Must match pattern [a-zA-Z_][a-zA-Z0-9_.-]*."
        )
    stem = name.split(".")[0].upper()
    if stem in _WINDOWS_RESERVED:
        raise ValueError(
            f"Field name '{name}' conflicts with Windows reserved name '{stem}'."
        )


def _raise_if_raw_replaces_typed_field(
    *,
    name: str,
    value: BaseField | ArrayLike,
    existing_field: BaseField,
) -> None:
    """Reject raw array replacement when existing field metadata would be lost."""
    if isinstance(value, BaseField):
        return
    if isinstance(existing_field, FieldCategorical | Field3D):
        raise TypeError(
            f"Field '{name}' is a {type(existing_field).__name__}. "
            "Replace it with an explicit field object to preserve metadata."
        )


def _slice_field_values(
    field: BaseField,
    observation_selector: slice | IntArray,
    asset_selector: slice | IntArray,
) -> AnyArray:
    """Slice field values on observation and asset axes."""
    if isinstance(field, Field3D):
        return field.values[observation_selector, :, :][:, asset_selector, :]
    return field.values[observation_selector, :][:, asset_selector]


def _validate_field_against_axes(
    *,
    name: str,
    field: BaseField,
    expected_shape: tuple[int, int],
    active_mask: BoolArray,
) -> None:
    """Validate a field against panel axes and active-mask invariants."""
    if field.values.shape[:2] != expected_shape:
        raise ValueError(
            f"Field '{name}' has shape {field.values.shape}; expected "
            f"({expected_shape[0]}, {expected_shape[1]}, ...)."
        )
    if not np.issubdtype(field.values.dtype, np.floating):
        return
    if isinstance(field, Field3D):
        n_bad = int(np.isfinite(field.values[~active_mask, :]).sum())
    else:
        n_bad = int(np.isfinite(field.values[~active_mask]).sum())
    if n_bad:
        raise ValueError(
            f"Field '{name}' has {n_bad:,} finite value(s) where active_mask "
            "is False. Float fields must be NaN outside the active universe."
        )


def _normalize_positional_selector(length: int, selector: Any) -> slice | IntArray:
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
    selector_arr = selector_arr.astype(np.intp, copy=False)
    selector_arr = np.where(selector_arr < 0, selector_arr + length, selector_arr)
    if ((selector_arr < 0) | (selector_arr >= length)).any():
        raise IndexError(
            "Positional selector contains index out of bounds for axis with "
            f"length {length}."
        )
    contiguous = _try_as_contiguous_slice(selector_arr)
    return contiguous if contiguous is not None else selector_arr


def _positions_from_unique_labels(labels: AnyArray, selected: Any) -> slice | IntArray:
    """Resolve unique-label selectors to positional selectors."""
    index = pd.Index(labels)
    if isinstance(selected, slice):
        return index.slice_indexer(selected.start, selected.stop, selected.step)
    if isinstance(selected, (str, bytes)) or np.isscalar(selected):
        loc = index.get_loc(selected)
        return slice(int(loc), int(loc) + 1)

    positions = []
    missing = []
    for label in list(selected):
        try:
            loc = index.get_loc(label)
        except KeyError:
            missing.append(label)
            continue
        positions.append(int(loc))
    if missing:
        raise KeyError(f"Labels not found: {missing}")
    return np.asarray(positions, dtype=np.intp)


def _positions_from_labels(labels: AnyArray, selected: Any) -> slice | IntArray:
    """Resolve possibly repeated label selectors to positional selectors."""
    index = pd.Index(labels)
    if isinstance(selected, slice):
        return index.slice_indexer(selected.start, selected.stop, selected.step)
    if isinstance(selected, (str, bytes)) or np.isscalar(selected):
        loc = index.get_loc(selected)
        if isinstance(loc, slice):
            return loc
        if isinstance(loc, np.ndarray):
            return np.flatnonzero(loc) if loc.dtype == np.bool_ else loc
        return slice(int(loc), int(loc) + 1)

    positions = []
    missing = []
    for label in list(selected):
        try:
            loc = index.get_loc(label)
        except KeyError:
            missing.append(label)
            continue
        if isinstance(loc, slice):
            positions.extend(range(*loc.indices(len(labels))))
        elif isinstance(loc, np.ndarray):
            positions.extend(np.flatnonzero(loc) if loc.dtype == np.bool_ else loc)
        else:
            positions.append(int(loc))
    if missing:
        raise KeyError(f"Labels not found: {missing}")
    return np.asarray(positions, dtype=np.intp)


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


def _materialize_selector(length: int, selector: slice | IntArray) -> IntArray:
    """Return explicit integer positions for a slice or positional array."""
    if isinstance(selector, slice):
        return np.arange(length)[selector]
    return selector


def _selector_length(length: int, selector: slice | IntArray) -> int:
    """Return the number of positions selected by a positional selector."""
    if isinstance(selector, slice):
        return len(range(*selector.indices(length)))
    return len(selector)


def _compose_observation_selectors(
    outer_selector: slice | IntArray,
    inner_selector: slice | IntArray,
    *,
    total_length: int,
) -> slice | IntArray:
    """Compose nested observation selectors relative to the owner panel."""
    if isinstance(outer_selector, slice) and isinstance(inner_selector, slice):
        outer_start, outer_stop, outer_step = outer_selector.indices(total_length)
        if outer_step == 1:
            outer_length = max(0, outer_stop - outer_start)
            inner_start, inner_stop, inner_step = inner_selector.indices(outer_length)
            if inner_step == 1:
                return slice(outer_start + inner_start, outer_start + inner_stop)

    outer_indices = _materialize_selector(total_length, outer_selector)
    composed = outer_indices[inner_selector]
    if isinstance(composed, np.ndarray):
        contiguous = _try_as_contiguous_slice(composed)
        return contiguous if contiguous is not None else composed
    index = int(composed)
    return slice(index, index + 1)


def _fill_2d(
    values: AnyArray,
    *,
    method: Literal["ffill", "bfill"],
    limit: int | None,
    mask: BoolArray | None,
) -> AnyArray:
    """Forward or backward fill a 2D array, optionally within a mask."""
    if mask is None:
        frame = pd.DataFrame(values, copy=False)
        if method == "ffill":
            return frame.ffill(limit=limit).to_numpy()
        return frame.bfill(limit=limit).to_numpy()

    masked = np.where(mask, values, np.inf)
    # Pandas fill is well optimized
    frame = pd.DataFrame(masked, copy=False)
    filled = frame.ffill(limit=limit) if method == "ffill" else frame.bfill(limit=limit)
    filled_values = filled.to_numpy()
    result = values.copy()
    patch = mask & np.isnan(values) & np.isfinite(filled_values)
    result[patch] = filled_values[patch]
    return result


def _format_observation_range(observations: AnyArray) -> str:
    """Format the first and last observation labels for display."""
    if len(observations) == 0:
        return ""
    if len(observations) == 1:
        return f"  ({observations[0]})"
    first = observations[0]
    last = observations[-1]
    try:
        if isinstance(first, (int, float, np.integer, np.floating)):
            raise TypeError
        first_str = str(pd.Timestamp(first).date())
        last_str = str(pd.Timestamp(last).date())
    except Exception:
        first_str = str(first)
        last_str = str(last)
    return f"  ({first_str} -> {last_str})"


def _to_dataframe(
    panel: AssetPanel | AssetPanelView,
    *,
    fields: str | Iterable[str] | None,
    assets: str | Iterable[str] | None,
    output_format: Literal["long", "wide"],
    decode_categoricals: bool,
) -> pd.DataFrame:
    """Convert selected 2D fields and masks to a pandas DataFrame."""
    available_field_names = list(panel.keys())
    if fields is None:
        field_names = available_field_names
    else:
        field_names = [fields] if isinstance(fields, str) else list(fields)
        available_set = set(available_field_names)
        missing = [name for name in field_names if name not in available_set]
        if missing:
            raise KeyError(f"Fields not found: {missing}")

    asset_selector = (
        slice(None)
        if assets is None
        else _positions_from_unique_labels(panel.assets, assets)
    )
    selected_assets = panel.assets[asset_selector]

    if isinstance(fields, str):
        return _field_to_dataframe(
            panel,
            name=field_names[0],
            asset_selector=asset_selector,
            selected_assets=selected_assets,
            decode=decode_categoricals,
        )

    if output_format == "long":
        return _to_long_dataframe(
            panel,
            field_names=field_names,
            asset_selector=asset_selector,
            selected_assets=selected_assets,
            decode_categoricals=decode_categoricals,
        )
    if output_format == "wide":
        return _to_wide_dataframe(
            panel,
            field_names=field_names,
            asset_selector=asset_selector,
            selected_assets=selected_assets,
            decode_categoricals=decode_categoricals,
        )
    raise ValueError(f"output_format must be 'long' or 'wide', got {output_format!r}.")


def _field_to_dataframe(
    panel: AssetPanel | AssetPanelView,
    *,
    name: str,
    asset_selector: slice | IntArray,
    selected_assets: StrArray,
    decode: bool,
) -> pd.DataFrame:
    """Convert a single 2D field to a DataFrame."""
    field = panel.get_field(name)
    if isinstance(field, Field3D):
        raise ValueError(
            f"Field '{name}' has {field.ndim} dimensions; only 2D fields are supported."
        )

    values = field.values[:, asset_selector]
    if decode and isinstance(field, FieldCategorical):
        values = field.decode()[:, asset_selector]
    return pd.DataFrame(values, index=panel.observations, columns=selected_assets)


def _to_long_dataframe(
    panel: AssetPanel | AssetPanelView,
    *,
    field_names: list[str],
    asset_selector: slice | IntArray,
    selected_assets: StrArray,
    decode_categoricals: bool,
) -> pd.DataFrame:
    """Convert fields to long format filtered by `active_mask`."""
    n_observations = len(panel.observations)
    n_assets = len(selected_assets)
    index = pd.MultiIndex.from_arrays(
        [
            np.repeat(panel.observations, n_assets),
            np.tile(selected_assets, n_observations),
        ],
        names=["observation", "asset"],
    )

    columns: dict[str, AnyArray] = {}
    for name in field_names:
        field = panel.get_field(name)
        if isinstance(field, Field3D):
            warnings.warn(
                f"Skipping Field3D '{name}' in to_dataframe.",
                UserWarning,
                stacklevel=3,
            )
            continue
        values = field.values[:, asset_selector]
        if decode_categoricals and isinstance(field, FieldCategorical):
            values = field.decode()[:, asset_selector]
        columns[name] = values.ravel()

    active_mask = panel.active_mask[:, asset_selector]
    estimation_mask = panel.estimation_mask[:, asset_selector]
    columns["estimation_mask"] = estimation_mask.ravel()
    df = pd.DataFrame(columns, index=index)
    return df[active_mask.ravel()]


def _to_wide_dataframe(
    panel: AssetPanel | AssetPanelView,
    *,
    field_names: list[str],
    asset_selector: slice | IntArray,
    selected_assets: StrArray,
    decode_categoricals: bool,
) -> pd.DataFrame:
    """Convert fields and masks to wide format with `(field, asset)` columns."""
    frames = []
    for name in field_names:
        field = panel.get_field(name)
        if isinstance(field, Field3D):
            warnings.warn(
                f"Skipping Field3D '{name}' in to_dataframe.",
                UserWarning,
                stacklevel=3,
            )
            continue
        values = field.values[:, asset_selector]
        if decode_categoricals and isinstance(field, FieldCategorical):
            values = field.decode()[:, asset_selector]
        frame = pd.DataFrame(values, columns=selected_assets)
        frame.columns = pd.MultiIndex.from_product([[name], selected_assets])
        frames.append(frame)

    for mask_name in ("active_mask", "estimation_mask"):
        values = getattr(panel, mask_name)[:, asset_selector]
        frame = pd.DataFrame(values, columns=selected_assets)
        frame.columns = pd.MultiIndex.from_product([[mask_name], selected_assets])
        frames.append(frame)

    result = pd.concat(frames, axis=1)
    result.index = panel.observations
    result.index.name = "observation"
    result.columns.names = ["field", "asset"]
    return result


def _validate_active_observations(active_mask: BoolArray) -> None:
    """Validate that each observation has at least one active asset."""
    if not active_mask.any(axis=1).all():
        empty_idx = np.where(~active_mask.any(axis=1))[0]
        raise ValueError(
            "`active_mask` must contain at least one active asset for every "
            f"observation. Found {len(empty_idx)} observation(s) with none; "
            f"first failing position is {empty_idx[0]}."
        )
