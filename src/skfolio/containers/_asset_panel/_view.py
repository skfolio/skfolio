"""Containers for aligned cross-sectional asset data."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from skfolio.containers._asset_panel._base import _BaseAssetPanel
from skfolio.containers._asset_panel._fields import BaseField
from skfolio.containers._asset_panel._utils import (
    _as_field,
    _compose_observation_selectors,
    _normalize_positional_selector,
    _raise_if_raw_replaces_typed_field,
    _selector_length,
    _validate_field_against_axes,
    _validate_field_name,
)
from skfolio.typing import AnyArray, ArrayLike, BoolArray, IntArray, StrArray

if TYPE_CHECKING:
    from skfolio.containers._asset_panel._panel import AssetPanel

__all__ = ["AssetPanelView"]


@dataclass(slots=True)
class AssetPanelView(_BaseAssetPanel):
    """Observation-sliced view into an `AssetPanel`.

    A view stores an observation selector, a reference to its owner and optional
    view-local fields. Owner field arrays are sliced lazily on access through `fields`,
    `__getitem__` and `get_field`, so building a view never copies owner field data.

    When the selector is a slice, all access remains zero-copy and composing nested
    views produces another zero-copy slice. When the selector is an integer or boolean
    array, NumPy fancy indexing is applied on access and the resulting arrays may be
    copies.

    New fields can also be added directly to a view. These view-local fields are useful
    for derived data that only belongs to one slice (e.g. values computed for a
    cross-validation fold). They are stored on the view, do not modify the owner panel,
    and must have shape (n_view_observations, n_assets, ...).

    Parameters
    ----------
    owner : AssetPanel
        Panel that owns the underlying arrays.

    observation_selector : slice or ndarray of integers, default=slice(None)
        Selector applied to the owner observation axis. Slices preserve zero-copy
        semantics. Integer arrays follow NumPy fancy-indexing semantics on access.

    _local_fields : dict[str, BaseField], optional
        View-local fields. This argument is for internal use. Use `view[name] = value`
        to add local fields.

    See Also
    --------
    AssetPanel : Owning container.
    AssetPanel.isel : Returns a view for observation-only selections.
    AssetPanel.sel : Label-based equivalent of `AssetPanel.isel`.
    """

    owner: AssetPanel
    observation_selector: slice | IntArray = slice(None)
    _local_fields: dict[str, BaseField] | None = None

    def __post_init__(self) -> None:
        if self.owner is None:
            raise ValueError("AssetPanelView requires an owner.")
        self.observation_selector = _normalize_positional_selector(
            self.owner.n_observations,
            self.observation_selector,
        )
        if self._local_fields is None:
            self._local_fields = {}
        for name, field in self._local_fields.items():
            _validate_field_name(name)
            self._validate_field(name, field)

    def __len__(self) -> int:
        """Return the number of observations in the view."""
        return self.n_observations

    def __contains__(self, name: str) -> bool:
        """Check whether a local or owner field exists."""
        return name in self._local_fields or name in self.owner.fields

    def __getitem__(self, key: Any) -> AnyArray | AssetPanelView:
        """Return field values or a nested observation view.

        Parameters
        ----------
        key : str, int, slice, or array-like
            If `key` is a string, return the field values resolved through local fields
            and the owner panel. Otherwise, interpret `key` as an observation selector
            relative to the current view.

        Returns
        -------
        values or view : ndarray or AssetPanelView
            For a string key, the field values sliced to the current view. For an
            observation selector, a nested `AssetPanelView` whose selector is composed
            with the current view selector.

        Notes
        -----
        Selector composition keeps slice-only chains zero-copy. When a  selector is an
        integer or boolean array, NumPy fancy indexing is applied on access and may copy.
        """
        if isinstance(key, tuple):
            raise TypeError(
                "AssetPanelView supports field access by name or one-dimensional "
                "observation selectors."
            )
        if isinstance(key, str):
            return self.get_field(key).values

        inner_selector = _normalize_positional_selector(self.n_observations, key)
        composed_selector = _compose_observation_selectors(
            outer_selector=self.observation_selector,
            inner_selector=inner_selector,
            total_length=self.owner.n_observations,
        )
        local_fields = {
            name: field.with_values(field.values[inner_selector])
            for name, field in self._local_fields.items()
        }
        return AssetPanelView(
            owner=self.owner,
            observation_selector=composed_selector,
            _local_fields=local_fields,
        )

    def __setitem__(self, name: str, value: BaseField | ArrayLike) -> None:
        """Add or replace a view-local field.

        Raw arrays are converted to `Field2D`. Replacing an existing `FieldCategorical`
        or `Field3D` requires an explicit field object so metadata is not discarded
        accidentally.

        Parameters
        ----------
        name : str
            Local field name.

        value : BaseField or array-like
            Field object or raw 2D array with first two axes
            (n_view_observations, n_assets).
        """
        _validate_field_name(name)
        if name in self:
            existing_field = (
                self._local_fields[name]
                if name in self._local_fields
                else self.owner.fields[name]
            )
            _raise_if_raw_replaces_typed_field(
                name=name, value=value, existing_field=existing_field
            )
        field = _as_field(value)
        self._validate_field(name, field)
        self._local_fields[name] = field

    def __delitem__(self, name: str) -> None:
        """Delete a view-local field.

        Owner fields cannot be deleted through a view.

        Parameters
        ----------
        name : str
            Name of the local field to delete.
        """
        if name not in self._local_fields:
            raise KeyError(
                f"Field '{name}' is not a local field. Owner fields cannot be "
                "deleted from a view."
            )
        del self._local_fields[name]

    def __repr__(self) -> str:
        return (
            f"AssetPanelView(n_observations={self.n_observations}, "
            f"n_assets={self.n_assets}, n_fields={len(self.keys())})"
        )

    @property
    def observations(self) -> AnyArray:
        """Observation labels selected by the view."""
        return self.owner.observations[self.observation_selector]

    @property
    def assets(self) -> StrArray:
        """Asset labels."""
        return self.owner.assets

    @property
    def active_mask(self) -> BoolArray:
        """Active mask selected by the view."""
        return self.owner.active_mask[self.observation_selector]

    @property
    def estimation_mask(self) -> BoolArray:
        """Estimation mask selected by the view."""
        return self.owner.estimation_mask[self.observation_selector]

    @property
    def fields(self) -> Mapping[str, BaseField]:
        """Lazy mapping of field objects with view-sized values.

        Field objects are constructed on access and reuse the owner array sliced by
        `observation_selector`. Iterating through this mapping does not materialize
        sliced arrays for fields that are not accessed.
        """
        return _ViewFieldMapping(self)

    @property
    def n_observations(self) -> int:
        """Number of observations in the view."""
        return _selector_length(self.owner.n_observations, self.observation_selector)

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self.owner.n_assets

    def keys(self) -> Iterable[str]:
        """Return field names visible from the view.

        Returns
        -------
        names : list of str
            Union of view-local field names and owner field names. Local fields shadow
            owner fields with the same name. Owner field order is preserved and
            local-only fields are appended in insertion order.
        """
        local_only = [
            name for name in self._local_fields if name not in self.owner.fields
        ]
        return [*self.owner.fields.keys(), *local_only]

    def get_field(self, name: str) -> BaseField:
        """Return a local field or an owner field sliced to the view.

        Parameters
        ----------
        name : str
            Field name.

        Returns
        -------
        field : BaseField
            Field object with first two axes matching the view.
        """
        if name in self._local_fields:
            return self._local_fields[name]
        field = self.owner.fields[name]
        return field.with_values(field.values[self.observation_selector])

    def copy(self, *, deep: bool = False, copy_owner: bool = True) -> AssetPanelView:
        """Return a copy of the view.

        Parameters
        ----------
        deep : bool, default=False
            If `True`, copy local field arrays and the observation selector when
            it is an ndarray.

        copy_owner : bool, default=True
            If `True`, copy the owner panel. If `False`, the copied view points
            to the same owner.

        Returns
        -------
        view : AssetPanelView
            Copied view.
        """
        owner = self.owner.copy(deep=deep) if copy_owner else self.owner
        selector = (
            self.observation_selector.copy()
            if deep and isinstance(self.observation_selector, np.ndarray)
            else self.observation_selector
        )
        local_fields = {
            name: field.copy(deep=deep) for name, field in self._local_fields.items()
        }
        return AssetPanelView(
            owner=owner,
            observation_selector=selector,
            _local_fields=local_fields,
        )

    def _validate_field(self, name: str, field: BaseField) -> None:
        """Validate a local field against view axes and masks."""
        _validate_field_against_axes(
            name=name,
            field=field,
            expected_shape=(self.n_observations, self.n_assets),
            active_mask=self.active_mask,
        )


class _ViewFieldMapping(Mapping[str, BaseField]):
    """Lazy mapping exposing field objects sized to an `AssetPanelView`."""

    def __init__(self, view: AssetPanelView) -> None:
        self._view = view

    def __getitem__(self, name: str) -> BaseField:
        return self._view.get_field(name)

    def __iter__(self) -> Iterable[str]:
        return iter(self._view.keys())

    def __len__(self) -> int:
        return len(self._view.keys())
