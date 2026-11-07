"""Concatenation for `AssetPanel` containers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.containers._asset_panel._base import _BaseAssetPanel
from skfolio.containers._asset_panel._fields import (
    BaseField,
    Field3D,
    FieldCategorical,
)
from skfolio.containers._asset_panel._panel import AssetPanel
from skfolio.typing import AnyArray, StrArray


def _concat_panels(
    *, panels: list[_BaseAssetPanel], observations: AnyArray
) -> AssetPanel:
    """Concatenate panels along the observation axis.

    This function performs strict vertical concatenation. All panels or panel views must
    have identical assets, field names, field types, field dtypes, `inactive_policy`
    values, categorical levels and 3D field metadata. Field arrays, `active_mask` and
    `estimation_mask` are concatenated on axis 0.

    Parameters
    ----------
    panels : list of AssetPanel or AssetPanelView
        Panels or panel views to concatenate. The list must contain at least one object.

    observations : AnyArray of shape (n_observations,)
        Concatenated observation labels, in the same order as `panels`.

    Returns
    -------
    panel : AssetPanel
        Concatenated panel.
    """
    reference_panel = panels[0]

    for position, panel in enumerate(panels[1:], start=1):
        _validate_concat_schema(
            reference_panel=reference_panel,
            panel=panel,
            position=position,
        )

    fields = {
        name: reference_panel.fields[name].with_values(
            np.concatenate([panel.fields[name].values for panel in panels], axis=0)
        )
        for name in reference_panel.fields
    }

    return AssetPanel(
        fields=fields,
        observations=observations,
        asset_names=reference_panel.asset_names.copy(),
        active_mask=np.concatenate([panel.active_mask for panel in panels], axis=0),
        estimation_mask=np.concatenate(
            [panel.estimation_mask for panel in panels],
            axis=0,
        ),
        _validate_on_init=False,
    )


def _validate_concat_schema(
    *, reference_panel: _BaseAssetPanel, panel: _BaseAssetPanel, position: int
) -> None:
    """Validate non-concatenated axes and field schema for one panel."""
    if not np.array_equal(panel.asset_names, reference_panel.asset_names):
        raise ValueError(
            "All panels must have identical assets for observation-axis concat; "
            f"panel at position {position} differs."
        )

    reference_field_names = list(reference_panel.fields)
    field_names = list(panel.fields)
    if field_names != reference_field_names:
        raise ValueError(
            "All panels must have the same fields in the same order for concat; "
            f"panel at position {position} has fields {field_names}, expected "
            f"{reference_field_names}."
        )

    for name in reference_field_names:
        _validate_field_schema(
            name=name,
            reference_field=reference_panel.fields[name],
            field=panel.fields[name],
            position=position,
        )


def _validate_field_schema(
    *, name: str, reference_field: BaseField, field: BaseField, position: int
) -> None:
    """Validate that two fields can be concatenated on the observation axis."""
    if type(field) is not type(reference_field):
        raise TypeError(
            f"Field '{name}' has type {type(field).__name__} in panel at position "
            f"{position}; expected {type(reference_field).__name__}."
        )
    if field.values.dtype != reference_field.values.dtype:
        raise TypeError(
            f"Field '{name}' has dtype {field.values.dtype} in panel at position "
            f"{position}; expected {reference_field.values.dtype}."
        )
    if field.inactive_policy != reference_field.inactive_policy:
        raise ValueError(
            f"Field '{name}' has inactive_policy={field.inactive_policy!r} in panel "
            f"at position {position}; expected {reference_field.inactive_policy!r}."
        )

    if isinstance(reference_field, FieldCategorical) and isinstance(
        field, FieldCategorical
    ):
        if not np.array_equal(field.levels, reference_field.levels):
            raise ValueError(
                f"Categorical field '{name}' has different levels in panel at "
                f"position {position}."
            )
        return

    if isinstance(reference_field, Field3D) and isinstance(field, Field3D):
        if field.third_axis_name != reference_field.third_axis_name:
            raise ValueError(
                f"Field3D '{name}' has a different third_axis_name in panel at "
                f"position {position}."
            )
        if not np.array_equal(
            field.third_axis_labels, reference_field.third_axis_labels
        ):
            raise ValueError(
                f"Field3D '{name}' has different third_axis_labels in panel at "
                f"position {position}."
            )
        _validate_optional_groups(
            name=name,
            reference_groups=reference_field.third_axis_groups,
            groups=field.third_axis_groups,
            position=position,
        )


def _validate_optional_groups(
    *,
    name: str,
    reference_groups: StrArray | None,
    groups: StrArray | None,
    position: int,
) -> None:
    """Validate optional third-axis groups for a 3D field."""
    if reference_groups is None or groups is None:
        if reference_groups is not groups:
            raise ValueError(
                f"Field3D '{name}' has inconsistent third_axis_groups in panel at "
                f"position {position}."
            )
        return

    if not np.array_equal(groups, reference_groups):
        raise ValueError(
            f"Field3D '{name}' has different third_axis_groups in panel at "
            f"position {position}."
        )
