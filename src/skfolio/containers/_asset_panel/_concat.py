"""Concatenation for `AssetPanel` containers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from skfolio.containers._asset_panel._fields import (
    BaseField,
    Field3D,
    FieldCategorical,
)
from skfolio.containers._asset_panel._panel import AssetPanel

__all__ = ["concat"]


def concat(
    panels: Iterable[AssetPanel], *, verify_observations: bool = False
) -> AssetPanel:
    """Concatenate panels along the observation axis.

    This function performs strict vertical concatenation. All panels must have
    identical assets, field names, field types, field dtypes, categorical levels and
    3D field metadata. Field arrays, observations, `active_mask` and
    `estimation_mask` are concatenated on axis 0.

    Parameters
    ----------
    panels : iterable of AssetPanel
        Panels to concatenate. The iterable must contain at least one panel.

    verify_observations : bool, default=False
        If `True`, raise an error when the concatenated observation labels contain
        duplicates.

    Returns
    -------
    panel : AssetPanel
        Concatenated panel.

    Examples
    --------
    >>> from skfolio.containers import AssetPanel, concat
    >>>
    >>> panel_1 = AssetPanel(
    ...     fields={"returns": [[0.01, 0.02]]},
    ...     observations=["2024-01-01"],
    ...     assets=["A", "B"],
    ... )
    >>> panel_2 = AssetPanel(
    ...     fields={"returns": [[0.03, 0.04]]},
    ...     observations=["2024-01-02"],
    ...     assets=["A", "B"],
    ... )
    >>> concat([panel_1, panel_2])
    AssetPanel(n_observations=2, n_assets=2, n_fields=1)
    """
    if isinstance(panels, AssetPanel):
        raise TypeError(
            "`panels` must be an iterable of AssetPanel instances, not a single "
            "AssetPanel."
        )

    panel_list = list(panels)
    if not panel_list:
        raise ValueError("`panels` must contain at least one AssetPanel.")

    for position, panel in enumerate(panel_list):
        if not isinstance(panel, AssetPanel):
            raise TypeError(
                "`panels` must contain only AssetPanel instances; "
                f"item at position {position} has type {type(panel).__name__}."
            )

    reference_panel = panel_list[0]

    for position, panel in enumerate(panel_list[1:], start=1):
        _validate_concat_schema(
            reference_panel=reference_panel,
            panel=panel,
            position=position,
        )

    observations = np.concatenate([panel.observations for panel in panel_list], axis=0)
    if verify_observations:
        _validate_unique_observations(observations)

    fields = {
        name: reference_panel.fields[name].with_values(
            np.concatenate([panel.fields[name].values for panel in panel_list], axis=0)
        )
        for name in reference_panel.fields
    }

    return AssetPanel(
        fields=fields,
        observations=observations,
        assets=reference_panel.assets.copy(),
        active_mask=np.concatenate([panel.active_mask for panel in panel_list], axis=0),
        estimation_mask=np.concatenate(
            [panel.estimation_mask for panel in panel_list],
            axis=0,
        ),
        _validate_on_init=False,
    )


def _validate_concat_schema(
    *, reference_panel: AssetPanel, panel: AssetPanel, position: int
) -> None:
    """Validate non-concatenated axes and field schema for one panel."""
    if not np.array_equal(panel.assets, reference_panel.assets):
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

    if isinstance(reference_field, FieldCategorical):
        if not np.array_equal(field.levels, reference_field.levels):
            raise ValueError(
                f"Categorical field '{name}' has different levels in panel at "
                f"position {position}."
            )
        return

    if isinstance(reference_field, Field3D):
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
    reference_groups: np.ndarray | None,
    groups: np.ndarray | None,
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


def _validate_unique_observations(observations: np.ndarray) -> None:
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
