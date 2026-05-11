"""Base Asset Panel."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Literal, TypeVar

import numpy as np
import pandas as pd

from skfolio.containers._asset_panel._fields import (
    BaseField,
    Field2D,
    Field3D,
    FieldCategorical,
    InactivePolicy,
)
from skfolio.containers._asset_panel._utils import (
    _materialize_selector,
    _positions_from_labels,
    _positions_from_unique_labels,
    _to_dataframe,
)
from skfolio.typing import AnyArray, ArrayLike, StrArray

_BaseAssetPanelT = TypeVar("_BaseAssetPanelT", bound="_BaseAssetPanel")


class _BaseAssetPanel(ABC):
    """Base AssetPanel class to share read utilities between `AssetPanel` and
    `AssetPanelView`.
    """

    def __contains__(self, name: str) -> bool:
        """Check whether a field exists."""
        return name in self.keys()

    @property
    @abstractmethod
    def n_observations(self) -> int:
        """Number of observations."""

    @property
    @abstractmethod
    def n_assets(self) -> int:
        """Number of assets."""

    @abstractmethod
    def keys(self) -> Iterable[str]:
        """Return field names."""

    @abstractmethod
    def get_field(self, name: str) -> BaseField:
        """Return a field object."""

    @abstractmethod
    def __setitem__(self, name: str, value: BaseField | ArrayLike) -> None:
        """Add or replace a field."""

    @property
    def shape(self) -> tuple[int]:
        """Shape tuple used by scikit-learn sample indexing."""
        return (len(self),)

    @property
    def ndim(self) -> int:
        """Number of dimensions used by scikit-learn sample indexing."""
        return 1

    def decode_categorical_field(
        self, name: str, *, missing_label: str = "MISSING"
    ) -> StrArray:
        """Decode a categorical field to labels.

        Parameters
        ----------
        name : str
            Name of a `FieldCategorical` field.

        missing_label : str, default="MISSING"
            Label assigned to missing or out-of-bound codes.

        Returns
        -------
        decoded : ndarray
            Decoded labels with shape (n_observations, n_assets).
        """
        field = self.get_field(name)
        if not isinstance(field, FieldCategorical):
            raise TypeError(f"Field '{name}' is not categorical.")
        return field.decode(missing_label=missing_label)

    def sel_3d(self, name: str, *, labels: Any = None, groups: Any = None) -> AnyArray:
        """Select entries from the third axis of a 3D field by label.

        Exactly one of `labels` or `groups` must be provided. Selecting a single label
        returns a 2D array with shape (n_observations, n_assets). Selecting multiple
        labels or any group returns a 3D array whose first two axes are unchanged.

        Parameters
        ----------
        name : str
            Name of a `Field3D`.

        labels : scalar, iterable, or None, optional
            Third-axis labels to select.

        groups : scalar, iterable, or None, optional
            Third-axis group labels to select. The field must define `third_axis_groups`.

        Returns
        -------
        values : ndarray
            Selected values. A scalar `labels` selection returns 2D values. All other
            selections return 3D values.
        """
        has_labels = labels is not None
        has_groups = groups is not None
        if has_labels == has_groups:
            raise ValueError("Exactly one of labels or groups must be provided.")

        field = self.get_field(name)
        if not isinstance(field, Field3D):
            raise TypeError(f"Field '{name}' is not a Field3D.")

        if has_labels:
            selector = _positions_from_unique_labels(field.third_axis_labels, labels)
            positions = _materialize_selector(len(field.third_axis_labels), selector)
            if isinstance(labels, (str, bytes)) or np.isscalar(labels):
                return field.values[:, :, int(positions[0])]
            return field.values[:, :, positions]

        if field.third_axis_groups is None:
            raise ValueError(f"Field '{name}' does not define third-axis groups.")
        selector = _positions_from_labels(field.third_axis_groups, groups)
        positions = _materialize_selector(len(field.third_axis_groups), selector)
        return field.values[:, :, positions]

    def add_2d_field(
        self: _BaseAssetPanelT,
        name: str,
        values: ArrayLike,
        *,
        inactive_policy: InactivePolicy = InactivePolicy.MISSING,
    ) -> _BaseAssetPanelT:
        """Add or replace a numeric 2D field.

        Parameters
        ----------
        name : str
            Field name.

        values : array-like of shape (n_observations, n_assets)
            Numeric 2D values.

        inactive_policy : InactivePolicy, default=InactivePolicy.MISSING
            Validation policy for values outside `active_mask`.

        Returns
        -------
        self : BaseAssetPanel
            The modified container.
        """
        self[name] = Field2D(values, inactive_policy=inactive_policy)
        return self

    def add_3d_field(
        self: _BaseAssetPanelT,
        name: str,
        values: ArrayLike,
        *,
        third_axis_name: str,
        third_axis_labels: StrArray | list[str] | list[Any],
        third_axis_groups: StrArray | list[str] | list[Any] | None = None,
        inactive_policy: InactivePolicy = InactivePolicy.MISSING,
    ) -> _BaseAssetPanelT:
        """Add or replace a numeric 3D field.

        This is a convenience wrapper around assigning a `Field3D`. The first two axes
        of `values` must be observations and assets with shape (n_observations, n_assets).
        The third axis stores a homogeneous block such as factors.

        Parameters
        ----------
        name : str
            Field name.

        values : array-like of shape (n_observations, n_assets, n_third_axis)
            Numeric 3D values.

        third_axis_name : str
            Name describing what the third axis represents (e.g. `factor`).

        third_axis_labels : array-like of shape (n_third_axis,)
            Labels for entries along the third axis such as factor names (e.g. `size`,
            `momentum`).

        third_axis_groups : array-like of shape (n_third_axis,), optional
            Optional group label for each third-axis entry such as factor families (e.g.
            `style`, `industry`).

        inactive_policy : InactivePolicy, default=InactivePolicy.MISSING
            Validation policy for values outside `active_mask`.

        Returns
        -------
        self : BaseAssetPanel
            The modified container.
        """
        self[name] = Field3D(
            values,
            third_axis_name=third_axis_name,
            third_axis_labels=third_axis_labels,
            third_axis_groups=third_axis_groups,
            inactive_policy=inactive_policy,
        )
        return self

    def add_categorical_field(
        self: _BaseAssetPanelT,
        name: str,
        values: ArrayLike,
        *,
        levels: StrArray | list[str] | list[Any],
        inactive_policy: InactivePolicy = InactivePolicy.MISSING,
    ) -> _BaseAssetPanelT:
        """Add or replace a 2D categorical field.

        This is a convenience wrapper around assigning a `FieldCategorical`. The field
        values must be integer codes with shape (n_observations, n_assets). Code -1 is
        reserved for missing values. Code 0 selects `levels[0]`, code 1 selects
        `levels[1]` and so on.

        Parameters
        ----------
        name : str
            Field name.

        values : array-like of integers, shape (n_observations, n_assets)
            Integer category codes.

        levels : array-like of shape (n_levels,)
            Category labels selected by codes 0, 1 and so on.

        inactive_policy : InactivePolicy, default=InactivePolicy.MISSING
            Validation policy for codes outside `active_mask`.

        Returns
        -------
        self : BaseAssetPanel
            The modified container.
        """
        self[name] = FieldCategorical(
            values,
            levels=levels,
            inactive_policy=inactive_policy,
        )
        return self

    def to_dataframe(
        self,
        *,
        fields: str | Iterable[str] | None = None,
        assets: str | Iterable[str] | None = None,
        output_format: Literal["long", "wide"] = "long",
        decode_categoricals: bool = True,
    ) -> pd.DataFrame:
        """Convert 2D fields to a pandas DataFrame.

        Parameters
        ----------
        fields : str, iterable of str, or None, optional
            Field names to include. If a single string is passed, the result is a simple
            field DataFrame with observations as index and assets as columns. If `None`,
            all 2D fields are included.

        assets : str, iterable of str, or None, optional
            Asset labels to include. If `None`, all assets are included.

        output_format : {"long", "wide"}, default="long"
            Output format used when `fields` is not a single string. In long format,
            rows are indexed by `(observation, asset)` and filtered by `active_mask`.
            In wide format, columns are indexed by `(field, asset)`.

        decode_categoricals : bool, default=True
            If `True`, categorical codes are decoded to labels.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame representation of the selected 2D fields.
        """
        return _to_dataframe(
            self,
            fields=fields,
            assets=assets,
            output_format=output_format,
            decode_categoricals=decode_categoricals,
        )
