"""Asset Panel container."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import shutil
from collections.abc import Generator, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from skfolio.containers._asset_panel._base import _BaseAssetPanel
from skfolio.containers._asset_panel._fields import (
    MISSING_CATEGORY_CODE,
    BaseField,
    Field2D,
    Field3D,
    FieldCategorical,
    _as_observation_array,
    _as_pickle_safe_array,
    _validate_unique_labels,
)
from skfolio.containers._asset_panel._utils import (
    _as_field,
    _fill_2d,
    _format_observation_range,
    _materialize_selector,
    _normalize_positional_selector,
    _positions_from_labels,
    _positions_from_unique_labels,
    _raise_if_raw_replaces_typed_field,
    _selector_length,
    _slice_field_values,
    _to_dataframe,
    _validate_active_observations,
    _validate_field_against_axes,
    _validate_field_name,
)
from skfolio.containers._asset_panel._view import AssetPanelView
from skfolio.typing import AnyArray, ArrayLike, BoolArray, IntArray, StrArray

__all__ = ["AssetPanel"]


_METADATA_FILENAME = "_metadata.json"
_SAVE_VERSION = 1


@dataclass(slots=True)
class AssetPanel(_BaseAssetPanel):
    """Container for aligned cross-sectional asset data.

    `AssetPanel` stores asset-level fields (e.g. returns, volums, industry
    classification, factor exposure), over shared observation and asset axes.
    Every field uses `observations` as the first axis and `assets` as the second
    axis. These two axes always have shape (n_observations, n_assets).

    Three kinds of fields are supported:

    - 2D numeric fields (e.g. `returns`, `volume`, `market_cap`). These are stored as 2D
      numpy array in a `Field2D`.
    - 2D categorical fields (e.g. `country`, `industry`). These are stored as 2D numpy
      array of integer codes in a `FieldCategorical`, together with the category labels
      (e.g. "bank", "technology")
    - 3D numeric fields (e.g. `factor_exposures`). These are stored in a 3D numpy array
      in a `Field3D` with shape (n_observations, n_assets, n_third_axis), together with
      the third axis labels such as factor names (e.g. "size", "momentum") and optional
      group labels such as factor families (e.g. "style", "industry")

    The container is scikit-learn compatible: `len(panel)` returns `n_observations`
    and `panel[start:stop]` returns an `AssetPanelView` that can be passed to
    cross-validation and hyper-parameter tuning utilities. View creation is zero-copy
    when the observation selector is a slice or a contiguous index.

    Parameters
    ----------
    fields : dict[str, BaseField or ndarray]
        Field mapping. Raw arrays must be 2D and are converted to `Field2D`.
        Use `FieldCategorical` for integer-coded categorical fields and `Field3D` for 3D
        fields; both carry the metadata needed to interpret their codes or third axes.

    observations : ndarray of shape (n_observations,)
        Unique observation labels for the sample axis. NumPy dtypes are preserved.
        Object-dtype datetime-like labels are converted with NumPy, object-dtype string
        labels are converted to strings and mixed object labels are rejected.

    assets : ndarray of shape (n_assets,)
        Unique asset labels. Object-dtype labels are converted to strings.

    active_mask : boolean ndarray of shape (n_observations, n_assets), optional
        Boolean mask indicating whether each asset belongs to the universe at each
        observation. This separates assets that are outside the universe (e.g. before
        listing, after delisting) from assets that are in the universe but have a
        missing observation (e.g. holiday, missing quote). If `None`, all pairs are
        active.

    estimation_mask :boolean ndarray of shape (n_observations, n_assets), optional
        Boolean mask indicating which active `(observation, asset)` pairs should be used
        for estimator-specific statistics by `skfolio` estimators that suport it (e.g.
        :class:`~skfolio.preprocessing.CSStandardScaler`,
        :class:`~skfolio.moments.RegimeAdjustedEWCovariance`
        If `None`, all active pairs are eligible for estimation. Values are always
        enforced as a subset of `active_mask`.

    Attributes
    ----------
    n_observations : int
        Number of observations.

    n_assets : int
        Number of assets.

    n_fields : int
        Number of fields.

    Notes
    -----
    `AssetPanel` is an optimized middle ground between raw NumPy arrays and
    general-purpose labeled containers (e.g. pandas, polars, xarray). It is optimized
    for portfolio, factor and alpha workflows:

    - observations are the sample axis, so scikit-learn cross-validation can slice over
      time without grouping rows
    - assets are fixed on axis 1, while `active_mask` represents listings, delistings
      and other universe changes
    - payload arrays remain numeric, with categorical fields stored as integer codes
    - categorical levels, third-axis labels and third-axis groups are stored with their
      fields
    - shape, mask and universe invariants are validated by the container
    - estimators can rely on validated axes, masks and float-field universe invariants
      without repeating full container validation

    Compared with DataFrames, this avoids repeated `groupby`, `pivot` and
    index-alignment work while keeping the arrays ready for vectorized cross-sectional
    and time-series operations. Compared with xarray, it keeps a smaller API optimized
    for quant worklows.

    Performance benefits come from the same layout:

    - observation slices return `AssetPanelView` objects, so walk-forward folds can
      reuse field arrays
    - native `Field3D` avoid restacking large lists of 2D arrays
    - integer-coded categoricals and dense boolean masks keep memory and convertion
      overhead low
    - with `Parallel(..., prefer="threads")`, workers can read the same big panel in
      memory instead of receiving separate process copies. This is useful for
      NumPy-heavy computations where the numeric kernels release the GIL.
    - saved panels use `.npy` files and support memory-mapped loading with
      `AssetPanel.load(..., mmap_mode=...)`.


    **Indexing.**

    - `panel["name"]` returns the underlying field array.
    - `panel.fields["name"]` returns the field object with its metadata.
    - `panel[start:stop]` returns an `AssetPanelView` with shared field
      arrays.
    - `panel.isel(...)` and `panel.sel(...)` select observations and assets
      by position or label.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.containers import AssetPanel, concat
    >>>
    >>> n_observations = 252
    >>> observations = np.arange(n_observations)
    >>> assets = ["AAPL", "MSFT", "GOOG", "AMZN"]
    >>> n_assets = len(assets)
    >>>
    >>> panel = AssetPanel(
    ...     fields={
    ...         "returns": np.random.randn(n_observations, n_assets),
    ...         "volume": np.random.lognormal(size=(n_observations, n_assets)),
    ...         "market_cap": np.random.lognormal(size=(n_observations, n_assets)),
    ...     },
    ...     observations=observations,
    ...     assets=assets,
    ... )
    >>> panel.add_categorical_field(
    ...     name="industry",
    ...     values=np.random.randint(0, 3, size=(n_observations, n_assets)),
    ...     levels=["energy", "bank", "technology"],
    ... )
    AssetPanel(n_observations=252, n_assets=4, n_fields=4)
    >>> panel.add_3d_field(
    ...     name="factor_exposure",
    ...     values=np.random.randn(n_observations, n_assets, len(factor_labels)),
    ...     third_axis_name="factor",
    ...     third_axis_labels=["size", "momentum", "value"],
    ...     third_axis_groups=["style", "style", "style"],
    ... )
    AssetPanel(n_observations=252, n_assets=4, n_fields=5)
    >>> panel.n_observations, panel.n_assets, panel.n_fields
    (252, 4, 5)

    Access raw NumPy arrays:

    >>> returns = panel["returns"]
    >>> industry_codes = panel["industry"]
    >>> factor_exposure = panel["factor_exposure"]

    Use field objects or decoding helpers when labels or metadata are needed:

    >>> industry_labels = panel.decode_categorical_field("industry")
    >>> exposure_field = panel.fields["factor_exposure"]
    >>> exposure_field.third_axis_labels
    array(['size', 'momentum', 'value'], dtype='<U8')

    Slice observations:

    >>> view = panel[100:200]
    >>> view.n_observations
    100

    Select observations and assets by position or label:

    >>> panel.isel(observations=slice(0, 60), assets=[0, 1])
    AssetPanel(n_observations=60, n_assets=2, n_fields=5)
    >>> panel.sel(observations=slice(0, 59), assets=["AAPL", "MSFT"])
    AssetPanel(n_observations=60, n_assets=2, n_fields=5)

    Select entries from a 3D field by third-axis label or group:

    >>> panel.sel_3d("factor_exposure", labels="momentum").shape
    (252, 4)
    >>> panel.sel_3d("factor_exposure", groups="style").shape
    (252, 4, 3)

    Convert to pandas:

    >>> df = panel.to_dataframe(fields=["returns", "industry"], output_format="wide")

    Get summary and inspect missingness:

    >>> summary = panel.describe(by="industry")
    >>> report = panel.info()

     Clean selected fields:

    >>> panel.ffill("returns", inplace=False)
    AssetPanel(n_observations=252, n_assets=4, n_fields=5)
    >>> panel.bfill("returns", inplace=False)
    AssetPanel(n_observations=252, n_assets=4, n_fields=5)
    >>> panel.align_active_mask_to("returns")
    0

    Rename field and drop assets:

    >>> panel.rename({"market_cap": "capitalization"})
    AssetPanel(n_observations=252, n_assets=4, n_fields=5)
    >>> panel.drop(assets=["AMZN"])

     Concatenate, copy, save and load panels:

    AssetPanel(n_observations=252, n_assets=3, n_fields=5)
    >>> concat([panel[:126], panel[126:]])
    AssetPanel(n_observations=252, n_assets=4, n_fields=5)
    >>> panel.copy(deep=True)
    AssetPanel(n_observations=252, n_assets=4, n_fields=5)
    >>> panel.save("asset_panel")
    >>> loaded = AssetPanel.load("asset_panel", mmap_mode="r")
    """

    fields: dict[str, BaseField]
    observations: AnyArray
    assets: StrArray | list[str]
    active_mask: BoolArray | None = None
    estimation_mask: BoolArray | None = None

    _validate_on_init: bool = True

    def __post_init__(self) -> None:
        """Normalize fields, labels and masks after dataclass initialization."""
        if not self.fields:
            raise ValueError("`fields` must contain at least one entry.")

        self.fields = {
            name: _as_field(field) for name, field in dict(self.fields).items()
        }

        for name in self.fields:
            _validate_field_name(name)

        self.observations = _as_observation_array(self.observations)
        self.assets = _as_pickle_safe_array(self.assets)
        expected_shape = (self.n_observations, self.n_assets)
        if self.active_mask is None:
            self.active_mask = np.ones(expected_shape, dtype=np.bool_)
        else:
            self.active_mask = np.asarray(self.active_mask)
        if self.estimation_mask is None:
            self.estimation_mask = np.ones(expected_shape, dtype=np.bool_)
        else:
            self.estimation_mask = np.asarray(self.estimation_mask)

        # validate observations and assets
        """Validate axis labels."""
        if self.observations.ndim != 1:
            raise ValueError("observations must be a 1D array.")
        if self.assets.ndim != 1:
            raise ValueError("assets must be a 1D array.")
        if self.observations.shape != (self.n_observations,):
            raise ValueError(f"observations must have shape ({self.n_observations},).")
        if self.assets.shape != (self.n_assets,):
            raise ValueError(f"assets must have shape ({self.n_assets},).")
        _validate_unique_labels(self.observations, name="observations")
        _validate_unique_labels(self.assets, name="assets")

        self._validate_masks_shape_and_type()
        self._enforce_estimation_mask_subset()
        self._validate_masks_invariants()
        if self._validate_on_init:
            for name, field in self.fields.items():
                self._validate_field(name, field)
        self._lock_masks()

    def __len__(self) -> int:
        """Return the number of observations."""
        return self.n_observations

    def __getitem__(self, key: Any) -> AnyArray | AssetPanelView:
        """Return field values or an observation view. Slice selectors are zero-copy.
        Integer or boolean array selectors follow NumPy fancy-indexing semantics on
         access and may copy.

        Parameters
        ----------
        key : str, int, slice or array-like
            If `key` is a string, return the underlying field array. Otherwise,
            interpret `key` as an observation selector and return an `AssetPanelView`.

        Returns
        -------
        values or view : ndarray or AssetPanelView
            For a string key, the underlying field array (no copy). For an observation
            selector, an `AssetPanelView` whose field arrays are views into this panel.
        """
        if isinstance(key, tuple):
            raise TypeError(
                "AssetPanel supports field access by name or one-dimensional "
                "observation selectors. Use isel(..., assets=...) or "
                "sel(..., assets=...) to select assets."
            )
        if isinstance(key, str):
            return self.fields[key].values
        selector = _normalize_positional_selector(self.n_observations, key)
        return AssetPanelView(owner=self, observation_selector=selector)

    def __setitem__(self, name: str, value: BaseField | ArrayLike) -> None:
        """Add or replace a field.

        Raw numpy arrays are coerced to `Field2D`. Pass `FieldCategorical` or `Field3D`
        explicitly when metadata is required. Replacing an existing `FieldCategorical`
        or `Field3D` with a raw numpy array raises `TypeError`, since this would
        silently discard categorical levels or third-axis metadata.

        Parameters
        ----------
        name : str
            Field name.

        value : BaseField or array-like
            Field object or raw 2D array with shape (n_observations, n_assets).
            When `value` is a raw float array, entries outside `active_mask` must be NaN.
        """
        _validate_field_name(name)
        if name in self.fields:
            _raise_if_raw_replaces_typed_field(
                name=name, value=value, existing_field=self.fields[name]
            )
        field = _as_field(value)
        self._validate_field(name, field)
        self.fields[name] = field

    def __delitem__(self, name: str) -> None:
        """Delete a field.

        Parameters
        ----------
        name : str
            Field name to remove.

        Raises
        ------
        KeyError
            If `name` is not a known field.

        ValueError
            If `name` is the only field. An `AssetPanel` must contain at  least one
            field so that observation and asset axes remain defined.
        """
        if name not in self.fields:
            raise KeyError(name)
        if len(self.fields) == 1:
            raise ValueError("Cannot delete the last field from an AssetPanel.")
        del self.fields[name]

    def __repr__(self) -> str:
        """Return a compact representation with panel dimensions."""
        return (
            f"AssetPanel(n_observations={self.n_observations}, "
            f"n_assets={self.n_assets}, n_fields={self.n_fields})"
        )

    @contextmanager
    def edit_masks(self, *, _validate: bool = True) -> Generator[None, None, None]:
        """Temporarily make masks editable.

        On exit, `estimation_mask` is re-enforced as a subset of `active_mask`, float
        values outside the active universe are set to NaN and both masks are locked
        again.

        Parameters
        ----------
        _validate : bool, default=True
            Internal flag controlling the per-observation non-empty mask check after the
            context exits.

        Yields
        ------
        None
            The panel with editable mask arrays.
        """
        self.active_mask.flags.writeable = True
        self.estimation_mask.flags.writeable = True
        try:
            yield
        finally:
            self._enforce_estimation_mask_subset()
            self._mask_inactive_float_values()
            self._lock_masks()

        if _validate:
            self._validate_masks_shape_and_type()
            self._validate_masks_invariants()

    @property
    def n_observations(self) -> int:
        """Number of observations."""
        return self._first_field().n_observations

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return self._first_field().n_assets

    @property
    def n_fields(self) -> int:
        """Number of fields."""
        return len(self.fields)

    def keys(self) -> Iterable[str]:
        """Return field names.

        Returns
        -------
        names : KeysView of str
            Names of all fields in the panel.
        """
        return self.fields.keys()

    def get_field(self, name: str) -> BaseField:
        """Return a field object.

        Parameters
        ----------
        name : str
            Field name.

        Returns
        -------
        field : BaseField
            Field object that owns the field values and metadata.
        """
        return self.fields[name]

    def isel(
        self, *, observations: Any = None, assets: Any = None
    ) -> AssetPanel | AssetPanelView:
        """Select observations and assets by integer position.

        Parameters
        ----------
        observations : int, slice, array-like, or None, optional
            Positional selector for the observation axis. If `None`, all  observations
            are selected.

        assets : int, slice, array-like, or None, optional
            Positional selector for the asset axis. If `None`, assets are not sliced and
            an `AssetPanelView` is returned.

        Returns
        -------
        panel or view : AssetPanel or AssetPanelView
            Observation-only selections return a view. Selections that slice assets
            return a new panel.
        """
        observation_selector = _normalize_positional_selector(
            self.n_observations, slice(None) if observations is None else observations
        )
        if assets is None:
            return AssetPanelView(owner=self, observation_selector=observation_selector)

        asset_selector = _normalize_positional_selector(self.n_assets, assets)
        return self._subset(
            observation_selector=observation_selector, asset_selector=asset_selector
        )

    def sel(
        self, *, observations: Any = None, assets: Any = None
    ) -> AssetPanel | AssetPanelView:
        """Select observations and assets by label.

        Parameters
        ----------
        observations : scalar, slice, iterable, or None, optional
            Observation labels to select. If `None`, all observations are selected.

        assets : scalar, slice, iterable, or None, optional
            Asset labels to select. If `None`, assets are not sliced and an
             `AssetPanelView` is returned.

        Returns
        -------
        panel or view : AssetPanel or AssetPanelView
            Observation-only selections return a view. Selections that slice assets
            return a new panel.
        """
        observation_selector = (
            slice(None)
            if observations is None
            else _positions_from_unique_labels(self.observations, observations)
        )
        if assets is None:
            return AssetPanelView(owner=self, observation_selector=observation_selector)

        asset_selector = _positions_from_unique_labels(self.assets, assets)
        return self._subset(
            observation_selector=observation_selector, asset_selector=asset_selector
        )

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

    def drop(self, *, observations: Any = None, assets: Any = None) -> AssetPanel:
        """Return a panel with selected labels removed.

        Parameters
        ----------
        observations : scalar, iterable, or None, optional
            Observation labels to remove.

        assets : scalar, iterable, or None, optional
            Asset labels to remove.

        Returns
        -------
        panel : AssetPanel
            New panel with the selected observations or assets removed.
        """
        if observations is None:
            observation_keep = slice(None)
        else:
            drop_positions = _materialize_selector(
                self.n_observations,
                _positions_from_unique_labels(self.observations, observations),
            )
            keep = np.ones(self.n_observations, dtype=np.bool_)
            keep[drop_positions] = False
            observation_keep = np.flatnonzero(keep)

        if assets is None:
            asset_keep = slice(None)
        else:
            drop_positions = _materialize_selector(
                self.n_assets,
                _positions_from_unique_labels(self.assets, assets),
            )
            keep = np.ones(self.n_assets, dtype=np.bool_)
            keep[drop_positions] = False
            asset_keep = np.flatnonzero(keep)

        return self._subset(
            observation_selector=observation_keep, asset_selector=asset_keep
        )

    def rename(
        self, fields: Mapping[str, str] | None = None, *, overwrite: bool = False
    ) -> AssetPanel:
        """Rename fields in place.

        Parameters
        ----------
        fields : mapping of str to str or None, optional
            Mapping from existing field names to replacement names.

        overwrite : bool, default=False
            If `True`, an existing target field can be replaced by a renamed field.
            If `False`, name conflicts raise `KeyError`.

        Returns
        -------
        self : AssetPanel
            The modified panel.
        """
        if fields is None:
            return self

        missing = [old for old in fields if old not in self.fields]
        if missing:
            raise KeyError(f"Fields not found: {missing}")

        existing = set(self.fields)
        targets = list(fields.values())
        for name in targets:
            _validate_field_name(name)

        duplicates = {name for name in targets if targets.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate target field names: {sorted(duplicates)}")

        conflicts = [
            new for old, new in fields.items() if new in existing and new != old
        ]
        if conflicts and not overwrite:
            raise KeyError(
                f"Target field names already exist: {conflicts}. "
                "Use overwrite=True to replace them."
            )

        renamed = {}
        for name, field in self.fields.items():
            if name in fields:
                renamed[fields[name]] = field
            elif overwrite and name in targets:
                continue
            else:
                renamed[name] = field

        self.fields = renamed
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

        `Field3D` entries are skipped with a warning when multiple fields are converted.
        Selecting a single `Field3D` raises `ValueError`.

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

    def add_categorical_field(
        self, name: str, values: ArrayLike, *, levels: StrArray | list[str] | list[Any]
    ) -> AssetPanel:
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

        Returns
        -------
        self : AssetPanel
            The modified panel.
        """
        self[name] = FieldCategorical(values, levels=levels)
        return self

    def add_3d_field(
        self,
        name: str,
        values: ArrayLike,
        *,
        third_axis_name: str,
        third_axis_labels: StrArray | list[str] | list[Any],
        third_axis_groups: StrArray | list[str] | list[Any] | None = None,
    ) -> AssetPanel:
        """Add or replace a numeric 3D field.

        This is a convenience wrapper around assigning a `Field3D`. The first two axes
        of `values` must be observations and assets with shape
        (n_observations, n_assets). The third axis stores a homogeneous block such as
        factors.

        Parameters
        ----------
        name : str
            Field name.

        values : array-like of shape (n_observations, n_assets, n_third_axis)
            Numeric 3D values.

        third_axis_name : str
            Name describing what the third axis represents (e.g. "factor")

        third_axis_labels : array-like of shape (n_third_axis,)
            Labels for entries along the third axis such as factor names (e.g. "size",
            "momentum")

        third_axis_groups : array-like of shape (n_third_axis,), optional
            Optional group label for each third-axis entry such factor families (e.g.
            "style", "industry")

        Returns
        -------
        self : AssetPanel
            The modified panel.
        """
        self[name] = Field3D(
            values,
            third_axis_name=third_axis_name,
            third_axis_labels=third_axis_labels,
            third_axis_groups=third_axis_groups,
        )
        return self

    def describe(self, *, by: str | None = None) -> pd.DataFrame:
        """Return a structured missingness summary.

        Parameters
        ----------
        by : str or None, optional
            Categorical field used to group missingness statistics. If `None`,
            missingness is summarized by field.

        Returns
        -------
        summary : pandas.DataFrame
            Missingness summary indexed by field, or by `(field, category)`
            when `by` is provided.
        """
        if by is None:
            rows = []
            active_entries_2d = int(self.active_mask.sum())
            for name, field in self.fields.items():
                missing = field.missing_mask
                active = self.active_mask
                if isinstance(field, Field3D):
                    active = active[:, :, np.newaxis]
                    type_name = "3D"
                elif isinstance(field, FieldCategorical):
                    type_name = "categorical"
                else:
                    type_name = "2D"

                total_entries = int(np.prod(field.values.shape))
                active_entries = (
                    active_entries_2d * field.values.shape[2]
                    if isinstance(field, Field3D)
                    else active_entries_2d
                )
                missing_total = int(missing.sum())
                missing_active = int((missing & active).sum())
                rows.append(
                    {
                        "field": name,
                        "type": type_name,
                        "dtype": str(field.values.dtype),
                        "missing_pct": missing_total / total_entries * 100,
                        "missing_pct_active": (
                            missing_active / active_entries * 100
                            if active_entries
                            else 0.0
                        ),
                    }
                )
            return pd.DataFrame(rows).set_index("field")

        category_field = self.fields[by]
        if not isinstance(category_field, FieldCategorical):
            raise TypeError(f"Field '{by}' is not categorical.")

        rows = []
        for code, label in enumerate(category_field.levels):
            category_mask = (category_field.values == code) & self.active_mask
            for name, field in self.fields.items():
                if name == by:
                    continue
                missing = field.missing_mask
                mask = category_mask
                if isinstance(field, Field3D):
                    mask = mask[:, :, np.newaxis]
                    n_entries = int(category_mask.sum()) * field.values.shape[2]
                else:
                    n_entries = int(mask.sum())
                n_missing = int((missing & mask).sum())
                rows.append(
                    {
                        "field": name,
                        by: label,
                        "n_entries": n_entries,
                        "missing_pct": n_missing / n_entries * 100
                        if n_entries
                        else 0.0,
                    }
                )
        return pd.DataFrame(rows).set_index(["field", by])

    def info(self) -> str:
        """Multi-line report with panel dimensions, mask coverage, field missingness
        and categorical field level coverage.

        Returns
        -------
        report : str
            Multi-line report.
        """
        n_assets = self.n_assets
        n_entries = self.n_observations * self.n_assets

        lines = [
            "AssetPanel Info",
            "=" * 60,
            f"Observations  : {self.n_observations:,}{_format_observation_range(self.observations)}",
            f"Assets        : {n_assets:,}",
            f"Fields        : {self.n_fields}",
            f"Panel entries : {n_entries:,}  (observations x assets)",
        ]

        active = self.active_mask
        active_entries = int(active.sum()) if active is not None else n_entries
        total_entries = 0
        headline_missing = 0
        total_active_entries = 0
        headline_missing_active = 0
        for field in self.fields.values():
            if field.ndim != 2:
                continue
            total_entries += n_entries
            total_active_entries += active_entries
            missing = field.missing_mask
            missing_count = int(missing.sum())
            missing_active = (
                int((missing & active).sum()) if active is not None else missing_count
            )
            headline_missing += missing_count
            headline_missing_active += missing_active
        if total_entries > 0:
            pct_miss = headline_missing / total_entries * 100
            pct_miss_u = (
                headline_missing_active / total_active_entries * 100
                if total_active_entries > 0
                else 0.0
            )
            lines.append(
                f"Missing       : {pct_miss:.1f}% total, {pct_miss_u:.1f}% in Active Mask"
            )

        def add_mask_block(
            title: str, mask: BoolArray | None, *, user_set: bool
        ) -> None:
            lines.append("")
            lines.append(title)
            lines.append("-" * len(title))
            if mask is None or not user_set:
                lines.append("Not set.")
                return
            in_mask = int(mask.sum())
            pct = in_mask / n_entries * 100 if n_entries > 0 else 0.0
            lines.append(
                f"In mask              : {in_mask:,} / {n_entries:,} entries"
                f"  ({pct:.1f}%)"
            )
            per_obs = mask.sum(axis=1)
            obs_min = int(per_obs.min())
            obs_med = int(np.median(per_obs))
            obs_max = int(per_obs.max())
            lines.append(
                f"Assets per obs       : min={obs_min:,}, "
                f"median={obs_med:,}, max={obs_max:,}"
            )
            per_asset = mask.sum(axis=0)
            n_in = int((per_asset > 0).sum())
            lines.append(f"Assets in mask       : {n_in:,} / {n_assets:,}")
            if n_in > 0:
                live_duration = per_asset[per_asset > 0]
                med = int(np.median(live_duration))
                lo = int(live_duration.min())
                hi = int(live_duration.max())
                lines.append(f"  median duration    : {med:,} observations")
                lines.append(f"  shortest / longest : {lo:,} / {hi:,} observations")

        active_set = active is not None and not np.all(active)
        add_mask_block("Active Mask", active, user_set=active_set)

        est = self.estimation_mask
        if est is not None and active is not None and np.array_equal(est, active):
            lines.append("")
            lines.append("Estimation Mask")
            lines.append("-" * len("Estimation Mask"))
            lines.append("Same as Active Mask.")
        else:
            est_set = est is not None and not np.all(est)
            add_mask_block("Estimation Mask", est, user_set=est_set)

        lines.append("")
        lines.append("Field Coverage")
        lines.append("-" * 60)

        name_w = max((min(len(n), 28) for n in self.fields), default=20)
        name_w = max(name_w, 8)

        hdr = (
            f"{'':>{name_w}s} {'dtype':>8s}   {'% missing':>9s}   {'% missing':>9s}"
            f"   {'fully missing':>14s}"
        )
        sub = (
            f"{'':>{name_w}s} {'':>8s}   {'total':>9s}   {'in Active Mask':>11s}"
            f"   {'assets (active)':>14s}"
        )
        lines.append(hdr)
        lines.append(sub)

        for name, field in self.fields.items():
            display_name = name if len(name) <= 28 else name[:25] + "..."
            arr = field.values

            if arr.ndim == 3:
                lines.append(
                    f"{display_name:>{name_w}s} {'3D':>8s}"
                    f"   {'--':>9s}   {'--':>11s}   {'--':>14s}"
                )
                continue

            dtype_s = str(arr.dtype)
            missing = field.missing_mask
            field_missing = int(missing.sum())
            pct_total = field_missing / n_entries * 100 if n_entries > 0 else 0.0

            if active is not None:
                active_entries = int(active.sum())
                miss_in_active = int((missing & active).sum())
                pct_active = (
                    miss_in_active / active_entries * 100 if active_entries > 0 else 0.0
                )

                per_asset_active = active.sum(axis=0).astype(float)
                per_asset_miss = (missing & active).sum(axis=0).astype(float)
                live = per_asset_active > 0
                fully_miss = int(((per_asset_miss == per_asset_active) & live).sum())
            else:
                pct_active = pct_total
                fully_miss = 0

            lines.append(
                f"{display_name:>{name_w}s} {dtype_s:>8s}   {pct_total:>8.1f}%"
                f"   {pct_active:>10.1f}%   {fully_miss:>14d}"
            )

        categorical_fields = {
            name: field
            for name, field in self.fields.items()
            if isinstance(field, FieldCategorical)
        }
        if categorical_fields:
            lines.append("")
            lines.append("Categorical Fields")
            lines.append("-" * 60)

            buckets = [(0, 10), (10, 20), (20, 50), (50, None)]

            for name, field in categorical_fields.items():
                arr = field.values
                levels = field.levels
                n_levels = len(levels)
                lines.append(f"{name} : {n_levels} levels")

                valid_mask = arr != MISSING_CATEGORY_CODE
                if active is not None:
                    valid_mask = valid_mask & active

                counts = np.zeros((arr.shape[0], n_levels), dtype=np.int64)
                for level_idx in range(n_levels):
                    counts[:, level_idx] = ((arr == level_idx) & valid_mask).sum(axis=1)

                min_counts = counts.min(axis=0)

                lines.append("  Min number of assets per level (over time):")
                for lo, hi in buckets:
                    if lo == 0:
                        in_bucket = min_counts < hi
                        label = f"< {hi}"
                    elif hi is not None:
                        in_bucket = (min_counts >= lo) & (min_counts < hi)
                        label = f"{lo} - {hi}"
                    else:
                        in_bucket = min_counts >= lo
                        label = f"> {lo}"

                    n_in = int(in_bucket.sum())
                    bucket_names = levels[in_bucket]

                    if n_in == 0:
                        lines.append(f"    {label:>7s} :  {n_in} levels")
                    elif n_in <= 6:
                        names_str = ", ".join(str(s) for s in bucket_names)
                        lines.append(f"    {label:>7s} :  {n_in} levels  ({names_str})")
                    else:
                        shown = ", ".join(str(s) for s in bucket_names[:4])
                        lines.append(
                            f"    {label:>7s} :  {n_in} levels"
                            f"  ({shown}, ... +{n_in - 4} more)"
                        )

        lines.append("")
        return "\n".join(lines)

    def ffill(
        self,
        fields: str | Iterable[str],
        *,
        limit: int | None = None,
        inplace: bool = True,
    ) -> AssetPanel:
        """Forward fill NaN values along the observation axis.

        Parameters
        ----------
        fields : str or iterable of str
            Numeric `Field2D` names to fill.

        limit : int or None, optional
            Maximum number of consecutive NaN values to fill. If `None`, all consecutive
            NaN values are eligible.

        inplace : bool, default=True
            If `True`, modify this panel. If `False`, return a shallow copy with filled
            fields.

        Returns
        -------
        panel : AssetPanel
            Modified panel or copied panel.
        """
        return self._fill(
            fields,
            method="ffill",
            limit=limit,
            inplace=inplace,
        )

    def bfill(
        self,
        fields: str | Iterable[str],
        *,
        limit: int | None = None,
        inplace: bool = True,
    ) -> AssetPanel:
        """Backward fill NaN values along the observation axis.

        Parameters
        ----------
        fields : str or iterable of str
            Numeric `Field2D` names to fill.

        limit : int or None, optional
            Maximum number of consecutive NaN values to fill. If `None`, all consecutive
            NaN values are eligible.

        inplace : bool, default=True
            If `True`, modify this panel. If `False`, return a shallow copy with filled
            fields.

        Returns
        -------
        panel : AssetPanel
            Modified panel or copied panel.
        """
        return self._fill(
            fields,
            method="bfill",
            limit=limit,
            inplace=inplace,
        )

    def align_active_mask_to(self, fields: str | Iterable[str]) -> int:
        """Align active periods to valid field values.

        For each asset, remove leading `active_mask` entries until the selected fields
        have valid values in the remaining active history. If an asset has no valid
        active value for a selected field, all active entries for that asset are
        removed. Only leading active entries are removed. Missing values after the first
        valid active value are left unchanged.

        Parameters
        ----------
        fields : str or iterable of str
            Field names used to determine when each asset can become active. Floating
            values must be finite, categorical values must not be missing, and 3D
             floating values must be finite across the third axis.

        Returns
        -------
        n_removed : int
            Number of `(observation, asset)` entries removed from `active_mask`.
        """
        field_names = [fields] if isinstance(fields, str) else list(fields)
        for name in field_names:
            if name not in self.fields:
                raise KeyError(f"Field '{name}' not found.")

        n_removed = 0
        new_active_mask = self.active_mask.copy()
        rows = np.arange(self.n_observations)[:, np.newaxis]

        for name in field_names:
            valid = ~self.fields[name].missing_mask
            valid_active = valid & new_active_mask
            has_active = new_active_mask.any(axis=0)
            has_valid = valid_active.any(axis=0)
            first_active = np.argmax(new_active_mask, axis=0)
            first_valid = np.argmax(valid_active, axis=0)

            remove = (
                new_active_mask
                & has_active
                & ((~has_valid) | ((first_valid > first_active) & (rows < first_valid)))
                & (rows >= first_active)
            )
            n_removed += int(remove.sum())
            new_active_mask[remove] = False

        _validate_active_observations(new_active_mask)
        with self.edit_masks():
            self.active_mask[:] = new_active_mask

        return n_removed

    def copy(self, *, deep: bool = False) -> AssetPanel:
        """Return a copy of the panel.

        Parameters
        ----------
        deep : bool, default=False
            If `True`, copy field arrays and label arrays. If `False`, field arrays and
            labels are shared. Masks are always copied so the copy owns independent
            lockable mask arrays.

        Returns
        -------
        panel : AssetPanel
            Copied panel.
        """
        return AssetPanel(
            fields={name: field.copy(deep=deep) for name, field in self.fields.items()},
            observations=self.observations.copy() if deep else self.observations,
            assets=self.assets.copy() if deep else self.assets,
            active_mask=self.active_mask.copy(),
            estimation_mask=self.estimation_mask.copy(),
            _validate_on_init=False,
        )

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Save the panel to a directory of `.npy` files.

        The directory contains one `.npy` file per field, small metadata files for
        categorical and third-axis labels and a `_metadata.json` manifest.
        Object-dtype axis labels and field metadata labels are converted to strings
        so the panel is loaded with `allow_pickle=False`.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination directory.

        overwrite : bool, default=False
            If `True`, replace an existing saved panel at `path`. Existing directories
            that do not contain `_metadata.json` are never overwritten.
        """
        path = Path(path)
        if path.exists():
            if path.is_file():
                raise ValueError(f"Path '{path}' is a file, not a directory.")
            if not (path / _METADATA_FILENAME).exists():
                raise ValueError(
                    f"Directory '{path}' exists but is not a saved AssetPanel "
                    f"(missing {_METADATA_FILENAME})."
                )
            if not overwrite:
                raise FileExistsError(
                    f"Directory '{path}' already contains a saved AssetPanel. "
                    "Use overwrite=True to replace it."
                )
            shutil.rmtree(path)

        for name in self.fields:
            _validate_field_name(name)

        path.mkdir(parents=True)
        fields_dir = path / "fields"
        fields_dir.mkdir()
        levels_dir = path / "levels"
        third_axis_dir = path / "third_axis"

        np.save(path / "observations.npy", _as_pickle_safe_array(self.observations))
        np.save(path / "assets.npy", _as_pickle_safe_array(self.assets))

        active_mask_storage = "all_true"
        if not self.active_mask.all():
            np.save(path / "active_mask.npy", self.active_mask)
            active_mask_storage = "file"

        estimation_mask_storage = "all_true"
        if not self.estimation_mask.all():
            np.save(path / "estimation_mask.npy", self.estimation_mask)
            estimation_mask_storage = "file"

        field_metadata: dict[str, dict[str, Any]] = {}
        for name, field in self.fields.items():
            np.save(fields_dir / f"{name}.npy", field.values)
            entry: dict[str, Any] = {
                "type": type(field).__name__,
                "dtype": str(field.values.dtype),
                "shape": list(field.values.shape),
            }

            if isinstance(field, FieldCategorical):
                levels_dir.mkdir(exist_ok=True)
                np.save(levels_dir / f"{name}.npy", _as_pickle_safe_array(field.levels))

            if isinstance(field, Field3D):
                third_axis_dir.mkdir(exist_ok=True)
                np.save(
                    third_axis_dir / f"{name}_labels.npy",
                    _as_pickle_safe_array(field.third_axis_labels),
                )
                entry["third_axis_name"] = field.third_axis_name
                if field.third_axis_groups is not None:
                    np.save(
                        third_axis_dir / f"{name}_groups.npy",
                        _as_pickle_safe_array(field.third_axis_groups),
                    )

            field_metadata[name] = entry

        metadata = {
            "version": _SAVE_VERSION,
            "n_observations": self.n_observations,
            "n_assets": self.n_assets,
            "active_mask": active_mask_storage,
            "estimation_mask": estimation_mask_storage,
            "fields": field_metadata,
        }
        with open(path / _METADATA_FILENAME, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        mmap_mode: str | None = None,
        fields: list[str] | None = None,
    ) -> AssetPanel:
        """Load a panel saved with `save`.

        Parameters
        ----------
        path : str or pathlib.Path
            Directory containing a saved panel.

        mmap_mode : str or None, optional
            Memory-mapping mode passed to `numpy.load` for field and mask arrays.
            Use `r` for read-only memory maps.

        fields : list of str or None, optional
            Field names to load. If `None`, all fields are loaded.

        Returns
        -------
        panel : AssetPanel
            Loaded panel.
        """
        path = Path(path)
        metadata_path = path / _METADATA_FILENAME
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Not a saved AssetPanel: '{path}' (missing {_METADATA_FILENAME})."
            )

        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata["version"] > _SAVE_VERSION:
            raise ValueError(
                f"Unsupported AssetPanel format version {metadata['version']}."
            )

        n_observations = metadata["n_observations"]
        n_assets = metadata["n_assets"]
        observations = _as_observation_array(
            np.load(path / "observations.npy", allow_pickle=False)
        )
        assets = _as_pickle_safe_array(np.load(path / "assets.npy", allow_pickle=False))

        if metadata["active_mask"] == "all_true":
            active_mask = np.ones((n_observations, n_assets), dtype=np.bool_)
        else:
            active_mask = np.load(
                path / "active_mask.npy",
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )

        if metadata["estimation_mask"] == "all_true":
            estimation_mask = np.ones((n_observations, n_assets), dtype=np.bool_)
        else:
            estimation_mask = np.load(
                path / "estimation_mask.npy",
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )

        available = list(metadata["fields"])
        available_set = set(available)
        requested = available if fields is None else list(fields)
        missing = [name for name in requested if name not in available_set]
        if missing:
            raise KeyError(f"Requested fields not found: {sorted(missing)}")

        loaded_fields: dict[str, BaseField] = {}
        fields_dir = path / "fields"
        levels_dir = path / "levels"
        third_axis_dir = path / "third_axis"
        for name in requested:
            entry = metadata["fields"][name]
            values = np.load(
                fields_dir / f"{name}.npy",
                allow_pickle=False,
                mmap_mode=mmap_mode,
            )
            field_type = entry["type"]
            if field_type == "Field2D":
                loaded_fields[name] = Field2D(values)
            elif field_type == "FieldCategorical":
                levels = _as_pickle_safe_array(
                    np.load(levels_dir / f"{name}.npy", allow_pickle=False)
                )
                loaded_fields[name] = FieldCategorical(values, levels=levels)
            elif field_type == "Field3D":
                labels = _as_pickle_safe_array(
                    np.load(third_axis_dir / f"{name}_labels.npy", allow_pickle=False)
                )
                groups = None
                if (third_axis_dir / f"{name}_groups.npy").exists():
                    groups = _as_pickle_safe_array(
                        np.load(
                            third_axis_dir / f"{name}_groups.npy",
                            allow_pickle=False,
                        )
                    )
                loaded_fields[name] = Field3D(
                    values,
                    third_axis_name=entry["third_axis_name"],
                    third_axis_labels=labels,
                    third_axis_groups=groups,
                )
            else:
                raise ValueError(f"Unsupported field type '{field_type}'.")

        return cls(
            fields=loaded_fields,
            observations=observations,
            assets=assets,
            active_mask=active_mask,
            estimation_mask=estimation_mask,
            _validate_on_init=False,
        )

    def _first_field(self) -> BaseField:
        """Return the first field used to derive panel dimensions."""
        return next(iter(self.fields.values()))

    def _lock_masks(self) -> None:
        """Make mask arrays read-only outside `edit_masks`."""
        self.active_mask.flags.writeable = False
        self.estimation_mask.flags.writeable = False

    def _validate_masks_shape_and_type(self) -> None:
        """Validate mask shape and dtype."""
        expected_mask_shape = (self.n_observations, self.n_assets)
        if self.active_mask.shape != expected_mask_shape:
            raise ValueError(f"active_mask must have shape {expected_mask_shape}.")
        if self.estimation_mask.shape != expected_mask_shape:
            raise ValueError(f"estimation_mask must have shape {expected_mask_shape}.")
        if self.active_mask.dtype != np.bool_:
            raise ValueError("active_mask must have dtype bool.")
        if self.estimation_mask.dtype != np.bool_:
            raise ValueError("estimation_mask must have dtype bool.")

    def _validate_masks_invariants(self) -> None:
        """Validate mask subset relation and non-empty observations."""
        if (self.estimation_mask & ~self.active_mask).any():
            raise ValueError("estimation_mask must be a subset of active_mask.")
        _validate_active_observations(self.active_mask)
        if not self.estimation_mask.any(axis=1).all():
            empty_idx = np.where(~self.estimation_mask.any(axis=1))[0]
            raise ValueError(
                "`estimation_mask` must contain at least one estimable asset for every "
                "observation after intersection with `active_mask`. "
                "`estimation_mask` is always enforced as a subset of `active_mask`; "
                "entries set to True where `active_mask` is False are ignored. "
                f"Found {len(empty_idx)} observation(s) with none; "
                f"first failing position is {empty_idx[0]}."
            )

    def _enforce_estimation_mask_subset(self) -> None:
        """Enforce `estimation_mask` as a subset of `active_mask`."""
        # Avoid mutating read-only inputs when load(..., mmap_mode="r").
        if self.estimation_mask.flags.writeable:
            self.estimation_mask &= self.active_mask
        else:
            self.estimation_mask = np.asarray(self.estimation_mask & self.active_mask)

    def _validate_field(self, name: str, field: BaseField) -> None:
        """Validate a field against panel shape and active-mask invariants."""
        _validate_field_against_axes(
            name=name,
            field=field,
            expected_shape=(self.n_observations, self.n_assets),
            active_mask=self.active_mask,
        )

    def _mask_inactive_float_values(self) -> None:
        """Set floating-point field values outside `active_mask` to NaN."""
        out = ~self.active_mask
        if not out.any():
            return
        for field in self.fields.values():
            if np.issubdtype(field.values.dtype, np.floating):
                if isinstance(field, Field3D):
                    field.values[out, :] = np.nan
                else:
                    field.values[out] = np.nan

    def _subset(
        self,
        *,
        observation_selector: slice | IntArray,
        asset_selector: slice | IntArray,
    ) -> AssetPanel:
        """Return a panel restricted to selected observations and assets."""
        if _selector_length(self.n_observations, observation_selector) == 0:
            raise ValueError(
                "Cannot remove all observations; at least one must remain."
            )
        if _selector_length(self.n_assets, asset_selector) == 0:
            raise ValueError("Cannot remove all assets; at least one must remain.")

        new_fields = {
            name: field.with_values(
                _slice_field_values(field, observation_selector, asset_selector)
            )
            for name, field in self.fields.items()
        }
        return AssetPanel(
            fields=new_fields,
            observations=self.observations[observation_selector],
            assets=self.assets[asset_selector],
            active_mask=self.active_mask[observation_selector, :][:, asset_selector],
            estimation_mask=(
                self.estimation_mask[observation_selector, :][:, asset_selector]
            ),
            _validate_on_init=False,
        )

    def _fill(
        self,
        fields: str | Iterable[str],
        method: Literal["ffill", "bfill"],
        *,
        limit: int | None,
        inplace: bool,
    ) -> AssetPanel:
        """Fill floating 2D fields along observations."""
        field_names = [fields] if isinstance(fields, str) else list(fields)
        panel = self if inplace else self.copy()
        for name in field_names:
            if name not in panel.fields:
                raise KeyError(f"Field '{name}' not found.")
            field = panel.fields[name]
            if not isinstance(field, Field2D) or isinstance(field, FieldCategorical):
                raise TypeError(f"{method} only supports numeric Field2D entries.")
            if not np.issubdtype(field.values.dtype, np.floating):
                raise TypeError(f"{method} only supports floating Field2D entries.")
            values = _fill_2d(
                field.values,
                method=method,
                limit=limit,
                mask=panel.active_mask,
            )
            values[~panel.active_mask] = np.nan
            panel.fields[name] = field.with_values(values)
        return panel
