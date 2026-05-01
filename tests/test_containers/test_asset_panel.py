"""Unit tests for AssetPanel, AssetPanelView, and field classes."""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from skfolio.containers import (
    MISSING_CATEGORY_CODE,
    AssetPanel,
    AssetPanelView,
    Field2D,
    Field3D,
    FieldCategorical,
    concat,
)

N_OBS = 20
N_ASSETS = 4
OBSERVATIONS = np.arange(N_OBS)
ASSETS = np.array(["AAPL", "MSFT", "GOOG", "AMZN"])


def _conform_to_universe(fields, active_mask):
    if active_mask is None:
        return fields

    out = ~active_mask
    result = {}
    for name, field in fields.items():
        if isinstance(field, FieldCategorical):
            result[name] = field.copy(deep=True)
            continue

        if isinstance(field, Field3D):
            values = field.values.copy()
            if np.issubdtype(values.dtype, np.floating):
                values[out, :] = np.nan
            result[name] = field.with_values(values)
            continue

        if isinstance(field, Field2D):
            values = field.values.copy()
            if np.issubdtype(values.dtype, np.floating):
                values[out] = np.nan
            result[name] = field.with_values(values)
            continue

        values = np.asarray(field).copy()
        if np.issubdtype(values.dtype, np.floating):
            if values.ndim == 3:
                values[out, :] = np.nan
            else:
                values[out] = np.nan
        result[name] = values
    return result


def _make_panel(**overrides) -> AssetPanel:
    defaults = {
        "fields": {
            "x": np.arange(N_OBS * N_ASSETS, dtype=float).reshape(N_OBS, N_ASSETS)
        },
        "observations": OBSERVATIONS.copy(),
        "assets": ASSETS.copy(),
    }
    defaults.update(overrides)
    defaults["fields"] = _conform_to_universe(
        defaults["fields"],
        defaults.get("active_mask"),
    )
    return AssetPanel(**defaults)


def _make_full_panel(*, observations=None) -> AssetPanel:
    rng = np.random.default_rng(42)
    active_mask = np.ones((N_OBS, N_ASSETS), dtype=np.bool_)
    active_mask[:2, :3] = False

    estimation_mask = np.ones((N_OBS, N_ASSETS), dtype=np.bool_)
    estimation_mask[-2:, :3] = False

    fields = {
        "momentum": rng.standard_normal((N_OBS, N_ASSETS)),
        "sector": FieldCategorical(
            rng.integers(0, 3, size=(N_OBS, N_ASSETS), dtype=np.int32),
            levels=np.array(["Tech", "Health", "Finance"], dtype=object),
        ),
        "exposures": Field3D(
            rng.standard_normal((N_OBS, N_ASSETS, 3)),
            third_axis_name="factor",
            third_axis_labels=np.array(["mkt", "size", "value"], dtype=object),
            third_axis_groups=np.array(["market", "style", "style"], dtype=object),
        ),
    }

    return AssetPanel(
        fields=_conform_to_universe(fields, active_mask),
        observations=OBSERVATIONS.copy() if observations is None else observations,
        assets=ASSETS.copy(),
        active_mask=active_mask,
        estimation_mask=estimation_mask,
    )


def _assert_panels_equal(left: AssetPanel, right: AssetPanel) -> None:
    assert left.n_observations == right.n_observations
    assert left.n_assets == right.n_assets
    assert left.n_fields == right.n_fields
    np.testing.assert_array_equal(left.observations, right.observations)
    np.testing.assert_array_equal(left.assets, right.assets)
    np.testing.assert_array_equal(left.active_mask, right.active_mask)
    np.testing.assert_array_equal(left.estimation_mask, right.estimation_mask)

    assert list(left.fields) == list(right.fields)
    for name, right_field in right.fields.items():
        left_field = left.fields[name]
        assert type(left_field) is type(right_field)
        np.testing.assert_array_equal(left_field.values, right_field.values)
        assert left_field.values.dtype == right_field.values.dtype

        if isinstance(right_field, FieldCategorical):
            np.testing.assert_array_equal(left_field.levels, right_field.levels)
            assert left_field.levels.dtype == right_field.levels.dtype

        if isinstance(right_field, Field3D):
            assert left_field.third_axis_name == right_field.third_axis_name
            np.testing.assert_array_equal(
                left_field.third_axis_labels,
                right_field.third_axis_labels,
            )
            if right_field.third_axis_groups is None:
                assert left_field.third_axis_groups is None
            else:
                np.testing.assert_array_equal(
                    left_field.third_axis_groups,
                    right_field.third_axis_groups,
                )


class TestFields:
    def test_field_2d_rejects_wrong_ndim(self):
        with pytest.raises(ValueError, match="2D"):
            Field2D(np.ones((2, 3, 4)))

    def test_field_2d_rejects_object_dtype(self):
        with pytest.raises(ValueError, match="dtype=object"):
            Field2D(np.array([["a", "b"]], dtype=object))

    def test_categorical_decodes_codes_and_missing_values(self):
        field = FieldCategorical(
            np.array([[0, 1, MISSING_CATEGORY_CODE]], dtype=np.int32),
            levels=["Tech", "Health"],
        )

        decoded = field.decode()

        np.testing.assert_array_equal(decoded, [["Tech", "Health", "MISSING"]])
        np.testing.assert_array_equal(field.missing_mask, [[False, False, True]])
        assert isinstance(field, FieldCategorical)

    def test_categorical_validates_codes(self):
        with pytest.raises(ValueError, match="out of bounds"):
            FieldCategorical(np.array([[2]], dtype=np.int32), levels=["A", "B"])

        with pytest.raises(ValueError, match="integer dtype"):
            FieldCategorical(np.array([[0.0]]), levels=["A"])

    def test_categorical_requires_unique_levels(self):
        with pytest.raises(ValueError, match="unique"):
            FieldCategorical(np.array([[0, 1]], dtype=np.int32), levels=["A", "A"])

    def test_categorical_decode_always_preserves_missing_codes(self):
        field = FieldCategorical(
            np.array([[0, MISSING_CATEGORY_CODE]], dtype=np.int32),
            levels=["A", "B"],
        )

        decoded = field.decode()

        np.testing.assert_array_equal(decoded, [["A", "MISSING"]])

    def test_categorical_decode_empty_array_preserves_shape(self):
        field = FieldCategorical(
            np.empty((0, 2), dtype=np.int32),
            levels=["A", "B"],
        )

        decoded = field.decode()

        assert decoded.shape == (0, 2)

    def test_field_3d_metadata_and_with_values(self):
        field = Field3D(
            np.ones((2, 3, 2)),
            third_axis_name="factor",
            third_axis_labels=["mkt", "size"],
            third_axis_groups=["market", "style"],
        )

        new_field = field.with_values(np.zeros((2, 3, 2)))

        assert isinstance(new_field, Field3D)
        assert new_field.third_axis_name == "factor"
        np.testing.assert_array_equal(new_field.third_axis_labels, ["mkt", "size"])
        np.testing.assert_array_equal(new_field.third_axis_groups, ["market", "style"])
        np.testing.assert_array_equal(new_field.values, 0.0)

    def test_object_metadata_labels_are_converted_to_strings(self):
        categorical = FieldCategorical(
            np.array([[0, 1]], dtype=np.int32),
            levels=np.array(["Tech", "Health"], dtype=object),
        )
        field_3d = Field3D(
            np.ones((1, 2, 2)),
            third_axis_name="factor",
            third_axis_labels=np.array(["mkt", "size"], dtype=object),
            third_axis_groups=np.array(["market", "style"], dtype=object),
        )

        assert categorical.levels.dtype != object
        assert field_3d.third_axis_labels.dtype != object
        assert field_3d.third_axis_groups.dtype != object

    def test_field_3d_validates_third_axis_metadata(self):
        with pytest.raises(ValueError, match="third_axis_labels"):
            Field3D(
                np.ones((2, 3, 2)),
                third_axis_name="factor",
                third_axis_labels=["mkt"],
            )

        with pytest.raises(ValueError, match="third_axis_groups"):
            Field3D(
                np.ones((2, 3, 2)),
                third_axis_name="factor",
                third_axis_labels=["mkt", "size"],
                third_axis_groups=["style"],
            )

        with pytest.raises(ValueError, match="non-empty string"):
            Field3D(
                np.ones((2, 3, 2)),
                third_axis_name="",
                third_axis_labels=["mkt", "size"],
            )

        with pytest.raises(ValueError, match="unique"):
            Field3D(
                np.ones((2, 3, 2)),
                third_axis_name="factor",
                third_axis_labels=["mkt", "mkt"],
            )


class TestPanelConstruction:
    def test_basic_construction_coerces_raw_arrays_to_field_2d(self):
        panel = _make_panel()

        assert panel.n_observations == N_OBS
        assert panel.n_assets == N_ASSETS
        assert panel.n_fields == 1
        assert isinstance(panel.fields["x"], Field2D)
        assert panel.shape == (N_OBS,)
        assert panel.ndim == 1

    def test_object_axis_labels_are_converted_to_strings(self):
        panel = AssetPanel(
            fields={"x": np.ones((2, 2))},
            observations=np.array(["t0", "t1"], dtype=object),
            assets=np.array(["A", "B"], dtype=object),
        )

        assert panel.observations.dtype != object
        assert panel.assets.dtype != object
        np.testing.assert_array_equal(panel.observations, ["t0", "t1"])
        np.testing.assert_array_equal(panel.assets, ["A", "B"])

    def test_numpy_observation_dtype_is_preserved(self):
        observations = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]")

        panel = AssetPanel(
            fields={"x": np.ones((2, 2))},
            observations=observations,
            assets=np.array(["A", "B"]),
        )

        assert panel.observations.dtype == observations.dtype
        np.testing.assert_array_equal(panel.observations, observations)

    def test_python_date_observations_are_converted_to_numpy_datetime(self):
        panel = AssetPanel(
            fields={"x": np.ones((2, 2))},
            observations=[date(2024, 1, 1), date(2024, 1, 2)],
            assets=np.array(["A", "B"]),
        )

        assert np.issubdtype(panel.observations.dtype, np.datetime64)
        np.testing.assert_array_equal(
            panel.observations,
            np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[D]"),
        )

    def test_mixed_object_observations_raise(self):
        with pytest.raises(ValueError, match="object dtype"):
            AssetPanel(
                fields={"x": np.ones((2, 2))},
                observations=np.array([date(2024, 1, 1), "2024-01-02"], dtype=object),
                assets=np.array(["A", "B"]),
            )

    def test_duplicate_observations_and_assets_raise(self):
        with pytest.raises(ValueError, match="observations must be unique"):
            AssetPanel(
                fields={"x": np.ones((2, 2))},
                observations=np.array(["t0", "t0"]),
                assets=np.array(["A", "B"]),
            )

        with pytest.raises(ValueError, match="assets must be unique"):
            AssetPanel(
                fields={"x": np.ones((2, 2))},
                observations=np.array(["t0", "t1"]),
                assets=np.array(["A", "A"]),
            )

    def test_typed_fields_keep_metadata(self):
        panel = _make_full_panel()

        assert isinstance(panel.fields["sector"], FieldCategorical)
        assert isinstance(panel.fields["exposures"], Field3D)
        np.testing.assert_array_equal(
            panel.fields["sector"].levels,
            ["Tech", "Health", "Finance"],
        )
        assert panel.fields["exposures"].third_axis_name == "factor"

    def test_default_masks_are_all_true_and_read_only(self):
        panel = _make_panel()

        assert panel.active_mask.all()
        assert panel.estimation_mask.all()
        assert not panel.active_mask.flags.writeable
        assert not panel.estimation_mask.flags.writeable

    def test_custom_active_mask_is_preserved(self):
        active_mask = np.ones((N_OBS, N_ASSETS), dtype=np.bool_)
        active_mask[:, 0] = False

        panel = _make_panel(active_mask=active_mask)

        assert not panel.active_mask[:, 0].any()
        assert panel.active_mask[:, 1:].all()

    def test_repr_includes_panel_shape(self):
        panel = _make_panel()

        result = repr(panel)

        assert "AssetPanel" in result
        assert f"n_observations={N_OBS}" in result
        assert f"n_assets={N_ASSETS}" in result

    def test_estimation_mask_is_enforced_as_subset_of_active_mask(self):
        active_mask = np.ones((3, 2), dtype=np.bool_)
        active_mask[0, 0] = False
        estimation_mask = np.ones((3, 2), dtype=np.bool_)

        panel = _make_panel(
            fields={"x": np.ones((3, 2))},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
            estimation_mask=estimation_mask,
        )

        assert not panel.estimation_mask[0, 0]
        assert panel.estimation_mask[0, 1]

    def test_estimation_mask_empty_after_active_intersection_raises(self):
        active_mask = np.array([[False, True], [True, True], [True, True]])
        estimation_mask = np.array([[True, False], [True, True], [True, True]])

        with pytest.raises(ValueError, match="after intersection with `active_mask`"):
            _make_panel(
                fields={"x": np.ones((3, 2))},
                observations=np.arange(3),
                assets=np.array(["A", "B"]),
                active_mask=active_mask,
                estimation_mask=estimation_mask,
            )

    def test_empty_fields_raise(self):
        with pytest.raises(ValueError, match="fields"):
            AssetPanel(fields={}, observations=np.arange(2), assets=np.array(["A"]))

    def test_axis_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="observations"):
            AssetPanel(
                fields={"x": np.ones((3, 2))},
                observations=np.arange(4),
                assets=np.array(["A", "B"]),
            )

        with pytest.raises(ValueError, match="assets"):
            AssetPanel(
                fields={"x": np.ones((3, 2))},
                observations=np.arange(3),
                assets=np.array(["A"]),
            )

    def test_mask_shape_and_dtype_mismatch_raise(self):
        with pytest.raises(ValueError, match="active_mask"):
            AssetPanel(
                fields={"x": np.ones((3, 2))},
                observations=np.arange(3),
                assets=np.array(["A", "B"]),
                active_mask=np.ones((3, 3), dtype=np.bool_),
            )

        with pytest.raises(ValueError, match="estimation_mask"):
            AssetPanel(
                fields={"x": np.ones((3, 2))},
                observations=np.arange(3),
                assets=np.array(["A", "B"]),
                estimation_mask=np.ones((3, 2), dtype=int),
            )

    def test_empty_mask_rows_raise_at_construction(self):
        active_mask = np.ones((3, 2), dtype=np.bool_)
        active_mask[1, :] = False
        estimation_mask = np.ones((3, 2), dtype=np.bool_)
        estimation_mask[1, :] = False

        with pytest.raises(ValueError, match="active_mask"):
            AssetPanel(
                fields={
                    "x": _conform_to_universe({"x": np.ones((3, 2))}, active_mask)["x"]
                },
                observations=np.arange(3),
                assets=np.array(["A", "B"]),
                active_mask=active_mask,
            )

        with pytest.raises(ValueError, match="estimation_mask"):
            AssetPanel(
                fields={"x": np.ones((3, 2))},
                observations=np.arange(3),
                assets=np.array(["A", "B"]),
                estimation_mask=estimation_mask,
            )

    def test_field_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            AssetPanel(
                fields={"x": np.ones((3, 2)), "y": np.ones((3, 3))},
                observations=np.arange(3),
                assets=np.array(["A", "B"]),
            )

    @pytest.mark.parametrize(
        "name",
        [
            "",
            "has space",
            "path/sep",
            "back\\slash",
            "1starts_with_num",
            "CON",
        ],
    )
    def test_invalid_field_names_raise_at_construction(self, name):
        with pytest.raises(ValueError):
            AssetPanel(
                fields={name: np.ones((3, 2))},
                observations=np.arange(3),
                assets=np.array(["A", "B"]),
            )

    def test_float_values_outside_active_universe_raise(self):
        active_mask = np.array([[False, True], [True, True]])

        with pytest.raises(ValueError, match="finite value"):
            AssetPanel(
                fields={"x": np.array([[1.0, 2.0], [3.0, 4.0]])},
                observations=np.arange(2),
                assets=np.array(["A", "B"]),
                active_mask=active_mask,
            )

    def test_nan_values_outside_active_universe_are_accepted(self):
        active_mask = np.array([[False, True], [True, True]])

        panel = AssetPanel(
            fields={"x": np.array([[np.nan, 2.0], [3.0, 4.0]])},
            observations=np.arange(2),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
        )

        assert np.isnan(panel["x"][0, 0])

    def test_categorical_values_outside_active_universe_are_accepted(self):
        active_mask = np.array([[False, True], [True, True]])

        panel = AssetPanel(
            fields={
                "sector": FieldCategorical(
                    np.array([[0, 1], [1, 0]], dtype=np.int32),
                    levels=["A", "B"],
                ),
            },
            observations=np.arange(2),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
        )

        assert panel["sector"][0, 0] == 0


class TestPanelAccessAndMutation:
    def test_getitem_returns_field_array_without_copy(self):
        values = np.ones((N_OBS, N_ASSETS))
        panel = _make_panel(fields={"x": values})

        result = panel["x"]

        assert result is panel.fields["x"].values
        np.testing.assert_array_equal(result, values)

    def test_getitem_observation_slice_returns_view(self):
        panel = _make_panel()

        view = panel[5:15]

        assert isinstance(view, AssetPanelView)
        assert view.n_observations == 10
        np.testing.assert_array_equal(view["x"], panel["x"][5:15])
        assert np.shares_memory(view["x"], panel["x"])

    def test_getitem_integer_and_array_selectors_return_views(self):
        panel = _make_panel()

        single = panel[3]
        non_contiguous = panel[np.array([0, 5, 10])]

        assert isinstance(single, AssetPanelView)
        assert single.n_observations == 1
        np.testing.assert_array_equal(single["x"], panel["x"][3:4])
        np.testing.assert_array_equal(non_contiguous["x"], panel["x"][[0, 5, 10]])

    def test_getitem_missing_field_raises(self):
        panel = _make_panel()

        with pytest.raises(KeyError):
            panel["missing"]

    def test_getitem_tuple_selector_raises(self):
        panel = _make_panel()

        with pytest.raises(TypeError, match="one-dimensional observation selectors"):
            panel[:, :]

    def test_setitem_raw_array_adds_field_2d(self):
        panel = _make_panel()
        values = np.zeros((N_OBS, N_ASSETS))

        panel["y"] = values

        assert isinstance(panel.fields["y"], Field2D)
        np.testing.assert_array_equal(panel["y"], values)

    def test_setitem_accepts_typed_field_3d(self):
        panel = _make_panel()
        values = np.ones((N_OBS, N_ASSETS, 2))

        panel["exposures"] = Field3D(
            values,
            third_axis_name="factor",
            third_axis_labels=["mkt", "size"],
        )

        assert panel["exposures"].shape == (N_OBS, N_ASSETS, 2)
        assert panel.fields["exposures"].third_axis_name == "factor"

    def test_add_categorical_field_convenience(self):
        panel = _make_panel()
        codes = np.zeros((N_OBS, N_ASSETS), dtype=np.int32)

        result = panel.add_categorical_field("industry", codes, levels=["energy"])

        assert result is panel
        assert isinstance(panel.fields["industry"], FieldCategorical)
        np.testing.assert_array_equal(panel["industry"], codes)
        np.testing.assert_array_equal(panel.fields["industry"].levels, ["energy"])

    def test_add_3d_field_convenience(self):
        panel = _make_panel()
        values = np.ones((N_OBS, N_ASSETS, 2))

        result = panel.add_3d_field(
            "factor_exposure",
            values,
            third_axis_name="factor",
            third_axis_labels=["size", "momentum"],
            third_axis_groups=["style", "style"],
        )

        assert result is panel
        assert isinstance(panel.fields["factor_exposure"], Field3D)
        np.testing.assert_array_equal(panel["factor_exposure"], values)
        assert panel.fields["factor_exposure"].third_axis_name == "factor"
        np.testing.assert_array_equal(
            panel.fields["factor_exposure"].third_axis_labels,
            ["size", "momentum"],
        )
        np.testing.assert_array_equal(
            panel.fields["factor_exposure"].third_axis_groups,
            ["style", "style"],
        )

    def test_add_typed_field_conveniences_use_assignment_validation(self):
        panel = _make_panel()

        with pytest.raises(ValueError, match="integer dtype"):
            panel.add_categorical_field(
                "industry",
                np.zeros((N_OBS, N_ASSETS)),
                levels=["energy"],
            )

        with pytest.raises(ValueError, match="third_axis_labels"):
            panel.add_3d_field(
                "factor_exposure",
                np.ones((N_OBS, N_ASSETS, 2)),
                third_axis_name="factor",
                third_axis_labels=["size"],
            )

    def test_setitem_rejects_raw_replacement_of_typed_field(self):
        panel = _make_full_panel()

        with pytest.raises(TypeError, match="FieldCategorical"):
            panel["sector"] = panel["sector"]

        with pytest.raises(TypeError, match="Field3D"):
            panel["exposures"] = panel["exposures"]

    def test_setitem_allows_explicit_typed_replacement(self):
        panel = _make_full_panel()
        codes = np.zeros((N_OBS, N_ASSETS), dtype=np.int32)

        panel["sector"] = FieldCategorical(codes, levels=["Only"])

        np.testing.assert_array_equal(panel.fields["sector"].levels, ["Only"])

    def test_setitem_rejects_wrong_shape(self):
        panel = _make_panel()

        with pytest.raises(ValueError, match="shape"):
            panel["bad"] = np.ones((N_OBS, N_ASSETS + 1))

    def test_setitem_rejects_float_values_outside_active_universe(self):
        active_mask = np.array([[False, True], [True, True]])
        panel = _make_panel(
            fields={"x": np.array([[np.nan, 1.0], [2.0, 3.0]])},
            observations=np.arange(2),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
        )

        with pytest.raises(ValueError, match="finite value"):
            panel["y"] = np.array([[99.0, 1.0], [2.0, 3.0]])

    def test_setitem_invalid_field_name_raises(self):
        panel = _make_panel()

        with pytest.raises(ValueError):
            panel["has space"] = np.ones((N_OBS, N_ASSETS))

    def test_delete_field_and_last_field_guard(self):
        panel = _make_full_panel()

        del panel["momentum"]

        assert "momentum" not in panel
        with pytest.raises(ValueError, match="last field"):
            only = _make_panel()
            del only["x"]

    def test_rename_fields_in_place(self):
        panel = _make_full_panel()

        result = panel.rename({"momentum": "returns"})

        assert result is panel
        assert "returns" in panel
        assert "momentum" not in panel

    def test_rename_conflicts_raise_unless_overwrite_is_true(self):
        panel = _make_full_panel()

        with pytest.raises(KeyError, match="already exist"):
            panel.rename({"momentum": "sector"})

        panel.rename({"momentum": "sector"}, overwrite=True)

        assert set(panel.fields) == {"sector", "exposures"}
        assert isinstance(panel.fields["sector"], Field2D)

    def test_rename_missing_duplicate_and_invalid_targets_raise(self):
        panel = _make_full_panel()

        with pytest.raises(KeyError, match="not found"):
            panel.rename({"missing": "x"})

        with pytest.raises(ValueError, match="Duplicate"):
            panel.rename({"momentum": "x", "sector": "x"})

        with pytest.raises(ValueError):
            panel.rename({"momentum": "has space"})


class TestSelectionAndDrop:
    def test_isel_observation_only_returns_view(self):
        panel = _make_full_panel()

        view = panel.isel(observations=slice(2, 8))

        assert isinstance(view, AssetPanelView)
        np.testing.assert_array_equal(view.observations, OBSERVATIONS[2:8])
        np.testing.assert_array_equal(view["momentum"], panel["momentum"][2:8])

    def test_isel_with_assets_returns_new_panel_and_preserves_metadata(self):
        panel = _make_full_panel()

        selected = panel.isel(observations=slice(2, 8), assets=[1, 3])

        assert isinstance(selected, AssetPanel)
        np.testing.assert_array_equal(selected.assets, ASSETS[[1, 3]])
        assert selected["momentum"].shape == (6, 2)
        assert selected["exposures"].shape == (6, 2, 3)
        np.testing.assert_array_equal(
            selected.fields["exposures"].third_axis_labels,
            panel.fields["exposures"].third_axis_labels,
        )

    def test_isel_rejects_invalid_boolean_selector_shape(self):
        panel = _make_panel()

        with pytest.raises(ValueError, match="Boolean selector"):
            panel.isel(observations=np.array([True, False]))

    def test_isel_rejects_out_of_bounds_positions(self):
        panel = _make_panel()

        with pytest.raises(IndexError, match="out of bounds"):
            panel.isel(observations=N_OBS)

        with pytest.raises(IndexError, match="out of bounds"):
            panel.isel(assets=[0, N_ASSETS])

    def test_sel_uses_observation_and_asset_labels(self):
        observations = pd.date_range("2024-01-01", periods=N_OBS, freq="B").to_numpy()
        panel = _make_full_panel(observations=observations)

        selected = panel.sel(
            observations=slice(observations[2], observations[5]),
            assets=["MSFT", "AMZN"],
        )

        np.testing.assert_array_equal(selected.assets, ["MSFT", "AMZN"])
        np.testing.assert_array_equal(selected.observations, observations[2:6])

    def test_sel_unknown_labels_raise(self):
        panel = _make_panel()

        with pytest.raises(KeyError, match="Labels not found"):
            panel.sel(assets=["MISSING"])

    def test_sel_3d_scalar_label_returns_2d_values(self):
        panel = _make_full_panel()

        values = panel.sel_3d("exposures", labels="size")

        assert values.shape == (N_OBS, N_ASSETS)
        np.testing.assert_array_equal(values, panel["exposures"][:, :, 1])

    def test_sel_3d_multiple_labels_returns_3d_values(self):
        panel = _make_full_panel()

        values = panel.sel_3d("exposures", labels=["mkt", "value"])

        assert values.shape == (N_OBS, N_ASSETS, 2)
        np.testing.assert_array_equal(values, panel["exposures"][:, :, [0, 2]])

    def test_sel_3d_group_returns_3d_values(self):
        panel = _make_full_panel()

        values = panel.sel_3d("exposures", groups="style")

        assert values.shape == (N_OBS, N_ASSETS, 2)
        np.testing.assert_array_equal(values, panel["exposures"][:, :, [1, 2]])

    def test_sel_3d_validates_arguments_and_field_type(self):
        panel = _make_full_panel()

        with pytest.raises(ValueError, match="Exactly one"):
            panel.sel_3d("exposures")

        with pytest.raises(ValueError, match="Exactly one"):
            panel.sel_3d("exposures", labels="size", groups="style")

        with pytest.raises(TypeError, match="Field3D"):
            panel.sel_3d("momentum", labels="size")

        with pytest.raises(KeyError):
            panel.sel_3d("exposures", labels="missing")

        with pytest.raises(KeyError):
            panel.sel_3d("exposures", groups="missing")

    def test_sel_3d_groups_require_group_metadata(self):
        panel = _make_panel()
        panel.add_3d_field(
            "scenarios",
            np.ones((N_OBS, N_ASSETS, 2)),
            third_axis_name="scenario",
            third_axis_labels=["base", "stress"],
        )

        with pytest.raises(ValueError, match="does not define"):
            panel.sel_3d("scenarios", groups="macro")

    def test_drop_removes_observation_and_asset_labels(self):
        panel = _make_full_panel()

        dropped = panel.drop(observations=[0, 1], assets=["AAPL"])

        np.testing.assert_array_equal(dropped.observations, OBSERVATIONS[2:])
        np.testing.assert_array_equal(dropped.assets, ASSETS[1:])
        assert dropped["exposures"].shape == (N_OBS - 2, N_ASSETS - 1, 3)

    def test_drop_cannot_remove_all_observations_or_assets(self):
        panel = _make_panel()

        with pytest.raises(ValueError, match="all observations"):
            panel.drop(observations=OBSERVATIONS)

        with pytest.raises(ValueError, match="all assets"):
            panel.drop(assets=ASSETS)


class TestMasksAndUniverse:
    def test_edit_masks_allows_mutation_and_relocks(self):
        panel = _make_panel()

        with panel.edit_masks():
            panel.active_mask[0, 0] = False

        assert not panel.active_mask[0, 0]
        assert not panel.estimation_mask[0, 0]
        assert np.isnan(panel["x"][0, 0])
        assert not panel.active_mask.flags.writeable
        assert not panel.estimation_mask.flags.writeable

    def test_direct_mask_mutation_raises(self):
        panel = _make_panel()

        with pytest.raises(ValueError, match="read-only"):
            panel.active_mask[0, 0] = False

        with pytest.raises(ValueError, match="read-only"):
            panel.estimation_mask[0, 0] = False

    def test_estimation_only_exclusions_are_preserved(self):
        active_mask = np.ones((3, 2), dtype=np.bool_)
        estimation_mask = np.ones((3, 2), dtype=np.bool_)
        estimation_mask[1, 0] = False

        panel = _make_panel(
            fields={"x": np.ones((3, 2))},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
            estimation_mask=estimation_mask,
        )

        assert panel.active_mask[1, 0]
        assert not panel.estimation_mask[1, 0]

    def test_edit_masks_relocks_on_exception(self):
        panel = _make_panel()

        with pytest.raises(RuntimeError, match="boom"):
            with panel.edit_masks():
                panel.active_mask[0, 0] = False
                raise RuntimeError("boom")

        assert not panel.active_mask.flags.writeable
        assert not panel.estimation_mask.flags.writeable

    def test_edit_masks_does_not_mask_user_exception_with_validation_error(self):
        panel = _make_panel(
            fields={"x": np.ones((3, 2))},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
        )

        with pytest.raises(RuntimeError, match="user error"):
            with panel.edit_masks():
                panel.active_mask[1, :] = False
                raise RuntimeError("user error")

        assert not panel.active_mask.flags.writeable
        assert not panel.estimation_mask.flags.writeable

    def test_edit_masks_validates_nonempty_rows_on_clean_exit(self):
        panel = _make_panel(
            fields={"x": np.ones((3, 2))},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
        )

        with pytest.raises(ValueError, match="active_mask"):
            with panel.edit_masks():
                panel.active_mask[1, :] = False

    def test_edit_masks_validates_nonempty_estimation_rows(self):
        panel = _make_panel(
            fields={"x": np.ones((3, 2))},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
        )

        with pytest.raises(ValueError, match="estimation_mask"):
            with panel.edit_masks():
                panel.estimation_mask[1, :] = False

    def test_align_universe_to_delays_entry_until_field_is_valid(self):
        values = np.array(
            [
                [np.nan, 1.0],
                [np.nan, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ],
        )
        panel = _make_panel(
            fields={"x": values.copy(), "y": np.arange(8, dtype=float).reshape(4, 2)},
            observations=np.arange(4),
            assets=np.array(["A", "B"]),
        )

        removed = panel.align_active_mask_to("x")

        assert removed == 2
        assert not panel.active_mask[0, 0]
        assert not panel.active_mask[1, 0]
        assert not panel.estimation_mask[0, 0]
        assert np.isnan(panel["y"][0, 0])
        assert panel["y"][2, 0] == 4.0

    def test_align_universe_to_uses_categorical_missing_code(self):
        codes = np.array(
            [
                [MISSING_CATEGORY_CODE, 0],
                [MISSING_CATEGORY_CODE, 1],
                [1, 0],
            ],
            dtype=np.int32,
        )
        panel = AssetPanel(
            fields={"sector": FieldCategorical(codes, levels=["A", "B"])},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
        )

        removed = panel.align_active_mask_to("sector")

        assert removed == 2
        assert not panel.active_mask[0, 0]
        assert not panel.active_mask[1, 0]

    def test_align_universe_to_no_change_when_already_valid(self):
        panel = _make_panel(
            fields={"x": np.array([[1.0, 2.0], [3.0, 4.0]])},
            observations=np.arange(2),
            assets=np.array(["A", "B"]),
        )

        removed = panel.align_active_mask_to("x")

        assert removed == 0
        assert panel.active_mask.all()

    def test_align_universe_to_all_missing_asset_removes_entire_asset_history(self):
        panel = _make_panel(
            fields={"x": np.array([[np.nan, 1.0], [np.nan, 2.0], [np.nan, 3.0]])},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
        )

        removed = panel.align_active_mask_to("x")

        assert removed == 3
        assert not panel.active_mask[:, 0].any()
        assert panel.active_mask[:, 1].all()

    def test_align_universe_to_does_not_remove_interior_missing_values(self):
        panel = _make_panel(
            fields={"x": np.array([[1.0], [2.0], [np.nan], [4.0]])},
            observations=np.arange(4),
            assets=np.array(["A"]),
        )

        removed = panel.align_active_mask_to("x")

        assert removed == 0
        assert panel.active_mask.all()

    def test_align_universe_to_multiple_fields_uses_latest_valid_start(self):
        panel = _make_panel(
            fields={
                "x": np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]),
                "y": np.array(
                    [[np.nan, 1.0], [np.nan, 2.0], [30.0, 30.0], [40.0, 40.0]]
                ),
            },
            observations=np.arange(4),
            assets=np.array(["A", "B"]),
        )

        removed = panel.align_active_mask_to(["x", "y"])

        assert removed == 2
        assert not panel.active_mask[0, 0]
        assert not panel.active_mask[1, 0]
        assert panel.active_mask[2, 0]
        assert panel.active_mask[:, 1].all()

    def test_align_universe_to_respects_existing_universe_start(self):
        active_mask = np.array([[False, True], [False, True], [True, True]])
        panel = _make_panel(
            fields={"x": np.array([[np.nan, 1.0], [np.nan, 2.0], [3.0, 3.0]])},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
        )

        removed = panel.align_active_mask_to("x")

        assert removed == 0

    def test_align_universe_to_rejects_empty_observations(self):
        panel = _make_panel(
            fields={"x": np.array([[np.nan], [1.0]])},
            observations=np.arange(2),
            assets=np.array(["A"]),
        )

        with pytest.raises(ValueError, match="active_mask"):
            panel.align_active_mask_to("x")

        assert panel.active_mask.all()

    def test_align_universe_to_missing_field_raises(self):
        panel = _make_panel()

        with pytest.raises(KeyError, match="not found"):
            panel.align_active_mask_to("missing")


class TestFill:
    def test_ffill_and_bfill_modify_in_place_by_default(self):
        panel = _make_panel(
            fields={
                "x": np.array(
                    [[1.0, np.nan], [np.nan, 2.0], [np.nan, np.nan], [4.0, 5.0]],
                ),
            },
            observations=np.arange(4),
            assets=np.array(["A", "B"]),
        )

        result = panel.ffill("x", limit=1)

        assert result is panel
        assert panel["x"][1, 0] == 1.0
        assert np.isnan(panel["x"][2, 0])

        panel.bfill("x")
        assert panel["x"][0, 1] == 2.0

    def test_fill_inplace_false_returns_copy(self):
        panel = _make_panel(
            fields={"x": np.array([[1.0], [np.nan], [3.0]])},
            observations=np.arange(3),
            assets=np.array(["A"]),
        )

        filled = panel.ffill("x", inplace=False)

        assert filled is not panel
        assert filled["x"][1, 0] == 1.0
        assert np.isnan(panel["x"][1, 0])

    def test_fill_multiple_fields(self):
        values = np.array(
            [[1.0, np.nan], [np.nan, 2.0], [np.nan, np.nan], [4.0, 5.0]],
        )
        panel = _make_panel(
            fields={"a": values.copy(), "b": values.copy()},
            observations=np.arange(4),
            assets=np.array(["A", "B"]),
        )

        panel.ffill(["a", "b"]).bfill(["a", "b"])

        assert not np.isnan(panel["a"]).any()
        assert not np.isnan(panel["b"]).any()

    def test_fill_rejects_non_numeric_2d_fields(self):
        panel = _make_panel()
        panel["sector"] = FieldCategorical(
            np.zeros((N_OBS, N_ASSETS), dtype=np.int32),
            levels=["A"],
        )

        with pytest.raises(TypeError, match="numeric Field2D"):
            panel.ffill("sector")

    def test_fill_rejects_missing_and_3d_fields(self):
        panel = _make_full_panel()

        with pytest.raises(KeyError, match="not found"):
            panel.ffill("missing")

        with pytest.raises(TypeError, match="numeric Field2D"):
            panel.ffill("exposures")

    def test_ffill_does_not_fill_before_universe_entry(self):
        active_mask = np.array(
            [[False, True], [True, True], [True, True], [True, True]]
        )
        panel = _make_panel(
            fields={
                "x": np.array(
                    [
                        [np.nan, np.nan],
                        [np.nan, 2.0],
                        [np.nan, np.nan],
                        [4.0, 5.0],
                    ],
                ),
            },
            observations=np.arange(4),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
        )

        panel.ffill("x")

        assert np.isnan(panel["x"][0, 0])
        assert np.isnan(panel["x"][1, 0])
        assert np.isnan(panel["x"][2, 0])
        assert np.isnan(panel["x"][0, 1])
        assert panel["x"][2, 1] == 2.0

    def test_bfill_does_not_fill_from_inactive_future(self):
        active_mask = np.array(
            [[True, True], [True, True], [True, True], [True, False]]
        )
        panel = _make_panel(
            fields={
                "x": np.array(
                    [[1.0, np.nan], [np.nan, 2.0], [np.nan, np.nan], [4.0, np.nan]],
                ),
            },
            observations=np.arange(4),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
        )

        panel.bfill("x")

        assert panel["x"][1, 0] == 4.0
        assert panel["x"][0, 1] == 2.0
        assert np.isnan(panel["x"][2, 1])
        assert np.isnan(panel["x"][3, 1])

    def test_ffill_treats_inactive_gap_as_barrier(self):
        active_mask = np.array(
            [[True, True], [False, True], [False, True], [True, True]]
        )
        panel = _make_panel(
            fields={
                "x": np.array([[1.0, 0.0], [np.nan, 0.0], [np.nan, 0.0], [np.nan, 0.0]])
            },
            observations=np.arange(4),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
        )

        panel.ffill("x")

        assert np.isnan(panel["x"][3, 0])
        assert np.isnan(panel["x"][1, 0])
        assert np.isnan(panel["x"][2, 0])

    def test_ffill_respects_limit(self):
        panel = _make_panel(
            fields={"x": np.array([[1.0], [np.nan], [np.nan], [np.nan], [5.0]])},
            observations=np.arange(5),
            assets=np.array(["A"]),
        )

        panel.ffill("x", limit=1)

        assert panel["x"][1, 0] == 1.0
        assert np.isnan(panel["x"][2, 0])


class TestDataFrameDescribeInfo:
    def test_to_dataframe_single_field_matches_old_field_to_dataframe_shape(self):
        panel = _make_full_panel()

        df = panel.to_dataframe(fields="momentum")

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (N_OBS, N_ASSETS)
        assert list(df.index) == list(OBSERVATIONS)
        assert list(df.columns) == list(ASSETS)

    def test_to_dataframe_single_categorical_decodes_by_default(self):
        panel = _make_full_panel()

        df = panel.to_dataframe(fields="sector")

        assert df.iloc[2, 0] in {"Tech", "Health", "Finance"}

    def test_to_dataframe_single_categorical_can_return_raw_codes(self):
        panel = _make_full_panel()

        df = panel.to_dataframe(fields="sector", decode_categoricals=False)

        assert df.dtypes.iloc[0] == np.dtype("int32")

    def test_to_dataframe_single_3d_raises(self):
        panel = _make_full_panel()

        with pytest.raises(ValueError, match="only 2D fields"):
            panel.to_dataframe(fields="exposures")

    def test_to_dataframe_long_filters_active_mask_and_skips_3d(self):
        panel = _make_full_panel()

        with pytest.warns(UserWarning, match="Skipping Field3D"):
            df = panel.to_dataframe(output_format="long")

        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["observation", "asset"]
        assert "momentum" in df.columns
        assert "sector" in df.columns
        assert "exposures" not in df.columns
        assert "active_mask" not in df.columns
        assert "estimation_mask" in df.columns
        assert len(df) == int(panel.active_mask.sum())

    def test_to_dataframe_long_keeps_active_estimation_exclusions(self):
        active_mask = np.ones((3, 2), dtype=np.bool_)
        estimation_mask = np.ones((3, 2), dtype=np.bool_)
        estimation_mask[1, 0] = False
        panel = _make_panel(
            fields={"x": np.ones((3, 2))},
            observations=np.arange(3),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
            estimation_mask=estimation_mask,
        )

        df = panel.to_dataframe(output_format="long")

        assert (1, "A") in df.index
        assert "active_mask" not in df.columns
        assert not df.loc[(1, "A"), "estimation_mask"]

    def test_to_dataframe_wide_has_field_asset_columns(self):
        panel = _make_full_panel()

        with pytest.warns(UserWarning, match="Skipping Field3D"):
            df = panel.to_dataframe(output_format="wide")

        assert isinstance(df.columns, pd.MultiIndex)
        assert df.columns.names == ["field", "asset"]
        assert ("momentum", "AAPL") in df.columns
        assert ("active_mask", "AAPL") in df.columns

    def test_to_dataframe_invalid_format_raises(self):
        panel = _make_panel()

        with pytest.raises(ValueError, match="output_format"):
            panel.to_dataframe(output_format="invalid")

    def test_to_dataframe_unknown_field_and_asset_raise(self):
        panel = _make_panel()

        with pytest.raises(KeyError, match="Fields not found"):
            panel.to_dataframe(fields=["missing"])

        with pytest.raises(KeyError, match="Labels not found"):
            panel.to_dataframe(assets=["MISSING"])

    def test_describe_summarizes_field_types(self):
        panel = _make_full_panel()

        summary = panel.describe()

        assert summary.loc["momentum", "type"] == "2D"
        assert summary.loc["sector", "type"] == "categorical"
        assert summary.loc["exposures", "type"] == "3D"

    def test_describe_by_categorical_field(self):
        panel = _make_full_panel()

        summary = panel.describe(by="sector")

        assert isinstance(summary.index, pd.MultiIndex)
        assert "momentum" in summary.index.get_level_values("field")

    def test_describe_by_non_categorical_field_raises(self):
        panel = _make_full_panel()

        with pytest.raises(TypeError, match="not categorical"):
            panel.describe(by="momentum")

    def test_info_reports_shape_masks_and_fields(self):
        panel = _make_full_panel()

        report = panel.info()

        assert "AssetPanel Info" in report
        assert f"Observations  : {N_OBS:,}" in report
        assert f"Assets        : {N_ASSETS:,}" in report
        assert "Active Mask" in report
        assert "Estimation Mask" in report
        assert "Field Coverage" in report
        assert "Categorical Fields" in report
        assert "exposures" in report


class TestView:
    def test_view_properties_are_sliced_to_observations(self):
        panel = _make_full_panel()

        view = panel[2:8]

        np.testing.assert_array_equal(view.observations, OBSERVATIONS[2:8])
        np.testing.assert_array_equal(view.assets, ASSETS)
        np.testing.assert_array_equal(view.active_mask, panel.active_mask[2:8])
        np.testing.assert_array_equal(view.estimation_mask, panel.estimation_mask[2:8])
        assert len(view) == 6
        assert view.shape == (6,)
        assert view.ndim == 1

    def test_view_repr_includes_shape(self):
        panel = _make_panel()
        view = panel[5:10]

        result = repr(view)

        assert "AssetPanelView" in result
        assert "n_observations=5" in result
        assert f"n_assets={N_ASSETS}" in result

    def test_view_fields_mapping_returns_view_sized_field_objects(self):
        panel = _make_full_panel()

        view = panel[2:8]
        field = view.fields["exposures"]

        assert isinstance(field, Field3D)
        assert field.values.shape == (6, N_ASSETS, 3)
        np.testing.assert_array_equal(
            field.third_axis_labels,
            panel.fields["exposures"].third_axis_labels,
        )

    def test_nested_slice_view_composes_selectors(self):
        panel = _make_panel()

        view = panel[2:18][3:8]

        np.testing.assert_array_equal(view["x"], panel["x"][5:10])
        assert np.shares_memory(view["x"], panel["x"])

    def test_direct_view_construction_normalizes_selector(self):
        panel = _make_panel()
        selector = np.zeros(N_OBS, dtype=np.bool_)
        selector[[1, 3]] = True

        view = AssetPanelView(owner=panel, observation_selector=selector)

        assert view.n_observations == 2
        np.testing.assert_array_equal(view["x"], panel["x"][[1, 3]])

    def test_direct_view_construction_rejects_out_of_bounds_selector(self):
        panel = _make_panel()

        with pytest.raises(IndexError, match="out of bounds"):
            AssetPanelView(owner=panel, observation_selector=[0, N_OBS])

    def test_view_local_fields_shadow_owner_without_modifying_owner(self):
        panel = _make_panel()
        view = panel[5:10]

        view["x"] = np.full((5, N_ASSETS), 99.0)
        view["local"] = np.zeros((5, N_ASSETS))

        assert "local" in view
        assert "local" not in panel
        assert view["x"][0, 0] == 99.0
        assert panel["x"][5, 0] != 99.0
        assert set(view.keys()) == {"x", "local"}
        assert list(view.keys()) == ["x", "local"]

    def test_nested_view_preserves_and_slices_local_fields(self):
        panel = _make_panel()
        view = panel[5:10]
        local_x = np.arange(5 * N_ASSETS, dtype=float).reshape(5, N_ASSETS)
        local_only = local_x + 100.0
        view["x"] = local_x
        view["local"] = local_only

        nested = view[1:4]

        np.testing.assert_array_equal(nested["x"], local_x[1:4])
        np.testing.assert_array_equal(nested["local"], local_only[1:4])
        assert "local" not in panel

    def test_view_local_fields_validate_active_universe(self):
        active_mask = np.array(
            [[False, True], [True, True], [True, True], [True, True]]
        )
        panel = _make_panel(
            fields={"x": np.ones((4, 2))},
            observations=np.arange(4),
            assets=np.array(["A", "B"]),
            active_mask=active_mask,
        )
        view = panel[:]

        with pytest.raises(ValueError, match="finite value"):
            view["bad"] = np.ones((4, 2))

        values = np.ones((4, 2))
        values[~active_mask] = np.nan
        view["good"] = values
        np.testing.assert_array_equal(view["good"], values)

    def test_view_to_dataframe_includes_local_fields(self):
        panel = _make_panel()
        view = panel[5:10]
        local = np.full((5, N_ASSETS), 1.5)
        view["local"] = local

        df = view.to_dataframe(fields=["x", "local"], output_format="wide")

        assert ("x", "AAPL") in df.columns
        assert ("local", "AAPL") in df.columns
        np.testing.assert_array_equal(df["local"].to_numpy(), local)

    def test_view_local_typed_field_and_delete(self):
        panel = _make_panel()
        view = panel[5:10]

        view["z"] = Field3D(
            np.ones((5, N_ASSETS, 2)),
            third_axis_name="scenario",
            third_axis_labels=["base", "stress"],
        )

        assert view["z"].shape == (5, N_ASSETS, 2)
        del view["z"]
        assert "z" not in view

    def test_view_cannot_delete_owner_field(self):
        panel = _make_panel()
        view = panel[5:10]

        with pytest.raises(KeyError, match="not a local field"):
            del view["x"]

    def test_view_missing_owner_field_raises(self):
        panel = _make_panel()
        view = panel[5:10]

        with pytest.raises(KeyError):
            view["missing"]

    def test_view_tuple_selector_raises(self):
        panel = _make_panel()
        view = panel[5:10]

        with pytest.raises(TypeError, match="one-dimensional observation selectors"):
            view[:, :]

    def test_view_rejects_wrong_local_shape(self):
        panel = _make_panel()
        view = panel[5:10]

        with pytest.raises(ValueError, match="shape"):
            view["bad"] = np.ones((4, N_ASSETS))

    def test_view_prevents_raw_replacement_of_owner_typed_field(self):
        panel = _make_full_panel()
        view = panel[5:10]

        with pytest.raises(TypeError, match="FieldCategorical"):
            view["sector"] = view["sector"]

    def test_view_copy_can_copy_owner(self):
        panel = _make_panel()
        view = panel[5:10]

        copied = view.copy(deep=True, copy_owner=True)

        assert copied.owner is not panel
        np.testing.assert_array_equal(copied["x"], view["x"])

    def test_view_copy_deep_copies_local_fields(self):
        panel = _make_panel()
        view = panel[5:10]
        view["local"] = np.zeros((5, N_ASSETS))

        copied = view.copy(deep=True, copy_owner=False)
        copied["local"][0, 0] = 1.0

        assert view["local"][0, 0] == 0.0
        assert copied.owner is panel


class TestCopy:
    def test_shallow_copy_shares_field_arrays_but_not_masks(self):
        panel = _make_panel()

        copied = panel.copy(deep=False)

        assert copied is not panel
        assert copied.fields is not panel.fields
        assert copied.fields["x"].values is panel.fields["x"].values
        assert copied.active_mask is not panel.active_mask
        with copied.edit_masks():
            copied.active_mask[0, 0] = False
        assert panel.active_mask[0, 0]

    def test_shallow_copy_has_independent_field_mapping(self):
        panel = _make_panel()
        copied = panel.copy(deep=False)

        copied["y"] = np.zeros((N_OBS, N_ASSETS))

        assert "y" not in panel

    def test_deep_copy_copies_field_arrays(self):
        panel = _make_panel()

        copied = panel.copy(deep=True)
        copied["x"][0, 0] = 999.0

        assert panel["x"][0, 0] != 999.0
        assert copied.observations is not panel.observations
        assert copied.assets is not panel.assets


class TestConcat:
    def test_concat_stacks_observations_fields_and_masks(self):
        left = _make_full_panel(observations=np.arange(N_OBS))
        right = _make_full_panel(observations=np.arange(N_OBS, 2 * N_OBS))

        result = concat([left, right])

        assert isinstance(result, AssetPanel)
        np.testing.assert_array_equal(
            result.observations,
            np.concatenate([left.observations, right.observations]),
        )
        np.testing.assert_array_equal(result.assets, left.assets)
        np.testing.assert_array_equal(
            result.active_mask,
            np.concatenate([left.active_mask, right.active_mask], axis=0),
        )
        np.testing.assert_array_equal(
            result.estimation_mask,
            np.concatenate([left.estimation_mask, right.estimation_mask], axis=0),
        )
        np.testing.assert_array_equal(
            result["momentum"],
            np.concatenate([left["momentum"], right["momentum"]], axis=0),
        )
        np.testing.assert_array_equal(
            result.fields["sector"].levels,
            left.fields["sector"].levels,
        )
        np.testing.assert_array_equal(
            result.fields["exposures"].third_axis_labels,
            left.fields["exposures"].third_axis_labels,
        )
        assert not result.active_mask.flags.writeable
        assert not result.estimation_mask.flags.writeable

    def test_concat_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            concat([])

    def test_concat_rejects_non_panel_input(self):
        with pytest.raises(TypeError, match="AssetPanel"):
            concat([_make_panel(), object()])

    def test_concat_rejects_single_panel_input(self):
        with pytest.raises(TypeError, match="single AssetPanel"):
            concat(_make_panel())

    def test_concat_requires_identical_assets(self):
        left = _make_panel(
            fields={"x": np.ones((N_OBS, 2))},
            assets=np.array(["A", "B"]),
        )
        right = _make_panel(
            fields={"x": np.ones((N_OBS, 2))},
            assets=np.array(["A", "C"]),
        )

        with pytest.raises(ValueError, match="identical assets"):
            concat([left, right])

    def test_concat_requires_same_field_names_and_order(self):
        left = _make_panel(fields={"x": np.ones((N_OBS, N_ASSETS))})
        right = _make_panel(fields={"y": np.ones((N_OBS, N_ASSETS))})

        with pytest.raises(ValueError, match="same fields"):
            concat([left, right])

    def test_concat_requires_same_field_type(self):
        left = _make_full_panel()
        right = _make_full_panel()
        right.fields["momentum"] = FieldCategorical(
            np.zeros((N_OBS, N_ASSETS), dtype=np.int32),
            levels=["zero"],
        )

        with pytest.raises(TypeError, match="Field 'momentum'"):
            concat([left, right])

    def test_concat_requires_same_field_dtype(self):
        left = _make_panel(fields={"x": np.ones((N_OBS, N_ASSETS), dtype=np.float32)})
        right = _make_panel(fields={"x": np.ones((N_OBS, N_ASSETS), dtype=np.float64)})

        with pytest.raises(TypeError, match="dtype"):
            concat([left, right])

    def test_concat_requires_matching_categorical_levels(self):
        left = _make_full_panel()
        right = _make_full_panel()
        right.fields["sector"] = FieldCategorical(
            right["sector"],
            levels=["Tech", "Health", "Other"],
        )

        with pytest.raises(ValueError, match="different levels"):
            concat([left, right])

    def test_concat_requires_matching_3d_metadata(self):
        left = _make_full_panel()
        right = _make_full_panel()
        right.fields["exposures"] = Field3D(
            right["exposures"],
            third_axis_name="factor",
            third_axis_labels=["mkt", "size", "quality"],
            third_axis_groups=["market", "style", "style"],
        )

        with pytest.raises(ValueError, match="third_axis_labels"):
            concat([left, right])

    def test_concat_verify_observations_rejects_duplicates(self):
        left = _make_panel(observations=np.arange(N_OBS))
        right = _make_panel(observations=np.arange(N_OBS))

        with pytest.raises(ValueError, match="duplicate"):
            concat([left, right], verify_observations=True)


class TestPersistence:
    def test_round_trip_preserves_values_and_metadata(self, tmp_path):
        observations = np.datetime64("2024-01-01") + np.arange(N_OBS)
        panel = _make_full_panel(observations=observations)

        panel.save(tmp_path / "panel")
        loaded = AssetPanel.load(tmp_path / "panel")

        _assert_panels_equal(loaded, panel)

    def test_round_trip_preserves_3d_field_without_groups(self, tmp_path):
        panel = _make_panel()
        panel["scenarios"] = Field3D(
            np.ones((N_OBS, N_ASSETS, 2)),
            third_axis_name="scenario",
            third_axis_labels=["base", "stress"],
        )

        panel.save(tmp_path / "panel")
        loaded = AssetPanel.load(tmp_path / "panel")

        assert loaded.fields["scenarios"].third_axis_groups is None
        np.testing.assert_array_equal(
            loaded.fields["scenarios"].third_axis_labels,
            ["base", "stress"],
        )

    def test_all_true_masks_are_omitted_from_disk(self, tmp_path):
        panel = _make_panel()

        panel.save(tmp_path / "panel")

        assert not (tmp_path / "panel" / "active_mask.npy").exists()
        assert not (tmp_path / "panel" / "estimation_mask.npy").exists()
        with open(tmp_path / "panel" / "_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        assert metadata["active_mask"] == "all_true"
        assert metadata["estimation_mask"] == "all_true"

    def test_nontrivial_masks_are_saved(self, tmp_path):
        panel = _make_full_panel()

        panel.save(tmp_path / "panel")

        assert (tmp_path / "panel" / "active_mask.npy").exists()
        assert (tmp_path / "panel" / "estimation_mask.npy").exists()

    def test_saved_arrays_load_without_pickle(self, tmp_path):
        _make_full_panel().save(tmp_path / "panel")

        for npy_file in (tmp_path / "panel").rglob("*.npy"):
            np.load(npy_file, allow_pickle=False)

    def test_save_load_keeps_object_labels_as_strings(self, tmp_path):
        panel = AssetPanel(
            fields={
                "sector": FieldCategorical(
                    np.array([[0, 1], [1, 0]], dtype=np.int32),
                    levels=np.array(["Tech", "Health"], dtype=object),
                ),
                "exposures": Field3D(
                    np.ones((2, 2, 2)),
                    third_axis_name="factor",
                    third_axis_labels=np.array(["mkt", "size"], dtype=object),
                    third_axis_groups=np.array(["market", "style"], dtype=object),
                ),
            },
            observations=np.array(["t0", "t1"], dtype=object),
            assets=np.array(["A", "B"], dtype=object),
        )

        panel.save(tmp_path / "panel")
        loaded = AssetPanel.load(tmp_path / "panel")

        assert loaded.observations.dtype != object
        assert loaded.assets.dtype != object
        assert loaded.fields["sector"].levels.dtype != object
        assert loaded.fields["exposures"].third_axis_labels.dtype != object
        assert loaded.fields["exposures"].third_axis_groups.dtype != object

        with open(tmp_path / "panel" / "_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)
        assert "observations_dtype" not in metadata
        assert "assets_dtype" not in metadata
        assert "levels_dtype" not in metadata["fields"]["sector"]
        assert "third_axis_labels_dtype" not in metadata["fields"]["exposures"]
        assert "third_axis_groups_dtype" not in metadata["fields"]["exposures"]

    def test_selective_load_loads_requested_fields_and_metadata(self, tmp_path):
        panel = _make_full_panel()
        panel.save(tmp_path / "panel")

        loaded_sector = AssetPanel.load(tmp_path / "panel", fields=["sector"])
        loaded_exposures = AssetPanel.load(tmp_path / "panel", fields=["exposures"])
        loaded_ordered = AssetPanel.load(
            tmp_path / "panel",
            fields=["exposures", "sector"],
        )

        assert set(loaded_sector.fields) == {"sector"}
        assert isinstance(loaded_sector.fields["sector"], FieldCategorical)
        np.testing.assert_array_equal(
            loaded_sector.fields["sector"].levels,
            panel.fields["sector"].levels,
        )
        assert set(loaded_exposures.fields) == {"exposures"}
        assert isinstance(loaded_exposures.fields["exposures"], Field3D)
        assert loaded_exposures.fields["exposures"].third_axis_name == "factor"
        assert list(loaded_ordered.fields) == ["exposures", "sector"]

    def test_load_unknown_field_raises(self, tmp_path):
        _make_panel().save(tmp_path / "panel")

        with pytest.raises(KeyError, match="not found"):
            AssetPanel.load(tmp_path / "panel", fields=["missing"])

    def test_load_missing_metadata_raises(self, tmp_path):
        (tmp_path / "not_panel").mkdir()

        with pytest.raises(FileNotFoundError, match="Not a saved AssetPanel"):
            AssetPanel.load(tmp_path / "not_panel")

    def test_overwrite_behavior(self, tmp_path):
        panel = _make_panel()
        panel.save(tmp_path / "panel")

        with pytest.raises(FileExistsError, match="overwrite=True"):
            panel.save(tmp_path / "panel")

        panel.save(tmp_path / "panel", overwrite=True)
        _assert_panels_equal(AssetPanel.load(tmp_path / "panel"), panel)

    def test_refuses_non_panel_directory_and_file_path(self, tmp_path):
        directory = tmp_path / "directory"
        directory.mkdir()
        (directory / "file.txt").write_text("hello")
        file_path = tmp_path / "file.txt"
        file_path.write_text("hello")

        with pytest.raises(ValueError, match="not a saved AssetPanel"):
            _make_panel().save(directory)

        with pytest.raises(ValueError, match="is a file"):
            _make_panel().save(file_path)

    def test_load_with_read_only_mmap(self, tmp_path):
        panel = _make_full_panel()
        panel.save(tmp_path / "panel")

        loaded = AssetPanel.load(tmp_path / "panel", mmap_mode="r")

        np.testing.assert_array_equal(loaded["momentum"], panel["momentum"])
        with pytest.raises((TypeError, ValueError)):
            loaded["momentum"][0, 0] = 999.0

    def test_metadata_content_uses_new_field_types(self, tmp_path):
        panel = _make_full_panel()

        panel.save(tmp_path / "panel")
        with open(tmp_path / "panel" / "_metadata.json", encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["version"] == 1
        assert metadata["fields"]["momentum"]["type"] == "Field2D"
        assert metadata["fields"]["sector"]["type"] == "FieldCategorical"
        assert metadata["fields"]["exposures"]["type"] == "Field3D"
        assert metadata["fields"]["exposures"]["third_axis_name"] == "factor"
