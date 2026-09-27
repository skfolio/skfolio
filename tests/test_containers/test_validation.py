"""Tests for AssetPanel validation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from skfolio.containers import AssetPanel, Field3D, validate_asset_panel


class DummyEstimator(BaseEstimator):
    """Dummy estimator for testing validation."""

    pass


def _asset_panel_with_field(values, field="field", active_mask=None):
    """Create a simple AssetPanel for field validation tests."""
    panel = AssetPanel(
        fields={field: np.asarray(values, dtype=float)},
        asset_names=np.array(["a", "b"]),
        observations=np.arange(np.asarray(values).shape[0]),
    )
    if active_mask is not None:
        panel.active_mask = np.asarray(active_mask, dtype=bool)
    return panel


class TestValidateAssetPanel:
    """Test suite for validate_asset_panel."""

    def test_finite_or_nan_allows_nan_and_rejects_inf(self):
        estimator = DummyEstimator()
        panel = _asset_panel_with_field([[1.0, np.nan], [2.0, np.inf]])

        with pytest.raises(ValueError, match='"field" contains infinite values'):
            validate_asset_panel(estimator, panel, finite_or_nan=["field"])

    def test_strictly_positive_or_nan_allows_nan_and_rejects_non_positive(self):
        estimator = DummyEstimator()
        panel = _asset_panel_with_field([[1.0, np.nan], [0.0, 2.0]])

        with pytest.raises(ValueError, match="non-positive"):
            validate_asset_panel(estimator, panel, strictly_positive_or_nan=["field"])

    def test_non_negative_or_nan_allows_zero_and_rejects_negative(self):
        estimator = DummyEstimator()
        panel = _asset_panel_with_field([[0.0, np.nan], [-1.0, 2.0]])

        with pytest.raises(ValueError, match="negative values"):
            validate_asset_panel(estimator, panel, non_negative_or_nan=["field"])

    def test_strictly_positive_when_active_rejects_active_nan_and_inf(self):
        estimator = DummyEstimator()
        panel = _asset_panel_with_field(
            [[1.0, np.nan], [np.inf, 2.0]],
            active_mask=[[True, False], [True, True]],
        )

        with pytest.raises(ValueError, match="non-finite or non-positive"):
            validate_asset_panel(
                estimator, panel, strictly_positive_when_active=["field"]
            )

    def test_reset_false_still_validates_field_rules(self):
        estimator = DummyEstimator()
        panel = _asset_panel_with_field([[1.0, 2.0]])
        validate_asset_panel(estimator, panel)

        bad_panel = _asset_panel_with_field([[np.inf, 2.0]])
        with pytest.raises(ValueError, match='"field" contains infinite values'):
            validate_asset_panel(
                estimator, bad_panel, finite_or_nan=["field"], reset=False
            )

    def test_empty_rule_lists_are_noops(self):
        estimator = DummyEstimator()
        panel = _asset_panel_with_field([[1.0, 2.0]])

        result = validate_asset_panel(
            estimator,
            panel,
            required_fields=[],
            reserved_fields=[],
            finite_or_nan=[],
            finite_when_active=[],
            strictly_positive_or_nan=[],
            strictly_positive_when_active=[],
            non_negative_or_nan=[],
        )

        assert result is panel

    def test_rejects_non_asset_panel(self):
        with pytest.raises(TypeError, match="AssetPanel"):
            validate_asset_panel(DummyEstimator(), np.ones((2, 2)))

    def test_reset_false_requires_matching_asset_names(self):
        estimator = DummyEstimator()
        panel = _asset_panel_with_field([[1.0, 2.0]])
        validate_asset_panel(estimator, panel)
        panel_with_reversed_assets = AssetPanel(
            fields={"field": np.array([[2.0, 1.0]])},
            asset_names=np.array(["b", "a"]),
            observations=np.array([0]),
        )

        with pytest.raises(ValueError, match="asset_names don't match"):
            validate_asset_panel(estimator, panel_with_reversed_assets, reset=False)

    @pytest.mark.parametrize(
        "rule,bad_value,error",
        [
            ("finite_when_active", np.nan, "NaN/inf"),
            ("strictly_positive_or_nan", 0.0, "non-positive"),
            (
                "strictly_positive_when_active",
                0.0,
                "non-finite or non-positive",
            ),
            ("non_negative_or_nan", -1.0, "negative values"),
        ],
    )
    def test_3d_field_rules_reject_invalid_components(self, rule, bad_value, error):
        values = np.ones((2, 2, 1))
        values[0, 0, 0] = bad_value
        panel = AssetPanel(
            fields={
                "field": Field3D(
                    values,
                    third_axis_name="factor",
                    third_axis_labels=["value"],
                )
            },
            asset_names=np.array(["a", "b"]),
            observations=np.arange(2),
        )

        with pytest.raises(ValueError, match=error):
            validate_asset_panel(DummyEstimator(), panel, **{rule: ["field"]})

    def test_rule_fields_must_exist_in_panel(self):
        panel = _asset_panel_with_field([[1.0, 2.0]])

        with pytest.raises(ValueError, match="not in the AssetPanel"):
            validate_asset_panel(
                DummyEstimator(), panel, finite_or_nan=["missing_field"]
            )
