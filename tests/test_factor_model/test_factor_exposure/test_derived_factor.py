"""Tests for DerivedFactor and dependency layer logic."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.factor_model.descriptor import Passthrough
from skfolio.factor_model.factor_exposure import (
    DerivedFactor,
    FixedWeightedFactor,
    GlobalFactor,
)
from skfolio.prior import CharacteristicsFactorModel
from tests.test_factor_model._characteristics_data import (
    generate_synthetic_characteristics,
)

# --- Fixtures ---


@pytest.fixture
def size_exposure(simple_panel):
    """Pre-computed size factor exposure."""
    size_factor = FixedWeightedFactor(
        descriptors=[("lnmcap", Passthrough("market_cap"))],
        family="style",
    )
    return size_factor.fit_transform(simple_panel)


# --- DerivedFactor Unit Tests ---


class TestDerivedFactor:
    """Unit tests for DerivedFactor."""

    def test_basic_transformation(self, simple_panel, size_exposure):
        """Test that func is applied and output is standardized."""
        nlsize = DerivedFactor(source="size", func=lambda x: x**3, family="style")
        result = nlsize.fit_transform(simple_panel, source_exposure=size_exposure)

        assert result.shape == size_exposure.shape
        assert not np.isnan(result).any()
        # Output should be standardized (mean ~0, std ~1 per observation)
        assert np.abs(np.nanmean(result, axis=1)).max() < 0.15
        assert np.abs(np.nanstd(result, axis=1) - 1).max() < 0.15

    def test_missing_source_exposure_raises(self, simple_panel):
        """Test that missing source_exposure raises ValueError."""
        nlsize = DerivedFactor(source="size", func=lambda x: x**3, family="style")

        with pytest.raises(ValueError, match="source_exposure"):
            nlsize.fit_transform(simple_panel)

    def test_invalid_source_exposure_shape_raises(self, simple_panel, size_exposure):
        """Test that source_exposure must be a single-factor exposure."""
        nlsize = DerivedFactor(source="size", func=lambda x: x**3, family="style")

        with pytest.raises(ValueError, match="source_exposure"):
            nlsize.fit_transform(
                simple_panel, source_exposure=size_exposure[:, :, np.newaxis]
            )

    def test_invalid_func_output_shape_raises(self, simple_panel, size_exposure):
        """Test that func must preserve the source exposure shape."""
        nlsize = DerivedFactor(source="size", func=lambda x: x[:, :-1], family="style")

        with pytest.raises(ValueError, match="same shape"):
            nlsize.fit_transform(simple_panel, source_exposure=size_exposure)

    def test_passthrough_outlier_transformer(self, simple_panel, size_exposure):
        """Test that passthrough outlier transformer skips winsorization."""
        nlsize = DerivedFactor(
            source="size",
            func=lambda x: x**3,
            outlier_transformer="passthrough",
            family="style",
        )
        result = nlsize.fit_transform(simple_panel, source_exposure=size_exposure)

        assert result.shape == size_exposure.shape

    def test_passthrough_scoring_transformer(self, simple_panel, size_exposure):
        """Test that passthrough scoring skips standardization."""
        nlsize = DerivedFactor(
            source="size",
            func=lambda x: x**3,
            scoring_transformer="passthrough",
            family="style",
        )
        result = nlsize.fit_transform(simple_panel, source_exposure=size_exposure)

        # Without scoring, output is raw cubed values (not standardized)
        assert result.shape == size_exposure.shape

    def test_custom_func(self, simple_panel, size_exposure):
        """Test with a custom transformation function."""
        nlsize = DerivedFactor(
            source="size", func=lambda x: np.sign(x) * np.abs(x) ** 0.5, family="style"
        )
        result = nlsize.fit_transform(simple_panel, source_exposure=size_exposure)

        assert result.shape == size_exposure.shape


# --- CharacteristicsFactorModel Integration Tests ---


class TestDependencyLayers:
    """Tests for dependency layer logic in CharacteristicsFactorModel."""

    def test_single_layer_no_derived(self):
        """Test that factors without dependencies form a single layer."""
        factors = [
            ("global", GlobalFactor(family="market")),
            (
                "size",
                FixedWeightedFactor(
                    descriptors=[("d", Passthrough("market_cap"))], family="style"
                ),
            ),
            (
                "value",
                FixedWeightedFactor(
                    descriptors=[("d", Passthrough("market_cap"))], family="style"
                ),
            ),
        ]

        model = CharacteristicsFactorModel(
            factors=factors,
        )
        model._validate_factors()
        model.named_factor_estimators_ = {name: est for name, est in factors}

        layers = model._get_dependency_layers()

        assert len(layers) == 1
        assert set(layers[0]) == {"global", "size", "value"}

    def test_two_layers_with_derived(self):
        """Test that DerivedFactor creates a second layer."""
        factors = [
            ("global", GlobalFactor(family="market")),
            (
                "size",
                FixedWeightedFactor(
                    descriptors=[("d", Passthrough("market_cap"))], family="style"
                ),
            ),
            (
                "nlsize",
                DerivedFactor(source="size", func=lambda x: x**3, family="style"),
            ),
        ]

        model = CharacteristicsFactorModel(
            factors=factors,
        )
        model._validate_factors()
        model.named_factor_estimators_ = {name: est for name, est in factors}

        layers = model._get_dependency_layers()

        assert len(layers) == 2
        assert "nlsize" not in layers[0]
        assert "nlsize" in layers[1]

    def test_chained_dependencies(self):
        """Test three-level dependency chain."""
        factors = [
            (
                "base",
                FixedWeightedFactor(
                    descriptors=[("d", Passthrough("market_cap"))], family="style"
                ),
            ),
            (
                "derived1",
                DerivedFactor(source="base", func=lambda x: x**2, family="style"),
            ),
            (
                "derived2",
                DerivedFactor(source="derived1", func=lambda x: x**2, family="style"),
            ),
        ]

        model = CharacteristicsFactorModel(
            factors=factors,
        )
        model._validate_factors()
        model.named_factor_estimators_ = {name: est for name, est in factors}

        layers = model._get_dependency_layers()

        assert len(layers) == 3
        assert "base" in layers[0]
        assert "derived1" in layers[1]
        assert "derived2" in layers[2]

    def test_undefined_source_raises(self):
        """Test that referencing undefined source raises ValueError."""
        factors = [
            (
                "nlsize",
                DerivedFactor(source="undefined", func=lambda x: x**3, family="style"),
            ),
        ]

        model = CharacteristicsFactorModel(
            factors=factors,
        )
        model._validate_factors()
        model.named_factor_estimators_ = {name: est for name, est in factors}

        with pytest.raises(ValueError, match="undefined"):
            model._get_dependency_layers()

    def test_circular_dependency_raises(self):
        """Test that circular dependencies raise CycleError."""
        from graphlib import CycleError

        # Create factors that would form a cycle if allowed
        factors = [
            ("a", DerivedFactor(source="b", func=lambda x: x, family="style")),
            ("b", DerivedFactor(source="a", func=lambda x: x, family="style")),
        ]

        model = CharacteristicsFactorModel(
            factors=factors,
        )
        model._validate_factors()
        model.named_factor_estimators_ = {name: est for name, est in factors}

        with pytest.raises(CycleError):
            model._get_dependency_layers()


class TestCharacteristicsFactorModelWithDerived:
    """End-to-end tests for CharacteristicsFactorModel with DerivedFactor."""

    def test_fit_with_derived_factor(self):
        """Test full model fit with a DerivedFactor."""
        # No listing gaps: `active_mask` is dense so model-written `benchmark_weights`
        # (zeros off-estimation) do not sit on inactive cells (AssetPanel float rule).
        characteristics = generate_synthetic_characteristics(
            n_assets=100,
            years=1,
            seed=42,
            delist_prob=0.0,
            list_late_prob=0.0,
            mid_nan_frac=0.0,
        )
        characteristics.ffill("market_cap").bfill("market_cap")
        X = characteristics.to_dataframe(fields="returns")

        factors = [
            ("global", GlobalFactor(family="market")),
            (
                "size",
                FixedWeightedFactor(
                    descriptors=[("mc", Passthrough("market_cap"))], family="style"
                ),
            ),
            (
                "nlsize",
                DerivedFactor(source="size", func=lambda x: x**3, family="style"),
            ),
        ]

        model = CharacteristicsFactorModel(
            factors=factors,
        )

        model.fit(X, characteristics=characteristics)

        assert "global" in model.factor_model_.factor_names
        assert "size" in model.factor_model_.factor_names
        assert "nlsize" in model.factor_model_.factor_names
        assert model.factor_model_.loading_matrix.shape[1] == 3

    def test_fit_with_neutralization(self):
        """Test that nlsize can be orthogonalized against size."""
        characteristics = generate_synthetic_characteristics(
            n_assets=100,
            years=1,
            seed=42,
            delist_prob=0.0,
            list_late_prob=0.0,
            mid_nan_frac=0.0,
        )
        characteristics.ffill("market_cap").bfill("market_cap")
        X = characteristics.to_dataframe(fields="returns")

        factors = [
            ("global", GlobalFactor(family="market")),
            (
                "size",
                FixedWeightedFactor(
                    descriptors=[("mc", Passthrough("market_cap"))], family="style"
                ),
            ),
            (
                "nlsize",
                DerivedFactor(source="size", func=lambda x: x**3, family="style"),
            ),
        ]

        model = CharacteristicsFactorModel(
            factors=factors,
            neutralize_against={"nlsize": ["size"]},
        )

        model.fit(X, characteristics=characteristics)

        # Verify orthogonalization reduced correlation
        factor_names = list(model.factor_model_.factor_names)
        loadings = model.factor_model_.loading_matrix
        size_idx = factor_names.index("size")
        nlsize_idx = factor_names.index("nlsize")

        corr = np.corrcoef(loadings[:, size_idx], loadings[:, nlsize_idx])[0, 1]
        # After orthogonalization, correlation should be reduced
        assert abs(corr) < 0.5
