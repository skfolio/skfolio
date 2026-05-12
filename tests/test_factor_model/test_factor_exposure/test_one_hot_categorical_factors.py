from __future__ import annotations

import numpy as np
import pytest

from skfolio.containers import MISSING_CATEGORY_CODE, FieldCategorical
from skfolio.factor_model.factor_exposure import OneHotCategoricalFactors


def test_fit_transform_one_hot_encodes_categorical_field(simple_panel):
    """Test one-hot factor exposures and factor names."""
    levels = np.array(["Energy", "Technology", "Utilities"])
    codes = np.tile(
        np.array([0, 1, 2, MISSING_CATEGORY_CODE, 0], dtype=np.int32),
        (simple_panel.n_observations, 1),
    )
    simple_panel["industry"] = FieldCategorical(codes, levels=levels)

    factor = OneHotCategoricalFactors(category="industry", family="industry")
    result = factor.fit_transform(simple_panel)

    assert result.shape == (simple_panel.n_observations, simple_panel.n_assets, 3)
    np.testing.assert_array_equal(factor.factor_names_, levels)
    np.testing.assert_array_equal(result[0, 0], np.array([1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(result[0, 1], np.array([0.0, 1.0, 0.0]))
    np.testing.assert_array_equal(result[0, 2], np.array([0.0, 0.0, 1.0]))
    assert np.isnan(result[0, 3]).all()


def test_partial_fit_transform_matches_fit_transform(simple_panel):
    """Test stateless partial_fit_transform delegation."""
    levels = np.array(["US", "CA"])
    codes = np.tile(
        np.array([0, 1, MISSING_CATEGORY_CODE, 0, 1], dtype=np.int32),
        (simple_panel.n_observations, 1),
    )
    simple_panel["country"] = FieldCategorical(codes, levels=levels)

    fit_result = OneHotCategoricalFactors(
        category="country", family="country"
    ).fit_transform(simple_panel)
    partial_fit_result = OneHotCategoricalFactors(
        category="country", family="country"
    ).partial_fit_transform(simple_panel)

    np.testing.assert_array_equal(partial_fit_result, fit_result)


def test_non_categorical_field_raises(simple_panel):
    """Test that the selected field must be categorical."""
    factor = OneHotCategoricalFactors(category="market_cap", family="industry")

    with pytest.raises(ValueError, match="CategoricalField"):
        factor.fit_transform(simple_panel)
