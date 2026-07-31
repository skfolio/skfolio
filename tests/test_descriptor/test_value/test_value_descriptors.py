"""Tests for Value descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import (
    BookToPrice,
    CashFlowToPrice,
    EarningsToPrice,
    SalesToPrice,
)

# Parametrize all four value descriptors with their numerator field.
VALUE_DESCRIPTORS = [
    (BookToPrice, "book_equity"),
    (SalesToPrice, "sales_ttm"),
    (CashFlowToPrice, "operating_cash_flow_ttm"),
    (EarningsToPrice, "net_income_ttm"),
]

VALUE_DIRECTORY_DESCRIPTORS = [
    BookToPrice,
    SalesToPrice,
    CashFlowToPrice,
]


@pytest.mark.parametrize(
    "descriptor_cls, numerator_field",
    VALUE_DESCRIPTORS,
    ids=[cls.__name__ for cls, _ in VALUE_DESCRIPTORS],
)
class TestValueDescriptors:
    """Shared tests for all value descriptors (numerator / market_cap)."""

    def test_output_shape(self, simple_panel, descriptor_cls, numerator_field):
        """Output shape matches (n_observations, n_assets)."""
        result = descriptor_cls().fit_transform(simple_panel)
        assert result.shape == simple_panel["market_cap"].shape

    def test_partial_fit_transform_matches_fit_transform(
        self, simple_panel, descriptor_cls, numerator_field
    ):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = descriptor_cls()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_ratio(self, simple_panel, descriptor_cls, numerator_field):
        """Output equals numerator / market_cap."""
        result = descriptor_cls().fit_transform(simple_panel)
        expected = simple_panel[numerator_field] / simple_panel["market_cap"]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel, descriptor_cls, numerator_field):
        """NaN in numerator or denominator propagates to output."""
        simple_panel[numerator_field][0, 0] = np.nan
        simple_panel["market_cap"][1, 0] = np.nan
        result = descriptor_cls().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        # Other values unaffected
        assert not np.isnan(result[0, 1])


@pytest.mark.parametrize(
    "descriptor_cls",
    VALUE_DIRECTORY_DESCRIPTORS,
    ids=[cls.__name__ for cls in VALUE_DIRECTORY_DESCRIPTORS],
)
class TestValueDirectoryDescriptors:
    """Validation tests for descriptors in the `_value` package."""

    def test_raise_when_market_cap_zero(self, simple_panel, descriptor_cls):
        """Non-missing market cap must be strictly positive."""
        simple_panel["market_cap"][0, 0] = 0.0
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            descriptor_cls().fit_transform(simple_panel)

    def test_raise_when_market_cap_negative(self, simple_panel, descriptor_cls):
        """Negative market cap is invalid."""
        simple_panel["market_cap"][0, 0] = -1.0
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            descriptor_cls().fit_transform(simple_panel)

    def test_raise_when_market_cap_infinite(self, simple_panel, descriptor_cls):
        """Infinite market cap is invalid."""
        simple_panel["market_cap"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            descriptor_cls().fit_transform(simple_panel)
