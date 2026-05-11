"""Tests for Earnings Yield descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.factor_model.descriptor import (
    EarningsToPrice,
    EbitdaToEnterpriseValue,
    ForwardEarningsToPrice,
)

# (descriptor_class, numerator_field, denominator_field)
EARNINGS_YIELD_DESCRIPTORS = [
    (EarningsToPrice, "net_income_ttm", "market_cap"),
    (ForwardEarningsToPrice, "eps_ntm", "adj_close"),
    (EbitdaToEnterpriseValue, "ebitda_ttm", "enterprise_value"),
]


@pytest.mark.parametrize(
    "descriptor_cls, numerator_field, denominator_field",
    EARNINGS_YIELD_DESCRIPTORS,
    ids=[cls.__name__ for cls, _, _ in EARNINGS_YIELD_DESCRIPTORS],
)
class TestEarningsYieldDescriptors:
    """Shared tests for earnings yield descriptors."""

    def test_output_shape(
        self, simple_panel, descriptor_cls, numerator_field, denominator_field
    ):
        """Output shape matches (n_observations, n_assets)."""
        result = descriptor_cls().fit_transform(simple_panel)
        assert result.shape == simple_panel[numerator_field].shape

    def test_partial_fit_transform_matches_fit_transform(
        self, simple_panel, descriptor_cls, numerator_field, denominator_field
    ):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = descriptor_cls()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_ratio(
        self, simple_panel, descriptor_cls, numerator_field, denominator_field
    ):
        """Output equals numerator / denominator."""
        result = descriptor_cls().fit_transform(simple_panel)
        expected = simple_panel[numerator_field] / simple_panel[denominator_field]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(
        self, simple_panel, descriptor_cls, numerator_field, denominator_field
    ):
        """NaN in numerator or denominator propagates to output."""
        simple_panel[numerator_field][0, 0] = np.nan
        simple_panel[denominator_field][1, 0] = np.nan
        result = descriptor_cls().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        # Other values unaffected
        assert not np.isnan(result[0, 1])

    def test_denominator_zero_handling(
        self, simple_panel, descriptor_cls, numerator_field, denominator_field
    ):
        """Zero enterprise value is masked; other zero denominators are invalid."""
        simple_panel[denominator_field][0, 0] = 0.0
        if denominator_field == "enterprise_value":
            result = descriptor_cls().fit_transform(simple_panel)
            assert np.isnan(result[0, 0])
            return
        with pytest.raises(
            ValueError, match=rf"{denominator_field}.*strictly positive"
        ):
            descriptor_cls().fit_transform(simple_panel)

    def test_denominator_negative_handling(
        self, simple_panel, descriptor_cls, numerator_field, denominator_field
    ):
        """Negative enterprise value is masked; other negative denominators are invalid."""
        simple_panel[denominator_field][0, 0] = -1.0
        if denominator_field == "enterprise_value":
            result = descriptor_cls().fit_transform(simple_panel)
            assert np.isnan(result[0, 0])
            return
        with pytest.raises(
            ValueError, match=rf"{denominator_field}.*strictly positive"
        ):
            descriptor_cls().fit_transform(simple_panel)

    def test_raise_when_denominator_infinite(
        self, simple_panel, descriptor_cls, numerator_field, denominator_field
    ):
        """Infinite denominators are invalid."""
        simple_panel[denominator_field][0, 0] = np.inf
        with pytest.raises(ValueError, match=rf"{denominator_field}.*finite"):
            descriptor_cls().fit_transform(simple_panel)

    def test_raise_when_numerator_infinite(
        self, simple_panel, descriptor_cls, numerator_field, denominator_field
    ):
        """Infinite numerators are invalid."""
        simple_panel[numerator_field][0, 0] = np.inf
        with pytest.raises(ValueError, match=rf"{numerator_field}.*finite"):
            descriptor_cls().fit_transform(simple_panel)
