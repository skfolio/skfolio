"""Tests for Profitability descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import (
    AssetTurnover,
    CashFlowToAssets,
    GrossMargin,
    GrossProfitability,
    ReturnOnAssets,
    ReturnOnEquity,
    SalesToEnterpriseValue,
)

# ── Simple ratio descriptors (numerator / denominator) ──────────────────────


class _SimpleRatioCase:
    """Helper to define a simple ratio test case."""

    def __init__(self, cls, numerator_field, denominator_field):
        self.cls = cls
        self.numerator_field = numerator_field
        self.denominator_field = denominator_field


_SIMPLE_RATIO_CASES = [
    pytest.param(
        _SimpleRatioCase(ReturnOnAssets, "net_income_ttm", "total_assets"),
        id="ReturnOnAssets",
    ),
    pytest.param(
        _SimpleRatioCase(AssetTurnover, "sales_ttm", "total_assets"),
        id="AssetTurnover",
    ),
    pytest.param(
        _SimpleRatioCase(CashFlowToAssets, "operating_cash_flow_ttm", "total_assets"),
        id="CashFlowToAssets",
    ),
    pytest.param(
        _SimpleRatioCase(SalesToEnterpriseValue, "sales_ttm", "enterprise_value"),
        id="SalesToEnterpriseValue",
    ),
]

_TOTAL_ASSETS_RATIO_CASES = [
    pytest.param(
        _SimpleRatioCase(ReturnOnAssets, "net_income_ttm", "total_assets"),
        id="ReturnOnAssets",
    ),
    pytest.param(
        _SimpleRatioCase(AssetTurnover, "sales_ttm", "total_assets"),
        id="AssetTurnover",
    ),
    pytest.param(
        _SimpleRatioCase(CashFlowToAssets, "operating_cash_flow_ttm", "total_assets"),
        id="CashFlowToAssets",
    ),
]


class TestSimpleRatioDescriptors:
    """Tests for profitability descriptors that compute numerator / denominator."""

    @pytest.mark.parametrize("case", _SIMPLE_RATIO_CASES)
    def test_output_shape(self, simple_panel, case):
        """Output shape matches input."""
        result = case.cls().fit_transform(simple_panel)
        assert result.shape == simple_panel[case.numerator_field].shape

    @pytest.mark.parametrize("case", _SIMPLE_RATIO_CASES)
    def test_partial_fit_transform_matches_fit_transform(self, simple_panel, case):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = case.cls()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    @pytest.mark.parametrize("case", _SIMPLE_RATIO_CASES)
    def test_values_match_formula(self, simple_panel, case):
        """Output equals numerator / denominator."""
        result = case.cls().fit_transform(simple_panel)
        expected = (
            simple_panel[case.numerator_field] / simple_panel[case.denominator_field]
        )
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("case", _SIMPLE_RATIO_CASES)
    def test_nan_propagation(self, simple_panel, case):
        """NaN in numerator or denominator propagates to output."""
        simple_panel[case.numerator_field][3, 0] = np.nan
        simple_panel[case.denominator_field][4, 1] = np.nan
        result = case.cls().fit_transform(simple_panel)
        assert np.isnan(result[3, 0])
        assert np.isnan(result[4, 1])

    @pytest.mark.parametrize("case", _TOTAL_ASSETS_RATIO_CASES)
    def test_nan_when_total_assets_zero(self, simple_panel, case):
        """Output is NaN when total assets are zero."""
        simple_panel["total_assets"][5, 0] = 0.0
        result = case.cls().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    @pytest.mark.parametrize("case", _TOTAL_ASSETS_RATIO_CASES)
    def test_nan_when_total_assets_negative(self, simple_panel, case):
        """Output is NaN when total assets are negative."""
        simple_panel["total_assets"][5, 0] = -1.0
        result = case.cls().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    @pytest.mark.parametrize("case", _TOTAL_ASSETS_RATIO_CASES)
    def test_raise_when_total_assets_infinite(self, simple_panel, case):
        """Infinite total assets are invalid."""
        simple_panel["total_assets"][5, 0] = np.inf
        with pytest.raises(ValueError, match=r"total_assets.*finite"):
            case.cls().fit_transform(simple_panel)

    def test_nan_when_enterprise_value_zero(self, simple_panel):
        """Output is NaN when enterprise value is zero."""
        simple_panel["enterprise_value"][5, 0] = 0.0
        result = SalesToEnterpriseValue().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_nan_when_enterprise_value_negative(self, simple_panel):
        """Output is NaN when enterprise value is negative."""
        simple_panel["enterprise_value"][5, 0] = -1.0
        result = SalesToEnterpriseValue().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_raise_when_enterprise_value_infinite(self, simple_panel):
        """Infinite enterprise value is invalid."""
        simple_panel["enterprise_value"][5, 0] = np.inf
        with pytest.raises(ValueError, match=r"enterprise_value.*finite"):
            SalesToEnterpriseValue().fit_transform(simple_panel)


# ── ReturnOnEquity (NaN masking for equity <= 0) ────────────────────────────


class TestReturnOnEquity:
    """Tests for ReturnOnEquity descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = ReturnOnEquity().fit_transform(simple_panel)
        assert result.shape == simple_panel["net_income_ttm"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = ReturnOnEquity()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_formula(self, simple_panel):
        """Output equals net_income / book_equity where equity > 0."""
        result = ReturnOnEquity().fit_transform(simple_panel)
        ni = simple_panel["net_income_ttm"]
        eq = simple_panel["book_equity"]
        expected = ni / eq
        # All fixture equity values are > 0
        np.testing.assert_array_equal(result, expected)

    def test_nan_when_equity_zero(self, simple_panel):
        """Output is NaN when book_equity == 0."""
        simple_panel["book_equity"][5, 0] = 0.0
        result = ReturnOnEquity().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_nan_when_equity_negative(self, simple_panel):
        """Output is NaN when book_equity < 0."""
        simple_panel["book_equity"][5, 0] = -1e8
        result = ReturnOnEquity().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_positive_equity_unaffected(self, simple_panel):
        """Positive equity observations remain valid."""
        simple_panel["book_equity"][5, 0] = -1e8
        result = ReturnOnEquity().fit_transform(simple_panel)
        # Other observations should still be valid (fixture equity > 0)
        assert not np.isnan(result[4, 0])
        assert not np.isnan(result[6, 0])

    def test_nan_propagation(self, simple_panel):
        """NaN in net_income or book_equity propagates to output."""
        simple_panel["net_income_ttm"][3, 0] = np.nan
        result = ReturnOnEquity().fit_transform(simple_panel)
        assert np.isnan(result[3, 0])


# ── GrossProfitability ((sales - cogs) / total_assets) ──────────────────────


class TestGrossProfitability:
    """Tests for GrossProfitability descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = GrossProfitability().fit_transform(simple_panel)
        assert result.shape == simple_panel["sales_ttm"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = GrossProfitability()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_formula(self, simple_panel):
        """Output equals (sales - cogs) / total_assets."""
        result = GrossProfitability().fit_transform(simple_panel)
        expected = (
            simple_panel["sales_ttm"] - simple_panel["cost_of_revenue_ttm"]
        ) / simple_panel["total_assets"]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in any input field propagates to output."""
        simple_panel["sales_ttm"][3, 0] = np.nan
        simple_panel["cost_of_revenue_ttm"][4, 1] = np.nan
        simple_panel["total_assets"][5, 2] = np.nan
        result = GrossProfitability().fit_transform(simple_panel)
        assert np.isnan(result[3, 0])
        assert np.isnan(result[4, 1])
        assert np.isnan(result[5, 2])

    def test_nan_when_total_assets_zero(self, simple_panel):
        """Output is NaN when total assets are zero."""
        simple_panel["total_assets"][5, 0] = 0.0
        result = GrossProfitability().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_nan_when_total_assets_negative(self, simple_panel):
        """Output is NaN when total assets are negative."""
        simple_panel["total_assets"][5, 0] = -1.0
        result = GrossProfitability().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_raise_when_total_assets_infinite(self, simple_panel):
        """Infinite total assets are invalid."""
        simple_panel["total_assets"][5, 0] = np.inf
        with pytest.raises(ValueError, match=r"total_assets.*finite"):
            GrossProfitability().fit_transform(simple_panel)


# ── GrossMargin ((sales - cogs) / sales) ────────────────────────────────────


class TestGrossMargin:
    """Tests for GrossMargin descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = GrossMargin().fit_transform(simple_panel)
        assert result.shape == simple_panel["sales_ttm"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = GrossMargin()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_formula(self, simple_panel):
        """Output equals (sales - cogs) / sales."""
        result = GrossMargin().fit_transform(simple_panel)
        expected = (
            simple_panel["sales_ttm"] - simple_panel["cost_of_revenue_ttm"]
        ) / simple_panel["sales_ttm"]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in sales or cost_of_revenue propagates to output."""
        simple_panel["sales_ttm"][3, 0] = np.nan
        simple_panel["cost_of_revenue_ttm"][4, 1] = np.nan
        result = GrossMargin().fit_transform(simple_panel)
        assert np.isnan(result[3, 0])
        assert np.isnan(result[4, 1])

    def test_nan_when_sales_zero(self, simple_panel):
        """Output is NaN when sales are zero."""
        simple_panel["sales_ttm"][5, 0] = 0.0
        result = GrossMargin().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_nan_when_sales_negative(self, simple_panel):
        """Output is NaN when sales are negative."""
        simple_panel["sales_ttm"][5, 0] = -1.0
        result = GrossMargin().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])
