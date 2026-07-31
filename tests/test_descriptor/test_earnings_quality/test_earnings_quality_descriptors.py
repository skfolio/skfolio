"""Tests for Earnings Quality descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import AccrualsCashFlow, AnalystDispersionToPrice


class TestAccrualsCashFlow:
    """Tests for AccrualsCashFlow descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = AccrualsCashFlow().fit_transform(simple_panel)
        assert result.shape == simple_panel["net_income_ttm"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = AccrualsCashFlow()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_formula(self, simple_panel):
        """Output equals (net_income - operating_cash_flow) / total_assets."""
        result = AccrualsCashFlow().fit_transform(simple_panel)
        expected = (
            simple_panel["net_income_ttm"] - simple_panel["operating_cash_flow_ttm"]
        ) / simple_panel["total_assets"]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in any input field propagates to output."""
        simple_panel["net_income_ttm"][3, 0] = np.nan
        simple_panel["operating_cash_flow_ttm"][4, 1] = np.nan
        simple_panel["total_assets"][5, 2] = np.nan
        result = AccrualsCashFlow().fit_transform(simple_panel)
        assert np.isnan(result[3, 0])
        assert np.isnan(result[4, 1])
        assert np.isnan(result[5, 2])

    def test_sign_convention(self, simple_panel):
        """Positive when earnings exceed cash flow (high accruals)."""
        # Force net_income > operating_cash_flow
        simple_panel["net_income_ttm"][0, 0] = 5e8
        simple_panel["operating_cash_flow_ttm"][0, 0] = 1e8
        result = AccrualsCashFlow().fit_transform(simple_panel)
        assert result[0, 0] > 0

        # Force net_income < operating_cash_flow
        simple_panel["net_income_ttm"][1, 0] = 1e8
        simple_panel["operating_cash_flow_ttm"][1, 0] = 5e8
        result = AccrualsCashFlow().fit_transform(simple_panel)
        assert result[1, 0] < 0

    def test_nan_when_total_assets_zero(self, simple_panel):
        """Output is NaN when total assets are zero."""
        simple_panel["total_assets"][0, 0] = 0.0
        result = AccrualsCashFlow().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])

    def test_nan_when_total_assets_negative(self, simple_panel):
        """Output is NaN when total assets are negative."""
        simple_panel["total_assets"][0, 0] = -1.0
        result = AccrualsCashFlow().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])

    def test_raise_when_total_assets_infinite(self, simple_panel):
        """Infinite total assets are invalid."""
        simple_panel["total_assets"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"total_assets.*finite"):
            AccrualsCashFlow().fit_transform(simple_panel)


class TestAnalystDispersionToPrice:
    """Tests for AnalystDispersionToPrice descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = AnalystDispersionToPrice().fit_transform(simple_panel)
        assert result.shape == simple_panel["eps_ntm_std"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = AnalystDispersionToPrice()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_formula(self, simple_panel):
        """Output equals eps_ntm_std / adj_close."""
        result = AnalystDispersionToPrice().fit_transform(simple_panel)
        expected = simple_panel["eps_ntm_std"] / simple_panel["adj_close"]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in either field propagates to output."""
        simple_panel["eps_ntm_std"][3, 0] = np.nan
        simple_panel["adj_close"][4, 1] = np.nan
        result = AnalystDispersionToPrice().fit_transform(simple_panel)
        assert np.isnan(result[3, 0])
        assert np.isnan(result[4, 1])

    def test_raise_when_eps_std_negative(self, simple_panel):
        """Forecast dispersion must be non-negative."""
        simple_panel["eps_ntm_std"][0, 0] = -1.0
        with pytest.raises(ValueError, match=r"eps_ntm_std.*non-negative"):
            AnalystDispersionToPrice().fit_transform(simple_panel)

    def test_raise_when_adj_close_non_positive(self, simple_panel):
        """Split-adjusted close must be strictly positive."""
        simple_panel["adj_close"][0, 0] = 0.0
        with pytest.raises(ValueError, match=r"adj_close.*strictly positive"):
            AnalystDispersionToPrice().fit_transform(simple_panel)

    def test_raise_when_adj_close_infinite(self, simple_panel):
        """Infinite split-adjusted close is invalid."""
        simple_panel["adj_close"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"adj_close.*strictly positive"):
            AnalystDispersionToPrice().fit_transform(simple_panel)
