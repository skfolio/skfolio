"""Tests for Dividend Yield descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import (
    DividendToPrice,
    ForwardDividendToPrice,
    ShareholderYield,
)


# ---------------------------------------------------------------------------
# DividendToPrice (aggregate: dividends_ttm / market_cap)
# ---------------------------------------------------------------------------
class TestDividendToPrice:
    """Tests for DividendToPrice descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = DividendToPrice().fit_transform(simple_panel)
        assert result.shape == simple_panel["market_cap"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = DividendToPrice()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_ratio(self, simple_panel):
        """Output equals dividends_ttm / market_cap."""
        result = DividendToPrice().fit_transform(simple_panel)
        expected = simple_panel["dividends_ttm"] / simple_panel["market_cap"]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in numerator or denominator propagates to output."""
        simple_panel["dividends_ttm"][0, 0] = np.nan
        simple_panel["market_cap"][1, 0] = np.nan
        result = DividendToPrice().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        assert not np.isnan(result[0, 1])

    def test_raise_when_dividends_negative(self, simple_panel):
        """Trailing dividends must be non-negative."""
        simple_panel["dividends_ttm"][0, 0] = -1.0
        with pytest.raises(ValueError, match=r"dividends_ttm.*non-negative"):
            DividendToPrice().fit_transform(simple_panel)

    def test_raise_when_market_cap_non_positive(self, simple_panel):
        """Market cap must be strictly positive."""
        simple_panel["market_cap"][0, 0] = 0.0
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            DividendToPrice().fit_transform(simple_panel)

    def test_raise_when_market_cap_infinite(self, simple_panel):
        """Infinite market cap is invalid."""
        simple_panel["market_cap"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            DividendToPrice().fit_transform(simple_panel)


# ---------------------------------------------------------------------------
# ForwardDividendToPrice (per-share: dps_ntm / adj_close)
# ---------------------------------------------------------------------------
class TestForwardDividendToPrice:
    """Tests for ForwardDividendToPrice descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = ForwardDividendToPrice().fit_transform(simple_panel)
        assert result.shape == simple_panel["adj_close"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = ForwardDividendToPrice()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_ratio(self, simple_panel):
        """Output equals dps_ntm / adj_close."""
        result = ForwardDividendToPrice().fit_transform(simple_panel)
        expected = simple_panel["dps_ntm"] / simple_panel["adj_close"]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in numerator or denominator propagates to output."""
        simple_panel["dps_ntm"][0, 0] = np.nan
        simple_panel["adj_close"][1, 0] = np.nan
        result = ForwardDividendToPrice().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        assert not np.isnan(result[0, 1])

    def test_raise_when_dps_negative(self, simple_panel):
        """Forward dividends must be non-negative."""
        simple_panel["dps_ntm"][0, 0] = -1.0
        with pytest.raises(ValueError, match=r"dps_ntm.*non-negative"):
            ForwardDividendToPrice().fit_transform(simple_panel)

    def test_raise_when_adj_close_non_positive(self, simple_panel):
        """Split-adjusted close must be strictly positive."""
        simple_panel["adj_close"][0, 0] = 0.0
        with pytest.raises(ValueError, match=r"adj_close.*strictly positive"):
            ForwardDividendToPrice().fit_transform(simple_panel)

    def test_raise_when_adj_close_infinite(self, simple_panel):
        """Infinite split-adjusted close is invalid."""
        simple_panel["adj_close"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"adj_close.*strictly positive"):
            ForwardDividendToPrice().fit_transform(simple_panel)


# ---------------------------------------------------------------------------
# ShareholderYield ((dividends_ttm + net_buybacks_ttm) / market_cap)
# ---------------------------------------------------------------------------
class TestShareholderYield:
    """Tests for ShareholderYield descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = ShareholderYield().fit_transform(simple_panel)
        assert result.shape == simple_panel["market_cap"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = ShareholderYield()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_formula(self, simple_panel):
        """Output equals (dividends_ttm + net_buybacks_ttm) / market_cap."""
        result = ShareholderYield().fit_transform(simple_panel)
        expected = (
            simple_panel["dividends_ttm"] + simple_panel["net_buybacks_ttm"]
        ) / simple_panel["market_cap"]
        np.testing.assert_array_equal(result, expected)

    def test_zero_buybacks_equals_dividend_yield(self, simple_panel):
        """With zero buybacks, shareholder yield equals dividend yield."""
        simple_panel["net_buybacks_ttm"][:] = 0.0
        sh_yield = ShareholderYield().fit_transform(simple_panel)
        div_yield = DividendToPrice().fit_transform(simple_panel)
        np.testing.assert_array_equal(sh_yield, div_yield)

    def test_negative_buybacks_reduces_yield(self, simple_panel):
        """Negative net buybacks (net issuance) reduces yield vs dividends."""
        simple_panel["net_buybacks_ttm"][:] = -1e8  # net issuance
        sh_yield = ShareholderYield().fit_transform(simple_panel)
        div_yield = DividendToPrice().fit_transform(simple_panel)
        assert np.all(sh_yield < div_yield)

    def test_nan_propagation(self, simple_panel):
        """NaN in any input field propagates to output."""
        simple_panel["dividends_ttm"][0, 0] = np.nan
        simple_panel["net_buybacks_ttm"][1, 0] = np.nan
        simple_panel["market_cap"][2, 0] = np.nan
        result = ShareholderYield().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        assert np.isnan(result[2, 0])
        assert not np.isnan(result[0, 1])

    def test_raise_when_dividends_negative(self, simple_panel):
        """Trailing dividends must be non-negative."""
        simple_panel["dividends_ttm"][0, 0] = -1.0
        with pytest.raises(ValueError, match=r"dividends_ttm.*non-negative"):
            ShareholderYield().fit_transform(simple_panel)

    def test_raise_when_net_buybacks_infinite(self, simple_panel):
        """Infinite net buybacks are invalid."""
        simple_panel["net_buybacks_ttm"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"net_buybacks_ttm.*finite"):
            ShareholderYield().fit_transform(simple_panel)

    def test_raise_when_market_cap_non_positive(self, simple_panel):
        """Market cap must be strictly positive."""
        simple_panel["market_cap"][0, 0] = 0.0
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            ShareholderYield().fit_transform(simple_panel)

    def test_raise_when_market_cap_infinite(self, simple_panel):
        """Infinite market cap is invalid."""
        simple_panel["market_cap"][0, 0] = np.inf
        with pytest.raises(ValueError, match=r"market_cap.*strictly positive"):
            ShareholderYield().fit_transform(simple_panel)
