"""Tests for Leverage descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import (
    BookLeverage,
    DebtToAssets,
    MarketLeverage,
)


# ---------------------------------------------------------------------------
# DebtToAssets (total_debt / total_assets)
# ---------------------------------------------------------------------------
class TestDebtToAssets:
    """Tests for DebtToAssets descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = DebtToAssets().fit_transform(simple_panel)
        assert result.shape == simple_panel["total_assets"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = DebtToAssets()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_ratio(self, simple_panel):
        """Output equals total_debt / total_assets."""
        result = DebtToAssets().fit_transform(simple_panel)
        expected = simple_panel["total_debt"] / simple_panel["total_assets"]
        np.testing.assert_array_equal(result, expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in numerator or denominator propagates to output."""
        simple_panel["total_debt"][0, 0] = np.nan
        simple_panel["total_assets"][1, 0] = np.nan
        result = DebtToAssets().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        assert not np.isnan(result[0, 1])

    def test_nan_when_total_assets_zero(self, simple_panel):
        """Output is NaN when total_assets is zero."""
        simple_panel["total_assets"][0, 0] = 0.0
        result = DebtToAssets().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])

    def test_nan_when_total_assets_negative(self, simple_panel):
        """Output is NaN when total_assets is negative."""
        simple_panel["total_assets"][0, 0] = -1.0
        result = DebtToAssets().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])

    def test_raises_on_infinite_debt(self, simple_panel):
        """Raises ValueError when total_debt contains infinite values."""
        simple_panel["total_debt"][0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            DebtToAssets().fit_transform(simple_panel)

    def test_raises_on_infinite_total_assets(self, simple_panel):
        """Raises ValueError when total_assets contains infinite values."""
        simple_panel["total_assets"][0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            DebtToAssets().fit_transform(simple_panel)


# ---------------------------------------------------------------------------
# BookLeverage (total_debt / (total_debt + book_equity))
# ---------------------------------------------------------------------------
class TestBookLeverage:
    """Tests for BookLeverage descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = BookLeverage().fit_transform(simple_panel)
        assert result.shape == simple_panel["total_debt"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = BookLeverage()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_formula(self, simple_panel):
        """Output equals total_debt / (total_debt + book_equity)."""
        result = BookLeverage().fit_transform(simple_panel)
        expected = simple_panel["total_debt"] / (
            simple_panel["total_debt"] + simple_panel["book_equity"]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_bounded_zero_one_healthy(self, simple_panel):
        """Output is in [0, 1] when book_equity > 0 and total_debt >= 0."""
        # Default fixture has positive book_equity and total_debt
        result = BookLeverage().fit_transform(simple_panel)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)
        assert np.all(result[valid] <= 1)

    def test_exceeds_one_with_negative_equity(self, simple_panel):
        """Ratio > 1 when book_equity < 0 but denominator > 0."""
        # Set book_equity negative but less than total_debt
        simple_panel["book_equity"][5, 0] = -1e8
        simple_panel["total_debt"][5, 0] = 5e8  # denom = 5e8 - 1e8 = 4e8 > 0
        result = BookLeverage().fit_transform(simple_panel)
        assert result[5, 0] > 1.0
        assert not np.isnan(result[5, 0])

    def test_nan_when_denominator_zero(self, simple_panel):
        """Output is NaN when total_debt + book_equity = 0."""
        simple_panel["total_debt"][5, 0] = 1e8
        simple_panel["book_equity"][5, 0] = -1e8  # denom = 0
        result = BookLeverage().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_nan_when_denominator_negative(self, simple_panel):
        """Output is NaN when total_debt + book_equity < 0."""
        simple_panel["total_debt"][5, 0] = 1e8
        simple_panel["book_equity"][5, 0] = -5e8  # denom = -4e8
        result = BookLeverage().fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_nan_propagation(self, simple_panel):
        """NaN in any input field propagates to output."""
        simple_panel["total_debt"][0, 0] = np.nan
        simple_panel["book_equity"][1, 0] = np.nan
        result = BookLeverage().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        assert not np.isnan(result[0, 1])

    def test_raises_on_infinite_debt(self, simple_panel):
        """Raises ValueError when total_debt contains infinite values."""
        simple_panel["total_debt"][0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            BookLeverage().fit_transform(simple_panel)

    def test_raises_on_infinite_book_equity(self, simple_panel):
        """Raises ValueError when book_equity contains infinite values."""
        simple_panel["book_equity"][0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            BookLeverage().fit_transform(simple_panel)

    def test_zero_debt_gives_zero(self, simple_panel):
        """Zero debt produces zero leverage."""
        simple_panel["total_debt"][:] = 0.0
        result = BookLeverage().fit_transform(simple_panel)
        valid = ~np.isnan(result)
        np.testing.assert_array_almost_equal(result[valid], 0.0)


# ---------------------------------------------------------------------------
# MarketLeverage (total_debt / (total_debt + market_cap))
# ---------------------------------------------------------------------------
class TestMarketLeverage:
    """Tests for MarketLeverage descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = MarketLeverage().fit_transform(simple_panel)
        assert result.shape == simple_panel["total_debt"].shape

    def test_partial_fit_transform_matches_fit_transform(self, simple_panel):
        """Stateless descriptor: partial_fit_transform equals fit_transform on the same panel."""
        est = MarketLeverage()
        result = est.fit_transform(simple_panel)
        result_partial = est.partial_fit_transform(simple_panel)
        np.testing.assert_allclose(
            result_partial, result, rtol=0, atol=0, equal_nan=True
        )

    def test_values_match_formula(self, simple_panel):
        """Output equals total_debt / (total_debt + market_cap)."""
        result = MarketLeverage().fit_transform(simple_panel)
        expected = simple_panel["total_debt"] / (
            simple_panel["total_debt"] + simple_panel["market_cap"]
        )
        np.testing.assert_array_equal(result, expected)

    def test_bounded_zero_one(self, simple_panel):
        """Output is in [0, 1) since market_cap > 0 and total_debt >= 0."""
        result = MarketLeverage().fit_transform(simple_panel)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)
        assert np.all(result[valid] < 1)

    def test_zero_debt_gives_zero(self, simple_panel):
        """Zero debt produces zero leverage."""
        simple_panel["total_debt"][:] = 0.0
        result = MarketLeverage().fit_transform(simple_panel)
        np.testing.assert_array_almost_equal(result, 0.0)

    def test_higher_debt_higher_leverage(self, simple_panel):
        """More debt relative to market cap increases leverage."""
        result_low = MarketLeverage().fit_transform(simple_panel)

        simple_panel["total_debt"][:] *= 10
        result_high = MarketLeverage().fit_transform(simple_panel)

        valid = ~np.isnan(result_low) & ~np.isnan(result_high)
        assert np.all(result_high[valid] > result_low[valid])

    def test_nan_propagation(self, simple_panel):
        """NaN in any input field propagates to output."""
        simple_panel["total_debt"][0, 0] = np.nan
        simple_panel["market_cap"][1, 0] = np.nan
        result = MarketLeverage().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[1, 0])
        assert not np.isnan(result[0, 1])

    def test_nan_when_market_cap_zero(self, simple_panel):
        """Output is NaN when market_cap is zero."""
        simple_panel["market_cap"][0, 0] = 0.0
        result = MarketLeverage().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])

    def test_nan_when_market_cap_negative(self, simple_panel):
        """Output is NaN when market_cap is negative."""
        simple_panel["market_cap"][0, 0] = -1.0
        result = MarketLeverage().fit_transform(simple_panel)
        assert np.isnan(result[0, 0])

    def test_raises_on_infinite_debt(self, simple_panel):
        """Raises ValueError when total_debt contains infinite values."""
        simple_panel["total_debt"][0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            MarketLeverage().fit_transform(simple_panel)

    def test_raises_on_infinite_market_cap(self, simple_panel):
        """Raises ValueError when market_cap contains infinite values."""
        simple_panel["market_cap"][0, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            MarketLeverage().fit_transform(simple_panel)
