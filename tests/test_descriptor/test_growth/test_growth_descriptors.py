"""Tests for Growth descriptors."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from skfolio.descriptor import (
    AssetsGrowthRate,
    CapexToAssetsChangeInIntensity,
    ChangeInIntensity,
    ChangeToScale,
    EarningsChangeToPrice,
    GrowthRate,
    IssuanceGrowthRate,
    SalesGrowthRate,
)


def _assert_streaming_correct(make, panel):
    """Verify chunked partial_fit_transform matches fit_transform."""
    full = make(lag=3).fit_transform(panel)

    # One-shot partial == fit
    np.testing.assert_array_equal(make(lag=3).partial_fit_transform(panel), full)

    # 3 chunks
    d = make(lag=3)
    parts = [d.partial_fit_transform(panel[s : s + 7]) for s in range(0, 20, 7)]
    np.testing.assert_array_almost_equal(np.concatenate(parts), full)

    # Chunks smaller than lag
    full5 = make(lag=5).fit_transform(panel)
    d = make(lag=5)
    parts = [d.partial_fit_transform(panel[s : s + 2]) for s in range(0, 20, 2)]
    np.testing.assert_array_almost_equal(np.concatenate(parts), full5)

    # fit_transform resets state
    d = make(lag=3)
    d.partial_fit_transform(panel[:10])
    np.testing.assert_array_equal(d.fit_transform(panel), full)


class TestGrowthRate:
    """Tests for GrowthRate descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = GrowthRate("sales_ttm", lag=3).fit_transform(simple_panel)
        assert result.shape == simple_panel["sales_ttm"].shape

    def test_first_lag_rows_are_nan(self, simple_panel):
        """First `lag` rows are NaN (no history)."""
        lag = 5
        result = GrowthRate("sales_ttm", lag=lag).fit_transform(simple_panel)
        assert np.all(np.isnan(result[:lag]))
        assert not np.all(np.isnan(result[lag:]))

    def test_values_match_formula(self, simple_panel):
        """Output equals x(t) / x(t-lag) - 1."""
        lag = 3
        result = GrowthRate("sales_ttm", lag=lag).fit_transform(simple_panel)
        sales = simple_panel["sales_ttm"]
        expected = sales[lag:] / sales[: len(sales) - lag] - 1
        np.testing.assert_array_equal(result[lag:], expected)

    def test_nan_propagation(self, simple_panel):
        """NaN in the field propagates to output."""
        simple_panel["sales_ttm"][5, 0] = np.nan
        result = GrowthRate("sales_ttm", lag=3).fit_transform(simple_panel)
        # NaN at t=5 affects growth at t=5 (current) and t=8 (as base)
        assert np.isnan(result[5, 0])
        assert np.isnan(result[8, 0])

    def test_raises_on_negative_values(self, simple_panel):
        """Raises ValueError when field contains negative values."""
        simple_panel["sales_ttm"][5, 0] = -100
        with pytest.raises(ValueError, match="negative values"):
            GrowthRate("sales_ttm", lag=3).fit_transform(simple_panel)

    def test_negative_value_message_suggests_cleaning(self, simple_panel):
        """Error message mentions cleaning the input field."""
        simple_panel["sales_ttm"][5, 0] = -100
        with pytest.raises(ValueError, match="clean the input field"):
            GrowthRate("sales_ttm", lag=3).fit_transform(simple_panel)

    def test_zero_base_produces_nan(self, simple_panel):
        """Zero lagged value produces NaN (division by zero)."""
        simple_panel["sales_ttm"][2, 0] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            result = GrowthRate("sales_ttm", lag=3).fit_transform(simple_panel)
        assert np.isnan(result[5, 0])

    def test_raises_on_infinite_values(self, simple_panel):
        """Raises ValueError when field contains infinite values."""
        simple_panel["sales_ttm"][5, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            GrowthRate("sales_ttm", lag=3).fit_transform(simple_panel)

    def test_growth_rate_fitted_attribute(self, simple_panel):
        """growth_rate_ stores the last growth-rate row."""
        descriptor = GrowthRate("sales_ttm", lag=3)
        result = descriptor.fit_transform(simple_panel)
        np.testing.assert_array_equal(descriptor.growth_rate_, result[-1])

    def test_raises_on_invalid_lag(self, simple_panel):
        """Raises ValueError when lag < 1."""
        with pytest.raises(ValueError, match="lag must be a positive integer"):
            GrowthRate("sales_ttm", lag=0).fit_transform(simple_panel)

    def test_lag_exceeds_observations(self, simple_panel):
        """All NaN when lag >= n_observations."""
        n_obs = simple_panel["sales_ttm"].shape[0]
        result = GrowthRate("sales_ttm", lag=n_obs).fit_transform(simple_panel)
        assert np.all(np.isnan(result))

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        lag = 3
        descriptor = GrowthRate("sales_ttm", lag=lag)
        full = descriptor.fit_transform(simple_panel)

        descriptor2 = GrowthRate("sales_ttm", lag=lag)
        partial = descriptor2.partial_fit_transform(simple_panel)

        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        lag = 3
        full = GrowthRate("sales_ttm", lag=lag).fit_transform(simple_panel)

        descriptor = GrowthRate("sales_ttm", lag=lag)
        # Split into 3 chunks
        chunk1 = simple_panel[:7]
        chunk2 = simple_panel[7:13]
        chunk3 = simple_panel[13:]

        r1 = descriptor.partial_fit_transform(chunk1)
        r2 = descriptor.partial_fit_transform(chunk2)
        r3 = descriptor.partial_fit_transform(chunk3)

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_small_chunks(self, simple_panel):
        """Chunks smaller than lag still produce correct results."""
        lag = 5
        full = GrowthRate("sales_ttm", lag=lag).fit_transform(simple_panel)

        descriptor = GrowthRate("sales_ttm", lag=lag)
        # Chunks of 2 (smaller than lag=5)
        chunks = []
        for start in range(0, 20, 2):
            view = simple_panel[start : start + 2]
            chunks.append(descriptor.partial_fit_transform(view))

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        lag = 3
        descriptor = GrowthRate("sales_ttm", lag=lag)

        # First run: partial with a chunk
        descriptor.partial_fit_transform(simple_panel[:10])

        # Second run: fit_transform should reset and produce clean result
        result = descriptor.fit_transform(simple_panel)
        expected = GrowthRate("sales_ttm", lag=lag).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)


class TestEarningsChangeToPrice:
    """Tests for EarningsChangeToPrice descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = EarningsChangeToPrice(lag=3).fit_transform(simple_panel)
        assert result.shape == simple_panel["net_income_ttm"].shape

    def test_first_lag_rows_are_nan(self, simple_panel):
        """First `lag` rows are NaN (no history)."""
        lag = 5
        result = EarningsChangeToPrice(lag=lag).fit_transform(simple_panel)
        assert np.all(np.isnan(result[:lag]))
        assert not np.all(np.isnan(result[lag:]))

    def test_values_match_formula(self, simple_panel):
        """Output equals (net_income(t) - net_income(t-lag)) / market_cap(t)."""
        lag = 3
        result = EarningsChangeToPrice(lag=lag).fit_transform(simple_panel)
        ni = simple_panel["net_income_ttm"]
        mc = simple_panel["market_cap"]
        expected = (ni[lag:] - ni[: len(ni) - lag]) / mc[lag:]
        np.testing.assert_array_equal(result[lag:], expected)

    def test_handles_negative_earnings(self, simple_panel):
        """Works correctly when earnings are negative."""
        # Force a negative -> positive transition
        simple_panel["net_income_ttm"][0, 0] = -1e8
        simple_panel["net_income_ttm"][3, 0] = 2e8
        result = EarningsChangeToPrice(lag=3).fit_transform(simple_panel)
        # Change is positive (improvement), market_cap is positive -> result > 0
        assert result[3, 0] > 0

    def test_handles_negative_to_more_negative(self, simple_panel):
        """Correctly shows deterioration for worsening losses."""
        simple_panel["net_income_ttm"][0, 0] = -1e8
        simple_panel["net_income_ttm"][3, 0] = -3e8
        result = EarningsChangeToPrice(lag=3).fit_transform(simple_panel)
        # Change is negative (deterioration) -> result < 0
        assert result[3, 0] < 0

    def test_nan_propagation(self, simple_panel):
        """NaN in earnings or market_cap propagates to output."""
        simple_panel["net_income_ttm"][5, 0] = np.nan
        simple_panel["market_cap"][6, 1] = np.nan
        result = EarningsChangeToPrice(lag=3).fit_transform(simple_panel)
        assert np.isnan(result[5, 0])  # NaN in current earnings
        assert np.isnan(result[8, 0])  # NaN in lagged earnings
        assert np.isnan(result[6, 1])  # NaN in market_cap

    def test_raises_on_invalid_lag(self, simple_panel):
        """Raises ValueError when lag < 1."""
        with pytest.raises(ValueError, match="lag must be a positive integer"):
            EarningsChangeToPrice(lag=0).fit_transform(simple_panel)

    def test_default_lag(self):
        """Default lag is 252."""
        descriptor = EarningsChangeToPrice()
        assert descriptor.lag == 252

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        lag = 3
        descriptor = EarningsChangeToPrice(lag=lag)
        full = descriptor.fit_transform(simple_panel)

        descriptor2 = EarningsChangeToPrice(lag=lag)
        partial = descriptor2.partial_fit_transform(simple_panel)

        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        lag = 3
        full = EarningsChangeToPrice(lag=lag).fit_transform(simple_panel)

        descriptor = EarningsChangeToPrice(lag=lag)
        # Split into 3 chunks
        chunk1 = simple_panel[:7]
        chunk2 = simple_panel[7:13]
        chunk3 = simple_panel[13:]

        r1 = descriptor.partial_fit_transform(chunk1)
        r2 = descriptor.partial_fit_transform(chunk2)
        r3 = descriptor.partial_fit_transform(chunk3)

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_small_chunks(self, simple_panel):
        """Chunks smaller than lag still produce correct results."""
        lag = 5
        full = EarningsChangeToPrice(lag=lag).fit_transform(simple_panel)

        descriptor = EarningsChangeToPrice(lag=lag)
        # Chunks of 2 (smaller than lag=5)
        chunks = []
        for start in range(0, 20, 2):
            view = simple_panel[start : start + 2]
            chunks.append(descriptor.partial_fit_transform(view))

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        lag = 3
        descriptor = EarningsChangeToPrice(lag=lag)

        # First run: partial with a chunk
        descriptor.partial_fit_transform(simple_panel[:10])

        # Second run: fit_transform should reset and produce clean result
        result = descriptor.fit_transform(simple_panel)
        expected = EarningsChangeToPrice(lag=lag).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)


class TestChangeToScale:
    """Tests for ChangeToScale descriptor."""

    def _make(self, lag):
        return ChangeToScale("net_income_ttm", "market_cap", lag=lag)

    def test_values_match_formula(self, simple_panel):
        """Output equals (A(t) - A(t-lag)) / S(t)."""
        lag = 3
        result = self._make(lag).fit_transform(simple_panel)
        ni = simple_panel["net_income_ttm"]
        mc = simple_panel["market_cap"]
        expected = (ni[lag:] - ni[: len(ni) - lag]) / mc[lag:]
        np.testing.assert_array_equal(result[lag:], expected)

    def test_first_lag_rows_are_nan(self, simple_panel):
        lag = 5
        result = self._make(lag).fit_transform(simple_panel)
        assert np.all(np.isnan(result[:lag]))
        assert not np.all(np.isnan(result[lag:]))

    def test_raises_on_non_positive_scale(self, simple_panel):
        simple_panel["market_cap"][3, 0] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            self._make(3).fit_transform(simple_panel)

    def test_raises_on_negative_scale(self, simple_panel):
        simple_panel["market_cap"][3, 0] = -1.0
        with pytest.raises(ValueError, match="non-positive"):
            self._make(3).fit_transform(simple_panel)

    def test_raises_on_infinite_field(self, simple_panel):
        simple_panel["net_income_ttm"][3, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            self._make(3).fit_transform(simple_panel)

    def test_raises_on_infinite_scale(self, simple_panel):
        simple_panel["market_cap"][3, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            self._make(3).fit_transform(simple_panel)

    def test_nan_propagation(self, simple_panel):
        simple_panel["net_income_ttm"][5, 0] = np.nan
        simple_panel["market_cap"][6, 1] = np.nan
        result = self._make(3).fit_transform(simple_panel)
        assert np.isnan(result[5, 0])
        assert np.isnan(result[8, 0])
        assert np.isnan(result[6, 1])

    def test_change_to_scale_fitted_attribute(self, simple_panel):
        descriptor = self._make(3)
        result = descriptor.fit_transform(simple_panel)
        np.testing.assert_array_equal(descriptor.change_to_scale_, result[-1])

    def test_raises_on_invalid_lag(self, simple_panel):
        with pytest.raises(ValueError, match="lag must be a positive integer"):
            self._make(0).fit_transform(simple_panel)

    def test_streaming(self, simple_panel):
        _assert_streaming_correct(self._make, simple_panel)


class TestChangeInIntensity:
    """Tests for ChangeInIntensity descriptor."""

    def _make(self, lag):
        return ChangeInIntensity("net_income_ttm", "market_cap", lag=lag)

    def test_values_match_formula(self, simple_panel):
        """Output equals A(t)/S(t) - A(t-lag)/S(t-lag)."""
        lag = 3
        result = self._make(lag).fit_transform(simple_panel)
        ni = simple_panel["net_income_ttm"]
        mc = simple_panel["market_cap"]
        ratio = ni / mc
        expected = ratio[lag:] - ratio[: len(ratio) - lag]
        np.testing.assert_array_almost_equal(result[lag:], expected)

    def test_first_lag_rows_are_nan(self, simple_panel):
        lag = 5
        result = self._make(lag).fit_transform(simple_panel)
        assert np.all(np.isnan(result[:lag]))
        assert not np.all(np.isnan(result[lag:]))

    def test_raises_on_non_positive_scale(self, simple_panel):
        simple_panel["market_cap"][3, 0] = 0.0
        with pytest.raises(ValueError, match="non-positive"):
            self._make(3).fit_transform(simple_panel)

    def test_raises_on_negative_scale(self, simple_panel):
        simple_panel["market_cap"][3, 0] = -1.0
        with pytest.raises(ValueError, match="non-positive"):
            self._make(3).fit_transform(simple_panel)

    def test_raises_on_infinite_field(self, simple_panel):
        simple_panel["net_income_ttm"][3, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            self._make(3).fit_transform(simple_panel)

    def test_raises_on_infinite_scale(self, simple_panel):
        simple_panel["market_cap"][3, 0] = np.inf
        with pytest.raises(ValueError, match="infinite values"):
            self._make(3).fit_transform(simple_panel)

    def test_nan_propagation(self, simple_panel):
        simple_panel["net_income_ttm"][5, 0] = np.nan
        simple_panel["market_cap"][6, 1] = np.nan
        result = self._make(3).fit_transform(simple_panel)
        assert np.isnan(result[5, 0])
        assert np.isnan(result[8, 0])
        assert np.isnan(result[6, 1])

    def test_change_in_intensity_fitted_attribute(self, simple_panel):
        descriptor = self._make(3)
        result = descriptor.fit_transform(simple_panel)
        np.testing.assert_array_equal(descriptor.change_in_intensity_, result[-1])

    def test_raises_on_invalid_lag(self, simple_panel):
        with pytest.raises(ValueError, match="lag must be a positive integer"):
            self._make(0).fit_transform(simple_panel)

    def test_streaming(self, simple_panel):
        _assert_streaming_correct(self._make, simple_panel)


class TestCapexToAssetsChangeInIntensity:
    """Tests for CapexToAssetsChangeInIntensity descriptor."""

    @staticmethod
    def _panel_with_capex(simple_panel):
        simple_panel["capex_ttm"] = np.abs(simple_panel["sales_ttm"]) * 0.1
        return simple_panel

    def test_default_parameters(self):
        """Default parameters are field=capex_ttm, scale=total_assets, lag=252."""
        descriptor = CapexToAssetsChangeInIntensity()
        assert descriptor.field == "capex_ttm"
        assert descriptor.scale_field == "total_assets"
        assert descriptor.lag == 252

    def test_values_match_change_in_intensity(self, simple_panel):
        """Output matches ChangeInIntensity(capex_ttm, total_assets)."""
        panel = self._panel_with_capex(simple_panel)
        lag = 3
        expected = ChangeInIntensity(
            field="capex_ttm", scale_field="total_assets", lag=lag
        ).fit_transform(panel)
        result = CapexToAssetsChangeInIntensity(lag=lag).fit_transform(panel)
        np.testing.assert_array_almost_equal(result, expected)


class TestSalesGrowthRate:
    """Tests for SalesGrowthRate descriptor."""

    def test_default_parameters(self):
        """Default parameters are field=sales_ttm, lag=252."""
        descriptor = SalesGrowthRate()
        assert descriptor.field == "sales_ttm"
        assert descriptor.lag == 252

    def test_values_match_growth_rate(self, simple_panel):
        """Output matches GrowthRate(sales_ttm)."""
        lag = 3
        expected = GrowthRate(field="sales_ttm", lag=lag).fit_transform(simple_panel)
        result = SalesGrowthRate(lag=lag).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)


class TestTotalAssetsGrowthRate:
    """Tests for TotalAssetsGrowthRate descriptor."""

    def test_default_parameters(self):
        """Default parameters are field=total_assets, lag=252."""
        descriptor = AssetsGrowthRate()
        assert descriptor.field == "total_assets"
        assert descriptor.lag == 252

    def test_values_match_growth_rate(self, simple_panel):
        """Output matches GrowthRate(total_assets)."""
        lag = 3
        expected = GrowthRate(field="total_assets", lag=lag).fit_transform(simple_panel)
        result = AssetsGrowthRate(lag=lag).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)


class TestIssuanceGrowthRate:
    """Tests for IssuanceGrowthRate descriptor."""

    def test_default_parameters(self):
        """Default parameters are field=adj_shares_outstanding, lag=252."""
        descriptor = IssuanceGrowthRate()
        assert descriptor.field == "adj_shares_outstanding"
        assert descriptor.lag == 252

    def test_values_match_growth_rate(self, simple_panel):
        """Output matches GrowthRate(adj_shares_outstanding)."""
        lag = 3
        expected = GrowthRate(field="adj_shares_outstanding", lag=lag).fit_transform(
            simple_panel
        )
        result = IssuanceGrowthRate(lag=lag).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)
