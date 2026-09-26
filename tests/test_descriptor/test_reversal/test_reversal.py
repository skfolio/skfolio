"""Tests for Reversal descriptor."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import Reversal


class TestReversal:
    """Tests for the fixed-window Reversal descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = Reversal(window=5).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_early_observations_are_nan(self, simple_panel):
        """Observations before window are NaN, window-th is finite."""
        window = 5
        result = Reversal(window=window).fit_transform(simple_panel)
        # First (window-1) rows are NaN
        assert np.all(np.isnan(result[: window - 1]))
        # Row at index (window-1) is the first valid observation
        assert not np.any(np.isnan(result[window - 1]))

    def test_window_1_equals_negative_log_return(self, simple_panel):
        """With window=1, reversal is -log(1+r)."""
        result = Reversal(window=1).fit_transform(simple_panel)
        expected = -np.log1p(simple_panel["returns"])
        np.testing.assert_array_almost_equal(result, expected)

    def test_log_space_formula(self, simple_panel):
        """Output matches -sum(log(1+r)) over the trailing window."""
        window = 5
        result = Reversal(window=window).fit_transform(simple_panel)
        returns = simple_panel["returns"]
        log_returns = np.log1p(returns)

        for t in range(window - 1, returns.shape[0]):
            expected = -np.sum(log_returns[t - window + 1 : t + 1], axis=0)
            np.testing.assert_array_almost_equal(result[t], expected)

    def test_monotonic_with_simple_return(self, simple_panel):
        """Log-space output preserves ranking vs simple cumulative return."""
        window = 5
        result = Reversal(window=window).fit_transform(simple_panel)
        returns = simple_panel["returns"]

        # Compute simple cumulative return for comparison
        for t in range(window - 1, returns.shape[0]):
            simple_cum = np.prod(1 + returns[t - window + 1 : t + 1], axis=0) - 1
            neg_simple = -simple_cum
            # Rankings should be identical (log is monotonic)
            log_rank = np.argsort(np.argsort(result[t]))
            simple_rank = np.argsort(np.argsort(neg_simple))
            np.testing.assert_array_equal(log_rank, simple_rank)

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        full = Reversal(window=5).fit_transform(simple_panel)
        partial = Reversal(window=5).partial_fit_transform(simple_panel)
        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        window = 5
        full = Reversal(window=window).fit_transform(simple_panel)

        descriptor = Reversal(window=window)
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_single_obs_chunks(self, simple_panel):
        """Single-observation chunks produce correct results."""
        window = 5
        full = Reversal(window=window).fit_transform(simple_panel)

        descriptor = Reversal(window=window)
        chunks = []
        for start in range(20):
            chunks.append(
                descriptor.partial_fit_transform(simple_panel[start : start + 1])
            )

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        descriptor = Reversal(window=5)
        descriptor.partial_fit_transform(simple_panel[:10])

        result = descriptor.fit_transform(simple_panel)
        expected = Reversal(window=5).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)

    def test_nan_return_holds_state(self, simple_panel):
        """NaN return contributes zero to the running sum (not NaN)."""
        window = 5
        simple_panel["returns"][8, 0] = np.nan
        result = Reversal(window=window).fit_transform(simple_panel)

        t = 9
        log_returns = np.log1p(simple_panel["returns"])
        log_returns[8, 0] = 0.0
        expected = -np.sum(log_returns[t - window + 1 : t + 1, 0])
        assert not np.isnan(result[t, 0])
        np.testing.assert_almost_equal(result[t, 0], expected)

    def test_late_listing_needs_full_active_window(self, simple_panel):
        """Late-listed assets need a full active lookback window before output."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = Reversal(window=3).fit_transform(simple_panel)

        assert np.all(np.isnan(result[:7, 0]))
        assert not np.isnan(result[7, 0])

    def test_inactive_gap_breaks_active_window(self, simple_panel):
        """Inactive observations break the reversal lookback window."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[5:7, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][5:7, 0] = np.nan

        result = Reversal(window=3).fit_transform(simple_panel)

        assert not np.isnan(result[4, 0])
        assert np.all(np.isnan(result[5:9, 0]))
        assert not np.isnan(result[9, 0])

    def test_raises_on_infinite_returns(self, simple_panel):
        """Raises ValueError when returns contain infinity."""
        simple_panel["returns"][5, 0] = np.inf
        with pytest.raises(ValueError, match='Field "returns" contains infinite'):
            Reversal(window=5).fit_transform(simple_panel)

    def test_raises_on_returns_less_than_or_equal_to_minus_one(self, simple_panel):
        """Raises ValueError when log return is undefined."""
        simple_panel["returns"][5, 0] = -1.0
        with pytest.raises(ValueError, match="less than or equal to -1"):
            Reversal(window=5).fit_transform(simple_panel)

    def test_formula_matches_reference_loop(self, simple_panel):
        """Reversal matches an explicit reference loop in batch and online modes."""
        window = 4
        returns = simple_panel["returns"]
        returns[4, 0] = np.nan
        returns[6, 1] = np.nan
        active_mask = simple_panel.active_mask.copy()
        active_mask[9, 2] = False
        simple_panel.active_mask = active_mask

        batch = Reversal(window=window).fit_transform(simple_panel)

        descriptor = Reversal(window=window)
        chunked = np.concatenate(
            [
                descriptor.partial_fit_transform(simple_panel[:3]),
                descriptor.partial_fit_transform(simple_panel[3:11]),
                descriptor.partial_fit_transform(simple_panel[11:]),
            ],
            axis=0,
        )

        n_observations, n_assets = returns.shape
        expected = np.full((n_observations, n_assets), np.nan, dtype=float)
        log_returns = np.zeros_like(returns, dtype=float)
        non_missing = ~np.isnan(returns)
        log_returns[non_missing] = np.log1p(returns[non_missing])

        for t in range(n_observations):
            start = t - window + 1
            if start < 0:
                continue

            active_window = active_mask[start : t + 1]
            window_sum = np.sum(log_returns[start : t + 1], axis=0)
            ready = np.all(active_window, axis=0)
            expected[t] = np.where(ready, -window_sum, np.nan)

        expected = np.where(active_mask, expected, np.nan)
        np.testing.assert_allclose(batch, expected, equal_nan=True)
        np.testing.assert_allclose(chunked, expected, equal_nan=True)

    def test_raises_on_invalid_window(self, simple_panel):
        """Raises ValueError when window < 1."""
        with pytest.raises(ValueError, match="window must be a positive integer"):
            Reversal(window=0).fit_transform(simple_panel)

    def test_different_windows_differ(self, simple_panel):
        """Different window sizes produce different outputs."""
        r5 = Reversal(window=5).fit_transform(simple_panel)
        r10 = Reversal(window=10).fit_transform(simple_panel)
        # At observation 14 both should be valid but different
        assert not np.allclose(r5[14], r10[14])

    def test_positive_return_gives_negative_reversal(self, simple_panel):
        """Positive cumulative return yields negative reversal (log space)."""
        # Make all returns positive
        simple_panel["returns"][:] = np.abs(simple_panel["returns"]) + 0.001
        result = Reversal(window=5).fit_transform(simple_panel)
        valid = ~np.isnan(result)
        # All cumulative returns positive → all reversals negative
        assert np.all(result[valid] < 0)

    def test_negative_return_gives_positive_reversal(self, simple_panel):
        """Negative cumulative return yields positive reversal (log space)."""
        # Make all returns negative
        simple_panel["returns"][:] = -np.abs(simple_panel["returns"]) - 0.001
        result = Reversal(window=5).fit_transform(simple_panel)
        valid = ~np.isnan(result)
        assert np.all(result[valid] > 0)

    def test_large_window_all_nan(self, simple_panel):
        """Window larger than n_observations makes all output NaN."""
        result = Reversal(window=100).fit_transform(simple_panel)
        assert np.all(np.isnan(result))
