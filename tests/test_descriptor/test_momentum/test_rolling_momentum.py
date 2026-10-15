"""Tests for RollingMomentum descriptor."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import RollingMomentum


class TestRollingMomentum:
    """Tests for the fixed-window RollingMomentum descriptor."""

    def test_output_shape(self, simple_panel):
        result = RollingMomentum(window=5, skip=2).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_early_observations_are_nan(self, simple_panel):
        window, skip = 5, 2
        result = RollingMomentum(window=window, skip=skip).fit_transform(simple_panel)
        first_valid = skip + window - 1
        assert np.all(np.isnan(result[:first_valid]))
        assert not np.any(np.isnan(result[first_valid]))

    def test_formula_exponentiate_true(self, simple_panel):
        """Output matches exp(sum(log(1+r))) - 1 over the trailing window."""
        window, skip = 5, 2
        result = RollingMomentum(
            window=window, skip=skip, exponentiate=True
        ).fit_transform(simple_panel)
        returns = simple_panel["returns"]
        log_returns = np.log1p(returns)

        first_valid = skip + window - 1
        for t in range(first_valid, returns.shape[0]):
            window_sum = np.sum(
                log_returns[t - skip - window + 1 : t - skip + 1], axis=0
            )
            expected = np.expm1(window_sum)
            np.testing.assert_array_almost_equal(result[t], expected)

    def test_formula_exponentiate_false(self, simple_panel):
        """Output matches sum(log(1+r)) when exponentiate=False."""
        window, skip = 5, 2
        result = RollingMomentum(
            window=window, skip=skip, exponentiate=False
        ).fit_transform(simple_panel)
        returns = simple_panel["returns"]
        log_returns = np.log1p(returns)

        first_valid = skip + window - 1
        for t in range(first_valid, returns.shape[0]):
            expected = np.sum(log_returns[t - skip - window + 1 : t - skip + 1], axis=0)
            np.testing.assert_array_almost_equal(result[t], expected)

    def test_skip_zero(self, simple_panel):
        """With skip=0, window ends at the current observation."""
        window = 5
        result = RollingMomentum(window=window, skip=0).fit_transform(simple_panel)
        returns = simple_panel["returns"]
        log_returns = np.log1p(returns)

        for t in range(window - 1, returns.shape[0]):
            window_sum = np.sum(log_returns[t - window + 1 : t + 1], axis=0)
            np.testing.assert_array_almost_equal(result[t], window_sum)

    def test_window_1_skip_0(self, simple_panel):
        """window=1, skip=0 gives r(t) (single-period return)."""
        result = RollingMomentum(window=1, skip=0, exponentiate=True).fit_transform(
            simple_panel
        )
        expected = simple_panel["returns"]
        np.testing.assert_array_almost_equal(result, expected)

    def test_ranking_preserved_across_exponentiate(self, simple_panel):
        """Cross-sectional ranking is identical regardless of exponentiate."""
        window, skip = 5, 2
        r_exp = RollingMomentum(
            window=window, skip=skip, exponentiate=True
        ).fit_transform(simple_panel)
        r_log = RollingMomentum(
            window=window, skip=skip, exponentiate=False
        ).fit_transform(simple_panel)

        first_valid = skip + window - 1
        for t in range(first_valid, simple_panel["returns"].shape[0]):
            rank_exp = np.argsort(np.argsort(r_exp[t]))
            rank_log = np.argsort(np.argsort(r_log[t]))
            np.testing.assert_array_equal(rank_exp, rank_log)

    def test_partial_fit_matches_fit(self, simple_panel):
        full = RollingMomentum(window=5, skip=2).fit_transform(simple_panel)
        partial = RollingMomentum(window=5, skip=2).partial_fit_transform(simple_panel)
        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        window, skip = 5, 2
        full = RollingMomentum(window=window, skip=skip).fit_transform(simple_panel)

        descriptor = RollingMomentum(window=window, skip=skip)
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_single_obs_chunks(self, simple_panel):
        """Single-observation chunks produce correct results."""
        window, skip = 5, 2
        full = RollingMomentum(window=window, skip=skip).fit_transform(simple_panel)

        descriptor = RollingMomentum(window=window, skip=skip)
        chunks = []
        for start in range(20):
            chunks.append(
                descriptor.partial_fit_transform(simple_panel[start : start + 1])
            )

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_chunked_exponentiate_false(self, simple_panel):
        """Chunked results match with exponentiate=False."""
        window, skip = 5, 2
        full = RollingMomentum(
            window=window, skip=skip, exponentiate=False
        ).fit_transform(simple_panel)

        descriptor = RollingMomentum(window=window, skip=skip, exponentiate=False)
        r1 = descriptor.partial_fit_transform(simple_panel[:4])
        r2 = descriptor.partial_fit_transform(simple_panel[4:])

        combined = np.concatenate([r1, r2], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_chunked_with_inactive_gap(self, simple_panel):
        """Chunked results preserve active-window readiness."""
        window, skip = 4, 1
        active_mask = simple_panel.active_mask.copy()
        active_mask[6:8, 1] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][6:8, 1] = np.nan

        full = RollingMomentum(window=window, skip=skip).fit_transform(simple_panel)

        descriptor = RollingMomentum(window=window, skip=skip)
        r1 = descriptor.partial_fit_transform(simple_panel[:5])
        r2 = descriptor.partial_fit_transform(simple_panel[5:12])
        r3 = descriptor.partial_fit_transform(simple_panel[12:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_allclose(combined, full, equal_nan=True)

    def test_fit_transform_resets_state(self, simple_panel):
        descriptor = RollingMomentum(window=5, skip=2)
        descriptor.partial_fit_transform(simple_panel[:10])

        result = descriptor.fit_transform(simple_panel)
        expected = RollingMomentum(window=5, skip=2).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)

    def test_nan_return_contributes_zero(self, simple_panel):
        """NaN return contributes 0 to the sum (not NaN)."""
        window, skip = 5, 2
        simple_panel["returns"][8, 0] = np.nan
        result = RollingMomentum(window=window, skip=skip).fit_transform(simple_panel)

        t = 10
        log_returns = np.log1p(simple_panel["returns"])
        log_returns[8, 0] = 0.0
        expected = np.sum(log_returns[t - skip - window + 1 : t - skip + 1, 0])
        assert not np.isnan(result[t, 0])
        np.testing.assert_almost_equal(result[t, 0], expected)

    def test_late_listing_needs_full_active_window(self, simple_panel):
        """Late-listed assets need a full active lookback window before output."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = RollingMomentum(window=3, skip=0).fit_transform(simple_panel)

        assert np.all(np.isnan(result[:7, 0]))
        assert not np.isnan(result[7, 0])

    def test_inactive_gap_breaks_active_window(self, simple_panel):
        """Inactive observations break the rolling lookback window."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[5:7, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][5:7, 0] = np.nan

        result = RollingMomentum(window=3, skip=0).fit_transform(simple_panel)

        assert not np.isnan(result[4, 0])
        assert np.all(np.isnan(result[5:9, 0]))
        assert not np.isnan(result[9, 0])

    def test_raises_on_infinite_returns(self, simple_panel):
        """Raises ValueError when returns contain infinity."""
        simple_panel["returns"][5, 0] = np.inf
        with pytest.raises(ValueError, match='Field "returns" contains infinite'):
            RollingMomentum(window=5, skip=2).fit_transform(simple_panel)

    def test_raises_on_returns_less_than_or_equal_to_minus_one(self, simple_panel):
        """Raises ValueError when log return is undefined."""
        simple_panel["returns"][5, 0] = -1.0
        with pytest.raises(ValueError, match="less than or equal to -1"):
            RollingMomentum(window=5, skip=2).fit_transform(simple_panel)

    def test_formula_with_skip_matches_reference_loop(self, simple_panel):
        """Rolling momentum with skip matches an explicit reference loop."""
        window, skip = 4, 2
        returns = simple_panel["returns"]
        returns[4, 0] = np.nan
        returns[6, 1] = np.nan
        active_mask = simple_panel.active_mask.copy()
        active_mask[9, 2] = False
        simple_panel.active_mask = active_mask

        batch = RollingMomentum(
            window=window, skip=skip, exponentiate=True
        ).fit_transform(simple_panel)

        descriptor = RollingMomentum(window=window, skip=skip, exponentiate=True)
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
            start = t - skip - window + 1
            end = t - skip + 1
            if start < 0:
                continue

            active_window = active_mask[start:end]
            window_sum = np.sum(log_returns[start:end], axis=0)
            ready = np.all(active_window, axis=0)
            expected[t] = np.where(ready, np.expm1(window_sum), np.nan)

        expected = np.where(active_mask, expected, np.nan)
        np.testing.assert_allclose(batch, expected, equal_nan=True)
        np.testing.assert_allclose(chunked, expected, equal_nan=True)

    def test_raises_on_invalid_window(self, simple_panel):
        with pytest.raises(ValueError, match="window must be a positive integer"):
            RollingMomentum(window=0).fit_transform(simple_panel)

    def test_raises_on_negative_skip(self, simple_panel):
        with pytest.raises(ValueError, match="skip must be a non-negative integer"):
            RollingMomentum(window=5, skip=-1).fit_transform(simple_panel)

    def test_large_window_all_nan(self, simple_panel):
        result = RollingMomentum(window=100, skip=0).fit_transform(simple_panel)
        assert np.all(np.isnan(result))

    def test_different_windows_differ(self, simple_panel):
        r5 = RollingMomentum(window=5, skip=2).fit_transform(simple_panel)
        r10 = RollingMomentum(window=10, skip=2).fit_transform(simple_panel)
        # Both valid at t=14, should differ
        assert not np.allclose(r5[14], r10[14])

    def test_different_skips_differ(self, simple_panel):
        r0 = RollingMomentum(window=5, skip=0).fit_transform(simple_panel)
        r3 = RollingMomentum(window=5, skip=3).fit_transform(simple_panel)
        # At t=9, both valid, should differ
        assert not np.allclose(r0[9], r3[9])

    def test_first_chunk_smaller_than_first_valid(self, simple_panel):
        """First chunk too small for any output, second chunk completes."""
        window, skip = 5, 2
        full = RollingMomentum(window=window, skip=skip).fit_transform(simple_panel)

        descriptor = RollingMomentum(window=window, skip=skip)
        r1 = descriptor.partial_fit_transform(simple_panel[:3])
        r2 = descriptor.partial_fit_transform(simple_panel[3:])

        assert np.all(np.isnan(r1))
        combined = np.concatenate([r1, r2], axis=0)
        np.testing.assert_array_almost_equal(combined, full)
