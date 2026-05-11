"""Tests for MaxReturn descriptor."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.factor_model.descriptor import MaxReturn


class TestMaxReturn:
    """Tests for the MaxReturn descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = MaxReturn(window=5).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_early_observations_are_nan(self, simple_panel):
        """Observations before window are NaN, window-th is finite."""
        window = 5
        result = MaxReturn(window=window).fit_transform(simple_panel)
        assert np.all(np.isnan(result[: window - 1]))
        assert not np.any(np.isnan(result[window - 1]))

    def test_raises_on_window_one(self, simple_panel):
        """Raises ValueError when window is one."""
        with pytest.raises(ValueError, match="window must be > 1"):
            MaxReturn(window=1).fit_transform(simple_panel)

    def test_rolling_max_formula(self, simple_panel):
        """Output matches manual rolling max computation."""
        window = 5
        result = MaxReturn(window=window).fit_transform(simple_panel)
        returns = simple_panel["returns"]

        for t in range(window - 1, returns.shape[0]):
            expected = np.max(returns[t - window + 1 : t + 1], axis=0)
            np.testing.assert_array_almost_equal(result[t], expected)

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        full = MaxReturn(window=5).fit_transform(simple_panel)
        partial = MaxReturn(window=5).partial_fit_transform(simple_panel)
        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        window = 5
        full = MaxReturn(window=window).fit_transform(simple_panel)

        descriptor = MaxReturn(window=window)
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_single_obs_chunks(self, simple_panel):
        """Single-observation chunks produce correct results."""
        window = 5
        full = MaxReturn(window=window).fit_transform(simple_panel)

        descriptor = MaxReturn(window=window)
        chunks = []
        for start in range(20):
            chunks.append(
                descriptor.partial_fit_transform(simple_panel[start : start + 1])
            )

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        descriptor = MaxReturn(window=5)
        descriptor.partial_fit_transform(simple_panel[:10])

        result = descriptor.fit_transform(simple_panel)
        expected = MaxReturn(window=5).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)

    def test_nan_return_ignored(self, simple_panel):
        """NaN return is ignored in the max (treated as -inf)."""
        simple_panel["returns"][8, 0] = np.nan
        result = MaxReturn(window=5).fit_transform(simple_panel)
        # Output should still be finite (NaN skipped)
        assert not np.isnan(result[9, 0])

    def test_all_nan_window_produces_nan(self, simple_panel):
        """All-missing windows produce NaN, not -inf."""
        simple_panel["returns"][:5, 0] = np.nan
        result = MaxReturn(window=5).fit_transform(simple_panel)
        assert np.isnan(result[4, 0])
        assert not np.isneginf(result[4, 0])

    def test_late_listing_needs_full_active_window(self, simple_panel):
        """Late-listed assets need a full active window before output."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = MaxReturn(window=3).fit_transform(simple_panel)

        assert np.all(np.isnan(result[:7, 0]))
        assert not np.isnan(result[7, 0])

    def test_delisting_breaks_active_window(self, simple_panel):
        """Inactive observations break the trailing active window."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[5:7, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][5:7, 0] = np.nan

        result = MaxReturn(window=3).fit_transform(simple_panel)

        assert not np.isnan(result[4, 0])
        assert np.all(np.isnan(result[5:9, 0]))
        assert not np.isnan(result[9, 0])

    def test_raises_on_infinite_returns(self, simple_panel):
        """Raises ValueError when returns contain infinity."""
        simple_panel["returns"][5, 0] = np.inf
        with pytest.raises(ValueError, match='Field "returns" contains infinite'):
            MaxReturn(window=5).fit_transform(simple_panel)

    def test_raises_on_invalid_window(self, simple_panel):
        """Raises ValueError when window is less than one."""
        with pytest.raises(ValueError, match="window must be > 1"):
            MaxReturn(window=0).fit_transform(simple_panel)

    def test_different_windows_differ(self, simple_panel):
        """Different window sizes produce different outputs."""
        r5 = MaxReturn(window=5).fit_transform(simple_panel)
        r10 = MaxReturn(window=10).fit_transform(simple_panel)
        assert not np.allclose(r5[14], r10[14])

    def test_max_geq_last_return(self, simple_panel):
        """Rolling max is always >= the current return."""
        window = 5
        result = MaxReturn(window=window).fit_transform(simple_panel)
        returns = simple_panel["returns"]
        for t in range(window - 1, returns.shape[0]):
            assert np.all(result[t] >= returns[t] - 1e-12)

    def test_large_window_all_nan(self, simple_panel):
        """Window larger than n_observations makes all output NaN."""
        result = MaxReturn(window=100).fit_transform(simple_panel)
        assert np.all(np.isnan(result))
