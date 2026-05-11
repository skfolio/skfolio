"""Tests for EWMomentum descriptor."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.factor_model.descriptor import EWMomentum
from skfolio.utils.tools import half_life_to_decay_factor


class TestEWMomentum:
    """Tests for EWMomentum descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches input."""
        result = EWMomentum(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_early_observations_are_nan(self, simple_panel):
        """Observations before skip + min_periods are NaN."""
        skip = 3
        min_periods = 4
        result = EWMomentum(
            half_life=5, skip=skip, min_periods=min_periods
        ).fit_transform(simple_panel)
        # First skip + min_periods - 1 rows should be NaN
        # (skip to fill delay buffer, then min_periods EWMA updates)
        assert np.all(np.isnan(result[: skip + min_periods - 1]))
        # Row at skip + min_periods - 1 should be valid
        assert not np.all(np.isnan(result[skip + min_periods - 1]))

    def test_default_half_life(self):
        """Default half_life is 87 (~1-year rolling window equivalent)."""
        m = EWMomentum()
        assert m.half_life == 87

    def test_default_min_periods(self):
        """Default min_periods is resolved during initialization."""
        m = EWMomentum(half_life=10)
        assert m.min_periods is None

    def test_fractional_half_life_default_min_periods(self, simple_panel):
        """Default min_periods is ceil(half_life), with minimum one."""
        descriptor = EWMomentum(half_life=1.2, skip=0)
        result = descriptor.fit_transform(simple_panel)
        assert descriptor.min_periods_ == 2
        assert np.all(np.isnan(result[0]))
        assert not np.all(np.isnan(result[1]))

        descriptor = EWMomentum(half_life=0.5, skip=0)
        descriptor.fit_transform(simple_panel)
        assert descriptor.min_periods_ == 1

    def test_no_skip_immediate_ewma(self, simple_panel):
        """With skip=0, EWMA starts from first observation."""
        min_periods = 2
        result = EWMomentum(half_life=5, skip=0, min_periods=min_periods).fit_transform(
            simple_panel
        )
        # First min_periods - 1 rows are NaN
        assert np.all(np.isnan(result[: min_periods - 1]))
        # Row at min_periods - 1 should be valid
        assert not np.all(np.isnan(result[min_periods - 1]))

    def test_ewma_formula_no_skip(self, simple_panel):
        """EWMA values match manual computation (skip=0)."""
        half_life = 5
        decay = half_life_to_decay_factor(half_life)
        min_periods = 1

        result = EWMomentum(
            half_life=half_life, skip=0, min_periods=min_periods
        ).fit_transform(simple_panel)

        returns = simple_panel["returns"]
        log_returns = np.log1p(returns)

        # Manually compute EWMA for first asset
        ewma = 0.0
        for t in range(returns.shape[0]):
            ewma = decay * ewma + (1 - decay) * log_returns[t, 0]
            np.testing.assert_almost_equal(result[t, 0], ewma)

    def test_ewma_formula_with_skip(self, simple_panel):
        """EWMA values match manual computation (skip>0)."""
        half_life = 5
        skip = 3
        decay = half_life_to_decay_factor(half_life)
        min_periods = 1

        result = EWMomentum(
            half_life=half_life, skip=skip, min_periods=min_periods
        ).fit_transform(simple_panel)

        returns = simple_panel["returns"]
        log_returns = np.log1p(returns)

        # Manually compute EWMA for first asset with delay
        ewma = 0.0
        for t in range(returns.shape[0]):
            if t < skip:
                assert np.isnan(result[t, 0])
            else:
                delayed = log_returns[t - skip, 0]
                ewma = decay * ewma + (1 - decay) * delayed
                np.testing.assert_almost_equal(result[t, 0], ewma)

    def test_ewma_formula_with_skip_matches_reference_loop(self, simple_panel):
        """EWMA with skip matches an explicit reference loop for all assets."""
        half_life = 5
        skip = 3
        min_periods = 2
        decay = half_life_to_decay_factor(half_life)

        returns = simple_panel["returns"]
        returns[4, 0] = np.nan
        returns[6, 1] = np.nan
        active_mask = simple_panel.active_mask.copy()
        active_mask[9, 2] = False
        simple_panel.active_mask = active_mask

        result = EWMomentum(
            half_life=half_life,
            skip=skip,
            min_periods=min_periods,
            exponentiate=True,
        ).fit_transform(simple_panel)

        n_observations, n_assets = returns.shape
        ewma = np.zeros(n_assets, dtype=float)
        n_valid = np.zeros(n_assets, dtype=int)
        expected = np.full((n_observations, n_assets), np.nan, dtype=float)

        for t in range(n_observations):
            source_t = t - skip
            if source_t >= 0:
                delayed = np.log1p(returns[source_t])
                valid = np.isfinite(delayed)
                if np.any(valid):
                    n_valid[valid] += 1
                    ewma[valid] = decay * ewma[valid] + (1 - decay) * delayed[valid]

            ready = n_valid >= min_periods
            expected[t] = np.where(ready, np.expm1(ewma), np.nan)

        expected = np.where(active_mask, expected, np.nan)
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        descriptor = EWMomentum(half_life=5, skip=2, min_periods=3)
        full = descriptor.fit_transform(simple_panel)

        descriptor2 = EWMomentum(half_life=5, skip=2, min_periods=3)
        partial = descriptor2.partial_fit_transform(simple_panel)

        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        half_life = 5
        skip = 2
        min_periods = 3

        full = EWMomentum(
            half_life=half_life, skip=skip, min_periods=min_periods
        ).fit_transform(simple_panel)

        descriptor = EWMomentum(half_life=half_life, skip=skip, min_periods=min_periods)
        chunk1 = simple_panel[:7]
        chunk2 = simple_panel[7:13]
        chunk3 = simple_panel[13:]

        r1 = descriptor.partial_fit_transform(chunk1)
        r2 = descriptor.partial_fit_transform(chunk2)
        r3 = descriptor.partial_fit_transform(chunk3)

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_small_chunks(self, simple_panel):
        """Chunks smaller than skip still produce correct results."""
        half_life = 5
        skip = 4
        min_periods = 2

        full = EWMomentum(
            half_life=half_life, skip=skip, min_periods=min_periods
        ).fit_transform(simple_panel)

        descriptor = EWMomentum(half_life=half_life, skip=skip, min_periods=min_periods)
        chunks = []
        for start in range(0, 20, 2):
            view = simple_panel[start : start + 2]
            chunks.append(descriptor.partial_fit_transform(view))

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        descriptor = EWMomentum(half_life=5, skip=2, min_periods=3)

        # First run: partial with a chunk
        descriptor.partial_fit_transform(simple_panel[:10])

        # Second run: fit_transform should reset and produce clean result
        result = descriptor.fit_transform(simple_panel)
        expected = EWMomentum(half_life=5, skip=2, min_periods=3).fit_transform(
            simple_panel
        )
        np.testing.assert_array_equal(result, expected)

    def test_nan_return_holds_ewma(self, simple_panel):
        """NaN return holds EWMA state for that asset."""
        simple_panel["returns"][5, 0] = np.nan
        result = EWMomentum(half_life=5, skip=0, min_periods=1).fit_transform(
            simple_panel
        )
        # EWMA should be held: result at t=5 equals result at t=4 for asset 0
        np.testing.assert_almost_equal(result[5, 0], result[4, 0])

    def test_nan_return_does_not_increment_valid_count(self, simple_panel):
        """NaN returns hold state and do not count toward min_periods."""
        simple_panel["returns"][1, 0] = np.nan
        result = EWMomentum(half_life=5, skip=0, min_periods=2).fit_transform(
            simple_panel
        )

        assert np.isnan(result[1, 0])
        assert not np.isnan(result[2, 0])

    def test_late_listing_needs_asset_min_periods(self, simple_panel):
        """Late-listed assets need min_periods valid delayed returns before output."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = EWMomentum(half_life=5, skip=0, min_periods=3).fit_transform(
            simple_panel
        )

        assert np.all(np.isnan(result[:7, 0]))
        assert not np.isnan(result[7, 0])

    def test_raises_on_infinite_returns(self, simple_panel):
        """Raises ValueError when returns contain infinity."""
        simple_panel["returns"][5, 0] = np.inf
        with pytest.raises(ValueError, match='Field "returns" contains infinite'):
            EWMomentum(half_life=5).fit_transform(simple_panel)

    def test_raises_on_returns_less_than_or_equal_to_minus_one(self, simple_panel):
        """Raises ValueError when log return is undefined."""
        simple_panel["returns"][5, 0] = -1.0
        with pytest.raises(ValueError, match="less than or equal to -1"):
            EWMomentum(half_life=5).fit_transform(simple_panel)

    def test_raises_on_invalid_half_life(self, simple_panel):
        """Raises ValueError when half_life <= 0."""
        with pytest.raises(ValueError, match="half_life must be positive"):
            EWMomentum(half_life=0).fit_transform(simple_panel)

    def test_raises_on_negative_skip(self, simple_panel):
        """Raises ValueError when skip < 0."""
        with pytest.raises(ValueError, match="skip must be non-negative"):
            EWMomentum(half_life=5, skip=-1).fit_transform(simple_panel)

    def test_raises_on_invalid_min_periods(self, simple_panel):
        """Raises ValueError when min_periods < 1."""
        with pytest.raises(ValueError, match="min_periods must be >= 1"):
            EWMomentum(half_life=5, min_periods=0).fit_transform(simple_panel)

    def test_higher_half_life_smoother(self, simple_panel):
        """Higher half_life produces smoother (less volatile) output."""
        result_fast = EWMomentum(half_life=2, skip=0, min_periods=1).fit_transform(
            simple_panel
        )
        result_slow = EWMomentum(half_life=10, skip=0, min_periods=1).fit_transform(
            simple_panel
        )
        # Variance of slow EWMA should be lower (smoother)
        valid_fast = result_fast[~np.isnan(result_fast)]
        valid_slow = result_slow[~np.isnan(result_slow)]
        assert np.var(valid_slow) < np.var(valid_fast)
