"""Tests for EWVolatility and EWDownsideVolatility descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import (
    EWDownsideVolatility,
    EWVolatility,
)
from skfolio.utils.tools import half_life_to_decay_factor


class TestEWVolatility:
    """Tests for EWVolatility descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = EWVolatility(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_early_observations_are_nan(self, simple_panel):
        """Observations before min_periods are NaN."""
        min_periods = 8
        result = EWVolatility(half_life=5, min_periods=min_periods).fit_transform(
            simple_panel
        )
        assert np.all(np.isnan(result[: min_periods - 1]))
        assert not np.all(np.isnan(result[min_periods - 1]))

    def test_default_min_periods_uses_half_life(self, simple_panel):
        """Default min_periods resolves to ceil(half_life)."""
        d = EWVolatility(half_life=8.2)
        assert d.min_periods is None
        result = d.fit_transform(simple_panel)
        assert np.all(np.isnan(result[:8]))
        assert not np.all(np.isnan(result[8]))

    def test_output_is_non_negative(self, simple_panel):
        """Volatility output is always non-negative."""
        result = EWVolatility(half_life=5, min_periods=1).fit_transform(simple_panel)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)

    def test_ewma_formula(self, simple_panel):
        """EWMA volatility matches manual computation."""
        half_life = 5
        decay = half_life_to_decay_factor(half_life)
        min_periods = 1

        result = EWVolatility(
            half_life=half_life, min_periods=min_periods
        ).fit_transform(simple_panel)

        returns = simple_panel["returns"]
        n_obs, n_assets = returns.shape
        var = np.zeros(n_assets)
        n_valid = np.zeros(n_assets, dtype=int)

        for t in range(n_obs):
            var = decay * var + (1 - decay) * returns[t] ** 2
            n_valid += 1
            corrected_var = var / (1 - decay**n_valid)
            np.testing.assert_array_almost_equal(result[t], np.sqrt(corrected_var))

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        full = EWVolatility(half_life=5, min_periods=2).fit_transform(simple_panel)
        partial = EWVolatility(half_life=5, min_periods=2).partial_fit_transform(
            simple_panel
        )
        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        kwargs = dict(half_life=5, min_periods=3)

        full = EWVolatility(**kwargs).fit_transform(simple_panel)

        descriptor = EWVolatility(**kwargs)
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_small_chunks(self, simple_panel):
        """Single-observation chunks produce correct results."""
        kwargs = dict(half_life=5, min_periods=2)

        full = EWVolatility(**kwargs).fit_transform(simple_panel)

        descriptor = EWVolatility(**kwargs)
        chunks = []
        for start in range(20):
            chunks.append(
                descriptor.partial_fit_transform(simple_panel[start : start + 1])
            )

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        descriptor = EWVolatility(half_life=5, min_periods=2)
        descriptor.partial_fit_transform(simple_panel[:10])

        result = descriptor.fit_transform(simple_panel)
        expected = EWVolatility(half_life=5, min_periods=2).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)

    def test_nan_return_holds_state(self, simple_panel):
        """NaN return freezes EWMA state for that asset."""
        simple_panel["returns"][5, 0] = np.nan
        result = EWVolatility(half_life=5, min_periods=1).fit_transform(simple_panel)
        np.testing.assert_almost_equal(result[5, 0], result[4, 0])

    def test_nan_return_does_not_increment_asset_min_periods(self, simple_panel):
        """Missing asset returns do not count toward per-asset readiness."""
        simple_panel["returns"][:2, 0] = np.nan
        result = EWVolatility(half_life=5, min_periods=3).fit_transform(simple_panel)
        assert np.isnan(result[2, 0])
        assert np.isnan(result[3, 0])
        assert not np.isnan(result[4, 0])

    def test_late_listing_needs_asset_min_periods(self, simple_panel):
        """Late-listed assets need min_periods valid returns before output."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = EWVolatility(half_life=5, min_periods=3).fit_transform(simple_panel)

        assert np.isnan(result[5, 0])
        assert np.isnan(result[6, 0])
        assert not np.isnan(result[7, 0])

    def test_inactive_gap_resets_warmup(self, simple_panel):
        """Inactive observations reset state and valid observation count."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[5:7, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][5:7, 0] = np.nan

        result = EWVolatility(half_life=5, min_periods=3).fit_transform(simple_panel)

        assert np.isnan(result[5, 0])
        assert np.isnan(result[6, 0])
        assert np.isnan(result[7, 0])
        assert np.isnan(result[8, 0])
        assert not np.isnan(result[9, 0])

    def test_active_mask_delistings(self, simple_panel):
        """Out-of-universe assets produce NaN output."""
        mask = np.ones_like(simple_panel["returns"], dtype=bool)
        mask[10:, 2] = False
        simple_panel.active_mask = mask

        result = EWVolatility(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert np.all(np.isnan(result[10:, 2]))
        assert not np.any(np.isnan(result[:10, 2]))

    def test_raises_on_invalid_half_life(self, simple_panel):
        """Raises ValueError when half_life <= 0."""
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            EWVolatility(half_life=0).fit_transform(simple_panel)

    def test_raises_on_invalid_min_periods(self, simple_panel):
        """Raises ValueError when min_periods < 1."""
        with pytest.raises(ValueError, match="min_periods must be a positive integer"):
            EWVolatility(half_life=5, min_periods=0).fit_transform(simple_panel)

    def test_raises_on_infinite_returns(self, simple_panel):
        simple_panel["returns"][0, 0] = np.inf
        with pytest.raises(
            ValueError, match='Field "returns" contains infinite values'
        ):
            EWVolatility(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_higher_half_life_smoother(self, simple_panel):
        """Higher half_life produces smoother (less volatile) output."""
        result_fast = EWVolatility(half_life=2, min_periods=1).fit_transform(
            simple_panel
        )
        result_slow = EWVolatility(half_life=10, min_periods=1).fit_transform(
            simple_panel
        )
        valid_fast = result_fast[~np.isnan(result_fast)]
        valid_slow = result_slow[~np.isnan(result_slow)]
        assert np.var(valid_slow) < np.var(valid_fast)

    def test_no_min_acceptable_return_param(self):
        """EWVolatility does not expose min_acceptable_return."""
        d = EWVolatility()
        params = d.get_params()
        assert "min_acceptable_return" not in params

    def test_volatility_attribute_set(self, simple_panel):
        """The fitted `volatility_` attribute stores the last row."""
        descriptor = EWVolatility(half_life=5, min_periods=1)
        result = descriptor.fit_transform(simple_panel)
        np.testing.assert_array_equal(descriptor.volatility_, result[-1])


class TestEWDownsideVolatility:
    """Tests for EWDownsideVolatility descriptor."""

    def test_downside_formula(self, simple_panel):
        """Downside volatility matches manual computation."""
        half_life = 5
        decay = half_life_to_decay_factor(half_life)
        mar = 0.0

        result = EWDownsideVolatility(
            half_life=half_life, min_acceptable_return=mar, min_periods=1
        ).fit_transform(simple_panel)

        returns = simple_panel["returns"]
        n_obs, n_assets = returns.shape
        var = np.zeros(n_assets)
        n_valid = np.zeros(n_assets, dtype=int)

        for t in range(n_obs):
            downside = np.minimum(returns[t] - mar, 0.0)
            var = decay * var + (1 - decay) * downside**2
            n_valid += 1
            corrected_var = var / (1 - decay**n_valid)
            np.testing.assert_array_almost_equal(result[t], np.sqrt(corrected_var))

    def test_downside_leq_total(self, simple_panel):
        """Downside volatility is always <= total volatility."""
        kwargs = dict(half_life=5, min_periods=1)

        total = EWVolatility(**kwargs).fit_transform(simple_panel)
        downside = EWDownsideVolatility(**kwargs).fit_transform(simple_panel)

        valid = ~np.isnan(total) & ~np.isnan(downside)
        assert np.all(downside[valid] <= total[valid] + 1e-12)

    def test_downside_chunked_matches_full(self, simple_panel):
        """Chunked downside processing matches full fit_transform."""
        kwargs = dict(half_life=5, min_periods=2)

        full = EWDownsideVolatility(**kwargs).fit_transform(simple_panel)

        descriptor = EWDownsideVolatility(**kwargs)
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_nonzero_threshold(self, simple_panel):
        """Higher threshold captures more downside events -> higher vol."""
        kwargs = dict(half_life=5, min_periods=1)

        result_zero = EWDownsideVolatility(
            min_acceptable_return=0.0, **kwargs
        ).fit_transform(simple_panel)
        result_pos = EWDownsideVolatility(
            min_acceptable_return=0.01, **kwargs
        ).fit_transform(simple_panel)

        valid = ~np.isnan(result_zero) & ~np.isnan(result_pos)
        assert np.all(result_pos[valid] >= result_zero[valid] - 1e-12)

    def test_exposes_min_acceptable_return_param(self):
        """EWDownsideVolatility exposes min_acceptable_return."""
        d = EWDownsideVolatility()
        params = d.get_params()
        assert "min_acceptable_return" in params
        assert params["min_acceptable_return"] == 0.0

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = EWDownsideVolatility(half_life=5, min_periods=1).fit_transform(
            simple_panel
        )
        assert result.shape == simple_panel["returns"].shape

    def test_output_is_non_negative(self, simple_panel):
        """Downside volatility output is always non-negative."""
        result = EWDownsideVolatility(half_life=5, min_periods=1).fit_transform(
            simple_panel
        )
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)

    def test_nan_return_holds_state(self, simple_panel):
        """NaN return freezes EWMA state for that asset."""
        simple_panel["returns"][5, 0] = np.nan
        result = EWDownsideVolatility(half_life=5, min_periods=1).fit_transform(
            simple_panel
        )
        np.testing.assert_almost_equal(result[5, 0], result[4, 0])
