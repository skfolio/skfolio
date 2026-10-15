"""Tests for EWDownsideBeta descriptor."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.descriptor import EWDownsideBeta
from skfolio.utils.stats import _market_returns
from skfolio.utils.tools import half_life_to_decay_factor


class TestEWDownsideBeta:
    """Tests for EWDownsideBeta descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = EWDownsideBeta(half_life=5, min_periods=1).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_early_observations_are_nan(self, simple_panel):
        """Observations before min_periods are NaN."""
        min_periods = 8
        result = EWDownsideBeta(half_life=5, min_periods=min_periods).fit_transform(
            simple_panel
        )
        assert np.all(np.isnan(result[: min_periods - 1]))
        assert not np.all(np.isnan(result[min_periods - 1]))

    def test_late_listing_needs_asset_min_periods(self, simple_panel):
        """Late-listed assets need min_periods valid returns before output."""
        active_mask = simple_panel.active_mask.copy()
        estimation_mask = simple_panel.estimation_mask.copy()
        active_mask[:5, 0] = False
        estimation_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel.estimation_mask = estimation_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = EWDownsideBeta(half_life=5, min_periods=3).fit_transform(simple_panel)

        assert np.all(np.isnan(result[:7, 0]))
        assert not np.isnan(result[7, 0])

    def test_default_min_periods(self):
        """Default min_periods is None (resolved to half_life internally)."""
        d = EWDownsideBeta(half_life=10)
        assert d.min_periods is None

    def test_lpm_formula(self, simple_panel):
        """LPM co-moment beta matches manual computation."""
        half_life = 5
        decay = half_life_to_decay_factor(half_life)
        eps = 1e-12
        mar = 0.0

        result = EWDownsideBeta(
            half_life=half_life,
            min_acceptable_return=mar,
            min_periods=1,
            eps=eps,
        ).fit_transform(simple_panel)

        returns = simple_panel["returns"]
        market_rets = _market_returns(
            asset_returns=returns,
            weights=simple_panel["market_cap"],
            estimation_mask=simple_panel.estimation_mask,
        )
        n_obs, n_assets = returns.shape

        var_down_market = 0.0
        cov_down = np.zeros(n_assets)

        for t in range(n_obs):
            r_m = market_rets[t]
            r_i = returns[t]

            d_m = min(r_m - mar, 0.0)
            d_i = np.minimum(r_i - mar, 0.0)

            var_down_market = decay * var_down_market + (1 - decay) * d_m * d_m
            cov_down = decay * cov_down + (1 - decay) * d_i * d_m

            expected = cov_down / (var_down_market + eps)
            np.testing.assert_array_almost_equal(result[t], expected)

    def test_all_positive_returns_gives_zero_beta(self, simple_panel):
        """When all returns are above threshold, downside beta is ~0."""
        # Make all returns positive
        simple_panel["returns"][:] = np.abs(simple_panel["returns"]) + 0.01

        result = EWDownsideBeta(
            half_life=5, min_acceptable_return=0.0, min_periods=1
        ).fit_transform(simple_panel)

        # All D_i and D_m are zero → cov_down and var_down_market are zero
        # beta = 0 / (0 + eps) ≈ 0
        valid = ~np.isnan(result)
        np.testing.assert_array_almost_equal(result[valid], 0.0)

    def test_downside_beta_geq_zero(self, simple_panel):
        """Downside beta is non-negative (D_i and D_m have same sign)."""
        result = EWDownsideBeta(half_life=5, min_periods=1).fit_transform(simple_panel)

        valid = ~np.isnan(result)
        assert np.all(result[valid] >= -1e-10)

    def test_negative_threshold(self, simple_panel):
        """Negative threshold restricts downside to larger losses."""
        kwargs = dict(half_life=5, min_periods=1)

        result_zero = EWDownsideBeta(min_acceptable_return=0.0, **kwargs).fit_transform(
            simple_panel
        )
        result_neg = EWDownsideBeta(
            min_acceptable_return=-0.05, **kwargs
        ).fit_transform(simple_panel)

        # With a more negative threshold, fewer observations trigger downside,
        # so the var_down_market and cov_down accumulators are smaller.
        # The beta values will differ.
        valid = ~np.isnan(result_zero) & ~np.isnan(result_neg)
        assert not np.allclose(result_zero[valid], result_neg[valid])

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        full = EWDownsideBeta(half_life=5, min_periods=2).fit_transform(simple_panel)

        partial = EWDownsideBeta(half_life=5, min_periods=2).partial_fit_transform(
            simple_panel
        )

        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        kwargs = dict(half_life=5, min_periods=3)

        full = EWDownsideBeta(**kwargs).fit_transform(simple_panel)

        descriptor = EWDownsideBeta(**kwargs)
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_small_chunks(self, simple_panel):
        """Single-observation chunks produce correct results."""
        kwargs = dict(half_life=5, min_periods=2)

        full = EWDownsideBeta(**kwargs).fit_transform(simple_panel)

        descriptor = EWDownsideBeta(**kwargs)
        chunks = []
        for start in range(20):
            chunks.append(
                descriptor.partial_fit_transform(simple_panel[start : start + 1])
            )

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        descriptor = EWDownsideBeta(half_life=5, min_periods=2)
        descriptor.partial_fit_transform(simple_panel[:10])

        result = descriptor.fit_transform(simple_panel)
        expected = EWDownsideBeta(half_life=5, min_periods=2).fit_transform(
            simple_panel
        )
        np.testing.assert_array_equal(result, expected)

    def test_nan_return_holds_cov_state(self, simple_panel):
        """NaN return holds co-moment state for that asset.

        The beta itself may change because the market downside variance
        (denominator) updates from other assets. This verifies that the
        asset's co-moment numerator is held, not that the beta is identical.
        """
        simple_panel["returns"][5, 0] = np.nan
        result = EWDownsideBeta(half_life=5, min_periods=1).fit_transform(simple_panel)
        # Output should still be a valid number (not NaN) even though
        # the asset's return was NaN -- state is held, not zeroed.
        assert not np.isnan(result[5, 0])

    def test_uses_estimation_mask_for_market_returns(self, simple_panel):
        """Market return is computed on the estimation universe."""
        simple_panel["returns"][:, 0] = -0.30
        simple_panel["market_cap"][:, 0] = 1e12
        mask = np.ones_like(simple_panel["returns"], dtype=bool)
        mask[:, 0] = False
        simple_panel.estimation_mask = mask

        result = EWDownsideBeta(half_life=5, min_periods=1).fit_transform(simple_panel)

        returns = simple_panel["returns"]
        market_rets = _market_returns(
            asset_returns=returns,
            weights=simple_panel["market_cap"],
            estimation_mask=mask,
        )
        decay = half_life_to_decay_factor(5)
        var_down_market = 0.0
        cov_down = np.zeros(returns.shape[1])

        for t in range(returns.shape[0]):
            d_m = min(market_rets[t], 0.0)
            d_i = np.minimum(returns[t], 0.0)
            var_down_market = decay * var_down_market + (1 - decay) * d_m * d_m
            cov_down = decay * cov_down + (1 - decay) * d_i * d_m

            expected = cov_down / (var_down_market + 1e-12)
            np.testing.assert_array_almost_equal(result[t], expected)

    def test_raises_on_undefined_market_return(self, simple_panel):
        """Raises when no estimable asset can define the market return."""
        simple_panel["returns"][5] = np.nan
        with pytest.raises(ValueError, match="Market return is undefined"):
            EWDownsideBeta(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_raises_on_invalid_half_life(self, simple_panel):
        """Raises ValueError when half_life <= 0."""
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            EWDownsideBeta(half_life=0).fit_transform(simple_panel)

    def test_raises_on_invalid_min_periods(self, simple_panel):
        """Raises ValueError when min_periods < 1."""
        with pytest.raises(ValueError, match="min_periods must be a positive integer"):
            EWDownsideBeta(half_life=5, min_periods=0).fit_transform(simple_panel)

    def test_raises_on_invalid_eps(self, simple_panel):
        """Raises ValueError when eps <= 0."""
        with pytest.raises(ValueError, match="eps must be a positive number"):
            EWDownsideBeta(half_life=5, eps=0).fit_transform(simple_panel)

    def test_fractional_half_life_default_min_periods_at_least_one(self, simple_panel):
        """Default min_periods is at least one for fractional half-lives."""
        result = EWDownsideBeta(half_life=0.5).fit_transform(simple_panel)
        assert not np.any(np.isnan(result[0]))

    def test_higher_half_life_smoother(self, simple_panel):
        """Higher half_life produces smoother output."""
        result_fast = EWDownsideBeta(half_life=2, min_periods=1).fit_transform(
            simple_panel
        )
        result_slow = EWDownsideBeta(half_life=10, min_periods=1).fit_transform(
            simple_panel
        )
        valid_fast = result_fast[~np.isnan(result_fast)]
        valid_slow = result_slow[~np.isnan(result_slow)]
        assert np.var(valid_slow) < np.var(valid_fast)
