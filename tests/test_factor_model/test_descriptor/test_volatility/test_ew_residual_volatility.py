"""Tests for EWResidualVolatility and EWResidualDownsideVolatility descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.factor_model._utils import _market_returns
from skfolio.factor_model.descriptor import (
    EWResidualDownsideVolatility,
    EWResidualVolatility,
    EWVolatility,
)
from skfolio.utils.tools import half_life_to_decay_factor


def _manual_residual_volatility(
    panel,
    half_life,
    beta_half_life,
    min_periods,
    eps=1e-12,
    min_acceptable_return=None,
):
    """Reference EWMA residual volatility calculation."""
    asset_returns = panel["returns"]
    market_caps = panel["market_cap"]
    market_returns = _market_returns(
        asset_returns=asset_returns,
        weights=market_caps,
        estimation_mask=panel.estimation_mask,
    )
    n_observations, n_assets = asset_returns.shape

    vol_decay = half_life_to_decay_factor(half_life)
    beta_decay = half_life_to_decay_factor(beta_half_life)

    mu_market = 0.0
    var_market = 0.0
    mu_assets = np.zeros(n_assets)
    cov_assets = np.zeros(n_assets)
    var_residual = np.zeros(n_assets)
    n_valid_assets = np.zeros(n_assets, dtype=int)
    is_active = np.ones(n_assets, dtype=bool)
    result = np.full((n_observations, n_assets), np.nan)

    for t in range(n_observations):
        returns_t = asset_returns[t]
        market_return_t = market_returns[t]
        active_assets = panel.active_mask[t]
        valid_returns = active_assets & np.isfinite(returns_t)

        newly_inactive = is_active & ~active_assets
        if np.any(newly_inactive):
            mu_assets[newly_inactive] = 0.0
            cov_assets[newly_inactive] = 0.0
            var_residual[newly_inactive] = 0.0
            n_valid_assets[newly_inactive] = 0
        is_active[:] = active_assets

        market_deviation = market_return_t - mu_market
        asset_deviations = returns_t[valid_returns] - mu_assets[valid_returns]

        mu_market = beta_decay * mu_market + (1 - beta_decay) * market_return_t
        mu_assets[valid_returns] = (
            beta_decay * mu_assets[valid_returns]
            + (1 - beta_decay) * returns_t[valid_returns]
        )

        var_market = (
            beta_decay * var_market
            + (1 - beta_decay) * market_deviation * market_deviation
        )
        cov_assets[valid_returns] = (
            beta_decay * cov_assets[valid_returns]
            + (1 - beta_decay) * asset_deviations * market_deviation
        )

        beta = cov_assets / (var_market + eps)
        residual = returns_t - beta * market_return_t
        if min_acceptable_return is None:
            contribution = residual
        else:
            contribution = np.minimum(residual - min_acceptable_return, 0.0)

        n_valid_assets[valid_returns] += 1
        var_residual[valid_returns] = (
            vol_decay * var_residual[valid_returns]
            + (1 - vol_decay) * contribution[valid_returns] ** 2
        )

        ready = n_valid_assets >= min_periods
        if np.any(ready):
            weight_sum = 1.0 - vol_decay ** n_valid_assets[ready]
            result[t, ready] = np.sqrt(var_residual[ready] / weight_sum)

    return result


class TestEWResidualVolatility:
    """Tests for EWResidualVolatility descriptor."""

    def test_output_shape(self, simple_panel):
        """Output shape matches (n_observations, n_assets)."""
        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=1
        ).fit_transform(simple_panel)
        assert result.shape == simple_panel["returns"].shape

    def test_early_observations_are_nan(self, simple_panel):
        """Observations before min_periods are NaN."""
        min_periods = 8
        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=min_periods
        ).fit_transform(simple_panel)
        assert np.all(np.isnan(result[: min_periods - 1]))
        assert not np.all(np.isnan(result[min_periods - 1]))

    def test_default_min_periods_uses_slower_half_life(self, simple_panel):
        """Default min_periods resolves to ceil of the slower half-life."""
        d = EWResidualVolatility(half_life=8.2, beta_half_life=5)
        assert d.min_periods is None
        result = d.fit_transform(simple_panel)
        assert np.all(np.isnan(result[:8]))
        assert not np.all(np.isnan(result[8]))

    def test_output_is_non_negative(self, simple_panel):
        """Volatility output is always non-negative."""
        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=1
        ).fit_transform(simple_panel)
        valid = ~np.isnan(result)
        assert np.all(result[valid] >= 0)

    def test_residual_vol_formula(self, simple_panel):
        """EWMA residual vol matches manual computation with separate decays."""
        half_life = 5
        beta_half_life = 8
        eps = 1e-12
        min_periods = 1

        result = EWResidualVolatility(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=min_periods,
            eps=eps,
        ).fit_transform(simple_panel)

        expected = _manual_residual_volatility(
            simple_panel,
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=min_periods,
            eps=eps,
        )
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_same_half_life_matches_single_decay(self, simple_panel):
        """When half_life == beta_half_life, result uses a single decay."""
        hl = 7
        result = EWResidualVolatility(
            half_life=hl, beta_half_life=hl, min_periods=1
        ).fit_transform(simple_panel)

        expected = _manual_residual_volatility(
            simple_panel,
            half_life=hl,
            beta_half_life=hl,
            min_periods=1,
        )
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_residual_vol_less_than_total_vol(self, simple_panel):
        """CAPM residual vol should generally be less than total vol."""
        half_life = 5
        min_periods = 1

        result = EWResidualVolatility(
            half_life=half_life, beta_half_life=half_life, min_periods=min_periods
        ).fit_transform(simple_panel)

        total_vol_series = EWVolatility(
            half_life=half_life, min_periods=min_periods
        ).fit_transform(simple_panel)

        late = result[10:]
        late_total = total_vol_series[10:]
        frac = np.mean(late <= late_total + 1e-10)
        assert frac > 0.7

    def test_partial_fit_matches_fit(self, simple_panel):
        """partial_fit_transform in one shot matches fit_transform."""
        full = EWResidualVolatility(
            half_life=5, beta_half_life=8, min_periods=2
        ).fit_transform(simple_panel)

        partial = EWResidualVolatility(
            half_life=5, beta_half_life=8, min_periods=2
        ).partial_fit_transform(simple_panel)

        np.testing.assert_array_equal(full, partial)

    def test_partial_fit_chunked(self, simple_panel):
        """Chunked partial_fit_transform matches fit_transform."""
        half_life = 5
        beta_half_life = 8
        min_periods = 3

        full = EWResidualVolatility(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=min_periods,
        ).fit_transform(simple_panel)

        descriptor = EWResidualVolatility(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=min_periods,
        )
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_partial_fit_small_chunks(self, simple_panel):
        """Single-observation chunks produce correct results."""
        half_life = 5
        beta_half_life = 8
        min_periods = 2

        full = EWResidualVolatility(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=min_periods,
        ).fit_transform(simple_panel)

        descriptor = EWResidualVolatility(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=min_periods,
        )
        chunks = []
        for start in range(20):
            chunks.append(
                descriptor.partial_fit_transform(simple_panel[start : start + 1])
            )

        combined = np.concatenate(chunks, axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_fit_transform_resets_state(self, simple_panel):
        """fit_transform resets state from a previous run."""
        descriptor = EWResidualVolatility(half_life=5, beta_half_life=8, min_periods=2)
        descriptor.partial_fit_transform(simple_panel[:10])

        result = descriptor.fit_transform(simple_panel)
        expected = EWResidualVolatility(
            half_life=5, beta_half_life=8, min_periods=2
        ).fit_transform(simple_panel)
        np.testing.assert_array_equal(result, expected)

    def test_nan_return_holds_state(self, simple_panel):
        """NaN return holds EWMA state for that asset."""
        simple_panel["returns"][5, 0] = np.nan
        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=1
        ).fit_transform(simple_panel)
        np.testing.assert_almost_equal(result[5, 0], result[4, 0])

    def test_nan_return_does_not_increment_asset_min_periods(self, simple_panel):
        """Missing asset returns do not count toward per-asset readiness."""
        simple_panel["returns"][:2, 0] = np.nan
        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=3
        ).fit_transform(simple_panel)
        assert np.isnan(result[2, 0])
        assert np.isnan(result[3, 0])
        assert not np.isnan(result[4, 0])

    def test_late_listing_needs_asset_min_periods(self, simple_panel):
        """Late-listed assets need min_periods valid returns before output."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=3
        ).fit_transform(simple_panel)

        assert np.isnan(result[5, 0])
        assert np.isnan(result[6, 0])
        assert not np.isnan(result[7, 0])

    def test_inactive_gap_resets_warmup(self, simple_panel):
        """Inactive observations reset asset-specific state and valid counts."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[5:7, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][5:7, 0] = np.nan

        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=3
        ).fit_transform(simple_panel)

        assert np.isnan(result[5, 0])
        assert np.isnan(result[6, 0])
        assert np.isnan(result[7, 0])
        assert np.isnan(result[8, 0])
        assert not np.isnan(result[9, 0])

    def test_raises_on_invalid_half_life(self, simple_panel):
        """Raises ValueError when half_life <= 0."""
        with pytest.raises(ValueError, match="half_life must be positive"):
            EWResidualVolatility(half_life=0).fit_transform(simple_panel)

    def test_raises_on_invalid_beta_half_life(self, simple_panel):
        """Raises ValueError when beta_half_life <= 0."""
        with pytest.raises(ValueError, match="beta_half_life must be positive"):
            EWResidualVolatility(half_life=5, beta_half_life=0).fit_transform(
                simple_panel
            )

    def test_raises_on_invalid_min_periods(self, simple_panel):
        """Raises ValueError when min_periods < 1."""
        with pytest.raises(ValueError, match="min_periods must be >= 1"):
            EWResidualVolatility(half_life=5, min_periods=0).fit_transform(simple_panel)

    def test_raises_on_invalid_eps(self, simple_panel):
        """Raises ValueError when eps <= 0."""
        with pytest.raises(ValueError, match="eps must be positive"):
            EWResidualVolatility(half_life=5, eps=0).fit_transform(simple_panel)

    def test_raises_on_infinite_returns(self, simple_panel):
        simple_panel["returns"][0, 0] = np.inf
        with pytest.raises(
            ValueError, match='Field "returns" contains infinite values'
        ):
            EWResidualVolatility(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_residual_volatility_attribute_set(self, simple_panel):
        """The fitted ``residual_volatility_`` attribute stores the last row."""
        descriptor = EWResidualVolatility(half_life=5, beta_half_life=5, min_periods=1)
        result = descriptor.fit_transform(simple_panel)
        np.testing.assert_array_equal(descriptor.residual_volatility_, result[-1])

    def test_higher_half_life_smoother(self, simple_panel):
        """Higher half_life produces smoother (less volatile) output."""
        result_fast = EWResidualVolatility(
            half_life=2, beta_half_life=5, min_periods=1
        ).fit_transform(simple_panel)
        result_slow = EWResidualVolatility(
            half_life=10, beta_half_life=5, min_periods=1
        ).fit_transform(simple_panel)
        valid_fast = result_fast[~np.isnan(result_fast)]
        valid_slow = result_slow[~np.isnan(result_slow)]
        assert np.var(valid_slow) < np.var(valid_fast)

    def test_no_min_acceptable_return_param(self):
        """EWResidualVolatility does not expose min_acceptable_return."""
        d = EWResidualVolatility()
        params = d.get_params()
        assert "min_acceptable_return" not in params


class TestEWResidualDownsideVolatility:
    """Tests for EWResidualDownsideVolatility descriptor."""

    def test_downside_formula(self, simple_panel):
        """Downside residual vol matches manual computation."""
        half_life = 5
        beta_half_life = 8
        eps = 1e-12
        mar = 0.0

        result = EWResidualDownsideVolatility(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_acceptable_return=mar,
            min_periods=1,
            eps=eps,
        ).fit_transform(simple_panel)

        expected = _manual_residual_volatility(
            simple_panel,
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=1,
            eps=eps,
            min_acceptable_return=mar,
        )
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_downside_leq_total(self, simple_panel):
        """Downside residual vol is always <= total residual vol."""
        kwargs = dict(half_life=5, beta_half_life=8, min_periods=1)

        total = EWResidualVolatility(**kwargs).fit_transform(simple_panel)
        downside = EWResidualDownsideVolatility(**kwargs).fit_transform(simple_panel)

        valid = ~np.isnan(total) & ~np.isnan(downside)
        assert np.all(downside[valid] <= total[valid] + 1e-12)

    def test_downside_chunked_matches_full(self, simple_panel):
        """Chunked downside processing matches full fit_transform."""
        kwargs = dict(half_life=5, beta_half_life=8, min_periods=2)

        full = EWResidualDownsideVolatility(**kwargs).fit_transform(simple_panel)

        descriptor = EWResidualDownsideVolatility(**kwargs)
        r1 = descriptor.partial_fit_transform(simple_panel[:7])
        r2 = descriptor.partial_fit_transform(simple_panel[7:13])
        r3 = descriptor.partial_fit_transform(simple_panel[13:])

        combined = np.concatenate([r1, r2, r3], axis=0)
        np.testing.assert_array_almost_equal(combined, full)

    def test_nonzero_threshold(self, simple_panel):
        """Non-zero threshold produces different results than zero."""
        kwargs = dict(half_life=5, beta_half_life=8, min_periods=1)

        result_zero = EWResidualDownsideVolatility(
            min_acceptable_return=0.0, **kwargs
        ).fit_transform(simple_panel)
        result_pos = EWResidualDownsideVolatility(
            min_acceptable_return=0.01, **kwargs
        ).fit_transform(simple_panel)

        # Higher threshold captures more "downside" events -> higher vol
        valid = ~np.isnan(result_zero) & ~np.isnan(result_pos)
        assert np.all(result_pos[valid] >= result_zero[valid] - 1e-12)

    def test_exposes_min_acceptable_return_param(self):
        """EWResidualDownsideVolatility exposes min_acceptable_return."""
        d = EWResidualDownsideVolatility()
        params = d.get_params()
        assert "min_acceptable_return" in params
        assert params["min_acceptable_return"] == 0.0
