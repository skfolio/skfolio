"""Tests for EWResidualVolatility and EWResidualDownsideVolatility descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.containers import AssetPanel
from skfolio.descriptor import (
    EWResidualDownsideVolatility,
    EWResidualVolatility,
    EWVolatility,
)
from tests.test_descriptor._ew_reference import residual_path


@pytest.mark.parametrize(
    "descriptor_cls", [EWResidualVolatility, EWResidualDownsideVolatility]
)
def test_squared_residual_overflow_raises(descriptor_cls):
    """Finite returns with overflowing squared residuals raise an explicit error."""
    n_observations = 12
    panel = AssetPanel(
        fields={
            "returns": np.column_stack(
                (
                    np.linspace(-0.01, 0.01, n_observations),
                    np.full(n_observations, -1e160),
                )
            ),
            "market_cap": np.tile([1.0, 0.0], (n_observations, 1)),
        },
        observations=np.arange(n_observations),
        asset_names=np.array(["market", "asset"]),
    )
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="Non-finite EW residual moments"),
    ):
        descriptor_cls(half_life=5, beta_half_life=5, min_periods=1).fit_transform(
            panel
        )


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
        assert np.isnan(result[:16]).all()
        assert np.isfinite(result[16]).all()

    def test_default_min_periods_uses_slower_half_life(self, simple_panel):
        """Default min_periods resolves to ceil of the slower half-life."""
        d = EWResidualVolatility(half_life=8.2, beta_half_life=5)
        assert d.min_periods is None
        result = d.fit_transform(simple_panel)
        assert d._min_periods == 9
        assert np.isnan(result[:17]).all()
        assert np.isfinite(result[17:]).all()

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
        min_periods = 1

        result = EWResidualVolatility(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=min_periods,
        ).fit_transform(simple_panel)

        expected = residual_path(
            simple_panel,
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=min_periods,
        )
        np.testing.assert_allclose(result, expected, equal_nan=True)

    def test_same_half_life_matches_single_decay(self, simple_panel):
        """When half_life == beta_half_life, result uses a single decay."""
        hl = 7
        result = EWResidualVolatility(
            half_life=hl, beta_half_life=hl, min_periods=1
        ).fit_transform(simple_panel)

        expected = residual_path(
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

        late = result[12:]
        late_total = total_vol_series[12:]
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
        simple_panel["returns"][15, 0] = np.nan
        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=1
        ).fit_transform(simple_panel)
        assert np.isfinite(result[14, 0])
        np.testing.assert_equal(result[15, 0], result[14, 0])

    def test_nan_return_does_not_increment_asset_min_periods(self, simple_panel):
        """Missing asset returns do not count toward per-asset readiness."""
        simple_panel["returns"][:2, 0] = np.nan
        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=3
        ).fit_transform(simple_panel)
        assert np.isnan(result[:13, 0]).all()
        assert np.isfinite(result[13, 0])

    def test_late_listing_needs_asset_min_periods(self, simple_panel):
        """Late-listed assets need min_periods valid returns before output."""
        active_mask = simple_panel.active_mask.copy()
        active_mask[:5, 0] = False
        simple_panel.active_mask = active_mask
        simple_panel["returns"][:5, 0] = np.nan

        result = EWResidualVolatility(
            half_life=5, beta_half_life=5, min_periods=3
        ).fit_transform(simple_panel)

        assert np.isnan(result[:16, 0]).all()
        assert np.isfinite(result[16, 0])

    def test_late_listed_residual_vol_matches_batch_reference(self):
        """Late listings use their own matched market and residual histories."""
        rng = np.random.default_rng(0)
        n_obs, n_assets = 400, 20
        half_life = 40
        beta_half_life = 60
        listing = 200
        market = rng.normal(0, 0.01, n_obs)
        true_beta = np.ones(n_assets) * 1.5
        returns = true_beta * market[:, None] + rng.normal(0, 0.01, (n_obs, n_assets))
        active = np.ones((n_obs, n_assets), dtype=bool)
        active[:listing, 0] = False
        returns[~active] = np.nan
        market_cap = np.ones((n_obs, n_assets))
        market_cap[~active] = np.nan
        panel = AssetPanel(
            fields={"returns": returns, "market_cap": market_cap},
            observations=np.arange(n_obs),
            asset_names=np.array([f"A{i}" for i in range(n_assets)]),
            active_mask=active,
        )

        result = EWResidualVolatility(
            half_life=half_life, beta_half_life=beta_half_life
        ).fit_transform(panel)

        expected = residual_path(panel, half_life, beta_half_life, 60)
        np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)
        assert np.isnan(result[listing : listing + 68, 0]).all()
        assert np.isfinite(result[listing + 68, 0])

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
        expected = residual_path(simple_panel, 5, 5, 3)
        np.testing.assert_allclose(result, expected, rtol=1e-12, equal_nan=True)

    def test_raises_on_invalid_half_life(self, simple_panel):
        """Raises ValueError when half_life <= 0."""
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            EWResidualVolatility(half_life=0).fit_transform(simple_panel)

    def test_raises_on_invalid_beta_half_life(self, simple_panel):
        """Raises ValueError when beta_half_life <= 0."""
        with pytest.raises(
            ValueError, match="beta_half_life must be a positive number"
        ):
            EWResidualVolatility(half_life=5, beta_half_life=0).fit_transform(
                simple_panel
            )

    def test_raises_on_invalid_min_periods(self, simple_panel):
        """Raises ValueError when min_periods < 1."""
        with pytest.raises(ValueError, match="min_periods must be a positive integer"):
            EWResidualVolatility(half_life=5, min_periods=0).fit_transform(simple_panel)

    def test_raises_on_invalid_eps(self, simple_panel):
        """Raises ValueError when eps <= 0."""
        with pytest.raises(ValueError, match="eps must be a positive number"):
            EWResidualVolatility(half_life=5, eps=0).fit_transform(simple_panel)

    def test_raises_on_infinite_returns(self, simple_panel):
        simple_panel["returns"][0, 0] = np.inf
        with pytest.raises(
            ValueError, match='Field "returns" contains infinite values'
        ):
            EWResidualVolatility(half_life=5, min_periods=1).fit_transform(simple_panel)

    def test_residual_volatility_attribute_set(self, simple_panel):
        """The fitted `residual_volatility_` attribute stores the last row."""
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
        mar = 0.0

        result = EWResidualDownsideVolatility(
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_acceptable_return=mar,
            min_periods=1,
        ).fit_transform(simple_panel)

        expected = residual_path(
            simple_panel,
            half_life=half_life,
            beta_half_life=beta_half_life,
            min_periods=1,
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
