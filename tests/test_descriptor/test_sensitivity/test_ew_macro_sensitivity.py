"""Tests for EWMacroSensitivity descriptor."""

import numpy as np
import pytest

from skfolio.containers import AssetPanel
from skfolio.descriptor import EWMacroSensitivity


def _partial_fit_in_chunks(estimator, X, reference_returns, chunk_size):
    """Feed `X` to `partial_fit_transform` in chunks and stack the outputs."""
    return np.vstack(
        [
            estimator.partial_fit_transform(
                X[i : i + chunk_size],
                reference_returns=reference_returns[i : i + chunk_size],
            )
            for i in range(0, X.n_observations, chunk_size)
        ]
    )


@pytest.fixture
def ref_returns(clean_panel_data):
    """Synthetic reference returns aligned with clean_panel_data."""
    rng = np.random.default_rng(99)
    return rng.standard_normal(clean_panel_data.n_observations) * 0.005


@pytest.fixture
def deterministic_panel():
    """Small deterministic panel with known bivariate factor structure.

    True model: r_i = beta_m_i * r_market + beta_ref_i * r_ref + noise
    with beta_m = [0.8, 1.2, 1.0], beta_ref = [0.3, -0.2, 0.5].
    """
    np.random.seed(42)
    n_obs, n_assets = 30, 3

    market_factor = np.random.randn(n_obs) * 0.015
    ref_factor = np.random.randn(n_obs) * 0.010

    betas_market = np.array([0.8, 1.2, 1.0])
    betas_ref = np.array([0.3, -0.2, 0.5])
    noise = np.random.randn(n_obs, n_assets) * 0.003

    asset_returns = (
        market_factor[:, None] * betas_market[None, :]
        + ref_factor[:, None] * betas_ref[None, :]
        + noise
    )
    market_cap = np.ones((n_obs, n_assets)) * 1e9

    panel = AssetPanel(
        fields={"returns": asset_returns, "market_cap": market_cap},
        observations=np.arange(n_obs),
        asset_names=np.array(["A", "B", "C"]),
    )
    return panel, ref_factor


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestBasic:
    """Basic API tests."""

    def test_output_shape(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(half_life=10, min_periods=5, aggregation_period=1)
        result = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)

        assert result.shape == (
            clean_panel_data.n_observations,
            clean_panel_data.n_assets,
        )

    def test_default_min_periods(self, clean_panel_data, ref_returns):
        """`min_periods=None` defaults to ceil(half_life)."""
        default = EWMacroSensitivity(half_life=10.2).fit_transform(
            clean_panel_data, reference_returns=ref_returns
        )
        explicit = EWMacroSensitivity(half_life=10.2, min_periods=11).fit_transform(
            clean_panel_data, reference_returns=ref_returns
        )
        np.testing.assert_array_equal(default, explicit)
        assert np.isnan(default[9]).all()
        assert not np.isnan(default[10]).any()

    def test_macro_sensitivity_fitted_attribute(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(half_life=10, min_periods=5, aggregation_period=1)
        result = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)

        np.testing.assert_allclose(ms.macro_sensitivity_, result[-1], equal_nan=True)

    def test_warm_up_nan(self, clean_panel_data, ref_returns):
        """Rows before min_periods should be NaN."""
        ms = EWMacroSensitivity(half_life=5, min_periods=5, aggregation_period=1)
        result = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)
        # First 4 rows (indices 0-3) must be all NaN
        assert np.all(np.isnan(result[:4]))
        # Row at index 4 (5th observation) should have values
        assert not np.all(np.isnan(result[4]))

    def test_missing_reference_returns_raises(self, clean_panel_data):
        ms = EWMacroSensitivity(half_life=5, min_periods=5, aggregation_period=1)
        with pytest.raises(ValueError, match="reference_returns must be provided"):
            ms.fit_transform(clean_panel_data)

    def test_reference_returns_length_mismatch(self, clean_panel_data):
        ms = EWMacroSensitivity(half_life=5, min_periods=5, aggregation_period=1)
        bad_ref = np.zeros(3)
        with pytest.raises(ValueError, match="reference_returns length"):
            ms.fit_transform(clean_panel_data, reference_returns=bad_ref)

    def test_reference_returns_must_be_1d(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(half_life=5, min_periods=5, aggregation_period=1)
        with pytest.raises(ValueError, match="reference_returns must be a 1D array"):
            ms.fit_transform(
                clean_panel_data, reference_returns=ref_returns.reshape(-1, 1)
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    """Tests for aggregation_period parameter."""

    def test_aggregation_delays_first_valid(self, clean_panel_data, ref_returns):
        ms1 = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=5)
        ms5 = EWMacroSensitivity(half_life=5, aggregation_period=5, min_periods=5)

        r1 = ms1.fit_transform(clean_panel_data, reference_returns=ref_returns)
        r5 = ms5.fit_transform(clean_panel_data, reference_returns=ref_returns)

        first_valid_1 = np.where(~np.isnan(r1[:, 0]))[0][0]
        first_valid_5 = np.where(~np.isnan(r5[:, 0]))[0][0]
        assert first_valid_5 > first_valid_1

    def test_beta_held_within_aggregation_window(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(half_life=5, aggregation_period=5, min_periods=2)
        result = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)

        first_valid_idx = np.where(~np.isnan(result[:, 0]))[0][0]
        for i in range(1, 5):
            if first_valid_idx + i < result.shape[0]:
                np.testing.assert_array_equal(
                    result[first_valid_idx],
                    result[first_valid_idx + i],
                )

    def test_buffer_persistence_across_partial_fit(self, clean_panel_data, ref_returns):
        """An incomplete aggregation window carries over to the next call."""
        expected = EWMacroSensitivity(
            half_life=5, aggregation_period=5, min_periods=2
        ).fit_transform(clean_panel_data, reference_returns=ref_returns)

        ms = EWMacroSensitivity(half_life=5, aggregation_period=5, min_periods=2)
        # First chunk: 3 observations (incomplete window). Second chunk: 4
        # observations (completes the window and starts the next one).
        result = np.vstack(
            [
                ms.partial_fit_transform(
                    clean_panel_data[s], reference_returns=ref_returns[s]
                )
                for s in (slice(0, 3), slice(3, 7), slice(7, None))
            ]
        )

        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Partial fit
# ---------------------------------------------------------------------------


class TestPartialFit:
    """Online / chunked processing tests."""

    def test_partial_fit_matches_fit(self, clean_panel_data, ref_returns):
        ms_fit = EWMacroSensitivity(half_life=10, aggregation_period=1, min_periods=5)
        result_fit = ms_fit.fit_transform(
            clean_panel_data, reference_returns=ref_returns
        )

        ms_pf = EWMacroSensitivity(half_life=10, aggregation_period=1, min_periods=5)
        result_pf = ms_pf.partial_fit_transform(
            clean_panel_data, reference_returns=ref_returns
        )

        np.testing.assert_array_almost_equal(result_fit, result_pf)

    def test_partial_fit_chunked_matches_fit(self, clean_panel_data, ref_returns):
        ms_fit = EWMacroSensitivity(half_life=10, aggregation_period=1, min_periods=5)
        result_fit = ms_fit.fit_transform(
            clean_panel_data, reference_returns=ref_returns
        )

        ms_pf = EWMacroSensitivity(half_life=10, aggregation_period=1, min_periods=5)
        result_pf = _partial_fit_in_chunks(
            ms_pf, clean_panel_data, ref_returns, chunk_size=7
        )

        np.testing.assert_array_equal(result_pf, result_fit)
        np.testing.assert_array_equal(
            ms_pf.macro_sensitivity_, ms_fit.macro_sensitivity_
        )

    def test_partial_fit_aggregation_chunked(self, clean_panel_data, ref_returns):
        ms_fit = EWMacroSensitivity(half_life=3, aggregation_period=5, min_periods=2)
        result_fit = ms_fit.fit_transform(
            clean_panel_data, reference_returns=ref_returns
        )

        ms_pf = EWMacroSensitivity(half_life=3, aggregation_period=5, min_periods=2)
        # Chunks of 3 do not align with the aggregation windows of 5.
        result_pf = _partial_fit_in_chunks(
            ms_pf, clean_panel_data, ref_returns, chunk_size=3
        )

        np.testing.assert_array_equal(result_pf, result_fit)


# ---------------------------------------------------------------------------
# Beta correctness
# ---------------------------------------------------------------------------


class TestBetaValues:
    """Tests for computational correctness."""

    def test_betas_reasonable_range(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(half_life=20, aggregation_period=1, min_periods=10)
        result = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)
        valid = result[~np.isnan(result)]
        assert np.all(np.abs(valid) < 20)

    def test_orthogonality_to_market(self, deterministic_panel):
        """When reference is uncorrelated with market, ref beta should
        approximate the true value and preserve signs."""
        panel, ref_factor = deterministic_panel

        ms = EWMacroSensitivity(half_life=8, aggregation_period=1, min_periods=5)
        result = ms.fit_transform(panel, reference_returns=ref_factor)
        final = result[-1]

        # True ref betas: [0.3, -0.2, 0.5]. Signs should agree.
        assert final[0] > 0, "Asset A ref beta should be positive"
        assert final[1] < 0, "Asset B ref beta should be negative"
        assert final[2] > 0, "Asset C ref beta should be positive"

    def test_zero_reference_gives_zero_beta(self, clean_panel_data):
        """A flat reference series should yield zero ref betas."""
        n_obs = clean_panel_data.n_observations
        zero_ref = np.zeros(n_obs)

        ms = EWMacroSensitivity(half_life=10, aggregation_period=1, min_periods=5)
        result = ms.fit_transform(clean_panel_data, reference_returns=zero_ref)

        valid = result[~np.isnan(result)]
        np.testing.assert_allclose(valid, 0.0, atol=1e-6)

    def test_ref_betas_match_frisch_waugh_formula(self, deterministic_panel):
        """Reference betas match the closed-form Frisch-Waugh decomposition."""
        panel, ref_factor = deterministic_panel

        half_life = 8
        min_periods = 5
        eps = 1e-12
        batch = EWMacroSensitivity(
            half_life=half_life, aggregation_period=1, min_periods=min_periods, eps=eps
        )
        batch_result = batch.fit_transform(panel, reference_returns=ref_factor)

        online = EWMacroSensitivity(
            half_life=half_life, aggregation_period=1, min_periods=min_periods, eps=eps
        )
        online_chunks = []
        for start in range(0, panel.n_observations, 7):
            end = min(start + 7, panel.n_observations)
            online_chunks.append(
                online.partial_fit_transform(
                    panel[start:end], reference_returns=ref_factor[start:end]
                )
            )
        online_result = np.vstack(online_chunks)

        returns = panel["returns"]
        market_returns = np.mean(returns, axis=1)
        decay = np.exp(-np.log(2) / half_life)

        mu_market = 0.0
        mu_ref = 0.0
        mu_assets = np.zeros(panel.n_assets)
        var_market = 0.0
        var_ref = 0.0
        cov_market_ref = 0.0
        cov_assets_market = np.zeros(panel.n_assets)
        cov_assets_ref = np.zeros(panel.n_assets)
        n_valid_assets = np.zeros(panel.n_assets, dtype=int)

        for t in range(panel.n_observations):
            market_deviation = market_returns[t] - mu_market
            ref_deviation = ref_factor[t] - mu_ref

            mu_market = decay * mu_market + (1 - decay) * market_returns[t]
            mu_ref = decay * mu_ref + (1 - decay) * ref_factor[t]

            var_market = decay * var_market + (1 - decay) * (
                market_deviation * market_deviation
            )
            var_ref = decay * var_ref + (1 - decay) * (ref_deviation * ref_deviation)
            cov_market_ref = decay * cov_market_ref + (1 - decay) * (
                market_deviation * ref_deviation
            )

            valid = np.isfinite(returns[t])
            n_valid_assets[valid] += 1
            asset_deviations = returns[t, valid] - mu_assets[valid]
            mu_assets[valid] = (
                decay * mu_assets[valid] + (1 - decay) * returns[t, valid]
            )
            cov_assets_market[valid] = (
                decay * cov_assets_market[valid]
                + (1 - decay) * asset_deviations * market_deviation
            )
            cov_assets_ref[valid] = (
                decay * cov_assets_ref[valid]
                + (1 - decay) * asset_deviations * ref_deviation
            )

        market_variance = var_market + eps
        reference_residual_variance = (
            var_ref - cov_market_ref**2 / market_variance + eps
        )
        expected = (
            cov_assets_ref - cov_assets_market * cov_market_ref / market_variance
        ) / reference_residual_variance

        ready = n_valid_assets >= min_periods
        expected = np.where(ready, expected, np.nan)

        np.testing.assert_allclose(batch_result[-1], expected, equal_nan=True)
        np.testing.assert_allclose(online_result[-1], expected, equal_nan=True)
        np.testing.assert_allclose(batch.macro_sensitivity_, expected, equal_nan=True)
        np.testing.assert_allclose(online.macro_sensitivity_, expected, equal_nan=True)
        np.testing.assert_allclose(batch_result, online_result, equal_nan=True)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


class TestNaNHandling:
    """NaN handling in asset returns."""

    def test_nan_in_data_does_not_crash(self, panel_data):
        rng = np.random.default_rng(42)
        n_obs = panel_data.n_observations
        ref = rng.standard_normal(n_obs) * 0.005

        ms = EWMacroSensitivity(half_life=10, aggregation_period=5, min_periods=5)
        result = ms.fit_transform(panel_data, reference_returns=ref)
        assert result is not None
        assert not np.all(np.isnan(result))

    def test_all_nan_estimable_date_raises(self, clean_panel_data, ref_returns):
        """All-NaN estimable date raises because market return is undefined."""
        X = clean_panel_data.copy()
        returns = X["returns"].copy()
        returns[10, :] = np.nan
        X["returns"] = returns

        ms = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=3)
        with pytest.raises(ValueError, match="Market return is undefined"):
            ms.fit_transform(X, reference_returns=ref_returns)

    def test_all_nan_estimable_window_raises_with_aggregation(
        self, clean_panel_data, ref_returns
    ):
        """All-NaN estimable window raises before aggregation."""
        X = clean_panel_data.copy()
        returns = X["returns"].copy()
        returns[5:10, :] = np.nan
        X["returns"] = returns

        ms = EWMacroSensitivity(half_life=3, aggregation_period=5, min_periods=1)
        with pytest.raises(ValueError, match="Market return is undefined"):
            ms.fit_transform(X, reference_returns=ref_returns)

    def test_nan_reference_date_freezes_state(self, clean_panel_data, ref_returns):
        """A non-finite reference value should freeze state and keep last beta."""
        ref = ref_returns.copy()
        ref[10] = np.nan

        ms = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=3)
        result = ms.fit_transform(clean_panel_data, reference_returns=ref)

        np.testing.assert_array_equal(
            result[10, :],
            result[9, :],
            err_msg="Betas should be held when reference return is non-finite",
        )

        # The frozen observation leaves no trace: the rest of the output matches a
        # fit on the data without it.
        keep = np.arange(clean_panel_data.n_observations) != 10
        without = EWMacroSensitivity(
            half_life=5, aggregation_period=1, min_periods=3
        ).fit_transform(clean_panel_data[keep], reference_returns=ref[keep])
        np.testing.assert_array_equal(result[keep], without)

    def test_nan_return_does_not_increment_asset_min_periods(
        self, clean_panel_data, ref_returns
    ):
        """Missing asset returns do not count toward per-asset readiness."""
        X = clean_panel_data.copy()
        returns = X["returns"].copy()
        returns[:2, 0] = np.nan
        X["returns"] = returns

        result = EWMacroSensitivity(
            half_life=5, aggregation_period=1, min_periods=3
        ).fit_transform(X, reference_returns=ref_returns)

        assert np.isnan(result[2, 0])
        assert np.isnan(result[3, 0])
        assert not np.isnan(result[4, 0])

    def test_late_listing_needs_asset_min_periods(self, clean_panel_data, ref_returns):
        """Late-listed assets need min_periods valid returns before output."""
        X = clean_panel_data.copy()
        active_mask = X.active_mask.copy()
        estimation_mask = X.estimation_mask.copy()
        active_mask[:5, 0] = False
        estimation_mask[:5, 0] = False
        X.active_mask = active_mask
        X.estimation_mask = estimation_mask
        X["returns"][:5, 0] = np.nan

        result = EWMacroSensitivity(
            half_life=5, aggregation_period=1, min_periods=3
        ).fit_transform(X, reference_returns=ref_returns)

        assert np.isnan(result[5, 0])
        assert np.isnan(result[6, 0])
        assert not np.isnan(result[7, 0])

    def test_raises_on_infinite_returns(self, clean_panel_data, ref_returns):
        X = clean_panel_data.copy()
        X["returns"][0, 0] = np.inf

        ms = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=3)
        with pytest.raises(
            ValueError, match='Field "returns" contains infinite values'
        ):
            ms.fit_transform(X, reference_returns=ref_returns)

    def test_raises_on_infinite_reference_returns(self, clean_panel_data, ref_returns):
        ref = ref_returns.copy()
        ref[0] = np.inf

        ms = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=3)
        with pytest.raises(
            ValueError, match="reference_returns contains infinite values"
        ):
            ms.fit_transform(clean_panel_data, reference_returns=ref)


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


class TestParameterValidation:
    """Validation of constructor parameters."""

    def test_invalid_half_life(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(half_life=0, aggregation_period=1)
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            ms.fit_transform(clean_panel_data, reference_returns=ref_returns)

    def test_invalid_aggregation_period(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(aggregation_period=0)
        with pytest.raises(
            ValueError, match="aggregation_period must be a positive integer"
        ):
            ms.fit_transform(clean_panel_data, reference_returns=ref_returns)

    def test_invalid_min_periods(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(min_periods=0, aggregation_period=1)
        with pytest.raises(ValueError, match="min_periods must be a positive integer"):
            ms.fit_transform(clean_panel_data, reference_returns=ref_returns)

    def test_invalid_eps(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(eps=0, aggregation_period=1)
        with pytest.raises(ValueError, match="eps must be a positive number"):
            ms.fit_transform(clean_panel_data, reference_returns=ref_returns)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case coverage."""

    def test_reset_on_fit_transform(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(half_life=10, aggregation_period=1, min_periods=5)
        first = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)
        second = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)
        np.testing.assert_array_equal(first, second)

    def test_eps_prevents_inf(self, clean_panel_data, ref_returns):
        ms = EWMacroSensitivity(
            half_life=10, aggregation_period=1, min_periods=5, eps=1e-12
        )
        result = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)
        valid = result[~np.isnan(result)]
        assert not np.any(np.isinf(valid))


# ---------------------------------------------------------------------------
# Regression (exact values)
# ---------------------------------------------------------------------------


class TestRegression:
    """Regression tests with exact expected values."""

    def test_partial_fit_exact_state_match(self, deterministic_panel):
        panel, ref_factor = deterministic_panel

        ms_fit = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=5)
        result_fit = ms_fit.fit_transform(panel, reference_returns=ref_factor)

        ms_pf = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=5)
        result_pf = _partial_fit_in_chunks(ms_pf, panel, ref_factor, chunk_size=7)

        np.testing.assert_array_equal(result_pf, result_fit)
        np.testing.assert_array_equal(
            ms_pf.macro_sensitivity_, ms_fit.macro_sensitivity_
        )
