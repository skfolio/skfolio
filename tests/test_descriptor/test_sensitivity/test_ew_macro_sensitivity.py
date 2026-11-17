"""Tests for EWMacroSensitivity descriptor."""

import numpy as np
import pytest

from skfolio.containers import AssetPanel
from skfolio.descriptor import EWMacroSensitivity
from tests.test_descriptor._ew_reference import regression_path


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
        ms = EWMacroSensitivity(half_life=10.2)
        result = ms.fit_transform(clean_panel_data, reference_returns=ref_returns)

        assert ms._min_periods == 11
        assert np.isnan(result[:10]).all()
        assert np.isfinite(result[10:]).all()

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
        first = ms.partial_fit_transform(
            clean_panel_data[:3], reference_returns=ref_returns[:3]
        )
        assert np.isnan(first).all()

        second = ms.partial_fit_transform(
            clean_panel_data[3:7], reference_returns=ref_returns[3:7]
        )
        assert np.isnan(second).all()

        rest = ms.partial_fit_transform(
            clean_panel_data[7:], reference_returns=ref_returns[7:]
        )
        result = np.vstack([first, second, rest])
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

    def test_zero_reference_is_unidentifiable(self, clean_panel_data):
        result = EWMacroSensitivity(half_life=5, min_periods=3).fit_transform(
            clean_panel_data,
            reference_returns=np.zeros(clean_panel_data.n_observations),
        )
        assert np.isnan(result).all()

    def test_ref_betas_match_batch_regression(self, deterministic_panel):
        panel, reference = deterministic_panel
        result = EWMacroSensitivity(half_life=5, min_periods=3).fit_transform(
            panel, reference_returns=reference
        )
        expected = regression_path(panel, 5, 3, reference=reference)
        np.testing.assert_allclose(result, expected, rtol=1e-11, atol=1e-14)


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

    @pytest.mark.parametrize("min_periods", [3, 11])
    def test_nan_reference_date_freezes_state(
        self, clean_panel_data, ref_returns, min_periods
    ):
        """A non-finite reference value should freeze state and keep last beta."""
        ref = ref_returns.copy()
        ref[10] = np.nan

        ms = EWMacroSensitivity(
            half_life=5, aggregation_period=1, min_periods=min_periods
        )
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
            half_life=5, aggregation_period=1, min_periods=min_periods
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

    def test_eps_is_ignored_with_warning(self, clean_panel_data, ref_returns):
        expected = EWMacroSensitivity(half_life=5, min_periods=3).fit_transform(
            clean_panel_data, reference_returns=ref_returns
        )
        with pytest.warns(FutureWarning, match="eps is deprecated"):
            result = EWMacroSensitivity(
                half_life=5, min_periods=3, eps=0.1
            ).fit_transform(clean_panel_data, reference_returns=ref_returns)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Regression (exact values)
# ---------------------------------------------------------------------------


class TestRegression:
    """Regression tests with exact expected values."""

    def test_partial_fit_exact_output_match(self, deterministic_panel):
        panel, ref_factor = deterministic_panel

        ms_fit = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=5)
        result_fit = ms_fit.fit_transform(panel[:-5], reference_returns=ref_factor[:-5])

        ms_pf = EWMacroSensitivity(half_life=5, aggregation_period=1, min_periods=5)
        result_pf = _partial_fit_in_chunks(
            ms_pf, panel[:-5], ref_factor[:-5], chunk_size=7
        )

        np.testing.assert_array_equal(result_pf, result_fit)
        np.testing.assert_array_equal(
            ms_pf.macro_sensitivity_, ms_fit.macro_sensitivity_
        )
        np.testing.assert_array_equal(
            ms_pf.partial_fit_transform(panel[-5:], reference_returns=ref_factor[-5:]),
            ms_fit.partial_fit_transform(panel[-5:], reference_returns=ref_factor[-5:]),
        )
