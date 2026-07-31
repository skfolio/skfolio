"""Tests for EWMarketBeta descriptor."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from skfolio.containers import AssetPanel, FieldCategorical
from skfolio.descriptor import EWMarketBeta
from skfolio.utils.stats import _market_returns


@pytest.fixture
def deterministic_data():
    """Deterministic test data with known market factor structure."""
    np.random.seed(42)
    n_obs = 20
    n_assets = 3

    market_factor = np.array(
        [
            0.01,
            -0.02,
            0.015,
            -0.005,
            0.02,
            -0.01,
            0.005,
            0.025,
            -0.015,
            0.01,
            0.005,
            -0.01,
            0.02,
            -0.005,
            0.015,
            -0.02,
            0.01,
            0.005,
            -0.01,
            0.015,
        ]
    )

    # Asset betas: 0.8, 1.2, 1.0 with small noise (seed=42)
    betas_true = np.array([0.8, 1.2, 1.0])
    noise = np.random.randn(n_obs, n_assets) * 0.005
    asset_returns = market_factor[:, None] * betas_true[None, :] + noise
    market_cap = np.ones((n_obs, n_assets)) * 1e9

    return AssetPanel(
        fields={"returns": asset_returns, "market_cap": market_cap},
        observations=np.arange(n_obs),
        asset_names=np.array(["A", "B", "C"]),
    )


@pytest.fixture
def data_with_groups():
    """Test data with industry groups for shrinkage tests."""
    np.random.seed(42)
    n_obs = 30
    n_assets = 9  # 3 industries x 3 assets each

    market_factor = np.random.randn(n_obs) * 0.015

    industry_betas = {"Tech": 1.3, "Finance": 0.9, "Utilities": 0.6}
    industries = [
        "Tech",
        "Tech",
        "Tech",
        "Finance",
        "Finance",
        "Finance",
        "Utilities",
        "Utilities",
        "Utilities",
    ]
    true_betas = np.array([industry_betas[ind] for ind in industries])

    asset_deviation = np.array([0.1, -0.1, 0.0, 0.15, -0.05, -0.1, 0.05, -0.05, 0.0])
    noise = np.random.randn(n_obs, n_assets) * 0.008

    asset_returns = (
        market_factor[:, None] * (true_betas + asset_deviation)[None, :] + noise
    )
    market_cap = np.ones((n_obs, n_assets)) * 1e9

    # Industry labels as integer codes
    industry_map = {"Tech": 0, "Finance": 1, "Utilities": 2}
    industry_codes = np.tile(
        [industry_map[ind] for ind in industries], (n_obs, 1)
    ).astype(np.intp)

    return AssetPanel(
        fields={
            "returns": asset_returns,
            "market_cap": market_cap,
            "industry": FieldCategorical(
                industry_codes.astype(np.int32, copy=False),
                levels=np.array(["Tech", "Finance", "Utilities"]),
            ),
        },
        observations=np.arange(n_obs),
        asset_names=np.array([f"A{i}" for i in range(n_assets)]),
    )


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


class TestMarketSensitivityBasic:
    """Basic functionality tests."""

    def test_output_shape(self, clean_panel_data):
        ms = EWMarketBeta(half_life=10, min_periods=5)
        result = ms.fit_transform(clean_panel_data)
        assert result.shape == (
            clean_panel_data.n_observations,
            clean_panel_data.n_assets,
        )

    def test_default_min_periods(self, clean_panel_data):
        half_life = 10.2
        ms = EWMarketBeta(half_life=half_life)
        ms.fit_transform(clean_panel_data)
        assert ms._min_periods == 11

    def test_market_beta_fitted_attribute(self, clean_panel_data):
        ms = EWMarketBeta(half_life=10, min_periods=5)
        result = ms.fit_transform(clean_panel_data)

        np.testing.assert_allclose(ms.market_beta_, result[-1], equal_nan=True)
        assert not hasattr(ms, "betas_")


class TestAggregationPeriod:
    """Tests for aggregation_period parameter."""

    def test_no_aggregation_default(self, clean_panel_data):
        ms = EWMarketBeta(half_life=10, aggregation_period=1, min_periods=5)
        result = ms.fit_transform(clean_panel_data)
        first_valid_idx = np.where(~np.isnan(result[:, 0]))[0][0]
        assert first_valid_idx == 5 - 1

    def test_aggregation_period_5(self, clean_panel_data):
        ms = EWMarketBeta(half_life=10, aggregation_period=5, min_periods=3)
        result = ms.fit_transform(clean_panel_data)
        first_valid_idx = np.where(~np.isnan(result[:, 0]))[0][0]
        assert first_valid_idx == 3 * 5 - 1

    def test_aggregation_delays_first_beta(self, clean_panel_data):
        ms1 = EWMarketBeta(half_life=5, aggregation_period=1, min_periods=5)
        ms5 = EWMarketBeta(half_life=5, aggregation_period=5, min_periods=5)

        r1 = ms1.fit_transform(clean_panel_data)
        r5 = ms5.fit_transform(clean_panel_data)

        first_valid_1 = np.where(~np.isnan(r1[:, 0]))[0][0]
        first_valid_5 = np.where(~np.isnan(r5[:, 0]))[0][0]

        assert first_valid_5 > first_valid_1
        assert first_valid_5 == first_valid_1 * 5 + 4

    def test_beta_held_during_incomplete_window(self, clean_panel_data):
        ms = EWMarketBeta(half_life=5, aggregation_period=5, min_periods=2)
        result = ms.fit_transform(clean_panel_data)

        first_valid_idx = np.where(~np.isnan(result[:, 0]))[0][0]
        for i in range(1, 5):
            if first_valid_idx + i < len(result):
                np.testing.assert_array_equal(
                    result[first_valid_idx, :],
                    result[first_valid_idx + i, :],
                )


class TestPartialFit:
    """Tests for partial_fit_transform (online learning) mode."""

    def test_partial_fit_matches_fit(self, clean_panel_data):
        ms_fit = EWMarketBeta(half_life=10, aggregation_period=1, min_periods=5)
        result_fit = ms_fit.fit_transform(clean_panel_data)

        ms_pf = EWMarketBeta(half_life=10, aggregation_period=1, min_periods=5)
        result_pf = ms_pf.partial_fit_transform(clean_panel_data)

        np.testing.assert_array_almost_equal(result_fit, result_pf)

    def test_partial_fit_chunked(self, clean_panel_data):
        ms = EWMarketBeta(half_life=10, aggregation_period=1, min_periods=5)

        n_obs = clean_panel_data.n_observations
        chunk_size = 20
        for i in range(0, n_obs, chunk_size):
            end = min(i + chunk_size, n_obs)
            chunk = clean_panel_data[i:end]
            ms.partial_fit_transform(chunk)

        assert not np.all(np.isnan(ms._betas))

    def test_partial_fit_buffer_persistence(self, clean_panel_data):
        ms = EWMarketBeta(half_life=5, aggregation_period=5, min_periods=2)

        chunk1 = clean_panel_data[:3]
        ms.partial_fit_transform(chunk1)
        assert ms._buffer_idx == 3

        chunk2 = clean_panel_data[3:7]
        ms.partial_fit_transform(chunk2)
        assert ms._buffer_idx == 2

    def test_partial_fit_with_aggregation(self, clean_panel_data):
        ms = EWMarketBeta(half_life=5, aggregation_period=5, min_periods=3)

        n_obs = clean_panel_data.n_observations
        chunk_size = 12
        for i in range(0, n_obs, chunk_size):
            end = min(i + chunk_size, n_obs)
            chunk = clean_panel_data[i:end]
            ms.partial_fit_transform(chunk)

        assert not np.all(np.isnan(ms._betas))


class TestNaNHandling:
    """Tests for NaN handling."""

    def test_nan_in_aggregation_window(self, panel_data):
        ms = EWMarketBeta(half_life=10, aggregation_period=5, min_periods=5)
        result = ms.fit_transform(panel_data)
        assert result is not None
        assert not np.all(np.isnan(result))

    def test_all_nan_asset_holds_beta(self, make_characteristics_panel):
        """An asset with all NaN in a window holds previous beta."""
        X = make_characteristics_panel(
            n_assets=5,
            n_observations=76,
            random_state=999,
        )

        # Inject NaN for asset 0 in observations 25-29
        returns = X["returns"].copy()
        returns[25:30, 0] = np.nan
        X["returns"] = returns

        ms = EWMarketBeta(half_life=5, aggregation_period=5, min_periods=3)
        result = ms.fit_transform(X)

        beta_before = result[24, 0]
        beta_during = result[29, 0]
        assert not np.isnan(beta_before), "Beta before NaN window should be valid"
        np.testing.assert_equal(
            beta_before,
            beta_during,
            err_msg="Beta should be held when asset has all NaN in window",
        )

    def test_all_nan_estimable_date_raises(self, clean_panel_data):
        """All-NaN estimable date raises because market return is undefined."""
        X = clean_panel_data.copy()
        returns = X["returns"].copy()
        returns[10, :] = np.nan
        X["returns"] = returns

        ms = EWMarketBeta(half_life=5, aggregation_period=1, min_periods=3)
        with pytest.raises(ValueError, match="Market return is undefined"):
            ms.fit_transform(X)

    def test_all_nan_estimable_window_raises_with_aggregation(self, deterministic_data):
        """All-NaN estimable window raises before aggregation."""
        X = deterministic_data.copy()
        returns = X["returns"].copy()
        returns[5:10, :] = np.nan
        X["returns"] = returns

        ms = EWMarketBeta(half_life=3, aggregation_period=5, min_periods=1)
        with pytest.raises(ValueError, match="Market return is undefined"):
            ms.fit_transform(X)

    def test_nan_return_does_not_increment_asset_min_periods(self, clean_panel_data):
        """Missing asset returns do not count toward per-asset readiness."""
        X = clean_panel_data.copy()
        returns = X["returns"].copy()
        returns[:2, 0] = np.nan
        X["returns"] = returns

        result = EWMarketBeta(half_life=5, min_periods=3).fit_transform(X)

        assert np.isnan(result[2, 0])
        assert np.isnan(result[3, 0])
        assert not np.isnan(result[4, 0])

    def test_late_listing_needs_asset_min_periods(self, clean_panel_data):
        """Late-listed assets need min_periods valid returns before output."""
        X = clean_panel_data.copy()
        active_mask = X.active_mask.copy()
        estimation_mask = X.estimation_mask.copy()
        active_mask[:5, 0] = False
        estimation_mask[:5, 0] = False
        X.active_mask = active_mask
        X.estimation_mask = estimation_mask
        X["returns"][:5, 0] = np.nan

        result = EWMarketBeta(half_life=5, min_periods=3).fit_transform(X)

        assert np.isnan(result[5, 0])
        assert np.isnan(result[6, 0])
        assert not np.isnan(result[7, 0])

    def test_raises_on_infinite_returns(self, clean_panel_data):
        X = clean_panel_data.copy()
        X["returns"][0, 0] = np.inf

        ms = EWMarketBeta(half_life=5, min_periods=3)
        with pytest.raises(
            ValueError, match='Field "returns" contains infinite values'
        ):
            ms.fit_transform(X)


class TestBetaValues:
    """Tests for beta computation correctness."""

    def test_betas_reasonable_range(self, clean_panel_data):
        ms = EWMarketBeta(half_life=20, aggregation_period=1, min_periods=10)
        result = ms.fit_transform(clean_panel_data)
        valid = result[~np.isnan(result)]
        assert np.all(valid > -5)
        assert np.all(valid < 10)

    def test_betas_around_one(self, clean_panel_data):
        ms = EWMarketBeta(half_life=30, aggregation_period=1, min_periods=20)
        result = ms.fit_transform(clean_panel_data)
        final_betas = result[-1, :]
        valid_final = final_betas[~np.isnan(final_betas)]
        mean_beta = np.mean(valid_final)
        assert 0.5 < mean_beta < 2.0

    def test_ewma_convergence(self, clean_panel_data):
        ms = EWMarketBeta(half_life=10, aggregation_period=1, min_periods=5)
        result = ms.fit_transform(clean_panel_data)
        asset_betas = result[:, 0]
        valid_betas = asset_betas[~np.isnan(asset_betas)]
        mid = len(valid_betas) // 2
        early_var = np.var(valid_betas[:mid])
        late_var = np.var(valid_betas[mid:])
        assert late_var < early_var * 3


class TestParameterValidation:
    """Tests for parameter validation."""

    def test_invalid_half_life_zero(self, clean_panel_data):
        ms = EWMarketBeta(half_life=0)
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            ms.fit_transform(clean_panel_data)

    def test_invalid_half_life_negative(self, clean_panel_data):
        ms = EWMarketBeta(half_life=-5)
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            ms.fit_transform(clean_panel_data)

    def test_invalid_aggregation_period_zero(self, clean_panel_data):
        ms = EWMarketBeta(aggregation_period=0)
        with pytest.raises(
            ValueError, match="aggregation_period must be a positive integer"
        ):
            ms.fit_transform(clean_panel_data)

    def test_invalid_aggregation_period_negative(self, clean_panel_data):
        ms = EWMarketBeta(aggregation_period=-1)
        with pytest.raises(
            ValueError, match="aggregation_period must be a positive integer"
        ):
            ms.fit_transform(clean_panel_data)

    def test_invalid_min_periods_zero(self, clean_panel_data):
        ms = EWMarketBeta(min_periods=0)
        with pytest.raises(ValueError, match="min_periods must be a positive integer"):
            ms.fit_transform(clean_panel_data)

    def test_invalid_eps_zero(self, clean_panel_data):
        ms = EWMarketBeta(eps=0)
        with pytest.raises(ValueError, match="eps must be a positive number"):
            ms.fit_transform(clean_panel_data)

    def test_invalid_eps_negative(self, clean_panel_data):
        ms = EWMarketBeta(eps=-1e-12)
        with pytest.raises(ValueError, match="eps must be a positive number"):
            ms.fit_transform(clean_panel_data)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_observation_partial_fit(self, clean_panel_data):
        ms = EWMarketBeta(half_life=5, aggregation_period=5, min_periods=2)
        for i in range(10):
            chunk = clean_panel_data[i : i + 1]
            ms.partial_fit_transform(chunk)
        assert ms._t == 2  # 10 obs / 5 aggregation = 2 complete periods

    def test_reset_on_fit_transform(self, clean_panel_data):
        ms = EWMarketBeta(half_life=10, aggregation_period=5, min_periods=3)
        first = ms.fit_transform(clean_panel_data)
        second = ms.fit_transform(clean_panel_data)
        np.testing.assert_array_equal(first, second)

    def test_aggregation_period_equals_data_length(self, clean_panel_data):
        n_obs = clean_panel_data.n_observations
        ms = EWMarketBeta(half_life=5, aggregation_period=n_obs, min_periods=1)
        result = ms.fit_transform(clean_panel_data)
        assert ms._t == 1
        assert not np.all(np.isnan(result[-1, :]))

    def test_eps_numerical_stability(self, clean_panel_data):
        ms = EWMarketBeta(half_life=10, aggregation_period=1, min_periods=5, eps=1e-12)
        result = ms.fit_transform(clean_panel_data)
        valid_mask = ~np.isnan(result)
        assert not np.any(np.isinf(result[valid_mask]))


# ---------------------------------------------------------------------------
# Regression tests with exact expected values
# ---------------------------------------------------------------------------


class TestRegression:
    """Regression tests to catch computation changes."""

    def test_exact_betas_no_aggregation(self, deterministic_data):
        ms = EWMarketBeta(half_life=5, aggregation_period=1, min_periods=5)
        result = ms.fit_transform(deterministic_data)

        expected_final = np.array([0.83738619, 1.14133883, 1.02127497])
        expected_mid = np.array([0.83281078, 1.2263914, 0.9407978])

        np.testing.assert_array_almost_equal(
            result[-1, :],
            expected_final,
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            result[10, :],
            expected_mid,
            decimal=6,
        )

    def test_exact_ewma_state_no_aggregation(self, deterministic_data):
        ms = EWMarketBeta(half_life=5, aggregation_period=1, min_periods=5)
        ms.fit_transform(deterministic_data)

        assert ms._t == 20
        np.testing.assert_almost_equal(ms._mu_market, 0.002390779182924834, decimal=10)
        np.testing.assert_almost_equal(
            ms._var_market, 0.00016516762777270087, decimal=10
        )
        np.testing.assert_array_almost_equal(
            ms._mu_assets,
            np.array([0.00215129, 0.00257907, 0.00244197]),
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            ms._cov_assets,
            np.array([0.00013831, 0.00018851, 0.00016868]),
            decimal=6,
        )

    def test_exact_betas_with_aggregation(self, deterministic_data):
        ms = EWMarketBeta(half_life=3, aggregation_period=5, min_periods=2)
        result = ms.fit_transform(deterministic_data)

        assert ms._t == 4

        expected_final = np.array([1.26360874, 1.54865494, 0.18773542])
        expected_idx9 = np.array([1.84586389, 0.69576948, 0.45836552])
        expected_idx14 = np.array([1.2625842, 1.40378224, 0.33363275])

        np.testing.assert_array_almost_equal(
            result[-1, :],
            expected_final,
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            result[9, :],
            expected_idx9,
            decimal=6,
        )
        np.testing.assert_array_almost_equal(
            result[14, :],
            expected_idx14,
            decimal=6,
        )

    def test_beta_held_within_aggregation_window(self, deterministic_data):
        ms = EWMarketBeta(half_life=3, aggregation_period=5, min_periods=2)
        result = ms.fit_transform(deterministic_data)

        # Indices 10-13 should all hold the beta computed at index 9
        for i in range(10, 14):
            np.testing.assert_array_equal(
                result[i, :],
                result[9, :],
                err_msg=f"Beta at index {i} should equal beta at index 9",
            )

    def test_partial_fit_exact_match(self, deterministic_data):
        ms_fit = EWMarketBeta(half_life=5, aggregation_period=1, min_periods=5)
        ms_fit.fit_transform(deterministic_data)

        ms_pf = EWMarketBeta(half_life=5, aggregation_period=1, min_periods=5)
        n_obs = deterministic_data.n_observations
        for i in range(0, n_obs, 7):
            end = min(i + 7, n_obs)
            chunk = deterministic_data[i:end]
            ms_pf.partial_fit_transform(chunk)

        assert ms_fit._t == ms_pf._t
        np.testing.assert_array_equal(ms_fit._mu_assets, ms_pf._mu_assets)
        np.testing.assert_array_equal(ms_fit._cov_assets, ms_pf._cov_assets)
        np.testing.assert_equal(ms_fit._mu_market, ms_pf._mu_market)
        np.testing.assert_equal(ms_fit._var_market, ms_pf._var_market)

    def test_partial_fit_aggregation_exact_match(self, deterministic_data):
        ms_fit = EWMarketBeta(half_life=3, aggregation_period=5, min_periods=2)
        ms_fit.fit_transform(deterministic_data)

        ms_pf = EWMarketBeta(half_life=3, aggregation_period=5, min_periods=2)
        n_obs = deterministic_data.n_observations
        for i in range(0, n_obs, 3):
            end = min(i + 3, n_obs)
            chunk = deterministic_data[i:end]
            ms_pf.partial_fit_transform(chunk)

        assert ms_fit._t == ms_pf._t
        np.testing.assert_array_equal(ms_fit._betas, ms_pf._betas)


# ---------------------------------------------------------------------------
# Limit-case tests
# ---------------------------------------------------------------------------


class TestLimitCases:
    """Asymptotic and limit-case behavior tests."""

    def test_long_half_life_approximates_ols_beta(self):
        """With large half-life, EW beta should approach static OLS beta."""
        rng = np.random.default_rng(123)
        n_obs = 5000
        n_assets = 6

        market_factor = rng.normal(0.0, 0.01, size=n_obs)
        true_betas = np.array([0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
        noise = rng.normal(0.0, 0.002, size=(n_obs, n_assets))
        asset_returns = market_factor[:, None] * true_betas[None, :] + noise
        market_cap = rng.lognormal(mean=0.0, sigma=0.5, size=(n_obs, n_assets))

        X = AssetPanel(
            fields={"returns": asset_returns, "market_cap": market_cap},
            observations=np.arange(n_obs),
            asset_names=np.array([f"A{i}" for i in range(n_assets)]),
        )

        ms = EWMarketBeta(half_life=1000, min_periods=2000, aggregation_period=1)
        ew_betas = ms.fit_transform(X)[-1]

        market_rets = _market_returns(asset_returns=asset_returns, weights=market_cap)
        ols_betas = np.array(
            [
                LinearRegression()
                .fit(market_rets.reshape(-1, 1), asset_returns[:, j])
                .coef_[0]
                for j in range(n_assets)
            ]
        )

        np.testing.assert_allclose(ew_betas, ols_betas, atol=0.01)


# ---------------------------------------------------------------------------
# Shrinkage tests
# ---------------------------------------------------------------------------


class TestShrinkage:
    """Tests for Bayesian shrinkage feature."""

    def test_shrinkage_disabled_by_default(self, data_with_groups):
        ms = EWMarketBeta(half_life=5, min_periods=5)
        ms.fit_transform(data_with_groups)
        assert ms._shrinkage_enabled is False

    def test_shrinkage_enabled_with_group(self, data_with_groups):
        ms = EWMarketBeta(half_life=5, min_periods=5, shrinkage_group="industry")
        ms.fit_transform(data_with_groups)
        assert ms._shrinkage_enabled is True
        assert hasattr(ms, "_var_residual")
        assert ms._var_residual is not None

    def test_shrinkage_group_must_be_categorical(self, data_with_groups):
        X = data_with_groups.copy()
        X["industry_numeric"] = X["industry"].astype(float)

        ms = EWMarketBeta(
            half_life=5,
            min_periods=5,
            shrinkage_group="industry_numeric",
        )
        with pytest.raises(ValueError, match="must be a CategoricalField"):
            ms.fit_transform(X)

    def test_shrinkage_changes_betas(self, data_with_groups):
        ms_raw = EWMarketBeta(half_life=5, min_periods=5)
        result_raw = ms_raw.fit_transform(data_with_groups)
        raw_betas = result_raw[-1, :]

        ms_shrunk = EWMarketBeta(half_life=5, min_periods=5, shrinkage_group="industry")
        result_shrunk = ms_shrunk.fit_transform(data_with_groups)
        shrunk_betas = result_shrunk[-1, :]

        assert not np.allclose(raw_betas, shrunk_betas)
        assert not np.any(np.isnan(raw_betas))
        assert not np.any(np.isnan(shrunk_betas))

    def test_shrinkage_reduces_cross_sectional_variance(self, data_with_groups):
        ms_raw = EWMarketBeta(half_life=5, min_periods=5)
        result_raw = ms_raw.fit_transform(data_with_groups)
        raw_betas = result_raw[-1, :]

        ms_shrunk = EWMarketBeta(half_life=5, min_periods=5, shrinkage_group="industry")
        result_shrunk = ms_shrunk.fit_transform(data_with_groups)
        shrunk_betas = result_shrunk[-1, :]

        for start, end in [(0, 3), (3, 6), (6, 9)]:
            raw_var = np.var(raw_betas[start:end])
            shrunk_var = np.var(shrunk_betas[start:end])
            assert shrunk_var <= raw_var, (
                f"Group [{start}:{end}]: shrinkage should reduce within-group variance"
            )

    def test_small_group_falls_back_to_global(self):
        np.random.seed(123)
        n_obs = 30
        n_assets = 6

        market_factor = np.random.randn(n_obs) * 0.015
        true_betas = np.array([1.2, 1.1, 0.9, 0.8, 1.5, 0.5])

        asset_returns = (
            market_factor[:, None] * true_betas[None, :]
            + np.random.randn(n_obs, n_assets) * 0.008
        )
        market_cap = np.ones((n_obs, n_assets)) * 1e9

        # 4 in "Big", 2 in "Small" (below min_group_size=5)
        industries = [0, 0, 0, 0, 1, 1]  # 0=Big, 1=Small
        industry_codes = np.tile(industries, (n_obs, 1)).astype(np.intp)

        X = AssetPanel(
            fields={
                "returns": asset_returns,
                "market_cap": market_cap,
                "industry": FieldCategorical(
                    industry_codes.astype(np.int32, copy=False),
                    levels=np.array(["Big", "Small"]),
                ),
            },
            observations=np.arange(n_obs),
            asset_names=np.array([f"A{i}" for i in range(n_assets)]),
        )

        ms = EWMarketBeta(
            half_life=5,
            min_periods=5,
            shrinkage_group="industry",
            min_group_size=5,
        )
        result = ms.fit_transform(X)
        assert not np.any(np.isnan(result[-1, :]))

    def test_shrinkage_with_aggregation(self, data_with_groups):
        ms = EWMarketBeta(
            half_life=3,
            aggregation_period=5,
            min_periods=2,
            shrinkage_group="industry",
        )
        result = ms.fit_transform(data_with_groups)
        assert not np.all(np.isnan(result[-1, :]))
        assert hasattr(ms, "_var_residual")

    def test_shrinkage_partial_fit(self, data_with_groups):
        ms_fit = EWMarketBeta(half_life=5, min_periods=5, shrinkage_group="industry")
        ms_fit.fit_transform(data_with_groups)

        ms_pf = EWMarketBeta(half_life=5, min_periods=5, shrinkage_group="industry")
        n_obs = data_with_groups.n_observations
        chunk_size = 7
        for i in range(0, n_obs, chunk_size):
            end = min(i + chunk_size, n_obs)
            chunk = data_with_groups[i:end]
            ms_pf.partial_fit_transform(chunk)

        np.testing.assert_array_almost_equal(
            ms_fit._var_residual, ms_pf._var_residual, decimal=10
        )

    def test_invalid_min_group_size(self, data_with_groups):
        ms = EWMarketBeta(
            half_life=5,
            min_periods=5,
            shrinkage_group="industry",
            min_group_size=0,
        )
        with pytest.raises(
            ValueError, match="min_group_size must be a positive integer"
        ):
            ms.fit_transform(data_with_groups)

    def test_shrinkage_held_within_aggregation_window(self, data_with_groups):
        ms = EWMarketBeta(
            half_life=3,
            aggregation_period=5,
            min_periods=2,
            shrinkage_group="industry",
        )
        result = ms.fit_transform(data_with_groups)

        first_valid_idx = np.where(~np.isnan(result[:, 0]))[0][0]
        window_end = min(first_valid_idx + 5, len(result))
        for i in range(first_valid_idx + 1, window_end):
            np.testing.assert_array_equal(
                result[first_valid_idx, :],
                result[i, :],
            )

    def test_shrinkage_bounds_effect(self, data_with_groups):
        ms_default = EWMarketBeta(
            half_life=5, min_periods=5, shrinkage_group="industry"
        )
        result_default = ms_default.fit_transform(data_with_groups)

        ms_bounded = EWMarketBeta(
            half_life=5,
            min_periods=5,
            shrinkage_group="industry",
            shrinkage_bounds=(0.3, 0.7),
        )
        result_bounded = ms_bounded.fit_transform(data_with_groups)

        assert not np.any(np.isnan(result_default[-1, :]))
        assert not np.any(np.isnan(result_bounded[-1, :]))
        assert not np.allclose(result_default[-1, :], result_bounded[-1, :])

    def test_shrinkage_bounds_maximum_effect(self, data_with_groups):
        ms_raw = EWMarketBeta(half_life=5, min_periods=5)
        result_raw = ms_raw.fit_transform(data_with_groups)
        raw_betas = result_raw[-1, :]

        ms_bounded = EWMarketBeta(
            half_life=5,
            min_periods=5,
            shrinkage_group="industry",
            shrinkage_bounds=(0.0, 0.5),
        )
        result_bounded = ms_bounded.fit_transform(data_with_groups)
        shrunk_betas = result_bounded[-1, :]

        assert not np.allclose(raw_betas, shrunk_betas, atol=0.01)

    def test_invalid_shrinkage_bounds(self, data_with_groups):
        # w_min > w_max
        ms = EWMarketBeta(
            half_life=5,
            shrinkage_group="industry",
            shrinkage_bounds=(0.8, 0.2),
        )
        with pytest.raises(ValueError, match="shrinkage_bounds"):
            ms.fit_transform(data_with_groups)

        # w_min < 0
        ms = EWMarketBeta(
            half_life=5,
            shrinkage_group="industry",
            shrinkage_bounds=(-0.1, 0.9),
        )
        with pytest.raises(ValueError, match="shrinkage_bounds"):
            ms.fit_transform(data_with_groups)

        # w_max > 1
        ms = EWMarketBeta(
            half_life=5,
            shrinkage_group="industry",
            shrinkage_bounds=(0.1, 1.1),
        )
        with pytest.raises(ValueError, match="shrinkage_bounds"):
            ms.fit_transform(data_with_groups)
