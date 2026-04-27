import numpy as np
import pandas as pd
import pytest

from skfolio.factor_model.attribution import (
    Attribution,
    Component,
    FactorBreakdown,
    FamilyBreakdown,
    realized_factor_attribution,
    rolling_realized_factor_attribution,
)

from ._utils import _create_realized_model


class TestRealizedFactorAttribution:
    """Tests for realized_factor_attribution function."""

    # === Output Structure Tests ===

    def test_raw_output_structure(self, static_realized_model):
        """Test that raw output contains all expected keys with correct types."""
        result = realized_factor_attribution(**static_realized_model)

        assert isinstance(result, Attribution)

        # Component objects - always present
        for comp in [result.systematic, result.idio, result.unexplained, result.total]:
            assert isinstance(comp, Component)

        # Component fields - systematic
        assert isinstance(result.systematic.vol, float)
        assert isinstance(result.systematic.vol_contrib, float)
        assert isinstance(result.systematic.pct_total_variance, float)
        assert isinstance(result.systematic.mu, float)
        assert isinstance(result.systematic.pct_total_mu, float)
        assert isinstance(result.systematic.corr_with_ptf, float)

        # Component fields - idio
        assert isinstance(result.idio.vol, float)
        assert isinstance(result.idio.vol_contrib, float)
        assert isinstance(result.idio.pct_total_variance, float)
        assert isinstance(result.idio.corr_with_ptf, float)

        # Nested factors FactorBreakdown
        assert isinstance(result.factors, FactorBreakdown)
        assert isinstance(result.factors.names, np.ndarray)
        assert result.factors.family is None
        assert isinstance(result.factors.exposure_std, np.ndarray)

        # Families is None when not provided
        assert result.families is None

    def test_raw_output_with_families(self, static_realized_model):
        """Test that families Bunch is populated when factor_families provided."""
        result = realized_factor_attribution(
            **static_realized_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
        )

        assert isinstance(result.families, FamilyBreakdown)
        assert isinstance(result.families.names, np.ndarray)
        assert isinstance(result.families.vol_contrib, np.ndarray)

    def test_df_output_structure(self, static_realized_model):
        """Test DataFrame output methods work correctly."""
        result = realized_factor_attribution(**static_realized_model)
        factors_df = result.factors_df(formatted=False)
        summary_df = result.summary_df(formatted=False)

        assert isinstance(factors_df, pd.DataFrame)
        assert isinstance(summary_df, pd.DataFrame)

        # families_df raises when families not provided
        with pytest.raises(ValueError, match="requires `factor_families`"):
            result.families_df()

        # Check columns
        expected_factors_cols = [
            "Factor",
            "Exposure Mean",
            "Exposure Std",
            "Volatility Contribution",
            "% of Total Variance",
            "Mean Return Contribution",
            "% of Total Mean Return",
            "Standalone Volatility",
            "Standalone Mean Return",
            "Correlation with Portfolio",
        ]
        assert list(factors_df.columns) == expected_factors_cols

        # Realized attribution includes Unexplained row
        assert list(summary_df["Component"]) == [
            "Systematic",
            "Idiosyncratic",
            "Unexplained",
            "Total",
        ]

    def test_df_output_with_families(self, static_realized_model):
        """Test DataFrame output with families."""
        result = realized_factor_attribution(
            **static_realized_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
        )
        factors_df = result.factors_df(formatted=False)
        families_df = result.families_df(formatted=False)

        assert "Family" in factors_df.columns
        assert "Exposure Mean" in families_df.columns
        assert "Exposure Std" in families_df.columns

    # === Mathematical Correctness Tests ===

    def test_volatility_decomposition_additive(self, static_realized_model):
        """Test that volatility contributions sum to total volatility."""
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )

        sum_vol_contrib = np.sum(result.factors.vol_contrib) + result.idio.vol_contrib
        np.testing.assert_almost_equal(sum_vol_contrib, result.total.vol, decimal=10)

    def test_pct_total_variance_sums_to_100(self, static_realized_model):
        """Test that % of total variance sums to 100%."""
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )

        total_pct = (
            np.sum(result.factors.pct_total_variance) + result.idio.pct_total_variance
        )
        np.testing.assert_almost_equal(total_pct, 1.0, decimal=10)

    def test_factor_pct_total_variance_sums_to_systematic_share(
        self, static_realized_model
    ):
        """Test that factor % of total variance sums to systematic share."""
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )

        np.testing.assert_almost_equal(
            np.sum(result.factors.pct_total_variance),
            result.systematic.pct_total_variance,
            decimal=10,
        )

    def test_pure_factor_stats(self, static_realized_model):
        """Test that vol and corr_with_ptf are pure factor stats."""
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )

        # Verify vol are pure factor volatilities
        factor_returns = static_realized_model["factor_returns"]
        expected_vol = np.std(factor_returns, axis=0, ddof=1)
        np.testing.assert_array_almost_equal(
            result.factors.vol, expected_vol, decimal=10
        )

        # Verify corr_with_ptf is pure factor correlation
        portfolio_returns = static_realized_model["portfolio_returns"]
        for k in range(factor_returns.shape[1]):
            if result.factors.vol[k] > 0:
                expected_corr = np.corrcoef(factor_returns[:, k], portfolio_returns)[
                    0, 1
                ]
                np.testing.assert_almost_equal(
                    result.factors.corr_with_ptf[k], expected_corr, decimal=10
                )

    def test_mean_return_decomposition(self, static_realized_model):
        """Test that mean return contributions sum to total mean return."""
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )

        sum_mu = np.sum(result.factors.mu_contrib) + result.idio.mu
        np.testing.assert_almost_equal(sum_mu, result.total.mu, decimal=10)

    def test_systematic_mu_equals_sum_factor_contribs(self, static_realized_model):
        """Test that systematic.mu equals sum of factor mu_contrib."""
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )

        np.testing.assert_almost_equal(
            result.systematic.mu, np.sum(result.factors.mu_contrib), decimal=10
        )

    # === Time-Varying Tests ===

    def test_time_varying_decomposition_additive(self, time_varying_realized_model):
        """Test that decomposition is additive with time-varying inputs."""
        result = realized_factor_attribution(
            **time_varying_realized_model, annualized_factor=1
        )

        sum_vol_contrib = np.sum(result.factors.vol_contrib) + result.idio.vol_contrib
        np.testing.assert_almost_equal(sum_vol_contrib, result.total.vol, decimal=10)

    def test_time_varying_exposure_std_nonzero(self, time_varying_realized_model):
        """Test that exposure_std is nonzero for time-varying exposures."""
        result = realized_factor_attribution(
            **time_varying_realized_model, annualized_factor=1
        )
        assert np.all(result.factors.exposure_std > 0)

    def test_static_exposure_std_zero(self, static_realized_model):
        """Test that exposure_std is zero for static exposures."""
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )
        np.testing.assert_array_almost_equal(
            result.factors.exposure_std, np.zeros(3), decimal=10
        )

    # === Annualization Tests ===

    @pytest.mark.parametrize(
        "attr,scale_by_sqrt",
        [
            ("total.vol", True),
            ("total.mu", False),
        ],
    )
    def test_annualization_scaling(self, static_realized_model, attr, scale_by_sqrt):
        """Test that annualization correctly scales metrics."""
        result_raw = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )
        result_ann = realized_factor_attribution(
            **static_realized_model, annualized_factor=252
        )

        raw_obj = result_raw
        for name in attr.split("."):
            raw_obj = getattr(raw_obj, name)
        raw_val = raw_obj
        ann_obj = result_ann
        for name in attr.split("."):
            ann_obj = getattr(ann_obj, name)
        ann_val = ann_obj

        if scale_by_sqrt:
            np.testing.assert_almost_equal(ann_val, raw_val * np.sqrt(252))
        else:
            np.testing.assert_almost_equal(ann_val, raw_val * 252)

    # === Input Validation Tests ===

    @pytest.mark.parametrize(
        "invalid_input,param,error_match",
        [
            (np.random.randn(100), "factor_returns", "must be 2D"),
            (np.random.randn(100, 2), "portfolio_returns", "must be 1D"),
        ],
    )
    def test_error_invalid_shape(
        self, static_realized_model, invalid_input, param, error_match
    ):
        """Test error for invalid input shapes."""
        model = {**static_realized_model, param: invalid_input}
        with pytest.raises(ValueError, match=error_match):
            realized_factor_attribution(**model)

    def test_error_mismatched_n_obs(self, static_realized_model):
        """Test error for mismatched number of observations."""
        model = {**static_realized_model, "portfolio_returns": np.random.randn(50)}
        with pytest.raises(ValueError, match="does not match n_obs"):
            realized_factor_attribution(**model)

    def test_error_invalid_exposures_shape(self, static_realized_model):
        """Test error for invalid exposures shape."""
        model = {**static_realized_model, "exposures": np.random.randn(5)}
        with pytest.raises(ValueError, match=r"must be 2D.*or.*3D"):
            realized_factor_attribution(**model)

    def test_error_invalid_weights_shape(self, static_realized_model):
        """Test error for invalid weights shape."""
        model = {**static_realized_model, "weights": np.random.randn(100, 5, 2)}
        with pytest.raises(ValueError, match=r"must be 1D.*or 2D"):
            realized_factor_attribution(**model)

    def test_error_wrong_length_factor_names(self, static_realized_model):
        """Test error for wrong-length factor_names."""
        model = {**static_realized_model, "factor_names": np.array(["Only", "Two"])}
        with pytest.raises(ValueError, match="does not match n_factors"):
            realized_factor_attribution(**model)

    def test_error_wrong_length_factor_families(self, static_realized_model):
        """Test error for wrong-length factor_families."""
        with pytest.raises(ValueError, match="does not match n_factors"):
            realized_factor_attribution(
                **static_realized_model, factor_families=np.array(["Only"])
            )

    # === Unexplained Component Tests ===

    def test_unexplained_variance_with_misaligned_inputs(self, static_realized_model):
        """Test unexplained variance is populated for misaligned inputs."""
        np.random.seed(42)
        model = {
            **static_realized_model,
            "portfolio_returns": np.random.randn(100) * 0.02,
        }
        result = realized_factor_attribution(**model)

        assert result.unexplained is not None
        assert result.unexplained.pct_total_variance > 0.01

    def test_unexplained_near_zero_for_aligned_inputs(self, static_realized_model):
        """Test unexplained variance is near zero for properly aligned inputs."""
        result = realized_factor_attribution(**static_realized_model)

        assert result.unexplained is not None
        assert abs(result.unexplained.pct_total_variance) < 0.01

    # === Families Tests ===

    def test_families_aggregation(self, static_realized_model):
        """Test that families correctly aggregate factor contributions."""
        result = realized_factor_attribution(
            **static_realized_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
        )

        # Should have 2 families
        assert len(result.families.names) == 2

        # Family vol contribs should sum correctly
        np.testing.assert_almost_equal(
            np.sum(result.families.vol_contrib), result.systematic.vol_contrib
        )

    def test_families_pct_sums_to_systematic_share(self, static_realized_model):
        """Test that family % of total variance sums to systematic share."""
        result = realized_factor_attribution(
            **static_realized_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
        )

        np.testing.assert_almost_equal(
            np.sum(result.families.pct_total_variance),
            result.systematic.pct_total_variance,
            decimal=10,
        )

    # === Formatting Tests ===

    @pytest.mark.parametrize(
        "formatted,expected_dtype", [(True, object), (False, np.float64)]
    )
    def test_formatted_output(self, static_realized_model, formatted, expected_dtype):
        """Test that formatted parameter controls output type."""
        result = realized_factor_attribution(**static_realized_model)
        factors_df = result.factors_df(formatted=formatted)

        assert factors_df["Standalone Volatility"].dtype == expected_dtype

    # === Edge Cases ===

    def test_single_factor(self):
        """Test with a single factor."""
        model = _create_realized_model(
            n_obs=50, n_assets=3, n_factors=1, include_observations=False
        )
        result = realized_factor_attribution(**model, annualized_factor=1)

        assert len(result.factors.exposure) == 1
        assert result.systematic.vol > 0

    def test_accepts_list_inputs(self):
        """Test that function accepts Python lists."""
        model = _create_realized_model(
            n_obs=20, n_assets=2, n_factors=2, include_observations=False
        )
        # Convert arrays to lists
        list_model = {k: v.tolist() for k, v in model.items()}

        result = realized_factor_attribution(**list_model, annualized_factor=1)
        assert result.total.vol > 0


class TestNaNHandling:
    """Tests for NaN validation and zero-fill behaviour."""

    # --- Rejection of NaN in factor_returns / portfolio_returns / weights ---

    @pytest.mark.parametrize(
        "param", ["factor_returns", "portfolio_returns", "weights"]
    )
    def test_nan_in_non_nullable_raises(self, static_realized_model, param):
        """NaN in factor_returns, portfolio_returns, or weights raises ValueError."""
        model = {**static_realized_model}
        arr = np.array(model[param], dtype=float)
        arr.flat[0] = np.nan
        model[param] = arr
        with pytest.raises(ValueError, match=f"`{param}` contains"):
            realized_factor_attribution(**model)

    # --- Zero-fill of exposures / idio_returns (static) ---

    def test_nan_exposures_static_decomposition(self):
        """NaN in static exposures is filled with 0; decomposition remains exact."""
        model = _create_realized_model(
            n_obs=50, n_assets=4, n_factors=2, include_observations=False
        )
        model["exposures"][1, 0] = np.nan
        result = realized_factor_attribution(**model, annualized_factor=1)

        assert np.all(np.isfinite(result.total.vol))
        total_vol_contrib = (
            np.sum(result.factors.vol_contrib)
            + result.idio.vol_contrib
            + result.unexplained.vol_contrib
        )
        np.testing.assert_almost_equal(total_vol_contrib, result.total.vol, decimal=10)

    def test_nan_idio_returns_static_decomposition(self):
        """NaN in idio_returns is filled with 0; decomposition remains exact."""
        model = _create_realized_model(
            n_obs=50, n_assets=4, n_factors=2, include_observations=False
        )
        model["idio_returns"][3, 2] = np.nan
        result = realized_factor_attribution(**model, annualized_factor=1)

        assert np.all(np.isfinite(result.total.vol))
        total_vol_contrib = (
            np.sum(result.factors.vol_contrib)
            + result.idio.vol_contrib
            + result.unexplained.vol_contrib
        )
        np.testing.assert_almost_equal(total_vol_contrib, result.total.vol, decimal=10)

    # --- Zero-fill of exposures / idio_returns (time-varying) ---

    def test_nan_exposures_time_varying_decomposition(self):
        """NaN in 3D exposures is filled; decomposition remains exact."""
        model = _create_realized_model(
            n_obs=50,
            n_assets=4,
            n_factors=2,
            static_exposures=False,
            static_weights=False,
            include_observations=False,
        )
        model["exposures"][5, 1, :] = np.nan
        result = realized_factor_attribution(**model, annualized_factor=1)

        assert np.all(np.isfinite(result.total.vol))
        total_vol_contrib = (
            np.sum(result.factors.vol_contrib)
            + result.idio.vol_contrib
            + result.unexplained.vol_contrib
        )
        np.testing.assert_almost_equal(total_vol_contrib, result.total.vol, decimal=10)

    # --- Equivalence: NaN with zero weight == explicit zero ---

    def test_nan_equivalent_to_zero(self):
        """NaN exposure with zero weight gives the same result as explicit 0."""
        base = _create_realized_model(
            n_obs=50, n_assets=4, n_factors=2, include_observations=False
        )

        model_nan = {**base}
        model_nan["exposures"] = base["exposures"].copy()
        model_nan["idio_returns"] = base["idio_returns"].copy()
        model_nan["exposures"][2, :] = np.nan
        model_nan["idio_returns"][:, 2] = np.nan

        model_zero = {**base}
        model_zero["exposures"] = base["exposures"].copy()
        model_zero["idio_returns"] = base["idio_returns"].copy()
        model_zero["exposures"][2, :] = 0.0
        model_zero["idio_returns"][:, 2] = 0.0

        r_nan = realized_factor_attribution(**model_nan, annualized_factor=1)
        r_zero = realized_factor_attribution(**model_zero, annualized_factor=1)

        np.testing.assert_almost_equal(r_nan.total.vol, r_zero.total.vol, decimal=12)
        np.testing.assert_almost_equal(r_nan.total.mu, r_zero.total.mu, decimal=12)
        np.testing.assert_array_almost_equal(
            r_nan.factors.vol_contrib, r_zero.factors.vol_contrib, decimal=12
        )

    # --- Rolling: NaN handling propagates correctly ---

    def test_rolling_nan_idio_returns(self):
        """NaN in idio_returns is handled correctly in rolling attribution."""
        model = _create_realized_model(
            n_obs=100, n_assets=4, n_factors=2, include_observations=True
        )
        model["idio_returns"][10, 0] = np.nan
        model["idio_returns"][50, 3] = np.nan

        result = rolling_realized_factor_attribution(
            **model, window_size=30, step=20, annualized_factor=1
        )

        assert np.all(np.isfinite(result.total.vol))
        for i in range(len(result.observations)):
            total_vol_contrib = (
                np.sum(result.factors.vol_contrib[i])
                + result.idio.vol_contrib[i]
                + result.unexplained.vol_contrib[i]
            )
            np.testing.assert_almost_equal(
                total_vol_contrib, result.total.vol[i], decimal=10
            )

    # --- Unexplained is zero when portfolio uses nan_to_num (matching Portfolio) ---

    def test_unexplained_zero_with_nan_idio_3d_exposures(self):
        """Unexplained is zero when NaN idio cells use 0 in portfolio returns.

        Reproduces the real scenario: 3D exposures, 2D weights, some (date,
        asset) cells have NaN in idio_returns (missing actual return) while
        exposures are finite. The portfolio was built with nan_to_num(R, nan=0),
        so attribution must also attribute zero for those cells.
        """
        np.random.seed(99)
        n_obs, n_assets, n_factors = 60, 6, 2

        factor_returns = np.random.randn(n_obs, n_factors) * 0.01
        exposures = np.random.randn(n_obs, n_assets, n_factors) * 0.5
        idio_returns = np.random.randn(n_obs, n_assets) * 0.005
        weights_raw = np.abs(np.random.randn(n_obs, n_assets)) + 0.1
        weights = weights_raw / weights_raw.sum(axis=1, keepdims=True)

        # Inject NaN in idio (and implicitly in asset returns) at several cells
        # while keeping exposures finite — this is the scenario that triggered
        # the bug.
        nan_cells = [(5, 0), (10, 3), (20, 1), (30, 5), (40, 2)]
        for t, i in nan_cells:
            idio_returns[t, i] = np.nan

        # Build asset returns from the factor identity with lagged exposures:
        # R_i(t) = B_i(t-1) f(t) + eps_i(t)
        lagged_exp = exposures[:-1]
        asset_returns = (
            np.einsum("tik,tk->ti", lagged_exp, factor_returns[1:]) + idio_returns[1:]
        )

        # Portfolio returns use nan_to_num, matching Portfolio.__init__
        portfolio_returns = np.zeros(n_obs)
        portfolio_returns[1:] = np.sum(
            weights[1:] * np.nan_to_num(asset_returns, nan=0.0), axis=1
        )

        result = realized_factor_attribution(
            factor_returns=factor_returns,
            portfolio_returns=portfolio_returns,
            exposures=exposures,
            weights=weights,
            idio_returns=idio_returns,
            factor_names=np.array([f"F{k}" for k in range(n_factors)]),
            asset_names=np.array([f"A{i}" for i in range(n_assets)]),
            annualized_factor=1,
        )

        np.testing.assert_almost_equal(result.unexplained.mu_contrib, 0.0, decimal=12)
        np.testing.assert_almost_equal(result.unexplained.vol_contrib, 0.0, decimal=12)

    def test_unexplained_zero_with_nan_idio_static_weights(self):
        """Unexplained is zero with 3D exposures, static weights, and NaN idio."""
        np.random.seed(77)
        n_obs, n_assets, n_factors = 50, 4, 2

        factor_returns = np.random.randn(n_obs, n_factors) * 0.01
        exposures = np.random.randn(n_obs, n_assets, n_factors) * 0.5
        idio_returns = np.random.randn(n_obs, n_assets) * 0.005
        weights_raw = np.abs(np.random.randn(n_assets)) + 0.1
        weights = weights_raw / weights_raw.sum()

        nan_cells = [(3, 1), (15, 0), (25, 3)]
        for t, i in nan_cells:
            idio_returns[t, i] = np.nan

        lagged_exp = exposures[:-1]
        asset_returns = (
            np.einsum("tik,tk->ti", lagged_exp, factor_returns[1:]) + idio_returns[1:]
        )

        portfolio_returns = np.zeros(n_obs)
        portfolio_returns[1:] = np.sum(
            weights * np.nan_to_num(asset_returns, nan=0.0), axis=1
        )

        result = realized_factor_attribution(
            factor_returns=factor_returns,
            portfolio_returns=portfolio_returns,
            exposures=exposures,
            weights=weights,
            idio_returns=idio_returns,
            factor_names=np.array([f"F{k}" for k in range(n_factors)]),
            asset_names=np.array([f"A{i}" for i in range(n_assets)]),
            annualized_factor=1,
        )

        np.testing.assert_almost_equal(result.unexplained.mu_contrib, 0.0, decimal=12)
        np.testing.assert_almost_equal(result.unexplained.vol_contrib, 0.0, decimal=12)


class TestRealizedFactorAttributionRegression:
    """Exact regression tests for realized_factor_attribution.

    These tests use fixed seeds and expected values to catch any
    changes in the computation logic.
    """

    def test_exact_values_static_exposures(self):
        """Test exact values with static exposures (regression test)."""
        np.random.seed(12345)
        n_obs, n_assets, n_factors = 50, 3, 2

        factor_returns = np.random.randn(n_obs, n_factors) * 0.01
        exposures = np.array([[0.8, 0.3], [0.5, 0.7], [0.2, 0.4]])
        weights = np.array([0.5, 0.3, 0.2])
        idio_returns = np.random.randn(n_obs, n_assets) * 0.005

        x_t = exposures.T @ weights
        s = factor_returns * x_t
        u_P = idio_returns @ weights
        portfolio_returns = np.sum(s, axis=1) + u_P

        result = realized_factor_attribution(
            factor_returns=factor_returns,
            portfolio_returns=portfolio_returns,
            exposures=exposures,
            weights=weights,
            idio_returns=idio_returns,
            factor_names=np.array(["Factor1", "Factor2"]),
            asset_names=["A1", "A2", "A3"],
            annualized_factor=1,
        )

        # Expected values (computed once and hardcoded)
        np.testing.assert_almost_equal(
            result.factors.exposure, np.array([0.59, 0.44]), decimal=2
        )
        np.testing.assert_almost_equal(result.total.vol, 0.008348, decimal=5)

        # Verify decomposition is exact
        sum_vol = np.sum(result.factors.vol_contrib) + result.idio.vol_contrib
        np.testing.assert_almost_equal(sum_vol, result.total.vol, decimal=10)

    def test_deterministic_output(self, static_realized_model):
        """Test that output is deterministic for same inputs."""
        result1 = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )
        result2 = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )

        np.testing.assert_array_equal(
            result1.factors.vol_contrib, result2.factors.vol_contrib
        )
        np.testing.assert_equal(result1.total.vol, result2.total.vol)
        np.testing.assert_equal(result1.total.mu, result2.total.mu)


class TestRollingRealizedFactorAttribution:
    """Tests for rolling_realized_factor_attribution function."""

    # === Output Structure Tests ===

    def test_output_is_attribution(self, rolling_static_model):
        """Test that output is an Attribution object with is_rolling=True."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        assert isinstance(result, Attribution)
        assert result.is_rolling is True

    def test_observations_populated(self, rolling_static_model):
        """Test that observations is populated for rolling attribution."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        assert result.observations is not None
        assert isinstance(result.observations, np.ndarray)

    def test_number_of_windows(self, rolling_static_model):
        """Test correct number of windows is produced."""
        n_obs, window_size, step = 200, 60, 20

        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=window_size, step=step
        )

        expected_n_windows = len(np.arange(0, n_obs - window_size + 1, step))
        assert len(result.observations) == expected_n_windows

    def test_component_fields_are_1d_arrays(self, rolling_static_model):
        """Test that Component fields are 1D arrays for rolling attribution."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )

        for comp in [result.systematic, result.idio, result.unexplained, result.total]:
            for attr in [
                "vol",
                "vol_contrib",
                "pct_total_variance",
                "mu",
                "pct_total_mu",
            ]:
                val = getattr(comp, attr)
                assert isinstance(val, np.ndarray)
                assert val.ndim == 1

    def test_factor_breakdown_fields_are_2d_arrays(self, rolling_static_model):
        """Test that FactorBreakdown fields are 2D arrays for rolling attribution."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )

        n_windows = len(result.observations)
        n_factors = len(rolling_static_model["factor_names"])

        for attr in [
            "exposure",
            "vol",
            "vol_contrib",
            "pct_total_variance",
            "mu",
            "mu_contrib",
            "pct_total_mu",
            "exposure_std",
        ]:
            val = getattr(result.factors, attr)
            assert val.shape == (n_windows, n_factors)

    def test_factor_names_preserved(self, rolling_static_model):
        """Test that factor names are preserved (1D)."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20
        )
        np.testing.assert_array_equal(
            result.factors.names, rolling_static_model["factor_names"]
        )

    # === Window Parameters Tests ===

    @pytest.mark.parametrize(
        "step,expected_factor",
        [
            (1, lambda n, w: n - w + 1),  # step=1 produces max windows
            (50, lambda n, w: n // w),  # step=window_size produces non-overlapping
        ],
    )
    def test_window_count_by_step(self, rolling_static_model, step, expected_factor):
        """Test that window count is correct for different step sizes."""
        n_obs, window_size = 200, 50
        rolling_static_model["observations"] = np.arange(n_obs)  # Ensure correct n_obs

        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=window_size, step=step
        )

        expected = len(np.arange(0, n_obs - window_size + 1, step))
        assert len(result.observations) == expected

    def test_observation_labels_are_window_ends(self, rolling_static_model):
        """Test that observations contain the last observation of each window."""
        window_size, step = 60, 20

        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=window_size, step=step
        )

        # First window ends at index window_size - 1
        assert result.observations[0] == window_size - 1
        # Second window ends at index window_size - 1 + step
        assert result.observations[1] == window_size - 1 + step

    # === Mathematical Correctness Tests ===

    def test_each_window_matches_single_attribution(self, rolling_static_model):
        """Test that each window's result matches single realized_factor_attribution."""
        window_size, step = 60, 30

        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            window_size=window_size,
            step=step,
            annualized_factor=1,
        )

        # Verify first window
        single_result = realized_factor_attribution(
            factor_returns=rolling_static_model["factor_returns"][:window_size],
            portfolio_returns=rolling_static_model["portfolio_returns"][:window_size],
            exposures=rolling_static_model["exposures"],
            weights=rolling_static_model["weights"],
            idio_returns=rolling_static_model["idio_returns"][:window_size],
            factor_names=rolling_static_model["factor_names"],
            asset_names=rolling_static_model["asset_names"],
            annualized_factor=1,
        )

        np.testing.assert_almost_equal(result.total.vol[0], single_result.total.vol)
        np.testing.assert_almost_equal(result.total.mu[0], single_result.total.mu)
        np.testing.assert_array_almost_equal(
            result.factors.vol_contrib[0], single_result.factors.vol_contrib
        )

    def test_decomposition_additive_per_window(self, rolling_static_model):
        """Test that decomposition is additive for each window."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20, annualized_factor=1
        )

        for i in range(len(result.observations)):
            # Volatility decomposition
            sum_vol_contrib = (
                np.sum(result.factors.vol_contrib[i]) + result.idio.vol_contrib[i]
            )
            np.testing.assert_almost_equal(
                sum_vol_contrib, result.total.vol[i], decimal=10
            )

            # Return decomposition
            sum_mu = np.sum(result.factors.mu_contrib[i]) + result.idio.mu[i]
            np.testing.assert_almost_equal(sum_mu, result.total.mu[i], decimal=10)

            # Percentage shares
            total_pct = (
                np.sum(result.factors.pct_total_variance[i])
                + result.idio.pct_total_variance[i]
            )
            np.testing.assert_almost_equal(total_pct, 1.0, decimal=10)

    # === Annualization Tests ===

    @pytest.mark.parametrize(
        "attr,scale_sqrt",
        [
            ("total.vol", True),
            ("total.mu", False),
        ],
    )
    def test_annualization_scaling(self, rolling_static_model, attr, scale_sqrt):
        """Test that annualization correctly scales metrics."""
        result_raw = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30, annualized_factor=1
        )
        result_ann = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30, annualized_factor=252
        )

        raw_obj = result_raw
        for name in attr.split("."):
            raw_obj = getattr(raw_obj, name)
        raw_val = raw_obj
        ann_obj = result_ann
        for name in attr.split("."):
            ann_obj = getattr(ann_obj, name)
        ann_val = ann_obj

        if scale_sqrt:
            np.testing.assert_array_almost_equal(ann_val, raw_val * np.sqrt(252))
        else:
            np.testing.assert_array_almost_equal(ann_val, raw_val * 252)

    def test_pct_shares_unchanged_by_annualization(self, rolling_static_model):
        """Test that percentage shares are unchanged by annualization."""
        result_raw = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30, annualized_factor=1
        )
        result_ann = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30, annualized_factor=252
        )

        np.testing.assert_array_almost_equal(
            result_ann.systematic.pct_total_variance,
            result_raw.systematic.pct_total_variance,
        )

    # === Factor Families Tests ===

    def test_families_populated_when_provided(self, rolling_static_model):
        """Test that families FactorBreakdown is populated when factor_families provided."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            window_size=60,
            step=30,
        )

        assert result.families is not None
        assert isinstance(result.families, FamilyBreakdown)

    def test_families_vol_contrib_sum_to_systematic(self, rolling_static_model):
        """Test that family vol contribs sum to systematic vol contrib."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            window_size=60,
            step=30,
            annualized_factor=1,
        )

        for i in range(len(result.observations)):
            np.testing.assert_almost_equal(
                np.sum(result.families.vol_contrib[i]),
                result.systematic.vol_contrib[i],
                decimal=10,
            )

    # === DataFrame Output Tests ===

    def test_factors_df_multiindex(self, rolling_static_model):
        """Test that factors_df returns MultiIndex DataFrame for rolling."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30
        )
        factors_df = result.factors_df(formatted=False)

        assert isinstance(factors_df.index, pd.MultiIndex)
        assert factors_df.index.names == ["Observation", "Factor"]

    def test_summary_df_multiindex(self, rolling_static_model):
        """Test that summary_df returns MultiIndex DataFrame for rolling."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=30
        )
        summary_df = result.summary_df(formatted=False)

        assert isinstance(summary_df.index, pd.MultiIndex)
        assert summary_df.index.names == ["Observation", "Component"]

    # === Input Validation Tests ===

    @pytest.mark.parametrize(
        "window_size,step,error_match",
        [
            (500, 20, "exceeds n_obs"),
            (1, 1, "must be >= 2"),
            (60, 0, "must be >= 1"),
        ],
    )
    def test_error_invalid_window_params(
        self, rolling_static_model, window_size, step, error_match
    ):
        """Test error for invalid window parameters."""
        with pytest.raises(ValueError, match=error_match):
            rolling_realized_factor_attribution(
                **rolling_static_model, window_size=window_size, step=step
            )

    def test_error_observations_wrong_length(self, rolling_static_model):
        """Test error when observations has wrong length."""
        model = {**rolling_static_model, "observations": np.array(["a", "b", "c"])}
        with pytest.raises(ValueError, match="does not match"):
            rolling_realized_factor_attribution(**model, window_size=60, step=20)

    # === Edge Cases ===

    def test_single_window(self, rolling_static_model):
        """Test with window_size equal to n_obs (single window)."""
        n_obs = 200
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=n_obs, step=1
        )

        assert len(result.observations) == 1
        assert result.observations[0] == n_obs - 1

    def test_minimum_window_size(self, rolling_static_model):
        """Test with minimum window_size=2."""
        result = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=2, step=1
        )
        assert len(result.observations) == 200 - 2 + 1

    def test_single_factor(self):
        """Test rolling attribution with a single factor."""
        model = _create_realized_model(
            n_obs=100, n_assets=3, n_factors=1, include_observations=True
        )

        result = rolling_realized_factor_attribution(**model, window_size=30, step=10)

        assert result.factors.exposure.shape[1] == 1
        assert len(result.factors.names) == 1

    def test_accepts_list_inputs(self):
        """Test that function accepts Python lists."""
        model = _create_realized_model(
            n_obs=80, n_assets=2, n_factors=2, include_observations=True
        )
        list_model = {
            k: v.tolist() if hasattr(v, "tolist") else list(v) for k, v in model.items()
        }

        result = rolling_realized_factor_attribution(
            **list_model, window_size=30, step=10
        )

        assert result.is_rolling
        assert len(result.observations) > 0

    # === Determinism Tests ===

    def test_deterministic_output(self, rolling_static_model):
        """Test that output is deterministic for same inputs."""
        result1 = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20, annualized_factor=1
        )
        result2 = rolling_realized_factor_attribution(
            **rolling_static_model, window_size=60, step=20, annualized_factor=1
        )

        np.testing.assert_array_equal(result1.total.vol, result2.total.vol)
        np.testing.assert_array_equal(result1.total.mu, result2.total.mu)
        np.testing.assert_array_equal(
            result1.factors.vol_contrib, result2.factors.vol_contrib
        )
