"""Tests for attribution uncertainty (standard errors and confidence intervals)."""

import numpy as np
import pytest

from skfolio.factor_model.attribution import (
    realized_factor_attribution,
    rolling_realized_factor_attribution,
)

from ._utils import _create_realized_model


def _create_uncertainty_model(
    n_obs=100,
    n_assets=5,
    n_factors=3,
    static_exposures=True,
    static_weights=True,
    seed=42,
    include_observations=False,
    gls=False,
):
    """Create test data with regression_weights and idio_variances.

    Parameters
    ----------
    gls : bool
        If True, regression_weights = 1 / idio_variances (GLS).
    """
    model = _create_realized_model(
        n_obs=n_obs,
        n_assets=n_assets,
        n_factors=n_factors,
        static_exposures=static_exposures,
        static_weights=static_weights,
        seed=seed,
        include_observations=include_observations,
    )

    rng = np.random.RandomState(seed + 100)
    idio_variances = np.abs(rng.randn(n_obs, n_assets)) * 0.001 + 0.0001

    if gls:
        regression_weights = 1.0 / idio_variances
    else:
        regression_weights = np.abs(rng.randn(n_obs, n_assets)) + 0.1

    model["regression_weights"] = regression_weights
    model["idio_variances"] = idio_variances
    model["compute_uncertainty"] = True
    return model


@pytest.fixture
def static_uncertainty_model():
    return _create_uncertainty_model(
        n_obs=100,
        n_assets=5,
        n_factors=3,
        static_exposures=True,
        static_weights=True,
    )


@pytest.fixture
def time_varying_uncertainty_model():
    return _create_uncertainty_model(
        n_obs=100,
        n_assets=5,
        n_factors=3,
        static_exposures=False,
        static_weights=False,
    )


@pytest.fixture
def rolling_uncertainty_model():
    return _create_uncertainty_model(
        n_obs=200,
        n_assets=5,
        n_factors=3,
        static_exposures=True,
        static_weights=True,
        include_observations=True,
    )


@pytest.fixture
def gls_uncertainty_model():
    return _create_uncertainty_model(
        n_obs=100,
        n_assets=5,
        n_factors=3,
        static_exposures=True,
        static_weights=True,
        gls=True,
    )


class TestAttributionUncertaintyPresence:
    """Tests that uncertainty fields are populated correctly."""

    def test_uncertainty_present_when_inputs_provided(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )

        assert result.systematic.mu_uncertainty is not None
        assert result.idio.mu_uncertainty is not None
        assert isinstance(result.systematic.mu_uncertainty, float)
        assert isinstance(result.idio.mu_uncertainty, float)
        assert result.factors.mu_contrib_uncertainty is not None
        assert isinstance(result.factors.mu_contrib_uncertainty, np.ndarray)

    def test_uncertainty_absent_when_inputs_missing(self, static_uncertainty_model):
        model = {
            k: v
            for k, v in static_uncertainty_model.items()
            if k not in ("regression_weights", "idio_variances", "compute_uncertainty")
        }
        result = realized_factor_attribution(**model, annualized_factor=1)

        assert result.systematic.mu_uncertainty is None
        assert result.idio.mu_uncertainty is None
        assert result.factors.mu_contrib_uncertainty is None

    def test_unexplained_and_total_have_no_uncertainty(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        assert result.unexplained.mu_uncertainty is None
        assert result.total.mu_uncertainty is None

    def test_families_uncertainty_present(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            annualized_factor=1,
        )
        assert result.families is not None
        assert result.families.mu_contrib_uncertainty is not None
        assert len(result.families.mu_contrib_uncertainty) == 2

    def test_families_uncertainty_absent_when_no_inputs(self, static_uncertainty_model):
        model = {
            k: v
            for k, v in static_uncertainty_model.items()
            if k not in ("regression_weights", "idio_variances", "compute_uncertainty")
        }
        result = realized_factor_attribution(
            **model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            annualized_factor=1,
        )
        assert result.families.mu_contrib_uncertainty is None


class TestAttributionUncertaintyMath:
    """Mathematical correctness of uncertainty computation."""

    def test_systematic_and_idio_se_are_equal(self, static_uncertainty_model):
        """Systematic and idio SE are equal (perfect negative correlation)."""
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        np.testing.assert_almost_equal(
            result.systematic.mu_uncertainty,
            result.idio.mu_uncertainty,
            decimal=14,
        )

    def test_per_factor_se_shape(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        n_factors = static_uncertainty_model["factor_returns"].shape[1]
        assert result.factors.mu_contrib_uncertainty.shape == (n_factors,)

    def test_se_positive(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        assert result.systematic.mu_uncertainty > 0
        assert result.idio.mu_uncertainty > 0
        assert np.all(result.factors.mu_contrib_uncertainty > 0)

    def test_total_se_leq_sum_of_factor_se(self, static_uncertainty_model):
        """Total systematic SE <= sum of per-factor SEs (triangle inequality)."""
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        assert result.systematic.mu_uncertainty <= np.sum(
            result.factors.mu_contrib_uncertainty
        )

    def test_family_se_consistent_with_factor_se(self, static_uncertainty_model):
        """Single-factor families should have SE equal to factor SE."""
        families = np.array(["A", "B", "C"])
        result = realized_factor_attribution(
            **static_uncertainty_model,
            factor_families=families,
            annualized_factor=1,
        )
        np.testing.assert_array_almost_equal(
            np.sort(result.families.mu_contrib_uncertainty),
            np.sort(result.factors.mu_contrib_uncertainty),
            decimal=12,
        )

    def test_se_scales_with_annualization(self, static_uncertainty_model):
        """SE should scale linearly with annualized_factor."""
        result_raw = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        result_ann = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=252
        )

        np.testing.assert_almost_equal(
            result_ann.systematic.mu_uncertainty,
            result_raw.systematic.mu_uncertainty * 252,
            decimal=10,
        )
        np.testing.assert_array_almost_equal(
            result_ann.factors.mu_contrib_uncertainty,
            result_raw.factors.mu_contrib_uncertainty * 252,
            decimal=10,
        )

    def test_se_decreases_with_more_obs(self):
        """SE should decrease with more observations (1/sqrt(T) scaling)."""
        model_short = _create_uncertainty_model(n_obs=100, seed=42)
        model_long = _create_uncertainty_model(n_obs=400, seed=42)

        result_short = realized_factor_attribution(**model_short, annualized_factor=1)
        result_long = realized_factor_attribution(**model_long, annualized_factor=1)

        assert result_long.systematic.mu_uncertainty < (
            result_short.systematic.mu_uncertainty
        )

    def test_gls_case_agrees_with_generic(self, gls_uncertainty_model):
        """GLS weights should give valid (positive, finite) SEs."""
        result = realized_factor_attribution(
            **gls_uncertainty_model, annualized_factor=1
        )
        assert result.systematic.mu_uncertainty > 0
        assert np.isfinite(result.systematic.mu_uncertainty)
        assert np.all(result.factors.mu_contrib_uncertainty > 0)
        assert np.all(np.isfinite(result.factors.mu_contrib_uncertainty))

    def test_time_varying_exposures(self, time_varying_uncertainty_model):
        """Uncertainty with time-varying (3D) exposures."""
        result = realized_factor_attribution(
            **time_varying_uncertainty_model, annualized_factor=1
        )
        assert result.systematic.mu_uncertainty is not None
        assert result.systematic.mu_uncertainty > 0
        assert result.idio.mu_uncertainty > 0

    def test_deterministic(self, static_uncertainty_model):
        """Uncertainty computation is deterministic."""
        r1 = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        r2 = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        np.testing.assert_equal(
            r1.systematic.mu_uncertainty, r2.systematic.mu_uncertainty
        )
        np.testing.assert_array_equal(
            r1.factors.mu_contrib_uncertainty,
            r2.factors.mu_contrib_uncertainty,
        )

    def test_brute_force_single_factor(self):
        """Compare SE with brute-force for a single-factor model."""
        rng = np.random.RandomState(0)
        T, n_assets, n_factors = 200, 3, 1

        B = rng.randn(n_assets, n_factors)
        w = np.array([0.5, 0.3, 0.2])
        f = rng.randn(T, n_factors) * 0.01
        eps = rng.randn(T, n_assets) * 0.005
        s2 = np.full((T, n_assets), 0.005**2)
        u = np.ones((T, n_assets))

        ptf_returns = (f @ B.T + eps) @ w

        result = realized_factor_attribution(
            factor_returns=f,
            portfolio_returns=ptf_returns,
            exposures=B,
            weights=w,
            idio_returns=eps,
            factor_names=np.array(["F1"]),
            asset_names=np.array(["A1", "A2", "A3"]),
            annualized_factor=1,
            regression_weights=u,
            idio_variances=s2,
            compute_uncertainty=True,
        )

        # Brute-force: Var(f_hat_t) = (B'WB)^-1 B'W Omega W B (B'WB)^-1
        # With W=I and Omega=sigma^2 I: simplifies to
        # sigma^2 (B'B)^-1 for each t
        BtB_inv = np.linalg.inv(B.T @ B)
        g = B.T @ w  # (k,)

        total_var = 0.0
        for t in range(T):
            meat_t = B.T @ np.diag(u[t] ** 2 * s2[t]) @ B
            Var_ft = BtB_inv @ meat_t @ BtB_inv
            total_var += float(g @ Var_ft @ g)

        expected_se = np.sqrt(total_var) / T

        np.testing.assert_almost_equal(
            result.systematic.mu_uncertainty, expected_se, decimal=12
        )


class TestAttributionUncertaintyDataFrame:
    """Tests for uncertainty columns in DataFrame output."""

    def test_factors_df_has_uncertainty_column(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        df = result.factors_df(formatted=False)
        assert "Mean Return Uncertainty" in df.columns

    def test_factors_df_no_uncertainty_without_inputs(self, static_uncertainty_model):
        model = {
            k: v
            for k, v in static_uncertainty_model.items()
            if k not in ("regression_weights", "idio_variances", "compute_uncertainty")
        }
        result = realized_factor_attribution(**model, annualized_factor=1)
        df = result.factors_df(formatted=False)
        assert "Mean Return Uncertainty" not in df.columns

    def test_factors_df_formatted_has_ci(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        df = result.factors_df(formatted=True)
        assert "Mean Return Uncertainty" not in df.columns
        merged = "Mean Return Contribution (95% CI)"
        assert merged in df.columns
        assert "Mean Return Contribution" not in df.columns
        cols = list(df.columns)
        i_pct_var = cols.index("% of Total Variance")
        assert cols.index(merged) == i_pct_var + 1
        sample = df.iloc[0][merged]
        assert " ± " in sample

    def test_factors_df_custom_confidence_level(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        df = result.factors_df(formatted=True, confidence_level=0.99)
        assert "Mean Return Contribution (99% CI)" in df.columns
        assert "Mean Return Contribution (95% CI)" not in df.columns

    def test_factors_df_no_ci_when_unformatted(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        df = result.factors_df(formatted=False)
        assert "Mean Return Contribution (95% CI)" not in df.columns

    def test_summary_df_has_uncertainty(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        df = result.summary_df(formatted=False)
        assert "Mean Return Uncertainty" in df.columns

        systematic_se = df.loc[
            df["Component"] == "Systematic", "Mean Return Uncertainty"
        ].iloc[0]
        idio_se = df.loc[
            df["Component"] == "Idiosyncratic", "Mean Return Uncertainty"
        ].iloc[0]
        total_se = df.loc[df["Component"] == "Total", "Mean Return Uncertainty"].iloc[0]

        assert systematic_se > 0
        assert idio_se > 0
        np.testing.assert_almost_equal(systematic_se, idio_se, decimal=14)
        assert np.isnan(total_se)

    def test_summary_df_formatted_has_ci(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        df = result.summary_df(formatted=True)
        merged = "Mean Return Contribution (95% CI)"
        assert merged in df.columns
        assert "Mean Return Uncertainty" not in df.columns
        assert "Mean Return Contribution" not in df.columns
        cols = list(df.columns)
        i_pct_var = cols.index("% of Total Variance")
        assert cols.index(merged) == i_pct_var + 1
        # Total and Unexplained: no SE, contribution only (no ± margin).
        total_cell = df.loc[df["Component"] == "Total", merged].iloc[0]
        assert " ± " not in total_cell
        assert "%" in total_cell
        if "Unexplained" in set(df["Component"]):
            unexplained_cell = df.loc[df["Component"] == "Unexplained", merged].iloc[0]
            assert " ± " not in unexplained_cell
        systematic_cell = df.loc[df["Component"] == "Systematic", merged].iloc[0]
        assert " ± " in systematic_cell

    def test_families_df_has_uncertainty(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            annualized_factor=1,
        )
        df = result.families_df(formatted=False)
        assert "Mean Return Uncertainty" in df.columns
        assert all(df["Mean Return Uncertainty"] > 0)


class TestAttributionUncertaintyRolling:
    """Tests for rolling attribution uncertainty."""

    def test_rolling_uncertainty_shape(self, rolling_uncertainty_model):
        result = rolling_realized_factor_attribution(
            **rolling_uncertainty_model,
            window_size=60,
            step=20,
            annualized_factor=1,
        )
        n_windows = len(result.observations)
        n_factors = rolling_uncertainty_model["factor_returns"].shape[1]

        assert isinstance(result.systematic.mu_uncertainty, np.ndarray)
        assert result.systematic.mu_uncertainty.shape == (n_windows,)
        assert result.factors.mu_contrib_uncertainty.shape == (
            n_windows,
            n_factors,
        )

    def test_rolling_systematic_equals_idio(self, rolling_uncertainty_model):
        result = rolling_realized_factor_attribution(
            **rolling_uncertainty_model,
            window_size=60,
            step=20,
            annualized_factor=1,
        )
        np.testing.assert_array_almost_equal(
            result.systematic.mu_uncertainty,
            result.idio.mu_uncertainty,
            decimal=14,
        )

    def test_rolling_uncertainty_all_positive(self, rolling_uncertainty_model):
        result = rolling_realized_factor_attribution(
            **rolling_uncertainty_model,
            window_size=60,
            step=20,
            annualized_factor=1,
        )
        assert np.all(result.systematic.mu_uncertainty > 0)
        assert np.all(result.factors.mu_contrib_uncertainty > 0)

    def test_rolling_matches_single_window(self, rolling_uncertainty_model):
        """First rolling window matches single-point attribution."""
        window_size = 60

        rolling_result = rolling_realized_factor_attribution(
            **rolling_uncertainty_model,
            window_size=window_size,
            step=30,
            annualized_factor=1,
        )

        single_result = realized_factor_attribution(
            factor_returns=rolling_uncertainty_model["factor_returns"][:window_size],
            portfolio_returns=rolling_uncertainty_model["portfolio_returns"][
                :window_size
            ],
            exposures=rolling_uncertainty_model["exposures"],
            weights=rolling_uncertainty_model["weights"],
            idio_returns=rolling_uncertainty_model["idio_returns"][:window_size],
            factor_names=rolling_uncertainty_model["factor_names"],
            asset_names=rolling_uncertainty_model["asset_names"],
            regression_weights=rolling_uncertainty_model["regression_weights"][
                :window_size
            ],
            idio_variances=rolling_uncertainty_model["idio_variances"][:window_size],
            compute_uncertainty=True,
            annualized_factor=1,
        )

        np.testing.assert_almost_equal(
            rolling_result.systematic.mu_uncertainty[0],
            single_result.systematic.mu_uncertainty,
            decimal=12,
        )
        np.testing.assert_array_almost_equal(
            rolling_result.factors.mu_contrib_uncertainty[0],
            single_result.factors.mu_contrib_uncertainty,
            decimal=12,
        )

    def test_rolling_df_multiindex_with_uncertainty(self, rolling_uncertainty_model):
        result = rolling_realized_factor_attribution(
            **rolling_uncertainty_model,
            window_size=60,
            step=30,
            annualized_factor=1,
        )
        df = result.factors_df(formatted=False)
        assert "Mean Return Uncertainty" in df.columns

    def test_rolling_summary_df_with_uncertainty(self, rolling_uncertainty_model):
        result = rolling_realized_factor_attribution(
            **rolling_uncertainty_model,
            window_size=60,
            step=30,
            annualized_factor=1,
        )
        df = result.summary_df(formatted=False)
        assert "Mean Return Uncertainty" in df.columns

    def test_rolling_families_uncertainty(self, rolling_uncertainty_model):
        result = rolling_realized_factor_attribution(
            **rolling_uncertainty_model,
            factor_families=np.array(["Style", "Style", "Risk"]),
            window_size=60,
            step=30,
            annualized_factor=1,
        )
        assert result.families.mu_contrib_uncertainty is not None
        n_windows = len(result.observations)
        assert result.families.mu_contrib_uncertainty.shape == (n_windows, 2)


class TestAttributionUncertaintyValidation:
    """Input validation for uncertainty parameters."""

    def test_compute_uncertainty_true_missing_both(self, static_uncertainty_model):
        model = {
            k: v
            for k, v in static_uncertainty_model.items()
            if k not in ("regression_weights", "idio_variances", "compute_uncertainty")
        }
        with pytest.raises(ValueError, match="compute_uncertainty=True"):
            realized_factor_attribution(
                **model, annualized_factor=1, compute_uncertainty=True
            )

    def test_compute_uncertainty_true_missing_idio_variances(
        self, static_uncertainty_model
    ):
        model = {
            k: v
            for k, v in static_uncertainty_model.items()
            if k not in ("idio_variances", "compute_uncertainty")
        }
        with pytest.raises(ValueError, match="compute_uncertainty=True"):
            realized_factor_attribution(
                **model, annualized_factor=1, compute_uncertainty=True
            )

    def test_compute_uncertainty_true_missing_regression_weights(
        self, static_uncertainty_model
    ):
        model = {
            k: v
            for k, v in static_uncertainty_model.items()
            if k not in ("regression_weights", "compute_uncertainty")
        }
        with pytest.raises(ValueError, match="compute_uncertainty=True"):
            realized_factor_attribution(
                **model, annualized_factor=1, compute_uncertainty=True
            )

    def test_rolling_compute_uncertainty_true_missing(self, rolling_uncertainty_model):
        model = {
            k: v
            for k, v in rolling_uncertainty_model.items()
            if k not in ("idio_variances", "compute_uncertainty")
        }
        with pytest.raises(ValueError, match="compute_uncertainty=True"):
            rolling_realized_factor_attribution(
                **model, window_size=60, step=20, compute_uncertainty=True
            )

    def test_compute_uncertainty_false_ignores_partial_inputs(
        self, static_uncertainty_model
    ):
        model = {
            k: v
            for k, v in static_uncertainty_model.items()
            if k not in ("idio_variances", "compute_uncertainty")
        }
        result = realized_factor_attribution(
            **model, annualized_factor=1, compute_uncertainty=False
        )
        assert result.systematic.mu_uncertainty is None

    def test_compute_uncertainty_false_ignores_both_inputs(
        self, static_uncertainty_model
    ):
        model = {
            k: v
            for k, v in static_uncertainty_model.items()
            if k != "compute_uncertainty"
        }
        result = realized_factor_attribution(
            **model,
            annualized_factor=1,
            compute_uncertainty=False,
        )
        assert result.systematic.mu_uncertainty is None

    def test_error_nan_in_regression_weights(self, static_uncertainty_model):
        model = {**static_uncertainty_model}
        model["regression_weights"] = model["regression_weights"].copy()
        model["regression_weights"][5, 2] = np.nan
        with pytest.raises(ValueError, match="`regression_weights` contains"):
            realized_factor_attribution(**model, annualized_factor=1)

    def test_rolling_error_nan_in_regression_weights(self, rolling_uncertainty_model):
        model = {**rolling_uncertainty_model}
        model["regression_weights"] = model["regression_weights"].copy()
        model["regression_weights"][10, 0] = np.nan
        with pytest.raises(ValueError, match="`regression_weights` contains"):
            rolling_realized_factor_attribution(**model, window_size=60, step=20)


class TestAttributionUncertaintyNaN:
    """Tests for NaN handling in uncertainty inputs."""

    def test_nan_idio_variances_zeroed(self):
        """NaN in idio_variances at inactive cells produces finite SEs."""
        model = _create_uncertainty_model(n_obs=50, n_assets=4, n_factors=2, seed=77)
        model["idio_variances"][3, 1] = np.nan
        model["idio_variances"][10, 1] = np.nan
        # Also mark the same asset as inactive in exposures/idio_returns
        model["exposures"][1, :] = np.nan  # static: asset 1 always inactive
        model["idio_returns"][:, 1] = np.nan

        result = realized_factor_attribution(**model, annualized_factor=1)

        assert result.systematic.mu_uncertainty is not None
        assert np.isfinite(result.systematic.mu_uncertainty)
        assert result.systematic.mu_uncertainty > 0
        assert np.all(np.isfinite(result.factors.mu_contrib_uncertainty))

    def test_nan_idio_variances_active_cell_raises(self):
        """NaN in idio_variances outside inactive cells raises."""
        model = _create_uncertainty_model(n_obs=50, n_assets=4, n_factors=2, seed=77)
        model["idio_variances"][10, 0] = np.nan

        with pytest.raises(ValueError, match="`idio_variances` contains"):
            realized_factor_attribution(**model, annualized_factor=1)

    def test_nan_idio_variances_3d_exposures(self):
        """NaN in idio_variances with 3D exposures produces finite SEs."""
        rng = np.random.RandomState(99)
        n_obs, n_assets, n_factors = 60, 6, 2

        factor_returns = rng.randn(n_obs, n_factors) * 0.01
        exposures = rng.randn(n_obs, n_assets, n_factors) * 0.5
        idio_returns = rng.randn(n_obs, n_assets) * 0.005
        weights_raw = np.abs(rng.randn(n_obs, n_assets)) + 0.1
        weights = weights_raw / weights_raw.sum(axis=1, keepdims=True)

        regression_weights = np.abs(rng.randn(n_obs, n_assets)) + 0.1
        idio_variances = np.abs(rng.randn(n_obs, n_assets)) * 0.001 + 0.0001

        # Inject NaN at some inactive cells
        nan_cells = [(5, 0), (10, 3), (20, 1)]
        for t, i in nan_cells:
            idio_returns[t, i] = np.nan
            idio_variances[t, i] = np.nan
            exposures[t, i, :] = np.nan

        lagged_exp = exposures[:-1]
        asset_returns = (
            np.einsum("tik,tk->ti", lagged_exp, factor_returns[1:]) + idio_returns[1:]
        )
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
            regression_weights=regression_weights,
            idio_variances=idio_variances,
            compute_uncertainty=True,
            annualized_factor=1,
        )

        assert np.isfinite(result.systematic.mu_uncertainty)
        assert result.systematic.mu_uncertainty > 0
        assert np.all(np.isfinite(result.factors.mu_contrib_uncertainty))

    def test_nan_idio_variances_equivalent_to_zero(self):
        """NaN idio_variances at inactive cells gives same SE as explicit 0."""
        base = _create_uncertainty_model(n_obs=50, n_assets=4, n_factors=2, seed=55)
        # Mark asset 2 as inactive via exposures and idio_returns
        base["exposures"][2, :] = np.nan
        base["idio_returns"][:, 2] = np.nan

        model_nan = {**base}
        model_nan["idio_variances"] = base["idio_variances"].copy()
        model_nan["idio_variances"][:, 2] = np.nan

        model_zero = {**base}
        model_zero["idio_variances"] = base["idio_variances"].copy()
        model_zero["idio_variances"][:, 2] = 0.0

        r_nan = realized_factor_attribution(**model_nan, annualized_factor=1)
        r_zero = realized_factor_attribution(**model_zero, annualized_factor=1)

        np.testing.assert_almost_equal(
            r_nan.systematic.mu_uncertainty,
            r_zero.systematic.mu_uncertainty,
            decimal=12,
        )
        np.testing.assert_array_almost_equal(
            r_nan.factors.mu_contrib_uncertainty,
            r_zero.factors.mu_contrib_uncertainty,
            decimal=12,
        )

    def test_rolling_nan_idio_variances(self):
        """NaN in idio_variances handled correctly in rolling attribution."""
        model = _create_uncertainty_model(
            n_obs=100,
            n_assets=4,
            n_factors=2,
            include_observations=True,
            seed=33,
        )
        model["idio_returns"][10, 0] = np.nan
        model["idio_returns"][50, 3] = np.nan
        model["idio_variances"][10, 0] = np.nan
        model["idio_variances"][50, 3] = np.nan

        result = rolling_realized_factor_attribution(
            **model, window_size=30, step=20, annualized_factor=1
        )

        assert np.all(np.isfinite(result.systematic.mu_uncertainty))
        assert np.all(result.systematic.mu_uncertainty > 0)
        assert np.all(np.isfinite(result.factors.mu_contrib_uncertainty))


class TestAttributionUncertaintyNumerical:
    """Tests for numerical robustness of uncertainty computation."""

    def test_ill_conditioned_no_math_domain_error(self):
        """Near-collinear exposures should not raise math domain error."""
        rng = np.random.RandomState(7)
        n_obs, n_assets, n_factors = 50, 10, 4

        factor_returns = rng.randn(n_obs, n_factors) * 0.01
        # Make two factors nearly collinear
        exposures = rng.randn(n_assets, n_factors) * 0.5
        exposures[:, 1] = exposures[:, 0] + rng.randn(n_assets) * 1e-6

        weights = np.ones(n_assets) / n_assets
        idio_returns = rng.randn(n_obs, n_assets) * 0.005
        ptf_returns = (factor_returns @ exposures.T + idio_returns) @ weights

        regression_weights = np.ones((n_obs, n_assets))
        idio_variances = np.full((n_obs, n_assets), 0.005**2)

        result = realized_factor_attribution(
            factor_returns=factor_returns,
            portfolio_returns=ptf_returns,
            exposures=exposures,
            weights=weights,
            idio_returns=idio_returns,
            factor_names=np.array([f"F{k}" for k in range(n_factors)]),
            asset_names=np.array([f"A{i}" for i in range(n_assets)]),
            regression_weights=regression_weights,
            idio_variances=idio_variances,
            compute_uncertainty=True,
            factor_families=np.array(["S", "S", "R", "R"]),
            annualized_factor=1,
        )

        assert np.isfinite(result.systematic.mu_uncertainty)
        assert result.systematic.mu_uncertainty >= 0
        assert np.all(np.isfinite(result.factors.mu_contrib_uncertainty))
        assert np.all(result.factors.mu_contrib_uncertainty >= 0)
        assert np.all(np.isfinite(result.families.mu_contrib_uncertainty))
        assert np.all(result.families.mu_contrib_uncertainty >= 0)


class TestAttributionUncertaintyBasketNeutral:
    """Tests for uncertainty with basket-neutral basis (industry factors)."""

    @staticmethod
    def _create_industry_model(n_obs=200, n_assets=20, seed=42):
        """Create a model with style + industry factors and a basket-neutral basis."""
        from skfolio.factor_model._family_constraint_basis import (
            compute_family_constraint_basis,
        )

        rng = np.random.RandomState(seed)
        n_style = 2
        n_industry = 4
        n_factors = n_style + n_industry

        # Style exposures: continuous
        style_exp = rng.randn(n_obs, n_assets, n_style) * 0.5

        # Industry exposures: binary partition (each asset in exactly one)
        ind_exp = np.zeros((n_obs, n_assets, n_industry))
        assignments = np.arange(n_assets) % n_industry
        for i in range(n_assets):
            ind_exp[:, i, assignments[i]] = 1.0

        exposures = np.concatenate([style_exp, ind_exp], axis=2)

        factor_names = np.array(
            [f"Style{k}" for k in range(n_style)]
            + [f"Ind{k}" for k in range(n_industry)]
        )
        factor_families = np.array(["Style"] * n_style + ["Industry"] * n_industry)

        # Factor returns
        factor_returns = rng.randn(n_obs, n_factors) * 0.01

        # Benchmark weights (equal weight)
        benchmark_weights = np.ones((n_obs, n_assets)) / n_assets

        # Build basket-neutral basis
        bnb, _ = compute_family_constraint_basis(
            constrained_families=[("Industry", None)],
            factor_exposures=exposures,
            benchmark_weights=benchmark_weights,
            factor_names=factor_names,
            factor_families=factor_families,
        )

        # Idiosyncratic
        idio_returns = rng.randn(n_obs, n_assets) * 0.005
        weights = np.ones(n_assets) / n_assets
        ptf_returns = np.sum(
            weights[np.newaxis, :]
            * (np.einsum("tik,tk->ti", exposures, factor_returns) + idio_returns),
            axis=1,
        )

        regression_weights = np.ones((n_obs, n_assets))
        idio_variances = np.full((n_obs, n_assets), 0.005**2)

        return dict(
            factor_returns=factor_returns,
            portfolio_returns=ptf_returns,
            exposures=exposures,
            weights=weights,
            idio_returns=idio_returns,
            factor_names=factor_names,
            asset_names=np.array([f"A{i}" for i in range(n_assets)]),
            factor_families=factor_families,
            regression_weights=regression_weights,
            idio_variances=idio_variances,
            bnb=bnb,
        )

    def test_without_basis_differs_from_with_basis(self):
        """Without basket-neutral basis, SEs differ from the basis result."""
        m = self._create_industry_model()
        common = dict(
            factor_returns=m["factor_returns"],
            portfolio_returns=m["portfolio_returns"],
            exposures=m["exposures"],
            weights=m["weights"],
            idio_returns=m["idio_returns"],
            factor_names=m["factor_names"],
            asset_names=m["asset_names"],
            factor_families=m["factor_families"],
            regression_weights=m["regression_weights"],
            idio_variances=m["idio_variances"],
            compute_uncertainty=True,
            annualized_factor=1,
            exposure_lag=0,
        )
        r_no_basis = realized_factor_attribution(**common)
        r_with_basis = realized_factor_attribution(
            **common, family_constraint_basis=m["bnb"]
        )
        # The basis changes the SE values (different conditioning)
        assert r_no_basis.systematic.mu_uncertainty != pytest.approx(
            r_with_basis.systematic.mu_uncertainty, rel=1e-6
        )

    def test_with_basis_produces_reasonable_values(self):
        """With basket-neutral basis, all SEs are finite and reasonable."""
        m = self._create_industry_model()
        result = realized_factor_attribution(
            factor_returns=m["factor_returns"],
            portfolio_returns=m["portfolio_returns"],
            exposures=m["exposures"],
            weights=m["weights"],
            idio_returns=m["idio_returns"],
            factor_names=m["factor_names"],
            asset_names=m["asset_names"],
            factor_families=m["factor_families"],
            regression_weights=m["regression_weights"],
            idio_variances=m["idio_variances"],
            compute_uncertainty=True,
            family_constraint_basis=m["bnb"],
            annualized_factor=1,
            exposure_lag=0,
        )
        assert np.isfinite(result.systematic.mu_uncertainty)
        assert result.systematic.mu_uncertainty > 0
        assert result.systematic.mu_uncertainty < 1.0

        assert np.all(np.isfinite(result.factors.mu_contrib_uncertainty))
        assert np.all(result.factors.mu_contrib_uncertainty >= 0)
        # All factor SEs should be reasonable (< 100%)
        assert np.all(result.factors.mu_contrib_uncertainty < 1.0)

    def test_basis_per_family_se_reasonable(self):
        """Per-family SEs are finite and reasonable with basis."""
        m = self._create_industry_model()
        result = realized_factor_attribution(
            factor_returns=m["factor_returns"],
            portfolio_returns=m["portfolio_returns"],
            exposures=m["exposures"],
            weights=m["weights"],
            idio_returns=m["idio_returns"],
            factor_names=m["factor_names"],
            asset_names=m["asset_names"],
            factor_families=m["factor_families"],
            regression_weights=m["regression_weights"],
            idio_variances=m["idio_variances"],
            compute_uncertainty=True,
            family_constraint_basis=m["bnb"],
            annualized_factor=1,
            exposure_lag=0,
        )
        assert result.families is not None
        assert np.all(np.isfinite(result.families.mu_contrib_uncertainty))
        assert np.all(result.families.mu_contrib_uncertainty >= 0)
        assert np.all(result.families.mu_contrib_uncertainty < 1.0)

    def test_basis_systematic_equals_idio(self):
        """Systematic and idiosyncratic SEs are equal with basis."""
        m = self._create_industry_model()
        result = realized_factor_attribution(
            factor_returns=m["factor_returns"],
            portfolio_returns=m["portfolio_returns"],
            exposures=m["exposures"],
            weights=m["weights"],
            idio_returns=m["idio_returns"],
            factor_names=m["factor_names"],
            asset_names=m["asset_names"],
            factor_families=m["factor_families"],
            regression_weights=m["regression_weights"],
            idio_variances=m["idio_variances"],
            compute_uncertainty=True,
            family_constraint_basis=m["bnb"],
            annualized_factor=1,
            exposure_lag=0,
        )
        np.testing.assert_almost_equal(
            result.systematic.mu_uncertainty,
            result.idio.mu_uncertainty,
            decimal=14,
        )

    def test_basis_does_not_affect_attribution_values(self):
        """The basis only affects uncertainty, not the attribution itself."""
        m = self._create_industry_model()
        common = dict(
            factor_returns=m["factor_returns"],
            portfolio_returns=m["portfolio_returns"],
            exposures=m["exposures"],
            weights=m["weights"],
            idio_returns=m["idio_returns"],
            factor_names=m["factor_names"],
            asset_names=m["asset_names"],
            factor_families=m["factor_families"],
            annualized_factor=1,
            exposure_lag=0,
        )
        r_no_basis = realized_factor_attribution(**common)
        r_with_basis = realized_factor_attribution(
            **common,
            regression_weights=m["regression_weights"],
            idio_variances=m["idio_variances"],
            compute_uncertainty=True,
            family_constraint_basis=m["bnb"],
        )
        np.testing.assert_almost_equal(
            r_no_basis.systematic.mu_contrib,
            r_with_basis.systematic.mu_contrib,
            decimal=14,
        )
        np.testing.assert_array_almost_equal(
            r_no_basis.factors.mu_contrib,
            r_with_basis.factors.mu_contrib,
            decimal=14,
        )


def _error_y_has_visible_interval(trace) -> bool:
    """True if a Bar trace has data-driven asymmetric error bars."""
    ey = trace.error_y
    if ey is None:
        return False
    arr = getattr(ey, "array", None)
    if arr is None:
        return False
    return any(a is not None for a in arr)


class TestPlotReturnContribUncertainty:
    """Return bar chart uncertainty: error bars (single-point) and hover."""

    def test_single_point_error_y_with_confidence_level(self, static_uncertainty_model):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        fig = result.plot_return_contrib(confidence_level=0.95)
        bar = fig.data[0]
        assert _error_y_has_visible_interval(bar)
        assert bar.error_y.type == "data"
        assert bar.hoverinfo == "text"
        ht = bar.hovertext[0]
        assert "(95% CI):" in ht
        assert " ± " in ht
        assert "Mean return SE" not in ht
        assert "<extra>" not in ht.lower()

    def test_single_point_no_error_y_when_confidence_level_none(
        self, static_uncertainty_model
    ):
        result = realized_factor_attribution(
            **static_uncertainty_model, annualized_factor=1
        )
        fig = result.plot_return_contrib(confidence_level=None)
        bar = fig.data[0]
        assert not _error_y_has_visible_interval(bar)
        assert bar.hoverinfo == "text"
        ht = bar.hovertext[0]
        assert "Mean return SE" in ht
        assert "(95% CI)" not in ht
        assert "<extra>" not in ht.lower()

    def test_rolling_grouped_no_error_y_hover_shows_ci(self, rolling_uncertainty_model):
        result = rolling_realized_factor_attribution(
            **rolling_uncertainty_model,
            window_size=60,
            step=30,
            annualized_factor=1,
        )
        fig = result.plot_return_contrib(confidence_level=0.95)
        assert fig.layout.barmode == "group"
        for tr in fig.data:
            assert not _error_y_has_visible_interval(tr)
        ht = fig.data[0].hovertext[0]
        assert "(95% CI):" in ht
        assert " ± " in ht
        assert "Mean return SE" not in ht
        assert "<extra>" not in ht.lower()

    def test_without_uncertainty_data_legacy_hover(self, static_realized_model):
        result = realized_factor_attribution(
            **static_realized_model, annualized_factor=1
        )
        fig = result.plot_return_contrib(confidence_level=0.95)
        bar = fig.data[0]
        assert not _error_y_has_visible_interval(bar)
        assert bar.hovertemplate is not None

    def test_predicted_attribution_legacy_hover(self, simple_factor_model_with_perf):
        from skfolio.factor_model.attribution import predicted_factor_attribution

        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        fig = result.plot_return_contrib(confidence_level=0.95)
        bar = fig.data[0]
        assert not _error_y_has_visible_interval(bar)
        assert bar.hovertemplate is not None
