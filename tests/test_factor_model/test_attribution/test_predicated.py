import warnings

import numpy as np
import pandas as pd
import pytest

from skfolio.factor_model.attribution import (
    Attribution,
    Component,
    FactorBreakdown,
    predicted_factor_attribution,
)


class TestPredictedFactorAttribution:
    """Tests for predicted_factor_attribution function."""

    # === Output Structure Tests ===

    def test_raw_output_structure_risk_only(self, simple_factor_model):
        """Test that raw output contains all expected keys with correct types."""
        result = predicted_factor_attribution(**simple_factor_model)

        assert isinstance(result, Attribution)

        # Component objects - always present
        assert isinstance(result.systematic, Component)
        assert isinstance(result.idio, Component)
        assert isinstance(result.total, Component)
        assert result.unexplained is None  # None for predicted attribution

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

        # Scalars - perf (defaults to 0 when factor_mu not provided)
        assert result.systematic.mu == 0.0
        assert result.idio.mu == 0.0
        assert result.total.mu == 0.0

        # Nested factors FactorBreakdown - always present
        assert isinstance(result.factors, FactorBreakdown)
        assert isinstance(result.factors.names, np.ndarray)
        assert result.factors.family is None  # None when not provided
        assert isinstance(result.factors.exposure, np.ndarray)
        assert isinstance(result.factors.vol, np.ndarray)
        assert isinstance(result.factors.corr_with_ptf, np.ndarray)
        assert isinstance(result.factors.vol_contrib, np.ndarray)
        assert isinstance(result.factors.pct_total_variance, np.ndarray)

        # Factors perf (defaults to zeros when factor_mu not provided)
        assert isinstance(result.factors.mu, np.ndarray)
        assert isinstance(result.factors.mu_contrib, np.ndarray)
        assert isinstance(result.factors.pct_total_mu, np.ndarray)
        np.testing.assert_array_equal(result.factors.mu, np.zeros(2) * 252)

        # Families FactorBreakdown - None when not provided
        assert result.families is None

    def test_raw_output_structure_with_perf(self, simple_factor_model_with_perf):
        """Test that raw output contains perf values when provided."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)

        # Component perf should be present
        assert isinstance(result.systematic.mu, float)
        assert isinstance(result.idio.mu, float)
        assert isinstance(result.total.mu, float)
        assert isinstance(result.systematic.pct_total_mu, float)
        assert isinstance(result.idio.pct_total_mu, float)

        # Factors perf should be present
        assert isinstance(result.factors.mu, np.ndarray)
        assert isinstance(result.factors.mu_contrib, np.ndarray)
        assert isinstance(result.factors.pct_total_mu, np.ndarray)

    # === DataFrame Output Tests ===

    def test_df_output_without_families_risk_only(self, simple_factor_model):
        """Test DataFrame output methods work correctly without families."""
        result = predicted_factor_attribution(**simple_factor_model)
        factors_df = result.factors_df(formatted=False)
        summary_df = result.summary_df(formatted=False)

        assert isinstance(factors_df, pd.DataFrame)
        assert isinstance(summary_df, pd.DataFrame)

        # families_df raises when families not provided
        with pytest.raises(ValueError, match="requires `factor_families`"):
            result.families_df()

        # No Family column when families not provided
        assert "Family" not in factors_df.columns

        # Columns for predicted attribution
        expected_factors_cols = [
            "Factor",
            "Exposure",
            "Volatility Contribution",
            "% of Total Variance",
            "Expected Return Contribution",
            "% of Total Expected Return",
            "Standalone Volatility",
            "Standalone Expected Return",
            "Correlation with Portfolio",
        ]
        expected_summary_cols = [
            "Component",
            "Volatility Contribution",
            "% of Total Variance",
            "Expected Return Contribution",
            "% of Total Expected Return",
        ]
        assert list(factors_df.columns) == expected_factors_cols
        assert list(summary_df.columns) == expected_summary_cols
        assert list(summary_df["Component"]) == ["Systematic", "Idiosyncratic", "Total"]

    def test_df_output_with_perf(self, simple_factor_model_with_perf):
        """Test DataFrame output includes perf columns when provided."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        factors_df = result.factors_df(formatted=False)
        summary_df = result.summary_df(formatted=False)

        # Check perf columns are present
        perf_cols = [
            "Standalone Expected Return",
            "Expected Return Contribution",
            "% of Total Expected Return",
        ]
        for col in perf_cols:
            assert col in factors_df.columns

        # Check summary has perf columns
        assert "Expected Return Contribution" in summary_df.columns
        assert "% of Total Expected Return" in summary_df.columns

    def test_df_output_with_families(self, multi_factor_model):
        """Test DataFrame output with families."""
        result = predicted_factor_attribution(**multi_factor_model)
        factors_df = result.factors_df(formatted=False)
        summary_df = result.summary_df(formatted=False)
        families_df = result.families_df(formatted=False)

        assert isinstance(factors_df, pd.DataFrame)
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(families_df, pd.DataFrame)

        # Family column present in factors_df
        assert "Family" in factors_df.columns

        # Families df has all columns
        expected_families_cols = [
            "Family",
            "Exposure",
            "Volatility Contribution",
            "% of Total Variance",
            "Expected Return Contribution",
            "% of Total Expected Return",
        ]
        assert list(families_df.columns) == expected_families_cols

    def test_df_output_with_families_and_perf(self, multi_factor_model_with_perf):
        """Test families DataFrame includes perf columns when provided."""
        result = predicted_factor_attribution(**multi_factor_model_with_perf)
        families_df = result.families_df(formatted=False)

        # Families df has perf columns
        assert "Expected Return Contribution" in families_df.columns
        assert "% of Total Expected Return" in families_df.columns

    # === Mathematical Correctness Tests (Risk) ===

    def test_variance_decomposition(self, simple_factor_model):
        """Test the core variance decomposition: total = systematic + idiosyncratic."""
        result = predicted_factor_attribution(
            **simple_factor_model, annualized_factor=1
        )

        B = simple_factor_model["loading_matrix"]
        F = simple_factor_model["factor_covariance"]
        w = simple_factor_model["weights"]
        D = np.diag(simple_factor_model["idio_covariance"])

        # Factor exposure: b = B.T @ w
        b = B.T @ w
        np.testing.assert_array_almost_equal(result.factors.exposure, b)

        # Systematic variance: b.T @ F @ b
        expected_systematic_var = b.T @ F @ b
        systematic_var = result.systematic.vol**2
        np.testing.assert_almost_equal(systematic_var, expected_systematic_var)

        # Idiosyncratic variance: w.T @ D @ w
        expected_idio_var = w.T @ D @ w
        idio_var = result.idio.vol**2
        np.testing.assert_almost_equal(idio_var, expected_idio_var)

        # Total variance = systematic + idiosyncratic
        total_var = result.total.vol**2
        np.testing.assert_almost_equal(total_var, systematic_var + idio_var)

        # Volatility = sqrt(variance)
        np.testing.assert_almost_equal(result.total.vol, np.sqrt(total_var))

    def test_euler_volatility_contributions(self, simple_factor_model):
        """Test Euler volatility contributions are additive."""
        result = predicted_factor_attribution(
            **simple_factor_model, annualized_factor=1
        )

        B = simple_factor_model["loading_matrix"]
        F = simple_factor_model["factor_covariance"]
        w = simple_factor_model["weights"]
        b = B.T @ w
        Fb = F @ b
        sigma_p = result.total.vol

        # Euler volatility contribution: b_k * (F @ b)_k / sigma_p
        expected_vol_contrib = b * Fb / sigma_p
        np.testing.assert_array_almost_equal(
            result.factors.vol_contrib, expected_vol_contrib
        )

        # Volatility contributions sum to systematic_variance / sigma_p
        systematic_var = result.systematic.vol**2
        expected_sum = systematic_var / sigma_p
        np.testing.assert_almost_equal(result.factors.vol_contrib.sum(), expected_sum)

    def test_summary_volatility_contribution_additive(self, simple_factor_model):
        """Test that summary volatility contributions sum to total volatility."""
        result = predicted_factor_attribution(**simple_factor_model)

        # Systematic + Idio = Total (Euler additivity)
        np.testing.assert_almost_equal(
            result.systematic.vol_contrib + result.idio.vol_contrib, result.total.vol
        )

    def test_pct_total_risk_from_factors_sums_to_systematic_share(
        self, simple_factor_model
    ):
        """Test that factor % of Total Variance sums to systematic share."""
        result = predicted_factor_attribution(**simple_factor_model)
        factors_df = result.factors_df(formatted=False)

        np.testing.assert_almost_equal(
            factors_df["% of Total Variance"].sum(),
            result.systematic.pct_total_variance,
        )

    def test_pct_total_variance_consistency(self, simple_factor_model):
        """Test that % of Total Variance is consistent between factors and summary."""
        result = predicted_factor_attribution(**simple_factor_model)
        factors_df = result.factors_df(formatted=False)
        summary_df = result.summary_df(formatted=False)

        expected_systematic_share = result.systematic.pct_total_variance
        np.testing.assert_almost_equal(
            factors_df["% of Total Variance"].sum(), expected_systematic_share
        )
        np.testing.assert_almost_equal(
            factors_df["% of Total Variance"].sum(),
            summary_df.loc[0, "% of Total Variance"],
        )

    def test_factor_correlations(self, simple_factor_model):
        """Test factor-total portfolio correlation calculation."""
        result = predicted_factor_attribution(
            **simple_factor_model, annualized_factor=1
        )

        B = simple_factor_model["loading_matrix"]
        F = simple_factor_model["factor_covariance"]
        w = simple_factor_model["weights"]
        b = B.T @ w
        Fb = F @ b
        factor_vol = np.sqrt(np.diag(F))
        sigma_p = result.total.vol

        # correlation_k = (F @ b)_k / (sigma_k * sigma_p)
        expected = Fb / (factor_vol * sigma_p)
        np.testing.assert_array_almost_equal(result.factors.corr_with_ptf, expected)

    def test_x_sigma_rho_identity(self, simple_factor_model):
        """Test that vol_contrib = exposure * factor_vol * correlation (x-sigma-rho)."""
        result = predicted_factor_attribution(
            **simple_factor_model, annualized_factor=1
        )

        # x-sigma-rho identity: VolContrib_k = b_k * sigma_k * rho_{k,p}
        expected_vol_contrib = (
            result.factors.exposure * result.factors.vol * result.factors.corr_with_ptf
        )
        np.testing.assert_array_almost_equal(
            result.factors.vol_contrib, expected_vol_contrib
        )

    # === Mathematical Correctness Tests (Perf) ===

    def test_expected_return_decomposition(self, simple_factor_model_with_perf):
        """Test the core expected return decomposition."""
        result = predicted_factor_attribution(
            **simple_factor_model_with_perf, annualized_factor=1
        )

        B = simple_factor_model_with_perf["loading_matrix"]
        w = simple_factor_model_with_perf["weights"]
        factor_mu = simple_factor_model_with_perf["factor_mu"]
        idio_mu = simple_factor_model_with_perf["idio_mu"]

        b = B.T @ w

        # Per-factor contribution: b_k * lambda_k
        expected_contribs = b * factor_mu
        np.testing.assert_array_almost_equal(
            result.factors.mu_contrib, expected_contribs
        )

        # Spanned expected return: b.T @ lambda
        expected_spanned = b.T @ factor_mu
        np.testing.assert_almost_equal(result.systematic.mu, expected_spanned)

        # Orthogonal expected return: w.T @ mu_perp
        expected_ortho = w.T @ idio_mu
        np.testing.assert_almost_equal(result.idio.mu, expected_ortho)

        # Total expected return = spanned + orthogonal
        np.testing.assert_almost_equal(
            result.total.mu, result.systematic.mu + result.idio.mu
        )

    def test_pct_total_mu_from_factors_sums_to_systematic_share(
        self, simple_factor_model_with_perf
    ):
        """Test that factor % of Total Expected Return sums to systematic share."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)
        factors_df = result.factors_df(formatted=False)

        np.testing.assert_almost_equal(
            factors_df["% of Total Expected Return"].sum(),
            result.systematic.pct_total_mu,
        )

    def test_summary_expected_return_additive(self, simple_factor_model_with_perf):
        """Test that summary expected returns sum correctly."""
        result = predicted_factor_attribution(**simple_factor_model_with_perf)

        # Systematic + Idio = Total
        np.testing.assert_almost_equal(
            result.systematic.mu + result.idio.mu, result.total.mu
        )

    def test_idio_mu_defaults_to_zeros(self, simple_factor_model):
        """Test that idio_mu defaults to zeros when only factor_mu provided."""
        result = predicted_factor_attribution(
            **simple_factor_model,
            factor_mu=np.array([0.05, 0.03]),
        )

        # Orthogonal contribution should be 0
        np.testing.assert_almost_equal(result.idio.mu, 0.0)
        # Total should equal systematic
        np.testing.assert_almost_equal(result.total.mu, result.systematic.mu)

    # === Annualization Tests ===

    def test_annualization_risk(self, simple_factor_model):
        """Test that annualization correctly scales variance and volatility."""
        result_raw = predicted_factor_attribution(
            **simple_factor_model, annualized_factor=1
        )
        result_ann = predicted_factor_attribution(
            **simple_factor_model, annualized_factor=252
        )

        # Variance (vol^2) scales by factor
        np.testing.assert_almost_equal(
            result_ann.total.vol**2, result_raw.total.vol**2 * 252
        )
        # Volatility scales by sqrt(factor)
        np.testing.assert_almost_equal(
            result_ann.total.vol, result_raw.total.vol * np.sqrt(252)
        )

    def test_annualization_perf(self, simple_factor_model_with_perf):
        """Test that annualization correctly scales expected returns."""
        result_raw = predicted_factor_attribution(
            **simple_factor_model_with_perf, annualized_factor=1
        )
        result_ann = predicted_factor_attribution(
            **simple_factor_model_with_perf, annualized_factor=252
        )

        # Expected returns scale by factor
        np.testing.assert_almost_equal(result_ann.total.mu, result_raw.total.mu * 252)
        np.testing.assert_almost_equal(
            result_ann.systematic.mu, result_raw.systematic.mu * 252
        )
        # Percentage shares should remain the same
        np.testing.assert_almost_equal(
            result_ann.systematic.pct_total_mu, result_raw.systematic.pct_total_mu
        )

    # === Input Handling Tests ===

    def test_1d_vs_2d_idio_covariance(self, simple_factor_model):
        """Test that 1D and equivalent 2D diagonal produce same results."""
        idio_1d = simple_factor_model["idio_covariance"]
        idio_2d = np.diag(idio_1d)

        result_1d = predicted_factor_attribution(**simple_factor_model)
        result_2d = predicted_factor_attribution(
            weights=simple_factor_model["weights"],
            loading_matrix=simple_factor_model["loading_matrix"],
            factor_covariance=simple_factor_model["factor_covariance"],
            idio_covariance=idio_2d,
            factor_names=simple_factor_model["factor_names"],
            asset_names=simple_factor_model["asset_names"],
        )

        np.testing.assert_almost_equal(result_1d.idio.vol, result_2d.idio.vol)

    def test_2d_idiosyncratic_with_off_diagonal(self, simple_factor_model):
        """Test 2D idiosyncratic covariance with off-diagonal correlations."""
        idio_2d = np.array(
            [
                [0.01, 0.002, 0.0],
                [0.002, 0.015, 0.001],
                [0.0, 0.001, 0.02],
            ]
        )

        result = predicted_factor_attribution(
            weights=simple_factor_model["weights"],
            loading_matrix=simple_factor_model["loading_matrix"],
            factor_covariance=simple_factor_model["factor_covariance"],
            idio_covariance=idio_2d,
            factor_names=simple_factor_model["factor_names"],
            asset_names=simple_factor_model["asset_names"],
            annualized_factor=252,
        )

        # Should produce different (higher) idio variance due to correlations
        w = simple_factor_model["weights"]
        expected_var = w.T @ idio_2d @ w * 252
        np.testing.assert_almost_equal(result.idio.vol**2, expected_var)

    def test_accepts_list_inputs(self):
        """Test that function accepts Python lists (not just numpy arrays)."""
        result = predicted_factor_attribution(
            weights=[0.4, 0.3, 0.3],
            loading_matrix=[[1.0, 0.5], [0.5, 1.0], [0.3, 0.2]],
            factor_covariance=[[0.04, 0.01], [0.01, 0.02]],
            idio_covariance=[0.01, 0.015, 0.02],
            factor_names=["F1", "F2"],
            asset_names=["A1", "A2", "A3"],
        )
        assert result.total.vol > 0

    # === Factor Names and Families Tests ===

    def test_factor_names_stored(self, simple_factor_model):
        """Test that factor names are correctly stored."""
        result = predicted_factor_attribution(**simple_factor_model)
        np.testing.assert_array_equal(result.factors.names, ["Momentum", "Value"])

    def test_factor_families_stored_when_provided(self, simple_factor_model):
        """Test that factor families are stored when provided."""
        result = predicted_factor_attribution(
            **simple_factor_model, factor_families=["Style", "Style"]
        )
        np.testing.assert_array_equal(result.factors.family, ["Style", "Style"])

    def test_factor_families_none_when_not_provided(self, simple_factor_model):
        """Test that factor_families is None when not provided."""
        result = predicted_factor_attribution(**simple_factor_model)
        assert result.factors.family is None

    # === DataFrame-Specific Tests ===

    def test_factors_df_sorted_by_abs_contribution(self, simple_factor_model):
        """Test that factors DataFrame is sorted by |% of Total Variance|."""
        result = predicted_factor_attribution(**simple_factor_model)
        factors_df = result.factors_df(formatted=False)

        abs_pct = factors_df["% of Total Variance"].abs().values
        assert np.all(abs_pct[:-1] >= abs_pct[1:])

    def test_summary_df_consistency(self, simple_factor_model):
        """Test summary DataFrame values are internally consistent."""
        result = predicted_factor_attribution(**simple_factor_model)
        summary_df = result.summary_df(formatted=False)

        # % of Total Variance sums to 1.0 (Systematic + Idiosyncratic)
        sys_pct = summary_df.loc[0, "% of Total Variance"]
        idio_pct = summary_df.loc[1, "% of Total Variance"]
        np.testing.assert_almost_equal(sys_pct + idio_pct, 1.0)

        # Total % of Total Variance is 1.0
        assert summary_df.loc[2, "% of Total Variance"] == 1.0

        # Volatility contribution for Systematic = systematic_variance / total_vol
        systematic_var = result.systematic.vol**2
        expected_sys_vol_contrib = systematic_var / result.total.vol
        np.testing.assert_almost_equal(
            summary_df.loc[0, "Volatility Contribution"], expected_sys_vol_contrib
        )

    def test_families_df_aggregation(self, multi_factor_model):
        """Test families DataFrame correctly aggregates factors."""
        result = predicted_factor_attribution(**multi_factor_model)
        families_df = result.families_df(formatted=False)

        # Two families
        assert len(families_df) == 2
        assert set(families_df["Family"]) == {"Style", "Industry"}

        # % of Total Variance should sum to systematic's share
        np.testing.assert_almost_equal(
            families_df["% of Total Variance"].sum(),
            result.systematic.pct_total_variance,
        )

    def test_families_df_with_perf(self, multi_factor_model_with_perf):
        """Test families DataFrame includes perf aggregation."""
        result = predicted_factor_attribution(**multi_factor_model_with_perf)
        families_df = result.families_df(formatted=False)

        # % of Total Expected Return should sum to systematic's share
        np.testing.assert_almost_equal(
            families_df["% of Total Expected Return"].sum(),
            result.systematic.pct_total_mu,
        )

    # === Error Handling Tests ===

    def test_error_invalid_weights(self, simple_factor_model):
        """Test error for invalid weights."""
        with pytest.raises(ValueError, match="`weights` must be 1D"):
            predicted_factor_attribution(
                weights=np.ones((3, 2)),
                loading_matrix=simple_factor_model["loading_matrix"],
                factor_covariance=simple_factor_model["factor_covariance"],
                idio_covariance=simple_factor_model["idio_covariance"],
                factor_names=simple_factor_model["factor_names"],
                asset_names=simple_factor_model["asset_names"],
            )

    @pytest.mark.parametrize(
        "invalid_loading,error_match",
        [
            (np.ones(6), "`loading_matrix` must be 2D"),
            (np.ones((5, 2)), "must have 3 rows"),
        ],
    )
    def test_error_invalid_loading_matrix(
        self, simple_factor_model, invalid_loading, error_match
    ):
        """Test errors for invalid loading_matrix."""
        with pytest.raises(ValueError, match=error_match):
            predicted_factor_attribution(
                weights=simple_factor_model["weights"],
                loading_matrix=invalid_loading,
                factor_covariance=simple_factor_model["factor_covariance"],
                idio_covariance=simple_factor_model["idio_covariance"],
                factor_names=simple_factor_model["factor_names"],
                asset_names=simple_factor_model["asset_names"],
            )

    @pytest.mark.parametrize(
        "invalid_factor_cov,error_match",
        [
            (np.ones((2, 3)), "square"),
            (np.eye(5), "does not match n_factors"),
        ],
    )
    def test_error_invalid_factor_covariance(
        self, simple_factor_model, invalid_factor_cov, error_match
    ):
        """Test errors for invalid factor_covariance."""
        with pytest.raises(ValueError, match=error_match):
            predicted_factor_attribution(
                weights=simple_factor_model["weights"],
                loading_matrix=simple_factor_model["loading_matrix"],
                factor_covariance=invalid_factor_cov,
                idio_covariance=simple_factor_model["idio_covariance"],
                factor_names=simple_factor_model["factor_names"],
                asset_names=simple_factor_model["asset_names"],
            )

    @pytest.mark.parametrize(
        "invalid_idio,error_match",
        [
            (np.ones(5), "does not match n_assets"),
            (np.eye(5), "does not match n_assets"),
            (np.ones((3, 3, 3)), "must be 1D or 2D"),
        ],
    )
    def test_error_invalid_idio_covariance(
        self, simple_factor_model, invalid_idio, error_match
    ):
        """Test errors for invalid idio_covariance."""
        with pytest.raises(ValueError, match=error_match):
            predicted_factor_attribution(
                weights=simple_factor_model["weights"],
                loading_matrix=simple_factor_model["loading_matrix"],
                factor_covariance=simple_factor_model["factor_covariance"],
                idio_covariance=invalid_idio,
                factor_names=simple_factor_model["factor_names"],
                asset_names=simple_factor_model["asset_names"],
            )

    def test_error_wrong_length_factor_names(self, simple_factor_model):
        """Test error for wrong-length factor_names."""
        with pytest.raises(ValueError, match="does not match n_factors"):
            predicted_factor_attribution(
                weights=simple_factor_model["weights"],
                loading_matrix=simple_factor_model["loading_matrix"],
                factor_covariance=simple_factor_model["factor_covariance"],
                idio_covariance=simple_factor_model["idio_covariance"],
                asset_names=simple_factor_model["asset_names"],
                factor_names=["Only one"],
            )

    def test_error_wrong_length_factor_families(self, simple_factor_model):
        """Test error for wrong-length factor_families."""
        with pytest.raises(ValueError, match="does not match n_factors"):
            predicted_factor_attribution(
                **simple_factor_model, factor_families=["Only one"]
            )

    def test_error_non_positive_variance(self):
        """Test error when total variance is non-positive."""
        with pytest.raises(ValueError, match="Non-positive total variance"):
            predicted_factor_attribution(
                weights=np.array([1.0, 0.0, 0.0]),
                loading_matrix=np.zeros((3, 2)),
                factor_covariance=np.eye(2) * 0.01,
                idio_covariance=np.zeros(3),
                factor_names=["F1", "F2"],
                asset_names=["A1", "A2", "A3"],
            )

    @pytest.mark.parametrize(
        "invalid_factor_mu,error_match",
        [
            (np.ones((2, 2)), "`factor_mu` must be 1D"),
            (np.ones(5), "does not match n_factors"),
        ],
    )
    def test_error_invalid_factor_mu(
        self, simple_factor_model, invalid_factor_mu, error_match
    ):
        """Test errors for invalid factor_mu."""
        with pytest.raises(ValueError, match=error_match):
            predicted_factor_attribution(
                **simple_factor_model, factor_mu=invalid_factor_mu
            )

    @pytest.mark.parametrize(
        "invalid_ortho_mu,error_match",
        [
            (np.ones((3, 2)), "`idio_mu` must be 1D"),
            (np.ones(5), "does not match n_assets"),
        ],
    )
    def test_error_invalid_idio_mu(
        self, simple_factor_model, invalid_ortho_mu, error_match
    ):
        """Test errors for invalid idio_mu."""
        with pytest.raises(ValueError, match=error_match):
            predicted_factor_attribution(
                **simple_factor_model,
                factor_mu=np.array([0.05, 0.03]),
                idio_mu=invalid_ortho_mu,
            )

    # === Edge Cases ===

    def test_single_factor(self):
        """Test with a single factor."""
        result = predicted_factor_attribution(
            weights=np.array([0.5, 0.5]),
            loading_matrix=np.array([[1.0], [0.8]]),
            factor_covariance=np.array([[0.04]]),
            idio_covariance=np.array([0.01, 0.01]),
            factor_names=["SingleFactor"],
            asset_names=["A1", "A2"],
        )

        assert len(result.factors.exposure) == 1
        assert result.systematic.vol > 0

    def test_single_asset(self):
        """Test with a single asset (weight=1)."""
        loading = np.array([[0.5, 0.3]])
        result = predicted_factor_attribution(
            weights=np.array([1.0]),
            loading_matrix=loading,
            factor_covariance=np.eye(2) * 0.02,
            idio_covariance=np.array([0.01]),
            factor_names=["F1", "F2"],
            asset_names=["A1"],
        )

        # Factor exposure equals loading for single asset with weight 1
        np.testing.assert_array_almost_equal(result.factors.exposure, loading[0])

    def test_zero_factor(self):
        """Test when loadings cancel out giving zero exposure to a factor."""
        result = predicted_factor_attribution(
            weights=np.array([0.5, 0.5]),
            loading_matrix=np.array([[1.0, 0.5], [-1.0, 0.5]]),  # cancel on factor 0
            factor_covariance=np.eye(2) * 0.04,
            idio_covariance=np.array([0.01, 0.01]),
            factor_names=["F1", "F2"],
            asset_names=["A1", "A2"],
        )

        np.testing.assert_almost_equal(result.factors.exposure[0], 0.0)
        np.testing.assert_almost_equal(result.factors.vol_contrib[0], 0.0)

    def test_factor_with_zero_variance(self):
        """Test handling of factor with zero variance (correlation undefined)."""
        result = predicted_factor_attribution(
            weights=np.array([0.5, 0.5]),
            loading_matrix=np.array([[1.0, 0.5], [0.8, 0.6]]),
            factor_covariance=np.array([[0.04, 0.0], [0.0, 0.0]]),  # factor 1 has 0 var
            idio_covariance=np.array([0.01, 0.01]),
            factor_names=["F1", "F2"],
            asset_names=["A1", "A2"],
        )

        assert result.factors.vol[1] == 0.0
        assert np.isnan(result.factors.corr_with_ptf[1])

    def test_zero_systematic_return(self):
        """Test handling when systematic expected return is zero."""
        result = predicted_factor_attribution(
            weights=np.array([0.5, 0.5]),
            loading_matrix=np.array([[1.0, 0.5], [0.8, 0.6]]),
            factor_covariance=np.eye(2) * 0.04,
            idio_covariance=np.array([0.01, 0.01]),
            factor_names=["F1", "F2"],
            asset_names=["A1", "A2"],
            factor_mu=np.array([0.0, 0.0]),  # zero factor returns
            idio_mu=np.array([0.01, 0.01]),
        )

        assert result.systematic.mu == 0.0
        assert result.systematic.pct_total_mu == 0.0

    def test_zero_total_return(self):
        """Test handling when total expected return is zero."""
        result = predicted_factor_attribution(
            weights=np.array([0.5, 0.5]),
            loading_matrix=np.array([[1.0, 0.5], [0.8, 0.6]]),
            factor_covariance=np.eye(2) * 0.04,
            idio_covariance=np.array([0.01, 0.01]),
            factor_names=["F1", "F2"],
            asset_names=["A1", "A2"],
            factor_mu=np.array([0.0, 0.0]),
            idio_mu=np.array([0.0, 0.0]),  # both zero
        )

        assert result.total.mu == 0.0
        assert np.isnan(result.systematic.pct_total_mu)
        assert np.isnan(result.idio.pct_total_mu)
        assert np.all(np.isnan(result.factors.pct_total_mu))

    def test_negative_expected_returns(self):
        """Test handling of negative expected returns."""
        result = predicted_factor_attribution(
            weights=np.array([0.5, 0.5]),
            loading_matrix=np.array([[1.0, 0.5], [0.8, 0.6]]),
            factor_covariance=np.eye(2) * 0.04,
            idio_covariance=np.array([0.01, 0.01]),
            factor_names=["F1", "F2"],
            asset_names=["A1", "A2"],
            factor_mu=np.array([-0.02, 0.05]),  # mix of positive/negative
            idio_mu=np.array([-0.01, 0.02]),
        )

        # Contributions can be negative
        assert result.factors.mu_contrib[0] < 0
        # Total should still be systematic + idio
        np.testing.assert_almost_equal(
            result.total.mu, result.systematic.mu + result.idio.mu
        )

    # === Formatting Tests ===

    def test_formatted_output(self, simple_factor_model_with_perf):
        """Test that formatted=True produces string values with readable names."""
        result = predicted_factor_attribution(
            **simple_factor_model_with_perf, factor_families=["Style", "Style"]
        )
        factors_df = result.factors_df(formatted=True)
        summary_df = result.summary_df(formatted=True)
        families_df = result.families_df(formatted=True)

        # Check formatted columns are strings ending with %
        for col in [
            "Standalone Volatility",
            "% of Total Variance",
            "Standalone Expected Return",
        ]:
            assert factors_df[col].dtype == object
            assert factors_df[col].iloc[0].endswith("%")

        # Check summary_df formatting
        assert summary_df["Volatility Contribution"].dtype == object
        assert summary_df["Volatility Contribution"].iloc[0].endswith("%")

        # Check families_df formatting
        assert families_df["Volatility Contribution"].dtype == object
        assert families_df["% of Total Variance"].iloc[0].endswith("%")

    def test_formatted_false_returns_floats(self, simple_factor_model):
        """Test that formatted=False returns float values."""
        result = predicted_factor_attribution(**simple_factor_model)
        factors_df = result.factors_df(formatted=False)
        summary_df = result.summary_df(formatted=False)

        assert factors_df["Standalone Volatility"].dtype == np.float64
        assert factors_df["% of Total Variance"].dtype == np.float64
        assert summary_df["Volatility Contribution"].dtype == np.float64


class TestPredictedNaNHandling:
    """Tests for NaN validation and zero-fill behaviour."""

    @pytest.fixture()
    def base_model(self):
        """5-asset, 2-factor model where asset 4 is non-investable (NaN)."""
        n_assets, n_factors = 5, 2
        loadings = np.random.default_rng(0).standard_normal((n_assets, n_factors))
        loadings[4, :] = np.nan
        idio_var = np.full(n_assets, 0.01)
        idio_var[4] = np.nan
        idio_mu = np.full(n_assets, 0.005)
        idio_mu[4] = np.nan
        weights = np.array([0.3, 0.3, 0.2, 0.2, 0.0])
        return dict(
            weights=weights,
            loading_matrix=loadings,
            factor_covariance=np.eye(n_factors) * 0.04,
            idio_covariance=idio_var,
            factor_names=np.array(["F0", "F1"]),
            asset_names=np.array(["A0", "A1", "A2", "A3", "A4"]),
            factor_mu=np.array([0.05, 0.03]),
            idio_mu=idio_mu,
        )

    # --- Rejection of NaN in always-finite inputs ---

    @pytest.mark.parametrize("param", ["weights", "factor_covariance", "factor_mu"])
    def test_nan_in_non_nullable_raises(self, simple_factor_model, param):
        """NaN in weights, factor_covariance, or factor_mu raises ValueError."""
        model = {**simple_factor_model, "factor_mu": np.array([0.05, 0.03])}
        arr = np.array(model[param], dtype=float)
        arr.flat[0] = np.nan
        model[param] = arr
        with pytest.raises(ValueError, match=f"`{param}` contains"):
            predicted_factor_attribution(**model)

    # --- Non-zero weight on non-investable asset warns ---

    def test_nonzero_weight_on_nan_asset_warns(self, base_model):
        """Non-zero weight on a NaN asset emits a warning and zeroes out."""
        base_model["weights"][4] = 0.1
        base_model["weights"][0] = 0.2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = predicted_factor_attribution(**base_model, annualized_factor=1)
            assert len(w) == 1
            assert "non-zero weight" in str(w[0].message)
            assert "A4" in str(w[0].message)
        assert np.all(np.isfinite(result.total.vol))

    # --- Zero-weight NaN assets: valid, decomposition exact ---

    def test_nan_assets_with_zero_weight_decomposition(self, base_model):
        """NaN non-investable assets with zero weight produce exact decomposition."""
        result = predicted_factor_attribution(**base_model, annualized_factor=1)

        assert np.all(np.isfinite(result.total.vol))
        np.testing.assert_almost_equal(
            result.systematic.vol_contrib + result.idio.vol_contrib,
            result.total.vol,
            decimal=10,
        )
        np.testing.assert_almost_equal(
            result.systematic.mu + result.idio.mu,
            result.total.mu,
            decimal=10,
        )

    def test_nan_equivalent_to_subset(self, base_model):
        """Full-universe model with NaN gives same result as clean subset."""
        result_full = predicted_factor_attribution(**base_model, annualized_factor=1)

        clean = {**base_model}
        inv = np.array([True, True, True, True, False])
        clean["weights"] = base_model["weights"][inv]
        clean["loading_matrix"] = base_model["loading_matrix"][inv]
        clean["idio_covariance"] = np.array([0.01, 0.01, 0.01, 0.01])
        clean["idio_mu"] = np.array([0.005, 0.005, 0.005, 0.005])
        clean["asset_names"] = base_model["asset_names"][inv]

        result_clean = predicted_factor_attribution(**clean, annualized_factor=1)

        np.testing.assert_almost_equal(
            result_full.total.vol, result_clean.total.vol, decimal=12
        )
        np.testing.assert_almost_equal(
            result_full.total.mu, result_clean.total.mu, decimal=12
        )
        np.testing.assert_array_almost_equal(
            result_full.factors.vol_contrib,
            result_clean.factors.vol_contrib,
            decimal=12,
        )

    # --- 2D idio_covariance with NaN ---

    def test_nan_2d_idio_covariance(self, base_model):
        """NaN in 2D idio_covariance for non-investable assets works."""
        idio_2d = np.diag(base_model["idio_covariance"])
        idio_2d[4, :] = np.nan
        idio_2d[:, 4] = np.nan
        base_model["idio_covariance"] = idio_2d
        result = predicted_factor_attribution(**base_model, annualized_factor=1)

        assert np.all(np.isfinite(result.total.vol))
        np.testing.assert_almost_equal(
            result.systematic.vol_contrib + result.idio.vol_contrib,
            result.total.vol,
            decimal=10,
        )
