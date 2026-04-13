import numpy as np
import pandas as pd
import pytest

from skfolio.moments import (
    EWCovariance,
    EmpiricalCovariance,
    RegimeAdjustedEWCovariance,
)
from skfolio.moments.covariance import RegimeAdjustmentMethod, RegimeAdjustmentTarget
from skfolio.utils.stats import cov_to_corr

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def X_synth(rng):
    """Small synthetic return matrix (200 obs, 5 assets), no NaN."""
    return rng.standard_normal((200, 5)) * 0.01


@pytest.fixture
def X_synth_wide(rng):
    """Wider universe (200 obs, 20 assets), no NaN."""
    return rng.standard_normal((200, 20)) * 0.01


# ---------------------------------------------------------------------------
# Original tests (use real SP500 X fixture from conftest.py)
# ---------------------------------------------------------------------------


class TestRegimeAdjustedEWCovariance:
    def test_basic(self, X):
        """Test basic functionality with auto-calibrated STVU."""
        model = RegimeAdjustedEWCovariance(corr_half_life=23)
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert hasattr(model, "regime_multiplier_")
        assert hasattr(model, "covariance_")
        assert 0.5 < model.regime_multiplier_ < 2.0
        assert np.all(np.isfinite(model.covariance_))
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_auto_stvu(self, X):
        model_auto = RegimeAdjustedEWCovariance(half_life=11)
        model_auto.fit(X)

        expected_regime_half_life = 0.5 * 11
        model_manual = RegimeAdjustedEWCovariance(
            half_life=11, regime_half_life=expected_regime_half_life
        )
        model_manual.fit(X)

        np.testing.assert_almost_equal(model_auto.covariance_, model_manual.covariance_)
        assert model_auto.regime_multiplier_ == model_manual.regime_multiplier_

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    def test_demeaning(self, X, regime_method):
        """Test RegimeAdjustedEWCovariance with EWMA de-meaning."""
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.MAHALANOBIS,
            regime_method=regime_method,
            half_life=23,
            assume_centered=False,
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    @pytest.mark.parametrize(
        "regime_target",
        [RegimeAdjustmentTarget.DIAGONAL, RegimeAdjustmentTarget.PORTFOLIO],
    )
    def test_demeaning_with_regime_target(self, X, regime_method, regime_target):
        """Test demeaning with non-default STVU targets."""
        model = RegimeAdjustedEWCovariance(
            regime_target=regime_target,
            regime_method=regime_method,
            half_life=23,
            assume_centered=False,
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    @pytest.mark.parametrize(
        "regime_target",
        [RegimeAdjustmentTarget.DIAGONAL, RegimeAdjustmentTarget.PORTFOLIO],
    )
    def test_stvu_params_with_regime_target(self, X, regime_method, regime_target):
        """Test custom STVU parameters and non-default targets."""
        model = RegimeAdjustedEWCovariance(
            regime_target=regime_target,
            regime_method=regime_method,
            half_life=11,
            regime_half_life=14,
            regime_min_observations=10,
            regime_multiplier_clip=(0.8, 1.5),
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert 0.8 <= model.regime_multiplier_ <= 1.5
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_insufficient_data(self):
        """Test with insufficient data (all below min_observations)."""
        X_small = np.random.randn(5, 3)
        model = RegimeAdjustedEWCovariance(half_life=23, regime_min_observations=100)
        model.fit(X_small)
        assert np.all(np.isnan(model.covariance_))

    def test_insufficient_data_regime_multiplier(self):
        """Test that regime_multiplier defaults to 1.0 with insufficient data."""
        X_small = np.random.randn(60, 3)
        model = RegimeAdjustedEWCovariance(half_life=23, regime_min_observations=100)
        model.fit(X_small)
        assert model.covariance_ is not None
        assert model.regime_multiplier_ == 1

    def test_invalid_regime_half_life(self):
        """Test that invalid regime_half_life values raise appropriate errors."""
        X_small = np.random.randn(50, 3)

        model = RegimeAdjustedEWCovariance(regime_half_life=0)
        with pytest.raises(
            ValueError,
            match=r"regime_half_life must be positive",
        ):
            model.fit(X_small)

        model = RegimeAdjustedEWCovariance(regime_half_life=-10)
        with pytest.raises(ValueError, match=r"regime_half_life must be positive"):
            model.fit(X_small)

        model = RegimeAdjustedEWCovariance(regime_half_life=200)
        with pytest.warns(UserWarning, match="very slow-moving STVU.*desynchronize"):
            model.fit(X_small)
        assert model.covariance_.shape == (3, 3)

    def test_no_clip(self, X):
        """Test with no STVU clipping."""
        model = RegimeAdjustedEWCovariance(
            corr_half_life=23, regime_multiplier_clip=None
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert model.regime_multiplier_ > 0

    def test_separate_var_decay(self, X):
        """Test with separate variance decay factor."""
        model_standard = RegimeAdjustedEWCovariance(corr_half_life=23)
        model_standard.fit(X)

        model_separate = RegimeAdjustedEWCovariance(
            half_life=23,
            corr_half_life=11,
        )
        model_separate.fit(X)

        assert model_standard.covariance_.shape == (20, 20)
        assert model_separate.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model_standard.covariance_))
        assert np.all(np.isfinite(model_separate.covariance_))

        assert not np.allclose(model_standard.covariance_, model_separate.covariance_)

        np.testing.assert_almost_equal(
            model_standard.covariance_, model_standard.covariance_.T
        )
        np.testing.assert_almost_equal(
            model_separate.covariance_, model_separate.covariance_.T
        )

        eigvals_standard = np.linalg.eigvalsh(model_standard.covariance_)
        eigvals_separate = np.linalg.eigvalsh(model_separate.covariance_)
        assert np.all(eigvals_standard >= -1e-10)
        assert np.all(eigvals_separate >= -1e-10)

    def test_corr_decay_with_demeaning(self, X):
        """Test separate variance decay with de-meaning."""
        model = RegimeAdjustedEWCovariance(
            half_life=23, corr_half_life=11, assume_centered=False
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))

        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_covariance_pure_calibration(self):
        """On IID N(0, Sigma*) with long burn-in, regime_multiplier_ should be ~1."""
        rng = np.random.default_rng(123)

        n_assets = 8
        n_burn = 10_000
        n_eval = 5_000

        A = rng.standard_normal((n_assets, n_assets))
        cov0 = A @ A.T
        target_std = rng.uniform(0.01, 0.03, size=n_assets)
        D = np.diag(target_std / np.sqrt(np.diag(cov0)))
        Sigma_true = D @ cov0 @ D
        Sigma_true.flat[:: n_assets + 1] += 1e-12 * float(np.mean(np.diag(Sigma_true)))

        X = rng.multivariate_normal(
            np.zeros(n_assets), Sigma_true, size=n_burn + n_eval
        )

        model = RegimeAdjustedEWCovariance(
            half_life=69,
            corr_half_life=77,
            regime_half_life=695,
            regime_min_observations=200,
        )
        with pytest.warns(UserWarning):
            model.fit(X)

        reg = model.regime_multiplier_
        assert reg is not None
        assert 0.97 <= reg <= 1.03, f"regime_multiplier_={reg:.4f} not ~ 1 on IID data"

    def test_regime_jump(self):
        """Test STVU response to regime change."""
        np.random.seed(43)
        n_obs, n_assets = 300, 8
        regime_change_at = 200
        vol_multiplier = 1.5

        true_corr = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                true_corr[i, j] = true_corr[j, i] = 0.4
        base_std = np.full(n_assets, 0.015)
        base_cov = np.outer(base_std, base_std) * true_corr

        X = np.random.multivariate_normal(np.zeros(n_assets), base_cov, size=n_obs)
        X[regime_change_at:] *= vol_multiplier

        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.MAHALANOBIS,
            half_life=23,
            regime_half_life=69,
            regime_min_observations=50,
            regime_multiplier_clip=(0.5, 2.0),
        )
        model.fit(X)

        assert model.regime_multiplier_ is not None
        assert 1.05 <= model.regime_multiplier_ <= 1.75, (
            f"STVU = {model.regime_multiplier_} didn't adapt to regime change"
        )

        fitted_corr, _ = cov_to_corr(model.covariance_)
        off_diag_corrs = fitted_corr[np.triu_indices(n_assets, k=1)]
        mean_corr = np.mean(off_diag_corrs)
        assert 0.25 <= mean_corr <= 0.55, (
            f"Mean correlation = {mean_corr} drifted from true value 0.4"
        )

    def test_separate_decay_dynamics(self):
        """Test separate variance/correlation decay dynamics."""
        np.random.seed(44)
        n_obs, n_assets = 400, 6
        vol_shock_at = 300

        true_corr = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                true_corr[i, j] = true_corr[j, i] = 0.5

        base_std = np.full(n_assets, 0.01)
        base_cov = np.outer(base_std, base_std) * true_corr
        X = np.random.multivariate_normal(np.zeros(n_assets), base_cov, size=n_obs)

        shocked_std = np.full(n_assets, 0.03)
        shocked_cov = np.outer(shocked_std, shocked_std) * true_corr
        X[vol_shock_at:] = np.random.multivariate_normal(
            np.zeros(n_assets), shocked_cov, size=n_obs - vol_shock_at
        )

        model_separate = RegimeAdjustedEWCovariance(
            half_life=23,
            corr_half_life=7,
        )
        model_separate.fit(X)

        model_uniform = RegimeAdjustedEWCovariance(
            half_life=11,
        )
        model_uniform.fit(X)

        corr_sep, std_sep = cov_to_corr(model_separate.covariance_)
        corr_uni, std_uni = cov_to_corr(model_uniform.covariance_)

        mean_vol_sep = np.mean(std_sep)
        mean_vol_uni = np.mean(std_uni)

        assert mean_vol_sep > mean_vol_uni * 0.95, (
            f"Separate decay variance ({mean_vol_sep:.4f}) should be >= uniform ({mean_vol_uni:.4f})"
        )

        off_diag_sep = corr_sep[np.triu_indices(n_assets, k=1)]
        off_diag_uni = corr_uni[np.triu_indices(n_assets, k=1)]

        mean_corr_sep = np.mean(off_diag_sep)
        mean_corr_uni = np.mean(off_diag_uni)

        assert 0.35 <= mean_corr_sep <= 0.65, (
            f"Separate decay correlation = {mean_corr_sep} drifted"
        )
        assert 0.35 <= mean_corr_uni <= 0.65, (
            f"Uniform decay correlation = {mean_corr_uni} drifted"
        )

        np.testing.assert_allclose(
            np.diag(corr_sep),
            np.ones(n_assets),
            rtol=1e-10,
            err_msg="DCC normalization failed: correlation diagonal != 1",
        )

    def test_exact_values(self, X):
        model = RegimeAdjustedEWCovariance(half_life=34)
        model.fit(X)

        model2 = RegimeAdjustedEWCovariance(
            half_life=34,
            corr_half_life=23,
        )
        model2.fit(X)

        model_ref = EWCovariance(half_life=50)
        model_ref.fit(X)

        dist = np.linalg.norm(model.covariance_ - model_ref.covariance_, ord="fro")
        assert dist < 0.001
        dist = np.linalg.norm(model.covariance_ - model2.covariance_, ord="fro")
        assert dist < 0.001

        model_emp = EmpiricalCovariance()
        model_emp.fit(X)

        dist = np.linalg.norm(model_ref.covariance_ - model_emp.covariance_, ord="fro")
        assert dist > 0.001

    def test_covariance_high_dimensional(self):
        """Test with many assets (100+) to verify numerical stability."""
        n_assets = 100
        n_obs = 500
        rng = np.random.default_rng(456)

        n_factors = 10
        factors = rng.standard_normal((n_obs, n_factors)) * 0.01
        loadings = rng.standard_normal((n_factors, n_assets))
        X = factors @ loadings

        model = RegimeAdjustedEWCovariance(half_life=23)
        with pytest.warns(
            UserWarning,
            match="The covariance matrix is not positive definite",
        ):
            model.fit(X)

        assert model.covariance_.shape == (n_assets, n_assets)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals > -1e-10), "Covariance not positive semi-definite"
        assert model.regime_multiplier_ > 0
        assert np.all(np.isfinite(model.covariance_))

        np.testing.assert_allclose(model.covariance_, model.covariance_.T, rtol=1e-10)

    def test_covariance_near_singular(self):
        """Test with nearly singular covariance."""
        n_assets = 5
        n_obs = 200
        rng = np.random.default_rng(789)

        common_factor = rng.standard_normal(n_obs) * 0.01
        idiosyncratic = rng.standard_normal((n_obs, n_assets)) * 0.0001
        X = np.column_stack([common_factor for _ in range(n_assets)]) + idiosyncratic

        model = RegimeAdjustedEWCovariance(half_life=11)
        model.fit(X)

        assert model.covariance_.shape == (n_assets, n_assets)
        assert np.all(np.isfinite(model.covariance_))

        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals > -1e-10)

        corr, _ = cov_to_corr(model.covariance_)
        off_diag = corr[np.triu_indices(n_assets, k=1)]
        assert np.mean(off_diag) > 0.95, "Expected high correlations"

    def test_covariance_window_size_demeaning_interaction(self):
        """Test interaction between corr_half_life and assume_centered=False."""
        n_obs = 300
        n_assets = 10
        rng = np.random.default_rng(101)

        mean_returns = rng.uniform(-0.001, 0.001, n_assets)
        X = rng.standard_normal((n_obs, n_assets)) * 0.01 + mean_returns

        model_windowed = RegimeAdjustedEWCovariance(half_life=11, assume_centered=False)
        model_windowed.fit(X)

        model_full = RegimeAdjustedEWCovariance(
            corr_half_life=11, assume_centered=False
        )
        model_full.fit(X)

        assert model_windowed.covariance_.shape == (n_assets, n_assets)
        assert model_full.covariance_.shape == (n_assets, n_assets)
        assert np.all(np.isfinite(model_windowed.covariance_))
        assert np.all(np.isfinite(model_full.covariance_))

        dist = np.linalg.norm(
            model_windowed.covariance_ - model_full.covariance_, ord="fro"
        )
        assert dist > 1e-6, "Different configs should affect results"

    def test_covariance_partial_fit_matches_batch(self, X):
        """Streaming updates should match batch fit results."""
        model_batch = RegimeAdjustedEWCovariance(corr_half_life=23)
        model_batch.fit(X)

        model_online = RegimeAdjustedEWCovariance(corr_half_life=23)
        chunk = 37
        for start in range(0, X.shape[0], chunk):
            model_online.partial_fit(X[start : start + chunk])

        np.testing.assert_allclose(
            model_online.covariance_,
            model_batch.covariance_,
            rtol=1e-10,
            atol=1e-12,
        )
        assert model_online.regime_multiplier_ == pytest.approx(
            model_batch.regime_multiplier_, rel=1e-10
        )

    def test_covariance_invalid_regime_multiplier_clip(self):
        """Test that invalid regime_multiplier_clip values raise appropriate errors."""
        X_small = np.random.randn(50, 3)

        model = RegimeAdjustedEWCovariance(regime_multiplier_clip=(1.5, 1.2))
        with pytest.raises(
            ValueError, match=r"regime_multiplier_clip must satisfy 0 < lo < hi"
        ):
            model.fit(X_small)

        model = RegimeAdjustedEWCovariance(regime_multiplier_clip=(-0.5, 1.5))
        with pytest.raises(
            ValueError, match=r"regime_multiplier_clip must satisfy 0 < lo < hi"
        ):
            model.fit(X_small)

        model = RegimeAdjustedEWCovariance(regime_multiplier_clip=(0.7, 1.5, 2.0))
        with pytest.raises(
            ValueError, match=r"regime_multiplier_clip must be a tuple of length 2"
        ):
            model.fit(X_small)

    def test_hac_basic(self, X):
        """Test basic HAC (Newey-West) functionality."""
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.MAHALANOBIS, half_life=23, hac_lags=5
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_hac_with_demeaning(self, X):
        """Test HAC with EWMA de-meaning."""
        model = RegimeAdjustedEWCovariance(
            half_life=23, hac_lags=3, assume_centered=False
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_hac_with_separate_decay(self, X):
        """Test HAC with separate variance/correlation half-lives."""
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.MAHALANOBIS,
            half_life=11,
            corr_half_life=23,
            hac_lags=3,
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    def test_hac_with_regime_methods(self, X, regime_method):
        """Test HAC with different STVU methods."""
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.MAHALANOBIS,
            regime_method=regime_method,
            half_life=23,
            hac_lags=5,
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    @pytest.mark.parametrize(
        "regime_target",
        [RegimeAdjustmentTarget.DIAGONAL, RegimeAdjustmentTarget.PORTFOLIO],
    )
    def test_hac_with_regime_methods_and_targets(self, X, regime_method, regime_target):
        """Test HAC with different STVU methods and non-default targets."""
        model = RegimeAdjustedEWCovariance(
            regime_target=regime_target,
            regime_method=regime_method,
            half_life=23,
            hac_lags=5,
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_hac_vs_no_hac(self, X):
        """Test that HAC produces different results than no HAC."""
        model_no_hac = RegimeAdjustedEWCovariance(half_life=23)
        model_no_hac.fit(X)

        model_hac = RegimeAdjustedEWCovariance(half_life=23, hac_lags=5)
        model_hac.fit(X)

        dist = np.linalg.norm(
            model_hac.covariance_ - model_no_hac.covariance_, ord="fro"
        )
        assert dist > 0.001, "HAC should produce different results"

    def test_hac_different_lags(self, X):
        """Test that different HAC lag values produce different results."""
        model_3 = RegimeAdjustedEWCovariance(half_life=23, hac_lags=3)
        model_3.fit(X)

        model_5 = RegimeAdjustedEWCovariance(half_life=23, hac_lags=5)
        model_5.fit(X)

        model_10 = RegimeAdjustedEWCovariance(half_life=23, hac_lags=10)
        model_10.fit(X)

        dist_3_5 = np.linalg.norm(model_3.covariance_ - model_5.covariance_, ord="fro")
        dist_5_10 = np.linalg.norm(
            model_5.covariance_ - model_10.covariance_, ord="fro"
        )

        assert dist_3_5 > 1e-6, "Different HAC lags should produce different results"
        assert dist_5_10 > 1e-6, "Different HAC lags should produce different results"

    def test_hac_covariance_properties(self, X):
        """Test that HAC-adjusted covariance has correct properties."""
        model = RegimeAdjustedEWCovariance(half_life=23, hac_lags=5)
        model.fit(X)

        np.testing.assert_array_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_hac_invalid_lags(self):
        """Test that invalid hac_lags values raise appropriate errors."""
        X_small = np.random.randn(50, 3)

        model = RegimeAdjustedEWCovariance(hac_lags=0)
        with pytest.raises(ValueError, match="hac_lags must be a positive integer"):
            model.fit(X_small)

        model = RegimeAdjustedEWCovariance(hac_lags=-1)
        with pytest.raises(ValueError, match="hac_lags must be a positive integer"):
            model.fit(X_small)

        model = RegimeAdjustedEWCovariance(hac_lags=2.5)
        with pytest.raises(ValueError, match="hac_lags must be a positive integer"):
            model.fit(X_small)

    def test_stvu_uses_pre_update_bias_correction(self):
        """STVU must use the pre-update covariance state for one-step-ahead scaling."""
        X = np.array([[1.0], [2.0]])
        model = RegimeAdjustedEWCovariance(
            half_life=2,
            min_observations=1,
            regime_min_observations=1,
            regime_multiplier_clip=None,
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            nearest=False,
        )
        model.fit(X)

        expected = 2.0 / np.sqrt(2.0 / np.pi)
        assert model.regime_multiplier_ == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        "regime_target",
        [
            RegimeAdjustmentTarget.MAHALANOBIS,
            RegimeAdjustmentTarget.DIAGONAL,
            RegimeAdjustmentTarget.PORTFOLIO,
        ],
    )
    def test_regime_targets_basic(self, X, regime_target):
        """Test basic functionality with different STVU targets."""
        model = RegimeAdjustedEWCovariance(
            regime_target=regime_target,
            half_life=23,
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_regime_targets_produce_different_results(self, X):
        """Test that different STVU targets produce different regime multipliers."""
        multipliers = {}
        for target in [
            RegimeAdjustmentTarget.MAHALANOBIS,
            RegimeAdjustmentTarget.DIAGONAL,
            RegimeAdjustmentTarget.PORTFOLIO,
        ]:
            model = RegimeAdjustedEWCovariance(
                regime_target=target,
                regime_method=RegimeAdjustmentMethod.FIRST_MOMENT,
                half_life=23,
                regime_multiplier_clip=None,
            )
            model.fit(X)
            multipliers[target] = model.regime_multiplier_

        assert (
            multipliers[RegimeAdjustmentTarget.MAHALANOBIS]
            != multipliers[RegimeAdjustmentTarget.DIAGONAL]
        )
        assert (
            multipliers[RegimeAdjustmentTarget.DIAGONAL]
            != multipliers[RegimeAdjustmentTarget.PORTFOLIO]
        )
        assert (
            multipliers[RegimeAdjustmentTarget.MAHALANOBIS]
            != multipliers[RegimeAdjustmentTarget.PORTFOLIO]
        )

    def test_regime_target_with_partial_fit(self, X):
        """Test STVU targets work correctly with partial_fit."""
        for target in [
            RegimeAdjustmentTarget.MAHALANOBIS,
            RegimeAdjustmentTarget.DIAGONAL,
            RegimeAdjustmentTarget.PORTFOLIO,
        ]:
            model_batch = RegimeAdjustedEWCovariance(
                regime_target=target,
                half_life=23,
            )
            model_batch.fit(X)

            model_online = RegimeAdjustedEWCovariance(
                regime_target=target,
                half_life=23,
            )
            chunk = 37
            for start in range(0, X.shape[0], chunk):
                model_online.partial_fit(X[start : start + chunk])

            np.testing.assert_allclose(
                model_online.covariance_,
                model_batch.covariance_,
                rtol=1e-10,
                atol=1e-12,
            )
            assert model_online.regime_multiplier_ == pytest.approx(
                model_batch.regime_multiplier_, rel=1e-10
            )

    def test_regime_target_with_separate_decay(self, X):
        """Test STVU targets work with separate variance/correlation decay."""
        for target in [
            RegimeAdjustmentTarget.MAHALANOBIS,
            RegimeAdjustmentTarget.DIAGONAL,
            RegimeAdjustmentTarget.PORTFOLIO,
        ]:
            model = RegimeAdjustedEWCovariance(
                regime_target=target,
                half_life=23,
                corr_half_life=11,
            )
            model.fit(X)
            assert model.covariance_.shape == (20, 20)
            assert np.all(np.isfinite(model.covariance_))
            assert model.regime_multiplier_ > 0

    def test_regime_target_enum_values(self):
        """Test STVU target enum values."""
        assert RegimeAdjustmentTarget.MAHALANOBIS.value == "mahalanobis"
        assert RegimeAdjustmentTarget.DIAGONAL.value == "diagonal"
        assert RegimeAdjustmentTarget.PORTFOLIO.value == "portfolio"

    def test_regime_portfolio_weights_2d_single_matches_1d(self, X):
        """2D weights with one row must match 1D weights."""
        w = np.ones(20) / 20
        model_1d = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            half_life=23,
            regime_portfolio_weights=w,
        )
        model_1d.fit(X)

        model_2d = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            half_life=23,
            regime_portfolio_weights=w[np.newaxis, :],
        )
        model_2d.fit(X)

        np.testing.assert_allclose(
            model_2d.covariance_, model_1d.covariance_, rtol=1e-12
        )
        assert model_2d.regime_multiplier_ == pytest.approx(
            model_1d.regime_multiplier_, rel=1e-12
        )

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    def test_regime_portfolio_weights_2d_multi(self, X, regime_method):
        """Multi-portfolio STVU produces valid results with all methods."""
        W = np.zeros((3, 20))
        W[0, :5] = 1.0 / 5
        W[1, 5:10] = 1.0 / 5
        W[2, :] = 1.0 / 20
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            regime_method=regime_method,
            half_life=23,
            regime_portfolio_weights=W,
        )
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        assert np.all(np.isfinite(model.covariance_))
        assert model.regime_multiplier_ > 0
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_regime_portfolio_weights_2d_partial_fit(self, X):
        """Multi-portfolio streaming matches batch."""
        W = np.zeros((2, 20))
        W[0, :10] = 1.0 / 10
        W[1, 10:] = 1.0 / 10
        model_batch = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            half_life=23,
            regime_portfolio_weights=W,
        )
        model_batch.fit(X)

        model_online = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            half_life=23,
            regime_portfolio_weights=W,
        )
        chunk = 37
        for start in range(0, X.shape[0], chunk):
            model_online.partial_fit(X[start : start + chunk])

        np.testing.assert_allclose(
            model_online.covariance_, model_batch.covariance_, rtol=1e-10
        )
        assert model_online.regime_multiplier_ == pytest.approx(
            model_batch.regime_multiplier_, rel=1e-10
        )

    def test_regime_portfolio_weights_2d_differs_from_single(self, X):
        """Multi-portfolio STVU differs from single-portfolio STVU."""
        w_single = np.ones(20) / 20
        W_multi = np.zeros((3, 20))
        W_multi[0, :7] = 1.0 / 7
        W_multi[1, 7:14] = 1.0 / 7
        W_multi[2, 14:] = 1.0 / 6

        model_single = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            half_life=23,
            regime_portfolio_weights=w_single,
            regime_multiplier_clip=None,
        )
        model_single.fit(X)

        model_multi = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            half_life=23,
            regime_portfolio_weights=W_multi,
            regime_multiplier_clip=None,
        )
        model_multi.fit(X)

        assert model_single.regime_multiplier_ != model_multi.regime_multiplier_

    def test_regime_portfolio_weights_validation_wrong_target(self):
        """regime_portfolio_weights with non-PORTFOLIO target raises."""
        X_small = np.random.randn(50, 3)
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.MAHALANOBIS,
            regime_portfolio_weights=np.ones(3) / 3,
        )
        with pytest.raises(ValueError, match="PORTFOLIO"):
            model.fit(X_small)

    def test_regime_portfolio_weights_validation_wrong_cols(self):
        """regime_portfolio_weights with wrong number of assets raises."""
        X_small = np.random.randn(50, 3)
        W = np.ones((2, 5)) / 5
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            regime_portfolio_weights=W,
        )
        with pytest.raises(ValueError, match="n_assets"):
            model.fit(X_small)

    def test_regime_portfolio_weights_validation_negative(self):
        """Negative weights raise."""
        X_small = np.random.randn(50, 3)
        W = np.array([[0.5, -0.3, 0.8]])
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            regime_portfolio_weights=W,
        )
        with pytest.raises(ValueError, match="non-negative"):
            model.fit(X_small)

    def test_regime_portfolio_weights_validation_zero_row(self):
        """Row with all zeros raises."""
        X_small = np.random.randn(50, 3)
        W = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 0.0]])
        model = RegimeAdjustedEWCovariance(
            regime_target=RegimeAdjustmentTarget.PORTFOLIO,
            regime_portfolio_weights=W,
        )
        with pytest.raises(ValueError, match="positive sum"):
            model.fit(X_small)

    def test_invalid_regime_target(self):
        """Invalid regime_target raises a clear ValueError."""
        X_small = np.random.randn(50, 3)
        model = RegimeAdjustedEWCovariance(regime_target="bad")
        with pytest.raises(
            ValueError, match="regime_target must be a RegimeAdjustmentTarget"
        ):
            model.fit(X_small)

    def test_invalid_regime_method(self):
        """Invalid regime_method raises a clear ValueError."""
        X_small = np.random.randn(50, 3)
        model = RegimeAdjustedEWCovariance(regime_method="bad")
        with pytest.raises(
            ValueError, match="regime_method must be a RegimeAdjustmentMethod"
        ):
            model.fit(X_small)


# ---------------------------------------------------------------------------
# Helper for NaN test classes
# ---------------------------------------------------------------------------

_DEFAULT_KWARGS = dict(half_life=10, min_observations=1)
"""Common kwargs for NaN tests: short half-life for fast convergence."""


def _fit(X, active_mask=None, estimation_mask=None, **extra):
    """Fit a RegimeAdjustedEWCovariance with NaN-test defaults."""
    kw = {**_DEFAULT_KWARGS, **extra}
    model = RegimeAdjustedEWCovariance(**kw)
    model.fit(X, active_mask=active_mask, estimation_mask=estimation_mask)
    return model


def _active_submatrix_psd(cov, active_idx):
    """Assert active submatrix is finite, symmetric, and PSD."""
    sub = cov[np.ix_(active_idx, active_idx)]
    assert not np.any(np.isnan(sub)), "Active submatrix contains NaN"
    np.testing.assert_array_almost_equal(sub, sub.T)
    eigvals = np.linalg.eigvalsh(sub)
    assert np.all(eigvals >= -1e-10), f"Active submatrix not PSD: {eigvals.min()}"


# ---------------------------------------------------------------------------
# NaN handling: late listing
# ---------------------------------------------------------------------------


class TestNaNLateListing:
    """Assets entering the universe mid-stream (NaN at beginning)."""

    def test_single_asset(self, X_synth):
        """Late-listed asset gets valid covariance after enough data."""
        X = X_synth.copy()
        X[:80, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:80, 3] = False

        model = _fit(X, active_mask=um)
        assert not np.any(np.isnan(model.covariance_))
        assert model.regime_multiplier_ > 0

    def test_covariance_differs_from_full_data(self, X_synth):
        """Late listing produces different estimate than full data."""
        model_full = _fit(X_synth)

        X = X_synth.copy()
        X[:100, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:100, 4] = False
        model_late = _fit(X, active_mask=um)

        np.testing.assert_array_almost_equal(
            model_full.covariance_[:4, :4],
            model_late.covariance_[:4, :4],
            decimal=3,
        )
        assert not np.allclose(
            model_full.covariance_[4, :], model_late.covariance_[4, :]
        )

    def test_staggered_multiple_assets(self, X_synth):
        """Multiple assets entering at different times."""
        X = X_synth.copy()
        X[:30, 2] = np.nan
        X[:80, 3] = np.nan
        X[:150, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:30, 2] = False
        um[:80, 3] = False
        um[:150, 4] = False

        model = _fit(X, active_mask=um)
        assert not np.any(np.isnan(model.covariance_))
        np.testing.assert_array_almost_equal(model.covariance_, model.covariance_.T)


# ---------------------------------------------------------------------------
# NaN handling: delisting
# ---------------------------------------------------------------------------


class TestNaNDelisting:
    """Assets leaving the universe (NaN at end, active_mask=False)."""

    def test_single_asset(self, X_synth):
        """Delisted asset has NaN row/col; active submatrix is PSD."""
        X = X_synth.copy()
        X[100:, 2] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 2] = False

        model = _fit(X, active_mask=um)

        assert np.all(np.isnan(model.covariance_[2, :]))
        assert np.all(np.isnan(model.covariance_[:, 2]))
        _active_submatrix_psd(model.covariance_, [0, 1, 3, 4])

    def test_multiple_assets(self, X_synth):
        """Multiple assets delisted at different times."""
        X = X_synth.copy()
        X[80:, 3] = np.nan
        X[120:, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[80:, 3] = False
        um[120:, 4] = False

        model = _fit(X, active_mask=um)
        assert np.all(np.isnan(model.covariance_[3, :]))
        assert np.all(np.isnan(model.covariance_[4, :]))
        assert not np.any(np.isnan(model.covariance_[np.ix_([0, 1, 2], [0, 1, 2])]))


# ---------------------------------------------------------------------------
# NaN handling: holidays (NaN in middle, active_mask=True)
# ---------------------------------------------------------------------------


class TestNaNHolidays:
    """Missing data for in-universe assets (covariance freezes, no NaN)."""

    def test_holiday_freeze_no_nan(self, X_synth):
        """Holiday NaN freezes covariance, no NaN in output."""
        X = X_synth.copy()
        X[80:90, 1] = np.nan

        model = _fit(X)
        assert not np.any(np.isnan(model.covariance_))

    def test_holiday_changes_affected_entries_only(self, X_synth):
        """Holiday changes cross-covariances of affected asset but not others."""
        model_full = _fit(X_synth)

        X = X_synth.copy()
        X[90:100, 2] = np.nan
        model_hol = _fit(X)

        assert not np.allclose(
            model_full.covariance_[2, :], model_hol.covariance_[2, :]
        )
        np.testing.assert_array_almost_equal(
            model_full.covariance_[np.ix_([0, 1], [0, 1])],
            model_hol.covariance_[np.ix_([0, 1], [0, 1])],
        )

    def test_holiday_vs_delisting_distinction(self, X_synth):
        """Same NaN pattern, different active_mask -> different output."""
        X = X_synth.copy()
        X[150:, 3] = np.nan

        model_hol = _fit(X)
        assert not np.any(np.isnan(model_hol.covariance_))

        um = np.ones_like(X, dtype=bool)
        um[150:, 3] = False
        model_del = _fit(X, active_mask=um)
        assert np.all(np.isnan(model_del.covariance_[3, :]))


# ---------------------------------------------------------------------------
# active_mask
# ---------------------------------------------------------------------------


class TestUniverseMask:
    """Validation and behavior of active_mask."""

    def test_shape_mismatch_raises(self, X_synth):
        um_bad = np.ones((X_synth.shape[0] + 1, X_synth.shape[1]), dtype=bool)
        with pytest.raises(ValueError, match="active_mask shape"):
            _fit(X_synth, active_mask=um_bad)

    def test_none_equivalent_to_all_true(self, X_synth):
        """active_mask=None behaves identically to all-True mask."""
        model_none = _fit(X_synth)
        um = np.ones_like(X_synth, dtype=bool)
        model_mask = _fit(X_synth, active_mask=um)

        np.testing.assert_array_almost_equal(
            model_none.covariance_, model_mask.covariance_
        )

    def test_valid_returns_ignored_outside_universe(self, X_synth):
        """Non-NaN returns with active_mask=False are still ignored."""
        um = np.ones_like(X_synth, dtype=bool)
        um[:, 4] = False

        model = _fit(X_synth, active_mask=um)
        assert np.all(np.isnan(model.covariance_[4, :]))


# ---------------------------------------------------------------------------
# min_observations warm-up
# ---------------------------------------------------------------------------


class TestMinObservations:
    """Warm-up behavior of min_observations."""

    def test_below_threshold_is_nan(self, X_synth):
        """Assets below min_observations have NaN covariance."""
        X = X_synth[:30].copy()
        X[:25, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:25, 4] = False

        model = _fit(X, active_mask=um, min_observations=10)

        assert np.all(np.isnan(model.covariance_[4, :]))
        assert np.all(np.isnan(model.covariance_[:, 4]))
        assert not np.any(
            np.isnan(model.covariance_[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])])
        )

    def test_crosses_threshold(self, X_synth):
        """Asset becomes active once it crosses min_observations."""
        X = X_synth.copy()
        X[:180, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:180, 4] = False

        model = _fit(X, active_mask=um, min_observations=15)
        assert not np.any(np.isnan(model.covariance_))

    def test_all_below_threshold(self, X_synth):
        """All assets below threshold -> all NaN covariance."""
        model = _fit(X_synth[:5], min_observations=100)
        assert np.all(np.isnan(model.covariance_))

    def test_validation(self, X_synth):
        """min_observations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_observations must be >= 1"):
            _fit(X_synth, min_observations=0)

    def test_default_equals_half_life(self, X_synth):
        """Default min_observations equals int(half_life)."""
        model = RegimeAdjustedEWCovariance(half_life=40, min_observations=None)
        model.fit(X_synth)
        assert not np.any(np.isnan(model.covariance_))

    def test_tracked_across_partial_fit(self, X_synth):
        """Warm-up threshold is tracked across partial_fit calls."""
        X = X_synth.copy()
        X[:, 4] = np.nan
        um = np.ones((X_synth.shape[0], X_synth.shape[1]), dtype=bool)
        um[:, 4] = False

        model = RegimeAdjustedEWCovariance(half_life=10, min_observations=5)

        model.partial_fit(X[:50], active_mask=um[:50])
        assert np.all(np.isnan(model.covariance_[4, :]))

        model.partial_fit(X_synth[50:53])
        assert np.all(np.isnan(model.covariance_[4, :]))

        model.partial_fit(X_synth[53:60])
        assert not np.any(np.isnan(model.covariance_[4, :]))


# ---------------------------------------------------------------------------
# partial_fit streaming with NaN
# ---------------------------------------------------------------------------


class TestPartialFitNaN:
    """Streaming/online updates via partial_fit with NaN."""

    def test_nan_matches_batch(self, X_synth):
        """partial_fit with NaN should match batch fit."""
        X = X_synth.copy()
        X[:50, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:50, 3] = False

        model_batch = _fit(X, active_mask=um)

        model_stream = RegimeAdjustedEWCovariance(**_DEFAULT_KWARGS)
        model_stream.partial_fit(X[:80], active_mask=um[:80])
        model_stream.partial_fit(X[80:], active_mask=um[80:])

        np.testing.assert_array_almost_equal(
            model_batch.covariance_, model_stream.covariance_
        )

    def test_asset_leaves_universe(self, X_synth):
        """Asset leaving universe across partial_fit calls."""
        model = RegimeAdjustedEWCovariance(**_DEFAULT_KWARGS)

        model.partial_fit(X_synth[:100])
        assert not np.any(np.isnan(model.covariance_))

        X_b2 = X_synth[100:].copy()
        X_b2[:, 2] = np.nan
        um_b2 = np.ones_like(X_b2, dtype=bool)
        um_b2[:, 2] = False
        model.partial_fit(X_b2, active_mask=um_b2)
        assert np.all(np.isnan(model.covariance_[2, :]))


# ---------------------------------------------------------------------------
# assume_centered=False with NaN
# ---------------------------------------------------------------------------


class TestAssumeCenteredFalseNaN:
    """Tests for assume_centered=False (EWMA mean tracking) with NaN."""

    def test_location_nan_for_delisted(self, X_synth):
        X = X_synth.copy()
        X[100:, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 3] = False

        model = _fit(X, active_mask=um, assume_centered=False)

        assert np.isnan(model.location_[3])
        assert np.all(np.isfinite(model.location_[[0, 1, 2, 4]]))

    def test_location_frozen_during_holiday(self, X_synth):
        X = X_synth.copy()
        X[180:190, 1] = np.nan

        model = _fit(X, assume_centered=False)
        assert np.all(np.isfinite(model.location_))

    def test_location_zeros_when_centered(self, X_synth):
        """location_ is always zeros when assume_centered=True."""
        X = X_synth.copy()
        X[100:, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 3] = False

        model = _fit(X, active_mask=um)
        np.testing.assert_array_equal(model.location_, np.zeros(X_synth.shape[1]))

    def test_demeaned_covariance_psd(self, X_synth):
        """Covariance with demeaning and late listing is PSD."""
        X = X_synth.copy()
        X[:50, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:50, 4] = False

        model = _fit(X, active_mask=um, assume_centered=False)
        assert not np.any(np.isnan(model.covariance_))
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)


# ---------------------------------------------------------------------------
# Inference on NaN-aware covariance
# ---------------------------------------------------------------------------


class TestInferenceWithNaN:
    """Tests for score and mahalanobis on NaN-aware regime-adjusted covariance."""

    def test_mahalanobis_and_score_with_delisted_asset(self, X_synth):
        """Inference excludes inactive assets row by row instead of failing."""
        X = X_synth.copy()
        X[120:, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[120:, 4] = False

        model = _fit(X, active_mask=um, assume_centered=False)

        distances = model.mahalanobis(X)
        score = model.score(X)

        assert distances.shape == (len(X),)
        assert np.all(np.isfinite(distances))
        assert np.isfinite(score)

        single_distance = model.mahalanobis(X[150])
        assert np.isscalar(single_distance)
        assert np.isfinite(single_distance)

    def test_score_all_invalid_rows_raises(self, X_synth):
        """score raises when no row has any finite retained observation."""
        model = _fit(X_synth)

        X_test = np.full((3, X_synth.shape[1]), np.nan)
        with pytest.raises(ValueError, match="finite retained observation"):
            model.score(X_test)

    def test_score_handles_holiday_missing_values(self, X_synth):
        """score remains finite when retained assets have row-wise holiday NaN."""
        model = _fit(X_synth)

        X_test = X_synth.copy()
        X_test[10:20, 1] = np.nan
        X_test[40:45, [0, 3]] = np.nan
        score = model.score(X_test)
        assert np.isfinite(score)

    def test_mahalanobis_validates_feature_names(self, X_synth):
        """mahalanobis still validates dataframe column order with NaN support."""
        columns = ["a", "b", "c", "d", "e"]
        X = pd.DataFrame(X_synth, columns=columns)
        model = _fit(X)

        with pytest.raises(ValueError, match="feature names"):
            model.mahalanobis(X[columns[::-1]])

    def test_mahalanobis_handles_nan_in_retained_subspace(self, X_synth):
        """mahalanobis computes on observed subspace when X_test has NaN."""
        model = _fit(X_synth)

        X_test = X_synth.copy()
        X_test[0, 0] = np.nan
        X_test[1, [1, 3]] = np.nan

        distances = model.mahalanobis(X_test)
        assert distances.shape == (len(X_test),)
        assert np.all(np.isfinite(distances))
        assert np.all(distances >= 0)

    def test_mahalanobis_all_nan_row_returns_nan(self, X_synth):
        """mahalanobis returns NaN for rows with no finite retained observation."""
        model = _fit(X_synth)

        X_test = X_synth[:5].copy()
        X_test[0, :] = np.nan
        distances = model.mahalanobis(X_test)
        assert np.isnan(distances[0])
        assert np.all(np.isfinite(distances[1:]))


# ---------------------------------------------------------------------------
# DCC (separate var/corr) with NaN
# ---------------------------------------------------------------------------


class TestDCCWithNaN:
    """Tests for separate variance/correlation half-lives with NaN."""

    def test_late_listing_dcc(self, X_synth):
        """Late-listed asset works with DCC."""
        X = X_synth.copy()
        X[:80, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:80, 3] = False

        model = _fit(X, active_mask=um, corr_half_life=20)
        assert not np.any(np.isnan(model.covariance_))
        np.testing.assert_array_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_delisting_dcc(self, X_synth):
        """Delisted asset produces NaN with DCC path."""
        X = X_synth.copy()
        X[100:, 2] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 2] = False

        model = _fit(X, active_mask=um, corr_half_life=20)
        assert np.all(np.isnan(model.covariance_[2, :]))
        _active_submatrix_psd(model.covariance_, [0, 1, 3, 4])

    def test_holiday_dcc(self, X_synth):
        """Holiday NaN with DCC produces valid output."""
        X = X_synth.copy()
        X[80:90, 1] = np.nan

        model = _fit(X, corr_half_life=20)
        assert not np.any(np.isnan(model.covariance_))

    def test_dcc_bias_correction_uses_pair_counts(self):
        """DCC correlation bias correction must use pairwise counts."""
        X = np.array(
            [
                [1.0, np.nan],
                [0.5, 0.5],
                [1.0, 1.0],
                [1.5, 1.5],
            ]
        )
        active_mask = np.array(
            [
                [True, False],
                [True, True],
                [True, True],
                [True, True],
            ]
        )

        model = RegimeAdjustedEWCovariance(
            half_life=2,
            corr_half_life=4,
            min_observations=1,
            regime_min_observations=100,
            nearest=False,
        )
        model.fit(X, active_mask=active_mask)

        corr_out, _ = cov_to_corr(model.covariance_)
        corr_raw = model._corr_state.copy()
        corr_counts = model._pair_obs_count.copy()
        corr_raw *= 1.0 / np.maximum(1.0 - model._corr_decay**corr_counts, 1e-15)

        diag_sqrt = np.sqrt(np.clip(np.diag(corr_raw), 1e-12, None))
        corr_expected = (
            corr_raw * (1.0 / diag_sqrt[:, None]) * (1.0 / diag_sqrt[None, :])
        )
        corr_expected = np.clip(corr_expected, -1.0, 1.0)
        corr_expected = 0.5 * (corr_expected + corr_expected.T)
        np.fill_diagonal(corr_expected, 1.0)

        assert model._pair_obs_count[0, 0] == 4
        assert model._pair_obs_count[1, 1] == 3
        assert model._pair_obs_count[0, 1] == 3
        np.testing.assert_allclose(corr_out, corr_expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------
# HAC with NaN
# ---------------------------------------------------------------------------


class TestHACWithNaN:
    """Tests for Newey-West HAC correction with NaN."""

    def test_late_listing_hac(self, X_synth):
        """Late-listed asset works with HAC enabled."""
        X = X_synth.copy()
        X[:80, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:80, 3] = False

        model = _fit(X, active_mask=um, hac_lags=3)
        assert not np.any(np.isnan(model.covariance_))

    def test_delisting_hac(self, X_synth):
        """Delisted asset produces NaN with HAC enabled."""
        X = X_synth.copy()
        X[100:, 2] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 2] = False

        model = _fit(X, active_mask=um, hac_lags=3)
        assert np.all(np.isnan(model.covariance_[2, :]))
        _active_submatrix_psd(model.covariance_, [0, 1, 3, 4])

    def test_holiday_hac(self, X_synth):
        """Holiday NaN with HAC produces valid (no NaN) output."""
        X = X_synth.copy()
        X[80:90, 1] = np.nan

        model = _fit(X, hac_lags=3)
        assert not np.any(np.isnan(model.covariance_))


# ---------------------------------------------------------------------------
# STVU with NaN (dynamic n_active, dynamic kappa)
# ---------------------------------------------------------------------------


class TestSTVUWithNaN:
    """Tests for STVU statistic computation with varying active asset counts."""

    @pytest.mark.parametrize("regime_target", list(RegimeAdjustmentTarget))
    def test_late_listing_regime_targets(self, X_synth, regime_target):
        """STVU works with all targets when an asset is late-listed."""
        X = X_synth.copy()
        X[:80, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:80, 3] = False

        kw = dict(regime_target=regime_target, regime_multiplier_clip=None)
        model = _fit(X, active_mask=um, **kw)
        assert not np.any(np.isnan(model.covariance_))
        assert model.regime_multiplier_ > 0

    @pytest.mark.parametrize("regime_method", list(RegimeAdjustmentMethod))
    def test_delisting_regime_methods(self, X_synth, regime_method):
        """STVU works with all methods when an asset is delisted."""
        X = X_synth.copy()
        X[100:, 2] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 2] = False

        kw = dict(regime_method=regime_method, regime_multiplier_clip=None)
        model = _fit(X, active_mask=um, **kw)
        assert np.all(np.isnan(model.covariance_[2, :]))
        assert model.regime_multiplier_ > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestNaNEdgeCases:
    def test_all_assets_delisted(self, X_synth):
        """All assets leave universe -> all NaN covariance."""
        um = np.ones_like(X_synth, dtype=bool)
        um[100:, :] = False
        X = X_synth.copy()
        X[100:, :] = np.nan

        model = _fit(X, active_mask=um)
        assert np.all(np.isnan(model.covariance_))

    def test_first_row_all_nan(self, X_synth):
        """First row all NaN should not break initialization."""
        X = X_synth.copy()
        X[0, :] = np.nan

        model = _fit(X)
        assert not np.any(np.isnan(model.covariance_))

    def test_listing_then_delisting(self, X_synth):
        """Asset is listed for a window then delisted -> NaN at end."""
        X = X_synth.copy()
        X[:50, 3] = np.nan
        X[150:, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:50, 3] = False
        um[150:, 3] = False

        model = _fit(X, active_mask=um)
        assert np.all(np.isnan(model.covariance_[3, :]))

    def test_symmetry_with_mixed_nan_pattern(self, X_synth):
        """Late listing + holiday + delisting -> symmetric output."""
        X = X_synth.copy()
        X[:30, 1] = np.nan
        X[80:90, 2] = np.nan
        X[150:, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:30, 1] = False
        um[150:, 4] = False

        model = _fit(X, active_mask=um)

        cov = model.covariance_
        nan_mask = np.isnan(cov)
        np.testing.assert_array_equal(nan_mask, nan_mask.T)
        finite = np.isfinite(cov)
        if np.any(finite):
            np.testing.assert_array_almost_equal(
                np.where(finite, cov, 0.0),
                np.where(finite, cov.T, 0.0),
            )

    def test_nearest_with_nan_submatrix(self, X_synth_wide):
        """nearest=True projects active submatrix to PD, preserves NaN frame."""
        X = X_synth_wide.copy()
        X[:, -3:] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:, -3:] = False

        model = _fit(X, active_mask=um, nearest=True)
        _active_submatrix_psd(model.covariance_, list(range(17)))

    def test_fit_resets_state(self, X_synth):
        """Calling fit() twice produces the same result (state is reset)."""
        model = RegimeAdjustedEWCovariance(**_DEFAULT_KWARGS)
        model.fit(X_synth[:100])
        cov_first = model.covariance_.copy()

        model.fit(X_synth[:100])
        np.testing.assert_array_equal(model.covariance_, cov_first)

        model.fit(X_synth[50:150])
        assert not np.allclose(model.covariance_, cov_first)

    def test_combined_dcc_hac_nan(self, X_synth):
        """DCC + HAC + NaN all combined."""
        X = X_synth.copy()
        X[:60, 3] = np.nan
        X[150:, 4] = np.nan
        X[90:100, 1] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:60, 3] = False
        um[150:, 4] = False

        model = _fit(X, active_mask=um, corr_half_life=20, hac_lags=3)

        assert np.all(np.isnan(model.covariance_[4, :]))
        assert not np.any(np.isnan(model.covariance_[3, [0, 1, 2, 3]]))
        _active_submatrix_psd(model.covariance_, [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# estimation_mask
# ---------------------------------------------------------------------------


class TestEstimationMask:
    """Tests for estimation_mask (STVU estimation universe)."""

    def test_basic(self, X_synth):
        """estimation_mask changes regime multiplier but not covariance structure."""
        model_all = _fit(X_synth)

        est = np.ones_like(X_synth, dtype=bool)
        est[:, :2] = False
        model_sub = _fit(X_synth, estimation_mask=est)

        assert model_all.regime_multiplier_ != model_sub.regime_multiplier_
        assert model_sub.covariance_.shape == model_all.covariance_.shape

    def test_invalid_shape_raises(self, X_synth):
        """Wrong shape raises ValueError."""
        est_bad = np.ones((X_synth.shape[0] + 1, X_synth.shape[1]), dtype=bool)
        model = RegimeAdjustedEWCovariance(**_DEFAULT_KWARGS)
        with pytest.raises(ValueError, match="estimation_mask shape"):
            model.fit(X_synth, estimation_mask=est_bad)

    def test_none_equivalent_to_all_true(self, X_synth):
        """estimation_mask=None behaves like all-True."""
        model_none = _fit(X_synth)
        est = np.ones_like(X_synth, dtype=bool)
        model_all = _fit(X_synth, estimation_mask=est)

        np.testing.assert_array_almost_equal(
            model_none.covariance_, model_all.covariance_
        )

    def test_all_false_freezes_regime_state(self, X_synth):
        """All-False estimation_mask disables STVU updates without affecting EWMA."""
        model_ref = _fit(X_synth, regime_min_observations=1)

        est = np.zeros_like(X_synth, dtype=bool)
        model_masked = _fit(X_synth, estimation_mask=est, regime_min_observations=1)

        np.testing.assert_array_almost_equal(
            model_masked.covariance_,
            model_ref.covariance_ / model_ref.regime_multiplier_**2,
        )
        assert model_masked.regime_multiplier_ == pytest.approx(1.0)

    def test_partial_fit_matches_batch(self, X_synth):
        """Streaming with estimation_mask matches batch fit."""
        est = np.ones_like(X_synth, dtype=bool)
        est[:, :2] = False

        model_batch = RegimeAdjustedEWCovariance(**_DEFAULT_KWARGS)
        model_batch.fit(X_synth, estimation_mask=est)

        model_online = RegimeAdjustedEWCovariance(**_DEFAULT_KWARGS)
        model_online.partial_fit(X_synth[:80], estimation_mask=est[:80])
        model_online.partial_fit(X_synth[80:], estimation_mask=est[80:])

        np.testing.assert_array_almost_equal(
            model_batch.covariance_, model_online.covariance_
        )

    def test_with_active_mask(self, X_synth):
        """Both masks work together."""
        X = X_synth.copy()
        X[100:, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 4] = False

        est = np.ones_like(X, dtype=bool)
        est[:, :2] = False

        model = _fit(X, active_mask=um, estimation_mask=est)

        assert np.all(np.isnan(model.covariance_[4, :]))
        _active_submatrix_psd(model.covariance_, [0, 1, 2, 3])

    @pytest.mark.parametrize("regime_target", list(RegimeAdjustmentTarget))
    def test_with_regime_targets(self, X_synth, regime_target):
        """estimation_mask works with all STVU targets."""
        est = np.ones_like(X_synth, dtype=bool)
        est[:, 0] = False

        model = _fit(
            X_synth,
            estimation_mask=est,
            regime_target=regime_target,
            regime_multiplier_clip=None,
        )
        assert not np.any(np.isnan(model.covariance_))
        assert model.regime_multiplier_ > 0


# ---------------------------------------------------------------------------
# Late listing PSD regression
# ---------------------------------------------------------------------------


class TestLateListingPSD:
    """Regression test: late-listed asset must not break positive definiteness."""

    def test_late_listing_psd_around_half_life(self):
        """Covariance stays PSD at the exact min_observations boundary."""
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        X = prices_to_returns(prices)
        X.loc[X.index[:500], "AAPL"] = np.nan

        model = RegimeAdjustedEWCovariance(half_life=80, nearest=False)

        for end in [579, 580, 650, 800]:
            model.fit(X.iloc[:end])
            cov = model.covariance_
            active = np.isfinite(np.diag(cov))
            if np.sum(active) > 1:
                sub = cov[np.ix_(active, active)]
                eigvals = np.linalg.eigvalsh(sub)
                assert np.all(eigvals >= -1e-10), (
                    f"Non-PD at end={end}: min eigval={eigvals.min():.2e}"
                )

    @pytest.mark.parametrize("half_life", [20, 40, 80, 120])
    def test_late_listing_psd_various_half_lives(self, half_life):
        """PSD is maintained for various half-lives with late listing."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((800, 20)) * 0.01
        X[:500, 0] = np.nan

        model = RegimeAdjustedEWCovariance(half_life=half_life, nearest=False)
        model.fit(X)

        cov = model.covariance_
        active = np.isfinite(np.diag(cov))
        sub = cov[np.ix_(active, active)]
        eigvals = np.linalg.eigvalsh(sub)
        assert np.all(eigvals >= -1e-10), (
            f"Non-PD with half_life={half_life}: min eigval={eigvals.min():.2e}"
        )

    @pytest.mark.parametrize("half_life", [20, 40, 80])
    def test_late_listing_psd_with_dcc(self, half_life):
        """PSD is maintained with separate variance/correlation decay (DCC)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((800, 20)) * 0.01
        X[:500, 0] = np.nan

        model = RegimeAdjustedEWCovariance(
            half_life=half_life,
            corr_half_life=half_life * 2,
            nearest=False,
        )
        model.fit(X)

        cov = model.covariance_
        active = np.isfinite(np.diag(cov))
        sub = cov[np.ix_(active, active)]
        eigvals = np.linalg.eigvalsh(sub)
        assert np.all(eigvals >= -1e-10), (
            f"Non-PD (DCC) with half_life={half_life}: min eigval={eigvals.min():.2e}"
        )
