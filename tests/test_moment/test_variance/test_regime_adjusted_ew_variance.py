import numpy as np
import pytest

from skfolio.moments.variance import RegimeAdjustedEWVariance, RegimeAdjustmentMethod


@pytest.fixture
def X():
    """Generate random returns for testing."""
    np.random.seed(42)
    return np.random.randn(500, 20) * 0.02


@pytest.fixture
def X_with_nans(X):
    """Generate returns with NaN values."""
    X_nan = X.copy()
    X_nan[50:60, 0] = np.nan
    X_nan[55:65, 5] = np.nan
    X_nan[100:110, 10] = np.nan
    return X_nan


class TestRegimeAdjustedEWVarianceBasic:
    def test_basic(self, X):
        """Test basic functionality with default parameters."""
        model = RegimeAdjustedEWVariance()
        model.fit(X)
        assert model.variance_.shape == (20,)
        assert hasattr(model, "regime_multiplier_")
        assert 0.5 < model.regime_multiplier_ < 2.0
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)

    def test_auto_stvu(self, X):
        """Test auto-calibrated STVU half-life."""
        model_auto = RegimeAdjustedEWVariance(half_life=11)
        model_auto.fit(X)

        model_manual = RegimeAdjustedEWVariance(half_life=11, regime_half_life=5.5)
        model_manual.fit(X)

        np.testing.assert_almost_equal(model_auto.variance_, model_manual.variance_)
        assert model_auto.regime_multiplier_ == model_manual.regime_multiplier_

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    def test_regime_methods(self, X, regime_method):
        """Test different STVU methods produce valid results."""
        model = RegimeAdjustedEWVariance(regime_method=regime_method, half_life=23)
        model.fit(X)
        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)
        assert 0.5 < model.regime_multiplier_ < 2.0

    def test_regime_methods_produce_different_results(self, X):
        """Test that different STVU methods produce different regime multipliers."""
        results = {}
        for method in RegimeAdjustmentMethod:
            model = RegimeAdjustedEWVariance(regime_method=method, half_life=23)
            model.fit(X)
            results[method] = model.regime_multiplier_

        values = list(results.values())
        assert len(set(values)) == 3, (
            "All STVU methods should produce different results"
        )

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    def test_assume_centered(self, X, regime_method):
        """Test RegimeAdjustedEWVariance with EWMA de-meaning."""
        model = RegimeAdjustedEWVariance(
            regime_method=regime_method, half_life=23, assume_centered=False
        )
        model.fit(X)
        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)
        assert 0.5 < model.regime_multiplier_ < 2.0

    def test_assume_centered_vs_not(self, X):
        """Test that centering produces different results."""
        model_centered = RegimeAdjustedEWVariance(half_life=23, assume_centered=True)
        model_centered.fit(X)

        model_not_centered = RegimeAdjustedEWVariance(
            half_life=23, assume_centered=False
        )
        model_not_centered.fit(X)

        dist = np.linalg.norm(model_centered.variance_ - model_not_centered.variance_)
        assert dist > 1e-10, "assume_centered should produce different results"

        assert hasattr(model_centered, "location_")
        assert model_centered.location_.shape == (20,)
        np.testing.assert_array_equal(model_centered.location_, np.zeros(20))

        assert hasattr(model_not_centered, "location_")
        assert model_not_centered.location_.shape == (20,)
        assert not np.allclose(model_not_centered.location_, np.zeros(20))

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    def test_regime_params(self, X, regime_method):
        """Test RegimeAdjustedEWVariance with custom STVU parameters."""
        model = RegimeAdjustedEWVariance(
            regime_method=regime_method,
            half_life=11,
            regime_half_life=14,
            regime_min_observations=10,
            regime_multiplier_clip=(0.8, 1.5),
        )
        model.fit(X)
        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert 0.8 <= model.regime_multiplier_ <= 1.5


class TestRegimeAdjustedEWVarianceEdgeCases:
    def test_insufficient_data(self):
        """Test with insufficient data for STVU."""
        X_small = np.random.randn(5, 3) * 0.02
        model = RegimeAdjustedEWVariance(half_life=23, regime_min_observations=100)
        model.fit(X_small)
        assert model.variance_ is not None
        assert np.all(np.isnan(model.variance_))

    def test_insufficient_data_regime_multiplier(self):
        """Test that regime_multiplier defaults to 1.0 with insufficient data."""
        X_small = np.random.randn(60, 3) * 0.02
        model = RegimeAdjustedEWVariance(half_life=23, regime_min_observations=100)
        model.fit(X_small)
        assert model.variance_ is not None
        assert model.regime_multiplier_ == 1.0

    def test_invalid_regime_half_life(self):
        """Test that invalid regime_half_life values raise appropriate errors."""
        X_small = np.random.randn(50, 3) * 0.02

        model = RegimeAdjustedEWVariance(regime_half_life=0)
        with pytest.raises(
            ValueError,
            match=r"regime_half_life must be positive",
        ):
            model.fit(X_small)

        model = RegimeAdjustedEWVariance(regime_half_life=-10)
        with pytest.raises(ValueError, match=r"regime_half_life must be positive"):
            model.fit(X_small)

        model = RegimeAdjustedEWVariance(regime_half_life=150)
        with pytest.warns(UserWarning, match="excessive memory.*desynchronize"):
            model.fit(X_small)
        assert model.variance_.shape == (3,)

    def test_no_clip(self, X):
        """Test with no STVU clipping."""
        model = RegimeAdjustedEWVariance(half_life=23, regime_multiplier_clip=None)
        model.fit(X)
        assert model.variance_.shape == (20,)
        assert model.regime_multiplier_ > 0

    def test_invalid_regime_multiplier_clip(self):
        """Test that invalid regime_multiplier_clip values raise appropriate errors."""
        X_small = np.random.randn(50, 3) * 0.02

        model = RegimeAdjustedEWVariance(regime_multiplier_clip=(1.5, 1.2))
        with pytest.raises(
            ValueError, match=r"regime_multiplier_clip must satisfy 0 < lo < hi"
        ):
            model.fit(X_small)

        model = RegimeAdjustedEWVariance(regime_multiplier_clip=(-0.5, 1.5))
        with pytest.raises(
            ValueError, match=r"regime_multiplier_clip must satisfy 0 < lo < hi"
        ):
            model.fit(X_small)

        model = RegimeAdjustedEWVariance(regime_multiplier_clip=(0.7, 1.5, 2.0))
        with pytest.raises(
            ValueError, match=r"regime_multiplier_clip must be a tuple of length 2"
        ):
            model.fit(X_small)

    def test_invalid_half_life(self):
        """Test that invalid half_life values raise appropriate errors."""
        X_small = np.random.randn(50, 3) * 0.02

        model = RegimeAdjustedEWVariance(half_life=0.0)
        with pytest.raises(ValueError, match=r"half_life must be positive"):
            model.fit(X_small)

        model = RegimeAdjustedEWVariance(half_life=-10)
        with pytest.raises(ValueError, match=r"half_life must be positive"):
            model.fit(X_small)

    def test_invalid_regime_method(self):
        """Invalid regime_method raises a clear ValueError."""
        X_small = np.random.randn(50, 3) * 0.02
        model = RegimeAdjustedEWVariance(regime_method="bad")
        with pytest.raises(
            ValueError, match="regime_method must be a RegimeAdjustmentMethod"
        ):
            model.fit(X_small)


class TestRegimeAdjustedEWVarianceHAC:
    def test_hac_basic(self, X):
        """Test basic HAC (Newey-West) functionality."""
        model = RegimeAdjustedEWVariance(half_life=23, hac_lags=5)
        model.fit(X)
        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)
        assert 0.5 < model.regime_multiplier_ < 2.0

    def test_hac_with_demeaning(self, X):
        """Test HAC with EWMA de-meaning."""
        model = RegimeAdjustedEWVariance(
            half_life=23, hac_lags=3, assume_centered=False
        )
        model.fit(X)
        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)

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
        model = RegimeAdjustedEWVariance(
            regime_method=regime_method, half_life=23, hac_lags=5
        )
        model.fit(X)
        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)

    def test_hac_vs_no_hac(self, X):
        """Test that HAC produces different results than no HAC."""
        model_no_hac = RegimeAdjustedEWVariance(half_life=23)
        model_no_hac.fit(X)

        model_hac = RegimeAdjustedEWVariance(half_life=23, hac_lags=5)
        model_hac.fit(X)

        dist = np.linalg.norm(model_hac.variance_ - model_no_hac.variance_)
        assert dist > 1e-6, "HAC should produce different results"

    def test_hac_different_lags(self, X):
        """Test that different HAC lag values produce different results."""
        model_3 = RegimeAdjustedEWVariance(half_life=23, hac_lags=3)
        model_3.fit(X)

        model_5 = RegimeAdjustedEWVariance(half_life=23, hac_lags=5)
        model_5.fit(X)

        model_10 = RegimeAdjustedEWVariance(half_life=23, hac_lags=10)
        model_10.fit(X)

        dist_3_5 = np.linalg.norm(model_3.variance_ - model_5.variance_)
        dist_5_10 = np.linalg.norm(model_5.variance_ - model_10.variance_)

        assert dist_3_5 > 1e-8, "Different HAC lags should produce different results"
        assert dist_5_10 > 1e-8, "Different HAC lags should produce different results"

    def test_hac_invalid_lags(self):
        """Test that invalid hac_lags values raise appropriate errors."""
        X_small = np.random.randn(50, 3) * 0.02

        model = RegimeAdjustedEWVariance(hac_lags=0)
        with pytest.raises(ValueError, match="hac_lags must be a positive integer"):
            model.fit(X_small)

        model = RegimeAdjustedEWVariance(hac_lags=-1)
        with pytest.raises(ValueError, match="hac_lags must be a positive integer"):
            model.fit(X_small)

        model = RegimeAdjustedEWVariance(hac_lags=2.5)
        with pytest.raises(ValueError, match="hac_lags must be a positive integer"):
            model.fit(X_small)

    def test_hac_ignores_inactive_returns(self, X):
        """Inactive rows must not leak into HAC lagged-return buffering."""
        X_ref = X.copy()
        X_bad = X.copy()
        active_mask = np.ones(X.shape, dtype=bool)
        active_mask[120:126, 2] = False
        X_bad[120:126, 2] = 10.0

        model_ref = RegimeAdjustedEWVariance(
            half_life=23,
            hac_lags=3,
            min_observations=1,
            regime_min_observations=1,
            regime_multiplier_clip=None,
        )
        model_ref.fit(X_ref, active_mask=active_mask)

        model_bad = RegimeAdjustedEWVariance(
            half_life=23,
            hac_lags=3,
            min_observations=1,
            regime_min_observations=1,
            regime_multiplier_clip=None,
        )
        model_bad.fit(X_bad, active_mask=active_mask)

        np.testing.assert_allclose(
            model_bad.variance_,
            model_ref.variance_,
            rtol=1e-10,
            atol=1e-12,
        )
        assert model_bad.regime_multiplier_ == pytest.approx(
            model_ref.regime_multiplier_, rel=1e-10
        )


class TestRegimeAdjustedEWVariancePartialFit:
    def test_partial_fit_matches_batch(self, X):
        """Streaming updates should match batch fit results."""
        model_batch = RegimeAdjustedEWVariance(half_life=23)
        model_batch.fit(X)

        model_online = RegimeAdjustedEWVariance(half_life=23)
        chunk = 37
        for start in range(0, X.shape[0], chunk):
            model_online.partial_fit(X[start : start + chunk])

        np.testing.assert_allclose(
            model_online.variance_,
            model_batch.variance_,
            rtol=1e-10,
            atol=1e-12,
        )
        assert model_online.regime_multiplier_ == pytest.approx(
            model_batch.regime_multiplier_, rel=1e-10
        )

    def test_partial_fit_incremental(self, X):
        """Test that partial_fit updates estimates incrementally."""
        model = RegimeAdjustedEWVariance(half_life=23)

        model.partial_fit(X[:100])
        var_1 = model.variance_.copy()

        model.partial_fit(X[100:200])
        var_2 = model.variance_.copy()

        model.partial_fit(X[200:300])
        var_3 = model.variance_.copy()

        assert not np.allclose(var_1, var_2)
        assert not np.allclose(var_2, var_3)

    def test_partial_fit_with_hac(self, X):
        """Test partial_fit with HAC correction."""
        model_batch = RegimeAdjustedEWVariance(half_life=23, hac_lags=3)
        model_batch.fit(X)

        model_online = RegimeAdjustedEWVariance(half_life=23, hac_lags=3)
        chunk = 50
        for start in range(0, X.shape[0], chunk):
            model_online.partial_fit(X[start : start + chunk])

        np.testing.assert_allclose(
            model_online.variance_,
            model_batch.variance_,
            rtol=1e-10,
            atol=1e-12,
        )


class TestRegimeAdjustedEWVarianceNaN:
    def test_nan_handling_basic(self, X_with_nans):
        """Test that NaN values are handled correctly."""
        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_with_nans)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)
        assert np.isfinite(model.regime_multiplier_)

    def test_nan_vs_no_nan(self, X, X_with_nans):
        """Test that NaN handling produces similar but not identical results."""
        model_clean = RegimeAdjustedEWVariance(half_life=23)
        model_clean.fit(X)

        model_nan = RegimeAdjustedEWVariance(half_life=23)
        model_nan.fit(X_with_nans)

        assert np.all(np.isfinite(model_clean.variance_))
        assert np.all(np.isfinite(model_nan.variance_))

        non_nan_assets = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        dist = np.linalg.norm(
            model_clean.variance_[non_nan_assets] - model_nan.variance_[non_nan_assets]
        )
        assert dist < 1e-3

    @pytest.mark.parametrize(
        "regime_method",
        [
            RegimeAdjustmentMethod.LOG,
            RegimeAdjustmentMethod.FIRST_MOMENT,
            RegimeAdjustmentMethod.RMS,
        ],
    )
    def test_nan_with_regime_methods(self, X_with_nans, regime_method):
        """Test NaN handling with all STVU methods."""
        model = RegimeAdjustedEWVariance(regime_method=regime_method, half_life=23)
        model.fit(X_with_nans)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.isfinite(model.regime_multiplier_)

    def test_nan_with_hac(self, X_with_nans):
        """Test NaN handling with HAC correction."""
        model = RegimeAdjustedEWVariance(half_life=23, hac_lags=3)
        model.fit(X_with_nans)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.isfinite(model.regime_multiplier_)

    def test_nan_with_demeaning(self, X_with_nans):
        """Test NaN handling with demeaning."""
        model = RegimeAdjustedEWVariance(half_life=23, assume_centered=False)
        model.fit(X_with_nans)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.isfinite(model.regime_multiplier_)

    def test_nan_partial_fit(self, X_with_nans):
        """Test NaN handling with partial_fit."""
        model = RegimeAdjustedEWVariance(half_life=23)

        chunk = 100
        for start in range(0, X_with_nans.shape[0], chunk):
            model.partial_fit(X_with_nans[start : start + chunk])

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.isfinite(model.regime_multiplier_)

    def test_all_nan_row(self, X):
        """Test handling of rows with all NaN values."""
        X_nan = X.copy()
        X_nan[50, :] = np.nan
        X_nan[100, :] = np.nan

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_nan)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))

    def test_heavy_nan_asset(self, X):
        """Test with one asset having many NaN values."""
        X_nan = X.copy()
        nan_indices = np.random.choice(
            range(15, X.shape[0]),
            size=(X.shape[0] - 15) // 2,
            replace=False,
        )
        X_nan[nan_indices, 0] = np.nan

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_nan)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert model.variance_[0] > 0


class TestRegimeAdjustedEWVarianceEstimationUniverse:
    def test_estimation_mask_basic(self, X):
        """Test basic estimation_mask functionality."""
        model_all = RegimeAdjustedEWVariance(half_life=23)
        model_all.fit(X)

        est_univ = np.ones(X.shape, dtype=bool)
        est_univ[:, :5] = False
        model_subset = RegimeAdjustedEWVariance(half_life=23)
        model_subset.fit(X, estimation_mask=est_univ)

        assert model_all.regime_multiplier_ != model_subset.regime_multiplier_
        assert model_all.variance_.shape == model_subset.variance_.shape

    def test_estimation_mask_vs_none(self, X):
        """Test that estimation_mask=None uses all assets."""
        model_none = RegimeAdjustedEWVariance(half_life=23)
        model_none.fit(X, estimation_mask=None)

        est_univ = np.ones(X.shape, dtype=bool)
        model_all = RegimeAdjustedEWVariance(half_life=23)
        model_all.fit(X, estimation_mask=est_univ)

        np.testing.assert_allclose(
            model_none.variance_, model_all.variance_, rtol=1e-10
        )
        assert model_none.regime_multiplier_ == pytest.approx(
            model_all.regime_multiplier_, rel=1e-10
        )

    def test_estimation_mask_all_false_freezes_regime_state(self, X):
        """All-False estimation_mask disables STVU updates without affecting EWMA."""
        model_ref = RegimeAdjustedEWVariance(half_life=23, regime_min_observations=1)
        model_ref.fit(X)

        est_univ = np.zeros(X.shape, dtype=bool)
        model_masked = RegimeAdjustedEWVariance(half_life=23, regime_min_observations=1)
        model_masked.fit(X, estimation_mask=est_univ)

        np.testing.assert_allclose(
            model_masked.variance_,
            model_ref.variance_ / model_ref.regime_multiplier_**2,
            rtol=1e-10,
            atol=1e-12,
        )
        assert model_masked.regime_multiplier_ == pytest.approx(1.0)

    def test_estimation_mask_time_varying(self, X):
        """Test time-varying estimation universe."""
        est_univ = np.ones(X.shape, dtype=bool)
        est_univ[:250, :10] = False
        est_univ[250:, 10:] = False

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X, estimation_mask=est_univ)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))

    def test_estimation_mask_invalid_shape(self, X):
        """Test that invalid estimation_mask shape raises error."""
        model = RegimeAdjustedEWVariance(half_life=23)

        est_univ_bad = np.ones((X.shape[0] - 10, X.shape[1]), dtype=bool)
        with pytest.raises(ValueError, match=r"estimation_mask shape.*does not match"):
            model.fit(X, estimation_mask=est_univ_bad)

        est_univ_bad = np.ones((X.shape[0], X.shape[1] - 5), dtype=bool)
        with pytest.raises(ValueError, match=r"estimation_mask shape.*does not match"):
            model.fit(X, estimation_mask=est_univ_bad)

    def test_estimation_mask_partial_fit(self, X):
        """Test estimation_mask with partial_fit."""
        est_univ = np.ones(X.shape, dtype=bool)
        est_univ[:, :5] = False

        model_batch = RegimeAdjustedEWVariance(half_life=23)
        model_batch.fit(X, estimation_mask=est_univ)

        model_online = RegimeAdjustedEWVariance(half_life=23)
        chunk = 100
        for start in range(0, X.shape[0], chunk):
            end = min(start + chunk, X.shape[0])
            model_online.partial_fit(X[start:end], estimation_mask=est_univ[start:end])

        np.testing.assert_allclose(
            model_online.variance_,
            model_batch.variance_,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_estimation_mask_with_nan(self, X_with_nans):
        """Test estimation_mask combined with NaN handling."""
        est_univ = np.ones(X_with_nans.shape, dtype=bool)
        est_univ[:, :5] = False

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_with_nans, estimation_mask=est_univ)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.isfinite(model.regime_multiplier_)

    def test_estimation_mask_exclude_nan_assets(self, X):
        """Test excluding assets that have many NaNs from STVU."""
        X_nan = X.copy()
        for i in range(3):
            nan_idx = np.random.choice(range(15, X.shape[0]), size=50, replace=False)
            X_nan[nan_idx, i] = np.nan

        est_univ = np.ones(X_nan.shape, dtype=bool)
        est_univ[:, :3] = False

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_nan, estimation_mask=est_univ)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))


class TestRegimeAdjustedEWVarianceCalibration:
    def test_variance_calibration(self):
        """On IID N(0, sigma^2) with long burn-in, regime_multiplier_ should be ~1."""
        rng = np.random.default_rng(123)

        n_assets = 8
        n_burn = 5_000
        n_eval = 2_000

        true_stds = rng.uniform(0.01, 0.03, size=n_assets)
        X = rng.standard_normal((n_burn + n_eval, n_assets)) * true_stds

        model = RegimeAdjustedEWVariance(
            half_life=70,
            regime_half_life=150,
            regime_min_observations=200,
        )
        with pytest.warns(UserWarning):
            model.fit(X)

        reg = model.regime_multiplier_
        assert reg is not None
        assert 0.90 <= reg <= 1.10, f"regime_multiplier_={reg:.4f} not ~ 1 on IID data"

    def test_regime_jump(self):
        """Test STVU response to regime change."""
        np.random.seed(43)
        n_obs, n_assets = 300, 8
        regime_change_at = 200
        vol_multiplier = 1.5

        base_std = np.full(n_assets, 0.015)
        X = np.random.randn(n_obs, n_assets) * base_std
        X[regime_change_at:] *= vol_multiplier

        model = RegimeAdjustedEWVariance(
            half_life=23,
            regime_half_life=70,
            regime_min_observations=50,
            regime_multiplier_clip=(0.5, 2.0),
        )
        model.fit(X)

        assert model.regime_multiplier_ is not None
        assert model.regime_multiplier_ > 1.0, (
            f"STVU = {model.regime_multiplier_} should be > 1.0 after vol increase"
        )


class TestRegimeAdjustedEWVarianceHighDimensional:
    def test_high_dimensional(self):
        """Test with many assets (100+) to verify numerical stability."""
        n_assets = 100
        n_obs = 500
        rng = np.random.default_rng(456)

        X = rng.standard_normal((n_obs, n_assets)) * 0.02

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X)

        assert model.variance_.shape == (n_assets,)
        assert np.all(model.variance_ > 0)
        assert model.regime_multiplier_ > 0
        assert np.all(np.isfinite(model.variance_))

    def test_near_constant_asset(self):
        """Test with nearly constant asset (very low variance)."""
        n_assets = 5
        n_obs = 200
        rng = np.random.default_rng(789)

        X = rng.standard_normal((n_obs, n_assets)) * 0.02
        X[:, 0] = X[:, 0] * 1e-10

        model = RegimeAdjustedEWVariance(half_life=11)
        model.fit(X)

        assert model.variance_.shape == (n_assets,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)


class TestRegimeAdjustedEWVarianceUniverseMask:
    def test_delisting_produces_nan(self, X):
        """Delisted asset has NaN variance; others remain finite."""
        X_copy = X.copy()
        X_copy[100:, 2] = np.nan
        um = np.ones(X.shape, dtype=bool)
        um[100:, 2] = False

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_copy, active_mask=um)

        assert np.isnan(model.variance_[2])
        assert np.all(np.isfinite(model.variance_[[0, 1, 3, 4]]))

    def test_holiday_vs_delisting(self, X):
        """Same NaN pattern, different active_mask -> different output."""
        X_nan = X.copy()
        X_nan[400:, 3] = np.nan

        model_hol = RegimeAdjustedEWVariance(half_life=23)
        model_hol.fit(X_nan)
        assert np.all(np.isfinite(model_hol.variance_))

        um = np.ones(X.shape, dtype=bool)
        um[400:, 3] = False
        model_del = RegimeAdjustedEWVariance(half_life=23)
        model_del.fit(X_nan, active_mask=um)
        assert np.isnan(model_del.variance_[3])

    def test_relisting_produces_valid_variance(self, X):
        """Asset re-entering universe gets valid variance via zero-init + bias correction."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[50:100, 1] = np.nan
        um[50:100, 1] = False

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_copy, active_mask=um)

        assert np.isfinite(model.variance_[1])
        assert model.variance_[1] > 0

    def test_invalid_shape_raises(self, X):
        """Wrong shape raises ValueError."""
        model = RegimeAdjustedEWVariance(half_life=23)
        um_bad = np.ones((X.shape[0] + 1, X.shape[1]), dtype=bool)
        with pytest.raises(ValueError, match=r"active_mask shape.*does not match"):
            model.fit(X, active_mask=um_bad)

    def test_none_equivalent_to_all_true(self, X):
        """active_mask=None behaves like all-True."""
        model_none = RegimeAdjustedEWVariance(half_life=23)
        model_none.fit(X)

        um = np.ones(X.shape, dtype=bool)
        model_all = RegimeAdjustedEWVariance(half_life=23)
        model_all.fit(X, active_mask=um)

        np.testing.assert_allclose(
            model_none.variance_, model_all.variance_, rtol=1e-10
        )

    def test_partial_fit_matches_batch(self, X):
        """Streaming with active_mask matches batch fit."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[100:, 2] = np.nan
        um[100:, 2] = False

        model_batch = RegimeAdjustedEWVariance(half_life=23)
        model_batch.fit(X_copy, active_mask=um)

        model_online = RegimeAdjustedEWVariance(half_life=23)
        chunk = 100
        for start in range(0, X.shape[0], chunk):
            end = min(start + chunk, X.shape[0])
            model_online.partial_fit(X_copy[start:end], active_mask=um[start:end])

        np.testing.assert_allclose(
            model_online.variance_, model_batch.variance_, rtol=1e-10, atol=1e-12
        )

    def test_with_estimation_mask(self, X):
        """Both masks work together."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[100:, 2] = np.nan
        um[100:, 2] = False

        est = np.ones(X.shape, dtype=bool)
        est[:, :5] = False

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_copy, active_mask=um, estimation_mask=est)

        assert np.isnan(model.variance_[2])
        assert np.all(np.isfinite(model.variance_[[0, 1, 3, 4]]))

    def test_never_in_universe_has_nan(self, X):
        """Asset never in universe has NaN variance."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[:, 4] = np.nan
        um[:, 4] = False

        model = RegimeAdjustedEWVariance(half_life=23)
        model.fit(X_copy, active_mask=um)

        assert np.isnan(model.variance_[4])
        assert np.all(np.isfinite(model.variance_[:4]))

    def test_location_nan_for_delisted(self, X):
        """location_ is NaN for delisted assets when assume_centered=False."""
        X_copy = X.copy()
        X_copy[100:, 3] = np.nan
        um = np.ones(X.shape, dtype=bool)
        um[100:, 3] = False

        model = RegimeAdjustedEWVariance(
            half_life=23, assume_centered=False, min_observations=1
        )
        model.fit(X_copy, active_mask=um)

        assert np.isnan(model.location_[3])
        assert np.all(np.isfinite(model.location_[[0, 1, 2, 4]]))


class TestRegimeAdjustedEWVarianceMinObservations:
    """Tests for min_observations warm-up gating."""

    def test_default_min_observations(self, X):
        """Default min_observations equals int(half_life); enough data -> all finite."""
        model = RegimeAdjustedEWVariance(half_life=40)
        model.fit(X)
        assert not np.any(np.isnan(model.variance_))

    def test_late_listing_below_threshold(self, X):
        """Late-listed asset below threshold has NaN variance."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[:490, 4] = np.nan
        um[:490, 4] = False

        model = RegimeAdjustedEWVariance(half_life=23, min_observations=15)
        model.fit(X_copy, active_mask=um)

        assert np.isnan(model.variance_[4])
        assert np.all(np.isfinite(model.variance_[[0, 1, 2, 3]]))

    def test_late_listing_above_threshold(self, X):
        """Late-listed asset above threshold has finite variance."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[:100, 4] = np.nan
        um[:100, 4] = False

        model = RegimeAdjustedEWVariance(half_life=23, min_observations=15)
        model.fit(X_copy, active_mask=um)

        assert np.isfinite(model.variance_[4])
        assert model.variance_[4] > 0

    def test_min_observations_validation(self):
        """min_observations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_observations must be >= 1"):
            RegimeAdjustedEWVariance(min_observations=0).fit(np.random.randn(100, 3))

    def test_min_observations_tracked_across_partial_fit(self, X):
        """Warm-up threshold tracked across partial_fit calls."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[:, 4] = np.nan
        um[:, 4] = False

        model = RegimeAdjustedEWVariance(half_life=23, min_observations=5)
        model.partial_fit(X_copy[:200], active_mask=um[:200])
        assert np.isnan(model.variance_[4])

        # Re-enter universe with valid data; 3 obs < min_observations=5
        model.partial_fit(X[200:203])
        assert np.isnan(model.variance_[4])

        # After 7 more obs total: 10 > min_observations=5
        model.partial_fit(X[203:210])
        assert np.isfinite(model.variance_[4])


class TestRegimeAdjustedEWVarianceBiasCorrection:
    """Tests for universal bias correction."""

    def test_relisting_restarts_bias_correction(self, X):
        """Delisted then re-listed asset restarts bias correction from zero."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[80:200, 3] = np.nan
        um[80:200, 3] = False

        model = RegimeAdjustedEWVariance(half_life=23, min_observations=1)
        model.fit(X_copy, active_mask=um)

        assert np.isfinite(model.variance_[3])
        assert model.variance_[3] > 0

    def test_bias_correction_applied_to_all_assets(self, X):
        """All assets receive universal bias correction (obs_count based)."""
        model = RegimeAdjustedEWVariance(half_life=23, min_observations=1)
        model.fit(X)

        # All assets should have obs_count > 0 and valid variance
        assert np.all(model._obs_count > 0)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)

    def test_late_listed_asset_gets_bias_correction(self, X):
        """Asset entering late gets proper bias correction."""
        X_copy = X.copy()
        um = np.ones(X.shape, dtype=bool)
        X_copy[:400, 4] = np.nan
        um[:400, 4] = False

        model = RegimeAdjustedEWVariance(half_life=23, min_observations=1)
        model.fit(X_copy, active_mask=um)

        # Asset 4 entered at obs 400 with 100 obs, other assets have 500 obs
        assert model._obs_count[4] == 100
        assert model._obs_count[0] == 500
        assert np.isfinite(model.variance_[4])
        assert model.variance_[4] > 0

    def test_stvu_uses_pre_update_bias_correction(self):
        """STVU must use the pre-update variance state for one-step-ahead scaling."""
        X = np.array([[1.0], [2.0]])
        model = RegimeAdjustedEWVariance(
            half_life=2,
            min_observations=1,
            regime_min_observations=1,
            regime_multiplier_clip=None,
        )
        model.fit(X)

        expected = 2.0 / np.sqrt(2.0 / np.pi)
        assert model.regime_multiplier_ == pytest.approx(expected, rel=1e-12)

    def test_default_regime_min_observations_is_never_zero(self):
        """Auto-calibrated regime warm-up keeps at least one observation."""
        X = np.array([[1.0], [2.0]])
        model = RegimeAdjustedEWVariance(
            half_life=1.5,
            min_observations=1,
            regime_multiplier_clip=None,
        )
        model.fit(X)

        assert model._regime_min_observations == 1
        assert model.regime_multiplier_ == pytest.approx(
            2.0 / np.sqrt(2.0 / np.pi), rel=1e-12
        )
