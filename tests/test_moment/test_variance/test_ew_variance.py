"""Tests for EWVariance estimator."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.moments import EWVariance


@pytest.fixture
def X_synth():
    """Small synthetic return matrix (200 obs, 5 assets), no NaN."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 5)) * 0.01


class TestEWVariance:
    def test_basic(self, X):
        """Test basic functionality with default parameters."""
        model = EWVariance(half_life=23)
        model.fit(X)

        assert model.variance_.shape == (20,)
        assert model.location_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)

    def test_default_half_life(self, X):
        """Test default half_life is 40."""
        model = EWVariance()
        model.fit(X)

        model_explicit = EWVariance(half_life=40)
        model_explicit.fit(X)

        np.testing.assert_allclose(model.variance_, model_explicit.variance_)

    @pytest.mark.parametrize("half_life", [3, 6, 11, 23, 70])
    def test_half_life_values(self, X, half_life):
        """Test various half-life values produce valid results."""
        model = EWVariance(half_life=half_life)
        model.fit(X)

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)

    def test_larger_half_life_more_stable(self, X):
        """Test that larger half-life produces more stable estimates."""
        model_stable = EWVariance(half_life=70)
        model_responsive = EWVariance(half_life=3)

        model_stable.fit(X)
        model_responsive.fit(X)

        # Both should produce valid results but differ
        assert not np.allclose(model_stable.variance_, model_responsive.variance_)

    def test_assume_centered_true(self, X):
        """Test with assume_centered=True (default)."""
        model = EWVariance(half_life=23, assume_centered=True)
        model.fit(X)

        np.testing.assert_array_equal(model.location_, np.zeros(20))

    def test_assume_centered_false(self, X):
        """Test with assume_centered=False."""
        model = EWVariance(half_life=23, assume_centered=False)
        model.fit(X)

        # location_ should be non-zero (EWMA mean)
        assert not np.allclose(model.location_, np.zeros(20))

    def test_assume_centered_produces_different_results(self, X):
        """Test that assume_centered produces different results."""
        model_centered = EWVariance(half_life=23, assume_centered=True)
        model_not_centered = EWVariance(half_life=23, assume_centered=False)

        model_centered.fit(X)
        model_not_centered.fit(X)

        assert not np.allclose(model_centered.variance_, model_not_centered.variance_)

    def test_window_size(self, X):
        """Test with window_size parameter."""
        model_full = EWVariance(half_life=23)
        model_window = EWVariance(half_life=23, window_size=100)

        model_full.fit(X)
        model_window.fit(X)

        # Results should differ
        assert not np.allclose(model_full.variance_, model_window.variance_)

    def test_partial_fit(self, X):
        """Test partial_fit for streaming updates."""
        X_arr = np.asarray(X)

        # Batch fit
        model_batch = EWVariance(half_life=23)
        model_batch.fit(X_arr)

        # Streaming fit
        model_stream = EWVariance(half_life=23)
        model_stream.partial_fit(X_arr[:100])
        model_stream.partial_fit(X_arr[100:200])
        model_stream.partial_fit(X_arr[200:])

        # Should produce identical results
        np.testing.assert_allclose(
            model_batch.variance_, model_stream.variance_, rtol=1e-10
        )

    def test_window_size_applies_only_to_first_partial_fit(self, X_synth):
        """window_size truncates only the first partial_fit batch."""
        um = np.ones_like(X_synth, dtype=bool)
        um[:40, 4] = False

        model_windowed = EWVariance(half_life=10, min_observations=1, window_size=50)
        model_windowed.partial_fit(X_synth[:80], active_mask=um[:80])
        model_windowed.partial_fit(X_synth[80:150], active_mask=um[80:150])

        model_ref = EWVariance(half_life=10, min_observations=1)
        model_ref.partial_fit(X_synth[30:80], active_mask=um[30:80])
        model_ref.partial_fit(X_synth[80:150], active_mask=um[80:150])

        np.testing.assert_allclose(
            model_windowed.variance_, model_ref.variance_, rtol=1e-10
        )

    def test_partial_fit_single_observation(self, X):
        """Test partial_fit with single observations."""
        X_arr = np.asarray(X)[:50]

        model = EWVariance(half_life=23)
        for row in X_arr:
            model.partial_fit(row.reshape(1, -1))

        assert model.variance_.shape == (20,)
        assert np.all(np.isfinite(model.variance_))
        assert np.all(model.variance_ > 0)

    def test_sklearn_api(self, X):
        """Test sklearn-compatible API."""
        model = EWVariance(half_life=23)

        # fit should return self
        result = model.fit(X)
        assert result is model

        # n_features_in_ should be set
        assert model.n_features_in_ == 20

    def test_invalid_half_life(self):
        """Test that invalid half_life raises error."""
        with pytest.raises(ValueError, match="half_life must be positive"):
            model = EWVariance(half_life=0)
            model.fit(np.random.randn(100, 5))

        with pytest.raises(ValueError, match="half_life must be positive"):
            model = EWVariance(half_life=-10)
            model.fit(np.random.randn(100, 5))

    def test_exact_values(self, X):
        """Regression test with exact expected values."""
        # half_life = -ln(2)/ln(0.97) gives exactly decay_factor = 0.97
        model = EWVariance(half_life=-np.log(2) / np.log(0.97))
        model.fit(X)

        expected_variance = np.array(
            [0.00056035, 0.00120763, 0.00032479, 0.00081175, 0.00033951]
        )

        np.testing.assert_allclose(model.variance_[:5], expected_variance, rtol=1e-5)


class TestEWVarianceNaN:
    """NaN handling without active_mask (all NaN treated as holidays)."""

    def test_nan_freezes_variance(self, X_synth):
        """NaN returns freeze variance, no NaN in output."""
        X = X_synth.copy()
        X[80:90, 1] = np.nan

        model = EWVariance(half_life=10)
        model.fit(X)
        assert np.all(np.isfinite(model.variance_))

    def test_late_listing(self, X_synth):
        """Asset with NaN at start gets valid variance after first valid obs."""
        X = X_synth.copy()
        X[:100, 3] = np.nan

        model = EWVariance(half_life=10)
        model.fit(X)
        assert np.all(np.isfinite(model.variance_))
        assert model.variance_[3] > 0

    def test_nan_partial_fit_matches_batch(self, X_synth):
        """Streaming with NaN matches batch fit."""
        X = X_synth.copy()
        X[50:60, 2] = np.nan

        model_batch = EWVariance(half_life=10)
        model_batch.fit(X)

        model_online = EWVariance(half_life=10)
        model_online.partial_fit(X[:80])
        model_online.partial_fit(X[80:])

        np.testing.assert_allclose(
            model_online.variance_, model_batch.variance_, rtol=1e-10
        )


class TestEWVarianceUniverseMask:
    """Tests for active_mask (holiday vs. delisting distinction)."""

    def test_delisting_produces_nan(self, X_synth):
        """Delisted asset has NaN variance; others remain finite."""
        X = X_synth.copy()
        X[100:, 2] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 2] = False

        model = EWVariance(half_life=10)
        model.fit(X, active_mask=um)

        assert np.isnan(model.variance_[2])
        assert np.all(np.isfinite(model.variance_[[0, 1, 3, 4]]))

    def test_holiday_vs_delisting(self, X_synth):
        """Same NaN pattern, different active_mask -> different output."""
        X = X_synth.copy()
        X[150:, 3] = np.nan

        model_hol = EWVariance(half_life=10)
        model_hol.fit(X)
        assert np.all(np.isfinite(model_hol.variance_))

        um = np.ones_like(X, dtype=bool)
        um[150:, 3] = False
        model_del = EWVariance(half_life=10)
        model_del.fit(X, active_mask=um)
        assert np.isnan(model_del.variance_[3])

    def test_cold_start_after_relisting(self, X_synth):
        """Asset re-entering universe gets valid variance via cold-start."""
        X = X_synth.copy()
        um = np.ones_like(X, dtype=bool)
        X[50:100, 1] = np.nan
        um[50:100, 1] = False

        model = EWVariance(half_life=10)
        model.fit(X, active_mask=um)

        assert np.isfinite(model.variance_[1])
        assert model.variance_[1] > 0

    def test_invalid_shape_raises(self, X_synth):
        """Wrong shape raises ValueError."""
        model = EWVariance(half_life=10)
        um_bad = np.ones((X_synth.shape[0] + 1, X_synth.shape[1]), dtype=bool)
        with pytest.raises(ValueError, match=r"active_mask shape.*does not match"):
            model.fit(X_synth, active_mask=um_bad)

    def test_none_equivalent_to_all_true(self, X_synth):
        """active_mask=None behaves like all-True (no NaN in data)."""
        model_none = EWVariance(half_life=10)
        model_none.fit(X_synth)

        um = np.ones_like(X_synth, dtype=bool)
        model_all = EWVariance(half_life=10)
        model_all.fit(X_synth, active_mask=um)

        np.testing.assert_allclose(
            model_none.variance_, model_all.variance_, rtol=1e-10
        )

    def test_partial_fit_matches_batch(self, X_synth):
        """Streaming with active_mask matches batch fit."""
        X = X_synth.copy()
        um = np.ones_like(X, dtype=bool)
        X[100:, 2] = np.nan
        um[100:, 2] = False

        model_batch = EWVariance(half_life=10)
        model_batch.fit(X, active_mask=um)

        model_online = EWVariance(half_life=10)
        model_online.partial_fit(X[:80], active_mask=um[:80])
        model_online.partial_fit(X[80:], active_mask=um[80:])

        np.testing.assert_allclose(
            model_online.variance_, model_batch.variance_, rtol=1e-10
        )

    def test_location_nan_for_delisted(self, X_synth):
        """location_ is NaN for delisted assets when assume_centered=False."""
        X = X_synth.copy()
        X[100:, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[100:, 3] = False

        model = EWVariance(half_life=10, assume_centered=False, min_observations=1)
        model.fit(X, active_mask=um)

        assert np.isnan(model.location_[3])
        assert np.all(np.isfinite(model.location_[[0, 1, 2, 4]]))


# ---------------------------------------------------------------------------
# min_observations warm-up
# ---------------------------------------------------------------------------


class TestEWVarianceMinObservations:
    """Warm-up behavior of min_observations."""

    def test_below_threshold_is_nan(self, X_synth):
        """Assets below min_observations have NaN variance."""
        X = X_synth[:30].copy()
        X[:25, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:25, 4] = False

        model = EWVariance(half_life=10, min_observations=10)
        model.fit(X, active_mask=um)

        assert np.isnan(model.variance_[4])
        assert not np.any(np.isnan(model.variance_[[0, 1, 2, 3]]))

    def test_crosses_threshold(self, X_synth):
        """Asset becomes active once it crosses min_observations."""
        X = X_synth.copy()
        X[:180, 4] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:180, 4] = False

        model = EWVariance(half_life=10, min_observations=15)
        model.fit(X, active_mask=um)

        assert not np.any(np.isnan(model.variance_))

    def test_all_below_threshold(self, X_synth):
        """All assets below threshold -> all NaN variance."""
        model = EWVariance(half_life=10, min_observations=100)
        model.fit(X_synth[:10])

        assert np.all(np.isnan(model.variance_))

    def test_default_min_observations(self, X_synth):
        """Default min_observations equals int(half_life)."""
        model = EWVariance(half_life=40, min_observations=None)
        model.fit(X_synth)
        assert not np.any(np.isnan(model.variance_))

    def test_validation(self):
        """min_observations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_observations must be >= 1"):
            EWVariance(min_observations=0).fit(np.random.randn(10, 3))

    def test_tracked_across_partial_fit(self, X_synth):
        """Warm-up threshold is tracked across partial_fit calls."""
        X = X_synth.copy()
        X[:, 4] = np.nan
        um = np.ones((X_synth.shape[0], X_synth.shape[1]), dtype=bool)
        um[:, 4] = False

        model = EWVariance(half_life=10, min_observations=5)

        model.partial_fit(X[:50], active_mask=um[:50])
        assert np.isnan(model.variance_[4])

        model.partial_fit(X_synth[50:53])
        assert np.isnan(model.variance_[4])

        model.partial_fit(X_synth[53:60])
        assert not np.isnan(model.variance_[4])


# ---------------------------------------------------------------------------
# Bias correction
# ---------------------------------------------------------------------------


class TestEWVarianceBiasCorrection:
    """Tests for the bias correction mechanism."""

    def test_single_observation_equals_squared_return(self, X_synth):
        """With 1 valid observation, bias-corrected variance equals r^2."""
        X = np.full((10, 3), np.nan)
        X[:, 0] = X_synth[:10, 0]
        X[:, 1] = X_synth[:10, 1]
        X[9, 2] = X_synth[9, 2]

        model = EWVariance(half_life=5, min_observations=1)
        model.fit(X)

        np.testing.assert_almost_equal(model.variance_[2], X_synth[9, 2] ** 2)

    def test_matches_ewcovariance_diagonal(self, X_synth):
        """EWVariance matches the diagonal of EWCovariance on clean data."""
        from skfolio.moments import EWCovariance

        half_life = 20
        model_var = EWVariance(half_life=half_life, min_observations=1)
        model_var.fit(X_synth)

        model_cov = EWCovariance(half_life=half_life, min_observations=1)
        model_cov.fit(X_synth)

        np.testing.assert_allclose(
            model_var.variance_,
            np.diag(model_cov.covariance_),
            rtol=1e-10,
        )

    def test_matches_ewcovariance_diagonal_with_nan(self, X_synth):
        """EWVariance matches diagonal of EWCovariance with late-listed asset."""
        from skfolio.moments import EWCovariance

        half_life = 10
        X = X_synth.copy()
        X[:80, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:80, 3] = False

        model_var = EWVariance(half_life=half_life, min_observations=1)
        model_var.fit(X, active_mask=um)

        model_cov = EWCovariance(half_life=half_life, min_observations=1)
        model_cov.fit(X, active_mask=um)

        np.testing.assert_allclose(
            model_var.variance_,
            np.diag(model_cov.covariance_),
            rtol=1e-10,
        )

    def test_pandas_ewm_equivalence(self):
        """EWVariance (centered) matches pandas adjusted EWMA of r^2."""
        import pandas as pd

        rng = np.random.default_rng(seed=123)
        X = rng.standard_normal((100, 3)) * 0.01
        half_life = 20

        model = EWVariance(half_life=half_life, min_observations=1)
        model.fit(X)

        df_sq = pd.DataFrame(X**2)
        expected = df_sq.ewm(halflife=half_life, adjust=True).mean().iloc[-1].values

        np.testing.assert_allclose(model.variance_, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Batch fast-path
# ---------------------------------------------------------------------------


class TestEWVarianceBatchFastPath:
    """Tests for the vectorized batch fast-path."""

    def test_batch_matches_row_loop(self, X_synth):
        """Vectorized fast path produces same results as row-by-row."""
        model_batch = EWVariance(half_life=20, min_observations=1)
        model_batch.fit(X_synth)

        um = np.ones_like(X_synth, dtype=bool)
        model_loop = EWVariance(half_life=20, min_observations=1)
        model_loop.fit(X_synth, active_mask=um)

        np.testing.assert_allclose(
            model_batch.variance_, model_loop.variance_, rtol=1e-10
        )

    def test_batch_with_partial_fit(self, X_synth):
        """Batch fast-path across multiple partial_fit calls."""
        model_batch = EWVariance(half_life=20, min_observations=1)
        model_batch.fit(X_synth)

        model_stream = EWVariance(half_life=20, min_observations=1)
        model_stream.partial_fit(X_synth[:80])
        model_stream.partial_fit(X_synth[80:])

        np.testing.assert_allclose(
            model_batch.variance_, model_stream.variance_, rtol=1e-10
        )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEWVarianceEdgeCases:
    def test_all_assets_delisted(self, X_synth):
        """All assets leave universe -> all NaN variance."""
        um = np.ones_like(X_synth, dtype=bool)
        um[100:, :] = False
        X = X_synth.copy()
        X[100:, :] = np.nan

        model = EWVariance(half_life=10, min_observations=1)
        model.fit(X, active_mask=um)

        assert np.all(np.isnan(model.variance_))

    def test_first_row_all_nan(self, X_synth):
        """First row all NaN should not break initialization."""
        X = X_synth.copy()
        X[0, :] = np.nan

        model = EWVariance(half_life=10, min_observations=1)
        model.fit(X)

        assert not np.any(np.isnan(model.variance_))

    def test_listing_then_delisting(self, X_synth):
        """Asset is listed for a window then delisted -> NaN at end."""
        X = X_synth.copy()
        X[:50, 3] = np.nan
        X[150:, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:50, 3] = False
        um[150:, 3] = False

        model = EWVariance(half_life=10, min_observations=1)
        model.fit(X, active_mask=um)

        assert np.isnan(model.variance_[3])

    def test_delisting_then_relisting(self, X_synth):
        """Asset delisted then re-listed restarts bias correction."""
        X = X_synth.copy()
        X[80:130, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[80:130, 3] = False

        model = EWVariance(half_life=10, min_observations=1)
        model.fit(X, active_mask=um)

        assert not np.isnan(model.variance_[3])
        assert model.variance_[3] > 0

    def test_fit_resets_state(self, X_synth):
        """Calling fit() twice produces the same result (state is reset)."""
        model = EWVariance(half_life=10, min_observations=1)
        model.fit(X_synth[:50])
        var_first = model.variance_.copy()

        model.fit(X_synth[:50])
        np.testing.assert_array_equal(model.variance_, var_first)

        model.fit(X_synth[50:150])
        assert not np.allclose(model.variance_, var_first)

    def test_assume_centered_false_with_nan(self, X_synth):
        """assume_centered=False with late listing and delisting."""
        X = X_synth.copy()
        X[:50, 4] = np.nan
        X[150:, 3] = np.nan
        um = np.ones_like(X, dtype=bool)
        um[:50, 4] = False
        um[150:, 3] = False

        model = EWVariance(half_life=10, assume_centered=False, min_observations=1)
        model.fit(X, active_mask=um)

        assert np.isnan(model.variance_[3])
        assert np.isnan(model.location_[3])
        assert np.isfinite(model.variance_[4])
        assert np.isfinite(model.location_[4])
        assert model.variance_[4] > 0
