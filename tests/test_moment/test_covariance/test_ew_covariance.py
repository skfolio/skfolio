"""Tests for EWCovariance estimator.

Covers:
- Basic EWMA functionality and parameter validation (original + extended)
- NaN handling: late listings, delistings, holidays
- active_mask interaction
- min_observations warm-up
- partial_fit streaming
- assume_centered=False with NaN
- Edge cases
"""

import numpy as np
import pandas as pd
import pytest

from skfolio.moments import EWCovariance

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


class TestEWCovariance:
    """Original tests using the real SP500 dataset (20 assets)."""

    def test_ew_covariance_default_half_life(self, X):
        """Test EWCovariance with default half_life=40."""
        model = EWCovariance()
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        # Should be symmetric and PSD
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_ew_covariance_custom_half_life(self, X):
        """Test EWCovariance with custom half_life values."""
        model = EWCovariance(half_life=20)
        model.fit(X)
        assert model.covariance_.shape == (20, 20)
        np.testing.assert_almost_equal(model.covariance_, model.covariance_.T)

    def test_ew_covariance_invalid_half_life(self, X):
        """Test that invalid half_life raises an error."""
        with pytest.raises(ValueError, match="half_life must be positive"):
            EWCovariance(half_life=0).fit(X)
        with pytest.raises(ValueError, match="half_life must be positive"):
            EWCovariance(half_life=-10).fit(X)

    def test_ew_covariance_legacy_alpha(self, X):
        """Test EWCovariance with legacy alpha parameter (backward compat)."""
        with pytest.warns(FutureWarning, match="alpha.*deprecated"):
            model = EWCovariance(alpha=0.02)
            model.fit(X)

        assert model.covariance_.shape == (20, 20)

        # alpha=0.02 means half_life=34, which is half_life ~= 34.3
        model_new = EWCovariance(half_life=34.3)
        model_new.fit(X)
        np.testing.assert_almost_equal(
            model.covariance_, model_new.covariance_, decimal=3
        )

    def test_ew_covariance_alpha_half_life_conflict(self, X):
        """Test that specifying both alpha and half_life raises an error."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            EWCovariance(alpha=0.2, half_life=20).fit(X)


# ---------------------------------------------------------------------------
# Core functionality (extends originals with new features)
# ---------------------------------------------------------------------------


class TestCoreFunctionality:
    """Tests for features not covered by the original TestEWCovariance."""

    def test_window_size(self, X_synth):
        """window_size truncates data to last N observations."""
        model_full = EWCovariance(half_life=10, min_observations=1)
        model_full.fit(X_synth)

        model_win = EWCovariance(half_life=10, min_observations=1, window_size=50)
        model_win.fit(X_synth)

        assert not np.allclose(model_full.covariance_, model_win.covariance_)

    def test_partial_fit_matches_batch(self, X_synth):
        """Splitting data across partial_fit calls matches single fit."""
        model_batch = EWCovariance(half_life=20, min_observations=1)
        model_batch.fit(X_synth)

        model_stream = EWCovariance(half_life=20, min_observations=1)
        model_stream.partial_fit(X_synth[:80])
        model_stream.partial_fit(X_synth[80:150])
        model_stream.partial_fit(X_synth[150:])

        np.testing.assert_array_almost_equal(
            model_batch.covariance_, model_stream.covariance_
        )

    def test_window_size_applies_only_to_first_partial_fit(self, X_synth):
        """window_size truncates only the first partial_fit batch."""
        um = np.ones_like(X_synth, dtype=bool)
        um[:40, 4] = False

        model_windowed = EWCovariance(half_life=10, min_observations=1, window_size=50)
        model_windowed.partial_fit(X_synth[:80], active_mask=um[:80])
        model_windowed.partial_fit(X_synth[80:150], active_mask=um[80:150])

        model_ref = EWCovariance(half_life=10, min_observations=1)
        model_ref.partial_fit(X_synth[30:80], active_mask=um[30:80])
        model_ref.partial_fit(X_synth[80:150], active_mask=um[80:150])

        np.testing.assert_array_almost_equal(
            model_windowed.covariance_, model_ref.covariance_
        )

    def test_assume_centered_false(self, X_synth):
        """assume_centered=False tracks EWMA mean in location_."""
        model = EWCovariance(half_life=20, assume_centered=False, min_observations=1)
        model.fit(X_synth)

        assert model.covariance_.shape == (X_synth.shape[1], X_synth.shape[1])
        assert np.all(np.isfinite(model.location_))
        assert not np.allclose(model.location_, 0.0)

    def test_min_observations_validation(self, X_synth):
        """min_observations < 1 raises ValueError."""
        with pytest.raises(ValueError, match="min_observations must be >= 1"):
            EWCovariance(min_observations=0).fit(X_synth)

    def test_min_observations_default(self, X_synth):
        """Default min_observations equals int(half_life)."""
        model = EWCovariance(half_life=40, min_observations=None)
        model.fit(X_synth)
        # 200 obs >= 40 threshold -> all active
        assert not np.any(np.isnan(model.covariance_))

    def test_fit_resets_state(self, X_synth):
        """Calling fit() twice produces the same result (state is reset)."""
        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_synth[:50])
        cov_first = model.covariance_.copy()

        # Same data -> identical result
        model.fit(X_synth[:50])
        np.testing.assert_array_equal(model.covariance_, cov_first)

        # Different data -> different result
        model.fit(X_synth[50:150])
        assert not np.allclose(model.covariance_, cov_first)


# ---------------------------------------------------------------------------
# NaN handling: late listing
# ---------------------------------------------------------------------------


class TestNaNLateListing:
    """Assets entering the universe mid-stream (NaN at beginning)."""

    def test_single_asset(self, X_synth):
        """Late-listed asset gets valid covariance after enough data."""
        X_nan = X_synth.copy()
        X_nan[:80, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:80, 3] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert not np.any(np.isnan(model.covariance_))

    def test_covariance_differs_from_full_data(self, X_synth):
        """Late listing produces different estimate than full data."""
        model_full = EWCovariance(half_life=10, min_observations=1)
        model_full.fit(X_synth)

        X_nan = X_synth.copy()
        X_nan[:100, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:100, 4] = False
        model_late = EWCovariance(half_life=10, min_observations=1)
        model_late.fit(X_nan, active_mask=um)

        # Unaffected assets: same covariance
        np.testing.assert_array_almost_equal(
            model_full.covariance_[:4, :4],
            model_late.covariance_[:4, :4],
            decimal=3,
        )
        # Affected asset: different covariance
        assert not np.allclose(
            model_full.covariance_[4, :], model_late.covariance_[4, :]
        )

    def test_staggered_multiple_assets(self, X_synth):
        """Multiple assets entering at different times."""
        X_nan = X_synth.copy()
        X_nan[:30, 2] = np.nan
        X_nan[:80, 3] = np.nan
        X_nan[:150, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:30, 2] = False
        um[:80, 3] = False
        um[:150, 4] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert not np.any(np.isnan(model.covariance_))
        np.testing.assert_array_almost_equal(model.covariance_, model.covariance_.T)


# ---------------------------------------------------------------------------
# NaN handling: delisting
# ---------------------------------------------------------------------------


class TestNaNDelisting:
    """Assets leaving the universe (NaN at end, active_mask=False)."""

    def test_single_asset(self, X_synth):
        """Delisted asset has NaN row/col; active submatrix is PSD."""
        X_nan = X_synth.copy()
        X_nan[100:, 2] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[100:, 2] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        # Delisted -> NaN row and column
        assert np.all(np.isnan(model.covariance_[2, :]))
        assert np.all(np.isnan(model.covariance_[:, 2]))
        # Active submatrix is finite and PSD
        active = [0, 1, 3, 4]
        sub = model.covariance_[np.ix_(active, active)]
        assert not np.any(np.isnan(sub))
        eigvals = np.linalg.eigvalsh(sub)
        assert np.all(eigvals >= -1e-10)

    def test_multiple_assets(self, X_synth):
        """Multiple assets delisted at different times."""
        X_nan = X_synth.copy()
        X_nan[80:, 3] = np.nan
        X_nan[120:, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[80:, 3] = False
        um[120:, 4] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert np.all(np.isnan(model.covariance_[3, :]))
        assert np.all(np.isnan(model.covariance_[4, :]))
        core = [0, 1, 2]
        assert not np.any(np.isnan(model.covariance_[np.ix_(core, core)]))


# ---------------------------------------------------------------------------
# NaN handling: holidays (NaN in middle, active_mask=True)
# ---------------------------------------------------------------------------


class TestNaNHolidays:
    """Missing data for in-universe assets (covariance freezes, no NaN)."""

    def test_holiday_freeze_no_nan(self, X_synth):
        """Holiday NaN freezes covariance, no NaN in output."""
        X_nan = X_synth.copy()
        X_nan[80:90, 1] = np.nan

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan)

        assert not np.any(np.isnan(model.covariance_))

    def test_holiday_changes_affected_entries_only(self, X_synth):
        """Holiday changes cross-covariances of affected asset but not others."""
        model_full = EWCovariance(half_life=10, min_observations=1)
        model_full.fit(X_synth)

        X_nan = X_synth.copy()
        X_nan[90:100, 2] = np.nan
        model_hol = EWCovariance(half_life=10, min_observations=1)
        model_hol.fit(X_nan)

        # Affected asset's row differs
        assert not np.allclose(
            model_full.covariance_[2, :], model_hol.covariance_[2, :]
        )
        # Unaffected pairs identical
        np.testing.assert_array_almost_equal(
            model_full.covariance_[np.ix_([0, 1], [0, 1])],
            model_hol.covariance_[np.ix_([0, 1], [0, 1])],
        )

    def test_holiday_vs_delisting_distinction(self, X_synth):
        """Same NaN pattern, different active_mask -> different output."""
        X_nan = X_synth.copy()
        X_nan[150:, 3] = np.nan

        # Holiday (default active_mask=True) -> freeze, no NaN
        model_hol = EWCovariance(half_life=10, min_observations=1)
        model_hol.fit(X_nan)
        assert not np.any(np.isnan(model_hol.covariance_))

        # Delisting (active_mask=False) -> NaN
        um = np.ones_like(X_nan, dtype=bool)
        um[150:, 3] = False
        model_del = EWCovariance(half_life=10, min_observations=1)
        model_del.fit(X_nan, active_mask=um)
        assert np.all(np.isnan(model_del.covariance_[3, :]))


# ---------------------------------------------------------------------------
# active_mask
# ---------------------------------------------------------------------------


class TestUniverseMask:
    """Validation and behavior of active_mask."""

    def test_shape_mismatch_raises(self, X_synth):
        um_bad = np.ones((X_synth.shape[0] + 1, X_synth.shape[1]), dtype=bool)
        with pytest.raises(ValueError, match="active_mask shape"):
            EWCovariance(half_life=10).fit(X_synth, active_mask=um_bad)

    def test_none_equivalent_to_all_true(self, X_synth):
        """active_mask=None behaves identically to all-True mask."""
        model_none = EWCovariance(half_life=20, min_observations=1)
        model_none.fit(X_synth)

        um = np.ones_like(X_synth, dtype=bool)
        model_mask = EWCovariance(half_life=20, min_observations=1)
        model_mask.fit(X_synth, active_mask=um)

        np.testing.assert_array_almost_equal(
            model_none.covariance_, model_mask.covariance_
        )

    def test_truncated_with_window_size(self, X_synth):
        """window_size truncates both X and active_mask consistently."""
        um = np.ones_like(X_synth, dtype=bool)
        um[:, 4] = False

        model = EWCovariance(half_life=10, min_observations=1, window_size=50)
        model.fit(X_synth, active_mask=um)

        assert np.all(np.isnan(model.covariance_[4, :]))

    def test_valid_returns_ignored_outside_universe(self, X_synth):
        """Non-NaN returns with active_mask=False are still ignored."""
        um = np.ones_like(X_synth, dtype=bool)
        um[:, 4] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_synth, active_mask=um)

        assert np.all(np.isnan(model.covariance_[4, :]))


# ---------------------------------------------------------------------------
# min_observations warm-up
# ---------------------------------------------------------------------------


class TestMinObservations:
    """Warm-up behavior of min_observations."""

    def test_below_threshold_is_nan(self, X_synth):
        """Assets below min_observations have NaN covariance."""
        X_nan = X_synth[:30].copy()
        X_nan[:25, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:25, 4] = False

        # Asset 4: 5 obs < 10 threshold -> NaN
        model = EWCovariance(half_life=10, min_observations=10)
        model.fit(X_nan, active_mask=um)

        assert np.all(np.isnan(model.covariance_[4, :]))
        assert np.all(np.isnan(model.covariance_[:, 4]))
        # Others: 30 obs >= 10 -> active
        core = [0, 1, 2, 3]
        assert not np.any(np.isnan(model.covariance_[np.ix_(core, core)]))

    def test_crosses_threshold(self, X_synth):
        """Asset becomes active once it crosses min_observations."""
        X_nan = X_synth.copy()
        X_nan[:180, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:180, 4] = False

        # 20 obs >= 15 threshold -> active
        model = EWCovariance(half_life=10, min_observations=15)
        model.fit(X_nan, active_mask=um)

        assert not np.any(np.isnan(model.covariance_))

    def test_all_below_threshold(self, X_synth):
        """All assets below threshold -> all NaN covariance."""
        model = EWCovariance(half_life=10, min_observations=100)
        model.fit(X_synth[:10])

        assert np.all(np.isnan(model.covariance_))

    def test_tracked_across_partial_fit(self, X_synth):
        """Warm-up threshold is tracked across partial_fit calls."""
        X_nan = X_synth.copy()
        X_nan[:, 4] = np.nan
        um = np.ones((X_synth.shape[0], X_synth.shape[1]), dtype=bool)
        um[:, 4] = False

        model = EWCovariance(half_life=10, min_observations=5)

        # Batch 1: asset 4 not in universe
        model.partial_fit(X_nan[:50], active_mask=um[:50])
        assert np.all(np.isnan(model.covariance_[4, :]))

        # Batch 2: asset 4 enters with 3 obs (below threshold)
        model.partial_fit(X_synth[50:53])
        assert np.all(np.isnan(model.covariance_[4, :]))

        # Batch 3: 10 total obs >= 5 threshold -> active
        model.partial_fit(X_synth[53:60])
        assert not np.any(np.isnan(model.covariance_[4, :]))


# ---------------------------------------------------------------------------
# partial_fit streaming with NaN
# ---------------------------------------------------------------------------


class TestPartialFit:
    """Streaming/online updates via partial_fit with NaN."""

    def test_nan_matches_batch(self, X_synth):
        """partial_fit with NaN should match batch fit."""
        X_nan = X_synth.copy()
        X_nan[:50, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:50, 3] = False

        model_batch = EWCovariance(half_life=10, min_observations=1)
        model_batch.fit(X_nan, active_mask=um)

        model_stream = EWCovariance(half_life=10, min_observations=1)
        model_stream.partial_fit(X_nan[:80], active_mask=um[:80])
        model_stream.partial_fit(X_nan[80:], active_mask=um[80:])

        np.testing.assert_array_almost_equal(
            model_batch.covariance_, model_stream.covariance_
        )

    def test_asset_leaves_universe(self, X_synth):
        """Asset leaving universe across partial_fit calls."""
        model = EWCovariance(half_life=10, min_observations=1)

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


class TestAssumeCenteredFalse:
    """Tests for assume_centered=False (EWMA mean tracking) with NaN."""

    def test_location_nan_for_delisted(self, X_synth):
        X_nan = X_synth.copy()
        X_nan[100:, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[100:, 3] = False

        model = EWCovariance(half_life=10, assume_centered=False, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert np.isnan(model.location_[3])
        assert np.all(np.isfinite(model.location_[[0, 1, 2, 4]]))

    def test_location_frozen_during_holiday(self, X_synth):
        X_nan = X_synth.copy()
        X_nan[180:190, 1] = np.nan

        model = EWCovariance(half_life=10, assume_centered=False, min_observations=1)
        model.fit(X_nan)

        assert np.all(np.isfinite(model.location_))

    def test_location_zeros_when_centered(self, X_synth):
        """location_ is always zeros when assume_centered=True."""
        X_nan = X_synth.copy()
        X_nan[100:, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[100:, 3] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        np.testing.assert_array_equal(model.location_, np.zeros(X_synth.shape[1]))

    def test_demeaned_covariance_psd(self, X_synth):
        """Covariance with demeaning and late listing is PSD."""
        X_nan = X_synth.copy()
        X_nan[:50, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:50, 4] = False

        model = EWCovariance(half_life=10, assume_centered=False, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert not np.any(np.isnan(model.covariance_))
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)


# ---------------------------------------------------------------------------
# Inference on NaN-aware covariance
# ---------------------------------------------------------------------------


class TestInference:
    """Tests for score and mahalanobis on NaN-aware covariance estimates."""

    def test_mahalanobis_and_score_with_delisted_asset(self, X_synth):
        """Inference excludes inactive assets row by row instead of failing."""
        X_nan = X_synth.copy()
        X_nan[120:, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[120:, 4] = False

        model = EWCovariance(half_life=10, assume_centered=False, min_observations=1)
        model.fit(X_nan, active_mask=um)

        distances = model.mahalanobis(X_nan)
        score = model.score(X_nan)

        assert distances.shape == (len(X_nan),)
        assert np.all(np.isfinite(distances))
        assert np.isfinite(score)

        single_distance = model.mahalanobis(X_nan[150])
        assert np.isscalar(single_distance)
        assert np.isfinite(single_distance)

    def test_score_all_invalid_rows_raises(self, X_synth):
        """score raises when no row has any finite retained observation."""
        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_synth)

        X_test = np.full((3, X_synth.shape[1]), np.nan)
        with pytest.raises(ValueError, match="finite retained observation"):
            model.score(X_test)

    def test_score_handles_holiday_missing_values(self, X_synth):
        """score remains finite when retained assets have row-wise holiday NaN."""
        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_synth)

        X_test = X_synth.copy()
        X_test[10:20, 1] = np.nan
        X_test[40:45, [0, 3]] = np.nan
        score = model.score(X_test)
        assert np.isfinite(score)

    def test_mahalanobis_validates_feature_names(self, X_synth):
        """mahalanobis still validates dataframe column order with NaN support."""
        columns = ["a", "b", "c", "d", "e"]
        X = pd.DataFrame(X_synth, columns=columns)
        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X)

        with pytest.raises(ValueError, match="feature names"):
            model.mahalanobis(X[columns[::-1]])

    def test_mahalanobis_raises_on_nan_in_retained_subspace(self, X_synth):
        """mahalanobis rejects NaN on assets retained by the fitted mask."""
        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_synth)

        X_test = X_synth.copy()
        X_test[0, 0] = np.nan
        with pytest.raises(ValueError, match="inference subspace"):
            model.mahalanobis(X_test)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_assets_delisted(self, X_synth):
        """All assets leave universe -> all NaN covariance."""
        um = np.ones_like(X_synth, dtype=bool)
        um[100:, :] = False
        X_nan = X_synth.copy()
        X_nan[100:, :] = np.nan

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert np.all(np.isnan(model.covariance_))

    def test_single_observation(self, X_synth):
        """Asset with exactly one valid observation."""
        X_nan = np.full((10, 3), np.nan)
        X_nan[:, 0] = X_synth[:10, 0]
        X_nan[:, 1] = X_synth[:10, 1]
        X_nan[9, 2] = X_synth[9, 2]  # 1 obs

        model = EWCovariance(half_life=5, min_observations=1)
        model.fit(X_nan)

        assert not np.any(np.isnan(model.covariance_))
        # With bias correction, variance for 1-obs asset is r^2
        # (correction factor 1/(1-lambda^1) normalizes it)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

    def test_first_row_all_nan(self, X_synth):
        """First row all NaN should not break initialization."""
        X_nan = X_synth.copy()
        X_nan[0, :] = np.nan

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan)

        assert not np.any(np.isnan(model.covariance_))

    def test_nearest_with_nan_submatrix(self, X_synth_wide):
        """nearest=True projects active submatrix to PD, preserves NaN frame."""
        X_nan = X_synth_wide.copy()
        X_nan[:, -3:] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:, -3:] = False

        model = EWCovariance(half_life=10, min_observations=1, nearest=True)
        model.fit(X_nan, active_mask=um)

        active = list(range(17))
        sub = model.covariance_[np.ix_(active, active)]
        eigvals = np.linalg.eigvalsh(sub)
        assert np.all(eigvals >= -1e-10)

    def test_nearest_false_preserves_nan(self, X_synth):
        """nearest=False still works with NaN covariance."""
        X_nan = X_synth.copy()
        X_nan[100:, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[100:, 4] = False

        model = EWCovariance(half_life=10, min_observations=1, nearest=False)
        model.fit(X_nan, active_mask=um)

        assert np.all(np.isnan(model.covariance_[4, :]))
        active = [0, 1, 2, 3]
        assert not np.any(np.isnan(model.covariance_[np.ix_(active, active)]))

    def test_listing_then_delisting(self, X_synth):
        """Asset is listed for a window then delisted -> NaN at end."""
        X_nan = X_synth.copy()
        X_nan[:50, 3] = np.nan
        X_nan[150:, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:50, 3] = False
        um[150:, 3] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert np.all(np.isnan(model.covariance_[3, :]))

    def test_delisting_then_relisting(self, X_synth):
        """Asset delisted then re-listed gets correct bias correction."""
        X_nan = X_synth.copy()
        X_nan[80:130, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[80:130, 3] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert not np.any(np.isnan(model.covariance_[3, :]))
        np.testing.assert_array_almost_equal(model.covariance_, model.covariance_.T)
        eigvals = np.linalg.eigvalsh(model.covariance_)
        assert np.all(eigvals >= -1e-10)

        # Variance of re-listed asset must be in a plausible range (not
        # crushed by stale obs_count bias).
        ref_model = EWCovariance(half_life=10, min_observations=1)
        ref_model.fit(X_synth)
        ratio = model.covariance_[3, 3] / ref_model.covariance_[3, 3]
        assert ratio > 0.1, f"Re-listed variance too small (ratio={ratio:.4f})"

    def test_symmetry_with_mixed_nan_pattern(self, X_synth):
        """Late listing + holiday + delisting -> symmetric output."""
        X_nan = X_synth.copy()
        X_nan[:30, 1] = np.nan  # late listing
        X_nan[80:90, 2] = np.nan  # holiday
        X_nan[150:, 4] = np.nan  # delisting
        um = np.ones_like(X_nan, dtype=bool)
        um[:30, 1] = False
        um[150:, 4] = False

        model = EWCovariance(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        cov = model.covariance_
        nan_mask = np.isnan(cov)
        # NaN pattern symmetric
        np.testing.assert_array_equal(nan_mask, nan_mask.T)
        # Finite part symmetric
        finite = np.isfinite(cov)
        if np.any(finite):
            np.testing.assert_array_almost_equal(
                np.where(finite, cov, 0.0),
                np.where(finite, cov.T, 0.0),
            )


# ---------------------------------------------------------------------------
# Late listing PSD regression
# ---------------------------------------------------------------------------


class TestLateListingPSD:
    """Regression test: late-listed asset must not break positive definiteness."""

    def test_late_listing_psd_around_half_life(self):
        """Covariance stays PSD at the exact min_observations boundary.

        Reproduces the issue where a new asset entering on day 500 with
        half_life=80 produced a non-PD matrix right when its entries were
        first exposed (day 580 = 500 + half_life).
        """
        from skfolio.datasets import load_sp500_dataset
        from skfolio.preprocessing import prices_to_returns

        prices = load_sp500_dataset()
        X = prices_to_returns(prices)
        X.loc[X.index[:500], "AAPL"] = np.nan

        model = EWCovariance(half_life=80, nearest=False)

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

        model = EWCovariance(half_life=half_life, nearest=False)
        model.fit(X)

        cov = model.covariance_
        active = np.isfinite(np.diag(cov))
        sub = cov[np.ix_(active, active)]
        eigvals = np.linalg.eigvalsh(sub)
        assert np.all(eigvals >= -1e-10), (
            f"Non-PD with half_life={half_life}: min eigval={eigvals.min():.2e}"
        )
