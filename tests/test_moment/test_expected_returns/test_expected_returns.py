from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context

from skfolio.moments import (
    EWMu,
    EmpiricalMu,
    EquilibriumMu,
    ImpliedCovariance,
    ShrunkMu,
    ShrunkMuMethods,
)


class TestEmpiricalMu:
    def test_empirical_mu(self, X):
        model = EmpiricalMu()
        model.fit(X)
        np.testing.assert_almost_equal(model.mu_, np.array(X).mean(axis=0))


class TestEWMu:
    def test_ew_mu_default(self, X):
        """Test EWMu with default half_life=40."""
        model = EWMu()
        model.fit(X)
        assert model.mu_.shape == (20,)
        # With half_life=40 (decay ~0.983), estimates are more stable
        np.testing.assert_almost_equal(
            model.mu_,
            np.array(
                [
                    -2.35639920e-03,
                    -2.16298397e-03,
                    -6.24939820e-04,
                    1.59720195e-03,
                    1.32326000e-03,
                    1.13191618e-03,
                    9.15787880e-04,
                    5.73781161e-04,
                    1.20165990e-03,
                    8.58470553e-04,
                    1.52664306e-03,
                    2.45540847e-03,
                    -7.87452631e-04,
                    7.28482574e-04,
                    9.98412295e-04,
                    1.24476313e-03,
                    -1.34977227e-03,
                    3.08834478e-04,
                    3.41145908e-04,
                    1.61234395e-03,
                ]
            ),
        )

    def test_ew_mu_deprecated_alpha(self, X):
        """Test backward compatibility with deprecated alpha parameter."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = EWMu(alpha=0.2)
            model.fit(X)
            # Check deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, FutureWarning)
            assert "alpha" in str(w[0].message)

        # These values were computed with the original pandas-based implementation
        np.testing.assert_almost_equal(
            model.mu_,
            np.array(
                [
                    -1.24726372e-02,
                    -9.45067113e-03,
                    1.61657194e-03,
                    -1.83966453e-03,
                    2.83363890e-03,
                    9.19409711e-04,
                    -2.64229605e-03,
                    -5.19784162e-04,
                    1.54353089e-03,
                    6.95047714e-05,
                    4.90700510e-04,
                    6.30137508e-04,
                    -6.19996526e-03,
                    -4.15448322e-04,
                    -3.22012113e-03,
                    4.39614645e-05,
                    -1.16913798e-02,
                    -5.00326373e-04,
                    -4.56242603e-03,
                    1.86622274e-03,
                ]
            ),
        )

    def test_ew_mu_partial_fit(self, X):
        """Test streaming/online updates with partial_fit."""
        X_arr = np.array(X)

        # Fit all at once
        model1 = EWMu(half_life=20)
        model1.fit(X_arr)

        # Fit in chunks using partial_fit
        model2 = EWMu(half_life=20)
        model2.partial_fit(X_arr[:100])
        model2.partial_fit(X_arr[100:500])
        model2.partial_fit(X_arr[500:])

        np.testing.assert_almost_equal(model1.mu_, model2.mu_)

    def test_ew_mu_window_size_first_partial_fit_only(self, X_synth):
        """window_size truncates only the first partial_fit batch."""
        um = np.ones_like(X_synth, dtype=bool)
        um[:40, 4] = False

        model_windowed = EWMu(half_life=10, min_observations=1, window_size=50)
        model_windowed.partial_fit(X_synth[:80], active_mask=um[:80])
        model_windowed.partial_fit(X_synth[80:150], active_mask=um[80:150])

        model_ref = EWMu(half_life=10, min_observations=1)
        model_ref.partial_fit(X_synth[30:80], active_mask=um[30:80])
        model_ref.partial_fit(X_synth[80:150], active_mask=um[80:150])

        np.testing.assert_almost_equal(model_windowed.mu_, model_ref.mu_)

    def test_ew_mu_window_size(self, X):
        """Test window_size parameter."""
        X_arr = np.array(X)

        # With window_size, only last N observations are used
        model1 = EWMu(half_life=20, window_size=100)
        model1.fit(X_arr)

        # Manually fit on last 100 observations
        model2 = EWMu(half_life=20)
        model2.fit(X_arr[-100:])

        np.testing.assert_almost_equal(model1.mu_, model2.mu_)

    def test_ew_mu_validation(self):
        """Test parameter validation."""
        # Cannot specify both alpha and half_life
        with pytest.raises(ValueError, match="Cannot specify both"):
            EWMu(alpha=0.2, half_life=40).fit(np.random.randn(10, 3))

        # half_life must be positive
        with pytest.raises(ValueError, match="half_life must be positive"):
            EWMu(half_life=-5).fit(np.random.randn(10, 3))

        # alpha must be in (0, 1)
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="alpha must satisfy"):
                EWMu(alpha=1.5).fit(np.random.randn(10, 3))

    def test_ew_mu_min_observations_validation(self):
        """Test min_observations parameter validation."""
        with pytest.raises(ValueError, match="min_observations must be >= 1"):
            EWMu(min_observations=0).fit(np.random.randn(10, 3))


# ---------------------------------------------------------------------------
# Fixtures for NaN tests
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)


@pytest.fixture
def X_synth(rng):
    """Small synthetic return matrix (200 obs, 5 assets), no NaN."""
    return rng.standard_normal((200, 5)) * 0.01


# ---------------------------------------------------------------------------
# NaN handling: late listing
# ---------------------------------------------------------------------------


class TestEWMuLateListing:
    """Assets entering the universe mid-stream (NaN at beginning)."""

    def test_single_asset(self, X_synth):
        """Late-listed asset gets valid mu after enough data."""
        X_nan = X_synth.copy()
        X_nan[:80, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:80, 3] = False

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert not np.any(np.isnan(model.mu_))

    def test_mu_differs_from_full_data(self, X_synth):
        """Late listing produces different estimate than full data."""
        model_full = EWMu(half_life=10, min_observations=1)
        model_full.fit(X_synth)

        X_nan = X_synth.copy()
        X_nan[:100, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:100, 4] = False
        model_late = EWMu(half_life=10, min_observations=1)
        model_late.fit(X_nan, active_mask=um)

        # Unaffected assets: same mu
        np.testing.assert_array_almost_equal(
            model_full.mu_[:4],
            model_late.mu_[:4],
            decimal=10,
        )
        # Affected asset: different mu
        assert not np.isclose(model_full.mu_[4], model_late.mu_[4])

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

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert not np.any(np.isnan(model.mu_))

    def test_bias_correction_single_obs(self, X_synth):
        """With 1 valid observation, bias-corrected mu equals that return."""
        X_nan = np.full((10, 3), np.nan)
        X_nan[:, 0] = X_synth[:10, 0]
        X_nan[:, 1] = X_synth[:10, 1]
        X_nan[9, 2] = X_synth[9, 2]

        model = EWMu(half_life=5, min_observations=1)
        model.fit(X_nan)

        np.testing.assert_almost_equal(model.mu_[2], X_synth[9, 2])


# ---------------------------------------------------------------------------
# NaN handling: delisting
# ---------------------------------------------------------------------------


class TestEWMuDelisting:
    """Assets leaving the universe (NaN at end, active_mask=False)."""

    def test_single_asset(self, X_synth):
        """Delisted asset has NaN mu; active assets are finite."""
        X_nan = X_synth.copy()
        X_nan[100:, 2] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[100:, 2] = False

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert np.isnan(model.mu_[2])
        active = [0, 1, 3, 4]
        assert not np.any(np.isnan(model.mu_[active]))

    def test_multiple_assets(self, X_synth):
        """Multiple assets delisted at different times."""
        X_nan = X_synth.copy()
        X_nan[80:, 3] = np.nan
        X_nan[120:, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[80:, 3] = False
        um[120:, 4] = False

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert np.isnan(model.mu_[3])
        assert np.isnan(model.mu_[4])
        assert not np.any(np.isnan(model.mu_[[0, 1, 2]]))


# ---------------------------------------------------------------------------
# NaN handling: holidays (NaN in middle, active_mask=True)
# ---------------------------------------------------------------------------


class TestEWMuHolidays:
    """Missing data for in-universe assets (mean freezes, no NaN)."""

    def test_holiday_freeze_no_nan(self, X_synth):
        """Holiday NaN freezes mu, no NaN in output."""
        X_nan = X_synth.copy()
        X_nan[80:90, 1] = np.nan

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan)

        assert not np.any(np.isnan(model.mu_))

    def test_holiday_changes_affected_asset_only(self, X_synth):
        """Holiday changes mu of affected asset but not others."""
        model_full = EWMu(half_life=10, min_observations=1)
        model_full.fit(X_synth)

        X_nan = X_synth.copy()
        X_nan[90:100, 2] = np.nan
        model_hol = EWMu(half_life=10, min_observations=1)
        model_hol.fit(X_nan)

        assert not np.isclose(model_full.mu_[2], model_hol.mu_[2])
        np.testing.assert_array_almost_equal(
            model_full.mu_[[0, 1]],
            model_hol.mu_[[0, 1]],
        )

    def test_holiday_vs_delisting_distinction(self, X_synth):
        """Same NaN pattern, different active_mask -> different output."""
        X_nan = X_synth.copy()
        X_nan[150:, 3] = np.nan

        model_hol = EWMu(half_life=10, min_observations=1)
        model_hol.fit(X_nan)
        assert not np.any(np.isnan(model_hol.mu_))

        um = np.ones_like(X_nan, dtype=bool)
        um[150:, 3] = False
        model_del = EWMu(half_life=10, min_observations=1)
        model_del.fit(X_nan, active_mask=um)
        assert np.isnan(model_del.mu_[3])


# ---------------------------------------------------------------------------
# active_mask
# ---------------------------------------------------------------------------


class TestEWMuUniverseMask:
    """Validation and behavior of active_mask."""

    def test_shape_mismatch_raises(self, X_synth):
        um_bad = np.ones((X_synth.shape[0] + 1, X_synth.shape[1]), dtype=bool)
        with pytest.raises(ValueError, match="active_mask shape"):
            EWMu(half_life=10).fit(X_synth, active_mask=um_bad)

    def test_none_equivalent_to_all_true(self, X_synth):
        """active_mask=None behaves identically to all-True mask."""
        model_none = EWMu(half_life=20, min_observations=1)
        model_none.fit(X_synth)

        um = np.ones_like(X_synth, dtype=bool)
        model_mask = EWMu(half_life=20, min_observations=1)
        model_mask.fit(X_synth, active_mask=um)

        np.testing.assert_array_almost_equal(model_none.mu_, model_mask.mu_)

    def test_truncated_with_window_size(self, X_synth):
        """window_size truncates both X and active_mask consistently."""
        um = np.ones_like(X_synth, dtype=bool)
        um[:, 4] = False

        model = EWMu(half_life=10, min_observations=1, window_size=50)
        model.fit(X_synth, active_mask=um)

        assert np.isnan(model.mu_[4])

    def test_valid_returns_ignored_outside_universe(self, X_synth):
        """Non-NaN returns with active_mask=False are still ignored."""
        um = np.ones_like(X_synth, dtype=bool)
        um[:, 4] = False

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_synth, active_mask=um)

        assert np.isnan(model.mu_[4])


# ---------------------------------------------------------------------------
# min_observations warm-up
# ---------------------------------------------------------------------------


class TestEWMuMinObservations:
    """Warm-up behavior of min_observations."""

    def test_below_threshold_is_nan(self, X_synth):
        """Assets below min_observations have NaN mu."""
        X_nan = X_synth[:30].copy()
        X_nan[:25, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:25, 4] = False

        model = EWMu(half_life=10, min_observations=10)
        model.fit(X_nan, active_mask=um)

        assert np.isnan(model.mu_[4])
        assert not np.any(np.isnan(model.mu_[[0, 1, 2, 3]]))

    def test_crosses_threshold(self, X_synth):
        """Asset becomes active once it crosses min_observations."""
        X_nan = X_synth.copy()
        X_nan[:180, 4] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:180, 4] = False

        model = EWMu(half_life=10, min_observations=15)
        model.fit(X_nan, active_mask=um)

        assert not np.any(np.isnan(model.mu_))

    def test_all_below_threshold(self, X_synth):
        """All assets below threshold -> all NaN mu."""
        model = EWMu(half_life=10, min_observations=100)
        model.fit(X_synth[:10])

        assert np.all(np.isnan(model.mu_))

    def test_default_min_observations(self, X_synth):
        """Default min_observations equals int(half_life)."""
        model = EWMu(half_life=40, min_observations=None)
        model.fit(X_synth)
        assert not np.any(np.isnan(model.mu_))

    def test_tracked_across_partial_fit(self, X_synth):
        """Warm-up threshold is tracked across partial_fit calls."""
        X_nan = X_synth.copy()
        X_nan[:, 4] = np.nan
        um = np.ones((X_synth.shape[0], X_synth.shape[1]), dtype=bool)
        um[:, 4] = False

        model = EWMu(half_life=10, min_observations=5)

        model.partial_fit(X_nan[:50], active_mask=um[:50])
        assert np.isnan(model.mu_[4])

        model.partial_fit(X_synth[50:53])
        assert np.isnan(model.mu_[4])

        model.partial_fit(X_synth[53:60])
        assert not np.isnan(model.mu_[4])


# ---------------------------------------------------------------------------
# partial_fit streaming with NaN
# ---------------------------------------------------------------------------


class TestEWMuPartialFit:
    """Streaming/online updates via partial_fit with NaN."""

    def test_nan_matches_batch(self, X_synth):
        """partial_fit with NaN should match batch fit."""
        X_nan = X_synth.copy()
        X_nan[:50, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:50, 3] = False

        model_batch = EWMu(half_life=10, min_observations=1)
        model_batch.fit(X_nan, active_mask=um)

        model_stream = EWMu(half_life=10, min_observations=1)
        model_stream.partial_fit(X_nan[:80], active_mask=um[:80])
        model_stream.partial_fit(X_nan[80:], active_mask=um[80:])

        np.testing.assert_array_almost_equal(model_batch.mu_, model_stream.mu_)

    def test_asset_leaves_universe(self, X_synth):
        """Asset leaving universe across partial_fit calls."""
        model = EWMu(half_life=10, min_observations=1)

        model.partial_fit(X_synth[:100])
        assert not np.any(np.isnan(model.mu_))

        X_b2 = X_synth[100:].copy()
        X_b2[:, 2] = np.nan
        um_b2 = np.ones_like(X_b2, dtype=bool)
        um_b2[:, 2] = False
        model.partial_fit(X_b2, active_mask=um_b2)
        assert np.isnan(model.mu_[2])

    def test_batch_fast_path_matches_row_loop(self, X_synth):
        """Vectorized fast path produces same results as row-by-row."""
        model_batch = EWMu(half_life=20, min_observations=1)
        model_batch.fit(X_synth)

        um = np.ones_like(X_synth, dtype=bool)
        model_loop = EWMu(half_life=20, min_observations=1)
        model_loop.fit(X_synth, active_mask=um)

        np.testing.assert_array_almost_equal(model_batch.mu_, model_loop.mu_)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEWMuEdgeCases:
    def test_all_assets_delisted(self, X_synth):
        """All assets leave universe -> all NaN mu."""
        um = np.ones_like(X_synth, dtype=bool)
        um[100:, :] = False
        X_nan = X_synth.copy()
        X_nan[100:, :] = np.nan

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert np.all(np.isnan(model.mu_))

    def test_first_row_all_nan(self, X_synth):
        """First row all NaN should not break initialization."""
        X_nan = X_synth.copy()
        X_nan[0, :] = np.nan

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan)

        assert not np.any(np.isnan(model.mu_))

    def test_listing_then_delisting(self, X_synth):
        """Asset is listed for a window then delisted -> NaN at end."""
        X_nan = X_synth.copy()
        X_nan[:50, 3] = np.nan
        X_nan[150:, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[:50, 3] = False
        um[150:, 3] = False

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert np.isnan(model.mu_[3])

    def test_delisting_then_relisting(self, X_synth):
        """Asset delisted then re-listed restarts bias correction."""
        X_nan = X_synth.copy()
        X_nan[80:130, 3] = np.nan
        um = np.ones_like(X_nan, dtype=bool)
        um[80:130, 3] = False

        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_nan, active_mask=um)

        assert not np.isnan(model.mu_[3])

    def test_fit_resets_state(self, X_synth):
        """Calling fit() twice produces the same result (state is reset)."""
        model = EWMu(half_life=10, min_observations=1)
        model.fit(X_synth[:50])
        mu_first = model.mu_.copy()

        model.fit(X_synth[:50])
        np.testing.assert_array_equal(model.mu_, mu_first)

        model.fit(X_synth[50:150])
        assert not np.allclose(model.mu_, mu_first)

    def test_pandas_ewm_equivalence(self):
        """EWMu matches pandas adjusted EWMA mean for clean data."""
        import pandas as pd

        rng = np.random.default_rng(seed=123)
        X = rng.standard_normal((100, 3)) * 0.01
        half_life = 20

        model = EWMu(half_life=half_life, min_observations=1)
        model.fit(X)

        df = pd.DataFrame(X)
        expected = df.ewm(halflife=half_life, adjust=True).mean().iloc[-1].values

        np.testing.assert_array_almost_equal(model.mu_, expected)


class TestEquilibriumMu:
    def test_equilibrium_mu(self, X):
        model = EquilibriumMu()
        model.fit(X)
        np.testing.assert_almost_equal(
            model.mu_,
            np.array(
                [
                    1.39281815e-04,
                    2.24154782e-04,
                    1.69252898e-04,
                    1.58471289e-04,
                    1.54063130e-04,
                    1.55075717e-04,
                    1.23265081e-04,
                    8.27725049e-05,
                    1.52196578e-04,
                    8.34019782e-05,
                    1.00899188e-04,
                    8.80173294e-05,
                    1.37526282e-04,
                    8.88137066e-05,
                    9.35022944e-05,
                    8.03837339e-05,
                    2.14880096e-04,
                    1.22283550e-04,
                    7.42303223e-05,
                    1.36071903e-04,
                ]
            ),
        )

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = EquilibriumMu(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )

            with pytest.raises(ValueError):
                model.fit(X)

            model.fit(X, implied_vol=implied_vol)

        # noinspection PyUnresolvedReferences
        assert model.covariance_estimator_.r2_scores_.shape == (20,)
        assert model.mu_.shape == (20,)


class TestShrunkMu:
    def test_shrinkage_mu(self, X):
        model = ShrunkMu()
        model.fit(X)
        np.testing.assert_almost_equal(
            model.mu_,
            np.array(
                [
                    9.47940177e-04,
                    1.59095814e-03,
                    6.01237831e-04,
                    7.18130766e-04,
                    5.45759046e-04,
                    9.15282071e-05,
                    7.68939857e-04,
                    5.15035127e-04,
                    6.31494691e-04,
                    4.59445265e-04,
                    9.92082170e-04,
                    6.12075663e-04,
                    9.40928476e-04,
                    5.67185534e-04,
                    5.35844339e-04,
                    5.14453650e-04,
                    3.23942637e-04,
                    9.59637316e-04,
                    4.91763301e-04,
                    4.40458356e-04,
                ]
            ),
        )

        model = ShrunkMu(vol_weighted_target=True)
        model.fit(X)
        np.testing.assert_almost_equal(
            model.mu_,
            np.array(
                [
                    9.26914220e-04,
                    1.62077956e-03,
                    5.52795992e-04,
                    6.78932369e-04,
                    4.92930159e-04,
                    2.78050083e-06,
                    7.33759246e-04,
                    4.59776711e-04,
                    5.85445448e-04,
                    3.99791018e-04,
                    9.74546792e-04,
                    5.64490839e-04,
                    9.19348060e-04,
                    5.16050971e-04,
                    4.82231436e-04,
                    4.59149253e-04,
                    2.53573366e-04,
                    9.39536323e-04,
                    4.34664640e-04,
                    3.79302697e-04,
                ]
            ),
        )

        model = ShrunkMu(method=ShrunkMuMethods.BAYES_STEIN)
        model.fit(X)
        np.testing.assert_almost_equal(
            model.mu_,
            np.array(
                [
                    0.00083316,
                    0.00121765,
                    0.00062585,
                    0.00069574,
                    0.00059267,
                    0.00032106,
                    0.00072612,
                    0.0005743,
                    0.00064394,
                    0.00054106,
                    0.00085955,
                    0.00063233,
                    0.00082896,
                    0.00060548,
                    0.00058674,
                    0.00057395,
                    0.00046004,
                    0.00084015,
                    0.00056038,
                    0.00052971,
                ]
            ),
        )

        model = ShrunkMu(method=ShrunkMuMethods.BODNAR_OKHRIN)
        model.fit(X)
        np.testing.assert_almost_equal(
            model.mu_,
            np.array(
                [
                    5.95933693e-05,
                    -1.35794787e-04,
                    1.64942731e-04,
                    1.29423513e-04,
                    1.81800576e-04,
                    3.19823666e-04,
                    1.13984606e-04,
                    1.91136381e-04,
                    1.55748848e-04,
                    2.08027978e-04,
                    4.61803341e-05,
                    1.61649535e-04,
                    6.17239528e-05,
                    1.75289900e-04,
                    1.84813270e-04,
                    1.91313069e-04,
                    2.49201958e-04,
                    5.60390638e-05,
                    1.98207784e-04,
                    2.13797362e-04,
                ]
            ),
        )

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = ShrunkMu(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )

            with pytest.raises(ValueError):
                model.fit(X)

            model.fit(X, implied_vol=implied_vol)

        # noinspection PyUnresolvedReferences
        assert model.covariance_estimator_.r2_scores_.shape == (20,)
        assert model.mu_.shape == (20,)
