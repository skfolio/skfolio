from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn import config_context

from skfolio import RiskMeasure
from skfolio.moments import EWCovariance, EWMu, ImpliedCovariance
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior


def test_empirical_prior(X):
    model = EmpiricalPrior()
    model.fit(X)
    res = model.return_distribution_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    np.testing.assert_almost_equal(
        res.mu,
        np.array(
            [
                1.04344495e-03,
                1.90156515e-03,
                5.80763817e-04,
                7.36759751e-04,
                5.06726281e-04,
                -9.94537558e-05,
                8.04565487e-04,
                4.65724603e-04,
                6.21142195e-04,
                3.91538834e-04,
                1.10235332e-03,
                5.95227119e-04,
                1.03408770e-03,
                5.35320353e-04,
                4.93494909e-04,
                4.64948611e-04,
                2.10707897e-04,
                1.05905502e-03,
                4.34667892e-04,
                3.66200428e-04,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        res.covariance[:5, :5],
        np.array(
            [
                [0.00033685, 0.00028313, 0.00015618, 0.00017581, 0.00012206],
                [0.00028313, 0.00139232, 0.00022058, 0.0002515, 0.00017624],
                [0.00015618, 0.00022058, 0.0003929, 0.00019494, 0.00022071],
                [0.00017581, 0.0002515, 0.00019494, 0.00061523, 0.00014713],
                [0.00012206, 0.00017624, 0.00022071, 0.00014713, 0.00036072],
            ]
        ),
    )
    np.testing.assert_almost_equal(res.returns, np.asarray(X))


def test_empirical_prior_log_normal(X):
    model1 = EmpiricalPrior()
    model2 = EmpiricalPrior(is_log_normal=True, investment_horizon=1)

    model1.fit(X)
    model2.fit(X)

    np.testing.assert_almost_equal(
        model1.return_distribution_.mu, model2.return_distribution_.mu, 4
    )

    np.testing.assert_almost_equal(
        model1.return_distribution_.covariance,
        model2.return_distribution_.covariance,
        4,
    )


def test_empirical_prior_log_normal_investment_horizon(X):
    model = EmpiricalPrior(is_log_normal=True, investment_horizon=252)
    model.fit(X)
    res = model.return_distribution_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    np.testing.assert_almost_equal(
        res.mu,
        np.array(
            [
                0.30067519,
                0.61216954,
                0.15758953,
                0.20469753,
                0.13647218,
                -0.02469321,
                0.2249461,
                0.12453549,
                0.16940725,
                0.10375612,
                0.3198692,
                0.16179255,
                0.29758174,
                0.14443353,
                0.13237546,
                0.12428224,
                0.05359132,
                0.3058039,
                0.11575736,
                0.09670825,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        res.covariance[:5, :5],
        np.array(
            [
                [0.15002375, 0.15579818, 0.06082337, 0.07137514, 0.04671937],
                [0.15579818, 1.05213292, 0.1079466, 0.12822831, 0.08449074],
                [0.06082337, 0.1079466, 0.13914677, 0.07028907, 0.0754463],
                [0.07137514, 0.12822831, 0.07028907, 0.24792103, 0.05187185],
                [0.04671937, 0.08449074, 0.0754463, 0.05187185, 0.12443769],
            ]
        ),
    )
    np.testing.assert_almost_equal(res.returns, np.asarray(X))


def test_metadata_routing(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = EmpiricalPrior(
            covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
        )

        with pytest.raises(ValueError):
            model.fit(X)

        model.fit(X, implied_vol=implied_vol)

    # noinspection PyUnresolvedReferences
    assert model.covariance_estimator_.r2_scores_.shape == (20,)


@pytest.mark.parametrize("max_history", [1.5, True])
def test_max_history_validation(X, max_history):
    model = EmpiricalPrior(max_history=max_history)

    with pytest.raises(ValueError, match="`max_history` must be a positive integer"):
        model.fit(X)


def _make_ew_prior(**kwargs):
    return EmpiricalPrior(
        mu_estimator=EWMu(half_life=40),
        covariance_estimator=EWCovariance(half_life=40),
        **kwargs,
    )


class TestPartialFit:
    def test_equivalence_with_fit(self, X):
        """Streaming partial_fit matches a single fit on the same data."""
        X_arr = np.asarray(X)
        split = len(X_arr) // 2

        ref = _make_ew_prior()
        ref.fit(X_arr)

        stream = _make_ew_prior()
        stream.partial_fit(X_arr[:split])
        stream.partial_fit(X_arr[split:])

        np.testing.assert_array_almost_equal(
            stream.return_distribution_.mu, ref.return_distribution_.mu
        )
        np.testing.assert_array_almost_equal(
            stream.return_distribution_.covariance,
            ref.return_distribution_.covariance,
        )

    def test_returns_accumulation(self, X):
        """partial_fit accumulates raw returns across calls."""
        X_arr = np.asarray(X)
        split = len(X_arr) // 2

        model = _make_ew_prior()
        model.partial_fit(X_arr[:split])
        assert model.return_distribution_.returns.shape[0] == split

        model.partial_fit(X_arr[split:])
        np.testing.assert_array_almost_equal(model.return_distribution_.returns, X_arr)

    def test_max_history(self, X):
        """max_history caps the stored returns."""
        X_arr = np.asarray(X)
        cap = 50

        model = _make_ew_prior(max_history=cap)
        model.partial_fit(X_arr)

        assert model.return_distribution_.returns.shape[0] == cap
        np.testing.assert_array_almost_equal(
            model.return_distribution_.returns, X_arr[-cap:]
        )

    def test_fit_resets_after_partial_fit(self, X):
        """fit() resets all accumulated state from prior partial_fit calls."""
        X_arr = np.asarray(X)

        model = _make_ew_prior()
        model.partial_fit(X_arr)
        model.partial_fit(X_arr)
        assert model.return_distribution_.returns.shape[0] == 2 * len(X_arr)

        model.fit(X_arr)
        assert model.return_distribution_.returns.shape[0] == len(X_arr)

    def test_non_incremental_estimator_raises(self, X):
        """partial_fit raises when sub-estimators lack partial_fit."""
        model = EmpiricalPrior()
        with pytest.raises(TypeError, match="partial_fit"):
            model.partial_fit(np.asarray(X))


def _make_short_warmup_ew_prior(**kwargs):
    return EmpiricalPrior(
        mu_estimator=EWMu(half_life=10, min_observations=5),
        covariance_estimator=EWCovariance(half_life=10, min_observations=5),
        **kwargs,
    )


class TestNaNHandling:
    def test_holiday_nans_zero_filled(self, X):
        """Sparse holiday NaNs are zero-filled without warning and other values
        are unchanged."""
        X_arr = np.asarray(X).copy()
        X_arr[[10, 50, 100], 0] = np.nan

        model = _make_ew_prior()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.fit(X_arr)

        returns = model.return_distribution_.returns
        assert np.isfinite(returns).all()
        np.testing.assert_array_equal(returns[[10, 50, 100], 0], 0.0)
        valid = ~np.isnan(X_arr)
        np.testing.assert_array_equal(returns[valid], X_arr[valid])

    def test_moments_unaffected_by_fill(self, X):
        """The moment estimators receive the raw NaN data, not the filled
        scenarios."""
        X_arr = np.asarray(X).copy()
        X_arr[[10, 50, 100], 0] = np.nan

        model = _make_ew_prior()
        model.fit(X_arr)

        expected_mu = EWMu(half_life=40).fit(X_arr).mu_
        np.testing.assert_array_equal(model.return_distribution_.mu, expected_mu)

    def test_late_listing_backfilled_with_warning(self, X):
        """Pre-listing history is zero-filled once the asset is investable and a
        warning names the asset."""
        X_arr = np.asarray(X).copy()
        X_arr[:-100, 0] = np.nan

        model = _make_short_warmup_ew_prior()
        with pytest.warns(UserWarning, match="zero-filled"):
            model.fit(X_arr)

        returns = model.return_distribution_.returns
        assert np.isfinite(returns).all()
        np.testing.assert_array_equal(returns[:-100, 0], 0.0)
        np.testing.assert_array_equal(returns[-100:, 0], X_arr[-100:, 0])

    def test_warmup_asset_column_untouched(self, X):
        """An asset still in warm-up is non-investable and its scenario column
        keeps its raw NaN values."""
        X_arr = np.asarray(X).copy()
        X_arr[:-3, 0] = np.nan

        model = _make_short_warmup_ew_prior()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.fit(X_arr)

        res = model.return_distribution_
        assert np.isnan(res.mu[0])
        returns = res.returns
        assert np.isnan(returns[:-3, 0]).all()
        np.testing.assert_array_equal(returns[-3:, 0], X_arr[-3:, 0])

        subset = res.investable_subset()
        assert np.isfinite(subset.mu).all()
        assert np.isfinite(subset.covariance).all()
        assert np.isfinite(subset.returns).all()

    def test_warning_names_assets(self, X):
        """The warning reports asset names when X has feature names."""
        X_nan = X.copy()
        X_nan.iloc[:-100, 0] = np.nan

        model = _make_short_warmup_ew_prior()
        with pytest.warns(UserWarning, match="AAPL"):
            model.fit(X_nan)

    def test_warning_once_per_asset_across_partial_fit(self, X):
        """A warned asset does not warn again and a newly crossing asset does."""
        X_nan = X.copy()
        split = len(X_nan) // 2
        X_nan.iloc[: split - 100, 0] = np.nan  # AAPL crosses in the first chunk
        X_nan.iloc[split:, 1] = np.nan  # AMD crosses in the second chunk

        model = _make_short_warmup_ew_prior()
        with pytest.warns(UserWarning, match="AAPL"):
            model.partial_fit(X_nan.iloc[:split])

        with pytest.warns(UserWarning) as record:
            model.partial_fit(X_nan.iloc[split:])
        messages = [str(w.message) for w in record if "zero-filled" in str(w.message)]
        assert len(messages) == 1
        assert "AMD" in messages[0]
        assert "AAPL" not in messages[0]

    def test_warning_reset_by_fit(self, X):
        """fit() resets the warned assets so the warning fires again."""
        X_arr = np.asarray(X).copy()
        X_arr[:-100, 0] = np.nan

        model = _make_short_warmup_ew_prior()
        with pytest.warns(UserWarning, match="zero-filled"):
            model.fit(X_arr)
        with pytest.warns(UserWarning, match="zero-filled"):
            model.fit(X_arr)

    def test_no_copy_when_all_finite(self, X):
        """All-finite input keeps the buffer view without copying."""
        model = _make_ew_prior()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.fit(np.asarray(X))

        assert np.shares_memory(
            model.return_distribution_.returns, model._returns_buffer.array
        )

    def test_max_history_window(self, X):
        """The fill and the warning apply to the truncated window only."""
        X_arr = np.asarray(X).copy()
        cap = 100
        X_arr[:-cap, 0] = np.nan

        model = _make_short_warmup_ew_prior(max_history=cap)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.fit(X_arr)

        np.testing.assert_array_equal(model.return_distribution_.returns, X_arr[-cap:])

    def test_default_estimator_rejects_nan(self, X):
        """The default EmpiricalMu still rejects NaN input."""
        X_arr = np.asarray(X).copy()
        X_arr[0, 0] = np.nan

        model = EmpiricalPrior()
        with pytest.raises(ValueError, match="NaN"):
            model.fit(X_arr)

    def test_mean_risk_cvar_end_to_end(self, X):
        """CVaR optimization fits on holiday NaN data with NaN-aware estimators."""
        X_arr = np.asarray(X)[-300:].copy()
        X_arr[::50, 0] = np.nan

        model = MeanRisk(
            risk_measure=RiskMeasure.CVAR, prior_estimator=_make_ew_prior()
        )
        model.fit(X_arr)
        assert np.isfinite(model.weights_).all()
