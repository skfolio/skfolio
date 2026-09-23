"""Tests for ParametricPortfolioPolicy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio import Portfolio
from skfolio.containers import AssetPanel
from skfolio.datasets import make_synthetic_characteristics
from skfolio.descriptor import (
    BookToPrice,
    EWMomentum,
    LogMarketCap,
    Passthrough,
    RollingMomentum,
)
from skfolio.factor_exposure import FixedWeightedFactor, OneHotCategoricalFactors
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import ParametricPortfolioPolicy
from skfolio.optimization.parametric._parametric_portfolio_policy import (
    _crra,
    _maximize_crra_utility,
)


@pytest.fixture(scope="module")
def panel():
    return make_synthetic_characteristics(
        n_assets=60, n_observations=300, random_state=0
    )


@pytest.fixture(scope="module")
def X(panel):
    return panel.to_dataframe(fields="returns")


def _exposures():
    return [
        ("size", FixedWeightedFactor(descriptors=[("mcap", LogMarketCap())])),
        ("value", FixedWeightedFactor(descriptors=[("btp", BookToPrice())])),
        ("momentum", FixedWeightedFactor(descriptors=[("mom", EWMomentum())])),
    ]


def _signal_panel(n_assets=40, n_observations=400, beta=0.02, seed=0):
    """Panel where next-period returns load on a known characteristic.

    `signal` at t is standard normal across assets and r[t + 1] = beta * signal[t]
    plus noise, so a positive tilt on `signal` is optimal.
    """
    rng = np.random.default_rng(seed)
    signal = rng.standard_normal((n_observations, n_assets))
    noise = rng.standard_normal((n_observations, n_assets)) * 0.01
    returns = np.zeros((n_observations, n_assets))
    returns[1:] = beta * signal[:-1] + noise[1:]
    returns[0] = noise[0]
    market_cap = np.full((n_observations, n_assets), 1e9)
    panel = AssetPanel(
        fields={"returns": returns, "market_cap": market_cap, "signal": signal},
        asset_names=np.array([f"a{i}" for i in range(n_assets)]),
        observations=pd.date_range("2020-01-01", periods=n_observations, freq="B"),
    )
    return panel, panel.to_dataframe(fields="returns")


def _signal_exposure(field="signal"):
    return [
        (
            "signal",
            FixedWeightedFactor(
                descriptors=[("s", Passthrough(field=field))],
                outlier_transformer="passthrough",
                scoring_transformer="passthrough",
            ),
        )
    ]


class TestFit:
    def test_fit_sets_attributes(self, panel, X):
        model = ParametricPortfolioPolicy(characteristics_exposures=_exposures())
        model.fit(X, characteristics=panel)

        assert model.n_features_in_ == X.shape[1]
        np.testing.assert_array_equal(model.feature_names_in_, X.columns)
        assert model.coef_.shape == (3,)
        np.testing.assert_array_equal(
            model.characteristic_names_, ["size", "value", "momentum"]
        )
        assert model.weights_.shape == (X.shape[1],)
        assert model.weights_history_.shape == X.shape
        assert np.isfinite(model.coef_).all()
        assert model.n_iter_ >= 1
        assert model.utility_ >= model.benchmark_utility_

    def test_weights_sum_to_one(self, panel, X):
        model = ParametricPortfolioPolicy(characteristics_exposures=_exposures())
        model.fit(X, characteristics=panel)

        np.testing.assert_almost_equal(model.weights_.sum(), 1.0)
        row_sums = np.nansum(model.weights_history_, axis=1)
        valid = ~np.isnan(model.weights_history_).all(axis=1)
        np.testing.assert_allclose(row_sums[valid], 1.0)

    def test_last_weights_match_history(self, panel, X):
        model = ParametricPortfolioPolicy(characteristics_exposures=_exposures())
        model.fit(X, characteristics=panel)
        np.testing.assert_array_equal(model.weights_, model.weights_history_[-1])

    def test_warmup_rows_are_nan(self, panel, X):
        model = ParametricPortfolioPolicy(
            characteristics_exposures=[
                (
                    "momentum",
                    FixedWeightedFactor(
                        descriptors=[("mom", RollingMomentum(window=10, skip=0))]
                    ),
                )
            ]
        )
        model.fit(X, characteristics=panel)
        assert np.isnan(model.weights_history_[:9]).all()
        assert not np.isnan(model.weights_history_[-1]).any()

    def test_inactive_assets_have_zero_weight(self, panel, X):
        model = ParametricPortfolioPolicy(characteristics_exposures=_exposures())
        model.fit(X, characteristics=panel)
        valid = ~np.isnan(model.weights_history_).all(axis=1)
        inactive = ~panel.active_mask & valid[:, np.newaxis]
        assert inactive.any()
        assert np.all(model.weights_history_[inactive] == 0)

    def test_benchmark_when_no_tilt_pays(self):
        """With pure-noise characteristics the in-sample utility gain over the
        benchmark is marginal compared with a predictive signal."""
        rng = np.random.default_rng(1)
        n_obs, n_assets = 500, 50
        returns = rng.standard_normal((n_obs, n_assets)) * 0.01
        noise = rng.standard_normal((n_obs, n_assets))
        panel = AssetPanel(
            fields={
                "returns": returns,
                "market_cap": np.full((n_obs, n_assets), 1e9),
                "noise": noise,
            },
            asset_names=np.array([f"a{i}" for i in range(n_assets)]),
            observations=np.arange(n_obs),
        )
        X = panel.to_dataframe(fields="returns")
        signal = ParametricPortfolioPolicy(
            characteristics_exposures=_signal_exposure(field="noise")
        ).fit(X, characteristics=panel)
        noise_gain = signal.utility_ - signal.benchmark_utility_

        signal_panel, signal_X = _signal_panel(n_assets=n_assets, n_observations=n_obs)
        predictive = ParametricPortfolioPolicy(
            characteristics_exposures=_signal_exposure()
        ).fit(signal_X, characteristics=signal_panel)
        signal_gain = predictive.utility_ - predictive.benchmark_utility_

        assert 0 <= noise_gain < 0.05 * signal_gain

    def test_recovers_positive_tilt_on_predictive_signal(self):
        panel, X = _signal_panel()
        model = ParametricPortfolioPolicy(
            characteristics_exposures=_signal_exposure(), risk_aversion=5.0
        )
        model.fit(X, characteristics=panel)

        assert model.coef_[0] > 0
        assert model.utility_ > model.benchmark_utility_
        # Signal-weighted return path: a higher signal gets a higher weight
        last_signal = panel["signal"][-1]
        corr = np.corrcoef(last_signal, model.weights_)[0, 1]
        assert corr > 0.99

    def test_higher_risk_aversion_shrinks_tilt(self):
        panel, X = _signal_panel()
        coefs = []
        for gamma in (1.0, 5.0, 20.0):
            model = ParametricPortfolioPolicy(
                characteristics_exposures=_signal_exposure(), risk_aversion=gamma
            ).fit(X, characteristics=panel)
            coefs.append(model.coef_[0])
        assert coefs[0] > coefs[1] > coefs[2] > 0

    def test_equal_weighted_benchmark(self, panel, X):
        model = ParametricPortfolioPolicy(
            characteristics_exposures=_exposures(), benchmark_mcap_power=0.0
        )
        model.fit(X, characteristics=panel)
        np.testing.assert_almost_equal(model.weights_.sum(), 1.0)

    def test_equal_weighted_benchmark_without_market_cap(self):
        panel, X = _signal_panel()
        panel = AssetPanel(
            fields={"returns": panel["returns"], "signal": panel["signal"]},
            asset_names=panel.asset_names,
            observations=panel.observations,
        )
        model = ParametricPortfolioPolicy(
            characteristics_exposures=_signal_exposure(), benchmark_mcap_power=0.0
        )
        model.fit(X, characteristics=panel)
        assert model.coef_[0] > 0

    def test_multi_factor_characteristics_expand(self, panel, X):
        model = ParametricPortfolioPolicy(
            characteristics_exposures=[
                ("size", FixedWeightedFactor(descriptors=[("mcap", LogMarketCap())])),
                (
                    "industry",
                    OneHotCategoricalFactors(category="industry", family="industry"),
                ),
            ]
        )
        model.fit(X, characteristics=panel)
        levels = list(panel.fields["industry"].levels)
        np.testing.assert_array_equal(model.characteristic_names_, ["size", *levels])
        assert model.coef_.shape == (1 + len(levels),)

    def test_investment_universe_subset_and_order(self, panel, X):
        columns = list(X.columns[::-1][:30])
        model = ParametricPortfolioPolicy(characteristics_exposures=_exposures())
        model.fit(X[columns], characteristics=panel)

        assert model.n_features_in_ == 30
        np.testing.assert_array_equal(model.feature_names_in_, columns)
        assert model.weights_.shape == (30,)
        np.testing.assert_almost_equal(model.weights_.sum(), 1.0)

    def test_missing_next_return_is_dropped_from_objective(self):
        panel, X = _signal_panel()
        X_missing = X.copy()
        X_missing.iloc[10:15, 3] = np.nan
        model = ParametricPortfolioPolicy(characteristics_exposures=_signal_exposure())
        model.fit(X_missing, characteristics=panel)
        assert np.isfinite(model.coef_).all()
        assert model.coef_[0] > 0

    def test_predict_returns_portfolio(self, panel, X):
        model = ParametricPortfolioPolicy(characteristics_exposures=_exposures())
        model.fit(X, characteristics=panel)
        portfolio = model.predict(X)
        assert isinstance(portfolio, Portfolio)
        np.testing.assert_array_equal(portfolio.weights, model.weights_)

    def test_walk_forward_cross_val_predict(self):
        panel, X = _signal_panel()
        model = ParametricPortfolioPolicy(characteristics_exposures=_signal_exposure())
        pred = cross_val_predict(
            model,
            X,
            cv=WalkForward(train_size=100, test_size=50),
            params={"characteristics": panel},
        )
        assert len(pred.returns) == 300
        assert np.isfinite(pred.returns).all()
        # The signal is predictive, so the policy beats its benchmark out of sample
        benchmark = ParametricPortfolioPolicy(
            characteristics_exposures=_signal_exposure(), max_iter=0
        )
        pred_benchmark = cross_val_predict(
            benchmark,
            X,
            cv=WalkForward(train_size=100, test_size=50),
            params={"characteristics": panel},
        )
        assert pred.mean > pred_benchmark.mean

    def test_does_not_mutate_input_panel(self, panel, X):
        fields_before = set(panel.fields)
        ParametricPortfolioPolicy(characteristics_exposures=_exposures()).fit(
            X, characteristics=panel
        )
        assert set(panel.fields) == fields_before

    def test_clone_does_not_mutate_estimators(self, panel, X):
        exposures = _exposures()
        ParametricPortfolioPolicy(characteristics_exposures=exposures).fit(
            X, characteristics=panel
        )
        assert not hasattr(exposures[0][1], "descriptors_")

    def test_fallback_on_failure(self, panel, X):
        model = ParametricPortfolioPolicy(
            characteristics_exposures=[
                ("missing", FixedWeightedFactor(descriptors=[("x", BookToPrice())]))
            ],
            fallback="previous_weights",
            previous_weights=np.full(X.shape[1], 1 / X.shape[1]),
        )
        bad_panel = AssetPanel(
            fields={"returns": panel["returns"], "market_cap": panel["market_cap"]},
            asset_names=panel.asset_names,
            observations=panel.observations,
            active_mask=panel.active_mask,
        )
        model.fit(X, characteristics=bad_panel)
        np.testing.assert_allclose(model.weights_, 1 / X.shape[1])
        assert model.fallback_chain_ is not None


class TestValidation:
    def test_missing_characteristics_raises(self, X):
        with pytest.raises(ValueError, match="`characteristics` must be provided"):
            ParametricPortfolioPolicy(characteristics_exposures=_exposures()).fit(X)

    def test_non_dataframe_raises(self, panel, X):
        with pytest.raises(ValueError, match=r"pd\.DataFrame"):
            ParametricPortfolioPolicy(characteristics_exposures=_exposures()).fit(
                X.to_numpy(), characteristics=panel
            )

    def test_unknown_asset_raises(self, panel, X):
        X = X.rename(columns={X.columns[0]: "unknown"})
        with pytest.raises(ValueError, match="missing from `characteristics`"):
            ParametricPortfolioPolicy(characteristics_exposures=_exposures()).fit(
                X, characteristics=panel
            )

    def test_observation_mismatch_raises(self, panel, X):
        with pytest.raises(ValueError, match="same number of observations"):
            ParametricPortfolioPolicy(characteristics_exposures=_exposures()).fit(
                X.iloc[:-1], characteristics=panel
            )
        with pytest.raises(ValueError, match=r"`X\.index` must match"):
            ParametricPortfolioPolicy(characteristics_exposures=_exposures()).fit(
                X.reset_index(drop=True), characteristics=panel
            )

    def test_market_cap_required(self):
        panel, X = _signal_panel()
        panel = AssetPanel(
            fields={"returns": panel["returns"], "signal": panel["signal"]},
            asset_names=panel.asset_names,
            observations=panel.observations,
        )
        with pytest.raises(ValueError, match="market_cap"):
            ParametricPortfolioPolicy(characteristics_exposures=_signal_exposure()).fit(
                X, characteristics=panel
            )

    @pytest.mark.parametrize(
        ("params", "match"),
        [
            ({"characteristics_exposures": []}, "non-empty list"),
            ({"characteristics_exposures": None}, "non-empty list"),
            ({"characteristics_exposures": ["size"]}, "tuples"),
            (
                {
                    "characteristics_exposures": [
                        (1, FixedWeightedFactor(descriptors=[]))
                    ]
                },
                "must be a string",
            ),
            (
                {"characteristics_exposures": [("size", LogMarketCap())]},
                "BaseFactorExposure",
            ),
            (
                {
                    "characteristics_exposures": [
                        ("a", FixedWeightedFactor(descriptors=[])),
                        ("a", FixedWeightedFactor(descriptors=[])),
                    ]
                },
                "unique",
            ),
            ({"risk_aversion": 0.0}, "risk_aversion"),
            ({"risk_aversion": -1.0}, "risk_aversion"),
            ({"benchmark_mcap_power": -1.0}, "benchmark_mcap_power"),
            ({"max_iter": -1}, "max_iter"),
            ({"tol": 0.0}, "tol"),
        ],
    )
    def test_invalid_params_raise(self, panel, X, params, match):
        kwargs = {"characteristics_exposures": _exposures(), **params}
        with pytest.raises(ValueError, match=match):
            ParametricPortfolioPolicy(**kwargs).fit(X, characteristics=panel)

    def test_no_eligible_observation_raises(self):
        panel, X = _signal_panel(n_observations=5)
        with pytest.raises(ValueError, match="No observation has an eligible asset"):
            ParametricPortfolioPolicy(
                characteristics_exposures=[
                    (
                        "mom",
                        FixedWeightedFactor(
                            descriptors=[("m", RollingMomentum(window=10, skip=0))]
                        ),
                    )
                ]
            ).fit(X, characteristics=panel)

    def test_categorical_characteristics(self, panel, X):
        model = ParametricPortfolioPolicy(
            characteristics_exposures=[
                (
                    "industry",
                    OneHotCategoricalFactors(category="industry", family="industry"),
                ),
                ("size", FixedWeightedFactor(descriptors=[("mcap", LogMarketCap())])),
            ]
        )
        model.fit(X, characteristics=panel)
        assert (
            len(model.characteristic_names_) == len(panel.fields["industry"].levels) + 1
        )
        np.testing.assert_almost_equal(model.weights_.sum(), 1.0)


class TestSolver:
    def test_crra_derivatives(self):
        r = np.array([-0.5, 0.0, 0.1, 2.0])
        for gamma in (1.0, 2.0, 5.0):
            _, du, d2u = _crra(r, gamma)
            eps = 1e-6
            num_du = (_crra(r + eps, gamma)[0] - _crra(r - eps, gamma)[0]) / (2 * eps)
            num_d2u = (_crra(r + eps, gamma)[1] - _crra(r - eps, gamma)[1]) / (2 * eps)
            np.testing.assert_allclose(du, num_du, rtol=1e-5)
            np.testing.assert_allclose(d2u, num_d2u, rtol=1e-4)
        np.testing.assert_allclose(_crra(r, 1.0)[0], np.log1p(r))

    def test_solution_is_stationary_and_beats_benchmark(self):
        rng = np.random.default_rng(3)
        benchmark = rng.standard_normal(1000) * 0.01
        tilts = rng.standard_normal((1000, 3)) * 0.01 + np.array([0.001, -0.002, 0.0])
        theta, n_iter, utility, benchmark_utility = _maximize_crra_utility(
            benchmark, tilts, risk_aversion=5.0, max_iter=100, tol=1e-12
        )
        assert utility > benchmark_utility
        assert n_iter >= 1
        _, du, _ = _crra(benchmark + tilts @ theta, 5.0)
        gradient = tilts.T @ du / len(du)
        np.testing.assert_allclose(gradient, 0.0, atol=1e-6)
        assert theta[0] > 0 > theta[1]

    def test_solution_stays_in_utility_domain(self):
        rng = np.random.default_rng(4)
        benchmark = rng.standard_normal(200) * 0.01
        # Large tilt returns so that an undamped step would leave 1 + r > 0
        tilts = rng.standard_normal((200, 2)) * 0.5 + 0.2
        theta, _, utility, _ = _maximize_crra_utility(
            benchmark, tilts, risk_aversion=2.0, max_iter=200, tol=1e-12
        )
        assert np.all(1.0 + benchmark + tilts @ theta > 0)
        assert np.isfinite(utility)

    def test_zero_iterations_returns_benchmark(self):
        benchmark = np.full(10, 0.01)
        tilts = np.ones((10, 1)) * 0.01
        theta, n_iter, utility, benchmark_utility = _maximize_crra_utility(
            benchmark, tilts, risk_aversion=5.0, max_iter=0, tol=1e-12
        )
        np.testing.assert_array_equal(theta, 0.0)
        assert n_iter == 0
        assert utility == benchmark_utility
