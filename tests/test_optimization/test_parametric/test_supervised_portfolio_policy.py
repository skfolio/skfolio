"""Timing, coefficient recovery and inference tests for supervised policies."""

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from sklearn import clone

from skfolio.containers import AssetPanel
from skfolio.descriptor import Passthrough
from skfolio.exceptions import GroupNotFoundError
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.optimization import (
    BaseOptimization,
    EqualWeighted,
    SupervisedPortfolioPolicy,
)


class RecordingOptimizer(BaseOptimization):
    """Use encoded target weights to audit exactly which returns enter each label."""

    windows: ClassVar[list[pd.DataFrame]] = []

    def fit(self, X, y=None):
        type(self).windows.append(X.copy())
        self.weights_ = X.mean().to_numpy()
        return self


def data(n=11, offset=0):
    signal = np.tile([-1.0, 0.0, 1.0], (n, 1))
    signal[:, 0] += np.arange(n) * 0.03
    signal -= signal.mean(axis=1, keepdims=True)
    returns = np.full((n, 3), 1 / 3)
    for t in range(0, n - 2, 2):
        returns[t + 1 : t + 3] += 0.4 * signal[t] / 3
    dates = pd.date_range("2020-01-01", periods=n + offset)[offset:]
    panel = AssetPanel(
        fields={"signal": signal, "returns": returns},
        asset_names=np.array(["a", "b", "c"]),
        observations=dates,
    )
    return panel.to_dataframe(fields="returns"), panel


def model(**kwargs):
    return SupervisedPortfolioPolicy(
        characteristics_exposures=[
            (
                "signal",
                FixedWeightedFactor(
                    descriptors=[("signal", Passthrough(field="signal"))],
                    outlier_transformer="passthrough",
                    scoring_transformer="passthrough",
                ),
            )
        ],
        optimizer=kwargs.pop("optimizer", RecordingOptimizer()),
        benchmark_mcap_power=0,
        window_size=kwargs.pop("window_size", 2),
        alpha=0,
        **kwargs,
    )


def test_exact_linear_recovery_and_window_boundaries():
    X, panel = data(n=12)
    RecordingOptimizer.windows = []
    estimator = model().fit(X, characteristics=panel)
    np.testing.assert_allclose(estimator.coef_, [0.4], atol=1e-12)
    assert len(RecordingOptimizer.windows) == 5
    for t, window in zip(range(0, 10, 2), RecordingOptimizer.windows, strict=True):
        pd.testing.assert_frame_equal(window, X.iloc[t + 1 : t + 3])
    assert estimator.training_windows_.iloc[-1]["end"] == X.index[10]
    assert not hasattr(estimator.optimizer, "weights_")
    np.testing.assert_allclose(estimator.weights_history_.sum(axis=1), 1)


def test_unused_return_rows_do_not_affect_coefficients():
    X, panel = data(n=12)
    first = model().fit(X, characteristics=panel)
    changed = X.copy()
    changed.iloc[[0, -1]] = 1000
    second = model().fit(changed, characteristics=panel)
    np.testing.assert_array_equal(first.coef_, second.coef_)


def test_new_characteristics_and_standard_predict():
    X, panel = data()
    estimator = model(long_only=True).fit(X, characteristics=panel)
    old_coef = estimator.coef_.copy()
    future_X, future_panel = data(n=4, offset=11)
    weights = estimator.predict_weights(future_panel)
    assert weights.index.equals(future_X.index)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert (weights.to_numpy() >= 0).all()
    np.testing.assert_array_equal(estimator.coef_, old_coef)
    assert len(estimator.predict(future_X).returns) == len(future_X)


def test_clone_and_refit_reset():
    X, panel = data()
    estimator = model().fit(X, characteristics=panel)
    fresh = clone(estimator)
    assert not hasattr(fresh, "coef_")
    estimator.set_params(optimizer=EqualWeighted()).fit(X, characteristics=panel)
    np.testing.assert_allclose(estimator.coef_, 0, atol=1e-12)
    np.testing.assert_allclose(estimator.weights_, 1 / 3)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reversed", "short"])
def test_invalid_training_data(change):
    X, panel = data(n=2 if change == "short" else 11)
    if change == "missing":
        X.iloc[2, 0] = np.nan
    elif change == "duplicate":
        X.index = [X.index[0]] * len(X)
    elif change == "reversed":
        X = X.iloc[::-1]
    with pytest.raises(ValueError):
        model().fit(X, characteristics=panel)


def test_reject_non_invested_optimizer():
    X, panel = data()
    X *= 2
    with pytest.raises(ValueError, match="fully invested"):
        model().fit(X, characteristics=panel)


def test_failed_refit_clears_previous_policy():
    from sklearn.exceptions import NotFittedError

    X, panel = data()
    estimator = model().fit(X, characteristics=panel)
    X.iloc[1, 0] = np.nan
    with pytest.raises(ValueError):
        estimator.fit(X, characteristics=panel)
    with pytest.raises(NotFittedError):
        estimator.predict_weights(panel)


def test_real_optimizer_labels_and_asset_alignment():
    from skfolio.optimization import InverseVolatility

    X, panel = data(n=13)
    X = X + np.random.default_rng(17).normal(0, 0.01, X.shape)
    estimator = model(optimizer=InverseVolatility(), window_size=4).fit(
        X, characteristics=panel
    )
    expected = InverseVolatility().fit(X.iloc[1:5]).weights_
    np.testing.assert_allclose(estimator.target_weights_.iloc[0], expected)
    _, future = data(n=4, offset=13)
    reference = estimator.predict_weights(future)
    assert list(reference.columns) == list(X.columns)
    reordered = AssetPanel(
        fields={"signal": future["signal"][:, ::-1]},
        asset_names=future.asset_names[::-1],
        observations=future.observations,
    )
    with pytest.raises(ValueError, match=r"asset_names don.t match"):
        estimator.predict_weights(reordered)


def test_nonlinear_pipeline_predictions_and_clone():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, panel = data()
    supplied = make_pipeline(
        StandardScaler(), RandomForestRegressor(n_estimators=5, random_state=1)
    )
    estimator = model(regressor=supplied).fit(X, characteristics=panel)
    assert not hasattr(supplied[-1], "estimators_")
    assert not hasattr(estimator, "coef_")
    _, future = data(n=4, offset=11)
    weights = estimator.predict_weights(future)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert np.isfinite(weights).all().all()
    assert not hasattr(clone(estimator), "regressor_")


def test_regressor_intercept_is_removed_from_portfolio_budget():
    from sklearn.dummy import DummyRegressor

    X, panel = data()
    estimator = model(regressor=DummyRegressor(strategy="constant", constant=2)).fit(
        X, characteristics=panel
    )
    np.testing.assert_allclose(estimator.weights_, 1 / 3)


def test_projection_enforces_bounds_and_groups_for_future_predictions():
    X, panel = data()
    estimator = model(
        long_only=True,
        min_weights={"a": 0.2},
        max_weights=0.5,
        groups=[["equity", "equity", "bond"]],
        linear_constraints=["equity <= 0.6"],
    ).fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    weights = estimator.predict_weights(future).to_numpy()
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-7)
    assert (weights >= -1e-7).all()
    assert (weights <= 0.5 + 1e-7).all()
    assert (weights[:, 0] >= 0.2 - 1e-7).all()
    assert (weights[:, :2].sum(axis=1) <= 0.6 + 1e-7).all()


@pytest.mark.parametrize(
    "constraints, message",
    [
        ({"max_weights": 0.2}, "infeasible"),
        ({"linear_constraints": ["unknown <= 0.2"]}, "unknown"),
    ],
)
def test_invalid_constraints_fail_before_label_fit(constraints, message):
    X, panel = data()
    RecordingOptimizer.windows = []
    with pytest.raises((ValueError, GroupNotFoundError), match=message):
        model(**constraints).fit(X, characteristics=panel)
    assert not RecordingOptimizer.windows


def test_projection_is_euclidean_not_clip_and_normalize():
    from skfolio.optimization.parametric._weight_projection import _WeightProjection

    projection = _WeightProjection(["a", "b", "c"], True, None, None, None, None)
    actual = projection.project(np.array([[0.8, 0.4, -0.2]]))
    np.testing.assert_allclose(actual, [[0.7, 0.3, 0]], atol=1e-6)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_reject_nonfinite_regressor_predictions(value):
    from sklearn.base import BaseEstimator, RegressorMixin

    class InvalidRegressor(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), value)

    X, panel = data()
    with pytest.raises(ValueError, match="one finite value"):
        model(regressor=InvalidRegressor()).fit(X, characteristics=panel)


def test_nonlinear_prediction_uses_fitted_regressor():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    X, panel = data()
    estimator = model(
        regressor=make_pipeline(PolynomialFeatures(2), Ridge(alpha=0.1))
    ).fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    signal = future["signal"]
    features = (signal - signal.mean(axis=1, keepdims=True)) / 3
    raw = estimator.regressor_.predict(features.reshape(-1, 1)).reshape(4, 3)
    expected = 1 / 3 + raw - raw.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(estimator.predict_weights(future), expected, atol=1e-7)


def test_risk_limit_at_prediction():
    X, panel = data()
    estimator = model(max_variance=0.34).fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    weights = estimator.predict_weights(future, covariance=np.eye(3)).to_numpy()
    assert (np.sum(weights**2, axis=1) <= 0.34 + 1e-6).all()
    with pytest.raises(ValueError, match="covariance"):
        estimator.predict_weights(future)
    with pytest.raises(ValueError, match="explicit risk"):
        estimator.predict(X)
    with pytest.raises(ValueError, match="infeasible"):
        estimator.predict_weights(future, covariance=10 * np.eye(3))


def test_turnover_uses_each_supplied_holdings_vector():
    X, panel = data()
    estimator = model(max_turnover=0).fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    holdings = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]] * 2)
    actual = estimator.predict_weights(future, holdings=holdings)
    np.testing.assert_allclose(actual, holdings, atol=1e-6)
    with pytest.raises(ValueError, match="holdings"):
        estimator.predict_weights(future)


def test_conflicting_risk_and_turnover_raise():
    X, panel = data()
    estimator = model(max_turnover=0, max_variance=0.34).fit(X, characteristics=panel)
    _, future = data(n=1, offset=11)
    with pytest.raises(ValueError, match="infeasible"):
        estimator.predict_weights(future, covariance=np.eye(3), holdings=[1, 0, 0])


def test_sequential_evaluation_uses_drift_and_is_repeatable():
    X, panel = data()
    estimator = model(max_turnover=0).fit(X, characteristics=panel)
    _, future = data(n=3, offset=11)
    returns = pd.DataFrame(
        [[0.1, 0, -0.1], [0, 0.1, 0], [0, 0, 0]],
        index=pd.date_range("2020-01-13", periods=3),
        columns=X.columns,
    )
    first = estimator.evaluate_sequential(returns, future, initial_holdings=[1 / 3] * 3)
    second = estimator.evaluate_sequential(
        returns, future, initial_holdings=[1 / 3] * 3
    )
    np.testing.assert_allclose(first.returns, second.returns)
    periods = first.portfolios
    np.testing.assert_allclose(periods[1].weights, [1.1 / 3, 1 / 3, 0.9 / 3], atol=1e-6)
    for previous, current in pairwise(periods):
        np.testing.assert_allclose(current.previous_weights, previous.ending_weights)
    assert all(p.turnover < 1e-6 for p in periods)
    assert estimator.last_observation_ == X.index[-1]


def test_sequential_evaluation_rejects_same_period_characteristics():
    X, panel = data()
    estimator = model().fit(X, characteristics=panel)
    returns, future = data(n=3, offset=11)
    with pytest.raises(ValueError, match="Decision dates"):
        estimator.evaluate_sequential(returns, future, initial_holdings=[1 / 3] * 3)


def test_sequential_limits_and_no_future_return_leakage():
    X, panel = data()
    estimator = model(max_turnover=0.1, max_variance=0.34).fit(X, characteristics=panel)
    _, future = data(n=3, offset=11)
    returns = pd.DataFrame(
        np.zeros((3, 3)),
        index=pd.date_range("2020-01-13", periods=3),
        columns=X.columns,
    )
    covariance = np.array([np.eye(3), 0.9 * np.eye(3), np.eye(3)])
    first = estimator.evaluate_sequential(
        returns, future, initial_holdings=[1 / 3] * 3, covariance=covariance
    )
    for i, period in enumerate(first.portfolios):
        assert period.turnover <= 0.1 + 1e-6
        assert period.weights @ covariance[i] @ period.weights <= 0.34 + 1e-6
    changed = returns.copy()
    changed.iloc[-1, 0] = 0.5
    second = estimator.evaluate_sequential(
        changed, future, initial_holdings=[1 / 3] * 3, covariance=covariance
    )
    for left, right in zip(first.portfolios, second.portfolios, strict=True):
        np.testing.assert_allclose(left.weights, right.weights)


def test_failed_sequential_evaluation_leaves_fitted_state_unchanged():
    X, panel = data()
    estimator = model(max_variance=0.34).fit(X, characteristics=panel)
    original_coef = estimator.coef_.copy()
    _, future = data(n=2, offset=11)
    returns = pd.DataFrame(
        np.zeros((2, 3)),
        index=pd.date_range("2020-01-13", periods=2),
        columns=X.columns,
    )
    with pytest.raises(ValueError, match="infeasible"):
        estimator.evaluate_sequential(
            returns,
            future,
            initial_holdings=[1 / 3] * 3,
            covariance=np.array([np.eye(3), 10 * np.eye(3)]),
        )
    np.testing.assert_array_equal(estimator.coef_, original_coef)
    assert estimator.last_observation_ == X.index[-1]
    result = estimator.evaluate_sequential(
        returns, future, initial_holdings=[1 / 3] * 3, covariance=np.eye(3)
    )
    assert len(result.portfolios) == 2


@pytest.mark.parametrize("rates", [0.01, [0.01, 0.02, 0.03], {"c": 0.03, "a": 0.01}])
def test_sequential_costs_use_drifted_holdings(rates):
    X, panel = data()
    estimator = model(max_turnover=0.5).fit(X, characteristics=panel)
    _, future = data(n=3, offset=11)
    returns = pd.DataFrame(
        [[0.1, 0, -0.1], [0, 0.1, 0], [0, 0, 0]],
        index=pd.date_range("2020-01-13", periods=3),
        columns=X.columns,
    )
    initial = np.array([0.5, 0.3, 0.2])
    result = estimator.evaluate_sequential(
        returns, future, initial_holdings=initial, transaction_costs=rates
    )
    vector = (
        np.array([rates.get(name, 0) for name in X.columns])
        if isinstance(rates, dict)
        else np.broadcast_to(rates, (3,))
    )
    anchor = initial
    for i, period in enumerate(result.portfolios):
        trades = np.abs(period.weights - anchor)
        expected_cost = vector @ trades
        np.testing.assert_allclose(period.total_cost, expected_cost)
        np.testing.assert_allclose(period.turnover, trades.sum())
        np.testing.assert_allclose(
            period.returns[0],
            period.weights @ returns.iloc[i].to_numpy() - expected_cost,
        )
        assert period.turnover <= 0.5 + 1e-6
        anchor = period.ending_weights
    assert result.portfolios[0].total_cost > 0
    gross = estimator.evaluate_sequential(returns, future, initial_holdings=initial)
    np.testing.assert_allclose(
        result.returns,
        gross.returns - np.array([p.total_cost for p in result.portfolios]),
    )


def test_cost_defaults_override_and_no_trade():
    X, panel = data()
    estimator = model(max_turnover=0, portfolio_params={"transaction_costs": 0.02}).fit(
        X, characteristics=panel
    )
    _, future = data(n=1, offset=11)
    returns = pd.DataFrame(
        [[0, 0, 0]], index=pd.date_range("2020-01-13", periods=1), columns=X.columns
    )
    result = estimator.evaluate_sequential(
        returns, future, initial_holdings=[1 / 3] * 3
    )
    assert result.portfolios[0].total_cost < 1e-8
    np.testing.assert_allclose(result.portfolios[0].transaction_costs, 0.02)
    override = estimator.evaluate_sequential(
        returns, future, initial_holdings=[1 / 3] * 3, transaction_costs=0
    )
    assert override.portfolios[0].total_cost == 0


@pytest.mark.parametrize("rates", [-0.1, np.nan, {"unknown": 0.1}, [0.1, 0.2]])
def test_invalid_transaction_costs(rates):
    X, panel = data()
    estimator = model().fit(X, characteristics=panel)
    _, future = data(n=1, offset=11)
    returns = pd.DataFrame(
        [[0, 0, 0]], index=pd.date_range("2020-01-13", periods=1), columns=X.columns
    )
    with pytest.raises(ValueError):
        estimator.evaluate_sequential(
            returns, future, initial_holdings=[1 / 3] * 3, transaction_costs=rates
        )


def test_prediction_state_is_explicit_and_committed_only_on_success():
    X, panel = data()
    estimator = model(max_variance=0.34).fit(X, characteristics=panel)
    _, future = data(n=2, offset=11)
    old_exposures = estimator.characteristics_exposures_
    first = estimator.predict_weights(future, covariance=np.eye(3))
    second = estimator.predict_weights(future, covariance=np.eye(3))
    pd.testing.assert_frame_equal(first, second)
    assert estimator.characteristics_exposures_ is old_exposures
    with pytest.raises(ValueError, match="infeasible"):
        estimator.predict_weights(future, covariance=10 * np.eye(3), update_state=True)
    assert estimator.characteristics_exposures_ is old_exposures
    assert estimator.last_prediction_observation_ == X.index[-1]
    estimator.predict_weights(future, covariance=np.eye(3), update_state=True)
    assert estimator.last_prediction_observation_ == future.observations[-1]
    with pytest.raises(ValueError, match="Prediction dates"):
        estimator.predict_weights(future, covariance=np.eye(3), update_state=True)


def test_rolling_prior_only_sees_returns_available_at_decision():
    from skfolio.prior import EmpiricalPrior

    class RecordingPrior(EmpiricalPrior):
        windows: ClassVar[list[pd.DataFrame]] = []

        def fit(self, X, y=None, **fit_params):
            type(self).windows.append(X.copy())
            return super().fit(X, y, **fit_params)

    X, panel = data()
    estimator = model(max_variance=10).fit(X, characteristics=panel)
    _, future = data(n=3, offset=11)
    returns = pd.DataFrame(
        [[0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]],
        index=pd.date_range("2020-01-13", periods=3),
        columns=X.columns,
    )
    from skfolio.moments import LedoitWolf

    prior = RecordingPrior(covariance_estimator=LedoitWolf())
    estimator.evaluate_sequential(
        returns,
        future,
        initial_holdings=[1 / 3] * 3,
        prior_estimator=prior,
        covariance_window=5,
    )
    assert not hasattr(prior, "return_distribution_")
    for t, window in enumerate(RecordingPrior.windows):
        pd.testing.assert_frame_equal(window, pd.concat([X, returns.iloc[:t]]).tail(5))
        assert window.index[-1] <= future.observations[t]
    assert len(RecordingPrior.windows) == 3


def test_default_rolling_prior_works_without_external_covariance():
    X, panel = data()
    estimator = model(max_variance=10).fit(X, characteristics=panel)
    _, future = data(n=1, offset=11)
    returns = pd.DataFrame(
        [[0.01, 0, 0]], index=pd.date_range("2020-01-13", periods=1), columns=X.columns
    )
    result = estimator.evaluate_sequential(
        returns, future, initial_holdings=[1 / 3] * 3
    )
    assert np.isfinite(result.returns).all()


def test_multi_observation_holding_intervals_charge_only_rebalances():
    X, panel = data()
    estimator = model(max_turnover=0.5).fit(X, characteristics=panel)
    _, future = data(n=5, offset=11)
    returns = pd.DataFrame(
        [[0.1, 0, -0.1]] * 5,
        index=pd.date_range("2020-01-13", periods=5),
        columns=X.columns,
    )
    result = estimator.evaluate_sequential(
        returns,
        future,
        initial_holdings=[0.5, 0.3, 0.2],
        transaction_costs=0.01,
        rebalance_every=2,
    )
    assert len(result.portfolios) == 5
    for i, period in enumerate(result.portfolios):
        if i % 2:
            assert period.turnover == 0
            assert period.total_cost == 0
            np.testing.assert_array_equal(
                period.weights, result.portfolios[i - 1].ending_weights
            )
        else:
            assert period.turnover <= 0.5 + 1e-6
            np.testing.assert_allclose(period.total_cost, 0.01 * period.turnover)
    assert result.portfolios[2].total_cost > 0
    with pytest.raises(ValueError, match="rebalance_every"):
        estimator.evaluate_sequential(
            returns, future, initial_holdings=[1 / 3] * 3, rebalance_every=0
        )


def test_rolling_characteristics_advance_on_nontrading_dates():
    from skfolio.descriptor import RollingMomentum

    X, panel = data()
    estimator = (
        model()
        .set_params(
            characteristics_exposures=[
                (
                    "momentum",
                    FixedWeightedFactor(
                        descriptors=[("momentum", RollingMomentum(window=3, skip=0))],
                        outlier_transformer="passthrough",
                        scoring_transformer="passthrough",
                    ),
                )
            ]
        )
        .fit(X, characteristics=panel)
    )
    _, future = data(n=5, offset=11)
    batch = estimator.predict_weights(future)
    pd.testing.assert_frame_equal(batch, estimator.predict_weights(future))
    returns = pd.DataFrame(
        np.zeros((5, 3)),
        index=pd.date_range("2020-01-13", periods=5),
        columns=X.columns,
    )
    result = estimator.evaluate_sequential(
        returns, future, initial_holdings=[1 / 3] * 3, rebalance_every=2
    )
    for t in (0, 2, 4):
        np.testing.assert_allclose(
            result.portfolios[t].weights, batch.iloc[t], atol=1e-7
        )
    first = estimator.predict_weights(
        future.isel(observations=slice(0, 2)), update_state=True
    )
    second = estimator.predict_weights(
        future.isel(observations=slice(2, 5)), update_state=True
    )
    pd.testing.assert_frame_equal(pd.concat([first, second]), batch)


def test_prior_metadata_is_sliced_to_causal_history():
    from skfolio.moments import LedoitWolf
    from skfolio.prior import EmpiricalPrior

    class MetadataPrior(EmpiricalPrior):
        seen: ClassVar[list] = []

        def fit(self, X, y=None, factors=None, **kwargs):
            pd.testing.assert_index_equal(factors.index, X.index)
            type(self).seen.append(factors.copy())
            return super().fit(X, y, **kwargs)

    X, panel = data()
    estimator = model(max_variance=10).fit(X, characteristics=panel)
    _, future = data(n=2, offset=11)
    returns = pd.DataFrame(
        [[0, 0, 0]] * 2, index=pd.date_range("2020-01-13", periods=2), columns=X.columns
    )
    factors = pd.DataFrame(
        {"factor": np.arange(30)}, index=pd.date_range("2020-01-01", periods=30)
    )
    estimator.evaluate_sequential(
        returns,
        future,
        initial_holdings=[1 / 3] * 3,
        covariance_window=5,
        prior_estimator=MetadataPrior(covariance_estimator=LedoitWolf()),
        prior_fit_params={"factors": factors},
    )
    for t, received in enumerate(MetadataPrior.seen):
        pd.testing.assert_frame_equal(
            received, factors.loc[pd.concat([X, returns.iloc[:t]]).tail(5).index]
        )
    with pytest.raises(ValueError, match="cover each history"):
        estimator.evaluate_sequential(
            returns,
            future,
            initial_holdings=[1 / 3] * 3,
            prior_fit_params={"factors": factors.iloc[20:]},
        )


def test_dynamic_eligibility_excludes_assets_and_charges_exit_turnover():
    X, panel = data()
    estimator = model(allow_universe_changes=True, long_only=True).fit(
        X, characteristics=panel
    )
    _, future = data(n=2, offset=11)
    dynamic = AssetPanel(
        fields={
            "signal": np.where(
                [[True, True, False], [False, True, True]], future["signal"], np.nan
            )
        },
        asset_names=future.asset_names,
        observations=future.observations,
        active_mask=np.array([[True, True, False], [False, True, True]]),
    )
    weights = estimator.predict_weights(dynamic).to_numpy()
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert abs(weights[0, 2]) < 1e-6
    assert abs(weights[1, 0]) < 1e-6
    estimator.set_params(max_turnover=0)
    with pytest.raises(ValueError, match="infeasible"):
        estimator.predict_weights(dynamic, holdings=np.full((2, 3), 1 / 3))


def test_dynamic_training_uses_decision_eligible_label_assets():
    X, panel = data()
    active = np.ones(X.shape, dtype=bool)
    active[:4, 2] = False
    dynamic = AssetPanel(
        fields={"signal": np.where(active, panel["signal"], np.nan)},
        asset_names=panel.asset_names,
        observations=panel.observations,
        active_mask=active,
    )
    estimator = model(allow_universe_changes=True, optimizer=EqualWeighted()).fit(
        X, characteristics=dynamic
    )
    assert (estimator.target_weights_.iloc[:2, 2] == 0).all()
    np.testing.assert_allclose(estimator.target_weights_.sum(axis=1), 1)


def test_cash_ledger_funds_costs_from_initial_cash_exactly():
    X, panel = data()
    estimator = model(long_only=True).fit(X, characteristics=panel)
    _, future = data(n=1, offset=11)
    returns = pd.DataFrame(
        [[0, 0, 0]], index=pd.date_range("2020-01-13", periods=1), columns=X.columns
    )
    result = estimator.evaluate_execution(
        returns,
        future,
        initial_holdings=[0, 0, 0],
        initial_cash=1,
        cash_weight=0.2,
        transaction_costs=0.01,
    )
    capital = 1 / (1 + 0.01 * 0.8)
    np.testing.assert_allclose(result.ledger.wealth, capital, atol=1e-8)
    np.testing.assert_allclose(result.ledger.cost, 1 - capital, atol=1e-8)
    np.testing.assert_allclose(result.targets.sum(axis=1), 0.8, atol=1e-7)
    np.testing.assert_allclose(result.ledger.target_cash, 0.2, atol=1e-7)
    np.testing.assert_allclose(
        result.ending_weights.sum(axis=1) + result.ledger.ending_cash, 1
    )
    np.testing.assert_allclose(result.trades.sum(axis=1), 0.8 * capital, atol=1e-8)


def test_cash_execution_handles_exits_all_cash_and_reentry():
    X, panel = data()
    estimator = model(long_only=True, allow_universe_changes=True, max_turnover=2).fit(
        X, characteristics=panel
    )
    _, future = data(n=4, offset=11)
    active = np.array(
        [
            [True, True, True],
            [True, True, False],
            [False, False, False],
            [True, True, True],
        ]
    )
    dynamic = future
    returns = pd.DataFrame(
        np.where(active, 0.01, np.nan),
        index=pd.date_range("2020-01-13", periods=4),
        columns=X.columns,
    )
    result = estimator.evaluate_execution(
        returns,
        dynamic,
        initial_holdings=[0.3] * 3,
        initial_cash=0.1,
        cash_weight=0.1,
        cash_returns=0.001,
        transaction_costs=0.001,
        rebalance_every=10,
        eligibility=pd.DataFrame(active, index=future.observations, columns=X.columns),
    )
    assert result.ledger.rebalanced.all()
    assert result.trades.iloc[1, 2] < 0
    np.testing.assert_allclose(result.targets.iloc[2], 0, atol=1e-7)
    np.testing.assert_allclose(result.ledger.ending_cash.iloc[2], 1)
    assert result.trades.iloc[3].sum() > 0
    np.testing.assert_allclose(
        result.ending_weights.sum(axis=1) + result.ledger.ending_cash, 1
    )
    np.testing.assert_allclose(
        result.ledger.cost, result.trades.abs().sum(axis=1) * 0.001
    )


def test_cash_execution_constraints_use_funded_trades_and_post_fee_weights():
    X, panel = data()
    estimator = model(long_only=True, max_variance=0.23, max_turnover=0.85).fit(
        X, characteristics=panel
    )
    _, future = data(n=2, offset=11)
    returns = pd.DataFrame(
        [[0.01, 0, 0]] * 2,
        index=pd.date_range("2020-01-13", periods=2),
        columns=X.columns,
    )
    result = estimator.evaluate_execution(
        returns,
        future,
        initial_holdings=[0, 0, 0],
        initial_cash=1,
        cash_weight=0.2,
        covariance=np.eye(3),
        transaction_costs=0.01,
    )
    assert (result.ledger.turnover <= 0.85 + 1e-6).all()
    assert ((result.targets**2).sum(axis=1) <= 0.23 + 1e-6).all()
    assert (result.ledger.cost > 0).all()


def test_cash_execution_rejects_unvalued_held_asset_and_unknown_assets():
    X, panel = data()
    estimator = model(long_only=True).fit(X, characteristics=panel)
    _, future = data(n=1, offset=11)
    returns = pd.DataFrame(
        [[np.nan, 0, 0]],
        index=pd.date_range("2020-01-13", periods=1),
        columns=X.columns,
    )
    with pytest.raises(ValueError, match="unobserved return"):
        estimator.evaluate_execution(
            returns, future, initial_holdings=[0.3] * 3, initial_cash=0.1
        )
    with pytest.raises(ValueError, match="registered asset"):
        estimator.evaluate_execution(
            returns.rename(columns={"a": "new"}),
            future,
            initial_holdings=[0.3] * 3,
            initial_cash=0.1,
        )


def test_execution_with_real_factor_prior_and_dated_metadata():
    from skfolio.prior import TimeSeriesFactorModel

    X, panel = data(n=31)
    rng = np.random.default_rng(19)
    X = X + rng.normal(0, 0.01, X.shape)
    estimator = model(long_only=True, optimizer=EqualWeighted(), max_variance=10).fit(
        X, characteristics=panel
    )
    _, future = data(n=2, offset=31)
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (2, 3)),
        index=pd.date_range("2020-02-02", periods=2),
        columns=X.columns,
    )
    factors = pd.DataFrame(
        rng.normal(0, 0.01, (40, 1)),
        index=pd.date_range("2020-01-01", periods=40),
        columns=["market"],
    )
    result = estimator.evaluate_execution(
        returns,
        future,
        initial_holdings=[0.3] * 3,
        initial_cash=0.1,
        cash_weight=0.1,
        prior_estimator=TimeSeriesFactorModel(),
        prior_fit_params={"factors": factors},
        covariance_window=20,
    )
    assert np.isfinite(result.ledger.wealth).all()
    np.testing.assert_allclose(
        result.ending_weights.sum(axis=1) + result.ledger.ending_cash, 1
    )


def test_universe_exit_does_not_silently_relax_turnover_limit():
    X, panel = data()
    estimator = model(long_only=True, allow_universe_changes=True, max_turnover=0).fit(
        X, characteristics=panel
    )
    _, future = data(n=1, offset=11)
    returns = pd.DataFrame(
        [[0, 0, 0]], index=pd.date_range("2020-01-13", periods=1), columns=X.columns
    )
    with pytest.raises(ValueError, match="infeasible"):
        estimator.evaluate_execution(
            returns,
            future,
            initial_holdings=[0.3] * 3,
            initial_cash=0.1,
            cash_weight=0.1,
            eligibility=pd.DataFrame(
                [[True, True, False]], index=future.observations, columns=X.columns
            ),
        )
    assert estimator.last_prediction_observation_ == X.index[-1]


def test_label_optimizer_receives_routed_factor_metadata():
    from sklearn import config_context

    from skfolio.optimization import InverseVolatility
    from skfolio.prior import TimeSeriesFactorModel

    X, panel = data(n=31)
    rng = np.random.default_rng(31)
    X = X + rng.normal(0, 0.01, X.shape)
    factors = pd.DataFrame(
        rng.normal(size=(40, 1)),
        index=pd.date_range("2020-01-01", periods=40),
        columns=["factor"],
    )
    with config_context(enable_metadata_routing=True):
        optimizer = InverseVolatility(
            prior_estimator=TimeSeriesFactorModel().set_fit_request(factors=True)
        )
        estimator = model(optimizer=optimizer, window_size=5).fit(
            X, characteristics=panel, optimizer_fit_params={"factors": factors}
        )
        reference = clone(optimizer).fit(X.iloc[1:6], factors=factors.loc[X.index[1:6]])
    np.testing.assert_allclose(estimator.target_weights_.iloc[0], reference.weights_)
    assert not hasattr(optimizer, "weights_")
    with pytest.raises(ValueError, match="cover each history"):
        model(window_size=5).fit(
            X,
            characteristics=panel,
            optimizer_fit_params={"factors": factors.iloc[20:]},
        )
