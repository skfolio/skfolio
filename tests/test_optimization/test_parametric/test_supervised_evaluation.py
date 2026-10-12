"""Supervised policy evaluation contracts."""

from itertools import pairwise
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.optimization import (
    EqualWeighted,
)

from ._supervised_helpers import data, model


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


@pytest.mark.parametrize("cost", [0.0, 0.001])
def test_cash_short_positions_reconcile_without_borrow_fees(cost):
    X, panel = data()
    estimator = model().fit(X, characteristics=panel)
    estimator.regressor_.coef_[:] = 4
    returns, future = data(n=3, offset=11)
    returns.index = future.observations + pd.Timedelta(days=1)
    returns.iloc[:] = [0.1, -0.02, -0.1]
    result = estimator.evaluate_execution(
        returns,
        future,
        initial_holdings=[-0.5, 0.5, 0.9],
        initial_cash=0.1,
        cash_weight=0.1,
        transaction_costs=cost,
        rebalance_every=2,
    )
    assert (result.targets["a"] < 0).all()
    np.testing.assert_allclose(
        result.ending_weights.sum(axis=1) + result.ledger.ending_cash, 1
    )
    np.testing.assert_allclose(
        result.ledger.cost, cost * result.trades.abs().sum(axis=1)
    )
    expected = (1 - result.ledger.cost) * (
        1 + (result.targets * returns).sum(axis=1)
    ) - 1
    np.testing.assert_allclose(result.ledger["return"], expected, atol=1e-10)
    assert not result.ledger.rebalanced.iloc[1]


def test_missing_short_return_rejected():
    X, panel = data()
    estimator = model().fit(X, characteristics=panel)
    estimator.regressor_.coef_[:] = 4
    returns, future = data(n=1, offset=11)
    returns.index = future.observations + pd.Timedelta(days=1)
    returns.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="held asset"):
        estimator.evaluate_execution(
            returns, future, initial_holdings=0, initial_cash=1
        )
