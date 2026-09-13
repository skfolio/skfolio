from __future__ import annotations

import operator
import pickle
from copy import copy

import numpy as np
import pandas as pd
import pytest

from skfolio import FailedPortfolio, MultiPeriodPortfolio, Portfolio


def _drift_loop(
    returns: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return held-before-return weights, gross returns and final weights."""
    position_values = weights.astype(float).copy()
    cash = 1 - weights.sum()
    wealth = 1.0
    weights_path = []
    portfolio_returns = []
    for asset_return in returns:
        weights_path.append(position_values / wealth)
        previous_wealth = wealth
        position_values *= 1 + asset_return
        wealth = position_values.sum() + cash
        portfolio_returns.append(wealth / previous_wealth - 1)
    return (
        np.asarray(weights_path),
        np.asarray(portfolio_returns),
        position_values / wealth,
    )


@pytest.fixture()
def returns() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.10, -0.04, 0.02], [-0.03, 0.08, 0.01], [0.05, -0.02, -0.01]],
        index=pd.date_range("2024-01-02", periods=3),
        columns=["A", "B", "C"],
    )


@pytest.mark.parametrize(
    "weights",
    [
        np.array([0.5, 0.3, 0.2]),
        np.array([1.1, -0.4, 0.3]),
        np.array([0.4, 0.2, 0.1]),
    ],
    ids=["long_only", "long_short", "partially_invested"],
)
def test_weight_drift_matches_self_financing_identity(returns, weights):
    portfolio = Portfolio(X=returns, weights=weights, weight_drift=True)
    path, gross_returns, final_weights = _drift_loop(returns.to_numpy(), weights)

    np.testing.assert_allclose(portfolio.returns, gross_returns, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        portfolio.weights_per_observation.to_numpy(),
        path[:, portfolio.nonzero_assets_index],
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(portfolio.ending_weights, final_weights)

    cash = 1 - weights.sum()
    gross_wealth = np.cumprod(1 + gross_returns)
    np.testing.assert_allclose(path.sum(axis=1) + cash / np.r_[1, gross_wealth[:-1]], 1)
    expected_wealth = (weights * np.prod(1 + returns.to_numpy(), axis=0)).sum() + cash
    np.testing.assert_allclose(gross_wealth[-1], expected_wealth, rtol=1e-14)


def test_identical_asset_returns_preserve_relative_weights():
    X = np.broadcast_to([0.02, -0.01, 0.03], (3, 3)).T
    weights = np.array([0.4, 0.2, 0.1])
    portfolio = Portfolio(X=X, weights=weights, weight_drift=True)
    path = portfolio.weights_per_observation.to_numpy()

    np.testing.assert_allclose(
        path / path[:, [0]], np.broadcast_to(weights / weights[0], path.shape)
    )
    assert not np.allclose(path, weights)

    fully_invested = Portfolio(X=X, weights=weights / weights.sum(), weight_drift=True)
    np.testing.assert_allclose(
        fully_invested.weights_per_observation.to_numpy(),
        np.broadcast_to(weights / weights.sum(), X.shape),
    )


def test_ending_weights_follow_selected_convention(returns):
    weights = np.array([0.5, 0.3, 0.2])
    default = Portfolio(X=returns, weights=weights)
    drifted = Portfolio(X=returns, weights=weights, weight_drift=True)

    assert drifted._weights_path is None
    np.testing.assert_array_equal(default.returns, returns.to_numpy() @ weights)
    np.testing.assert_array_equal(default.ending_weights, weights)
    np.testing.assert_array_equal(
        default.weights_per_observation.to_numpy(),
        np.broadcast_to(weights, returns.shape),
    )
    assert drifted._weights_path is None
    assert not np.allclose(
        drifted.ending_weights,
        drifted.weights_per_observation.iloc[-1].to_numpy(),
    )
    assert drifted._weights_path is not None


def test_weight_drift_and_compounded_are_independent(returns):
    arithmetic = Portfolio(X=returns, weights=[0.5, 0.3, 0.2], weight_drift=True)
    compounded = Portfolio(
        X=returns,
        weights=[0.5, 0.3, 0.2],
        weight_drift=True,
        compounded=True,
    )

    np.testing.assert_array_equal(compounded.returns, arithmetic.returns)
    pd.testing.assert_frame_equal(
        compounded.weights_per_observation, arithmetic.weights_per_observation
    )
    np.testing.assert_array_equal(compounded.ending_weights, arithmetic.ending_weights)
    assert arithmetic.cumulative_returns[-1] == pytest.approx(arithmetic.returns.sum())
    assert compounded.cumulative_returns[-1] == pytest.approx(
        np.prod(1 + compounded.returns)
    )


def test_weight_drift_treats_nan_asset_returns_as_zero(returns):
    with_nan = returns.copy()
    with_nan.iloc[1, 0] = np.nan
    with_zero = with_nan.fillna(0)
    weights = [0.5, 0.3, 0.2]

    portfolio_nan = Portfolio(X=with_nan, weights=weights, weight_drift=True)
    portfolio_zero = Portfolio(X=with_zero, weights=weights, weight_drift=True)

    np.testing.assert_array_equal(portfolio_nan.returns, portfolio_zero.returns)
    pd.testing.assert_frame_equal(
        portfolio_nan.weights_per_observation,
        portfolio_zero.weights_per_observation,
    )
    np.testing.assert_array_equal(
        portfolio_nan.ending_weights,
        portfolio_zero.ending_weights,
    )


def test_costs_fees_and_turnover_are_unchanged(returns):
    weights = np.array([0.5, 0.3, 0.2])
    previous_weights = np.array([0.4, 0.4, 0.2])
    portfolio = Portfolio(
        X=returns,
        weights=weights,
        previous_weights=previous_weights,
        transaction_costs=0.002,
        management_fees=0.001,
        weight_drift=True,
    )
    _, gross_returns, _ = _drift_loop(returns.to_numpy(), weights)

    assert portfolio.turnover == pytest.approx(0.2)
    assert portfolio.total_cost == pytest.approx(0.002 * portfolio.turnover)
    assert portfolio.total_fee == pytest.approx(0.001)
    np.testing.assert_allclose(
        portfolio.returns,
        gross_returns - portfolio.total_cost - portfolio.total_fee,
    )


def test_first_period_turnover_and_failed_portfolio(returns):
    weights = np.array([0.5, -0.3, 0.2])
    portfolio = Portfolio(X=returns, weights=weights)
    assert portfolio.turnover == pytest.approx(np.abs(weights).sum())

    failed = FailedPortfolio(X=returns, weight_drift=True)
    assert failed.weight_drift is True
    assert np.isnan(failed.turnover)
    assert np.isnan(failed.ending_weights).all()


def test_non_positive_wealth_names_first_observation():
    observations = pd.Index(["first", "second"])
    X = pd.DataFrame([[-0.1, 0.0], [-1.0, 0.0]], index=observations)
    with pytest.raises(ValueError, match="'second'"):
        Portfolio(X=X, weights=[2.0, 0.0], weight_drift=True)

    default = Portfolio(X=X, weights=[2.0, 0.0])
    np.testing.assert_array_equal(default.ending_weights, default.weights)


@pytest.mark.parametrize("weight_drift", [False, True])
def test_empty_window_ending_weights_equal_targets(weight_drift):
    portfolio = Portfolio(
        X=np.empty((0, 2)), weights=[0.6, 0.4], weight_drift=weight_drift
    )

    np.testing.assert_array_equal(portfolio.ending_weights, portfolio.weights)
    assert portfolio.weights_per_observation.shape == (0, 2)


def test_operators_copy_and_pickle_preserve_convention(returns):
    drifted = Portfolio(X=returns, weights=[0.5, 0.3, 0.2], weight_drift=True)
    default = Portfolio(X=returns, weights=[0.5, 0.3, 0.2])

    assert copy(drifted).weight_drift is True
    assert pickle.loads(pickle.dumps(drifted)).weight_drift is True
    assert (drifted * 2).weight_drift is True
    with pytest.raises(ValueError, match="weight_drift"):
        _ = drifted + default
    with pytest.raises(ValueError, match="weight_drift"):
        _ = drifted - default


def test_multi_period_drift_properties_skip_failed_children(returns):
    first = Portfolio(
        X=returns.iloc[:2],
        weights=[0.5, 0.3, 0.2],
        previous_weights=[0.4, 0.4, 0.2],
        transaction_costs=0.002,
        weight_drift=True,
        name="period",
    )
    failed = FailedPortfolio(X=returns.iloc[[2]], weight_drift=True, name="failed")
    third = Portfolio(
        X=returns.iloc[[2]],
        weights=[0.3, 0.5, 0.2],
        previous_weights=first.ending_weights,
        transaction_costs=0.002,
        weight_drift=True,
        name="period",
    )
    portfolio = MultiPeriodPortfolio([first, failed, third])

    expected_turnover = pd.Series(
        [first.turnover, np.nan, third.turnover],
        index=[p.observations[0] for p in portfolio],
        name="turnover",
    )
    pd.testing.assert_series_equal(portfolio.turnover, expected_turnover)
    assert list(portfolio.ending_weights_dict) == [
        "period",
        "failed",
        "period_1",
    ]
    assert np.isnan(portfolio.ending_weights_dict["failed"]["A"])


@pytest.mark.parametrize(
    "transaction_costs, expected_cost",
    [
        (0.0, 0.0),
        (0.001, 0.002),
        ({"A": 0.002, "S": 0.003, "B": 0.001}, 0.0032),
        ({"A": 0.002, "S": 0.003, "B": 0.0}, 0.0022),
    ],
)
def test_liquidations_include_long_and_short_positions(
    returns, transaction_costs, expected_cost
):
    portfolio = Portfolio(
        X=returns[["B"]],
        weights=[1.0],
        previous_weights={"A": 0.8, "S": -0.2},
        transaction_costs=transaction_costs,
        weight_drift=True,
    )
    for restored in (
        portfolio,
        copy(portfolio),
        pickle.loads(pickle.dumps(portfolio)),
        portfolio * 1,
    ):
        assert restored.turnover == pytest.approx(2.0)
        assert restored.total_cost == pytest.approx(expected_cost)
        np.testing.assert_allclose(restored.returns, returns["B"] - expected_cost)
        np.testing.assert_array_equal(restored.previous_weights, [0.0])


def test_liquidation_inputs_survive_parameter_reconstruction(returns):
    previous = {"A": 1.0}
    costs = {"A": 0.002, "B": 0.001}
    portfolio = Portfolio(
        X=returns[["B"]],
        weights=[1.0],
        previous_weights=previous,
        transaction_costs=costs,
    )
    previous["A"] = 9.0
    costs["A"] = 0.5
    scaled = portfolio * 0.5
    assert scaled.turnover == pytest.approx(1.5)
    assert scaled.total_cost == pytest.approx(0.0025)
    restored = pickle.loads(pickle.dumps(MultiPeriodPortfolio([portfolio])))
    assert restored[0].total_cost == pytest.approx(0.003)


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
@pytest.mark.parametrize(
    "parameter, different_value",
    [
        ("previous_weights", {"A": 0.5}),
        ("transaction_costs", {"A": 0.002, "B": 0.001}),
    ],
)
def test_binary_operations_check_excluded_assets(
    returns, operation, parameter, different_value
):
    params = {
        "X": returns[["B"]],
        "weights": [1.0],
        "previous_weights": {"A": 1.0},
        "transaction_costs": {"A": 0.001, "B": 0.001},
    }
    first = Portfolio(**params)
    identical = Portfolio(**params)
    combined = operation(first, identical)
    expected_turnover = 1.0 + abs(operation(1.0, 1.0))
    assert combined.turnover == pytest.approx(expected_turnover)
    assert combined.total_cost == pytest.approx(0.001 * expected_turnover)

    different = Portfolio(**(params | {parameter: different_value}))
    for left, right in ((first, different), (different, first)):
        with pytest.raises(ValueError, match=parameter):
            operation(left, right)


@pytest.mark.parametrize("operation", [operator.add, operator.sub])
def test_binary_operations_accept_equivalent_named_and_array_inputs(returns, operation):
    named = Portfolio(
        X=returns,
        weights=[0.5, 0.3, 0.2],
        previous_weights={"A": 0.6, "B": 0.4},
        transaction_costs={"A": 0.001},
    )
    array = Portfolio(
        X=returns,
        weights=[0.5, 0.3, 0.2],
        previous_weights=[0.6, 0.4, 0.0],
        transaction_costs=[0.001, 0.0, 0.0],
    )
    combined = operation(named, array)
    np.testing.assert_array_equal(
        combined.weights, operation(named.weights, array.weights)
    )
    assert combined.total_cost == pytest.approx(0.001 * abs(combined.weights[0] - 0.6))


def test_liquidation_requires_identifiable_cost_rates(returns):
    with pytest.raises(ValueError, match="costs for liquidated assets are unavailable"):
        Portfolio(
            X=returns[["B"]],
            weights=[1.0],
            previous_weights={"A": 1.0},
            transaction_costs=[0.001],
        )


def test_pickle_does_not_depend_on_constructor_parameter_order(returns, monkeypatch):
    portfolio = Portfolio(
        X=returns,
        weights=[0.5, 0.3, 0.2],
        weight_drift=True,
        compounded=True,
        sample_weight=np.array([0.2, 0.3, 0.5]),
        min_acceptable_return=0.001,
        cvar_beta=0.9,
    )
    payload = pickle.dumps(portfolio)
    original_init = Portfolio.__init__

    def reordered_init(self, weights, X, **kwargs):
        original_init(self, X=X, weights=weights, **kwargs)

    monkeypatch.setattr(Portfolio, "__init__", reordered_init)
    restored = pickle.loads(payload)
    assert restored.weight_drift is True
    assert restored.compounded is True
    assert restored.min_acceptable_return == 0.001
    assert restored.cvar_beta == 0.9
    np.testing.assert_array_equal(restored.sample_weight, portfolio.sample_weight)
    np.testing.assert_array_equal(restored.returns, portfolio.returns)


def test_turnover_omits_empty_children(returns):
    first = Portfolio(X=returns.iloc[:2], weights=[0.5, 0.3, 0.2])
    empty = Portfolio(X=returns.iloc[:0], weights=[0.5, 0.3, 0.2])
    failed = FailedPortfolio(X=returns.iloc[2:], weight_drift=True)
    path = MultiPeriodPortfolio([empty, first, empty, failed])
    pd.testing.assert_series_equal(
        path.turnover,
        pd.Series(
            [1.0, np.nan], index=[returns.index[0], returns.index[2]], name="turnover"
        ),
    )
    assert len(path) == 4
    empty_turnover = MultiPeriodPortfolio([empty]).turnover
    assert empty_turnover.empty
    assert empty_turnover.dtype == float
    assert empty_turnover.name == "turnover"
    restored = pickle.loads(pickle.dumps(failed))
    assert restored.weight_drift is True
    assert np.isnan(restored.turnover)
