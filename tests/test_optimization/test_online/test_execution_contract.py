"""Execution boundaries for the port of Carlo's online framework."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._winner import FollowTheWinner


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_loss_uses_pre_observation_target(kind):
    model = (
        FollowTheWinner(learning_rate=2.0, update_mode="omd")
        if kind == "winner"
        else FollowTheLoser(strategy="pamr", epsilon=0.5)
    )
    X = np.array([[0.2, -0.1], [-0.1, 0.3]])
    before = np.array([0.5, 0.5])
    cumulative = 0.0
    for row in X:
        model.partial_fit(row)
        expected = -np.log(before @ (1 + row))
        assert_allclose(model.loss_, expected)
        cumulative += expected
        before = model.weights_.copy()
    if kind == "winner":
        assert_allclose(model.cumulative_loss_, cumulative)


@pytest.mark.parametrize("strategy", ["eg", "pamr", "olmar", "rmr", "cwmr"])
def test_turnover_uses_fresh_external_holdings_including_cold_start(strategy):
    model = (
        FollowTheWinner(strategy=strategy, learning_rate=0.2)
        if strategy == "eg"
        else FollowTheLoser(strategy=strategy)
    )
    model.set_params(max_turnover=0.0, previous_weights=[0.8, 0.2])
    model.partial_fit([0.2, -0.1])
    assert_allclose(model.weights_, [0.8, 0.2], atol=1e-7)
    model.set_params(previous_weights={"a": 0.2, "b": 0.8})
    # Use positional holdings here; named alignment has its own test.
    model.set_params(previous_weights=[0.2, 0.8]).partial_fit([-0.1, 0.2])
    assert_allclose(model.weights_, [0.2, 0.8], atol=1e-7)


def test_zero_turnover_does_not_project_actual_holdings_into_feasibility():
    from skfolio.optimization.online._projection import project_with_turnover

    anchor = np.array([0.9, 0.1])
    with pytest.raises((ValueError, RuntimeError), match=r"holdings|projection failed"):
        project_with_turnover(
            [0.5, 0.5],
            previous_weights=anchor,
            max_turnover=0,
            lower=0,
            upper=0.6,
            budget=1,
        )
    assert_allclose(anchor, [0.9, 0.1])


def test_positive_turnover_cap_rejects_infeasible_funding():
    from skfolio.optimization.online._projection import project_with_turnover

    with pytest.raises((ValueError, RuntimeError), match=r"holdings|projection failed"):
        project_with_turnover(
            [0.5, 0.5],
            previous_weights=[0, 0],
            max_turnover=0.1,
            lower=0,
            upper=1,
            budget=1,
        )


def test_mixed_eg_target_is_next_engine_reference():
    model = FollowTheWinner(
        update_mode="omd", learning_rate=2, eg_tilde=True, eg_tilde_alpha=0.3
    )
    model.partial_fit([0.2, -0.1])
    assert_allclose(model.weights_, model._foco_engine._x_t)


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_warmup_only_final_target_uses_entry_holdings(kind):
    model = (
        FollowTheWinner(learning_rate=0.5, update_mode="omd")
        if kind == "winner"
        else FollowTheLoser(strategy="pamr", epsilon=0.5)
    )
    model.set_params(max_turnover=0, previous_weights=[0.8, 0.2])
    model.fit(np.array([[0.2, -0.1], [-0.1, 0.3], [0.1, -0.1]]))
    assert_allclose(model.initial_weights_, [0.5, 0.5])
    assert not np.allclose(model.all_weights_[1], [0.8, 0.2])
    assert_allclose(model.weights_, [0.8, 0.2], atol=1e-7)
    assert model.n_observations_ == 3
    with pytest.raises(ValueError, match="row-aligned"):
        model.partial_fit([[0.1, -0.1], [-0.1, 0.1]])


def test_failure_requires_reset_before_continuing():
    model = FollowTheWinner(
        max_turnover=0, previous_weights=[0.9, 0.1], max_weights=0.6
    )
    with pytest.raises((ValueError, RuntimeError), match=r"holdings|projection failed"):
        model.partial_fit([0.1, -0.1])
    with pytest.raises(RuntimeError, match="reset"):
        model.partial_fit([0.0, 0.0])
    model.set_params(previous_weights=[0.5, 0.5]).fit([[0.0, 0.0]])
    assert_allclose(model.weights_, [0.5, 0.5])


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_learning_does_not_book_wealth_or_trades(kind):
    model = FollowTheWinner() if kind == "winner" else FollowTheLoser(strategy="olmar")
    model.fit([[0.2, -0.1], [-0.1, 0.3]])
    assert not hasattr(model, "wealth_")
    assert not hasattr(model, "all_wealth_")
    assert not hasattr(model, "_current_previous_weights_")


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_evaluator_matches_manual_execution_with_drift_costs_and_turnover(kind):
    from sklearn.base import clone

    from skfolio.measures import RatioMeasure
    from skfolio.model_selection import online_predict, online_score
    from skfolio.portfolio import Portfolio

    model = (
        FollowTheWinner(learning_rate=0.3, update_mode="omd")
        if kind == "winner"
        else FollowTheLoser(strategy="pamr", epsilon=0.7)
    )
    model.set_params(
        previous_weights=[0.6, 0.4],
        max_turnover=0.2,
        transaction_costs=0.002,
        management_fees=0.001,
        portfolio_params={"weight_drift": True},
    )
    X = np.random.default_rng(123).normal(0.002, 0.04, (14, 2))
    result = online_predict(model, X, warmup_size=3, test_size=1)
    manual = clone(model).fit(X[:3])
    expected = []
    for row in X[3:]:
        anchor = np.asarray(manual.previous_weights)
        assert np.abs(manual.weights_ - anchor).sum() <= 0.2 + 1e-7
        p = Portfolio(
            row[None, :],
            weights=manual.weights_,
            previous_weights=anchor,
            transaction_costs=0.002,
            management_fees=0.001,
            weight_drift=True,
        )
        expected.append(p.returns[0])
        manual.set_params(previous_weights=p.ending_weights).partial_fit(row)
    assert_allclose(result.returns, expected, atol=1e-10)
    assert_allclose(
        online_score(
            model, X, warmup_size=3, test_size=1, scoring=RatioMeasure.SHARPE_RATIO
        ),
        result.sharpe_ratio,
    )


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_no_warmup_evaluation_uses_portfolio_fee_convention(kind):
    model = (
        FollowTheWinner(learning_rate=0.2)
        if kind == "winner"
        else FollowTheLoser(strategy="pamr")
    )
    model.set_params(
        management_fees=0.01,
        transaction_costs=0.002,
        portfolio_params={"weight_drift": True},
    )
    p = model.fit_predict(np.array([[0.2, -0.1], [-0.1, 0.2]]))
    assert_allclose(p.returns[0], 0.05 - 0.01 - 0.002)
    assert model.n_observations_ == 2
    assert not hasattr(model, "wealth_")


def test_fee_aware_algorithm_loss_matches_additive_management_fee():
    model = FollowTheWinner(management_fees=0.01).partial_fit([0.2, -0.1])
    assert_allclose(model.loss_, -np.log(1 + 0.05 - 0.01))


def test_failure_also_blocks_prediction():
    model = FollowTheWinner(
        max_turnover=0, previous_weights=[0.9, 0.1], max_weights=0.6
    )
    with pytest.raises(RuntimeError, match="projection failed"):
        model.partial_fit([0.1, -0.1])
    with pytest.raises(RuntimeError, match="reset"):
        model.predict([[0.0, 0.0]])


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_optional_target_history_does_not_change_learning(kind):
    from sklearn.base import clone

    model = (
        FollowTheWinner(learning_rate=0.2)
        if kind == "winner"
        else FollowTheLoser(strategy="olmar")
    )
    X = np.random.default_rng(52).normal(0, 0.01, (20, 3))
    expected = clone(model).fit(X)
    model.set_params(store_weights=False).fit(X)
    assert_allclose(model.weights_, expected.weights_)
    assert not hasattr(model, "all_weights_")
    model.partial_fit(X[:1])
    assert not hasattr(model, "all_weights_")


def test_named_holdings_align_to_feature_order_and_are_not_mutated():
    import pandas as pd

    holdings = {"b": 0.8, "a": 0.2}
    model = FollowTheWinner(max_turnover=0, previous_weights=holdings)
    model.partial_fit(pd.DataFrame([[0.2, -0.1]], columns=["a", "b"]))
    assert_allclose(model.weights_, [0.2, 0.8])
    assert holdings == {"b": 0.8, "a": 0.2}
    assert not model._execution_anchor.flags.writeable


def test_entry_override_is_not_applied_to_later_rebalances():
    from skfolio.model_selection import online_predict

    model = FollowTheWinner(
        learning_rate=0.5,
        max_turnover=0,
        previous_weights=[0.0, 0.0],
        portfolio_params={"weight_drift": True},
    )
    X = np.array([[0.2, -0.1], [-0.1, 0.2], [0.1, 0.2], [0.2, -0.1], [0.1, -0.1]])
    with pytest.raises((ValueError, RuntimeError), match=r"holdings|projection failed"):
        online_predict(model, X, warmup_size=2, test_size=1)
    path = online_predict(
        model,
        X,
        warmup_size=2,
        test_size=1,
        entry_rebalancing_params={"max_turnover": None},
    )
    for prev, p in zip(list(path)[:-1], list(path)[1:], strict=True):
        assert_allclose(p.weights, prev.ending_weights, atol=1e-7)
    assert model.max_turnover == 0


def test_cost_aware_blocks_require_execution_context():
    model = FollowTheLoser(
        strategy="pamr", transaction_costs=0.01, previous_weights=[0.5, 0.5]
    )
    with pytest.raises(ValueError, match="row-aligned"):
        model.partial_fit([[0.1, -0.1], [-0.1, 0.1]])
    model.fit([[0.1, -0.1], [-0.1, 0.1]])
    assert model.n_observations_ == 2


def test_purged_evaluation_rejected_before_learning():
    from skfolio.model_selection import online_predict

    with pytest.raises(ValueError, match="purged"):
        online_predict(
            FollowTheWinner(),
            np.zeros((12, 2)),
            warmup_size=3,
            test_size=1,
            purged_size=1,
        )


@pytest.mark.parametrize("kind", ["winner", "loser"])
def test_grid_search_refits_last_without_extra_execution(kind):
    from skfolio.model_selection import OnlineGridSearch, online_predict

    model = (
        FollowTheWinner(learning_rate=0.1)
        if kind == "winner"
        else FollowTheLoser(strategy="pamr")
    )
    model.set_params(
        previous_weights=[0.5, 0.5],
        max_turnover=0.1,
        portfolio_params={"weight_drift": True},
    )
    X = np.random.default_rng(9).normal(0.001, 0.01, (18, 2))
    search = OnlineGridSearch(
        model, param_grid={"learning_rate": [0.05, 0.1]}, warmup_size=4, test_size=1
    ).fit(X)
    assert search.best_estimator_.n_observations_ == len(X)
    assert not hasattr(search.best_estimator_, "wealth_")
    path = online_predict(search.best_estimator_, X, warmup_size=4, test_size=1)
    assert len(path.returns) == len(X) - 4


def test_nonfinite_candidate_invalidates_stream(monkeypatch):
    model = FollowTheWinner().partial_fit([0.01, -0.01])

    def invalid_update(row):
        model.weights_ = np.array([np.nan, np.nan])

    monkeypatch.setattr(model, "_update_one", invalid_update)
    with pytest.raises(RuntimeError, match="invalid target"):
        model.partial_fit([0.0, 0.0])
    with pytest.raises(RuntimeError, match="reset"):
        model.predict([[0.0, 0.0]])
