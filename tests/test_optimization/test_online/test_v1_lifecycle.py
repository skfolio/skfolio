"""Integration contracts shared by Carlo's existing strategy families."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skfolio.measures import PerfMeasure
from skfolio.optimization.online._base import BaseOnlineOptimization
from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._winner import FollowTheWinner


@pytest.fixture(
    params=[
        FollowTheWinner(strategy="eg", update_mode="omd", learning_rate=0.1),
        FollowTheWinner(strategy="eg", update_mode="ftrl", learning_rate=0.1),
        FollowTheWinner(strategy="ogd", update_mode="omd", learning_rate=0.1),
        FollowTheWinner(strategy="adagrad", learning_rate=0.1),
        FollowTheLoser(strategy="olmar"),
        FollowTheLoser(strategy="pamr"),
        FollowTheLoser(strategy="cwmr"),
        FollowTheLoser(strategy="rmr"),
    ]
)
def estimator(request):
    return clone(request.param)


@pytest.fixture
def returns():
    return np.random.default_rng(26).normal(0.001, 0.02, (15, 3))


def test_blocks_and_rows_preserve_the_same_strategy_state(estimator, returns):
    batch = clone(estimator).fit(returns)
    streamed = clone(estimator)
    for row in returns:
        streamed.partial_fit(row)
    blocked = estimator.partial_fit(returns[:4]).partial_fit(returns[4:11])
    blocked.partial_fit(returns[11:])
    for model in [streamed, blocked]:
        assert isinstance(model, BaseOnlineOptimization)
        assert_allclose(model.weights_, batch.weights_, atol=1e-12)
        assert_allclose(model.all_weights_, batch.all_weights_, atol=1e-12)
        assert not hasattr(model, "wealth_")
        assert model.n_observations_ == len(returns)
        assert_allclose(model.initial_weights_, batch.all_weights_[0])
        # A further update also checks hidden predictors and adaptive state.
        model.partial_fit(returns[:1])
    batch.partial_fit(returns[:1])
    assert_allclose(streamed.weights_, batch.weights_, atol=1e-12)
    assert_allclose(blocked.weights_, batch.weights_, atol=1e-12)


def test_fit_resets_even_with_legacy_warm_start(estimator, returns):
    estimator.set_params(warm_start=True)
    estimator.partial_fit(returns).fit(returns[:6])
    fresh = clone(estimator).fit(returns[:6])
    assert_allclose(estimator.weights_, fresh.weights_)
    assert_allclose(estimator.all_weights_, fresh.all_weights_)
    assert estimator.n_observations_ == 6


def test_feature_names_and_new_universe_on_fresh_fit(estimator, returns):
    frame = pd.DataFrame(returns, columns=["a", "b", "c"])
    estimator.partial_fit(frame[:5]).partial_fit(frame[5:])
    assert list(estimator.feature_names_in_) == ["a", "b", "c"]
    previous = estimator.weights_.copy()
    with pytest.raises(ValueError, match="feature names"):
        estimator.partial_fit(frame[["b", "a", "c"]])
    assert_allclose(estimator.weights_, previous)
    estimator.fit(frame[["b", "a"]])
    assert list(estimator.feature_names_in_) == ["b", "a"]
    assert estimator.weights_.shape == (2,)


def test_validate_whole_block_before_learning(estimator, returns):
    estimator.partial_fit(returns[:3])
    previous = estimator.weights_.copy()
    bad = returns[3:].copy()
    bad[-1, 0] = np.nan
    with pytest.raises(ValueError):
        estimator.partial_fit(bad)
    assert estimator.n_observations_ == 3
    assert_allclose(estimator.weights_, previous)


def test_one_asset_and_one_dimensional_fit(estimator):
    estimator.fit(np.array([0.01]))
    estimator.partial_fit(np.array([[0.02], [-0.01]]))
    assert_allclose(estimator.weights_, [1.0])
    assert estimator.n_observations_ == 3


def test_reset_recreates_objective(returns):
    model = FollowTheWinner(learning_rate=0.1).fit(returns)
    model.set_params(objective=PerfMeasure.MEAN).fit(returns)
    assert_allclose(model.weights_, clone(model).fit(returns).weights_)
