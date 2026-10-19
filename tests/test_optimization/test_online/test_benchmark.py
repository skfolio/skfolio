"""Tests for online hindsight benchmarks."""

import numpy as np
import pytest

from skfolio.optimization.online._benchmark import (
    BestAsset,
    ConstantRebalancedPortfolio,
    _constant_wealth_history,
)


def test_constant_rebalanced_with_custom_weights(X_small):
    """Test ConstantRebalancedPortfolio with custom weights."""
    n_assets = X_small.shape[1]
    weights = np.ones(n_assets) / n_assets
    constant_rebalanced = ConstantRebalancedPortfolio(weights=weights)
    constant_rebalanced.fit(X_small)


def test_comparator_uniform_weights(X_small):
    model = ConstantRebalancedPortfolio(weights=None).fit(X_small)
    expected_returns = np.asarray(X_small).mean(axis=1)
    np.testing.assert_allclose(model.weights_, 1 / X_small.shape[1])
    np.testing.assert_allclose(
        model.all_wealth_, np.r_[1.0, np.cumprod(1 + expected_returns)]
    )
    np.testing.assert_allclose(model.all_weights_, 1 / X_small.shape[1])
    model.partial_fit(X_small.iloc[:1])
    np.testing.assert_allclose(model.weights_, 1 / X_small.shape[1])


def test_best_asset_picks_best_performer(X_small):
    """Test BestAsset picks the asset with highest cumulative log return."""
    bs = BestAsset()
    bs.fit(X_small)

    # Verify one-hot vector
    assert np.sum(bs.weights_) == pytest.approx(1.0)
    assert np.sum(bs.weights_ > 0) == 1

    # Verify it picks the best asset
    relatives = 1.0 + X_small.values
    log_returns = np.sum(np.log(relatives), axis=0)
    best_idx = np.argmax(log_returns)
    assert bs.weights_[best_idx] == pytest.approx(1.0)


def test_best_constant_solver_rejects_infeasible_nonconvergence():
    """An inaccurate infeasible allocation cannot serve as a regret comparator."""
    from skfolio.exceptions import ConvexOptimizationError
    from skfolio.optimization.online._regret import _solve_best_constant_weights

    # Construct data where the best-constant portfolio is far from uniform (one asset dominates),
    # so the solver needs many iterations to converge from uniform init.
    rng = np.random.default_rng(42)
    relatives = 1.0 + rng.standard_normal((100, 5)) * 0.02
    relatives[:, 0] += 0.05  # asset 0 strongly dominates

    with pytest.warns(UserWarning, match="inaccurate"):
        with pytest.raises(ConvexOptimizationError, match="infeasible"):
            _solve_best_constant_weights(relatives, solver_params={"max_iter": 1})


@pytest.mark.parametrize("weights", [[0.8, 0.2], [0.4, 0.1], [1.2, -0.2]])
def test_constant_wealth_preserves_cash_and_short_positions(weights):
    returns = np.array([[0.1, -0.05], [-0.02, 0.03], [0.04, 0.01]])
    model = ConstantRebalancedPortfolio(weights=weights).fit(returns)
    # Independent account calculation: cash earns zero and targets are restored
    # before each observation, including when an asset is held short.
    value = 1.0
    expected = [value]
    for row in returns:
        cash = value * (1.0 - sum(weights))
        positions = value * np.array(weights)
        value = cash + sum(positions * (1.0 + row))
        expected.append(value)
    np.testing.assert_allclose(model.all_wealth_, expected)
    assert model.wealth_ == pytest.approx(value)
    np.testing.assert_allclose(model.all_weights_, [weights] * len(returns))
    model.all_weights_[0, 0] = 99
    assert model.all_weights_[1, 0] == weights[0]
    assert model.weights_[0] == weights[0]


def test_best_asset_wealth_and_refit():
    model = BestAsset().fit([[0.1, 0.05], [-0.2, 0.05]])
    np.testing.assert_array_equal(model.weights_, [0.0, 1.0])
    np.testing.assert_allclose(model.all_wealth_, [1.0, 1.05, 1.1025])
    model.fit([[0.2, 0.1]])
    np.testing.assert_array_equal(model.weights_, [1.0, 0.0])
    np.testing.assert_allclose(model.all_wealth_, [1.0, 1.2])
    assert model.all_weights_.shape == (1, 2)


@pytest.mark.parametrize(
    "factory",
    [
        BestAsset,
        lambda: ConstantRebalancedPortfolio(weights=None),
    ],
)
@pytest.mark.parametrize(
    "returns", [[], [[np.nan, 0.0]], [[np.inf, 0.0]], [[-1.0, 0.0]], [0.1, 0.2]]
)
def test_benchmarks_reject_invalid_returns(factory, returns):
    with pytest.raises(ValueError):
        factory().fit(returns)


@pytest.mark.parametrize("initial_wealth", [0.0, -1.0, np.nan, np.inf, [1.0, 2.0]])
def test_initial_wealth_must_be_positive_scalar(initial_wealth, X_small):
    with pytest.raises(
        ValueError, match="initial_wealth must be a finite positive scalar"
    ):
        _constant_wealth_history(
            relatives=1.0 + np.asarray(X_small),
            weights=np.full(X_small.shape[1], 1 / X_small.shape[1]),
            initial_wealth=initial_wealth,
        )
