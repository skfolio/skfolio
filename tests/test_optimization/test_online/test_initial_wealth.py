"""Initial capital configures allocation; Portfolio owns evaluated wealth."""

import numpy as np
import pytest
from sklearn.base import clone

from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._winner import FollowTheWinner


@pytest.fixture(
    params=[FollowTheWinner(learning_rate=0.05), FollowTheLoser(strategy="pamr")]
)
def model(request):
    return clone(request.param)


@pytest.mark.parametrize(
    "capital", [None, 1.0, 10000.0, np.array([100.0, 300.0, 600.0])]
)
def test_initial_capital_does_not_create_an_execution_account(model, capital):
    model.set_params(initial_wealth=capital)
    model.fit([[0.01, 0.02, -0.01], [0.02, -0.01, 0.01]])
    assert not hasattr(model, "wealth_")
    assert not hasattr(model, "all_wealth_")
    expected = (
        capital / capital.sum() if isinstance(capital, np.ndarray) else np.ones(3) / 3
    )
    np.testing.assert_allclose(model.initial_weights_, expected)


@pytest.mark.parametrize(
    "fee,cost", [(0.0, 0.0), (0.001, 0.0), (0.0, 0.002), (0.001, 0.002)]
)
def test_execution_returns_and_compounding_are_owned_by_portfolio(model, fee, cost):
    model.set_params(
        initial_wealth=10000.0,
        transaction_costs=cost,
        management_fees=fee,
        portfolio_params={"weight_drift": True},
    )
    X = np.array([[0.01, 0.02, -0.01], [0.02, -0.01, 0.01], [-0.01, 0.03, 0.01]])
    path = model.fit_predict(X)
    returns = []
    prev = np.zeros(3)
    for p, row in zip(path, X, strict=True):
        np.testing.assert_allclose(p.previous_weights, prev)
        returns.append(
            p.weights @ row
            - fee * p.weights.sum()
            - cost * np.abs(p.weights - prev).sum()
        )
        prev = p.ending_weights
    np.testing.assert_allclose(path.returns, returns)
    np.testing.assert_allclose(
        np.prod(1 + path.returns), np.prod(1 + np.array(returns))
    )
    assert model.n_observations_ == len(X)
    assert not hasattr(model, "wealth_")


def test_fit_resets_learning_without_resetting_external_holdings(model):
    X = np.array([[0.01, 0.02, -0.01], [0.02, 0.01, 0.0]])
    model.set_params(previous_weights=[0.2, 0.3, 0.5])
    model.fit(X)
    expected = model.weights_.copy()
    model.fit(X)
    np.testing.assert_allclose(model.weights_, expected)
    np.testing.assert_allclose(model.previous_weights, [0.2, 0.3, 0.5])
    model.partial_fit(X)
    assert model.n_observations_ == 4


class TestInitialWealthValidation:
    """Test validation of initial_wealth parameter."""

    def test_array_weights_and_array_wealth_raises_error(self):
        """Test that both array initial_weights and initial_wealth raises ValueError."""
        with pytest.raises(ValueError, match="cannot both be arrays"):
            est = FollowTheWinner(
                strategy="eg",
                initial_weights=np.array([0.3, 0.3, 0.4]),
                initial_wealth=np.array([10000.0, 20000.0, 30000.0]),
                learning_rate=0.05,
            )
            X = np.array([[0.01, 0.02, -0.01]])
            est.fit(X)

    def test_scalar_wealth_and_array_weights_compatible(self):
        """Test scalar initial_wealth with array initial_weights is allowed."""
        initial_wealth = 100000.0
        initial_weights = np.array([0.2, 0.3, 0.5])

        est = FollowTheWinner(
            strategy="eg",
            initial_wealth=initial_wealth,
            initial_weights=initial_weights,
            learning_rate=0.05,
        )

        X = np.array([[0.01, 0.02, -0.01]])
        est.fit(X)

        # Should work without error
        assert not hasattr(est, "wealth_")
        assert np.allclose(est.all_weights_[0], initial_weights)

    def test_negative_wealth_raises_error(self):
        """Test negative initial_wealth raises ValueError."""
        with pytest.raises(ValueError, match="initial_wealth"):
            est = FollowTheWinner(
                strategy="eg",
                initial_wealth=-1000.0,
                learning_rate=0.05,
            )
            X = np.array([[0.01, 0.02, -0.01]])
            est.fit(X)

    def test_zero_wealth_raises_error(self):
        """Test zero initial_wealth raises ValueError."""
        with pytest.raises(ValueError, match="initial_wealth"):
            est = FollowTheWinner(
                strategy="eg",
                initial_wealth=0.0,
                learning_rate=0.05,
            )
            X = np.array([[0.01, 0.02, -0.01]])
            est.fit(X)

    def test_array_wealth_shape_mismatch(self):
        """Test array initial_wealth with wrong shape raises error."""
        with pytest.raises(ValueError, match="must have shape"):
            est = FollowTheWinner(
                strategy="eg",
                initial_wealth=np.array([10000.0, 20000.0]),  # Wrong shape
                learning_rate=0.05,
            )
            X = np.array([[0.01, 0.02, -0.01]])  # 3 assets
            est.fit(X)
