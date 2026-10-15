"""Decision penalties and booked costs are separate operations."""

import numpy as np

from skfolio.optimization.online._loser import FollowTheLoser


def test_loser_penalty_does_not_book_costs_inside_learning():
    model = FollowTheLoser(
        strategy="pamr",
        epsilon=0.8,
        transaction_costs=0.01,
        previous_weights=[0.8, 0.2],
    )
    model.partial_fit([0.2, -0.1])
    model._strategy_impl.penalize_turnover = True
    model.partial_fit([-0.1, 0.2])
    assert not hasattr(model, "wealth_")
    p = model.predict([[0.1, -0.1]])
    expected = (
        model.weights_ @ np.array([0.1, -0.1])
        - 0.01 * np.abs(model.weights_ - [0.8, 0.2]).sum()
    )
    np.testing.assert_allclose(p.returns, [expected])
