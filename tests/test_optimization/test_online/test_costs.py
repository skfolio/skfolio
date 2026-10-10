import warnings

import numpy as np
import pytest

from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._mixins import FTLStrategy, FTWStrategy
from skfolio.optimization.online._winner import FollowTheWinner


def test_fees_within_bounds_behave_normally():
    x = np.array([0.01, -0.005, 0.0], dtype=float)

    a = FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate=0.1,
        management_fees=0.002,
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        a.partial_fit(x)
        # no warnings expected from fee handling within [0,1)
        msgs = "\n".join(str(ww.message).lower() for ww in w)
        assert "fee" not in msgs


def test_fees_upper_bound_validation():
    x = np.array([0.01, 0.0, 0.0], dtype=float)
    # management_fees >= 1.0 should be rejected by parameter validation
    est = FollowTheLoser(
        strategy=FTLStrategy.OLMAR, management_fees=np.array([1.0, 0.0, 0.0])
    )
    with pytest.raises(ValueError):
        est.partial_fit(x)


def test_apply_fees_to_phi_subtracts_future_fee_once():
    # three periods to allow OLMAR-1 to update once with window=2
    x1 = np.array([0.99, 1.01, 1.0], dtype=float) - 1.0
    x2 = np.array([1.02, 0.98, 1.0], dtype=float) - 1.0

    # Prepare two estimators with identical initial state except apply_fees_to_phi
    a = FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        olmar_window=2,
        epsilon=1.0,
        management_fees=0.2,
        apply_fees_to_phi=False,
    )
    b = FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        olmar_window=2,
        epsilon=1.0,
        management_fees=0.2,
        apply_fees_to_phi=True,
    )
    # Initialize both with first sample to seed predictor histories
    a.partial_fit(x1)
    b.partial_fit(x1)

    # Compute second-step effective relatives for each and the corresponding phi
    x2_gross = 1.0 + x2
    d = x2_gross.shape[0]
    phi_a = a._compute_predictor(x2_gross, d)

    phi_b = b._compute_predictor(x2_gross, d)

    # Historical predictor inputs are gross; charge the future fee once.
    assert np.allclose(phi_b, phi_a - 0.2)


@pytest.mark.parametrize("strategy", ["eg", "adagrad", "olmar", "pamr"])
@pytest.mark.parametrize("fee,cost", [(0.0, 0.002), (0.002, 0.0), (0.002, 0.002)])
def test_fees_and_costs_booked_once_by_portfolio(strategy, fee, cost):
    from skfolio.model_selection import online_predict

    model = (
        FollowTheWinner(strategy=strategy, learning_rate=0.1)
        if strategy in ("eg", "adagrad")
        else FollowTheLoser(strategy=strategy)
    )
    model.set_params(
        transaction_costs=cost,
        management_fees=fee,
        portfolio_params={"weight_drift": True},
    )
    X = np.random.default_rng(12).normal(0.001, 0.03, (20, 3))
    path = online_predict(model, X, warmup_size=4, test_size=1)
    prev = np.zeros(3)
    for p, row in zip(path, X[4:], strict=True):
        np.testing.assert_allclose(p.previous_weights, prev)
        expected = (
            p.weights @ row
            - cost * np.abs(p.weights - prev).sum()
            - fee * p.weights.sum()
        )
        np.testing.assert_allclose(p.returns[0], expected)
        prev = p.ending_weights


def test_turnover_uses_actual_drift_instead_of_old_target():
    from skfolio.model_selection import online_predict

    model = FollowTheLoser(
        strategy="pamr",
        max_turnover=0,
        previous_weights=[0.5, 0.5],
        portfolio_params={"weight_drift": True},
    )
    X = np.array([[0.1, -0.1], [0.2, -0.1], [-0.1, 0.2], [0.2, -0.1]])
    path = online_predict(model, X, warmup_size=1, test_size=1)
    np.testing.assert_allclose(path[1].weights, path[0].ending_weights)
    assert not np.allclose(path[1].weights, path[0].weights)
