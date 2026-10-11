import numpy as np
import pytest

from skfolio.optimization.online import RegretType, regret
from skfolio.optimization.online._mixins import FTWStrategy
from skfolio.optimization.online._winner import FollowTheWinner


def _iid_relatives(T, n, seed=42, sigma=0.02):
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=0.0, sigma=sigma, size=(T, n))


def test_worst_case_dynamic_nonnegativity():
    T, n = 60, 5
    R = _iid_relatives(T, n, seed=1)
    X = R - 1.0

    est = FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.1, warm_start=False)
    rd = regret(
        estimator=est, X=X, regret_type=RegretType.DYNAMIC_WORST_CASE, average=False
    )

    # Worst-case dynamic regret is per-step nonnegative; cumulative is nonnegative
    assert np.all(rd >= -1e-12)


def test_universal_dynamic_pt_zero_equals_static():
    T, n = 50, 6
    R = _iid_relatives(T, n, seed=2)
    X = R - 1.0

    est = FollowTheWinner(strategy=FTWStrategy.OGD, learning_rate=0.1, warm_start=False)

    r_static = regret(estimator=est, X=X, regret_type=RegretType.STATIC, average=False)
    r_univ_pt0 = regret(
        estimator=FollowTheWinner(
            strategy=FTWStrategy.OGD, learning_rate=0.1, warm_start=False
        ),
        X=X,
        regret_type=RegretType.DYNAMIC_UNIVERSAL,
        dynamic_config={"path_length": 0.0},  # reduces to static comparator
        average=False,
    )
    assert np.allclose(r_univ_pt0, r_static, atol=1e-8)


def test_universal_dynamic_monotonic_in_pt():
    """
    As path_length increases, the universal comparator's feasible set enlarges
    => comparator loss decreases
    => dynamic regret (online - comparator) is non-decreasing in PT.
    """
    T, n = 80, 4
    R = _iid_relatives(T, n, seed=3)
    X = R - 1.0
    est = FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.05, warm_start=False)

    PTs = [
        0.0,
        1.0,
        5.0,
        2.0 * (T - 1),
    ]  # L1: 2 per step suffices to be effectively unconstrained
    final_vals = []
    for PT in PTs:
        r = regret(
            estimator=est,
            X=X,
            regret_type=RegretType.DYNAMIC_UNIVERSAL,
            dynamic_config={"path_length": PT, "norm": "l1"},
        )
        final_vals.append(r[-1])

    # Non-decreasing sequence in PT (allow small numerical slack)
    assert all(
        final_vals[i] <= final_vals[i + 1] + 1e-6 for i in range(len(final_vals) - 1)
    )

    # Compare with worst-case dynamic regret
    r_wc = regret(
        estimator=FollowTheWinner(
            strategy=FTWStrategy.EG, learning_rate=0.05, warm_start=False
        ),
        X=X,
        regret_type=RegretType.DYNAMIC_WORST_CASE,
    )

    # With PT large enough, universal dynamic regret should match worst-case (within tolerance)
    assert abs(final_vals[-1] - r_wc[-1]) <= 1e-5


def test_dynamic_legacy_alias_warns_and_matches():
    T, n = 40, 3
    R = _iid_relatives(T, n, seed=4)
    X = R - 1.0
    est = FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.1, warm_start=False)

    with pytest.warns(UserWarning):
        r_dyn = regret(estimator=est, X=X, regret_type=RegretType.DYNAMIC)

    r_leg = regret(
        estimator=FollowTheWinner(
            strategy=FTWStrategy.EG, learning_rate=0.1, warm_start=False
        ),
        X=X,
        regret_type=RegretType.DYNAMIC_LEGACY,
    )
    assert np.allclose(r_dyn, r_leg, atol=1e-12)
