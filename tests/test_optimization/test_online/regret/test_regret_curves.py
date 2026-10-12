import numpy as np
import pytest

from skfolio.optimization.online import RegretType, regret
from skfolio.optimization.online._mixins import FTWStrategy
from skfolio.optimization.online._regret import (
    _solve_best_constant_weights,
)
from skfolio.optimization.online._utils import CLIP_EPSILON, net_to_relatives
from skfolio.optimization.online._winner import FollowTheWinner


def _manual_losses_from_weights(
    relatives: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Vectorized per-period negative log-wealth: -log(max(w_t^T x_t, eps))
    relatives: (T, n), weights: (T, n) or (n,)
    """
    R = np.asarray(relatives, dtype=float)
    if weights.ndim == 1:
        dots = R @ weights
    else:
        dots = np.sum(weights * R, axis=1)
    dots = np.maximum(dots, CLIP_EPSILON)
    return -np.log(dots)


def _iid_relatives(T: int, n: int, seed: int = 0, sigma: float = 0.02) -> np.ndarray:
    """Generate i.i.d. gross relatives with no trend."""
    rng = np.random.default_rng(seed)
    # lognormal with small volatility around 1.0
    relatives = rng.lognormal(mean=0.0, sigma=sigma, size=(T, n))
    return relatives


@pytest.mark.parametrize(
    "objective", [FTWStrategy.EG, FTWStrategy.OGD, FTWStrategy.ADAGRAD]
)
def test_static_regret_matches_manual(objective):
    """
    Verify that regret(estimator, X, STATIC, average=False) equals the manual cumulative difference
    between online losses and BestConstantRebalancedPortfolio losses. Also check final static regret is >= -tol.
    """
    T, n = 60, 6
    relatives = _iid_relatives(T=T, n=n, seed=1)
    X_net = relatives - 1.0

    # Online estimator
    est_for_regret = FollowTheWinner(strategy=objective, learning_rate=0.1)

    # Compute regret curve via public API
    r_curve = regret(
        estimator=est_for_regret, X=X_net, regret_type=RegretType.STATIC, average=False
    )
    assert r_curve.shape == (T,)
    assert np.all(np.isfinite(r_curve))

    # Manual recomputation using fresh estimators
    est_manual = FollowTheWinner(strategy=objective, learning_rate=0.1)
    est_manual.fit_predict(X_net)
    W_online = est_manual.all_weights_
    R = net_to_relatives(X_net)
    online_losses = _manual_losses_from_weights(R, W_online)

    w_star = _solve_best_constant_weights(R)
    comp_losses = _manual_losses_from_weights(R, w_star)

    manual_curve = np.cumsum(online_losses - comp_losses)
    # Should match the library result closely
    assert np.allclose(r_curve, manual_curve, atol=1e-8, rtol=0)

    # Final static regret should be non-negative up to tiny numerical error
    assert r_curve[-1] >= -1e-6


def test_regret_average_modes_consistency_static():
    """
    Check that average='final' returns a constant array equal to the last entry of
    running-average regret; and that average=True equals 'running'.
    """
    T, n = 80, 5
    relatives = _iid_relatives(T=T, n=n, seed=2)
    X_net = relatives - 1.0

    est = FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.1)

    r_running = regret(
        estimator=est, X=X_net, regret_type=RegretType.STATIC, average=True
    )
    r_running_str = regret(
        estimator=FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.1),
        X=X_net,
        regret_type=RegretType.STATIC,
        average="running",
    )
    r_final = regret(
        estimator=FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.1),
        X=X_net,
        regret_type=RegretType.STATIC,
        average="final",
    )

    # average=True and 'running' are the same
    assert np.allclose(r_running, r_running_str, atol=1e-12)
    # 'final' is constant, equal to the last value of running average
    assert np.allclose(r_final, np.full(T, r_running[-1]), atol=1e-12)


def test_windowed_regret_consistency():
    """
    Verify sliding-window regret and running average agree with a manual computation.
    """
    T, n = 100, 7
    relatives = _iid_relatives(T=T, n=n, seed=3)
    X_net = relatives - 1.0
    window = 10

    est = FollowTheWinner(strategy=FTWStrategy.OGD, learning_rate=0.1)
    # Library curve
    r_win = regret(
        estimator=est,
        X=X_net,
        regret_type=RegretType.STATIC,
        average=True,
        window=window,
    )

    # Manual computation
    est_m = FollowTheWinner(strategy=FTWStrategy.OGD, learning_rate=0.1)
    est_m.fit_predict(X_net)
    W_online = est_m.all_weights_
    R = net_to_relatives(X_net)
    online_losses = _manual_losses_from_weights(R, W_online)

    w_star = _solve_best_constant_weights(R)
    comp_losses = _manual_losses_from_weights(R, w_star)

    # Manual windowed regret averaged by window size
    rw = np.zeros(T)
    for t in range(window - 1, T):
        diff = np.sum(online_losses[t - window + 1 : t + 1]) - np.sum(
            comp_losses[t - window + 1 : t + 1]
        )
        rw[t] = diff / float(window)
    assert np.allclose(r_win, rw, atol=1e-10)


@pytest.mark.parametrize(
    "objective",
    [
        FTWStrategy.EG,
        FTWStrategy.OGD,
        FTWStrategy.ADAGRAD,
        FTWStrategy.SWORD_VAR,
        FTWStrategy.SWORD_SMALL,
        FTWStrategy.SWORD_BEST,
        FTWStrategy.SWORD_PP,
    ],
)
def test_random_relatives_static_and_dynamic_regret_curves(objective):
    """
    Compare regret curves (static and 'dynamic' as currently implemented) for i.i.d. price relatives.
    We check shapes, finiteness, and non-negativity of final static regret (up to tiny tolerance).
    """
    T, n = 120, 8
    relatives = _iid_relatives(T=T, n=n, seed=4)
    X_net = relatives - 1.0

    model = FollowTheWinner(strategy=objective, learning_rate=0.1)

    rs = regret(estimator=model, X=X_net, regret_type=RegretType.STATIC, average=False)
    rd = regret(
        estimator=FollowTheWinner(strategy=objective, learning_rate=0.1),
        X=X_net,
        regret_type=RegretType.DYNAMIC_UNIVERSAL,
        average=False,
        dynamic_config={"path_length": 5.0},
    )

    assert rs.shape == (T,)
    assert rd.shape == (T,)
    assert np.all(np.isfinite(rs))
    assert np.all(np.isfinite(rd))

    # Final static regret should be non-negative up to tiny numerical slack
    assert rs[-1] >= -1e-6


def test_worst_case_regret_dominates_fixed_comparator():
    """
    The per-period best asset dominates the best fixed portfolio at each prefix.
    """
    T, n = 50, 6
    relatives = _iid_relatives(T=T, n=n, seed=5)
    X_net = relatives - 1.0

    # Online EG
    est = FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.05)
    est.fit_predict(X_net)
    W_online = est.all_weights_
    R = net_to_relatives(X_net)
    online_losses = _manual_losses_from_weights(R, W_online)

    comp_weights = _solve_best_constant_weights(R)
    comp_losses = _manual_losses_from_weights(R, comp_weights)

    manual_dr = np.cumsum(online_losses - comp_losses)

    # Library computation for worst-case dynamic regret
    rd_wc = regret(
        estimator=FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.05),
        X=X_net,
        regret_type=RegretType.DYNAMIC_WORST_CASE,
        average=False,
    )

    # Worst-case comparator should dominate fixed-comparator regret (manual_dr)
    assert rd_wc.shape == manual_dr.shape
    assert np.all(rd_wc >= manual_dr - 1e-8)
