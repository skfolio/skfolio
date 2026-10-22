# tests/test_ftloser_olmar.py
import numpy as np

from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._prediction import OLMAR2Predictor
from skfolio.optimization.online._projection import project_box_and_sum
from skfolio.optimization.online._utils import CLIP_EPSILON


def test_olmar_ewma_predictor_phi_recursion():
    alpha = 0.6
    pred = OLMAR2Predictor(alpha=alpha)
    pred.reset(d=2)
    x1 = np.array([1.1, 0.9], dtype=float)
    x2 = np.array([0.9, 1.3], dtype=float)
    phi1 = pred.update_and_predict(x1)
    phi2 = pred.update_and_predict(x2)

    phi_1_ref = alpha * np.ones(2) + (1 - alpha) * (
        np.ones(2) / np.maximum(x1, CLIP_EPSILON)
    )
    phi_2_ref = alpha * np.ones(2) + (1 - alpha) * (
        phi_1_ref / np.maximum(x2, CLIP_EPSILON)
    )
    np.testing.assert_allclose(phi1, phi_1_ref, atol=1e-12)
    np.testing.assert_allclose(phi2, phi_2_ref, atol=1e-12)


def test_pa_step_equals_closed_form_projection():
    model = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=2,
        epsilon=1.05,
        update_mode="pa",
    )
    X = np.array(
        [
            [0.00, 0.00],
            [0.02, -0.03],
            [-0.01, 0.04],
        ]
    )
    model.fit(X[:2])
    w_prev = model.weights_.copy()
    model.partial_fit(X[2:3])
    w_new = model.weights_.copy()

    # Two prices include the current price, so only the latest relative is used.
    phi = (1.0 + 1.0 / (1.0 + X[-1])) / 2.0

    margin = float(phi @ w_prev)
    ell = max(0.0, 1.05 - margin)
    c = phi - phi.mean()
    if (c @ c) > 0:
        lam = ell / float(c @ c)
        w_ref = project_box_and_sum(w_prev + lam * c, lower=0.0, upper=1.0, budget=1.0)
        np.testing.assert_allclose(w_new, w_ref, atol=1e-8)
    else:
        np.testing.assert_allclose(w_new, w_prev, atol=1e-12)


def test_turnover_constraint_enforced():
    from skfolio.model_selection import online_predict
    from skfolio.optimization.online._loser import FollowTheLoser

    X = np.array([[0.2, -0.2, 0], [-0.2, 0.2, 0]] * 8)
    model = FollowTheLoser(strategy="olmar", olmar_window=2)
    model.set_params(
        previous_weights=np.ones(3) / 3,
        max_turnover=0.1,
        portfolio_params={"weight_drift": True},
    )
    path = online_predict(model, X, warmup_size=3, test_size=1)
    for portfolio in path:
        assert (
            np.abs(portfolio.weights - portfolio.previous_weights).sum() <= 0.1 + 1e-7
        )


def test_trading_vs_postupdate_weights_semantics():
    X_net = np.array([[0.0, 0.0], [0.1, -0.1]], dtype=float)
    model = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=2,
        epsilon=2.0,
        update_mode="pa",
    )
    model.fit(X_net)
    W = model.all_weights_
    np.testing.assert_allclose(W[0], np.array([0.5, 0.5]), atol=1e-12)
    np.testing.assert_allclose(W[1].sum(), 1.0, atol=1e-12)


def test_olmar1_day3_is_non_uniform():
    # Use non-flat data so day 3 can be non-uniform
    X = np.array([[0.0, 0.0], [0.1, -0.1], [0.0, 0.0]], dtype=float)
    model = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=2,
        epsilon=2.0,
        update_mode="pa",
    )
    model.fit(X)
    W = model.all_weights_
    np.testing.assert_allclose(W[0], np.array([0.5, 0.5]), atol=1e-12)
    np.testing.assert_allclose(W[1], np.array([0.5, 0.5]), atol=1e-12)
    assert not np.allclose(W[2], np.array([0.5, 0.5]), atol=1e-12)


def test_day3_uniform_when_history_flat():
    X = np.array([[0.0, 0.0], [0.0, 0.0], [0.1, -0.1]], dtype=float)
    model = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=2,
        epsilon=2.0,
        update_mode="pa",
    )
    model.fit(X)
    W = model.all_weights_
    np.testing.assert_allclose(W[0], np.array([0.5, 0.5]), atol=1e-12)
    np.testing.assert_allclose(W[1], np.array([0.5, 0.5]), atol=1e-12)
    np.testing.assert_allclose(W[2], np.array([0.5, 0.5]), atol=1e-12)


def test_day3_nonuniform_with_nonflat_day2():
    X = np.array([[0.0, 0.0], [0.2, -0.2], [0.1, -0.1]], dtype=float)
    model = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=2,
        epsilon=2.0,
        update_mode="pa",
    )
    model.fit(X)
    W = model.all_weights_
    assert not np.allclose(W[2], np.array([0.5, 0.5]), atol=1e-12)


def test_md_zero_move_when_margin_satisfied():
    X = np.array([[0.0, 0.0]], dtype=float)
    model_pa = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=1,
        epsilon=0.5,
        update_mode="pa",
    )
    model_md = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=1,
        epsilon=0.5,
        update_mode="md",
        mirror="euclidean",
        learning_rate=0.5,
    )
    model_pa.fit(X)
    model_md.fit(X)
    np.testing.assert_allclose(model_pa.all_weights_, model_md.all_weights_, atol=1e-12)
