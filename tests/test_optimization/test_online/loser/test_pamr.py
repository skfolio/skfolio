# tests/test_optimization/test_online/test_ftloser_pamr.py
import numpy as np
import pytest

from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._mixins import FTLStrategy
from skfolio.optimization.online._projection import project_box_and_sum


def test_pamr_pa_closed_form_projection():
    # Uniform previous weight
    w_prev = np.array([0.5, 0.5], dtype=float)
    X = np.array([[0.00, 0.00], [0.2, -0.1]], dtype=float)  # second day triggers update
    # Build model but we will do a single PA step by hand for comparison
    model = FollowTheLoser(
        strategy=FTLStrategy.PAMR,
        epsilon=1.0,
        pamr_C=500,
        update_mode="pa",
    )
    model.fit(X[:1])  # day-1: uniform
    model.partial_fit(X[1:2])  # update with day-2
    w_new = model.weights_.copy()

    x_t = 1.0 + X[1]
    margin = float(w_prev @ x_t)
    ell = max(0.0, margin - 1.0)
    c = x_t - x_t.mean()
    denom = float(np.dot(c, c))
    if denom > 0.0 and ell > 0.0:
        tau = ell / denom
        w_ref = project_box_and_sum(w_prev - tau * c, lower=0.0, upper=1.0, budget=1.0)
    else:
        w_ref = w_prev.copy()

    np.testing.assert_allclose(w_new, w_ref, atol=1e-8)


def test_pamr_md_gradient_direction_alignment():
    # Show that MD gradient points along +x when margin>ε; centered -> direction ~ -(x-mean x)
    X = np.array([[0.0, 0.0], [0.3, -0.2]], dtype=float)
    base = FollowTheLoser(
        strategy="pamr",
        epsilon=1.0,
        update_mode="md",
        mirror="euclidean",
        learning_rate=0.5,
    )
    base.fit(X[:1])  # day-1 uniform
    w_prev = base.weights_.copy()
    base.partial_fit(X[1:2])
    w_new = base.weights_.copy()
    x_t = 1.0 + X[1]
    # PA direction for PAMR is - (x - mean(x))
    direction_pa = -(x_t - x_t.mean())
    update_dir = w_new - w_prev
    # Cosine between update_dir and PA direction should be positive
    num = float(np.dot(update_dir, direction_pa))
    den = float(np.linalg.norm(update_dir) * np.linalg.norm(direction_pa) + 1e-16)
    assert num / (den + 1e-16) >= 0.5  # fairly aligned


def test_pamr_param_validation_raises_on_negative_epsilon():
    # epsilon must be >= 0 by parameter constraints
    model = FollowTheLoser(strategy="pamr", epsilon=-1.0, update_mode="pa")
    X = np.array([[0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError):
        # validation should occur during fit
        model.fit(X)
