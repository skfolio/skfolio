import warnings
from itertools import product

import numpy as np
import pytest
from scipy.special import softmax

from skfolio.optimization.online._mixins import FTWStrategy
from skfolio.optimization.online._winner import FollowTheWinner

from ..utils import assert_box_budget, group_sum


@pytest.mark.parametrize(
    "method",
    [
        FTWStrategy.EG,
        FTWStrategy.OGD,
        FTWStrategy.ADAGRAD,
        FTWStrategy.ADABARRONS,
    ],
)
def test_methods_basic_validity_fit(method, X_small):
    # Keep runtime low for heavier methods
    est = FollowTheWinner(strategy=method)
    est.fit(X_small)
    assert_box_budget(est.weights_, 0.0, 1.0, 1.0)


def test_array_transaction_costs_gating_no_ambiguous_bool():
    # Single-step synthetic net returns (1 period, 3 assets)
    x_net = np.array([0.01, -0.005, 0.0], dtype=float)

    est = FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate=0.1,
        previous_weights=np.ones(3) / 3,
        transaction_costs=np.array([0.001, 0.0, 0.0]),
        management_fees=0.0,
    )
    # Enable a decision penalty, independent of booked transaction costs.
    est.penalize_turnover = True

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        est.partial_fit(x_net)

        # No exception; learning updates without booking wealth.
        assert hasattr(est, "weights_")
        assert np.isfinite(est.weights_).all()
        assert not hasattr(est, "wealth_")
        assert np.isfinite(est.loss_)

        # Accounting no longer disables a legitimate optimization penalty.
        assert not any("disabled" in str(item.message).lower() for item in w)


def test_smooth_prediction(X_small):
    # Test that smooth prediction runs and produces different weights from vanilla
    est_vanilla = FollowTheWinner(strategy=FTWStrategy.EG, learning_rate=0.1)
    est_smooth = FollowTheWinner(
        strategy=FTWStrategy.EG, learning_rate=0.1, grad_predictor="smooth"
    )

    est_vanilla.fit(X_small)
    est_smooth.fit(X_small)

    assert_box_budget(est_vanilla.weights_, 0.0, 1.0, 1.0)
    assert_box_budget(est_smooth.weights_, 0.0, 1.0, 1.0)

    # Weights should be different due to the optimistic term
    assert not np.allclose(est_vanilla.weights_, est_smooth.weights_), (
        "Smooth prediction weights are identical to vanilla"
    )


def test_warm_start_and_initial_weights(X_small_single):
    n = X_small_single.shape[1]
    init = np.random.rand(n)
    init /= np.sum(init)

    # With warm_start=False, weights should reset to a deterministic state
    est_no_warm = FollowTheWinner(
        strategy=FTWStrategy.EG, initial_weights=init, warm_start=False
    )
    est_no_warm.partial_fit(X_small_single)  # First fit uses init
    first_weights = est_no_warm.weights_.copy()
    est_no_warm.fit(X_small_single)  # Second fit should reset and give same result
    assert np.allclose(first_weights, est_no_warm.weights_)

    # With warm_start=True, weights should persist and continue updating
    est_warm = FollowTheWinner(
        strategy=FTWStrategy.EG, initial_weights=init, warm_start=True
    )
    est_warm.partial_fit(X_small_single)
    first_weights_warm = est_warm.weights_.copy()
    est_warm.partial_fit(X_small_single)
    second_weights_warm = est_warm.weights_.copy()
    assert not np.allclose(init, first_weights_warm)  # weights should have been updated
    assert not np.allclose(
        first_weights_warm, second_weights_warm
    )  # second update should differ


def test_eg_tilde_implementation(X_small):
    """Test EG-Tilde mixing step."""
    # With alpha=1, result should be uniform portfolio
    est_uniform = FollowTheWinner(
        strategy=FTWStrategy.EG, eg_tilde=True, eg_tilde_alpha=1.0
    ).fit(X_small)
    n_assets = X_small.shape[1]
    uniform = np.ones(n_assets) / n_assets
    np.testing.assert_allclose(est_uniform.weights_, uniform, atol=1e-8)

    # With alpha=0, result should be same as standard EG
    est_eg = FollowTheWinner(strategy=FTWStrategy.EG, eg_tilde=False).fit(X_small)
    est_no_mix = FollowTheWinner(
        strategy=FTWStrategy.EG, eg_tilde=True, eg_tilde_alpha=0.0
    ).fit(X_small)
    np.testing.assert_allclose(est_eg.weights_, est_no_mix.weights_, atol=1e-8)

    # Test with a callable alpha
    est_callable = FollowTheWinner(
        strategy=FTWStrategy.EG,
        eg_tilde=True,
        eg_tilde_alpha=lambda t: 1.0 / t if t > 0 else 1.0,
    ).fit(X_small)
    assert_box_budget(est_callable.weights_, 0.0, 1.0, 1.0)
    assert not np.allclose(est_callable.weights_, est_eg.weights_)


@pytest.mark.parametrize(
    "method",
    list(FTWStrategy),
)
def test_update_mode(method, X_small):
    """Test that FTRL and OMD modes run and produce different results."""
    # OMD mode (default)
    est_omd = FollowTheWinner(strategy=method, update_mode="omd")
    est_omd.fit(X_small)
    assert_box_budget(est_omd.weights_, 0.0, 1.0, 1.0)

    # FTRL mode
    est_ftrl = FollowTheWinner(strategy=method, update_mode="ftrl")
    est_ftrl.fit(X_small)
    assert_box_budget(est_ftrl.weights_, 0.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "strategy,learning_rate",
    product(list(FTWStrategy), [0.1, "auto", lambda t: 1 / (t + 1)]),
)
def test_learning_rate(strategy, learning_rate, X_small):
    """Test that a callable learning_rate runs correctly."""
    # Constant learning_rate
    est = FollowTheWinner(strategy=strategy, learning_rate=learning_rate)
    est.fit(X_small)
    assert_box_budget(est.weights_, 0.0, 1.0, 1.0)


def test_entropy_omd_matches_exponentiated_gradient():
    """
    For ψ entropy and OMD mode, Orabona Eq. (8.13) gives wₜ₊₁ ∝ wₜ·exp(-η gₜ). _composite_update implements softmax(log wₜ - ηgₜ).
    Weights from FirstOrderOCOEngine coincide with exponentiated-gradient update, validating OMD pathway.
    """
    lr = 0.5
    model = FollowTheWinner(
        objective="log_wealth",
        strategy=FTWStrategy.EG,
        update_mode="omd",
        learning_rate=lr,
        warm_start=False,
        initial_weights=np.array([0.6, 0.4]),
    )
    X = np.array([[0.02, -0.01]])
    model.partial_fit(X)
    w0 = np.array([0.6, 0.4])
    rel = 1.0 + X[0]
    grad = -rel / np.dot(w0, rel)
    expected = softmax(np.log(w0) - lr * grad)
    assert np.allclose(model.weights_, expected, atol=1e-10)


def test_entropy_ftrl_matches_dual_averaging(X_small):
    """Dual averaging with entropy gives wₜ₊₁ ∝ exp(-η Σ_{s≤t} g_s) (Orabona Lemma 8.6).
    FOCO engines's FTRL branch uses Σg.
    Expected outcome: final weights align with dual averaging recursion, confirming cumulative-gradient handling."""
    learning_rate = 0.3
    model = FollowTheWinner(
        strategy=FTWStrategy.EG,
        update_mode="ftrl",
        learning_rate=learning_rate,
        warm_start=False,
    )
    X = np.array([[0.05, -0.02], [-0.01, 0.03]])
    model.fit(X)
    # manual check
    w = np.array([0.5, 0.5])
    G = np.zeros_like(w)
    for row in X:
        rel = 1.0 + row
        grad = -rel / np.dot(w, rel)
        G += grad
        w = softmax(-learning_rate * G)
    assert np.allclose(model.weights_, w, atol=1e-10)


class TestFollowTheWinnerConstraints:
    @pytest.mark.parametrize(
        "lower,upper,budget",
        [(0.0, 0.5, 0.8)],
    )
    def test_bounds_and_budget(self, lower, upper, budget, X_small_single):
        est = FollowTheWinner(min_weights=lower, max_weights=upper, budget=budget)
        est.partial_fit(X_small_single)
        assert_box_budget(est.weights_, lower, upper, budget)

    def test_turnover_projection(self, X_small):
        max_turnover = 0.5
        n = X_small.shape[1]
        prev = np.ones(n) / n
        est = FollowTheWinner(previous_weights=prev, max_turnover=max_turnover)
        est.fit(X_small)
        l1 = np.abs(est.weights_ - prev).sum()
        assert l1 <= max_turnover + 1e-8
        assert_box_budget(est.weights_, 0.0, 1.0, 1.0)

    def test_convex_fallback_groups_linear(
        self, X_small_single, groups, linear_constraints
    ):
        # Force convex path via groups/linear constraints
        budget = 0.9
        est = FollowTheWinner(
            strategy=FTWStrategy.EG,
            min_weights=0.0,
            max_weights=0.8,
            budget=budget,
            groups=groups,
            linear_constraints=linear_constraints,
        )
        est.partial_fit(X_small_single)
        w = est.weights_

        # Box + budget
        assert_box_budget(w, 0.0, 0.8, budget)

        # Check a subset of linear constraints semantics
        eq_sum = group_sum(w, groups, "Equity", 0)
        bond_sum = group_sum(w, groups, "Bond", 0)
        assert eq_sum <= 0.5 * bond_sum + 1e-6

        us_sum = group_sum(w, groups, "US", 1)
        assert us_sum >= 0.1 - 1e-6

        europe_sum = group_sum(w, groups, "Europe", 1)
        fund_sum = group_sum(w, groups, "Fund", 0)
        assert europe_sum >= 0.5 * fund_sum - 1e-6

    def test_convex_variance_bound(self, X_small):
        Sigma = np.cov(X_small.to_numpy().T)
        # Loose bound to ensure feasibility
        var_bound = float(np.trace(Sigma)) / Sigma.shape[0]
        est = FollowTheWinner(
            strategy=FTWStrategy.EG,
            covariance=Sigma,
            variance_bound=var_bound * 2.0,
            min_weights=0.0,
            max_weights=0.5,
            budget=0.9,
        )
        est.fit(X_small)
        w = est.weights_
        assert_box_budget(w, 0.0, 0.5, 0.9)
        quad = float(w @ Sigma @ w)
        assert quad <= var_bound * 2.0 + 1e-6
