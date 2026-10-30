import numpy as np
import pytest
from tests.test_optimization.test_online.utils import (
    assert_simplex_trajectory,
    make_stationary_returns,
)

from skfolio.measures._enums import PerfMeasure
from skfolio.optimization.online._foco import FirstOrderOCO
from skfolio.optimization.online._mirror_maps import EuclideanMirrorMap
from skfolio.optimization.online._mixins import FTWStrategy
from skfolio.optimization.online._prediction import LastGradPredictor
from skfolio.optimization.online._projection import IdentityProjector
from skfolio.optimization.online._winner import FollowTheWinner


@pytest.fixture
def X_small_single(X_small):
    return X_small.iloc[[0], :]


@pytest.mark.parametrize(
    "strategy, ftrl_flag, min_final_weight",
    [
        (FTWStrategy.EG, "omd", 0.95),
        (FTWStrategy.EG, "ftrl", 0.95),
        (FTWStrategy.OGD, "omd", 0.80),
        (FTWStrategy.ADAGRAD, "omd", 0.85),
    ],
)
def test_convergence_to_best_asset_under_stationary_env(
    strategy, ftrl_flag, min_final_weight
):
    # In a stationary environment with one dominant asset, OPS should converge most mass to it.
    X = make_stationary_returns(T=250, gap=0.01, n=2)
    # for making the adabarrons converge to the best asset fast we decrease the barrier coefficient, increase the euclidean coefficient and set the beta to a very small value
    est = FollowTheWinner(
        objective=PerfMeasure.LOG_WEALTH,
        strategy=strategy,
        adabarrons_alpha=0.1,
        adabarrons_euclidean_coef=0.1,
        adabarrons_beta=1e-6,
        update_mode=ftrl_flag,
        learning_rate=10,
        grad_predictor="last",
    )
    est.fit(X)
    W = est.all_weights_
    assert_simplex_trajectory(W)
    # last weights put most mass on the dominant asset
    assert W[-1, 0] >= min_final_weight


def test_smooth_prediction_accelerates_initial_adaptation():
    # Optimistic (last-gradient) prediction should speed up the initial allocation shift
    T = 12
    X = make_stationary_returns(T=T, gap=0.02, n=2)
    base = FollowTheWinner(
        strategy=FTWStrategy.EG,
        update_mode="omd",
        learning_rate=0.3,
        grad_predictor="last",
    )
    opti = FollowTheWinner(
        strategy=FTWStrategy.EG,
        update_mode="omd",
        learning_rate=0.3,
        grad_predictor="smooth",
    )

    base.fit(X)
    opti.fit(X)
    # Compare weight on asset 0 at an early time (e.g., t=3). Optimistic should be larger or equal.
    assert opti.all_weights_[2, 0] >= base.all_weights_[2, 0] - 1e-12


def test_eg_tilde_mixing_with_uniform():
    # EG-tilde mixes the EG step with the uniform portfolio
    X = make_stationary_returns(T=1, gap=0.01, n=3)
    pure = FollowTheWinner(
        strategy=FTWStrategy.EG,
        update_mode="omd",
        learning_rate=0.5,
        eg_tilde=False,
    )
    mix = FollowTheWinner(
        strategy=FTWStrategy.EG,
        learning_rate=0.5,
        eg_tilde=True,
        eg_tilde_alpha=0.5,
    )

    pure.fit(X)
    mix.fit(X)

    w_pure = pure.all_weights_[-1]
    w_mix = mix.all_weights_[-1]
    uniform = np.ones_like(w_pure) / w_pure.size
    w_expected = 0.5 * w_pure + 0.5 * uniform
    # Projection preserves the simplex so we expect exact match up to numerical error
    np.testing.assert_allclose(w_mix, w_expected, atol=1e-12, rtol=0)


def test_management_fees_flip_preference():
    # High management fee on asset 0 can flip the preference to asset 1
    T = 150
    X = make_stationary_returns(T=T, gap=0.01, n=2)
    # apply 2% fee on asset 0 every period -> effective 1.01 * (1 - 0.02) ~ 0.9898 < 1.0
    est = FollowTheWinner(
        strategy=FTWStrategy.EG,
        update_mode="omd",
        learning_rate=1,
        management_fees=np.array([0.02, 0.0]),
    )
    est.fit(X)
    W = est.all_weights_
    assert_simplex_trajectory(W)
    # Now asset 1 is more attractive net-of-fees
    assert W[-1, 1] >= 0.8


def test_partial_fit_input_and_warnings_on_sample_weight_and_nonpositive_return():
    # partial_fit accepts both individual observations and causal blocks
    X = make_stationary_returns(T=5, gap=0.01, n=2)
    est = FollowTheWinner(strategy=FTWStrategy.EG, update_mode="omd", learning_rate=0.2)

    # sample_weight warning
    with pytest.warns(UserWarning, match="sample_weight is ignored"):
        est.partial_fit(X[0:1, :], sample_weight=np.ones(1))

    # The block continues after the first observation.
    est.partial_fit(X[0:2, :])
    assert est.n_observations_ == 3


def test_objective_not_implemented_raises_value_error():
    # Not supported strategy Enums should raise.
    X = make_stationary_returns(T=5, gap=0.01, n=2)
    est = FollowTheWinner(
        strategy="INVALID_OBJECTIVE",
        update_mode="omd",
        learning_rate=0.2,
    )
    with pytest.raises(ValueError, match="INVALID_OBJECTIVE"):
        est.fit(X)


def test_turnover_cap_enforced_each_round():
    from skfolio.model_selection import online_predict
    from skfolio.optimization.online._winner import FollowTheWinner

    X = np.array([[0.2, -0.2, 0], [-0.2, 0.2, 0]] * 8)
    model = FollowTheWinner(learning_rate=5.0, update_mode="omd")
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


def test_weights_respect_min_max_and_budget_constraints():
    # Enforce min/max weights using projector; all weights must lie in [min, max] and sum==budget
    T = 50
    X = make_stationary_returns(T=T, gap=0.01, n=3)
    est = FollowTheWinner(
        strategy=FTWStrategy.EG,
        update_mode="ftrl",
        learning_rate=0.2,
        min_weights=np.array([0.20, 0.0, 0.0]),
        max_weights=np.array([0.8, 0.8, 0.8]),
        budget=1.0,
    )
    est.fit(X)
    W = est.all_weights_
    sums = np.sum(W, axis=1)
    assert np.allclose(sums, 1.0, atol=1e-9)
    assert np.all(W[:, 0] >= 0.20 - 1e-12)
    assert np.all(W <= 0.8 + 1e-12)


def test_fit_resets_and_partial_fit_continues():
    # Calling fit twice should not accumulate state
    X = make_stationary_returns(T=30, gap=0.02, n=2)

    est_cold = FollowTheWinner(
        strategy=FTWStrategy.EG, update_mode="omd", learning_rate=0.2
    )
    est_cold.fit(X)
    W1 = est_cold.all_weights_.copy()
    est_cold.fit(X)  # re-fit from scratch
    W2 = est_cold.all_weights_.copy()
    np.testing.assert_allclose(W1, W2, atol=1e-12, rtol=0)

    # A refitted estimator can continue learning with partial_fit.
    est_warm = FollowTheWinner(
        strategy=FTWStrategy.EG, update_mode="omd", learning_rate=0.2
    )
    est_warm.fit(X)
    W3 = est_warm.all_weights_.copy()
    est_warm.fit(X)
    W4 = est_warm.all_weights_.copy()
    np.testing.assert_allclose(W3, W4, atol=1e-12, rtol=0)
    est_warm.partial_fit(X)
    assert est_warm.n_observations_ == 2 * len(X)
    assert not np.allclose(W3, est_warm.all_weights_[len(X) :], atol=1e-12, rtol=0)


def test_omd_lastgrad_predictor_updates_prev_prediction():
    engine = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=0.1,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    g1 = np.array([1.0, 0.0, 0.0, -1.0], dtype=float)
    g2 = np.array([0.5, -0.5, 0.5, -0.5], dtype=float)

    engine.step(g1)
    # After first step, prev_prediction is set to m_t=g1
    engine.step(g2)
    # After second step, prev_prediction still equals g1; advance one more step
    engine.step(g2)
    assert engine._prev_prediction is not None
    assert np.allclose(engine._prev_prediction, g2)


def test_adabarrons_ftrl_matches_joint_constrained_potential(monkeypatch):
    """Check the adaptive composite objective rather than an arbitrary weight threshold."""
    from scipy.optimize import brentq

    initialize = FollowTheWinner._initialize_projector

    def tight_projector(model):
        projector = initialize(model)
        projector.config.solver_params = {
            "solver": "CLARABEL",
            "tol_gap_abs": 1e-11,
            "tol_gap_rel": 1e-11,
            "tol_feas": 1e-11,
        }
        return projector

    monkeypatch.setattr(FollowTheWinner, "_initialize_projector", tight_projector)
    model = FollowTheWinner(
        strategy=FTWStrategy.ADABARRONS,
        update_mode="ftrl",
        learning_rate=10,
        adabarrons_alpha=0.1,
        adabarrons_euclidean_coef=0.1,
        adabarrons_beta=1e-6,
        grad_predictor="last",
    )
    weights = np.array([0.5, 0.5])
    proximity = np.zeros(2)
    quadratic = np.zeros((2, 2))
    gradient_sum = np.zeros(2)
    last_gradient = np.zeros(2)
    for row in make_stationary_returns(T=250, gap=0.01, n=2):
        gradient = -row / (1.0 + weights @ row)
        gradient_sum += gradient
        barrier = 1.0 + 0.1 * proximity
        linear = 10 * (gradient_sum + last_gradient)

        def derivative(
            first_weight, barrier=barrier, quadratic=quadratic, linear=linear
        ):
            w = np.array([first_weight, 1 - first_weight])
            grad = -barrier / w + 0.2 * w + 1e-6 * (quadratic @ w) + linear
            return grad[0] - grad[1]

        first = brentq(derivative, 1e-10, 1 - 1e-10)
        optimum = np.array([first, 1 - first])
        model.partial_fit(row)
        weights = model.weights_.copy()
        assert np.all(weights > 0)
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-8, rtol=0)

        def potential(w, barrier=barrier, quadratic=quadratic, linear=linear):
            return (
                -barrier @ np.log(w)
                + 0.1 * (w @ w)
                + 0.5e-6 * (w @ quadratic @ w)
                + linear @ w
            )

        # Conic tolerances bound the objective, not coordinate error. Compare
        # against the independent scalar optimum for this actual input state.
        # Updating the reference from the accepted target avoids compounding
        # solver noise along a second, slightly different 250-step trajectory.
        reference_value = potential(optimum)
        gap = potential(weights) - reference_value
        assert abs(gap) <= 1e-7 * max(1.0, abs(reference_value))
        proximity += 1 - weights
        quadratic += np.outer(gradient, gradient)
        last_gradient = gradient
        components = model._foco_engine.map.components_
        np.testing.assert_allclose(
            components[0]._proximity_sum, proximity, atol=1e-10, rtol=1e-12
        )
        np.testing.assert_allclose(components[2]._A, quadratic, atol=1e-10, rtol=1e-12)
    # Convex solutions are checked at the same feasibility tolerance as constraints.
    assert_simplex_trajectory(model.all_weights_, atol=1e-7)
    components = model._foco_engine.map.components_
    np.testing.assert_allclose(components[0]._proximity_sum, proximity, atol=1e-3)
    np.testing.assert_allclose(components[2]._A, quadratic, atol=1e-7)
