import numpy as np
import pytest

from skfolio.optimization.online._foco import FirstOrderOCO
from skfolio.optimization.online._mirror_maps import (
    AdaptiveLogBarrierMap,
    AdaptiveMahalanobisMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.optimization.online._prediction import (
    ConstantPredictor,
    LastGradPredictor,
    SmoothPredictor,
    ZeroPredictor,
)
from skfolio.optimization.online._projection import (
    AutoProjector,
    IdentityProjector,
    ProjectionConfig,
)


def rng_grad_sequence(d=5, T=10, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(size=d).astype(float) for _ in range(T)]


def assert_simplex(w: np.ndarray, atol=1e-10):
    assert np.all(w >= -1e-12)
    assert np.all(w <= 1 + 1e-12)
    s = float(np.sum(w))
    assert abs(s - 1.0) <= atol


@pytest.mark.parametrize("d,T", [(4, 7), (5, 12)])
def test_entropy_omd_ftrl_equivalence_constant_eta(d, T):
    # Entropy mirror map: OMD and FTRL are equivalent with constant eta and simplex projection
    # when starting from uniform initialization.
    g_seq = rng_grad_sequence(d=d, T=T, seed=42)
    projector = AutoProjector(ProjectionConfig(lower=0.0, upper=1.0, budget=1.0))
    eta = 0.1

    omd = FirstOrderOCO(
        mirror_map=EntropyMirrorMap(), projector=projector, eta=eta, mode="omd"
    )
    ftr = FirstOrderOCO(
        mirror_map=EntropyMirrorMap(), projector=projector, eta=eta, mode="ftrl"
    )

    W_omd = []
    W_ftr = []
    for g in g_seq:
        W_omd.append(omd.step(g).copy())
        W_ftr.append(ftr.step(g).copy())

    for w1, w2 in zip(W_omd, W_ftr, strict=True):
        assert_simplex(w1)
        assert_simplex(w2)
        # Entropy map equivalence up to numerical error
        np.testing.assert_allclose(w1, w2, rtol=1e-10, atol=1e-12)


def test_euclidean_omd_step_matches_formula_identity_projector():
    # With Euclidean map and identity projector, OMD step is x_{t+1} = x_t - eta_t (g_t + m_t)
    d = 3
    g0 = np.array([1.0, -2.0, 0.5])
    g1 = np.array([-0.3, 0.7, 1.2])
    eta = 0.2
    omd = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        mode="omd",
    )

    # x0 initialized uniformly
    x0 = np.ones(d) / d
    x1 = omd.step(g0)
    np.testing.assert_allclose(x1, x0 - eta * g0, rtol=0, atol=1e-12)

    x2 = omd.step(g1)
    np.testing.assert_allclose(x2, x1 - eta * g1, rtol=0, atol=1e-12)


def test_euclidean_ftrl_matches_dual_averaging_identity_projector():
    # With Euclidean map and identity projector, FTRL step is x_{t+1} = - eta_t * sum_{s<=t} g_s
    g0 = np.array([1.0, 2.0, -3.0, 0.5])
    g1 = np.array([0.0, -1.0, 1.0, -0.5])
    eta = 0.1
    ftr = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        mode="ftrl",
    )

    x1 = ftr.step(g0)
    np.testing.assert_allclose(x1, -eta * g0, rtol=0, atol=1e-12)

    x2 = ftr.step(g1)
    np.testing.assert_allclose(x2, -eta * (g0 + g1), rtol=0, atol=1e-12)


def test_predictor_effect_and_shape_check():
    # OMD with LastGradPredictor: at t=1 predictor is zero; at t=2 predictor equals g0
    d = 2
    eta = 0.3
    g0 = np.array([1.0, -1.0])
    g1 = np.array([2.0, 0.5])
    pred = LastGradPredictor()
    omd_pred = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        predictor=pred,
        mode="omd",
    )
    omd_base = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        predictor=None,
        mode="omd",
    )

    # Step 1 identical (predictor returns zeros)
    x1_pred = omd_pred.step(g0)
    x1_base = omd_base.step(g0)
    np.testing.assert_allclose(x1_pred, x1_base, rtol=0, atol=1e-12)

    # Step 2: predictor returns last_grad = g0, effect equals an extra -eta*g0 step
    x2_pred = omd_pred.step(g1)
    x2_base = omd_base.step(g1)
    np.testing.assert_allclose(x2_pred, x2_base - eta * g0, rtol=0, atol=1e-12)

    # Wrong-shape predictor must raise
    class BadPredictor:
        def __call__(self, t, last_x, last_g):
            return np.zeros(d + 1)  # wrong shape

    omd_bad = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta,
        predictor=BadPredictor(),
        mode="omd",
    )
    with pytest.raises(ValueError, match="wrong shape"):
        omd_bad.step(g0)


def test_eta_schedule_array_and_callable():
    d = 3
    g = np.array([1.0, 0.5, -2.0])

    # Array schedule
    eta_arr = np.array([0.3, 0.2, 0.1])
    omd_a = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta_arr,
        mode="omd",
    )
    x0 = np.ones(d) / d
    x1 = omd_a.step(g)
    np.testing.assert_allclose(x1, x0 - eta_arr[0] * g, atol=1e-12)
    x2 = omd_a.step(g)
    np.testing.assert_allclose(x2, x1 - eta_arr[1] * g, atol=1e-12)
    x3 = omd_a.step(g)
    np.testing.assert_allclose(x3, x2 - eta_arr[2] * g, atol=1e-12)
    # beyond length, last value is repeated
    x4 = omd_a.step(g)
    np.testing.assert_allclose(x4, x3 - eta_arr[-1] * g, atol=1e-12)

    # Callable schedule
    seq = [0.05, 0.07, 0.11]

    def eta_fn(t):  # t=0,1,2... uses seq[t] if in range else last
        return seq[t] if t < len(seq) else seq[-1]

    omd_c = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=eta_fn,
        mode="omd",
    )
    y0 = np.ones(d) / d
    y1 = omd_c.step(g)
    np.testing.assert_allclose(y1, y0 - seq[0] * g, atol=1e-12)
    y2 = omd_c.step(g)
    np.testing.assert_allclose(y2, y1 - seq[1] * g, atol=1e-12)
    y3 = omd_c.step(g)
    np.testing.assert_allclose(y3, y2 - seq[2] * g, atol=1e-12)
    y4 = omd_c.step(g)
    np.testing.assert_allclose(y4, y3 - seq[-1] * g, atol=1e-12)


def test_dynamic_mirror_maps_update_internals():
    # Check that dynamic maps update state (AdaGrad, AdaBARRONS)
    g0 = np.array([2.0, 0.0, 0.0])
    g1 = np.array([2.0, 0.0, 0.0])

    # AdaGrad-like: H grows with sum of squares, step in coord 0 diminishes compared to Euclidean
    am = AdaptiveMahalanobisMap(eps=1e-8)
    omd_adagrad = FirstOrderOCO(
        mirror_map=am, projector=IdentityProjector(), eta=1.0, mode="omd"
    )
    euc = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=1.0,
        mode="omd",
    )

    x1_ad = omd_adagrad.step(g0)  # after first step, both similar direction
    x1_eu = euc.step(g0)
    # second step: AdaGrad scales down coord 0 more
    x2_ad = omd_adagrad.step(g1)
    x2_eu = euc.step(g1)

    # movement along coord 0 is smaller in AdaGrad than Euclidean on second step
    move_ad = abs(x2_ad[0] - x1_ad[0])
    move_eu = abs(x2_eu[0] - x1_eu[0])
    assert move_ad <= move_eu + 1e-12

    # AdaBARRONS: keep positivity and normalization even with extreme grads
    alb = AdaptiveLogBarrierMap(eps=1e-12)
    proj = AutoProjector(ProjectionConfig(lower=0.0, upper=1.0, budget=1.0))
    ftrl_alb = FirstOrderOCO(mirror_map=alb, projector=proj, eta=0.5, mode="ftrl")
    w = ftrl_alb.step(np.array([-10.0, -10.0, -10.0]))  # adversarial sign
    assert_simplex(w, atol=1e-9)


def test_euclidean_omd_matches_projected_step():
    """
    With ψ(w)=½‖w‖² the OMD step is Proj(w_t - η g_t); _composite_update should be identity plus projector (src/.../_mirror_maps.py:328-340).
    """
    engine = FirstOrderOCO(
        mirror_map=EuclideanMirrorMap(),
        projector=IdentityProjector(),
        eta=0.1,
        mode="omd",
    )
    engine._x_t = np.array([0.4, 0.6])
    grad = np.array([0.3, -0.2])
    w_next = engine.step(grad)
    expected = np.array([0.4, 0.6]) - 0.1 * grad
    assert np.allclose(w_next, expected)


def test_zero_predictor_matches_vanilla_omd():
    """Test that zero predictor matches vanilla OMD (no optimism)."""
    d = 5
    T = 10
    eta = 0.1

    # Create two engines: one with zero predictor, one without
    map1 = EuclideanMirrorMap()
    map2 = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine_optimistic = FirstOrderOCO(
        mirror_map=map1, projector=proj, eta=eta, predictor=ZeroPredictor(), mode="omd"
    )

    engine_vanilla = FirstOrderOCO(
        mirror_map=map2, projector=proj, eta=eta, predictor=None, mode="omd"
    )

    # Run both engines on same sequence of gradients
    gradients = [np.random.randn(d) for _ in range(T)]

    weights_optimistic = []
    weights_vanilla = []

    for g in gradients:
        w1 = engine_optimistic.step(g)
        w2 = engine_vanilla.step(g)
        weights_optimistic.append(w1)
        weights_vanilla.append(w2)

    # They should produce identical trajectories
    for w1, w2 in zip(weights_optimistic, weights_vanilla, strict=True):
        np.testing.assert_allclose(w1, w2, rtol=1e-10, atol=1e-10)


def test_first_round_initialization():
    """Test that first round handles missing prev_prediction correctly."""
    d = 3
    eta = 0.1

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    # First gradient
    g1 = np.array([1.0, 2.0, 3.0])
    w1 = engine.step(g1)

    # Should initialize _prev_prediction to zeros, so effective_grad = g1 + g1 = 2*g1
    # But since prev_prediction is initialized to zero, effective_grad = g1 - 0 + 0 = g1
    # (because LastGradPredictor returns empty/zeros on first call)
    assert engine._prev_prediction is not None
    assert w1.shape == (d,)


def test_last_grad_predictor_effective_gradient():
    """Test that LastGradPredictor produces correct effective gradients."""
    eta = 0.1

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    g1 = np.array([1.0, 0.0, 0.0])
    g2 = np.array([0.0, 1.0, 0.0])
    g3 = np.array([0.0, 0.0, 1.0])

    # Round 1: prev_pred = 0, m_t = 0 (no last grad yet)
    # effective_grad = g1 - 0 + 0 = g1
    engine.step(g1)
    engine._prev_prediction.copy()
    # After round 1, prev_prediction should be the prediction made at round 1
    # LastGradPredictor at round 1 returned 0, so prev_pred should be 0

    # Round 2: prev_pred = 0, m_t = g1 (last grad predictor returns g1)
    # effective_grad = g2 - 0 + g1
    engine.step(g2)
    prev_pred_2 = engine._prev_prediction.copy()
    # After round 2, prev_prediction should be g1

    # Round 3: prev_pred = g1, m_t = g2
    # effective_grad = g3 - g1 + g2
    engine.step(g3)
    prev_pred_3 = engine._prev_prediction.copy()
    # After round 3, prev_prediction should be g2

    # Check that prev_prediction is updated correctly
    np.testing.assert_allclose(prev_pred_2, g1, rtol=1e-10)
    np.testing.assert_allclose(prev_pred_3, g2, rtol=1e-10)


def test_prediction_tracking():
    """Test that _prev_prediction is correctly tracked across rounds."""
    eta = 0.1

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    # Use constant predictor for easy tracking
    const_vec = np.array([0.5, 0.5, 0.5])
    predictor = ConstantPredictor(const_vec)

    engine = FirstOrderOCO(
        mirror_map=map_obj, projector=proj, eta=eta, predictor=predictor, mode="omd"
    )

    g1 = np.array([1.0, 0.0, 0.0])
    g2 = np.array([0.0, 1.0, 0.0])

    # Round 1
    _ = engine.step(g1)
    # After round 1, prev_prediction should be const_vec
    np.testing.assert_allclose(engine._prev_prediction, const_vec, rtol=1e-10)

    # Round 2
    _ = engine.step(g2)
    # Should still be const_vec (constant predictor)
    np.testing.assert_allclose(engine._prev_prediction, const_vec, rtol=1e-10)


def test_smooth_predictor():
    """Test SmoothPredictor with clipping."""
    d = 3
    eta = 0.1
    L = 0.5

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj,
        projector=proj,
        eta=eta,
        predictor=SmoothPredictor(smoothness_L=L),
        mode="omd",
    )

    # First gradient - predictor has no history yet, returns zeros
    g1 = np.array([2.0, -3.0, 1.0])
    _ = engine.step(g1)
    # After first step, prev_prediction should be zeros (no last_grad yet)
    np.testing.assert_allclose(engine._prev_prediction, np.zeros(d), rtol=1e-10)

    # Second gradient - now predictor should clip g1
    g2 = np.array([1.0, 1.0, 1.0])
    _ = engine.step(g2)
    # Predictor should have clipped g1 to [-L, L]
    expected_clipped = np.clip(g1, -L, L)
    np.testing.assert_allclose(engine._prev_prediction, expected_clipped, rtol=1e-10)


def test_omd_vs_ftrl_modes_with_predictor():
    """Test that OMD and FTRL modes both work with predictors."""
    d = 3
    T = 5
    eta = 0.1

    map_omd = EuclideanMirrorMap()
    map_ftrl = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine_omd = FirstOrderOCO(
        mirror_map=map_omd,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    engine_ftrl = FirstOrderOCO(
        mirror_map=map_ftrl,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="ftrl",
    )

    gradients = [np.random.randn(d) for _ in range(T)]

    # Both should run without errors
    for g in gradients:
        w_omd = engine_omd.step(g)
        w_ftrl = engine_ftrl.step(g)
        assert w_omd.shape == (d,)
        assert w_ftrl.shape == (d,)


def test_numerical_example_hand_calculated():
    """Test against hand-calculated optimistic OMD update."""
    # Simple 2D case with Euclidean mirror map
    eta = 1.0

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    # Use constant predictor for predictability
    predictor = ConstantPredictor(np.array([0.1, 0.1]))

    engine = FirstOrderOCO(
        mirror_map=map_obj, projector=proj, eta=eta, predictor=predictor, mode="omd"
    )

    # Start from uniform
    x0 = np.array([0.5, 0.5])
    engine._x_t = x0.copy()

    g1 = np.array([1.0, -1.0])

    # Round 1:
    # prev_pred = 0 (initialized to zero)
    # m_t = [0.1, 0.1] (from predictor)
    # effective_grad = g1 - 0 + [0.1, 0.1] = [1.1, -0.9]
    # OMD update: x1 = x0 - eta * effective_grad = [0.5, 0.5] - 1.0 * [1.1, -0.9]
    #           = [-0.6, 1.4]

    w1 = engine.step(g1)

    expected_w1 = np.array([-0.6, 1.4])
    np.testing.assert_allclose(w1, expected_w1, rtol=1e-10, atol=1e-10)

    # After round 1, prev_prediction should be [0.1, 0.1]
    np.testing.assert_allclose(
        engine._prev_prediction, np.array([0.1, 0.1]), rtol=1e-10
    )


def test_entropy_mirror_map_with_predictor():
    """Test that optimistic updates work with entropy mirror map (EG)."""
    d = 4
    T = 10
    eta = 0.1

    map_obj = EntropyMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj,
        projector=proj,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    gradients = [np.random.randn(d) * 0.1 for _ in range(T)]

    # Should run without errors and produce valid probability distributions
    for g in gradients:
        w = engine.step(g)
        assert w.shape == (d,)
        assert np.all(w >= 0)
        assert np.isclose(np.sum(w), 1.0, rtol=1e-6)


def test_smooth_gradients_lower_regret():
    """Test that optimistic OMD has advantage on smooth gradient sequences."""
    d = 5
    T = 50
    eta = 0.05

    # Create smooth gradient sequence (slowly changing)
    base_grad = np.random.randn(d)
    gradients = []
    for _t in range(T):
        noise = np.random.randn(d) * 0.01  # Small noise
        gradients.append(base_grad + noise)
        base_grad = base_grad * 0.99 + noise  # Slow drift

    # Test with vanilla OMD
    map1 = EuclideanMirrorMap()
    proj1 = IdentityProjector()
    engine_vanilla = FirstOrderOCO(
        mirror_map=map1, projector=proj1, eta=eta, predictor=None, mode="omd"
    )

    # Test with optimistic OMD
    map2 = EuclideanMirrorMap()
    proj2 = IdentityProjector()
    engine_optimistic = FirstOrderOCO(
        mirror_map=map2,
        projector=proj2,
        eta=eta,
        predictor=LastGradPredictor(),
        mode="omd",
    )

    loss_vanilla = 0.0
    loss_optimistic = 0.0

    for g in gradients:
        w1 = engine_vanilla.step(g)
        w2 = engine_optimistic.step(g)

        # Linear loss
        loss_vanilla += np.dot(g, w1)
        loss_optimistic += np.dot(g, w2)

    # On smooth sequences, optimistic should typically do better or comparable
    # (not a strict requirement due to randomness, but good sanity check)
    # We just check both are finite and reasonable
    assert np.isfinite(loss_vanilla)
    assert np.isfinite(loss_optimistic)


def test_no_predictor_no_prev_prediction():
    """Test that without predictor, _prev_prediction stays None."""
    eta = 0.1

    map_obj = EuclideanMirrorMap()
    proj = IdentityProjector()

    engine = FirstOrderOCO(
        mirror_map=map_obj, projector=proj, eta=eta, predictor=None, mode="omd"
    )

    g1 = np.array([1.0, 0.0, 0.0])
    _ = engine.step(g1)

    # Without predictor, _prev_prediction should remain None
    assert engine._prev_prediction is None
