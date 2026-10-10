"""Native constrained updates checked against analytical stationarity conditions."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._foco import FirstOrderOCO
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
    BurgMirrorMap,
    EntropyMirrorMap,
    EuclideanMap,
    FullQuadraticMap,
    LogBarrierMap,
    TsallisMirrorMap,
)
from skfolio.optimization.online._projection import AutoProjector, ProjectionConfig


@pytest.mark.parametrize("constraint", ["box", "linear"])
def test_entropy_preserves_uncapped_asset_ratio(constraint):
    kwargs = (
        {"upper": [0.5, 1, 1]}
        if constraint == "box"
        else {"left_inequality": [[1, 0, 0]], "right_inequality": [0.5]}
    )
    projector = AutoProjector(ProjectionConfig(**kwargs))
    dual = np.log([0.8, 0.15, 0.05]) + 1
    result = projector.solve_update(EntropyMirrorMap(), dual)
    assert_allclose(result, [0.5, 0.375, 0.125], atol=2e-5)
    assert not np.allclose(result, projector.project([0.8, 0.15, 0.05]))


def test_entropy_free_budget_does_not_normalize_away_dual_scale():
    result = AutoProjector(ProjectionConfig(budget=None)).solve_update(
        EntropyMirrorMap(),
        np.log([0.1, 0.2]) + 1,
    )
    assert_allclose(result, [0.1, 0.2], atol=1e-5)


def test_entropy_engine_enforces_turnover_in_native_update():
    projector = AutoProjector(
        ProjectionConfig(previous_weights=[0.5, 0.5], max_turnover=0.2)
    )
    engine = FirstOrderOCO(EntropyMirrorMap(), projector, eta=1, mode="omd")
    assert_allclose(engine.step(np.array([-2.0, 2.0])), [0.6, 0.4], atol=1e-6)


def test_adagrad_pending_curvature_matches_weighted_projection():
    mirror = AdaptiveMahalanobisMap(eps=1e-8)
    engine = FirstOrderOCO(
        mirror, AutoProjector(ProjectionConfig()), eta=0.1, mode="omd"
    )
    center = np.array([0.5, 0.5])
    squared_sum = np.zeros(2)
    for gradient in [np.array([2.0, 4.0]), np.array([1.0, -2.0])]:
        squared_sum += gradient**2
        curvature = np.sqrt(squared_sum + mirror.eps)
        candidate = center - 0.1 * gradient / curvature
        multiplier = (candidate.sum() - 1) / (1 / curvature).sum()
        expected = candidate - multiplier / curvature
        actual = engine.step(gradient)
        assert_allclose(actual, expected, atol=1e-7)
        assert_allclose(mirror._Gsq, squared_sum)
        center = actual


def test_composite_potential_is_solved_jointly_without_state_mutation():
    quadratic = FullQuadraticMap(beta=2)
    quadratic.update_state(np.array([0.3, -0.2]))
    mirror = LogBarrierMap(2) + EuclideanMap(0.3) + quadratic
    center = np.array([0.4, 0.6])
    dual = mirror.grad_psi(center) - np.array([0.1, -0.2])
    before = deepcopy(mirror)
    result = AutoProjector(
        ProjectionConfig(
            lower=0.1,
            upper=0.9,
            solver_params={
                "solver": "CLARABEL",
                "tol_gap_abs": 1e-11,
                "tol_gap_rel": 1e-11,
                "tol_feas": 1e-11,
            },
        )
    ).solve_update(mirror, dual)
    residual = mirror.grad_psi(result) - dual
    assert_allclose(residual[0], residual[1], atol=2e-4)
    assert_allclose(result.sum(), 1, atol=1e-8)
    assert_allclose(mirror.grad_psi(center), before.grad_psi(center))


@pytest.mark.parametrize("mode", ["omd", "ftrl"])
def test_burg_update_respects_coefficient_and_mode(mode):
    mirror = BurgMirrorMap(barrier_coef=2)
    engine = FirstOrderOCO(
        mirror,
        AutoProjector(
            ProjectionConfig(
                solver_params={
                    "solver": "CLARABEL",
                    "tol_gap_abs": 1e-11,
                    "tol_gap_rel": 1e-11,
                    "tol_feas": 1e-11,
                }
            )
        ),
        eta=0.5,
        mode=mode,
    )
    center = np.array([0.5, 0.5])
    total = np.zeros(2)
    for gradient in [np.array([0.1, -0.3]), np.array([-0.2, 0.05])]:
        total += gradient
        dual = (
            mirror.grad_psi(center) - 0.5 * gradient if mode == "omd" else -0.5 * total
        )
        result = engine.step(gradient)
        residual = mirror.grad_psi(result) - dual
        assert_allclose(residual[0], residual[1], atol=2e-4)
        center = result


def test_unrepresented_geometry_fails_explicitly():
    engine = FirstOrderOCO(
        TsallisMirrorMap(), AutoProjector(ProjectionConfig()), eta=0.1
    )
    with pytest.raises(NotImplementedError, match="TsallisMirrorMap"):
        engine.step(np.array([0.1, -0.1]))


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize(
    "name",
    [
        "EntropyMirrorMap",
        "EuclideanMirrorMap",
        "EuclideanMap",
        "AdaptiveMahalanobisMap",
        "AdaptiveVariationMap",
        "DiagonalQuadraticMap",
        "FullQuadraticMap",
        "BurgMirrorMap",
        "LogBarrierMap",
        "AdaptiveLogBarrierMap",
        "AdaBarronsBarrierMap",
    ],
)
def test_convex_potential_derivative_matches_source_map(name, pending):
    import cvxpy as cp

    from skfolio.optimization.online import _mirror_maps
    from skfolio.optimization.online._geometry import mirror_potential

    mirror = getattr(_mirror_maps, name)()
    if isinstance(mirror, _mirror_maps.DynamicMirrorMap):
        mirror.ensure_dim(2)
        mirror.update_state(np.array([0.2, 0.3]))
    gradient = np.array([0.4, -0.1]) if pending else None
    variable = cp.Variable(2)
    expression = mirror_potential(mirror, 2, gradient)(variable)
    weights = np.array([0.4, 0.6])
    numerical = np.zeros(2)
    for index in range(2):
        offset = np.eye(2)[index] * 1e-6
        variable.value = weights + offset
        upper = expression.value
        variable.value = weights - offset
        numerical[index] = (upper - expression.value) / 2e-6
    expected = (
        mirror.grad_psi_after(weights, gradient)
        if pending and isinstance(mirror, _mirror_maps.DynamicMirrorMap)
        else mirror.grad_psi(weights)
    )
    assert_allclose(numerical, expected, atol=1e-7)


def test_custom_entropy_subclass_requires_explicit_potential_support():
    class DifferentEntropy(EntropyMirrorMap):
        def grad_psi(self, w):
            return 2 * super().grad_psi(w)

    with pytest.raises(NotImplementedError, match="DifferentEntropy"):
        AutoProjector(ProjectionConfig()).solve_update(DifferentEntropy(), np.ones(2))


def test_zero_turnover_rejects_barrier_domain_boundary():
    projector = AutoProjector(
        ProjectionConfig(previous_weights=[0.0, 1.0], max_turnover=0)
    )
    with pytest.raises(RuntimeError, match="mirror domain"):
        projector.solve_update(BurgMirrorMap(), np.ones(2))
