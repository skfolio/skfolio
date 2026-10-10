"""Convex potentials for solving constrained mirror updates in dual coordinates."""

from collections.abc import Callable

import cvxpy as cp
import numpy as np

from skfolio.optimization.online._mirror_maps import (
    AdaBarronsBarrierMap,
    AdaptiveLogBarrierMap,
    AdaptiveMahalanobisMap,
    AdaptiveVariationMap,
    BaseMirrorMap,
    BurgMirrorMap,
    CompositeMirrorMap,
    DiagonalQuadraticMap,
    DynamicMirrorMap,
    EntropyMirrorMap,
    EuclideanMap,
    EuclideanMirrorMap,
    FullQuadraticMap,
    LogBarrierMap,
)


def mirror_potential(
    mirror: BaseMirrorMap, n: int, gradient: np.ndarray | None = None
) -> Callable[[cp.Variable], cp.Expression]:
    """Represent the map's potential, using its pending-state hook when supplied.

    Constants are omitted. Composite maps use their existing current-state
    convention (their inherited pending-state hook does not advance components).
    Unsupported maps raise instead of silently using Euclidean geometry.
    """
    supported = (
        CompositeMirrorMap,
        EntropyMirrorMap,
        EuclideanMirrorMap,
        EuclideanMap,
        AdaptiveMahalanobisMap,
        AdaptiveVariationMap,
        DiagonalQuadraticMap,
        FullQuadraticMap,
        BurgMirrorMap,
        LogBarrierMap,
        AdaptiveLogBarrierMap,
        AdaBarronsBarrierMap,
    )
    if type(mirror) not in supported:
        raise NotImplementedError(
            f"Native constrained geometry is not implemented for {type(mirror).__name__}"
        )
    if isinstance(mirror, CompositeMirrorMap):
        terms = [mirror_potential(part, n) for part in mirror.components_]

        def composite(w: cp.Variable) -> cp.Expression:
            """Sum component potentials without separately projecting components."""
            return sum((term(w) for term in terms), cp.Constant(0.0))

        return composite
    if isinstance(mirror, EntropyMirrorMap):
        return lambda w: -cp.sum(cp.entr(w))
    if isinstance(mirror, EuclideanMirrorMap):
        return lambda w: 0.5 * cp.sum_squares(w)

    def forward(w: np.ndarray) -> np.ndarray:
        """Read the same geometry used to construct the engine's dual center."""
        if gradient is not None and isinstance(mirror, DynamicMirrorMap):
            return mirror.grad_psi_after(w, gradient)
        return mirror.grad_psi(w)

    if isinstance(mirror, FullQuadraticMap):
        matrix = np.column_stack([forward(row) for row in np.eye(n)])
        if not np.isfinite(matrix).all() or np.linalg.eigvalsh(matrix).min() < -1e-10:
            raise ValueError(
                "Mirror curvature must be finite and positive semidefinite"
            )
        return lambda w: 0.5 * cp.quad_form(w, cp.psd_wrap(matrix))
    if isinstance(
        mirror,
        (
            AdaptiveMahalanobisMap,
            AdaptiveVariationMap,
            DiagonalQuadraticMap,
            EuclideanMap,
        ),
    ):
        diagonal = forward(np.ones(n))
        if not np.isfinite(diagonal).all() or np.any(diagonal < 0):
            raise ValueError("Mirror curvature must be finite and nonnegative")
        return lambda w: 0.5 * cp.sum(cp.multiply(diagonal, cp.square(w)))
    if isinstance(
        mirror,
        (BurgMirrorMap, LogBarrierMap, AdaptiveLogBarrierMap, AdaBarronsBarrierMap),
    ):
        coefficients = -forward(np.ones(n))
        if not np.isfinite(coefficients).all() or np.any(coefficients < 0):
            raise ValueError("Barrier coefficients must be finite and nonnegative")
        active = coefficients > 0
        if not active.any():
            return lambda w: cp.Constant(0.0)
        return lambda w: -cp.sum(cp.multiply(coefficients[active], cp.log(w[active])))
    raise NotImplementedError(
        f"Native constrained geometry is not implemented for {type(mirror).__name__}"
    )
