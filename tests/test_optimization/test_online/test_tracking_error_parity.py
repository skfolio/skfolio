"""Tracking-error constraints share the batch optimizer's uncentred definition."""

import cvxpy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization import MeanRisk
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
    EntropyMirrorMap,
)
from skfolio.optimization.online._projection import AutoProjector, ProjectionConfig
from skfolio.prior import ReturnDistribution


@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
def test_constant_active_return_is_not_removed(geometry):
    projector = AutoProjector(
        ProjectionConfig(
            X_tracking=[[0.1, 0.0], [0.1, 0.0]],
            tracking_error_benchmark=[0.02, 0.02],
            max_tracking_error=0.02 * np.sqrt(2),
        )
    )
    candidate = np.array([0.9, 0.1])
    if geometry == "euclidean":
        weights = projector.project(candidate)
    else:
        mirror = (
            EntropyMirrorMap() if geometry == "entropy" else AdaptiveMahalanobisMap()
        )
        if geometry == "adaptive":
            mirror.update_state(np.array([1.0, 2.0]))
        weights = projector.solve_update(mirror, mirror.grad_psi(candidate))
    assert_allclose(weights, [0.4, 0.6], atol=2e-5)


def test_batch_tracking_expression_and_projector_match():
    X = np.array([[0.1, -0.02], [0.04, 0.01], [-0.02, 0.03]])
    y = np.array([0.02, 0.01, 0.0])
    rd = ReturnDistribution(mu=X.mean(axis=0), covariance=np.cov(X.T), returns=X)
    w = cp.Variable(2)
    expression = MeanRisk()._tracking_error(
        return_distribution=rd, w=w, y=y, factor=cp.Constant(1.0)
    )
    candidate = np.array([0.9, 0.1])
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(w - candidate)),
        [w >= 0, cp.sum(w) == 1, expression <= 0.025],
    )
    problem.solve(solver="CLARABEL")
    actual = AutoProjector(
        ProjectionConfig(
            X_tracking=X, tracking_error_benchmark=y, max_tracking_error=0.025
        )
    ).project(candidate)
    assert_allclose(actual, w.value, atol=2e-5)
    assert_allclose(
        np.linalg.norm(X @ actual - y) / np.sqrt(len(X) - 1), 0.025, atol=1e-6
    )


@pytest.mark.parametrize(
    "X,y,match",
    [
        ([[0.1, 0]], [0.0], "two observations"),
        ([[np.nan, 0], [0, 0]], [0.0, 0.0], "finite"),
        ([[0.1, 0], [0, 0]], [0.0, np.inf], "finite"),
        ([[0.1, 0], [0, 0]], [0.0], "shapes"),
    ],
)
def test_invalid_tracking_inputs(X, y, match):
    with pytest.raises(ValueError, match=match):
        AutoProjector(
            ProjectionConfig(
                X_tracking=X, tracking_error_benchmark=y, max_tracking_error=0.1
            )
        ).project(np.array([0.5, 0.5]))


def test_zero_turnover_does_not_bypass_uncentred_limit():
    with pytest.raises(RuntimeError, match="zero turnover"):
        AutoProjector(
            ProjectionConfig(
                X_tracking=[[0.1, 0], [0.1, 0]],
                tracking_error_benchmark=[0, 0],
                max_tracking_error=0.01,
                previous_weights=[0.5, 0.5],
                max_turnover=0,
            )
        ).project(np.array([0.5, 0.5]))
