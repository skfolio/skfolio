"""Minimum position sizes retain exact zero-or-threshold semantics."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skfolio.optimization.online import (
    AdaGrad,
    ExponentiatedGradient,
    OnlineMovingAverageReversion,
    PassiveAggressiveMeanReversion,
)
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
    EntropyMirrorMap,
)
from skfolio.optimization.online._projection import AutoProjector, ProjectionConfig

scip = pytest.mark.skipif("SCIP" not in cp.installed_solvers(), reason="SCIP required")


@scip
@pytest.mark.parametrize(
    "threshold, candidate, expected",
    [
        (0.4, [0.5, 0.3, 0.2], [0.6, 0.4, 0]),
        (0.6, [0.5, 0.3, 0.2], [1, 0, 0]),
        ([0.6, 0, 0], [0.5, 0.3, 0.2], [0.6, 0.25, 0.15]),
        (0.3, [0.8, 0.15, 0.05], [0.7, 0.3, 0]),
    ],
)
def test_threshold_projection_known_optimum(threshold, candidate, expected):
    weights = AutoProjector(
        ProjectionConfig(threshold_long=threshold, solver_params={"solver": "SCIP"})
    ).project(np.array(candidate))
    assert_allclose(weights, expected, atol=5e-5)


@scip
def test_threshold_combined_with_support_and_custom_constraint():
    config = ProjectionConfig(
        threshold_long=0.3,
        cardinality=2,
        group_cardinalities={"a": 1},
        groups=[["a", "a", "b"]],
        add_constraints=lambda w: w[0] <= 0.6,
        solver_params={"solver": "SCIP"},
    )
    mirror = AdaptiveMahalanobisMap()
    mirror.update_state(np.array([1.0, 2.0, 3.0]))
    weights = AutoProjector(config).solve_update(
        mirror, mirror.grad_psi(np.array([0.5, 0.3, 0.2]))
    )
    assert np.count_nonzero(weights > 1e-6) <= 2
    assert np.count_nonzero(weights[:2] > 1e-6) <= 1
    assert np.all(weights[weights > 1e-6] >= 0.3 - 1e-6)
    assert weights[0] <= 0.6 + 1e-6
    assert_allclose(weights.sum(), 1, atol=1e-6)


@pytest.mark.parametrize(
    "threshold", [-0.1, np.nan, np.inf, [0.0, -0.1, 0.0], [0.0, 0.0]]
)
def test_invalid_thresholds_raise(threshold):
    with pytest.raises(ValueError, match="threshold_long"):
        AutoProjector(ProjectionConfig(threshold_long=threshold)).project(
            np.ones(3) / 3
        )


def test_zero_threshold_needs_no_integer_solver():
    actual = AutoProjector(ProjectionConfig(threshold_long=[0, 0, 0])).project(
        np.array([0.5, 0.3, 0.2])
    )
    assert_allclose(actual, [0.5, 0.3, 0.2], atol=1e-6)
    model = ExponentiatedGradient(threshold_long=0).fit([[0.1, 0.02, -0.03]])
    assert_allclose(model.weights_.sum(), 1, atol=1e-6)


def test_positive_threshold_requires_mip_and_rejects_entropy():
    projector = AutoProjector(ProjectionConfig(threshold_long=0.3))
    with pytest.raises(ValueError, match="mixed-integer"):
        projector.project(np.ones(3) / 3)
    with pytest.raises(ValueError, match="entropy"):
        projector.solve_update(EntropyMirrorMap(), np.ones(3))


@scip
@pytest.mark.parametrize(
    "estimator_class",
    [AdaGrad, OnlineMovingAverageReversion, PassiveAggressiveMeanReversion],
)
def test_named_thresholds_align_and_clone(estimator_class):
    X = pd.DataFrame([[0.1, 0.02, -0.03], [0.04, -0.01, 0.03]], columns=["a", "b", "c"])
    model = estimator_class(threshold_long={"a": 0.6}, solver="SCIP")
    fitted = clone(model).fit(X)
    assert fitted.weights_[0] < 1e-6 or fitted.weights_[0] >= 0.6 - 1e-6
    permuted = clone(model).fit(X[["c", "a", "b"]])
    assert_allclose(permuted.weights_[[1, 2, 0]], fitted.weights_, atol=1e-4)


@scip
def test_infeasible_threshold_and_locked_small_holdings():
    with pytest.raises(RuntimeError, match="failed"):
        AutoProjector(
            ProjectionConfig(
                threshold_long=0.6, upper=0.5, solver_params={"solver": "SCIP"}
            )
        ).project(np.ones(3) / 3)
    with pytest.raises(RuntimeError, match="failed"):
        AutoProjector(
            ProjectionConfig(
                threshold_long=0.3,
                previous_weights=[0.8, 0.2, 0],
                max_turnover=0,
                solver_params={"solver": "SCIP"},
            )
        ).project(np.ones(3) / 3)


@scip
def test_thresholds_jointly_enforce_risk_and_turnover():
    weights = AutoProjector(
        ProjectionConfig(
            threshold_long=0.3,
            cardinality=2,
            covariance=np.eye(3),
            variance_bound=0.52,
            previous_weights=[0.5, 0.5, 0],
            max_turnover=0.2,
            solver_params={"solver": "SCIP"},
        )
    ).project(np.array([0.7, 0.2, 0.1]))
    assert_allclose(weights, [0.6, 0.4, 0], atol=5e-5)
    assert weights @ weights <= 0.52 + 1e-6
    assert np.abs(weights - [0.5, 0.5, 0]).sum() <= 0.2 + 1e-6
