"""Discrete projections enforce support limits without relaxing integrality."""

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skfolio.exceptions import GroupNotFoundError
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
    "options, expected",
    [
        ({"cardinality": 2}, [0.6, 0.4, 0]),
        (
            {"group_cardinalities": {"a": 1}, "groups": [["a", "a", "b"]]},
            [0.65, 0, 0.35],
        ),
        ({"cardinality": 1}, [1, 0, 0]),
    ],
)
def test_support_projection_known_optimum(options, expected):
    weights = AutoProjector(
        ProjectionConfig(**options, solver_params={"solver": "SCIP"})
    ).project(np.array([0.5, 0.3, 0.2]))
    assert_allclose(weights, expected, atol=3e-5)


@scip
def test_adaptive_support_projection_and_turnover():
    mirror = AdaptiveMahalanobisMap()
    mirror.update_state(np.array([1.0, 2.0, 3.0]))
    config = ProjectionConfig(
        cardinality=2,
        previous_weights=np.array([0.6, 0.4, 0]),
        max_turnover=0,
        solver_params={"solver": "SCIP"},
    )
    weights = AutoProjector(config).solve_update(
        mirror, mirror.grad_psi(np.array([0.5, 0.3, 0.2]))
    )
    assert_allclose(weights, [0.6, 0.4, 0], atol=0)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_cardinality(count):
    with pytest.raises(ValueError, match="nonnegative integers"):
        AutoProjector(ProjectionConfig(cardinality=count)).project(np.ones(3) / 3)


def test_default_solver_does_not_relax_cardinality():
    with pytest.raises(ValueError, match="mixed-integer"):
        AutoProjector(ProjectionConfig(cardinality=1)).project(np.ones(3) / 3)


def test_entropy_update_is_rejected():
    with pytest.raises(ValueError, match="entropy"):
        AutoProjector(ProjectionConfig(cardinality=1)).solve_update(
            EntropyMirrorMap(), np.ones(3)
        )


@scip
@pytest.mark.parametrize(
    "groups, error", [(None, ValueError), ([["a", "a", "b"]], GroupNotFoundError)]
)
def test_missing_groups_raise(groups, error):
    with pytest.raises(error):
        AutoProjector(
            ProjectionConfig(
                group_cardinalities={"unknown": 1},
                groups=groups,
                solver_params={"solver": "SCIP"},
            )
        ).project(np.ones(3) / 3)


@scip
@pytest.mark.parametrize(
    "estimator_class",
    [AdaGrad, OnlineMovingAverageReversion, PassiveAggressiveMeanReversion],
)
def test_public_discrete_estimators(estimator_class):
    X = pd.DataFrame([[0.1, 0.02, -0.03], [0.04, -0.01, 0.03]], columns=["a", "b", "c"])
    model = estimator_class(
        cardinality=2,
        group_cardinalities={"sector": 1},
        groups={"a": ["sector"], "b": ["sector"], "c": ["other"]},
        solver="SCIP",
    )
    fitted = clone(model).fit(X)
    assert np.count_nonzero(fitted.weights_ > 1e-6) <= 2
    assert np.count_nonzero(fitted.weights_[:2] > 1e-6) <= 1
    streamed = clone(model)
    for i in range(len(X)):
        streamed.partial_fit(X.iloc[[i]])
    assert_allclose(streamed.weights_, fitted.weights_, atol=1e-5)


@scip
def test_infeasible_support_and_solver_cones_fail():
    with pytest.raises(RuntimeError, match="failed"):
        AutoProjector(
            ProjectionConfig(cardinality=1, upper=0.4, solver_params={"solver": "SCIP"})
        ).project(np.ones(3) / 3)
    with pytest.raises(ValueError, match="cone support"):
        AutoProjector(
            ProjectionConfig(cardinality=1, solver_params={"solver": "SCIPY"})
        ).project(np.ones(3) / 3)


def test_nonbinding_cardinality_needs_no_integer_solver():
    weights = AutoProjector(ProjectionConfig(cardinality=3)).project(
        np.array([0.5, 0.3, 0.2])
    )
    assert_allclose(weights, [0.5, 0.3, 0.2], atol=1e-6)


@scip
def test_eg_does_not_silently_switch_geometry():
    with pytest.raises(ValueError, match="entropy"):
        ExponentiatedGradient(cardinality=1, solver="SCIP").fit([[0.1, 0.02, -0.03]])
