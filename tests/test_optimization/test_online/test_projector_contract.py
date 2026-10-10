"""Regression tests for source projector boundary corrections."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._projection import AutoProjector, ProjectionConfig


def test_no_budget_leaves_box_feasible_weights_unchanged():
    projector = AutoProjector(ProjectionConfig(budget=None))
    assert_allclose(projector.project([0.1, 0.2]), [0.1, 0.2])


def test_infeasible_budget_is_not_silently_replaced():
    with pytest.raises(ValueError, match="infeasible"):
        AutoProjector(ProjectionConfig(upper=0.2)).project([0.5, 0.5])


def test_zero_turnover_never_repairs_infeasible_holdings():
    projector = AutoProjector(
        ProjectionConfig(previous_weights=[0.9, 0.1], max_turnover=0, upper=0.6)
    )
    with pytest.raises(ValueError, match="Zero turnover"):
        projector.project([0.5, 0.5])


def test_turnover_intersection_rejects_insufficient_cap():
    projector = AutoProjector(
        ProjectionConfig(previous_weights=[0.9, 0.1], max_turnover=0.1, upper=0.6)
    )
    with pytest.raises(RuntimeError, match="projection failed"):
        projector.project([0.5, 0.5])


def test_short_exposure_uses_convex_upper_bound():
    projector = AutoProjector(ProjectionConfig(lower=-1, upper=2, max_short=0.2))
    weights = projector.project([-0.5, 1.5])
    assert_allclose(weights, [-0.2, 1.2], atol=1e-7)


def test_tracking_and_variance_constraints_both_apply():
    projector = AutoProjector(
        ProjectionConfig(
            X_tracking=np.eye(2),
            tracking_error_benchmark=[0, 0],
            max_tracking_error=10,
            covariance=np.eye(2),
            variance_bound=0.5,
        )
    )
    assert_allclose(projector.project([1, 0]), [0.5, 0.5], atol=1e-5)
