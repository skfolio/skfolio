"""Supervised policy projection contracts."""

import numpy as np
import pytest

from skfolio.exceptions import GroupNotFoundError

from ._supervised_helpers import RecordingOptimizer, data, model


def test_projection_enforces_bounds_and_groups_for_future_predictions():
    X, panel = data()
    estimator = model(
        long_only=True,
        min_weights={"a": 0.2},
        max_weights=0.5,
        groups=[["equity", "equity", "bond"]],
        linear_constraints=["equity <= 0.6"],
    ).fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    weights = estimator.predict_weights(future).to_numpy()
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-7)
    assert (weights >= -1e-7).all()
    assert (weights <= 0.5 + 1e-7).all()
    assert (weights[:, 0] >= 0.2 - 1e-7).all()
    assert (weights[:, :2].sum(axis=1) <= 0.6 + 1e-7).all()


@pytest.mark.parametrize(
    "constraints, message",
    [
        ({"max_weights": 0.2}, "infeasible"),
        ({"linear_constraints": ["unknown <= 0.2"]}, "unknown"),
    ],
)
def test_invalid_constraints_fail_before_label_fit(constraints, message):
    X, panel = data()
    RecordingOptimizer.windows = []
    with pytest.raises((ValueError, GroupNotFoundError), match=message):
        model(**constraints).fit(X, characteristics=panel)
    assert not RecordingOptimizer.windows


def test_projection_is_euclidean_not_clip_and_normalize():
    from skfolio.optimization.parametric._weight_projection import _WeightProjection

    projection = _WeightProjection(["a", "b", "c"], True, None, None, None, None)
    actual = projection.project(np.array([[0.8, 0.4, -0.2]]))
    np.testing.assert_allclose(actual, [[0.7, 0.3, 0]], atol=1e-6)


def test_risk_limit_at_prediction():
    X, panel = data()
    estimator = model(max_variance=0.34).fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    weights = estimator.predict_weights(future, covariance=np.eye(3)).to_numpy()
    assert (np.sum(weights**2, axis=1) <= 0.34 + 1e-6).all()
    with pytest.raises(ValueError, match="covariance"):
        estimator.predict_weights(future)
    with pytest.raises(ValueError, match="explicit risk"):
        estimator.predict(X)
    with pytest.raises(ValueError, match="infeasible"):
        estimator.predict_weights(future, covariance=10 * np.eye(3))


def test_turnover_uses_each_supplied_holdings_vector():
    X, panel = data()
    estimator = model(max_turnover=0).fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    holdings = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]] * 2)
    actual = estimator.predict_weights(future, holdings=holdings)
    np.testing.assert_allclose(actual, holdings, atol=1e-6)
    with pytest.raises(ValueError, match="holdings"):
        estimator.predict_weights(future)


def test_conflicting_risk_and_turnover_raise():
    X, panel = data()
    estimator = model(max_turnover=0, max_variance=0.34).fit(X, characteristics=panel)
    _, future = data(n=1, offset=11)
    with pytest.raises(ValueError, match="infeasible"):
        estimator.predict_weights(future, covariance=np.eye(3), holdings=[1, 0, 0])


def test_reused_projection_matches_fresh_solves():
    from skfolio.optimization.parametric._weight_projection import _WeightProjection

    def make_projection():
        return _WeightProjection(["a", "b", "c"], True, None, 0.8, None, None)

    projection = make_projection()
    proposals = np.array([[0.8, 0.4, -0.2], [-0.1, 0.3, 0.8], [0.3, 0.3, 0.4]])
    expected = np.vstack([make_projection().project(row[None, :]) for row in proposals])
    np.testing.assert_allclose(projection.project(proposals), expected, atol=1e-6)
    # A dynamic call must not leak its exclusion into later static calls.
    excluded = projection.project(
        proposals[:1], eligible=np.array([[True, True, False]])
    )
    assert abs(excluded[0, 2]) < 1e-6
    np.testing.assert_allclose(projection.project(proposals), expected, atol=1e-6)
    funded = projection.project(proposals, budget=0.8)
    np.testing.assert_allclose(funded.sum(axis=1), 0.8, atol=1e-6)


def test_projection_solver_configuration(monkeypatch):
    import cvxpy as cp
    from sklearn.base import clone

    seen = []
    original = cp.Problem.solve

    def recording_solve(problem, *args, **kwargs):
        seen.append(kwargs.copy())
        return original(problem, *args, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", recording_solve)
    settings = {"tol_gap_abs": 1e-9}
    estimator = clone(
        model(projection_solver="CLARABEL", projection_solver_params=settings)
    )
    X, panel = data()
    estimator.fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    estimator.predict_weights(future)
    assert seen
    assert all(
        call["solver"] == "CLARABEL" and call["tol_gap_abs"] == 1e-9 for call in seen
    )
    assert settings == {"tol_gap_abs": 1e-9}


def test_projection_invalid_solver_fails_without_fallback():
    import cvxpy as cp

    X, panel = data()
    with pytest.raises(cp.error.SolverError):
        model(projection_solver="NOT_A_SOLVER").fit(X, characteristics=panel)
