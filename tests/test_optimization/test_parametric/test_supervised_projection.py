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
