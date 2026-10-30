"""Custom constraints are rebuilt for each causal projection."""

import cvxpy as cp
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skfolio.model_selection import online_predict
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
from skfolio.prior import BasePrior, ReturnDistribution


@pytest.mark.parametrize("geometry", ["euclidean", "entropy", "adaptive"])
@pytest.mark.parametrize(
    "callback",
    [lambda w: w[0] <= 0.3, lambda w: [w[0] <= 0.3, cp.sum_squares(w) <= 0.6]],
)
def test_binding_callback_across_geometries(geometry, callback):
    projector = AutoProjector(ProjectionConfig(add_constraints=callback))
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
    assert_allclose(weights, [0.3, 0.7], atol=3e-5)


@pytest.mark.parametrize(
    "estimator_class",
    [
        ExponentiatedGradient,
        AdaGrad,
        OnlineMovingAverageReversion,
        PassiveAggressiveMeanReversion,
    ],
)
def test_public_callbacks_clone_and_continue(estimator_class):
    def cap(w, model):
        assert isinstance(model, estimator_class)
        assert model.n_features_in_ == 2
        return w[0] <= 0.3

    model = estimator_class(add_constraints=cap, store_weights=True)
    X = np.array([[0.1, -0.01], [0.05, -0.01], [0.08, -0.01]])
    fitted = clone(model).fit(X)
    streamed = clone(model)
    for row in X:
        streamed.partial_fit(row)
    assert_allclose(fitted.weights_, streamed.weights_, atol=1e-6)
    assert np.all(fitted.all_weights_[:, 0] <= 0.3 + 1e-6)
    result = clone(model).fit_predict(X)
    assert len(result.returns) == len(X)
    assert fitted.initial_weights_[0] <= 0.3 + 1e-6


@pytest.mark.parametrize(
    "callback,error,match",
    [
        (lambda w: None, TypeError, "must return"),
        (lambda w: True, TypeError, "must return"),
        (lambda w: [w >= 0, 1], TypeError, "must return"),
        (lambda w: cp.sum_squares(w) >= 0.6, ValueError, "convex DCP"),
        (lambda w: w == cp.Variable(2, boolean=True), ValueError, "integer"),
        (lambda w, model, extra: w >= 0, ValueError, "1 or 2"),
        (lambda w: w[9] >= 0, TypeError, "Error while calling"),
        (lambda w: w[0] >= 2, RuntimeError, "failed"),
    ],
)
def test_invalid_callbacks_fail_explicitly(callback, error, match):
    with pytest.raises(error, match=match):
        AutoProjector(ProjectionConfig(add_constraints=callback)).project(
            np.array([0.5, 0.5])
        )


@pytest.mark.parametrize("anchor", [np.array([0.2, 0.8]), np.array([0.8, 0.2])])
def test_zero_turnover_custom_auxiliary_variables(anchor):
    def cap(w):
        bound = cp.Variable()
        return [bound >= cp.sum_squares(w), bound <= 0.7, w[0] <= 0.3]

    projector = AutoProjector(
        ProjectionConfig(add_constraints=cap, previous_weights=anchor, max_turnover=0)
    )
    if anchor[0] > 0.3:
        with pytest.raises(RuntimeError, match="failed"):
            projector.project(np.array([0.5, 0.5]))
    else:
        assert_allclose(projector.project(np.array([0.5, 0.5])), anchor, atol=0)


class CausalPrior(BasePrior):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        raise AssertionError("Use causal updates")

    def partial_fit(self, X, y=None):
        self.rows_ = np.vstack([getattr(self, "rows_", np.empty((0, 2))), X])
        self.return_distribution_ = ReturnDistribution(
            mu=self.rows_.mean(axis=0), covariance=np.eye(2), returns=self.rows_.copy()
        )
        return self


def test_callback_observes_only_causal_prior_and_rebuilds():
    seen = []

    def cap(w, model):
        rows = model.prior_estimator_.rows_
        seen.append(rows.copy())
        return w[0] == rows[-1, 0]

    X = np.array([[0.1, 0], [0.2, 0], [0.3, 0]])
    model = ExponentiatedGradient(
        add_constraints=cap, prior_estimator=CausalPrior(), store_weights=True
    ).fit(X)
    assert_allclose(model.all_weights_[1:, 0], X[:-1, 0], atol=1e-6)
    assert_allclose(model.weights_[0], X[-1, 0], atol=1e-6)
    assert seen
    for rows in seen:
        assert_allclose(rows, X[: len(rows)])
    assert {len(rows) for rows in seen} == {1, 2, 3}
    result = online_predict(clone(model), X, warmup_size=1, test_size=1)
    assert len(result.returns) == 2
    # No warmup must not silently bypass a prior-dependent callback at entry.
    with pytest.raises(TypeError, match="Error while calling"):
        clone(model).fit_predict(X)


def test_failed_callback_requires_reset():
    model = ExponentiatedGradient(add_constraints=lambda w: w[0] <= 0.3).fit([[0.1, 0]])
    model.set_params(add_constraints=lambda w: w[0] >= 2)
    with pytest.raises(RuntimeError, match="failed"):
        model.partial_fit([0.1, 0])
    with pytest.raises(RuntimeError, match="reset"):
        model.partial_fit([0.1, 0])
    model.set_params(add_constraints=None).fit([[0.1, 0]])
