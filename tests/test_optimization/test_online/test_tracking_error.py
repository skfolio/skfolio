"""Public benchmark inputs are validated and consumed causally."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from skfolio.optimization.online import (
    AdaGrad,
    ExponentiatedGradient,
    OnlineMovingAverageReversion,
    PassiveAggressiveMeanReversion,
)


@pytest.mark.parametrize(
    "estimator",
    [
        ExponentiatedGradient,
        AdaGrad,
        OnlineMovingAverageReversion,
        PassiveAggressiveMeanReversion,
    ],
)
def test_tracking_error_stream_matches_blocks_and_reset(estimator):
    X = np.array([[0.03, -0.01], [-0.02, 0.04], [0.01, -0.03], [0.02, 0.01]])
    y = X @ np.array([0.8, 0.2])
    cap = 0.001
    model = estimator(max_tracking_error=cap, store_weights=True)
    streamed = clone(model)
    for i in range(len(X)):
        streamed.partial_fit(X[i : i + 1], y[i : i + 1])
        if i:
            active = X[: i + 1] @ streamed.weights_ - y[: i + 1]
            assert np.linalg.norm(active) / np.sqrt(i) <= cap + 1e-7
    batch = model.fit(X, y)
    np.testing.assert_allclose(batch.weights_, streamed.weights_, atol=1e-7)
    np.testing.assert_allclose(batch.all_weights_, streamed.all_weights_, atol=1e-7)
    # A future benchmark change cannot alter the target before that observation.
    changed = y.copy()
    changed[-1] += 0.001
    future = clone(model).fit(X, changed)
    np.testing.assert_allclose(future.all_weights_, batch.all_weights_, atol=1e-7)
    model.fit(X[:2], y[:2])
    assert len(model._tracking_returns_) == 2
    np.testing.assert_allclose(model.weights_, clone(model).fit(X[:2], y[:2]).weights_)


@pytest.mark.parametrize("y", [None, [0.01], [np.nan, 0.01], [[0, 0], [0, 0]], 0.01])
def test_invalid_benchmark_rejected_before_update(y):
    model = ExponentiatedGradient(max_tracking_error=0.01).fit(
        [[0.01, 0.02], [0.02, 0.01]], [0.015, 0.015]
    )
    weights = model.weights_.copy()
    with pytest.raises(ValueError, match="y"):
        model.partial_fit([[0.01, 0.02], [0.02, 0.01]], y)
    assert len(model._tracking_returns_) == 2
    np.testing.assert_array_equal(model.weights_, weights)
    assert not model._needs_reset


def test_named_benchmark_alignment_and_single_column():
    X = pd.DataFrame([[0.01, 0.02], [0.02, 0.01]], index=[1, 2])
    y = pd.Series([0.015, 0.015], index=X.index)
    model = ExponentiatedGradient(max_tracking_error=0.01)
    with pytest.raises(ValueError, match="aligned indices"):
        model.fit(X, y.iloc[::-1])
    model.fit(X, y.tolist())
    model.fit(X, y.to_frame())
    np.testing.assert_array_equal(model._tracking_benchmark_.array, y)


def test_online_evaluation_routes_benchmark_after_observation():
    from skfolio.model_selection import online_predict

    X = np.random.default_rng(421).normal(0, 0.01, (8, 2))
    y = X @ np.array([0.8, 0.2])
    model = ExponentiatedGradient(max_tracking_error=0.001)
    result = online_predict(model, X, y, warmup_size=2, test_size=1)
    manual = clone(model).fit(X[:2], y[:2])
    expected = []
    for i in range(2, len(X)):
        expected.append(X[i] @ manual.weights_)
        manual.partial_fit(X[i : i + 1], y[i : i + 1])
    np.testing.assert_allclose(result.returns, expected)
