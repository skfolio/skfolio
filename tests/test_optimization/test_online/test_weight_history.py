"""Streaming history preserves numerical results without repeated full copies."""

import pickle
from copy import deepcopy

import numpy as np
import pytest

from skfolio.optimization.online import ExponentiatedGradient


def test_history_grows_geometrically_and_matches_block_updates():
    returns = np.random.default_rng(261).normal(0, 0.01, (65, 3))
    streamed = ExponentiatedGradient(store_weights=True)
    buffers = []
    for row in returns:
        streamed.partial_fit(row[None, :])
        if not buffers or buffers[-1] is not streamed._weight_history_buffer_:
            buffers.append(streamed._weight_history_buffer_)
    assert [len(buffer) for buffer in buffers] == [16, 32, 64, 128]
    batch = ExponentiatedGradient(store_weights=True).fit(returns)
    np.testing.assert_allclose(streamed.all_weights_, batch.all_weights_)
    assert streamed.all_weights_.shape == returns.shape
    streamed.fit(returns[:2])
    assert streamed.all_weights_.shape == (2, 3)
    np.testing.assert_allclose(streamed.all_weights_, batch.all_weights_[:2])


@pytest.mark.parametrize(
    "serialize", [deepcopy, lambda model: pickle.loads(pickle.dumps(model))]
)
def test_copied_history_can_continue_without_changing_source(serialize):
    returns = np.random.default_rng(41).normal(0, 0.01, (21, 3))
    source = ExponentiatedGradient(store_weights=True).fit(returns[:5])
    history = source.all_weights_.copy()
    restored = serialize(source)
    restored.partial_fit(returns[5:])
    reference = ExponentiatedGradient(store_weights=True).fit(returns)
    np.testing.assert_allclose(restored.all_weights_, reference.all_weights_)
    np.testing.assert_array_equal(source.all_weights_, history)


def test_failed_block_preserves_successful_history(monkeypatch):
    model = ExponentiatedGradient(store_weights=True)
    update = model._update_one
    calls = 0

    def fail_second(row):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("failed observation")
        update(row)

    monkeypatch.setattr(model, "_update_one", fail_second)
    with pytest.raises(RuntimeError, match="failed observation"):
        model.partial_fit([[0.01, 0.02], [0.03, -0.01]])
    assert model.all_weights_.shape == (1, 2)
    assert model._needs_reset
