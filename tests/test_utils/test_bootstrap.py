from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.utils.bootstrap import stationary_bootstrap


@pytest.fixture(scope="module")
def returns():
    prices = load_sp500_dataset()
    prices = prices.loc[pd.Timestamp(2017, 1, 1) :]
    X = prices_to_returns(X=prices)
    returns = np.array(X)
    return returns


def test_stationary_bootstrap(returns):
    res = stationary_bootstrap(
        returns=returns,
        n_bootstrap_samples=20,
        block_size=5,
    )

    assert res.shape == (20, 1507, 20)


@pytest.mark.parametrize("block_size", [3, None])
def test_stationary_bootstrap_accepts_dataframe(block_size):
    """DataFrames produce the same samples as arrays without modifying the input."""
    returns = pd.DataFrame(
        np.random.default_rng(7).normal(size=(64, 3)), columns=["A", "B", "C"]
    )
    before = returns.copy()
    expected = stationary_bootstrap(
        returns.to_numpy(), 20, block_size=block_size, seed=42
    )

    actual = stationary_bootstrap(returns, 20, block_size=block_size, seed=42)

    np.testing.assert_array_equal(actual, expected)
    pd.testing.assert_frame_equal(returns, before)


@pytest.mark.parametrize("seed", [42, None])
def test_stationary_bootstrap_does_not_change_global_random_state(seed):
    returns = np.arange(12, dtype=float).reshape(6, 2)
    previous_state = np.random.get_state()
    try:
        np.random.seed(7)
        expected_next = np.random.random()
        np.random.seed(7)
        samples = stationary_bootstrap(
            returns, n_bootstrap_samples=3, block_size=2, seed=seed
        )
        assert np.random.random() == expected_next
        if seed == 42:
            np.testing.assert_array_equal(
                samples[:, :, 0],
                [
                    [6, 8, 10, 0, 8, 2],
                    [4, 4, 4, 6, 6, 4],
                    [10, 8, 10, 6, 8, 10],
                ],
            )
    finally:
        np.random.set_state(previous_state)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
@pytest.mark.parametrize("block_size", [1, 3, np.inf])
@pytest.mark.parametrize(
    ("n_observations", "n_bootstrap_samples"),
    [(1, 1), (8, 0), (8, 1), (8, 4), (32, 20)],
)
def test_stationary_bootstrap_preserves_samples_and_draws(
    dtype, block_size, n_observations, n_bootstrap_samples
):
    """Dense samples retain the circular row schedule, dtype and RNG state."""
    returns = np.arange(n_observations * 3).reshape(n_observations, 3).astype(dtype)
    before = returns.copy()
    returns.setflags(write=False)
    rng = np.random.RandomState(42)
    starts = rng.randint(n_observations, size=(n_bootstrap_samples, n_observations))
    continues = rng.rand(n_bootstrap_samples, n_observations) >= 1 / block_size
    expected = np.empty((n_bootstrap_samples, n_observations, 3), dtype=dtype)
    # A scalar circular walk independently checks block continuation and wraparound.
    for sample in range(n_bootstrap_samples):
        row = starts[sample, 0]
        for observation in range(n_observations):
            if observation:
                row = (
                    row + 1
                    if continues[sample, observation]
                    else starts[sample, observation]
                )
            expected[sample, observation] = before[row % n_observations]

    expected_state = np.random.get_state()
    actual = stationary_bootstrap(
        returns, n_bootstrap_samples, block_size=block_size, seed=42
    )

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype
    actual_state = np.random.get_state()
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[0] == expected_state[0]
    assert actual_state[2:] == expected_state[2:]
    np.testing.assert_array_equal(returns, before)
    assert not returns.flags.writeable
