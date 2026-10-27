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
