import datetime as dt

import numpy as np
import pytest
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.utils.bootstrap import stationary_bootstrap


@pytest.fixture(scope="module")
def returns():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2017, 1, 1) :]
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
