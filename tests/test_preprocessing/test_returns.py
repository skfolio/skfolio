import numpy as np
import pytest

from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def prices():
    prices = load_sp500_dataset()
    return prices


@pytest.fixture(scope="module")
def factor_prices():
    factor_prices = load_factors_dataset()
    return factor_prices


def test_returns(prices, factor_prices):
    # insert random nan
    for col in prices.columns:
        prices.loc[prices.sample(frac=0.1).index, col] = np.nan
    p = prices.ffill().dropna()
    X = prices_to_returns(X=prices)
    np.testing.assert_almost_equal(X.to_numpy(), p.pct_change().iloc[1:].to_numpy())

    X = prices_to_returns(X=prices, log_returns=True)
    np.testing.assert_almost_equal(
        X.to_numpy(), np.log(p / p.shift()).iloc[1:].to_numpy()
    )

    X = prices_to_returns(X=prices, nan_threshold=0.01)
    assert X.shape[0] < prices.shape[0] - 2

    X, y = prices_to_returns(X=prices, y=factor_prices)
    assert np.all(X.columns == prices.columns)
    assert np.all(y.columns == factor_prices.columns)
    assert np.all(X.index == y.index)
