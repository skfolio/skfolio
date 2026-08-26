from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.preprocessing import prices_to_returns


@pytest.fixture
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


def test_returns_nan_threshold_ignores_y():
    """Verify target NaNs do not affect the asset missingness threshold."""
    index = pd.date_range("2026-01-01", periods=4)
    asset_prices = pd.DataFrame({"asset": [100.0, 101.0, 102.0, 103.0]}, index=index)
    target_prices = pd.DataFrame(
        {
            "factor_1": [10.0, np.nan, 12.0, 13.0],
            "factor_2": [20.0, np.nan, 22.0, 23.0],
        },
        index=index,
    )

    result = prices_to_returns(
        X=asset_prices,
        y=target_prices,
        nan_threshold=1.0,
        drop_inceptions_nan=False,
        fill_nan=False,
    )
    assert isinstance(result, tuple)
    X, y = result

    expected_X = asset_prices.pct_change(fill_method=None).iloc[1:]
    pd.testing.assert_frame_equal(X, expected_X)
    pd.testing.assert_index_equal(X.index, y.index)


def test_returns_drop_inceptions_nan(prices):
    # Test index_intersect by making the first column mostly 0's
    prices.loc[: prices.index[-10], "AAPL"] = np.nan

    X = prices_to_returns(X=prices)
    assert X.shape[0] == 8

    X = prices_to_returns(X=prices, drop_inceptions_nan=False)
    assert X.shape[0] == prices.shape[0] - 1
