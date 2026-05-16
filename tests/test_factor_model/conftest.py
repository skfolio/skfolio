"""Shared fixtures for factor model tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio._constants import _BENCHMARK_WEIGHTS, _IDIO_RETURNS, _IDIO_VARIANCES
from skfolio.containers import AssetPanel


@pytest.fixture
def simple_panel():
    """Create a simple AssetPanel for testing.

    Contains ``returns``, ``market_cap``, and common fundamentals
    for 20 observations and 5 assets.
    """
    np.random.seed(42)
    n_obs, n_assets = 20, 5

    returns = np.random.randn(n_obs, n_assets) * 0.02
    adj_close = np.abs(np.random.randn(n_obs, n_assets)) * 100 + 10
    market_cap = np.abs(np.random.randn(n_obs, n_assets)) * 1e9 + 1e8
    book_equity = np.abs(np.random.randn(n_obs, n_assets)) * 5e8 + 1e7
    sales_ttm = np.abs(np.random.randn(n_obs, n_assets)) * 2e9 + 1e8
    operating_cash_flow_ttm = np.random.randn(n_obs, n_assets) * 3e8 + 1e8
    net_income_ttm = np.random.randn(n_obs, n_assets) * 2e8 + 5e7
    eps_ntm = np.random.randn(n_obs, n_assets) * 5 + 3
    ebitda_ttm = np.abs(np.random.randn(n_obs, n_assets)) * 4e8 + 5e7
    enterprise_value = market_cap + np.abs(np.random.randn(n_obs, n_assets)) * 3e8
    total_assets = np.abs(np.random.randn(n_obs, n_assets)) * 3e9 + 5e8
    total_equity = np.abs(np.random.randn(n_obs, n_assets)) * 1e9 + 1e8
    cost_of_revenue_ttm = np.abs(np.random.randn(n_obs, n_assets)) * 1e9 + 5e7
    eps_ntm_std = np.abs(np.random.randn(n_obs, n_assets)) * 2 + 0.5
    dividends_ttm = np.abs(np.random.randn(n_obs, n_assets)) * 1e8 + 1e7
    dps_ntm = np.abs(np.random.randn(n_obs, n_assets)) * 3 + 0.5
    net_buybacks_ttm = np.random.randn(n_obs, n_assets) * 5e7
    total_debt = np.abs(np.random.randn(n_obs, n_assets)) * 1e9 + 5e7
    adj_volume = np.abs(np.random.randn(n_obs, n_assets)) * 1e6 + 1e4
    adj_shares_outstanding = np.abs(np.random.randn(n_obs, n_assets)) * 1e8 + 1e7
    short_interest = np.abs(np.random.randn(n_obs, n_assets)) * 5e6 + 1e5
    benchmark_weights = np.ones((n_obs, n_assets)) / n_assets

    panel = AssetPanel(
        fields={
            "returns": returns,
            "adj_close": adj_close,
            "market_cap": market_cap,
            "book_equity": book_equity,
            "sales_ttm": sales_ttm,
            "operating_cash_flow_ttm": operating_cash_flow_ttm,
            "net_income_ttm": net_income_ttm,
            "eps_ntm": eps_ntm,
            "ebitda_ttm": ebitda_ttm,
            "enterprise_value": enterprise_value,
            "total_assets": total_assets,
            "total_equity": total_equity,
            "cost_of_revenue_ttm": cost_of_revenue_ttm,
            "eps_ntm_std": eps_ntm_std,
            "dividends_ttm": dividends_ttm,
            "dps_ntm": dps_ntm,
            "net_buybacks_ttm": net_buybacks_ttm,
            "total_debt": total_debt,
            "adj_volume": adj_volume,
            "adj_shares_outstanding": adj_shares_outstanding,
            "short_interest": short_interest,
        },
        asset_names=np.array([f"asset_{i}" for i in range(n_assets)]),
        observations=np.arange(n_obs),
    )
    panel[_BENCHMARK_WEIGHTS] = benchmark_weights
    return panel


@pytest.fixture
def alpha_deterministic_panel():
    """Create simple panel data with required columns for PredictorAlpha."""
    np.random.seed(123)  # Fixed seed for reproducibility
    n_obs = 20
    n_assets = 4

    observations = pd.bdate_range("2020-01-01", periods=n_obs).to_numpy()
    assets = np.array(["X", "Y", "Z", "W"])

    # Deterministic returns pattern
    base_returns = np.array(
        [
            [0.01, -0.02, 0.015, -0.005],
            [-0.01, 0.025, -0.01, 0.02],
            [0.02, -0.015, 0.005, -0.01],
            [-0.005, 0.01, -0.02, 0.015],
            [0.015, -0.005, 0.02, -0.015],
        ]
    )
    # Tile to get n_obs rows
    returns = np.tile(base_returns, (4, 1))

    # Idio returns = returns + small noise (deterministic with seed)
    idio_returns = returns + np.random.randn(n_obs, n_assets) * 0.005

    # Idio vol: constant per asset
    idio_vol = np.tile([0.02, 0.025, 0.018, 0.022], (n_obs, 1))

    # Signal: cumulative returns (predictive signal)
    signal = np.cumsum(returns, axis=0)
    benchmark_weights = np.ones((n_obs, n_assets)) / n_assets

    panel = AssetPanel(
        fields={
            "returns": returns,
            _IDIO_RETURNS: idio_returns,
            _IDIO_VARIANCES: idio_vol**2,
            "signal": signal,
        },
        observations=observations,
        asset_names=assets,
    )
    panel[_BENCHMARK_WEIGHTS] = benchmark_weights
    return panel
