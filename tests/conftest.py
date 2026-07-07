"""conftest module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio import RiskMeasure
from skfolio._constants import _BENCHMARK_WEIGHTS
from skfolio.cluster import LinkageMethod
from skfolio.containers import AssetPanel
from skfolio.datasets import (
    load_factors_dataset,
    load_sp500_dataset,
    load_sp500_implied_vol_dataset,
    make_synthetic_characteristics,
)
from skfolio.preprocessing import prices_to_returns


def pytest_configure(config):
    # globally turn off scientific notation in every test session
    np.set_printoptions(suppress=True, precision=6)


@pytest.fixture
def random_data():
    """Fixture that returns a random numpy array in [0,1] of shape (100, 2)."""
    rng = np.random.default_rng(seed=42)
    return rng.random((100, 2))


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    prices = prices.loc[pd.Timestamp(2014, 1, 1) :]
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="module")
def factors():
    factor_prices = load_factors_dataset()
    factor_prices = factor_prices.loc[pd.Timestamp(2014, 1, 1) :]
    return prices_to_returns(factor_prices)


@pytest.fixture(scope="module")
def returns(X):
    returns = X[["AAPL"]]
    return returns


@pytest.fixture(scope="module")
def implied_vol():
    implied_vol = load_sp500_implied_vol_dataset()
    implied_vol = implied_vol.loc[pd.Timestamp(2014, 1, 3) :]
    return implied_vol


@pytest.fixture(scope="module")
def X_medium(X):
    X_medium = X["2020":]
    return X_medium


@pytest.fixture(scope="module")
def factors_medium(factors):
    return factors["2020":]


@pytest.fixture(scope="module")
def X_small(X):
    X_small = X["2022":]
    return X_small


@pytest.fixture(scope="module")
def implied_vol_medium(implied_vol):
    implied_vol_medium = implied_vol["2020":]
    return implied_vol_medium


@pytest.fixture(scope="module")
def implied_vol_small(implied_vol):
    implied_vol_medium = implied_vol["2022":]
    return implied_vol_medium


@pytest.fixture(
    scope="module",
    params=[rm for rm in RiskMeasure if not rm.is_annualized],
)
def risk_measure(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=list(LinkageMethod),
)
def linkage_method(request):
    return request.param


@pytest.fixture
def simple_panel():
    """Create a simple `AssetPanel` for testing.

    Contains `returns`, `market_cap`, and common fundamentals for 20 observations and
    5 assets.
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
def make_characteristics_panel():
    """Create synthetic characteristics panels."""

    def _make_characteristics_panel(
        *,
        n_assets: int = 100,
        n_observations: int = 252,
        random_state: int = 42,
        missing_ratio: float = 0.0,
        delisting_proba: float = 0.0,
        late_listing_proba: float = 0.0,
        ffill_market_cap: bool = False,
        bfill_market_cap: bool = False,
    ) -> AssetPanel:
        panel = make_synthetic_characteristics(
            n_assets=n_assets,
            n_observations=n_observations,
            random_state=random_state,
            missing_ratio=missing_ratio,
            delisting_proba=delisting_proba,
            late_listing_proba=late_listing_proba,
        )
        if ffill_market_cap:
            panel.ffill("market_cap")
        if bfill_market_cap:
            panel.bfill("market_cap")
        return panel

    return _make_characteristics_panel


@pytest.fixture
def panel_data(make_characteristics_panel):
    """Synthetic panel with NaN holes, late listings, and delistings."""
    return make_characteristics_panel(
        n_assets=20,
        n_observations=126,
        random_state=42,
        missing_ratio=0.01,
        delisting_proba=0.1,
        late_listing_proba=0.1,
        ffill_market_cap=True,
    )


@pytest.fixture
def clean_panel_data(make_characteristics_panel):
    """Synthetic panel without listing gaps or random missing values."""
    return make_characteristics_panel(
        n_assets=10,
        n_observations=76,
        random_state=123,
    )


@pytest.fixture
def dense_characteristics_panel(make_characteristics_panel):
    """Synthetic characteristics panel with a dense active mask."""
    return make_characteristics_panel(
        n_assets=100,
        n_observations=252,
        random_state=42,
        ffill_market_cap=True,
        bfill_market_cap=True,
    )


@pytest.fixture(scope="module")
def previous_weights():
    return np.array(
        [
            0.06663786,
            -0.02609581,
            -0.12200097,
            -0.03729676,
            -0.18604607,
            -0.09291357,
            -0.22839449,
            -0.08750029,
            0.01262641,
            0.08712638,
            -0.15731865,
            0.14594815,
            0.11637876,
            0.02163102,
            0.03458678,
            -0.1106219,
            -0.05892651,
            0.05990245,
            -0.08750029,
            0.01262641,
        ]
    )


@pytest.fixture(scope="module")
def transaction_costs():
    return np.array(
        [
            1.35823376e-06,
            5.43149178e-06,
            5.78932342e-05,
            2.25837045e-06,
            1.38853806e-06,
            6.10805422e-06,
            4.49537883e-06,
            7.10354498e-06,
            9.57317662e-08,
            5.04014556e-06,
            3.95397852e-06,
            3.22918558e-05,
            8.05391670e-05,
            8.83970181e-05,
            3.78429663e-06,
            6.10805422e-06,
            4.49537883e-06,
            4.49537883e-06,
            5.04014556e-05,
            3.95397852e-06,
        ]
    )


@pytest.fixture(scope="module")
def groups():
    return [
        ["Equity"] * 3 + ["Fund"] * 5 + ["Bond"] * 12,
        ["US"] * 2 + ["Europe"] * 8 + ["Japan"] * 10,
    ]


@pytest.fixture(scope="module")
def groups_dict():
    return {
        "AAPL": ["Equity", "US"],
        "AMD": ["Equity", "US"],
        "BAC": ["Equity", "Europe"],
        "BBY": ["Fund", "Europe"],
        "CVX": ["Fund", "Europe"],
        "GE": ["Fund", "Europe"],
        "HD": ["Bond", "Europe"],
        "JNJ": ["Bond", "Europe"],
        "JPM": ["Bond", "Europe"],
        "KO": ["Bond", "Europe"],
        "LLY": ["Bond", "Japan"],
        "MRK": ["Bond", "Japan"],
        "MSFT": ["Bond", "Japan"],
        "PEP": ["Bond", "Japan"],
        "PFE": ["Bond", "Japan"],
        "PG": ["Bond", "Japan"],
        "RRC": ["Bond", "Japan"],
        "UNH": ["Bond", "Japan"],
        "WMT": ["Bond", "Japan"],
        "XOM": ["Bond", "Japan"],
    }


@pytest.fixture(scope="module")
def linear_constraints():
    return [
        "Equity <= 0.5 * Bond",
        "US >= 0.1",
        "Europe >= 0.5 * Fund",
        "Japan <= 1",
    ]
