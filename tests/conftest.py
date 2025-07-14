"""conftest module."""

import datetime as dt

import numpy as np
import pytest

from skfolio import RiskMeasure
from skfolio.cluster import LinkageMethod
from skfolio.datasets import (
    load_factors_dataset,
    load_sp500_dataset,
    load_sp500_implied_vol_dataset,
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
    prices = prices.loc[dt.date(2014, 1, 1) :]
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="module")
def y():
    factor_prices = load_factors_dataset()
    factor_prices = factor_prices.loc[dt.date(2014, 1, 1) :]
    y = prices_to_returns(factor_prices)
    return y


@pytest.fixture(scope="module")
def returns(X):
    returns = X[["AAPL"]]
    return returns


@pytest.fixture(scope="module")
def implied_vol():
    implied_vol = load_sp500_implied_vol_dataset()
    implied_vol = implied_vol.loc[dt.date(2014, 1, 3) :]
    return implied_vol


@pytest.fixture(scope="module")
def X_medium(X):
    X_medium = X["2020":]
    return X_medium


@pytest.fixture(scope="module")
def y_medium(y):
    y_medium = y["2020":]
    return y_medium


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
