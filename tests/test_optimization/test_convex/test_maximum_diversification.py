import datetime as dt

import numpy as np
import pytest

from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.optimization.convex import MaximumDiversification
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import FactorModel


@pytest.fixture(scope="module")
def X_y():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2018, 1, 1) :]
    factor_prices = load_factors_dataset()
    factor_prices = factor_prices.loc[dt.date(2018, 1, 1) :]
    X, y = prices_to_returns(X=prices, y=factor_prices)
    return X, y


@pytest.fixture(scope="module")
def X(X_y):
    return X_y[0]


@pytest.fixture(scope="module")
def y(X_y):
    return X_y[1]


def test_maximum_diversification(X):
    model = MaximumDiversification()
    model.fit(X)
    ptf = model.predict(X)
    diversification = model.problem_values_["expected_return"] / np.sqrt(
        model.problem_values_["risk"]
    )
    np.testing.assert_almost_equal(ptf.diversification, diversification, 3)


def test_maximum_diversification_factor(X, y):
    model = MaximumDiversification(prior_estimator=FactorModel())
    model.fit(X, y)
    ptf = model.predict(X)
    diversification = model.problem_values_["expected_return"] / np.sqrt(
        model.problem_values_["risk"]
    )
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(ptf.diversification, diversification, 3)
