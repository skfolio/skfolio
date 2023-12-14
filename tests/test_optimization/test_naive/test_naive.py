import datetime as dt

import numpy as np
import pytest

from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.optimization.naive import EqualWeighted, InverseVolatility, Random
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import FactorModel


@pytest.fixture(scope="module")
def X_y():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2014, 1, 1) :]
    factor_prices = load_factors_dataset()
    factor_prices = factor_prices.loc[dt.date(2014, 1, 1) :]
    X, y = prices_to_returns(X=prices, y=factor_prices)
    return X, y


@pytest.fixture(scope="module")
def X(X_y):
    return X_y[0]


@pytest.fixture(scope="module")
def y(X_y):
    return X_y[1]


def test_inverse_volatility(X, y):
    model = InverseVolatility()
    model.fit(X)
    np.testing.assert_almost_equal(sum(model.weights_), 1)
    w = 1 / np.std(np.asarray(X), axis=0)
    w /= sum(w)
    np.testing.assert_almost_equal(model.weights_, w)

    model = InverseVolatility(prior_estimator=FactorModel())
    model.fit(X, y)


def test_equal_weighted(X):
    model = EqualWeighted()
    model.fit(X)
    weights = model.weights_
    np.testing.assert_almost_equal(sum(weights), 1)
    w = 1 / X.shape[1]
    np.testing.assert_almost_equal(weights, w)


def test_random(X):
    model = Random()
    weights = model.fit(X).weights_
    np.testing.assert_almost_equal(sum(weights), 1)
