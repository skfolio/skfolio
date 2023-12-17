import datetime as dt

import numpy as np
import pytest
from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization.convex import (
    DistributionallyRobustCVaR,
    MeanRisk,
    ObjectiveFunction,
)
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2022, 1, 1) :]
    X = prices_to_returns(X=prices)
    return X


def test_distributionally_robust_cvar(X):
    risk_aversion = 2
    model = DistributionallyRobustCVaR(
        budget=1, min_weights=0, wasserstein_ball_radius=0, risk_aversion=risk_aversion
    )

    p1 = model.fit_predict(X)
    w1 = model.weights_

    model2 = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=RiskMeasure.CVAR,
        risk_aversion=risk_aversion,
        budget=1,
        min_weights=0,
    )
    model2.fit(X)
    w2 = model.weights_
    np.testing.assert_almost_equal(w1, w2, 5)
    # noinspection PyTypeChecker
    model.set_params(wasserstein_ball_radius=1e-2)
    p3 = model.fit_predict(X)
    assert p1.mean > p3.mean
    assert p1.cvar < p3.cvar
