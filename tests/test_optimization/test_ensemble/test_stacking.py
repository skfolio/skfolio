import datetime as dt

import numpy as np
import pytest
import sklearn.model_selection as skm

from skfolio import RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.optimization import MeanRisk, StackingOptimization
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


def test_stacking(X):
    estimators = [
        ("model1", MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ("model2", MeanRisk(risk_measure=RiskMeasure.VARIANCE)),
    ]

    model = StackingOptimization(
        estimators=estimators,
        final_estimator=MeanRisk(),
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
            5.38939636e-07,
            3.16420812e-08,
            3.71662947e-08,
            6.56782862e-07,
            5.92517812e-08,
            1.63046256e-07,
            1.16700513e-02,
            1.83449084e-01,
            5.83914715e-08,
            2.13148251e-01,
            3.08632985e-03,
            1.06269323e-01,
            6.66415810e-08,
            3.13343320e-07,
            8.36960758e-02,
            1.50643004e-01,
            6.10966130e-03,
            1.09941897e-07,
            1.93748955e-01,
            4.81772287e-02,
        ]),
    )


def test_stacking_factor(X, y):
    estimators = [
        (
            "model1",
            MeanRisk(risk_measure=RiskMeasure.CVAR, prior_estimator=FactorModel()),
        ),
        ("model2", MeanRisk(risk_measure=RiskMeasure.VARIANCE)),
    ]

    model = StackingOptimization(
        estimators=estimators,
        final_estimator=MeanRisk(),
    )
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
            6.24026750e-07,
            3.65906868e-08,
            4.30063673e-08,
            7.60534823e-07,
            6.85072122e-08,
            1.88782117e-07,
            1.25529582e-02,
            1.94422897e-01,
            6.75780780e-08,
            2.21089337e-01,
            1.59445913e-03,
            1.01442307e-01,
            7.71116094e-08,
            3.61028614e-07,
            7.47964592e-02,
            1.44303370e-01,
            4.00353798e-03,
            1.26943884e-07,
            1.92686697e-01,
            5.31056244e-02,
        ]),
    )


def test_stacking_cv(X):
    X_train, X_test = skm.train_test_split(X, test_size=0.33, shuffle=False)

    estimators = [
        ("model1", MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ("model2", MeanRisk(risk_measure=RiskMeasure.VARIANCE)),
    ]

    model = StackingOptimization(
        estimators=estimators,
        final_estimator=MeanRisk(),
    )

    model.fit(X_train)

    model2 = StackingOptimization(
        estimators=estimators, final_estimator=MeanRisk(), n_jobs=2
    )
    model2.fit(X_train)

    np.testing.assert_almost_equal(model.weights_, model2.weights_)

    model3 = StackingOptimization(
        estimators=estimators,
        final_estimator=MeanRisk(),
        n_jobs=2,
        cv=CombinatorialPurgedCV(),
    )
    model3.fit(X_train)

    assert model.get_params(deep=True)
    gs = skm.GridSearchCV(
        estimator=model,
        cv=skm.KFold(n_splits=5, shuffle=False),
        n_jobs=-1,
        param_grid={
            "model1__cvar_beta": [0.95, 0.80],
            "final_estimator__risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CDAR],
        },
    )
    gs.fit(X_train)
