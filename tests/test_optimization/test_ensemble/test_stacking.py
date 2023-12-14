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
        np.array(
            [
                2.84377724e-06,
                3.29513257e-07,
                2.90518530e-07,
                1.77264271e-06,
                4.57671135e-07,
                1.37098879e-06,
                1.16754284e-02,
                1.83625166e-01,
                4.58439920e-07,
                2.13258384e-01,
                3.00462228e-03,
                1.06216172e-01,
                5.19031283e-07,
                2.80568868e-06,
                8.35849054e-02,
                1.50558385e-01,
                6.07374098e-03,
                8.74059544e-07,
                1.93738020e-01,
                4.82534535e-02,
            ]
        ),
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
        np.array(
            [
                3.28623245e-06,
                3.80780816e-07,
                3.35719184e-07,
                2.04844276e-06,
                5.28877634e-07,
                1.58429757e-06,
                1.25445198e-02,
                1.94443936e-01,
                5.29766964e-07,
                2.21083870e-01,
                1.52459133e-03,
                1.01459465e-01,
                5.99785186e-07,
                3.24218818e-06,
                7.48138742e-02,
                1.44308692e-01,
                3.99667339e-03,
                1.01004500e-06,
                1.92698749e-01,
                5.31120831e-02,
            ]
        ),
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
