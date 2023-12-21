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
    prices = prices.loc[dt.date(2020, 1, 1) :]
    factor_prices = load_factors_dataset()
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
            4.14712340e-07,
            6.30177808e-07,
            1.93291519e-07,
            4.99502695e-07,
            2.83295643e-07,
            4.30655193e-07,
            6.47726679e-07,
            2.72459010e-01,
            3.21357278e-07,
            1.47898551e-01,
            6.22014698e-07,
            1.78790660e-01,
            4.15677703e-07,
            6.08486611e-07,
            5.34806929e-02,
            4.07012545e-02,
            6.55843498e-07,
            2.70811727e-07,
            2.68970614e-01,
            3.76932238e-02,
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
        estimators=estimators, final_estimator=MeanRisk(), n_jobs=-1
    )
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
            3.75619933e-07,
            5.70711164e-07,
            1.75092940e-07,
            4.52393785e-07,
            2.56615594e-07,
            3.90043142e-07,
            5.86634418e-07,
            2.46726487e-01,
            2.91072041e-07,
            1.33930284e-01,
            5.12046309e-07,
            1.61904465e-01,
            3.76494088e-07,
            5.51161003e-07,
            4.84295934e-02,
            3.68571303e-02,
            5.42851438e-07,
            2.45331929e-07,
            3.38013445e-01,
            3.41332699e-02,
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
