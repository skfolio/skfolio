"""Test Scorer module."""

import numpy as np
import sklearn.model_selection as sks

from skfolio import RatioMeasure, RiskMeasure
from skfolio.metrics import make_scorer
from skfolio.optimization import MeanRisk, ObjectiveFunction


def test_default_score(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)
    grid_search = sks.GridSearchCV(
        estimator=model, cv=cv, n_jobs=-1, param_grid={"l2_coef": l2_coefs}
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.sharpe_ratio
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_ratio(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)

    # ratio measure
    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(RatioMeasure.CDAR_RATIO),
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cdar_ratio
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_risk_measure(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)
    # risk measure
    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(RiskMeasure.CVAR),
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cvar
        res[f"split{i}_test_score"] = -d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_custom(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)

    # Custom
    def custom(prediction):
        return prediction.cvar - 2 * prediction.cdar

    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(custom),
    )

    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cvar - 2 * pred.cdar
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )
