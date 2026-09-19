from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn.model_selection as sks
from sklearn import config_context
from sklearn.exceptions import NotFittedError

from skfolio import RiskMeasure
from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.moments import ImpliedCovariance
from skfolio.optimization import EqualWeighted, MeanRisk, StackingOptimization
from skfolio.prior import EmpiricalPrior, TimeSeriesFactorModel


def test_stacking(X_medium):
    estimators = [
        ("model1", MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ("model2", MeanRisk(risk_measure=RiskMeasure.VARIANCE)),
    ]

    model = StackingOptimization(
        estimators=estimators,
        final_estimator=MeanRisk(),
    )
    model.fit(X_medium)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                3.99739931e-07,
                6.03152100e-07,
                1.86626824e-07,
                4.82851689e-07,
                2.73813779e-07,
                4.13381765e-07,
                6.25186283e-07,
                2.72402456e-01,
                3.10208157e-07,
                1.47969334e-01,
                6.06730174e-07,
                1.78672692e-01,
                4.00826603e-07,
                5.87809472e-07,
                5.35000921e-02,
                4.08781684e-02,
                6.47868626e-07,
                2.61714137e-07,
                2.68921638e-01,
                3.76498188e-02,
            ]
        ),
    )


def test_stacking_factor(X_medium, factors_medium):
    estimators = [
        (
            "model1",
            MeanRisk(
                risk_measure=RiskMeasure.CVAR, prior_estimator=TimeSeriesFactorModel()
            ),
        ),
        ("model2", MeanRisk(risk_measure=RiskMeasure.VARIANCE)),
    ]

    model = StackingOptimization(
        estimators=estimators, final_estimator=MeanRisk(), n_jobs=-1
    )
    model.fit(X_medium, factors=factors_medium)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                3.62263765e-07,
                5.46552549e-07,
                1.69148295e-07,
                4.37563016e-07,
                2.48163560e-07,
                3.74612339e-07,
                5.66544044e-07,
                2.46820960e-01,
                2.81131218e-07,
                1.34073524e-01,
                4.94835562e-07,
                1.61893180e-01,
                3.63248260e-07,
                5.32727893e-07,
                4.84757670e-02,
                3.70391927e-02,
                5.32322322e-07,
                2.37218098e-07,
                3.37578129e-01,
                3.41141002e-02,
            ]
        ),
    )


def test_stacking_cv(X_medium):
    X_test_data = X_medium.iloc[-300:, :10]
    X_train, _X_test = sks.train_test_split(X_test_data, test_size=0.33, shuffle=False)

    estimators = [
        ("model1", MeanRisk(risk_measure=RiskMeasure.CVAR)),
        ("model2", MeanRisk(risk_measure=RiskMeasure.VARIANCE)),
    ]

    model = StackingOptimization(
        estimators=estimators,
        final_estimator=MeanRisk(),
        cv=3,
    )

    model.fit(X_train)

    model2 = StackingOptimization(
        estimators=estimators, final_estimator=MeanRisk(), cv=3, n_jobs=2
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
    assert np.isfinite(model3.weights_).all()

    assert model.get_params(deep=True)
    gs = sks.GridSearchCV(
        estimator=model,
        cv=sks.KFold(n_splits=3, shuffle=False),
        n_jobs=-1,
        param_grid={
            "model1__cvar_beta": [0.95, 0.80],
            "final_estimator__risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CDAR],
        },
    )
    gs.fit(X_train)
    assert gs.best_estimator_ is not None


def test_get_metadata_routing_without_fit():
    # Test that metadata_routing() doesn't raise when called before fit.
    with config_context(enable_metadata_routing=True):
        est = MeanRisk(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )
        model = StackingOptimization(estimators=[("est", est)])
        model.get_metadata_routing()


@pytest.mark.filterwarnings("ignore:The covariance matrix is not positive definite")
def test_metadata_routing_for_stacking_estimators(X_medium, implied_vol_medium):
    """Test that metadata is routed correctly for Stacking*."""
    with config_context(enable_metadata_routing=True):
        est1 = MeanRisk(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        est2 = MeanRisk(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )
        est3 = MeanRisk()
        model = StackingOptimization(
            estimators=[("est1", est1), ("est2", est2), ("est3", est3)],
        )

        model.fit(X_medium, implied_vol=implied_vol_medium)

        model.predict(X_medium)

    for i in range(2):
        # noinspection PyUnresolvedReferences
        assert model.estimators_[
            i
        ].prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)

    # noinspection PyUnresolvedReferences
    assert not hasattr(
        model.estimators_[2].prior_estimator_.covariance_estimator_, "r2_scores_"
    )


@pytest.fixture(scope="module")
def X_tiny():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.0005, 0.01, (60, 6)), columns=list("ABCDEF"))


def test_stacking_named_estimators():
    estimators = [("model1", MeanRisk()), ("model2", EqualWeighted())]
    model = StackingOptimization(estimators=estimators)
    named = model.named_estimators
    assert list(named) == ["model1", "model2"]
    assert named.model2 is estimators[1][1]


@pytest.mark.parametrize("estimators", [None, []])
def test_stacking_invalid_estimators(X_tiny, estimators):
    model = StackingOptimization(estimators=estimators)
    with pytest.raises(ValueError, match="Invalid 'estimators' attribute"):
        model.fit(X_tiny)


def test_stacking_prefit(X_tiny):
    est1 = EqualWeighted().fit(X_tiny)
    est2 = MeanRisk().fit(X_tiny)
    model = StackingOptimization(
        estimators=[("ew", est1), ("mr", est2)],
        final_estimator=EqualWeighted(),
        cv="prefit",
    )
    model.fit(X_tiny)
    assert model.estimators_[0] is est1
    assert model.estimators_[1] is est2
    np.testing.assert_almost_equal(
        model.weights_, 0.5 * est1.weights_ + 0.5 * est2.weights_
    )


def test_stacking_prefit_requires_fitted_estimators(X_tiny):
    model = StackingOptimization(estimators=[("ew", EqualWeighted())], cv="prefit")
    with pytest.raises(NotFittedError):
        model.fit(X_tiny)


def test_stacking_cv_with_y(X_tiny):
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(0.0, 0.01, len(X_tiny)), name="benchmark")
    model = StackingOptimization(
        estimators=[("ew", EqualWeighted()), ("mr", MeanRisk())],
        final_estimator=EqualWeighted(),
        cv=sks.KFold(n_splits=3),
    )
    model.fit(X_tiny, y)
    np.testing.assert_almost_equal(np.sum(model.weights_), 1.0)
