import numpy as np
import pytest
from sklearn import config_context

from skfolio import RiskMeasure
from skfolio.moments import ImpliedCovariance
from skfolio.optimization.convex import (
    DistributionallyRobustCVaR,
    MeanRisk,
    ObjectiveFunction,
)
from skfolio.prior import EmpiricalPrior


def test_distributionally_robust_cvar(X_small):
    risk_aversion = 2
    model = DistributionallyRobustCVaR(
        budget=1, min_weights=0, wasserstein_ball_radius=0, risk_aversion=risk_aversion
    )

    p1 = model.fit_predict(X_small)
    w1 = model.weights_

    model2 = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=RiskMeasure.CVAR,
        risk_aversion=risk_aversion,
        budget=1,
        min_weights=0,
    )
    model2.fit(X_small)
    w2 = model.weights_
    np.testing.assert_almost_equal(w1, w2, 5)
    # noinspection PyTypeChecker
    model.set_params(wasserstein_ball_radius=1e-2)
    p3 = model.fit_predict(X_small)
    assert p1.mean > p3.mean
    assert p1.cvar < p3.cvar


def test_metadata_routing(X_small, implied_vol_small):
    with config_context(enable_metadata_routing=True):
        model = DistributionallyRobustCVaR(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        with pytest.raises(ValueError):
            model.fit(X_small)

        model.fit(X_small, implied_vol=implied_vol_small)

    # noinspection PyUnresolvedReferences
    assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)
