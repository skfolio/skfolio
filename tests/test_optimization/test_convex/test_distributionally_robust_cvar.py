import numpy as np
import pytest
from sklearn import clone, config_context

from skfolio import RiskMeasure
from skfolio.moments import ImpliedCovariance
from skfolio.optimization.convex import (
    DistributionallyRobustCVaR,
    MeanRisk,
    ObjectiveFunction,
)
from skfolio.prior import (
    EmpiricalPrior,
    EntropyPooling,
)


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


def test_optim_with_equal_weighted_sample_weight(X_small):
    """No sample weight and equal-weighted sample weight should give the same result"""
    ref = DistributionallyRobustCVaR()
    ref.fit(X_small)

    model = DistributionallyRobustCVaR(prior_estimator=EntropyPooling())
    model.fit(X_small)

    np.testing.assert_almost_equal(model.weights_, ref.weights_, 6)


@pytest.mark.parametrize(
    "view_params,expected_weights",
    [
        (
            dict(cvar_views=["PG == 0.06"]),
            [
                1.23663e-10,
                4.55651e-11,
                9.95123e-02,
                4.30269e-11,
                9.95123e-02,
                9.95123e-02,
                4.00570e-10,
                9.95123e-02,
                9.95123e-02,
                6.63676e-10,
                9.95123e-02,
                9.95123e-02,
                2.52827e-10,
                3.93614e-09,
                9.95123e-02,
                1.76854e-09,
                3.09124e-10,
                9.95123e-02,
                4.87690e-03,
                9.95123e-02,
            ],
        ),
    ],
)
def test_sample_weight(X_small, view_params, expected_weights):
    ref = DistributionallyRobustCVaR(wasserstein_ball_radius=0.005)
    ref.fit(X_small)

    model = clone(ref)
    model = model.set_params(prior_estimator=EntropyPooling(**view_params))
    model.fit(X_small)

    assert model.weights_[15] < ref.weights_[15]

    np.testing.assert_almost_equal(model.weights_, expected_weights, 5)

    ref_ptf = ref.predict(X_small)
    ptf = model.predict(X_small)

    assert ref_ptf.cvar < ptf.cvar

    sample_weight = model.prior_estimator_.return_distribution_.sample_weight

    ref_ptf.sample_weight = sample_weight
    ptf.sample_weight = sample_weight

    assert ref_ptf.cvar > ptf.cvar
