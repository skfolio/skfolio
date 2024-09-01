import numpy as np
import pytest
from sklearn import config_context

from skfolio import ExtraRiskMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.moments import EWCovariance, ImpliedCovariance
from skfolio.optimization import HierarchicalRiskParity
from skfolio.prior import EmpiricalPrior, FactorModel


@pytest.fixture(scope="module")
def small_X(X):
    return X[["AAPL", "AMD", "BAC"]]


@pytest.fixture(
    scope="module",
    params=list(RiskMeasure) + list(ExtraRiskMeasure),
)
def risk_measure(request):
    return request.param


def test_hrp_default(X):
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=LinkageMethod.SINGLE
        ),
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.04948508,
                0.03483034,
                0.04967809,
                0.05488211,
                0.02595736,
                0.04482042,
                0.03332249,
                0.08091826,
                0.02840749,
                0.04412388,
                0.06297933,
                0.05717246,
                0.05285248,
                0.07430329,
                0.05499559,
                0.07914531,
                0.03391673,
                0.0396792,
                0.04812428,
                0.0504058,
            ]
        ),
    )


def test_hrp_empirical_prior(X):
    model = HierarchicalRiskParity(
        prior_estimator=EmpiricalPrior(covariance_estimator=EWCovariance())
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.00923669,
                0.00427779,
                0.06213615,
                0.02367877,
                0.03596591,
                0.02150797,
                0.04545975,
                0.1480145,
                0.0800653,
                0.04264111,
                0.06024382,
                0.10169225,
                0.02671939,
                0.07198196,
                0.02399284,
                0.09213609,
                0.00500037,
                0.0520981,
                0.06382561,
                0.02932562,
            ]
        ),
    )


def test_hrp_factor_model(X, y):
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.CVAR, prior_estimator=FactorModel()
    )
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.04674381,
                0.03733258,
                0.04217222,
                0.02712771,
                0.0293525,
                0.02431272,
                0.02907417,
                0.08493361,
                0.046902,
                0.07729156,
                0.0372962,
                0.04266708,
                0.04608497,
                0.07247913,
                0.07839451,
                0.08150043,
                0.02103614,
                0.06087054,
                0.05491043,
                0.05951769,
            ]
        ),
    )


def test_hrp(X, linkage_method, risk_measure):
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=linkage_method
        ),
    )
    model.fit(X)


def test_transaction_costs(X, previous_weights, transaction_costs):
    model = HierarchicalRiskParity(risk_measure=RiskMeasure.WORST_REALIZATION)
    model.fit(X)

    model_tc = HierarchicalRiskParity(
        risk_measure=RiskMeasure.WORST_REALIZATION,
        transaction_costs=transaction_costs * 1000,
        previous_weights=previous_weights,
    )
    model_tc.fit(X)
    assert np.sum(np.abs(model.weights_ - model_tc.weights_)) > 0.1


def test_hrp_small_X(small_X):
    model = HierarchicalRiskParity()
    model.fit(small_X)
    assert model.hierarchical_clustering_estimator_.n_clusters_ == 1


def test_metadata_routing(X_medium, implied_vol_medium):
    with config_context(enable_metadata_routing=True):
        model = HierarchicalRiskParity(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        with pytest.raises(ValueError):
            model.fit(X_medium)

        model.fit(X_medium, implied_vol=implied_vol_medium)

    # noinspection PyUnresolvedReferences
    assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)
