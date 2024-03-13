import numpy as np
import pytest

from skfolio import ExtraRiskMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.moments import EWCovariance
from skfolio.optimization import (
    HierarchicalRiskParity,
    MeanRisk,
    SchurComplementaryAllocation,
)
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior, FactorModel


@pytest.fixture(scope="module")
def X_y():
    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()
    X, y = prices_to_returns(X=prices, y=factor_prices)
    return X, y


@pytest.fixture(scope="module")
def X(X_y):
    return X_y[0]


@pytest.fixture(scope="module")
def y(X_y):
    return X_y[1]


@pytest.fixture(
    scope="module",
    params=list(LinkageMethod),
)
def linkage_method(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=list(RiskMeasure) + list(ExtraRiskMeasure),
)
def risk_measure(request):
    return request.param


def test_schur_default(X):
    model = SchurComplementaryAllocation()
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.00290673,
                0.00080765,
                0.00947158,
                0.02841936,
                0.02198382,
                0.05570452,
                0.0727225,
                0.10366611,
                0.01190645,
                0.06332227,
                0.07082402,
                0.11164499,
                0.00610297,
                0.06378797,
                0.04052787,
                0.11672992,
                0.00708368,
                0.0310986,
                0.14080058,
                0.0404884,
            ]
        ),
    )


def test_schur_hrp_min_var(X):
    hrp = HierarchicalRiskParity()
    min_var = MeanRisk()
    schur_0 = SchurComplementaryAllocation(gamma=0)
    schur_05 = SchurComplementaryAllocation(gamma=0.5)

    ptf_hrp = hrp.fit_predict(X)
    ptf_min_var = min_var.fit_predict(X)
    ptf_schur_0 = schur_0.fit_predict(X)
    ptf_schur_05 = schur_05.fit_predict(X)

    np.testing.assert_array_almost_equal(ptf_hrp.weights, ptf_schur_0.weights)

    assert ptf_hrp.variance > ptf_min_var.variance
    assert ptf_hrp.mean > ptf_min_var.mean

    assert ptf_schur_05.variance > ptf_min_var.variance
    assert ptf_schur_05.mean > ptf_min_var.mean

    assert ptf_schur_05.variance < ptf_hrp.variance
    assert ptf_schur_05.mean < ptf_hrp.mean


def test_schur_frontier(X):
    prev_ptf = None
    for gamma in np.linspace(0, 0.7, 50):
        schur = SchurComplementaryAllocation(gamma=gamma)
        ptf = schur.fit_predict(X)
        if prev_ptf is not None:
            assert ptf.variance < prev_ptf.variance
            assert ptf.mean < prev_ptf.mean
        prev_ptf = ptf


def test_schur_empirical_prior(X):
    model = SchurComplementaryAllocation(
        prior_estimator=EmpiricalPrior(covariance_estimator=EWCovariance())
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                1.85170271e-04,
                9.11546502e-05,
                1.51946606e-01,
                8.24774798e-03,
                9.14795270e-02,
                9.36807937e-03,
                7.75445408e-03,
                9.70469517e-02,
                1.41814048e-01,
                2.85882437e-02,
                7.49046401e-02,
                1.53485794e-01,
                6.00625896e-04,
                6.55605288e-02,
                1.67444978e-02,
                6.20626545e-02,
                5.51153123e-03,
                3.25614228e-02,
                2.65538500e-02,
                2.54924720e-02,
            ]
        ),
    )


def test_schur_factor_model(X, y):
    model = SchurComplementaryAllocation(prior_estimator=FactorModel())
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                4.15629632e-02,
                5.17333290e-03,
                1.28044356e-03,
                8.18308555e-03,
                1.51901220e-02,
                7.67274438e-04,
                4.15165100e-02,
                1.54964187e-01,
                4.28966350e-02,
                1.23134172e-01,
                2.59949447e-02,
                4.41419429e-02,
                4.92108313e-02,
                1.27388339e-01,
                8.80510879e-02,
                8.83112162e-02,
                1.48928229e-04,
                4.59360721e-02,
                5.40522238e-02,
                4.20956885e-02,
            ]
        ),
    )


def test_schur_linkage(X, linkage_method, risk_measure):
    model = SchurComplementaryAllocation(
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=linkage_method
        ),
    )
    model.fit(X)
