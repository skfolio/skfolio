import numpy as np
import pytest

from skfolio import ExtraRiskMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.moments import EWCovariance
from skfolio.optimization import HierarchicalRiskParity
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

@pytest.fixture(scope="module")
def small_X(X):
    return X[["AAPL", "AMD", "BAC"]]


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


@pytest.fixture(scope="module")
def previous_weights():
    previous_weights = np.array([
        0.06663786,
        -0.02609581,
        -0.12200097,
        -0.03729676,
        -0.18604607,
        -0.09291357,
        -0.22839449,
        -0.08750029,
        0.01262641,
        0.08712638,
        -0.15731865,
        0.14594815,
        0.11637876,
        0.02163102,
        0.03458678,
        -0.1106219,
        -0.05892651,
        0.05990245,
        -0.08750029,
        0.01262641,
    ])
    return previous_weights


@pytest.fixture(scope="module")
def transaction_costs():
    transaction_costs = np.array([
        3.07368300e-03,
        1.22914659e-02,
        1.31012389e-01,
        5.11069233e-03,
        3.14226164e-03,
        1.38225267e-02,
        1.01730423e-02,
        1.60753223e-02,
        2.16640987e-04,
        1.14058494e-02,
        8.94785339e-03,
        7.30764696e-02,
        1.82260135e-01,
        2.00042452e-01,
        8.56386327e-03,
        1.38225267e-02,
        1.01730423e-02,
        1.01730423e-02,
        1.14058494e-01,
        8.94785339e-03,
    ])
    return transaction_costs


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
        np.array([
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
        ]),
    )


def test_hrp_empirical_prior(X):
    model = HierarchicalRiskParity(
        prior_estimator=EmpiricalPrior(covariance_estimator=EWCovariance())
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
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
        ]),
    )


def test_hrp_factor_model(X, y):
    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.CVAR, prior_estimator=FactorModel()
    )
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
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
        ]),
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
        transaction_costs=transaction_costs,
        previous_weights=previous_weights,
    )
    model_tc.fit(X)
    assert np.sum(np.abs(model.weights_ - model_tc.weights_)) > 0.1


def test_hrp_small_X(small_X):
    model = HierarchicalRiskParity()
    model.fit(small_X)
    assert model.hierarchical_clustering_estimator_.n_clusters_ == 1
