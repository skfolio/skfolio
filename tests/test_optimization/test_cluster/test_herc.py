import numpy as np
import pytest

from skfolio import ExtraRiskMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.moments import EWCovariance
from skfolio.optimization import HierarchicalEqualRiskContribution
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


def test_herc_default(X):
    model = HierarchicalEqualRiskContribution(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        hierarchical_clustering_estimator=HierarchicalClustering(
            max_clusters=5, linkage_method=LinkageMethod.SINGLE
        ),
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.00169182,
                0.67774352,
                0.00156652,
                0.00125186,
                0.00163489,
                0.00141334,
                0.00202476,
                0.00271361,
                0.00178921,
                0.00268109,
                0.00185844,
                0.00227296,
                0.00180695,
                0.00263724,
                0.00218642,
                0.00265416,
                0.16490106,
                0.02059527,
                0.10480936,
                0.00176751,
            ]
        ),
    )


def test_herc_empirical_prior(X):
    model = HierarchicalEqualRiskContribution(
        prior_estimator=EmpiricalPrior(covariance_estimator=EWCovariance())
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.01019466,
                0.01422122,
                0.11426137,
                0.01236976,
                0.00528518,
                0.03575707,
                0.02374812,
                0.13490786,
                0.14723107,
                0.05943483,
                0.04107982,
                0.06934321,
                0.01727373,
                0.10033125,
                0.02942026,
                0.06858159,
                0.00184601,
                0.0638832,
                0.04530751,
                0.00552226,
            ]
        ),
    )


def test_herc_factor_model(X, y):
    model = HierarchicalEqualRiskContribution(
        risk_measure=RiskMeasure.CVAR, prior_estimator=FactorModel()
    )
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.0659256,
                0.04795941,
                0.00749981,
                0.00856635,
                0.00926889,
                0.00853666,
                0.08794308,
                0.05855691,
                0.00845622,
                0.05228904,
                0.05042196,
                0.05768304,
                0.06499639,
                0.04903335,
                0.05404857,
                0.05515687,
                0.04420527,
                0.09262964,
                0.16609219,
                0.01073077,
            ]
        ),
    )


def test_herc(X, linkage_method, risk_measure):
    model = HierarchicalEqualRiskContribution(
        risk_measure=risk_measure,
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=linkage_method
        ),
    )
    model.fit(X)
