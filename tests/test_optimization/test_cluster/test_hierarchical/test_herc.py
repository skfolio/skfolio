import numpy as np
import pytest
from sklearn import config_context

from skfolio import ExtraRiskMeasure, RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.moments import EWCovariance, ImpliedCovariance
from skfolio.optimization import HierarchicalEqualRiskContribution
from skfolio.prior import EmpiricalPrior, FactorModel


@pytest.fixture(
    scope="module",
    params=[
        x
        for x in [*RiskMeasure, *ExtraRiskMeasure]
        if x not in [ExtraRiskMeasure.SKEW, ExtraRiskMeasure.KURTOSIS]
    ],
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


def test_metadata_routing(X_medium, implied_vol_medium):
    with config_context(enable_metadata_routing=True):
        model = HierarchicalEqualRiskContribution(
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


def test_herc_weight_constraints(X):
    model = HierarchicalEqualRiskContribution(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
    )
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.02584786935368332)
    np.testing.assert_almost_equal(model.weights_[-1], 0.0421175871045949)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Min Weights
    model.set_params(min_weights={"AAPL": 0.05, "XOM": 0.08})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.05)
    np.testing.assert_almost_equal(model.weights_[-1], 0.08)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(min_weights=0.05)
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) * 0.05)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Max Weights
    model.set_params(min_weights=0)
    model.set_params(max_weights={"AAPL": 0.01, "XOM": 0.03})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.01)
    np.testing.assert_almost_equal(model.weights_[-1], 0.03)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(max_weights=0.05)
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) * 0.05)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Both
    model.set_params(min_weights={"AAPL": 0.05}, max_weights={"XOM": 0.03})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.05)
    np.testing.assert_almost_equal(model.weights_[-1], 0.03)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(min_weights=0.03, max_weights=0.06)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.03218923,
                0.05585784,
                0.06,
                0.03,
                0.05336249,
                0.06,
                0.04001747,
                0.05895233,
                0.06,
                0.06,
                0.03674032,
                0.04708938,
                0.03484055,
                0.06,
                0.04486381,
                0.06,
                0.05002204,
                0.03796618,
                0.05914401,
                0.05895433,
            ]
        ),
    )
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)
