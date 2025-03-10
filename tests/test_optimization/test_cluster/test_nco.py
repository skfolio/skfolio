import numpy as np
import pytest
import sklearn.cluster as skc
import sklearn.model_selection as sks
from sklearn import config_context

from skfolio import RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.moments import ImpliedCovariance
from skfolio.optimization.cluster import NestedClustersOptimization
from skfolio.optimization.convex import MeanRisk, ObjectiveFunction
from skfolio.population import Population
from skfolio.prior import EmpiricalPrior


def test_nco_hierarchical_clustering(X_medium, linkage_method):
    model = NestedClustersOptimization(
        clustering_estimator=HierarchicalClustering(linkage_method=linkage_method),
    )
    model.fit(X_medium)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_nco_kmeans(X_medium):
    model = NestedClustersOptimization(
        clustering_estimator=skc.KMeans(n_init="auto"),
    )
    model.fit(X_medium)


def test_nco_1(X_medium):
    model = NestedClustersOptimization(
        inner_estimator=MeanRisk(),
        outer_estimator=MeanRisk(),
        clustering_estimator=HierarchicalClustering(
            max_clusters=5, linkage_method=LinkageMethod.SINGLE
        ),
        cv="ignore",
    )
    model.fit(X_medium)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                7.48090405e-08,
                6.65907470e-03,
                1.33799237e-08,
                2.66759969e-07,
                1.87400294e-08,
                2.41895367e-08,
                3.19054689e-06,
                2.60198953e-01,
                2.17543488e-08,
                1.16085432e-01,
                2.78580541e-07,
                1.46806945e-01,
                6.62201210e-08,
                1.35006362e-07,
                5.55266608e-02,
                1.32610690e-01,
                3.98971651e-07,
                2.02099841e-08,
                2.48898039e-01,
                3.32096961e-02,
            ]
        ),
    )


# TODO: remove warnings at the estimator level
@pytest.mark.filterwarnings("ignore:Wrong pattern encountered")
def test_nco_2(X_medium):
    model = NestedClustersOptimization(
        inner_estimator=MeanRisk(linear_constraints=["AAPL>=0.001"]),
    )
    model.fit(X_medium)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                9.90295327e-08,
                6.09420913e-15,
                1.27830085e-14,
                2.79509419e-08,
                4.03265886e-14,
                3.13445245e-08,
                3.06112838e-07,
                2.75858059e-01,
                2.23761523e-07,
                1.78437570e-01,
                9.88372651e-07,
                1.86399772e-01,
                1.74968956e-07,
                8.69295473e-07,
                5.60528828e-02,
                2.71089332e-02,
                4.99439491e-03,
                4.16455429e-07,
                2.71145088e-01,
                1.62253145e-07,
            ]
        ),
    )


def test_nco_array(X_medium):
    model = NestedClustersOptimization()
    model.fit(np.array(X_medium))


def test_nco_cv(X_medium):
    model = NestedClustersOptimization(
        cv=sks.KFold(),
    )
    model.fit(X_medium)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                9.90467618e-08,
                1.02959378e-14,
                1.27835593e-14,
                2.79629675e-08,
                4.03283261e-14,
                3.13458750e-08,
                3.06077945e-07,
                2.75858060e-01,
                2.23771164e-07,
                1.78437570e-01,
                9.88372654e-07,
                1.86399773e-01,
                1.74934573e-07,
                8.69295475e-07,
                5.60528830e-02,
                2.71089333e-02,
                4.99439191e-03,
                4.16455431e-07,
                2.71145089e-01,
                1.62260136e-07,
            ]
        ),
    )

    model2 = NestedClustersOptimization(
        cv=sks.KFold(),
        n_jobs=2,
    )
    model2.fit(X_medium)
    np.testing.assert_almost_equal(
        model.weights_,
        model2.weights_,
    )


def test_nco_combinatorial_cv(X_medium):
    model = NestedClustersOptimization(
        cv=CombinatorialPurgedCV(),
    )
    model.fit(X_medium)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                6.52721585e-07,
                6.78505860e-14,
                2.80817587e-11,
                1.84276923e-07,
                8.85895937e-11,
                6.88577633e-05,
                2.01706424e-06,
                2.75357640e-01,
                4.91560113e-04,
                1.78113876e-01,
                9.86579696e-07,
                1.86061634e-01,
                1.15282488e-06,
                8.67718530e-07,
                5.59512002e-02,
                2.70597563e-02,
                5.87954260e-03,
                4.15699959e-07,
                2.70653218e-01,
                3.56438289e-04,
            ]
        ),
    )

    model2 = NestedClustersOptimization(
        cv=CombinatorialPurgedCV(),
        n_jobs=2,
    )
    model2.fit(X_medium)
    np.testing.assert_almost_equal(
        model.weights_,
        model2.weights_,
    )


def test_nco_train_tests(X_medium):
    X_train, X_test = sks.train_test_split(
        X_medium, test_size=0.33, shuffle=False, random_state=42
    )

    # cv vs no cs
    estimator = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=RiskMeasure.VARIANCE,
    )

    model1 = NestedClustersOptimization(
        inner_estimator=estimator, outer_estimator=estimator, cv="ignore"
    )
    model1.fit(X_train)
    assert model1.score(X_test)
    ptf1 = model1.predict(X_test)

    model2 = NestedClustersOptimization(
        inner_estimator=estimator, outer_estimator=estimator
    )
    model2.fit(X_train)
    assert model2.score(X_test)
    ptf2 = model2.predict(X_test)

    model3 = NestedClustersOptimization(
        inner_estimator=estimator, outer_estimator=estimator, cv=CombinatorialPurgedCV()
    )
    model3.fit(X=X_train)
    assert model3.score(X_test)
    ptf3 = model3.predict(X_test)

    pop = Population([ptf1, ptf2, ptf3])
    assert pop.plot_cumulative_returns()

    assert model2.get_params(deep=True)
    gs = sks.GridSearchCV(
        estimator=model2,
        cv=sks.KFold(n_splits=5, shuffle=False),
        n_jobs=2,
        param_grid={
            "inner_estimator__risk_measure": [
                RiskMeasure.VARIANCE,
                RiskMeasure.SEMI_VARIANCE,
            ],
            "outer_estimator__risk_measure": [RiskMeasure.VARIANCE, RiskMeasure.CDAR],
        },
    )
    gs.fit(X_train)


def test_metadata_routing(X_medium, implied_vol, implied_vol_medium):
    with config_context(enable_metadata_routing=True):
        est = MeanRisk(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )
        model = NestedClustersOptimization(inner_estimator=est)

        with pytest.raises(ValueError):
            model.fit(X_medium)

        with pytest.raises(ValueError):
            model.fit(X_medium, implied_vol=implied_vol)

        model.fit(X_medium, implied_vol=implied_vol_medium)

    # noinspection PyUnresolvedReferences
    assert model.inner_estimators_[
        0
    ].prior_estimator_.covariance_estimator_.r2_scores_.shape == (5,)
