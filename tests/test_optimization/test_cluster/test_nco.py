import datetime as dt

import numpy as np
import pytest
import sklearn.cluster as skc
import sklearn.model_selection as skm
from skfolio import RiskMeasure
from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import CombinatorialPurgedCV
from skfolio.optimization.cluster import NestedClustersOptimization
from skfolio.optimization.convex import MeanRisk, ObjectiveFunction
from skfolio.population import Population
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices)
    X = X[2020:]
    return X


@pytest.fixture(scope="module")
def X2(X):
    return X[2022:]


@pytest.fixture(
    scope="module",
    params=list(LinkageMethod),
)
def linkage_method(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[rm for rm in RiskMeasure if not rm.is_annualized],
)
def risk_measure(request):
    return request.param


def test_nco_hierarchical_clustering(X2, linkage_method):
    model = NestedClustersOptimization(
        clustering_estimator=HierarchicalClustering(linkage_method=linkage_method),
    )
    model.fit(X2)


def test_nco_kmeans(X):
    model = NestedClustersOptimization(
        clustering_estimator=skc.KMeans(n_init="auto"),
    )
    model.fit(X)


def test_nco_1(X):
    model = NestedClustersOptimization(
        inner_estimator=MeanRisk(),
        outer_estimator=MeanRisk(),
        clustering_estimator=HierarchicalClustering(
            max_clusters=5, linkage_method=LinkageMethod.SINGLE
        ),
        cv="ignore",
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
            1.16623894e-02,
            6.56162853e-07,
            3.38449210e-08,
            5.42940423e-07,
            3.10070110e-02,
            1.03539586e-02,
            6.99658791e-03,
            2.27835418e-01,
            4.01358494e-08,
            1.44958851e-01,
            2.29672618e-02,
            3.10982667e-02,
            2.75689634e-02,
            1.23399338e-01,
            3.61583289e-02,
            1.44129384e-01,
            3.65095139e-03,
            1.14280366e-02,
            1.04433176e-01,
            6.23508037e-02,
        ]),
    )


def test_nco_2(X):
    model = NestedClustersOptimization(
        inner_estimator=MeanRisk(linear_constraints=["AAPL>=0.001"]),
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
            1.39380159e-02,
            6.70347749e-08,
            7.68906415e-10,
            1.11551820e-03,
            3.47779594e-02,
            5.44774443e-08,
            2.04997265e-02,
            2.21222594e-01,
            2.48494504e-08,
            1.39564537e-01,
            2.02086877e-02,
            2.80797059e-02,
            3.97896332e-02,
            1.19999376e-01,
            3.22916078e-02,
            1.39987403e-01,
            2.78434215e-04,
            2.88437111e-02,
            9.41843278e-02,
            6.52186149e-02,
        ]),
    )


def test_nco_array(X):
    model = NestedClustersOptimization()
    model.fit(np.array(X))


def test_nco_cv(X):
    model = NestedClustersOptimization(
        cv=skm.KFold(),
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
            1.39400185e-02,
            9.45988919e-07,
            7.69039602e-10,
            1.11291529e-03,
            3.47780004e-02,
            5.44868807e-08,
            2.05015713e-02,
            2.21227680e-01,
            2.48537548e-08,
            1.39567746e-01,
            2.02091524e-02,
            2.80803515e-02,
            3.97843265e-02,
            1.20002135e-01,
            3.22923502e-02,
            1.39990622e-01,
            2.78434543e-04,
            2.88443743e-02,
            9.41706045e-02,
            6.52186917e-02,
        ]),
    )

    model2 = NestedClustersOptimization(
        cv=skm.KFold(),
        n_jobs=2,
    )
    model2.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        model2.weights_,
    )


def test_nco_combinatorial_cv(X):
    model = NestedClustersOptimization(
        cv=CombinatorialPurgedCV(),
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array([
            1.32934659e-02,
            9.02112969e-07,
            1.19847489e-09,
            1.06129712e-03,
            3.25014951e-02,
            8.49126084e-08,
            1.95506872e-02,
            2.25598174e-01,
            3.87322070e-08,
            1.42324996e-01,
            2.06083971e-02,
            2.86350968e-02,
            3.79390881e-02,
            1.22372854e-01,
            3.29303062e-02,
            1.42756226e-01,
            2.60208719e-04,
            2.94142133e-02,
            8.98028740e-02,
            6.09495936e-02,
        ]),
    )

    model2 = NestedClustersOptimization(
        cv=CombinatorialPurgedCV(),
        n_jobs=2,
    )
    model2.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        model2.weights_,
    )


def test_nco_train_tests(X):
    X = X.loc[dt.date(2014, 1, 1) :]
    X_train, X_test = skm.train_test_split(
        X, test_size=0.33, shuffle=False, random_state=42
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
    gs = skm.GridSearchCV(
        estimator=model2,
        cv=skm.KFold(n_splits=5, shuffle=False),
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
