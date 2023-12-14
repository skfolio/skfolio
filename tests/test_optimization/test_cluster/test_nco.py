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
    return X


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


def test_nco_hierarchical_clustering(X, linkage_method):
    model = NestedClustersOptimization(
        clustering_estimator=HierarchicalClustering(linkage_method=linkage_method),
    )
    model.fit(X)


def test_nco_kmeans(X):
    model = NestedClustersOptimization(
        clustering_estimator=skc.KMeans(),
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
        np.array(
            [
                1.36204652e-02,
                5.73048550e-07,
                4.95508951e-08,
                3.67252658e-03,
                4.98268546e-02,
                2.03300425e-02,
                1.19674442e-02,
                1.73406108e-01,
                1.01024022e-07,
                1.29537590e-01,
                3.46786126e-02,
                3.47415643e-02,
                2.93637645e-02,
                1.10858115e-01,
                3.18184246e-02,
                1.50220161e-01,
                1.13923845e-02,
                7.60830699e-03,
                1.05607706e-01,
                8.13492053e-02,
            ]
        ),
    )


def test_nco_2(X):
    model = NestedClustersOptimization(
        inner_estimator=MeanRisk(linear_constraints=["AAPL>=0.001"]),
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                1.24568919e-02,
                1.00660387e-03,
                2.06959172e-08,
                4.36081112e-03,
                6.02088192e-02,
                2.57276311e-07,
                2.21172751e-02,
                1.36872625e-01,
                9.10312270e-08,
                1.37318071e-01,
                3.03437966e-02,
                3.40701364e-02,
                3.43654216e-02,
                1.18173123e-01,
                3.15415503e-02,
                1.65949744e-01,
                3.09068431e-03,
                2.60863141e-02,
                7.81880002e-02,
                1.03849763e-01,
            ]
        ),
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
        np.array(
            [
                1.24452068e-02,
                1.03864607e-03,
                2.06980819e-08,
                4.36133346e-03,
                6.02083546e-02,
                2.57303220e-07,
                2.20998754e-02,
                1.36872123e-01,
                9.10407482e-08,
                1.37318430e-01,
                3.03436851e-02,
                3.40700112e-02,
                3.43459347e-02,
                1.18173431e-01,
                3.15414344e-02,
                1.65950177e-01,
                3.09066046e-03,
                2.60862183e-02,
                7.82051484e-02,
                1.03848962e-01,
            ]
        ),
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
        np.array(
            [
                1.16416722e-02,
                9.71585068e-04,
                1.46512564e-07,
                4.07974052e-03,
                6.37711421e-02,
                1.82133565e-06,
                2.06729795e-02,
                1.33920610e-01,
                6.44437175e-07,
                1.39122270e-01,
                2.96893534e-02,
                3.33353250e-02,
                3.21283623e-02,
                1.19725779e-01,
                3.08612744e-02,
                1.68130129e-01,
                3.27354815e-03,
                2.55236947e-02,
                7.31557712e-02,
                1.09994152e-01,
            ]
        ),
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
