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
        np.array([
            1.36123961e-02,
            1.22651861e-07,
            1.57591946e-07,
            3.64502468e-03,
            4.97928490e-02,
            2.03212855e-02,
            1.19578776e-02,
            1.73479863e-01,
            1.97189682e-07,
            1.29586851e-01,
            3.46796741e-02,
            3.47396175e-02,
            2.93451892e-02,
            1.10903622e-01,
            3.18123394e-02,
            1.50276222e-01,
            1.14587375e-02,
            7.53529941e-03,
            1.05551447e-01,
            8.13012271e-02,
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
            1.24490349e-02,
            1.02334125e-03,
            3.92451518e-09,
            4.36056999e-03,
            6.02539916e-02,
            4.86736380e-08,
            2.21039122e-02,
            1.36868295e-01,
            1.72130341e-08,
            1.37342099e-01,
            3.03549592e-02,
            3.40773510e-02,
            3.43468656e-02,
            1.18238505e-01,
            3.15538103e-02,
            1.65849045e-01,
            3.08594247e-03,
            2.60929888e-02,
            7.81770293e-02,
            1.03822190e-01,
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
            1.24585278e-02,
            1.00969765e-03,
            3.92311510e-09,
            4.36377087e-03,
            6.02547103e-02,
            4.86562735e-08,
            2.21111193e-02,
            1.36868668e-01,
            1.72068933e-08,
            1.37341404e-01,
            3.03550420e-02,
            3.40774439e-02,
            3.43466548e-02,
            1.18237907e-01,
            3.15538964e-02,
            1.65848206e-01,
            3.08597928e-03,
            2.60930600e-02,
            7.81704145e-02,
            1.03823428e-01,
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
            1.16214392e-02,
            9.41856056e-04,
            2.39916044e-09,
            4.07056905e-03,
            6.37564987e-02,
            2.97554886e-08,
            2.06254729e-02,
            1.33892561e-01,
            1.05227853e-08,
            1.39348752e-01,
            2.96949942e-02,
            3.33364553e-02,
            3.20389027e-02,
            1.19966043e-01,
            3.08677804e-02,
            1.68272203e-01,
            3.26532537e-03,
            2.55256858e-02,
            7.29181434e-02,
            1.09857275e-01,
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
