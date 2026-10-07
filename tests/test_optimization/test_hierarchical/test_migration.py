"""Compatibility checks for the hierarchical optimizer package migration."""

import importlib
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn import clone, config_context
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.utils.validation import validate_data

from skfolio.cluster import HierarchicalClustering, LinkageMethod
from skfolio.measures import RiskMeasure
from skfolio.optimization import BaseOptimization
from skfolio.optimization.cluster.hierarchical._base import BaseHierarchicalOptimization
from skfolio.optimization.hierarchical import (
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    NestedClustersOptimization,
    SchurComplementary,
)
from skfolio.prior import EmpiricalPrior
from skfolio.seriation import HierarchicalSeriation


@pytest.fixture
def returns():
    return pd.DataFrame(
        np.random.default_rng(16).normal(0.001, 0.01, (60, 4)), columns=list("abcd")
    )


@pytest.mark.parametrize(
    "module_name",
    ["skfolio.optimization.cluster", "skfolio.optimization.cluster.hierarchical"],
)
def test_deprecated_optimizer_imports(module_name):
    module = importlib.import_module(module_name)
    estimators = [
        BaseHierarchicalOptimization,
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
        SchurComplementary,
    ]
    if module_name == "skfolio.optimization.cluster":
        estimators.append(NestedClustersOptimization)
    for expected in estimators:
        name = expected.__name__
        with pytest.warns(FutureWarning, match=f"{name}.*2.0") as caught:
            assert getattr(module, name) is expected
        assert caught[0].filename == __file__
        assert "skfolio.optimization." in str(caught[0].message)
        namespace = {}
        with pytest.warns(FutureWarning, match=f"{name}.*2.0"):
            exec(f"from {module_name} import {name}", namespace)
        assert namespace[name] is expected

    namespace = {}
    with pytest.warns(FutureWarning, match="deprecated.*2.0"):
        exec(f"from {module_name} import *", namespace)
    for expected in estimators:
        assert namespace[expected.__name__] is expected


def test_deprecated_top_level_base():
    import skfolio.optimization as optimization

    with pytest.warns(FutureWarning, match="BaseHierarchicalOptimization.*2.0"):
        assert optimization.BaseHierarchicalOptimization is BaseHierarchicalOptimization
    with pytest.warns(FutureWarning, match="BaseHierarchicalOptimization.*2.0"):
        from skfolio.optimization import BaseHierarchicalOptimization as legacy_base
    assert legacy_base is BaseHierarchicalOptimization


def test_preferred_imports_and_construction_are_quiet():
    # A fresh process also checks the internal imports during package initialization.
    subprocess.run(
        [
            sys.executable,
            "-W",
            "error::FutureWarning",
            "-c",
            (
                "from skfolio import optimization\n"
                "from skfolio.optimization import hierarchical\n"
                "import sys\n"
                "assert 'skfolio.optimization.cluster.hierarchical._base' not in sys.modules\n"
                "for name in hierarchical.__all__:\n"
                "    estimator = getattr(hierarchical, name)\n"
                "    assert estimator is getattr(optimization, name)\n"
                "    estimator()\n"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "estimator",
    [
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
        NestedClustersOptimization,
        SchurComplementary,
    ],
)
def test_optimizers_are_independent_of_legacy_base(estimator):
    assert issubclass(estimator, BaseOptimization)
    assert not issubclass(estimator, BaseHierarchicalOptimization)
    assert not isinstance(estimator(), BaseHierarchicalOptimization)


class CustomHierarchicalOptimization(BaseHierarchicalOptimization):
    def __init__(self, min_weights=0.0, max_weights=1.0):
        super().__init__(min_weights=min_weights, max_weights=max_weights)

    def fit(self, X, y=None):
        validate_data(self, X)
        distribution = EmpiricalPrior().fit(X).return_distribution_
        self.bounds_ = self._convert_weights_bounds(self.n_features_in_)
        self.asset_risks_ = self._unitary_risks(distribution)
        self.weights_ = np.full(self.n_features_in_, 1 / self.n_features_in_)
        self.risk_ = self._risk(self.weights_, distribution)
        return self


def test_legacy_custom_subclass(returns):
    model = clone(CustomHierarchicalOptimization(min_weights={"a": 0.1})).fit(returns)
    np.testing.assert_array_equal(model.bounds_[0], [0.1, 0, 0, 0])
    np.testing.assert_array_equal(model.bounds_[1], np.ones(4))
    np.testing.assert_allclose(model.asset_risks_, returns.var().to_numpy())
    assert model.risk_ == pytest.approx(model.predict(returns).variance)


@pytest.mark.parametrize("estimator", [HierarchicalRiskParity, SchurComplementary])
@pytest.mark.parametrize("legacy", [False, True])
def test_nested_clustering_parameter_search(returns, estimator, legacy):
    clustering = HierarchicalClustering()
    if legacy:
        model = estimator(hierarchical_clustering_estimator=clustering)
        parameter = "hierarchical_clustering_estimator__linkage_method"
    else:
        model = estimator(
            seriation_estimator=HierarchicalSeriation(
                hierarchical_clustering_estimator=clustering
            )
        )
        parameter = (
            "seriation_estimator__hierarchical_clustering_estimator__linkage_method"
        )
    search = GridSearchCV(
        model,
        {parameter: [LinkageMethod.SINGLE, LinkageMethod.WARD]},
        cv=KFold(2),
        error_score="raise",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore" if legacy else "error", FutureWarning)
        search.fit(returns)
    fitted = (
        search.best_estimator_.seriation_estimator_.hierarchical_clustering_estimator_
    )
    assert fitted.linkage_method == search.best_params_[parameter]
    assert model.get_params()[parameter] == LinkageMethod.WARD


class RecordingClustering(HierarchicalClustering):
    def fit(self, X, y=None, groups=None):
        self.groups_ = groups
        return super().fit(X, y)


@pytest.mark.parametrize("estimator", [HierarchicalRiskParity, SchurComplementary])
def test_seriation_clustering_metadata(returns, estimator):
    with config_context(enable_metadata_routing=True):
        clustering = RecordingClustering().set_fit_request(groups=True)
        model = clone(
            estimator(
                seriation_estimator=HierarchicalSeriation(
                    hierarchical_clustering_estimator=clustering
                )
            )
        )
        groups = np.array([0, 0, 1, 1])
        model.fit(returns, groups=groups)
    fitted = model.seriation_estimator_.hierarchical_clustering_estimator_
    np.testing.assert_array_equal(fitted.groups_, groups)


@pytest.mark.parametrize(
    "estimator, args, expected",
    [
        (
            HierarchicalRiskParity,
            (RiskMeasure.CVAR, None, None, None, 0.01, 0.8),
            {"risk_measure": RiskMeasure.CVAR, "min_weights": 0.01, "max_weights": 0.8},
        ),
        (
            HierarchicalEqualRiskContribution,
            (RiskMeasure.CVAR, None, None, None, 0.01, 0.8, "CLARABEL"),
            {"risk_measure": RiskMeasure.CVAR, "min_weights": 0.01, "max_weights": 0.8},
        ),
        (
            SchurComplementary,
            (0.7, False, None, None, None, 0.01, 0.8),
            {
                "gamma": 0.7,
                "keep_monotonic": False,
                "min_weights": 0.01,
                "max_weights": 0.8,
            },
        ),
        (
            NestedClustersOptimization,
            (None, None, None, None, "ignore"),
            {"cv": "ignore"},
        ),
    ],
)
def test_positional_parameters(estimator, args, expected):
    params = clone(estimator(*args)).get_params()
    for name, value in expected.items():
        assert params[name] == value
