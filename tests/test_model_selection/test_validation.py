"""Test Validation module."""

from __future__ import annotations

import pickle
import warnings
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import sklearn.model_selection as sks
import sklearn.utils as sku
from sklearn import config_context
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from skfolio import FailedPortfolio, MultiPeriodPortfolio, Population
from skfolio.measures import PerfMeasure, RiskMeasure
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.model_selection._validation import _route_params
from skfolio.moments import (
    EWCovariance,
    ImpliedCovariance,
)
from skfolio.optimization import (
    EqualWeighted,
    InverseVolatility,
    MeanRisk,
    ObjectiveFunction,
)
from skfolio.optimization._base import BaseOptimization
from skfolio.pre_selection import SelectKExtremes
from skfolio.prior import EmpiricalPrior


def assert_weights_dict_subset_equal(d1: dict, d2: dict, tol: float = 1e-15) -> None:
    """True iff for every key k in d2 d1.get(k, 0.0) matches d2[k] within tol."""
    for k, b in d2.items():
        assert abs(d1.get(k, 0.0) - b) < tol


class PreviousWeightsAwareOptimization(BaseOptimization):
    """Optimization estimator whose weights reveal previous weights and scale."""

    def __init__(
        self,
        portfolio_params: dict | None = None,
        fallback=None,
        previous_weights=None,
        raise_on_failure: bool = True,
        scale: float = 1.0,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            previous_weights=previous_weights,
            raise_on_failure=raise_on_failure,
        )
        self.scale = scale

    @property
    def needs_previous_weights(self) -> bool:
        return True

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        n_assets = X_arr.shape[1]
        self.n_features_in_ = n_assets
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        previous_weights = self._previous_weights_array(X, n_assets)
        increment = np.zeros(n_assets)
        increment[0] = self.scale
        self.weights_ = previous_weights + increment
        return self

    def _previous_weights_array(self, X, n_assets: int):
        if self.previous_weights is None:
            return np.zeros(n_assets)
        if np.isscalar(self.previous_weights):
            return np.full(n_assets, float(self.previous_weights))
        if isinstance(self.previous_weights, dict):
            return np.asarray(
                [self.previous_weights.get(asset, 0.0) for asset in X.columns],
                dtype=float,
            )
        return np.asarray(self.previous_weights, dtype=float)


class FailingFixedOptimization(BaseOptimization):
    """Fixed allocation that can fail for selected training-window lengths."""

    def __init__(
        self,
        fail_on_n_observations: tuple[int, ...] = (),
        portfolio_params: dict | None = None,
        previous_weights=None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            previous_weights=previous_weights,
            raise_on_failure=raise_on_failure,
        )
        self.fail_on_n_observations = fail_on_n_observations

    @property
    def needs_previous_weights(self) -> bool:
        return True

    def fit(self, X, y=None):
        if len(X) in self.fail_on_n_observations:
            raise RuntimeError("forced failure")
        self.n_features_in_ = X.shape[1]
        weights = np.zeros(X.shape[1])
        weights[:2] = [0.6, 0.4]
        self.weights_ = weights
        return self


class PopulationOptimization(BaseOptimization):
    """Optimization used to exercise the 2D-weights prediction path."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        weights = np.zeros(X.shape[1])
        weights[:2] = [0.6, 0.4]
        self.weights_ = np.vstack((weights, weights[::-1]))
        return self


def test_validation(X):
    model = MeanRisk()
    n_observations = X.shape[0]
    for cv in [
        sks.KFold(),
        WalkForward(test_size=n_observations // 5, train_size=n_observations // 5),
    ]:
        pred = cross_val_predict(
            model, X, cv=cv, portfolio_params=dict(name="ptf_test")
        )

        pred2 = MultiPeriodPortfolio()
        for train, test in cv.split(X):
            model.fit(X.take(train))
            pred2.append(model.predict(X.take(test)))

        assert isinstance(pred, MultiPeriodPortfolio)
        assert pred.name == "ptf_test"
        assert np.array_equal(pred.returns_df.index, pred2.returns_df.index)
        np.testing.assert_almost_equal(np.asarray(pred), np.asarray(pred2))

        assert len(pred.portfolios) == cv.get_n_splits(X)


def test_validation_combinatorial(X):
    model = MeanRisk()
    n_observations = X.shape[0]
    cv = CombinatorialPurgedCV()

    pred = cross_val_predict(model, X, cv=cv, portfolio_params=dict(name="test"))

    cv.split(X)
    cv.get_path_ids()

    assert isinstance(pred, Population)
    assert len(pred) == cv.n_test_paths
    for p in pred:
        assert isinstance(p, MultiPeriodPortfolio)
        assert len(p.portfolios) == cv.n_folds
        assert len(p) == cv.n_folds
        assert p.n_observations == n_observations


def test_meta_data_routing_cross_validation(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = InverseVolatility(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        cv = KFold()

        _ = cross_val_predict(model, X, params={"implied_vol": implied_vol}, cv=cv)


def test_route_params_partial_fit_error_message(X):
    active_mask = np.ones(X.shape, dtype=bool)

    with config_context(enable_metadata_routing=True):
        with pytest.raises(
            Exception,
            match="online_score",
        ) as exc_info:
            _route_params(
                EWCovariance(),
                params={"active_mask": active_mask},
                owner="online_score",
                callee="partial_fit",
            )

    message = str(exc_info.value)
    assert "set_partial_fit_request" in message
    assert "set_fit_request" not in message


def test_route_params_picklable_without_metadata():
    # `process_routing` returns a private placeholder that cannot be pickled when no
    # metadata is passed, which breaks the process-based parallel paths.
    with config_context(enable_metadata_routing=True):
        routed_params = _route_params(
            EWCovariance(),
            owner="online_score",
            callee="partial_fit",
            cv=KFold(2),
        )

    assert isinstance(routed_params, sku.Bunch)
    assert set(routed_params) == {"estimator_params", "splitter"}
    assert routed_params.estimator_params == {}
    assert routed_params.splitter.split == {}

    restored = pickle.loads(pickle.dumps(routed_params))
    assert restored.estimator_params == {}
    assert restored.splitter.split == {}


def test_route_params_picklable_with_metadata(X):
    active_mask = np.ones(X.shape, dtype=bool)

    with config_context(enable_metadata_routing=True):
        routed_params = _route_params(
            EWCovariance().set_partial_fit_request(active_mask=True),
            params={"active_mask": active_mask},
            owner="online_score",
            callee="partial_fit",
            cv=KFold(2),
        )

    assert isinstance(routed_params, sku.Bunch)
    assert set(routed_params) == {"estimator_params", "splitter"}
    np.testing.assert_array_equal(
        routed_params.estimator_params["active_mask"], active_mask
    )
    assert routed_params.splitter.split == {}

    restored = pickle.loads(pickle.dumps(routed_params))
    np.testing.assert_array_equal(restored.estimator_params["active_mask"], active_mask)
    assert restored.splitter.split == {}


def test_cross_val_predict_non_portfolio_estimator_raises(X):
    model = ImpliedCovariance()

    with pytest.raises(
        TypeError,
        match=(r"skfolio's `cross_val_predict` only supports"),
    ):
        cross_val_predict(model, X, cv=KFold())


def test_optim_with_previous_weights_walk_forward(X):
    cv = WalkForward(test_size=300, train_size=400)

    ref = MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY)
    assert ref.needs_previous_weights is False
    pred_ref = cross_val_predict(ref, X, cv=cv)

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY, transaction_costs=0.001
    )
    assert model.needs_previous_weights is True
    pred = cross_val_predict(model, X, cv=cv)

    assert abs((pred_ref.composition - pred.composition)["MeanRisk_5"].sum()) > 0.2

    assert np.all(pred[0].previous_weights == 0)

    for i in range(1, len(pred)):
        np.testing.assert_almost_equal(pred[i - 1].weights, pred[i].previous_weights)
        assert_weights_dict_subset_equal(
            pred[i - 1].weights_dict, pred[i].previous_weights_dict
        )


def test_entry_rebalancing_params_walk_forward(X):
    cv = WalkForward(test_size=300, train_size=400)
    pred = cross_val_predict(
        PreviousWeightsAwareOptimization(),
        X,
        cv=cv,
        entry_rebalancing_params={"scale": 2.0},
    )

    assert len(pred) >= 2
    first_expected = np.zeros(X.shape[1])
    first_expected[0] = 2.0
    second_expected = first_expected.copy()
    second_expected[0] = 3.0

    np.testing.assert_array_equal(pred[0].weights, first_expected)
    np.testing.assert_array_equal(pred[1].previous_weights, first_expected)
    np.testing.assert_array_equal(pred[1].weights, second_expected)


def test_entry_rebalancing_params_rejects_non_sequential_cv(X):
    with pytest.raises(ValueError, match="entry_rebalancing_params"):
        cross_val_predict(
            MeanRisk(),
            X,
            cv=KFold(),
            entry_rebalancing_params={"max_weights": 0.1},
        )


def test_pipeline_with_previous_weights_walk_forward(X):
    cv = WalkForward(test_size=300, train_size=400)

    pipe_ref = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=10)),
            ("optim", MeanRisk(ObjectiveFunction.MAXIMIZE_UTILITY)),
        ]
    )

    pipe = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=10)),
            (
                "optim",
                MeanRisk(ObjectiveFunction.MAXIMIZE_UTILITY, transaction_costs=0.01),
            ),
        ]
    )

    with config_context(transform_output="pandas"):
        pred_ref = cross_val_predict(pipe_ref, X, cv=cv)
        pred = cross_val_predict(pipe, X, cv=cv)

    assert abs((pred_ref.composition - pred.composition)["MeanRisk_5"].sum()) > 0.2

    assert np.all(pred[0].previous_weights == 0)

    for i in range(1, len(pred)):
        assert not np.allclose(pred[i].previous_weights, 0)
        assert_weights_dict_subset_equal(
            pred[i - 1].weights_dict, pred[i].previous_weights_dict
        )


def test_pipeline_with_previous_weights_walk_forward_initial_pre_w(X):
    cv = WalkForward(test_size=300, train_size=400)
    previous_weights = {name: 0.2 for name in X.columns}

    pipe = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=10)),
            (
                "optim",
                MeanRisk(
                    ObjectiveFunction.MAXIMIZE_UTILITY,
                    transaction_costs=0.01,
                    previous_weights=previous_weights,
                ),
            ),
        ]
    )

    with config_context(transform_output="pandas"):
        pred = cross_val_predict(pipe, X, cv=cv)

    prev_w = pred[0].previous_weights
    assert np.all(prev_w == 0.2)
    for i in range(1, len(pred)):
        assert not np.allclose(pred[i].previous_weights, 0)
        assert_weights_dict_subset_equal(
            pred[i - 1].weights_dict, pred[i].previous_weights_dict
        )


def test_pipeline_with_previous_weights_multiple_randomized_cv(X):
    cv = MultipleRandomizedCV(
        walk_forward=WalkForward(test_size=300, train_size=400),
        n_subsamples=5,
        asset_subset_size=5,
        window_size=1200,
        random_state=0,
    )

    pipe_ref = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=10)),
            ("optim", MeanRisk(ObjectiveFunction.MAXIMIZE_UTILITY)),
        ]
    )

    pipe = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=10)),
            (
                "optim",
                MeanRisk(ObjectiveFunction.MAXIMIZE_UTILITY, transaction_costs=1e-20),
            ),
        ]
    )

    pipe_tc = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=10)),
            (
                "optim",
                MeanRisk(
                    ObjectiveFunction.MAXIMIZE_UTILITY,
                    transaction_costs=0.001,
                    previous_weights={name: 0.1 for name in X.columns},
                ),
            ),
        ]
    )

    with config_context(transform_output="pandas"):
        pred_ref = cross_val_predict(pipe_ref, X, cv=cv)
        pred = cross_val_predict(pipe, X, cv=cv)
        pred_tc = cross_val_predict(pipe_tc, X, cv=cv)

    assert abs(pred_ref.composition() - pred.composition()).sum().sum() < 1e-3
    assert abs(pred_ref.composition() - pred_tc.composition()).sum().sum() > 7

    for mpp in pred_tc:
        assert np.all(mpp[0].previous_weights == 0.1)
        for i in range(1, len(mpp)):
            assert not np.allclose(mpp[i].previous_weights, 0.1)
            assert_weights_dict_subset_equal(
                mpp[i - 1].weights_dict, mpp[i].previous_weights_dict
            )


def test_fallback_previous_weights_propagation(X):
    cv = WalkForward(test_size=300, train_size=400)
    ref = MeanRisk(
        min_weights=1,
        fallback=MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        ),
    )
    assert ref.needs_previous_weights is False

    model = MeanRisk(
        min_weights=1,
        fallback=MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            transaction_costs=0.001,
        ),
    )
    assert model.needs_previous_weights is True

    pred_ref = cross_val_predict(ref, X, cv=cv)
    pred = cross_val_predict(model, X, cv=cv)

    assert abs((pred_ref.composition - pred.composition)["MeanRisk_5"]).sum() > 0.5

    assert np.all(pred[0].previous_weights == 0)
    for i in range(1, len(pred)):
        assert not np.allclose(pred[i].previous_weights, 0)
        np.testing.assert_almost_equal(pred[i - 1].weights, pred[i].previous_weights)
        assert_weights_dict_subset_equal(
            pred[i - 1].weights_dict, pred[i].previous_weights_dict
        )


def test_weight_drift_function_routing_and_propagation(X):
    model = InverseVolatility()
    assert model.needs_previous_weights is False
    cv = WalkForward(test_size=300, train_size=400)

    pred = cross_val_predict(
        model,
        X,
        cv=cv,
        portfolio_params={"weight_drift": True, "compounded": True},
    )

    assert pred.compounded is True
    assert model.portfolio_params is None
    assert len(pred) >= 2
    assert all(portfolio.weight_drift for portfolio in pred)
    assert all(portfolio.compounded for portfolio in pred)
    for previous, current in pairwise(pred):
        np.testing.assert_allclose(current.previous_weights, previous.ending_weights)


def test_measurement_params_inherit_from_estimator(X):
    measurement_params = {
        "compounded": True,
        "risk_free_rate": 0.001,
        "annualization_factor": 12,
        "fitness_measures": [PerfMeasure.ANNUALIZED_MEAN, RiskMeasure.CVAR],
        "min_acceptable_return": -0.002,
        "value_at_risk_beta": 0.91,
        "entropic_risk_measure_theta": 2.0,
        "entropic_risk_measure_beta": 0.92,
        "cvar_beta": 0.93,
        "evar_beta": 0.94,
        "drawdown_at_risk_beta": 0.95,
        "cdar_beta": 0.96,
        "edar_beta": 0.97,
    }
    model = InverseVolatility(portfolio_params=measurement_params.copy())

    pred = cross_val_predict(model, X.iloc[:60, :3], cv=KFold(n_splits=3))

    for param, expected in measurement_params.items():
        assert getattr(pred, param) == expected
        assert all(getattr(portfolio, param) == expected for portfolio in pred)
    assert model.portfolio_params == measurement_params


def test_function_measurement_params_override_estimator_and_resolve_none(X):
    model = InverseVolatility(
        portfolio_params={
            "compounded": False,
            "risk_free_rate": 0.001,
            "annualization_factor": 12,
            "fitness_measures": [PerfMeasure.ANNUALIZED_MEAN],
        }
    )
    portfolio_params = {
        "compounded": True,
        "risk_free_rate": 0.002,
        "annualization_factor": None,
        "fitness_measures": None,
    }

    pred = cross_val_predict(
        model,
        X.iloc[:60, :3],
        cv=KFold(n_splits=3),
        portfolio_params=portfolio_params,
    )

    assert pred.compounded is True
    assert pred.risk_free_rate == 0.002
    assert pred.annualization_factor == 252
    assert pred.fitness_measures == [PerfMeasure.MEAN, RiskMeasure.VARIANCE]
    for portfolio in pred:
        assert portfolio.compounded is True
        assert portfolio.risk_free_rate == 0.002
        assert portfolio.annualization_factor == 252
        assert portfolio.fitness_measures == [PerfMeasure.MEAN, RiskMeasure.VARIANCE]
    assert portfolio_params == {
        "compounded": True,
        "risk_free_rate": 0.002,
        "annualization_factor": None,
        "fitness_measures": None,
    }
    assert model.portfolio_params == {
        "compounded": False,
        "risk_free_rate": 0.001,
        "annualization_factor": 12,
        "fitness_measures": [PerfMeasure.ANNUALIZED_MEAN],
    }


def test_aggregate_only_params_are_not_forwarded_to_children(X):
    X_small = X.iloc[:60, :3]
    sample_weight = np.full(len(X_small), 1 / len(X_small))

    pred = cross_val_predict(
        InverseVolatility(),
        X_small,
        cv=KFold(n_splits=3),
        portfolio_params={
            "name": "aggregate",
            "tag": "evaluation",
            "sample_weight": sample_weight,
            "check_observations_order": True,
        },
    )

    assert pred.name == "aggregate"
    assert pred.tag == "evaluation"
    assert pred.check_observations_order is True
    np.testing.assert_array_equal(pred.sample_weight, sample_weight)
    assert all(portfolio.name == "InverseVolatility" for portfolio in pred)
    assert all(portfolio.tag is None for portfolio in pred)
    assert all(portfolio.sample_weight is None for portfolio in pred)


@pytest.mark.parametrize(
    ("estimator_params", "evaluation_params", "expected"),
    [
        ({"annualized_factor": 365}, {"annualization_factor": 12}, 12),
        ({"annualization_factor": 12}, {"annualized_factor": 365}, 365),
    ],
)
def test_annualized_factor_alias_resolution(
    X, estimator_params, evaluation_params, expected
):
    model = InverseVolatility(portfolio_params=estimator_params)

    with pytest.warns(FutureWarning, match="annualized_factor"):
        pred = cross_val_predict(
            model,
            X.iloc[:60, :3],
            cv=KFold(n_splits=3),
            portfolio_params=evaluation_params,
        )

    assert pred.annualization_factor == expected
    assert all(portfolio.annualization_factor == expected for portfolio in pred)
    assert model.portfolio_params == estimator_params


@pytest.mark.parametrize(
    ("estimator_params", "evaluation_params"),
    [
        (
            None,
            {"annualization_factor": 12, "annualized_factor": 365},
        ),
        (
            {"annualization_factor": 12, "annualized_factor": 365},
            None,
        ),
    ],
)
def test_annualized_factor_alias_conflict_is_preserved(
    X, estimator_params, evaluation_params
):
    with pytest.raises(ValueError, match="pass only `annualization_factor`"):
        cross_val_predict(
            InverseVolatility(portfolio_params=estimator_params),
            X.iloc[:60, :3],
            cv=KFold(n_splits=3),
            portfolio_params=evaluation_params,
        )


@pytest.mark.parametrize(
    ("portfolio_params", "expected"),
    [(None, 0.001), ({"risk_free_rate": 0.002}, 0.002)],
)
def test_risk_free_rate_estimator_precedence(X, portfolio_params, expected):
    model = MeanRisk(
        risk_free_rate=0.001,
        portfolio_params=portfolio_params,
    )

    pred = cross_val_predict(model, X.iloc[:60, :3], cv=KFold(n_splits=2))

    assert pred.risk_free_rate == expected
    assert all(portfolio.risk_free_rate == expected for portfolio in pred)


def test_measurement_params_resolve_when_all_folds_fail(X):
    model = FailingFixedOptimization(
        fail_on_n_observations=(5, 10, 15),
        raise_on_failure=False,
        portfolio_params={"compounded": True, "annualization_factor": 12},
    )

    with pytest.warns(UserWarning, match="forced failure"):
        pred = cross_val_predict(
            model,
            X.iloc[:20, :3],
            cv=sks.TimeSeriesSplit(n_splits=3),
        )

    assert all(isinstance(portfolio, FailedPortfolio) for portfolio in pred)
    assert pred.compounded is True
    assert pred.annualization_factor == 12
    assert all(portfolio.compounded for portfolio in pred)
    assert all(portfolio.annualization_factor == 12 for portfolio in pred)


def test_measurement_params_do_not_trigger_sequential_path(X):
    model = InverseVolatility(portfolio_params={"compounded": True})
    assert model.needs_previous_weights is False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred = cross_val_predict(
            model,
            X,
            cv=WalkForward(test_size=300, train_size=400),
            n_jobs=2,
            portfolio_params={"risk_free_rate": 0.001},
        )

    assert not any("sequential processing" in str(w.message) for w in caught)
    assert pred.compounded is True
    assert pred.risk_free_rate == 0.001
    assert all(portfolio.risk_free_rate == 0.001 for portfolio in pred)


def test_multiple_randomized_paths_receive_measurement_params(X):
    cv = MultipleRandomizedCV(
        walk_forward=WalkForward(test_size=300, train_size=400),
        n_subsamples=3,
        asset_subset_size=5,
        window_size=1200,
        random_state=0,
    )
    prediction = cross_val_predict(
        InverseVolatility(portfolio_params={"annualization_factor": 12}),
        X,
        cv=cv,
        portfolio_params={"compounded": True, "risk_free_rate": 0.001},
    )

    assert isinstance(prediction, Population)
    assert len(prediction) == 3
    for path in prediction:
        assert path.compounded is True
        assert path.risk_free_rate == 0.001
        assert path.annualization_factor == 12
        for portfolio in path:
            assert portfolio.compounded is True
            assert portfolio.risk_free_rate == 0.001
            assert portfolio.annualization_factor == 12


def test_weight_drift_function_routing_through_pipeline(X):
    model = Pipeline(
        [
            (
                "optimization",
                InverseVolatility(portfolio_params={"compounded": True}),
            )
        ]
    )
    pred = cross_val_predict(
        model,
        X,
        cv=WalkForward(test_size=300, train_size=400),
        portfolio_params={"weight_drift": True},
    )

    assert model[-1].portfolio_params == {"compounded": True}
    assert pred.compounded is True
    assert all(portfolio.compounded for portfolio in pred)
    assert all(portfolio.weight_drift for portfolio in pred)


def test_weight_drift_function_value_overrides_estimator(X):
    model = InverseVolatility(portfolio_params={"weight_drift": True})
    pred = cross_val_predict(
        model,
        X,
        cv=KFold(n_splits=3),
        portfolio_params={"weight_drift": False},
    )

    assert model.portfolio_params == {"weight_drift": True}
    assert all(not portfolio.weight_drift for portfolio in pred)


def test_weight_drift_uses_executed_turnover_for_costs(X):
    transaction_cost = 0.001
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        transaction_costs=transaction_cost,
    )
    cv = WalkForward(test_size=300, train_size=400)
    default = cross_val_predict(model, X, cv=cv)
    drifted = cross_val_predict(
        model, X, cv=cv, portfolio_params={"weight_drift": True}
    )

    for previous, current in pairwise(default):
        np.testing.assert_allclose(current.previous_weights, previous.ending_weights)
    for previous, current in pairwise(drifted):
        np.testing.assert_allclose(current.previous_weights, previous.ending_weights)
    np.testing.assert_allclose(
        [portfolio.total_cost for portfolio in default],
        transaction_cost * default.turnover,
    )
    np.testing.assert_allclose(
        [portfolio.total_cost for portfolio in drifted],
        transaction_cost * drifted.turnover,
    )


@pytest.mark.parametrize("weight_drift", [False, True])
def test_failed_fold_keeps_last_successful_weights(X, weight_drift):
    X_small = X.iloc[:20, :3]
    model = FailingFixedOptimization(
        fail_on_n_observations=(10,),
        raise_on_failure=False,
        portfolio_params={"weight_drift": weight_drift},
    )
    with pytest.warns(UserWarning, match="forced failure"):
        pred = cross_val_predict(model, X_small, cv=sks.TimeSeriesSplit(n_splits=3))

    assert isinstance(pred[1], FailedPortfolio)
    np.testing.assert_allclose(pred[2].previous_weights, pred[0].ending_weights)


def test_population_prediction_forwards_weight_drift(X):
    model = PopulationOptimization(portfolio_params={"weight_drift": True}).fit(X)
    population = model.predict(X.iloc[:5])

    assert isinstance(population, Population)
    assert all(portfolio.weight_drift for portfolio in population)


def test_combinatorial_population_routes_weight_drift(X):
    prediction = cross_val_predict(
        InverseVolatility(),
        X.iloc[:120, :3],
        cv=CombinatorialPurgedCV(n_folds=4, n_test_folds=2),
        portfolio_params={"weight_drift": True, "compounded": True},
    )

    assert isinstance(prediction, Population)
    assert all(path.compounded for path in prediction)
    assert all(portfolio.compounded for path in prediction for portfolio in path)
    assert all(portfolio.weight_drift for path in prediction for portfolio in path)


def test_sequential_population_prediction_raises_clear_error(X):
    with pytest.raises(ValueError, match="estimator returned a Population"):
        cross_val_predict(
            PopulationOptimization(),
            X.iloc[:30],
            cv=sks.TimeSeriesSplit(n_splits=3),
            portfolio_params={"weight_drift": True},
        )


@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("input_type", ["named", "numeric_columns", "array"])
@pytest.mark.parametrize(
    "cv",
    [
        WalkForward(train_size=20, test_size=10),
        sks.TimeSeriesSplit(n_splits=3),
        MultipleRandomizedCV(
            walk_forward=WalkForward(train_size=20, test_size=10),
            n_subsamples=2,
            asset_subset_size=2,
            random_state=0,
        ),
    ],
)
def test_target_turnover_without_costs(cv, n_jobs, input_type):
    X = pd.DataFrame(
        np.random.default_rng(0).normal(0, 0.01, (60, 4)), columns=list("ABCD")
    )
    if input_type == "array":
        X = X.to_numpy()
    elif input_type == "numeric_columns":
        X.columns = range(4)
    model = EqualWeighted()
    with warnings.catch_warnings(record=True) as caught:
        pred = cross_val_predict(model, X, cv=cv, n_jobs=n_jobs)
    assert not any("sequential processing" in str(w.message) for w in caught)
    for path in pred if isinstance(pred, Population) else [pred]:
        np.testing.assert_allclose(path.turnover, [1.0, *np.zeros(len(path) - 1)])
        for previous, current in pairwise(path):
            np.testing.assert_array_equal(current.previous_weights, previous.weights)
    assert model.previous_weights is None
    assert model.portfolio_params is None


@pytest.mark.parametrize("output", ["global", "pipeline"])
@pytest.mark.parametrize("weight_drift", [False, True])
@pytest.mark.parametrize(
    "costs, expected_cost",
    [
        (0.0, 0.0),
        (0.001, 0.002),
        ({"A": 0.002, "B": 0.001}, 0.003),
        ({"A": 0.002, "B": 0.0}, 0.002),
    ],
)
def test_pipeline_asset_replacement_counts_both_trades(
    output, weight_drift, costs, expected_cost
):
    X = pd.DataFrame(
        [
            [0.08, 0],
            [0.11, 0.01],
            [0.09, -0.01],
            [0.12, 0],
            [0, 0.08],
            [0.01, 0.11],
            [-0.01, 0.09],
            [0, 0.12],
            [0, 0.08],
            [0.01, 0.11],
            [-0.01, 0.09],
            [0, 0.12],
        ],
        columns=["A", "B"],
    )
    pipe = Pipeline(
        [
            ("select", SelectKExtremes(k=1, measure=PerfMeasure.MEAN)),
            ("optimization", MeanRisk(transaction_costs=costs)),
        ]
    )
    if output == "pipeline":
        pipe.set_output(transform="pandas")
    with config_context(transform_output="pandas" if output == "global" else "default"):
        pred = cross_val_predict(
            pipe,
            X,
            cv=WalkForward(train_size=4, test_size=4),
            portfolio_params={"weight_drift": weight_drift},
        )
    assert pred[0].assets.tolist() == ["A"]
    assert pred[1].assets.tolist() == ["B"]
    assert pred[1].turnover == pytest.approx(2.0)
    assert pred[1].total_cost == pytest.approx(expected_cost)
    np.testing.assert_allclose(
        pred[1].returns, X["B"].iloc[8:] - expected_cost, atol=1e-10
    )
    assert pipe[-1].previous_weights is None


def test_independent_fits_keep_holdings_across_failed_period(X, monkeypatch):
    monkeypatch.setattr(
        FailingFixedOptimization, "needs_previous_weights", property(lambda self: False)
    )
    model = FailingFixedOptimization(
        fail_on_n_observations=(10,), raise_on_failure=False
    )
    with pytest.warns(UserWarning, match="forced failure"):
        pred = cross_val_predict(
            model, X.iloc[:20, :3], cv=sks.TimeSeriesSplit(n_splits=3)
        )
    assert isinstance(pred[1], FailedPortfolio)
    np.testing.assert_allclose(pred.turnover, [1.0, np.nan, 0.0], equal_nan=True)
    np.testing.assert_array_equal(pred[2].previous_weights, pred[0].ending_weights)
