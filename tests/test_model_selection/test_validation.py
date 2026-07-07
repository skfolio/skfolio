"""Test Validation module."""

from __future__ import annotations

import numpy as np
import pytest
import sklearn.model_selection as sks
import sklearn.utils as sku
from sklearn import config_context
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from skfolio import MultiPeriodPortfolio, Population
from skfolio.model_selection import (
    CombinatorialPurgedCV,
    MultipleRandomizedCV,
    WalkForward,
    cross_val_predict,
)
from skfolio.model_selection import _validation as msv
from skfolio.model_selection._validation import _route_params
from skfolio.moments import (
    EWCovariance,
    ImpliedCovariance,
)
from skfolio.optimization import InverseVolatility, MeanRisk, ObjectiveFunction
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


def test_route_params_raises_on_unexpected_estimator_payload(monkeypatch):
    malformed = sku.Bunch(estimator=sku.Bunch(unexpected={"foo": "bar"}))

    monkeypatch.setattr(msv, "_routing_enabled", lambda: True)
    monkeypatch.setattr(msv.skm, "process_routing", lambda *args, **kwargs: malformed)

    with pytest.raises(
        RuntimeError,
        match="unexpected estimator payload",
    ):
        _route_params(
            MeanRisk(),
            params={"foo": np.array([1.0])},
            owner="cross_val_predict",
            callee="fit",
        )


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
