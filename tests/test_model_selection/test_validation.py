"""Test Validation module."""

import numpy as np
import sklearn.model_selection as sks
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
from skfolio.moments import (
    ImpliedCovariance,
)
from skfolio.optimization import InverseVolatility, MeanRisk, ObjectiveFunction
from skfolio.pre_selection import SelectKExtremes
from skfolio.prior import EmpiricalPrior


def assert_weights_dict_subset_equal(d1: dict, d2: dict, tol: float = 1e-15) -> None:
    """True iff for every key k in d2 d1.get(k, 0.0) matches d2[k] within tol."""
    for k, b in d2.items():
        assert abs(d1.get(k, 0.0) - b) < tol


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
