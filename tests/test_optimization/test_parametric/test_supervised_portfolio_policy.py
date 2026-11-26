"""Supervised policy learning contracts."""

import numpy as np
import pandas as pd
import pytest
from sklearn import clone

from skfolio.containers import AssetPanel
from skfolio.optimization import (
    EqualWeighted,
)

from ._supervised_helpers import RecordingOptimizer, data, model


def test_exact_linear_recovery_and_window_boundaries():
    X, panel = data(n=12)
    RecordingOptimizer.windows = []
    estimator = model().fit(X, characteristics=panel)
    np.testing.assert_allclose(estimator.coef_, [0.4], atol=1e-12)
    assert len(RecordingOptimizer.windows) == 5
    for t, window in zip(range(0, 10, 2), RecordingOptimizer.windows, strict=True):
        pd.testing.assert_frame_equal(window, X.iloc[t + 1 : t + 3])
    assert estimator.training_windows_.iloc[-1]["end"] == X.index[10]
    assert not hasattr(estimator.optimizer, "weights_")
    np.testing.assert_allclose(estimator.weights_history_.sum(axis=1), 1)


def test_unused_return_rows_do_not_affect_coefficients():
    X, panel = data(n=12)
    first = model().fit(X, characteristics=panel)
    changed = X.copy()
    changed.iloc[[0, -1]] = 1000
    second = model().fit(changed, characteristics=panel)
    np.testing.assert_array_equal(first.coef_, second.coef_)


def test_new_characteristics_and_standard_predict():
    X, panel = data()
    estimator = model(long_only=True).fit(X, characteristics=panel)
    old_coef = estimator.coef_.copy()
    future_X, future_panel = data(n=4, offset=11)
    weights = estimator.predict_weights(future_panel)
    assert weights.index.equals(future_X.index)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert (weights.to_numpy() >= 0).all()
    np.testing.assert_array_equal(estimator.coef_, old_coef)
    assert len(estimator.predict(future_X).returns) == len(future_X)


def test_clone_and_refit_reset():
    X, panel = data()
    estimator = model().fit(X, characteristics=panel)
    fresh = clone(estimator)
    assert not hasattr(fresh, "coef_")
    estimator.set_params(optimizer=EqualWeighted()).fit(X, characteristics=panel)
    np.testing.assert_allclose(estimator.coef_, 0, atol=1e-12)
    np.testing.assert_allclose(estimator.weights_, 1 / 3)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reversed", "short"])
def test_invalid_training_data(change):
    X, panel = data(n=2 if change == "short" else 11)
    if change == "missing":
        X.iloc[2, 0] = np.nan
    elif change == "duplicate":
        X.index = [X.index[0]] * len(X)
    elif change == "reversed":
        X = X.iloc[::-1]
    with pytest.raises(ValueError):
        model().fit(X, characteristics=panel)


def test_reject_non_invested_optimizer():
    X, panel = data()
    X *= 2
    with pytest.raises(ValueError, match="fully invested"):
        model().fit(X, characteristics=panel)


def test_failed_refit_clears_previous_policy():
    from sklearn.exceptions import NotFittedError

    X, panel = data()
    estimator = model().fit(X, characteristics=panel)
    X.iloc[1, 0] = np.nan
    with pytest.raises(ValueError):
        estimator.fit(X, characteristics=panel)
    with pytest.raises(NotFittedError):
        estimator.predict_weights(panel)


def test_real_optimizer_labels_and_asset_alignment():
    from skfolio.optimization import InverseVolatility

    X, panel = data(n=13)
    X = X + np.random.default_rng(17).normal(0, 0.01, X.shape)
    estimator = model(optimizer=InverseVolatility(), window_size=4).fit(
        X, characteristics=panel
    )
    expected = InverseVolatility().fit(X.iloc[1:5]).weights_
    np.testing.assert_allclose(estimator.target_weights_.iloc[0], expected)
    _, future = data(n=4, offset=13)
    reference = estimator.predict_weights(future)
    assert list(reference.columns) == list(X.columns)
    reordered = AssetPanel(
        fields={"signal": future["signal"][:, ::-1]},
        asset_names=future.asset_names[::-1],
        observations=future.observations,
    )
    with pytest.raises(ValueError, match=r"asset_names don.t match"):
        estimator.predict_weights(reordered)


def test_nonlinear_pipeline_predictions_and_clone():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, panel = data()
    supplied = make_pipeline(
        StandardScaler(), RandomForestRegressor(n_estimators=5, random_state=1)
    )
    estimator = model(regressor=supplied).fit(X, characteristics=panel)
    assert not hasattr(supplied[-1], "estimators_")
    assert not hasattr(estimator, "coef_")
    _, future = data(n=4, offset=11)
    weights = estimator.predict_weights(future)
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert np.isfinite(weights).all().all()
    assert not hasattr(clone(estimator), "regressor_")


def test_regressor_intercept_is_removed_from_portfolio_budget():
    from sklearn.dummy import DummyRegressor

    X, panel = data()
    estimator = model(regressor=DummyRegressor(strategy="constant", constant=2)).fit(
        X, characteristics=panel
    )
    np.testing.assert_allclose(estimator.weights_, 1 / 3)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_reject_nonfinite_regressor_predictions(value):
    from sklearn.base import BaseEstimator, RegressorMixin

    class InvalidRegressor(RegressorMixin, BaseEstimator):
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), value)

    X, panel = data()
    with pytest.raises(ValueError, match="one finite value"):
        model(regressor=InvalidRegressor()).fit(X, characteristics=panel)


def test_nonlinear_prediction_uses_fitted_regressor():
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    X, panel = data()
    estimator = model(
        regressor=make_pipeline(PolynomialFeatures(2), Ridge(alpha=0.1))
    ).fit(X, characteristics=panel)
    _, future = data(n=4, offset=11)
    signal = future["signal"]
    features = (signal - signal.mean(axis=1, keepdims=True)) / 3
    raw = estimator.regressor_.predict(features.reshape(-1, 1)).reshape(4, 3)
    expected = 1 / 3 + raw - raw.mean(axis=1, keepdims=True)
    np.testing.assert_allclose(estimator.predict_weights(future), expected, atol=1e-7)


def test_prediction_state_is_explicit_and_committed_only_on_success():
    X, panel = data()
    estimator = model(max_variance=0.34).fit(X, characteristics=panel)
    _, future = data(n=2, offset=11)
    old_exposures = estimator.characteristics_exposures_
    first = estimator.predict_weights(future, covariance=np.eye(3))
    second = estimator.predict_weights(future, covariance=np.eye(3))
    pd.testing.assert_frame_equal(first, second)
    assert estimator.characteristics_exposures_ is old_exposures
    with pytest.raises(ValueError, match="infeasible"):
        estimator.predict_weights(future, covariance=10 * np.eye(3), update_state=True)
    assert estimator.characteristics_exposures_ is old_exposures
    assert estimator.last_prediction_observation_ == X.index[-1]
    estimator.predict_weights(future, covariance=np.eye(3), update_state=True)
    assert estimator.last_prediction_observation_ == future.observations[-1]
    with pytest.raises(ValueError, match="Prediction dates"):
        estimator.predict_weights(future, covariance=np.eye(3), update_state=True)


def test_label_optimizer_receives_routed_factor_metadata():
    from sklearn import config_context

    from skfolio.optimization import InverseVolatility
    from skfolio.prior import TimeSeriesFactorModel

    X, panel = data(n=31)
    rng = np.random.default_rng(31)
    X = X + rng.normal(0, 0.01, X.shape)
    factors = pd.DataFrame(
        rng.normal(size=(40, 1)),
        index=pd.date_range("2020-01-01", periods=40),
        columns=["factor"],
    )
    with config_context(enable_metadata_routing=True):
        optimizer = InverseVolatility(
            prior_estimator=TimeSeriesFactorModel().set_fit_request(factors=True)
        )
        estimator = model(optimizer=optimizer, window_size=5).fit(
            X, characteristics=panel, optimizer_fit_params={"factors": factors}
        )
        reference = clone(optimizer).fit(X.iloc[1:6], factors=factors.loc[X.index[1:6]])
    np.testing.assert_allclose(estimator.target_weights_.iloc[0], reference.weights_)
    assert not hasattr(optimizer, "weights_")
    with pytest.raises(ValueError, match="cover each history"):
        model(window_size=5).fit(
            X,
            characteristics=panel,
            optimizer_fit_params={"factors": factors.iloc[20:]},
        )
