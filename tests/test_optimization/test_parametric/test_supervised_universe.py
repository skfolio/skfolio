"""Supervised policy universe contracts."""

import numpy as np
import pandas as pd
import pytest

from skfolio.containers import AssetPanel
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.optimization import (
    EqualWeighted,
)

from ._supervised_helpers import data, expanded_data, model


def test_dynamic_eligibility_excludes_assets_and_charges_exit_turnover():
    X, panel = data()
    estimator = model(allow_universe_changes=True, long_only=True).fit(
        X, characteristics=panel
    )
    _, future = data(n=2, offset=11)
    dynamic = AssetPanel(
        fields={
            "signal": np.where(
                [[True, True, False], [False, True, True]], future["signal"], np.nan
            )
        },
        asset_names=future.asset_names,
        observations=future.observations,
        active_mask=np.array([[True, True, False], [False, True, True]]),
    )
    weights = estimator.predict_weights(dynamic).to_numpy()
    np.testing.assert_allclose(weights.sum(axis=1), 1)
    assert abs(weights[0, 2]) < 1e-6
    assert abs(weights[1, 0]) < 1e-6
    estimator.set_params(max_turnover=0)
    with pytest.raises(ValueError, match="infeasible"):
        estimator.predict_weights(dynamic, holdings=np.full((2, 3), 1 / 3))


def test_dynamic_training_uses_decision_eligible_label_assets():
    X, panel = data()
    active = np.ones(X.shape, dtype=bool)
    active[:4, 2] = False
    dynamic = AssetPanel(
        fields={"signal": np.where(active, panel["signal"], np.nan)},
        asset_names=panel.asset_names,
        observations=panel.observations,
        active_mask=active,
    )
    estimator = model(allow_universe_changes=True, optimizer=EqualWeighted()).fit(
        X, characteristics=dynamic
    )
    assert (estimator.target_weights_.iloc[:2, 2] == 0).all()
    np.testing.assert_allclose(estimator.target_weights_.sum(axis=1), 1)


def test_expand_universe_preserves_regressor_and_original_model():
    X, panel = data()
    estimator = model(allow_universe_changes=True).fit(X, characteristics=panel)
    history, expanded = expanded_data()
    adapted = estimator.expand_universe(history, expanded)
    np.testing.assert_array_equal(adapted.regressor_.coef_, estimator.regressor_.coef_)
    assert list(estimator.feature_names_in_) == ["a", "b", "c"]
    assert list(adapted.feature_names_in_) == ["a", "b", "c", "new"]
    returns, future = expanded_data(n=3, offset=11)
    prediction = adapted.predict_weights(future)
    assert np.isfinite(prediction.to_numpy()).all()
    assert (prediction["new"] > 0).all()
    returns.index = future.observations + pd.Timedelta(days=1)
    result = adapted.evaluate_execution(
        returns,
        future,
        initial_holdings=0,
        initial_cash=1,
    )
    np.testing.assert_allclose(
        result.ending_weights.sum(axis=1) + result.ledger.ending_cash, 1
    )
    # Historical labels describe the original fit, not invented labels for new assets.
    pd.testing.assert_frame_equal(adapted.target_weights_, estimator.target_weights_)


def test_expand_universe_rejects_future_history_and_changes_to_old_returns():
    X, panel = data()
    estimator = model(allow_universe_changes=True).fit(X, characteristics=panel)
    history, expanded = expanded_data(n=12)
    with pytest.raises(ValueError, match="current prediction date"):
        estimator.expand_universe(history, expanded)
    history, expanded = expanded_data()
    history.iloc[0, 0] += 0.01
    with pytest.raises(ValueError, match="preserved exactly"):
        estimator.expand_universe(history, expanded)
    np.testing.assert_allclose(estimator.regressor_.coef_, [0.4])


def test_expand_universe_requires_explicit_positional_bounds():
    X, panel = data()
    estimator = model(allow_universe_changes=True, max_weights=[1, 1, 1]).fit(
        X, characteristics=panel
    )
    history, expanded = expanded_data()
    with pytest.raises(ValueError, match="expanded positional max_weights"):
        estimator.expand_universe(history, expanded)
    adapted = estimator.expand_universe(history, expanded, max_weights=[1, 1, 1, 1])
    assert len(adapted.weights_) == 4


def test_new_assets_warm_up_rolling_descriptors_and_causal_risk():
    from skfolio.descriptor import RollingMomentum

    X, panel = data()
    estimator = (
        model(allow_universe_changes=True, max_variance=10)
        .set_params(
            characteristics_exposures=[
                (
                    "momentum",
                    FixedWeightedFactor(
                        descriptors=[("momentum", RollingMomentum(window=3, skip=0))],
                        outlier_transformer="passthrough",
                        scoring_transformer="passthrough",
                    ),
                )
            ],
        )
        .fit(X, characteristics=panel)
    )
    history, expanded = expanded_data()
    adapted = estimator.expand_universe(history, expanded)
    returns, future = expanded_data(n=3, offset=11)
    batch = adapted.predict_weights(future, covariance=np.eye(4))
    streamed = pd.concat(
        [
            adapted.predict_weights(
                future.isel(observations=slice(t, t + 1)),
                covariance=np.eye(4),
                update_state=True,
            )
            for t in range(3)
        ]
    )
    pd.testing.assert_frame_equal(batch, streamed)
    fresh = estimator.expand_universe(history, expanded)
    returns.index = future.observations + pd.Timedelta(days=1)
    result = fresh.evaluate_execution(
        returns, future, initial_holdings=0, initial_cash=1
    )
    assert np.isfinite(result.ledger.to_numpy(dtype=float)).all()


def test_new_assets_require_finite_warmup():
    X, panel = data()
    estimator = model(allow_universe_changes=True).fit(X, characteristics=panel)
    history, expanded = expanded_data()
    signal = expanded["signal"].copy()
    signal[-1, -1] = np.nan
    missing = AssetPanel(
        fields={"signal": signal},
        asset_names=history.columns.to_numpy(),
        observations=history.index,
    )
    with pytest.raises(ValueError, match="warmed-up"):
        estimator.expand_universe(history, missing)
    assert estimator.n_features_in_ == 3
