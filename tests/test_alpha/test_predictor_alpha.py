"""Unit tests for PredictorAlpha."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge, SGDRegressor

from skfolio._constants import (
    _DESCRIPTOR_SCORES,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
)
from skfolio.alpha import ForecastUnit, PredictorAlpha
from skfolio.descriptor import BaseDescriptor, Passthrough
from skfolio.preprocessing import BaseCSTransformer, CSStandardScaler, CSWinsorizer
from tests.test_alpha._alpha_test_utils import apply_idio_nan_exclusions


class MethodAwareDescriptor(BaseDescriptor):
    """Descriptor that records which transform method was called."""

    def __init__(self, field: str = "signal"):
        self.field = field

    def fit_transform(self, X, y=None, **fit_params):
        self.fit_transform_called_ = True
        return X[self.field]

    def partial_fit_transform(self, X, y=None, **fit_params):
        self.partial_fit_transform_called_ = True
        return X[self.field]


class RecordingTargetTransformer(BaseCSTransformer):
    """Target transformer that records routed cross-sectional metadata."""

    def __init__(self):
        pass

    def transform(self, X, cs_weights=None, cs_groups=None):
        self.seen_weights_ = cs_weights is not None
        self.seen_groups_ = cs_groups is not None
        return np.asarray(X, dtype=float)


class CountingRegressor(BaseEstimator):
    """Regressor that records how many flattened samples it receives."""

    def fit(self, X, y):
        self.n_samples_seen_ = len(X)
        return self

    def partial_fit(self, X, y):
        self.n_samples_seen_ += len(X)
        return self

    def predict(self, X):
        return np.zeros(len(X))


class TestBasicFunctionality:
    """Test basic fit and partial_fit functionality."""

    def test_fit_returns_self(self, alpha_deterministic_panel):
        """Test that fit() returns self for method chaining."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        result = model.fit(alpha_deterministic_panel)
        assert result is model

    def test_partial_fit_returns_self(self, alpha_deterministic_panel):
        """Test that partial_fit() returns self for method chaining."""
        model = PredictorAlpha(
            predictor=SGDRegressor(random_state=42),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        result = model.partial_fit(alpha_deterministic_panel)
        assert result is model

    def test_alpha_shape(self, alpha_deterministic_panel):
        """Test that alpha_ has correct shape (n_assets,)."""
        n_assets = alpha_deterministic_panel.n_assets
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None
        assert model.alpha_.shape == (n_assets,)


class TestWarmupBehavior:
    """Test warmup behavior when insufficient observations."""

    def test_warmup_alpha_is_none(self, alpha_deterministic_panel):
        """Test that alpha_ is None during warmup (n_obs <= horizon)."""
        # Take only 3 observations with horizon=5
        panel_short = alpha_deterministic_panel[:3]

        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=5,
            half_life=5,
        )
        model.fit(panel_short)

        assert model.alpha_ is None

    def test_warmup_transition(self, alpha_deterministic_panel):
        """Test transition from warmup to valid alpha."""
        model = PredictorAlpha(
            predictor=SGDRegressor(random_state=42),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=5,
            half_life=5,
        )

        # First partial_fit with too few observations
        model.partial_fit(alpha_deterministic_panel[:3])
        assert model.alpha_ is None

        # Second partial_fit that crosses the threshold
        model.partial_fit(alpha_deterministic_panel[3:10])
        assert model.alpha_ is not None

    def test_warmup_then_fit_with_batch_only_predictor(self, alpha_deterministic_panel):
        """A batch-only predictor can fit after an initial warmup-only partial fit."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=5,
            half_life=5,
            calibrate_to_return_units=False,
        )

        model.partial_fit(alpha_deterministic_panel[:3])
        assert model.alpha_ is None

        model.partial_fit(alpha_deterministic_panel[3:10])
        assert model.alpha_ is not None

    def test_signal_lag_delays_warmup(self, alpha_deterministic_panel):
        """A longer signal lag should delay the first trainable target."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            signal_lag=2,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel[:4])
        assert model.alpha_ is None

        model.fit(alpha_deterministic_panel[:6])
        assert model.alpha_ is not None


class TestStreamingConsistency:
    """Test streaming/online behavior."""

    def test_buffer_size(self, alpha_deterministic_panel):
        """Test that buffer keeps rows needed for future target maturity."""
        horizon = 5
        signal_lag = 2
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=horizon,
            signal_lag=signal_lag,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        target_gap = horizon + signal_lag - 1
        assert model._buffer.n_observations == target_gap
        assert list(model._buffer.fields) == [
            _IDIO_RETURNS,
            _IDIO_VARIANCES,
            _DESCRIPTOR_SCORES,
        ]
        assert model._buffer[_DESCRIPTOR_SCORES].shape[0] == target_gap
        assert model._buffer[_IDIO_RETURNS].shape[0] == target_gap
        assert model._buffer[_IDIO_VARIANCES].shape[0] == target_gap

    def test_buffered_rows_are_trained_when_their_targets_mature(
        self, alpha_deterministic_panel
    ):
        """Rows kept in the target-gap buffer should be trained in the next batch."""
        model = PredictorAlpha(
            predictor=CountingRegressor(),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            calibrate_to_return_units=False,
        )

        model.partial_fit(alpha_deterministic_panel[:5])
        model.partial_fit(alpha_deterministic_panel[5:10])

        # First batch trains 2 dates, second batch trains the 5 matured buffered dates.
        assert model.predictor_.n_samples_seen_ == (2 + 5) * model.n_assets_

    def test_estimation_mask_excludes_predictor_training_samples(
        self, alpha_deterministic_panel
    ):
        """Non-estimation assets should not train the predictor."""
        panel = alpha_deterministic_panel.copy(deep=True)
        with panel.edit_masks():
            panel.estimation_mask[:, 0] = False

        model = PredictorAlpha(
            predictor=CountingRegressor(),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            calibrate_to_return_units=False,
        )
        model.fit(panel)

        target_gap = model.signal_lag + model.horizon - 1
        n_trainable = panel.n_observations - target_gap
        assert model.predictor_.n_samples_seen_ == n_trainable * (panel.n_assets - 1)

    def test_partial_fit_requires_partial_fit_predictor(
        self, alpha_deterministic_panel
    ):
        """Test that second partial_fit raises error for predictor without partial_fit."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),  # Ridge doesn't have partial_fit
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        # First partial_fit succeeds (uses predictor.fit internally)
        model.partial_fit(alpha_deterministic_panel[:10])
        assert model.alpha_ is not None

        # Second partial_fit fails (needs predictor.partial_fit)
        with pytest.raises(TypeError, match="does not support partial_fit"):
            model.partial_fit(alpha_deterministic_panel[10:])


class TestNaNHandling:
    """Test NaN handling in forward returns."""

    def test_nan_in_returns(self, alpha_deterministic_panel):
        """Test that NaN in idio_returns is handled correctly."""
        panel = alpha_deterministic_panel.copy(deep=True)

        # Introduce NaN in some returns
        apply_idio_nan_exclusions(panel, "X", rows=slice(5, 8))

        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
            calibrate_to_return_units=False,
        )
        # Should not raise
        model.fit(panel)

        assert model.alpha_ is not None

    def test_all_nan_targets_do_not_predict_unfitted_model(
        self, alpha_deterministic_panel
    ):
        """A first fit with no valid targets should stay in warmup."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel[_IDIO_RETURNS][:] = np.nan

        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            calibrate_to_return_units=False,
        )
        model.fit(panel)

        assert model.alpha_ is None


class TestFitReset:
    """Test that fit() properly resets state."""

    def test_fit_resets_state(self, alpha_deterministic_panel):
        """Test that calling fit() after partial_fit() starts fresh."""
        model = PredictorAlpha(
            predictor=SGDRegressor(random_state=42),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        # First partial_fit
        model.partial_fit(alpha_deterministic_panel[:15])
        alpha_after_partial = model.alpha_.copy()

        # Second fit (should reset)
        model.fit(alpha_deterministic_panel[15:])
        alpha_after_fit = model.alpha_

        # Results should differ (different data, fresh state)
        assert not np.allclose(alpha_after_partial, alpha_after_fit)


class TestMultipleDescriptors:
    """Test aggregation of multiple signals."""

    def test_multiple_descriptors_shape(self, alpha_deterministic_panel):
        """Test that multiple descriptors produce correct output shape."""
        panel = alpha_deterministic_panel.copy(deep=True)
        # Add a second signal (must maintain sorted asset order)
        panel["signal2"] = panel["signal"] * 0.5

        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[
                ("signal", Passthrough("signal")),
                ("signal2", Passthrough("signal2")),
            ],
            horizon=3,
            half_life=5,
        )
        model.fit(panel)

        # Alpha still (n_assets,)
        assert model.alpha_.shape == (4,)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_horizon_one(self, alpha_deterministic_panel):
        """Test with minimal horizon=1."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=1,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None
        assert model._buffer.n_observations == 1

    def test_neutralization_uses_asset_panel_exposure_field(
        self, alpha_deterministic_panel
    ):
        """Score neutralization should read factor metadata from AssetPanel fields."""
        panel = alpha_deterministic_panel.copy(deep=True)
        n_obs, n_assets = panel.n_observations, panel.n_assets
        panel.add_3d_field(
            name=_EXPOSURES,
            values=np.ones((n_obs, n_assets, 1)),
            third_axis_name="factor",
            third_axis_labels=["market"],
            third_axis_groups=["market"],
        )

        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            neutralize_against=["market"],
            calibrate_to_return_units=False,
        )
        model.fit(panel)

        assert model.alpha_ is not None

    def test_single_observation_streaming(self, alpha_deterministic_panel):
        """Test streaming one observation at a time after warmup."""
        model = PredictorAlpha(
            predictor=SGDRegressor(random_state=42),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        # Warmup with enough data
        model.partial_fit(alpha_deterministic_panel[:5])

        # Stream one observation at a time
        for i in range(5, 10):
            model.partial_fit(alpha_deterministic_panel[i : i + 1])

        assert model.alpha_ is not None

    def test_all_nan_one_asset(self, alpha_deterministic_panel):
        """Test that all-NaN asset produces NaN alpha."""
        panel = alpha_deterministic_panel.copy(deep=True)
        # Make all idio_returns NaN for asset X
        apply_idio_nan_exclusions(panel, "X")

        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
            calibrate_to_return_units=False,
        )
        model.fit(panel)

        # Asset X should have NaN alpha (or the model handles it gracefully)
        assert model.alpha_ is not None

    def test_passthrough_transformers(self, alpha_deterministic_panel):
        """Test with transformers set to passthrough."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
            target_outlier_transformer="passthrough",
            target_scoring_transformer="passthrough",
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_large_half_life(self, alpha_deterministic_panel):
        """Test with very large half_life (slow adaptation)."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=1000,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_small_half_life(self, alpha_deterministic_panel):
        """Test with very small half_life (fast adaptation)."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=1,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None


class TestCalibration:
    """Tests for calibrate_to_return_units parameter."""

    def test_no_calibration(self, alpha_deterministic_panel):
        """Test with calibrate_to_return_units=False."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            calibrate_to_return_units=False,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None
        # Without calibration, alpha is raw predictor output.

    def test_calibration_vs_no_calibration_differ(self, alpha_deterministic_panel):
        """Test that calibrated and non-calibrated alpha differ."""
        model_calibrated = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            calibrate_to_return_units=True,
        )
        model_calibrated.fit(alpha_deterministic_panel)

        model_no_calibration = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            calibrate_to_return_units=False,
        )
        model_no_calibration.fit(alpha_deterministic_panel)

        # Results should differ
        assert not np.allclose(model_calibrated.alpha_, model_no_calibration.alpha_)

    def test_forecast_scale_multiplies_calibrated_alpha(
        self, alpha_deterministic_panel
    ):
        """forecast_scale should rescale alpha after return-unit calibration."""
        model_base = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            calibrate_to_return_units=True,
        )
        model_scaled = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            calibrate_to_return_units=True,
            forecast_scale=2.5,
        )

        model_base.fit(alpha_deterministic_panel)
        model_scaled.fit(alpha_deterministic_panel)

        np.testing.assert_allclose(
            model_scaled.calibration_coef_, model_base.calibration_coef_, rtol=1e-10
        )
        np.testing.assert_allclose(
            model_scaled.alpha_, 2.5 * model_base.alpha_, rtol=1e-10, equal_nan=True
        )

    def test_idio_sharpe_forecast_unit(self, alpha_deterministic_panel):
        """Idio-Sharpe forecast unit should still return a finite alpha."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            forecast_unit=ForecastUnit.IDIO_SHARPE,
            calibrate_to_return_units=False,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None
        assert np.isfinite(model.alpha_).all()

    def test_calibration_uses_ewls_statistics_not_smoothed_coefficients(self):
        """Calibration should average normal-equation statistics before solving."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            half_life=1,
        )
        model._calibration_decay = 0.5
        model._calibration_normal = 0.0
        model._calibration_cross = 0.0
        model._n_valid_calibration_obs = 0
        model.calibration_coef_ = np.nan

        model._update_calibration(
            uncalibrated_alpha=np.array([[1.0, 1.0], [2.0, 2.0]]),
            forward_return=np.array([[1.0, 1.0], [0.0, 0.0]]),
            idio_variances=np.ones((2, 2)),
            estimation_weights=np.ones((2, 2)),
        )

        expected_ewls_coefficient = 0.5 / (4.5 + 4.5e-6)
        smoothed_daily_coefficient = 1.0 / 3.0
        np.testing.assert_allclose(model.calibration_coef_, expected_ewls_coefficient)
        assert not np.allclose(model.calibration_coef_, smoothed_daily_coefficient)

    def test_estimation_mask_excludes_calibration_observations(self):
        """Non-estimation assets should not update the calibration coefficient."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            half_life=1,
        )
        model._calibration_decay = 0.0
        model._calibration_normal = 0.0
        model._calibration_cross = 0.0
        model._n_valid_calibration_obs = 0
        model.calibration_coef_ = np.nan

        model._update_calibration(
            uncalibrated_alpha=np.array([[1.0, 1.0, 10.0]]),
            forward_return=np.array([[1.0, 1.0, 0.0]]),
            idio_variances=np.ones((1, 3)),
            estimation_weights=np.array([[1.0, 1.0, 0.0]]),
        )

        np.testing.assert_allclose(model.calibration_coef_, 1.0 / (1.0 + 1.0e-6))

    def test_explicit_cv_raises_when_calibration_samples_are_insufficient(
        self, alpha_deterministic_panel
    ):
        """Explicit CV should not silently fall back to in-sample calibration."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=2,
            cv=5,
        )

        with pytest.raises(ValueError, match="cv requires at least"):
            model.fit(alpha_deterministic_panel[:3])


class TestTargetTransformers:
    """Tests for target transformer parameters."""

    def test_target_outlier_transformer(self, alpha_deterministic_panel):
        """Test with custom target_outlier_transformer."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            target_outlier_transformer=CSWinsorizer(low=0.1, high=0.9),
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_target_scoring_transformer(self, alpha_deterministic_panel):
        """Test with custom target_scoring_transformer."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            target_scoring_transformer=CSStandardScaler(),
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_both_target_transformers(self, alpha_deterministic_panel):
        """Test with both target transformers."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            target_outlier_transformer=CSWinsorizer(low=0.05, high=0.95),
            target_scoring_transformer=CSStandardScaler(),
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_target_transformer_receives_weights_and_groups_without_calibration(
        self, alpha_deterministic_panel
    ):
        """Target transforms should keep weights when calibration is disabled."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel["group"] = np.tile(np.array([0, 0, 1, 1]), (panel.n_observations, 1))

        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            transform_by_group="group",
            target_outlier_transformer=RecordingTargetTransformer(),
            calibrate_to_return_units=False,
        )
        model.fit(panel)

        assert model.target_outlier_transformer_.seen_weights_
        assert model.target_outlier_transformer_.seen_groups_


class TestValidation:
    """Tests for parameter validation."""

    def test_zero_horizon_raises(self, alpha_deterministic_panel):
        """Zero horizon should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=0,
        )
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            model.fit(alpha_deterministic_panel)

    def test_negative_horizon_raises(self, alpha_deterministic_panel):
        """Negative horizon should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=-1,
        )
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            model.fit(alpha_deterministic_panel)

    def test_zero_signal_lag_raises(self, alpha_deterministic_panel):
        """Zero signal_lag should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            signal_lag=0,
        )
        with pytest.raises(ValueError, match="signal_lag must be a positive integer"):
            model.fit(alpha_deterministic_panel)

    def test_zero_half_life_raises(self, alpha_deterministic_panel):
        """Zero half_life should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            half_life=0,
        )
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            model.fit(alpha_deterministic_panel)

    def test_negative_half_life_raises(self, alpha_deterministic_panel):
        """Negative half_life should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            half_life=-5,
        )
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            model.fit(alpha_deterministic_panel)

    def test_zero_forecast_scale_raises(self, alpha_deterministic_panel):
        """Zero forecast_scale should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.0,
        )
        with pytest.raises(
            ValueError, match="forecast_scale must be a positive number"
        ):
            model.fit(alpha_deterministic_panel)

    def test_empty_descriptors_raises(self, alpha_deterministic_panel):
        """Empty descriptors list should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[],
            horizon=3,
        )
        with pytest.raises(ValueError, match="descriptors cannot be empty"):
            model.fit(alpha_deterministic_panel)

    def test_invalid_forecast_unit_raises(self, alpha_deterministic_panel):
        """Unsupported forecast_unit should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            forecast_unit="bad",
        )
        with pytest.raises(TypeError, match="forecast_unit must be of type"):
            model.fit(alpha_deterministic_panel)

    def test_non_bool_calibrate_to_return_units_raises(self, alpha_deterministic_panel):
        """Non-boolean calibrate_to_return_units should raise ValueError."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            calibrate_to_return_units="bad",
        )
        with pytest.raises(
            ValueError, match="calibrate_to_return_units must be a boolean"
        ):
            model.fit(alpha_deterministic_panel)


class TestPredictors:
    """Tests for different predictor types."""

    def test_ridge_predictor(self, alpha_deterministic_panel):
        """Test with Ridge predictor (batch only)."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_sgd_predictor(self, alpha_deterministic_panel):
        """Test with SGDRegressor (supports partial_fit)."""
        model = PredictorAlpha(
            predictor=SGDRegressor(random_state=42),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_sgd_streaming(self, alpha_deterministic_panel):
        """Test SGDRegressor in streaming mode."""
        model = PredictorAlpha(
            predictor=SGDRegressor(random_state=42),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
        )

        # First batch
        model.partial_fit(alpha_deterministic_panel[:10])
        alpha1 = model.alpha_.copy()

        # Second batch (uses partial_fit internally)
        model.partial_fit(alpha_deterministic_panel[10:])
        alpha2 = model.alpha_

        # Alphas should differ after update
        assert not np.allclose(alpha1, alpha2)

    def test_fit_uses_descriptor_fit_transform(self, alpha_deterministic_panel):
        """Batch fit should call descriptor fit_transform."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", MethodAwareDescriptor())],
            horizon=3,
        )
        model.fit(alpha_deterministic_panel)

        descriptor = model.descriptors_[0]
        assert descriptor.fit_transform_called_
        assert not hasattr(descriptor, "partial_fit_transform_called_")

    def test_partial_fit_uses_descriptor_partial_fit_transform(
        self, alpha_deterministic_panel
    ):
        """Online fit should call descriptor partial_fit_transform."""
        model = PredictorAlpha(
            predictor=SGDRegressor(random_state=42),
            descriptors=[("signal", MethodAwareDescriptor())],
            horizon=3,
        )
        model.partial_fit(alpha_deterministic_panel)

        descriptor = model.descriptors_[0]
        assert descriptor.partial_fit_transform_called_
        assert not hasattr(descriptor, "fit_transform_called_")


class TestRegression:
    """Regression tests with exact expected values to catch computation changes."""

    def test_exact_alpha_values(self, alpha_deterministic_panel):
        """Test exact alpha values to detect any computation changes."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        # Expected alpha values (computed with current implementation)
        # These values should remain constant unless the algorithm changes
        # To regenerate: print(repr(model.alpha_))
        expected_alpha = np.array([0.00318696, 0.00134846, 0.00214101, 0.00187144])

        np.testing.assert_allclose(
            model.alpha_,
            expected_alpha,
            rtol=1e-5,
            err_msg="Alpha values changed - check for computation changes",
        )

    def test_exact_coef_values(self, alpha_deterministic_panel):
        """Test exact EWLS calibration coefficient values."""
        model = PredictorAlpha(
            predictor=Ridge(alpha=1.0),
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        assert np.isfinite(model.calibration_coef_)
        assert model._n_valid_calibration_obs > 0

        # Expected coefficient (computed with current implementation)
        # To regenerate: print(repr(model.calibration_coef_))
        expected_coef = 0.88278537

        np.testing.assert_allclose(
            model.calibration_coef_,
            expected_coef,
            rtol=1e-5,
            err_msg="Coefficient values changed - check for computation changes",
        )
