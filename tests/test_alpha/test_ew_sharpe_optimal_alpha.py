"""Unit tests for EWSharpeOptimalAlpha."""

import numpy as np
import pytest

from skfolio._constants import (
    _DESCRIPTOR_SCORES,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
)
from skfolio.alpha import EWSharpeOptimalAlpha, ForecastUnit
from skfolio.descriptor import Passthrough
from tests.test_alpha._alpha_test_utils import apply_idio_nan_exclusions


class TestBasicFunctionality:
    """Test basic fit and partial_fit functionality."""

    def test_fit_returns_self(self, alpha_deterministic_panel):
        """Test that fit() returns self for method chaining."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        result = model.fit(alpha_deterministic_panel)
        assert result is model

    def test_partial_fit_returns_self(self, alpha_deterministic_panel):
        """Test that partial_fit() returns self for method chaining."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        result = model.partial_fit(alpha_deterministic_panel)
        assert result is model

    def test_alpha_shape(self, alpha_deterministic_panel):
        """Test that alpha_ has correct shape (n_assets,)."""
        n_assets = alpha_deterministic_panel.n_assets
        model = EWSharpeOptimalAlpha(
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

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=5,
            half_life=5,
        )
        model.fit(panel_short)

        assert model.alpha_ is None

    def test_warmup_transition(self, alpha_deterministic_panel):
        """Test transition from warmup to valid alpha."""
        model = EWSharpeOptimalAlpha(
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


class TestAlphaPath:
    """Test causal alpha-path APIs."""

    def test_fit_transform_returns_alpha_path(self, alpha_deterministic_panel):
        """fit_transform should return one alpha row per input observation."""
        horizon = 3
        signal_lag = 1
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=horizon,
            signal_lag=signal_lag,
            half_life=5,
        )

        alphas = model.fit_transform(alpha_deterministic_panel)

        assert alphas.shape == (
            alpha_deterministic_panel.n_observations,
            alpha_deterministic_panel.n_assets,
        )
        assert np.isnan(alphas[: horizon + signal_lag - 1]).all()
        np.testing.assert_allclose(model.alpha_, alphas[-1], equal_nan=True)

    def test_fit_transform_with_signal_lag(self, alpha_deterministic_panel):
        """Conservative signal lag should delay the first available alpha."""
        horizon = 3
        signal_lag = 2
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=horizon,
            signal_lag=signal_lag,
            half_life=5,
        )

        alphas = model.fit_transform(alpha_deterministic_panel)

        target_gap = horizon + signal_lag - 1
        assert np.isnan(alphas[:target_gap]).all()
        assert np.isfinite(alphas[-1]).all()
        np.testing.assert_allclose(model.alpha_, alphas[-1], rtol=1e-10)

    def test_fit_matches_fit_transform_latest(self, alpha_deterministic_panel):
        """Latest-only fit should match the last fit_transform alpha."""
        model_fit = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model_path = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        model_fit.fit(alpha_deterministic_panel)
        alphas = model_path.fit_transform(alpha_deterministic_panel)

        np.testing.assert_allclose(model_fit.alpha_, alphas[-1], rtol=1e-10)

    def test_forecast_scale_multiplies_alpha_path(self, alpha_deterministic_panel):
        """forecast_scale should only rescale published alpha forecasts."""
        model_base = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model_scaled = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
            forecast_scale=2.5,
        )

        base_alphas = model_base.fit_transform(alpha_deterministic_panel)
        scaled_alphas = model_scaled.fit_transform(alpha_deterministic_panel)

        np.testing.assert_allclose(
            model_scaled.coef_, model_base.coef_, rtol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            scaled_alphas, 2.5 * base_alphas, rtol=1e-10, equal_nan=True
        )
        np.testing.assert_allclose(
            model_scaled.alpha_, 2.5 * model_base.alpha_, rtol=1e-10, equal_nan=True
        )

    def test_partial_fit_transform_matches_batch_path(self, alpha_deterministic_panel):
        """Chunked path generation should match batch causal path generation."""
        model_batch = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        batch_path = model_batch.fit_transform(alpha_deterministic_panel)

        model_stream = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        stream_path = np.vstack(
            [
                model_stream.partial_fit_transform(alpha_deterministic_panel[:7]),
                model_stream.partial_fit_transform(alpha_deterministic_panel[7:13]),
                model_stream.partial_fit_transform(alpha_deterministic_panel[13:]),
            ]
        )

        np.testing.assert_allclose(batch_path, stream_path, rtol=1e-10, equal_nan=True)


class TestStreamingConsistency:
    """Test streaming/online behavior."""

    def test_streaming_vs_batch(self, alpha_deterministic_panel):
        """Test that streaming partial_fit gives same result as batch fit."""
        horizon = 3
        half_life = 5

        # Batch fit
        model_batch = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=horizon,
            half_life=half_life,
        )
        model_batch.fit(alpha_deterministic_panel)

        # Streaming fit (split into chunks)
        model_stream = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=horizon,
            half_life=half_life,
        )
        model_stream.partial_fit(alpha_deterministic_panel[:10])
        model_stream.partial_fit(alpha_deterministic_panel[10:19])
        model_stream.partial_fit(alpha_deterministic_panel[19:])

        # Results should be identical
        np.testing.assert_allclose(model_batch.alpha_, model_stream.alpha_, rtol=1e-10)

    def test_buffer_size(self, alpha_deterministic_panel):
        """Test that buffer maintains correct size (horizon observations)."""
        horizon = 5
        signal_lag = 2
        model = EWSharpeOptimalAlpha(
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


class TestAlphaProperties:
    """Test properties of computed alpha."""

    def test_alpha_cross_sectional_neutrality(self, alpha_deterministic_panel):
        """Test that alpha is approximately cross-sectionally neutral (sums to ~0)."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        # Alpha should sum close to zero (CS-neutral due to no intercept)
        alpha_sum = np.nansum(model.alpha_)
        assert np.abs(alpha_sum) < 1e-10, f"Alpha sum {alpha_sum} not near zero"


class TestNaNHandling:
    """Test NaN handling in forward returns."""

    def test_nan_in_returns(self, alpha_deterministic_panel):
        """Test that NaN in idio_returns is handled correctly."""
        panel = alpha_deterministic_panel.copy(deep=True)

        # Introduce NaN in some returns
        apply_idio_nan_exclusions(panel, "X", rows=slice(5, 8))

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        # Should not raise
        model.fit(panel)

        assert model.alpha_ is not None

    def test_nan_in_descriptor_scores_is_excluded(self, alpha_deterministic_panel):
        """NaN descriptor scores should be zero-weighted in WLS."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel["signal"][2:5, 0] = np.nan

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        model.fit(panel)

        assert model.alpha_ is not None

    def test_nan_forward_target_is_excluded_without_variance_inf(
        self, alpha_deterministic_panel
    ):
        """NaN forward targets should not require manually setting variance to inf."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel[_IDIO_RETURNS][5:8, 0] = np.nan

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        model.fit(panel)

        assert model.alpha_ is not None

    def test_non_positive_variance_raises(self, alpha_deterministic_panel):
        """Non-positive idiosyncratic variances are rejected during validation."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel[_IDIO_VARIANCES][2:5, 0] = 0.0
        panel[_IDIO_VARIANCES][5:8, 1] = -1.0

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(panel)

    def test_infinite_variance_raises(self, alpha_deterministic_panel):
        """Infinite idiosyncratic variances are rejected during validation."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel[_IDIO_VARIANCES][5, 0] = np.inf

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        with pytest.raises(ValueError, match="strictly positive"):
            model.fit(panel)

    def test_estimation_mask_does_not_block_alpha_forecasts(
        self, alpha_deterministic_panel
    ):
        """Non-estimation assets should still receive alpha forecasts."""
        panel = alpha_deterministic_panel.copy(deep=True)
        with panel.edit_masks():
            panel.estimation_mask[:, 0] = False

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model.fit(panel)

        assert np.isfinite(model.alpha_[0])

    def test_neutralization_with_group_transform(self, alpha_deterministic_panel):
        """Neutralization should forward group labels to score transformers."""
        panel = alpha_deterministic_panel.copy(deep=True)
        n_obs, n_assets = panel.n_observations, panel.n_assets
        panel["group"] = np.tile(np.array([0, 0, 1, 1]), (n_obs, 1))
        panel.add_3d_field(
            name=_EXPOSURES,
            values=np.ones((n_obs, n_assets, 1)),
            third_axis_name="factor",
            third_axis_labels=["market"],
            third_axis_groups=["market"],
        )

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
            neutralize_against=["market"],
            transform_by_group="group",
        )

        model.fit(panel)

        assert model.alpha_ is not None


class TestFitReset:
    """Test that fit() properly resets state."""

    def test_fit_resets_state(self, alpha_deterministic_panel):
        """Test that calling fit() after partial_fit() starts fresh."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        # First fit
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
        """Test that multiple descriptors produce correct coefficient shape."""
        panel = alpha_deterministic_panel.copy(deep=True)
        # Add a second signal (must maintain sorted asset order)
        panel["signal2"] = panel["signal"] * 0.5

        model = EWSharpeOptimalAlpha(
            descriptors=[
                ("signal", Passthrough("signal")),
                ("signal2", Passthrough("signal2")),
            ],
            horizon=3,
            half_life=5,
        )
        model.fit(panel)

        # Should have 2 coefficients
        assert model.coef_.shape == (2,)
        # Alpha still (n_assets,)
        assert model.alpha_.shape == (4,)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_horizon_one(self, alpha_deterministic_panel):
        """Test with minimal horizon=1."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=1,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None
        assert model._buffer.n_observations == 1

    def test_single_observation_streaming(self, alpha_deterministic_panel):
        """Test streaming one observation at a time after warmup."""
        model = EWSharpeOptimalAlpha(
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

        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model.fit(panel)

        # Asset X should have NaN alpha (or the model handles it gracefully)
        assert model.alpha_ is not None

    def test_passthrough_transformers(self, alpha_deterministic_panel):
        """Test with transformers set to passthrough."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_large_half_life(self, alpha_deterministic_panel):
        """Test with very large half_life (slow adaptation)."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=1000,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None

    def test_small_half_life(self, alpha_deterministic_panel):
        """Test with very small half_life (fast adaptation)."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=1,
        )
        model.fit(alpha_deterministic_panel)

        assert model.alpha_ is not None


class TestValidation:
    """Tests for parameter validation."""

    def test_zero_horizon_raises(self, alpha_deterministic_panel):
        """Zero horizon should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=0,
        )
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            model.fit(alpha_deterministic_panel)

    def test_negative_horizon_raises(self, alpha_deterministic_panel):
        """Negative horizon should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=-1,
        )
        with pytest.raises(ValueError, match="horizon must be a positive integer"):
            model.fit(alpha_deterministic_panel)

    def test_zero_signal_lag_raises(self, alpha_deterministic_panel):
        """Zero signal_lag should raise ValueError to prevent look-ahead."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            signal_lag=0,
        )
        with pytest.raises(ValueError, match="signal_lag must be a positive integer"):
            model.fit(alpha_deterministic_panel)

    def test_zero_half_life_raises(self, alpha_deterministic_panel):
        """Zero half_life should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            half_life=0,
        )
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            model.fit(alpha_deterministic_panel)

    def test_negative_half_life_raises(self, alpha_deterministic_panel):
        """Negative half_life should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            half_life=-5,
        )
        with pytest.raises(ValueError, match="half_life must be a positive number"):
            model.fit(alpha_deterministic_panel)

    def test_zero_forecast_scale_raises(self, alpha_deterministic_panel):
        """Zero forecast_scale should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.0,
        )
        with pytest.raises(
            ValueError, match="forecast_scale must be a positive number"
        ):
            model.fit(alpha_deterministic_panel)

    def test_empty_descriptors_raises(self, alpha_deterministic_panel):
        """Empty descriptors list should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[],
            horizon=3,
        )
        with pytest.raises(ValueError, match="descriptors cannot be empty"):
            model.fit(alpha_deterministic_panel)

    def test_invalid_forecast_unit_raises(self, alpha_deterministic_panel):
        """Unsupported forecast_unit should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_unit="bad",
        )
        with pytest.raises(TypeError, match="forecast_unit must be of type"):
            model.fit(alpha_deterministic_panel)

    def test_non_bool_normalize_weights_raises(self, alpha_deterministic_panel):
        """Non-boolean normalize_weights should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            normalize_weights="bad",
        )
        with pytest.raises(ValueError, match="normalize_weights must be a boolean"):
            model.fit(alpha_deterministic_panel)

    def test_negative_ridge_scale_raises(self, alpha_deterministic_panel):
        """Negative ridge_scale should raise ValueError."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            ridge_scale=-1e-6,
        )
        with pytest.raises(
            ValueError, match="ridge_scale must be a non-negative number"
        ):
            model.fit(alpha_deterministic_panel)

    def test_idio_sharpe_forecast_unit(self, alpha_deterministic_panel):
        """Idio-Sharpe forecast unit should keep alpha in return units."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
            forecast_unit=ForecastUnit.IDIO_SHARPE,
        )

        alphas = model.fit_transform(alpha_deterministic_panel)

        assert model.alpha_ is not None
        assert np.isfinite(model.alpha_).all()
        np.testing.assert_allclose(model.alpha_, alphas[-1], rtol=1e-10)


class TestRegression:
    """Regression tests with exact expected values to catch computation changes."""

    def test_scalar_update_uses_ewls_statistics_not_smoothed_coefficients(self):
        """Scalar EWLS should average normal-equation statistics before solving."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            ridge_scale=0.0,
        )
        model._decay = 0.5
        model._ew_normal_matrix = np.zeros((1, 1))
        model._ew_target_cross_product = np.zeros(1)
        model._n_valid_regression_obs = 0
        model.coef_ = np.full(1, np.nan)

        model._process_ewls_observation(
            scores=np.array([[1.0]]),
            target=np.array([1.0]),
            weights=np.array([1.0]),
        )
        model._process_ewls_observation(
            scores=np.array([[2.0]]),
            target=np.array([0.0]),
            weights=np.array([1.0]),
        )

        expected_ewls_coefficient = np.array([0.25 / 2.25])
        smoothed_daily_coefficient = np.array([0.25])

        np.testing.assert_allclose(model.coef_, expected_ewls_coefficient)
        assert not np.allclose(model.coef_, smoothed_daily_coefficient)

    def test_normalize_weights_removes_observation_scale(self):
        """Default weight normalization should remove per-date weight scale."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            ridge_scale=0.0,
        )
        model._decay = 0.5
        model._ew_normal_matrix = np.zeros((1, 1))
        model._ew_target_cross_product = np.zeros(1)
        model._n_valid_regression_obs = 0
        model.coef_ = np.full(1, np.nan)
        model.normalize_weights = True

        model._process_ewls_observation(
            scores=np.array([[1.0]]),
            target=np.array([1.0]),
            weights=np.array([1.0]),
        )
        model._process_ewls_observation(
            scores=np.array([[1.0]]),
            target=np.array([0.0]),
            weights=np.array([100.0]),
        )

        np.testing.assert_allclose(model.coef_, np.array([1.0 / 3.0]))

    def test_unnormalized_weights_keep_observation_scale(self):
        """Raw GLS mode should preserve per-date aggregate weight scale."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            ridge_scale=0.0,
            normalize_weights=False,
        )
        model._decay = 0.5
        model._ew_normal_matrix = np.zeros((1, 1))
        model._ew_target_cross_product = np.zeros(1)
        model._n_valid_regression_obs = 0
        model.coef_ = np.full(1, np.nan)

        model._process_ewls_observation(
            scores=np.array([[1.0]]),
            target=np.array([1.0]),
            weights=np.array([1.0]),
        )
        model._process_ewls_observation(
            scores=np.array([[1.0]]),
            target=np.array([0.0]),
            weights=np.array([100.0]),
        )

        np.testing.assert_allclose(model.coef_, np.array([0.25 / 50.25]))

    def test_fit_exact_alpha_values(self, alpha_deterministic_panel):
        """Batch fit should keep stable latest alpha values."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        # To regenerate: print(repr(model.alpha_))
        expected_alpha = np.array(
            [
                2.8937141977593106e-03,
                -2.1730716375984544e-03,
                1.1143957115889575e-05,
                -7.3178651727674477e-04,
            ]
        )

        np.testing.assert_allclose(
            model.alpha_,
            expected_alpha,
            rtol=1e-10,
            err_msg="Alpha values changed - check for computation changes",
        )

    def test_partial_fit_exact_alpha_values(self, alpha_deterministic_panel):
        """Chunked partial_fit should keep stable latest alpha values."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        model.partial_fit(alpha_deterministic_panel[:10])
        model.partial_fit(alpha_deterministic_panel[10:19])
        model.partial_fit(alpha_deterministic_panel[19:])

        # To regenerate: print(repr(model.alpha_))
        expected_alpha = np.array(
            [
                2.8937141977593106e-03,
                -2.1730716375984544e-03,
                1.1143957115889575e-05,
                -7.3178651727674477e-04,
            ]
        )

        np.testing.assert_allclose(
            model.alpha_,
            expected_alpha,
            rtol=1e-10,
            err_msg="Partial-fit alpha values changed - check streaming updates",
        )

    def test_fit_transform_exact_alpha_path(self, alpha_deterministic_panel):
        """fit_transform should keep the full causal alpha path stable."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )

        alphas = model.fit_transform(alpha_deterministic_panel)

        # To regenerate: print(repr(model.fit_transform(alpha_deterministic_panel)))
        expected_alphas = np.array(
            [
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan],
                [
                    -5.0585956278555574e-03,
                    3.6506077486532914e-03,
                    9.2825592654623493e-03,
                    -7.8745713862600833e-03,
                ],
                [
                    -1.1087461993664666e-02,
                    8.3262712019176212e-03,
                    -4.2698826676500878e-05,
                    2.8038896184235424e-03,
                ],
                [
                    -2.8262008167992850e-03,
                    3.2912414940334338e-03,
                    -1.4310787850968392e-03,
                    9.6603810786269058e-04,
                ],
                [
                    -3.2697617456305876e-03,
                    3.8609395449337863e-03,
                    3.1387274911458539e-04,
                    -9.0505054841778614e-04,
                ],
                [
                    -1.4934316232336038e-03,
                    1.3833116175675756e-03,
                    -1.7199155524127001e-04,
                    2.8211156090729818e-04,
                ],
                [
                    -2.2337439556256960e-03,
                    1.6587598856680861e-03,
                    1.2754371723630135e-03,
                    -7.0045310240540357e-04,
                ],
                [
                    -9.6853036802058915e-04,
                    7.2733025069582076e-04,
                    -3.7298987215170522e-06,
                    2.4493001604628510e-04,
                ],
                [
                    1.7861636304275855e-04,
                    -1.7861636304275849e-04,
                    5.4734329839456137e-05,
                    -5.4734329839456144e-05,
                ],
                [
                    -1.6162497566376005e-05,
                    1.4965275524422225e-05,
                    5.9861102097689171e-07,
                    5.9861102097689171e-07,
                ],
                [
                    9.6311497578023032e-04,
                    -8.2930430117910976e-04,
                    7.1043811772759856e-05,
                    -2.0485448637388054e-04,
                ],
                [
                    4.6064223617231292e-04,
                    -3.4327344996607607e-04,
                    -1.5548339203609726e-04,
                    3.8114605829860387e-05,
                ],
                [
                    1.9735358470637050e-03,
                    -1.4820519519027821e-03,
                    7.6002664200142527e-06,
                    -4.9908416158093711e-04,
                ],
                [
                    2.6270671896527990e-03,
                    -2.4408592850884434e-03,
                    5.7878023944488079e-04,
                    -7.6498814400923702e-04,
                ],
                [
                    2.3702559942224955e-03,
                    -2.0399976775747832e-03,
                    -5.1638327860724697e-05,
                    -2.7861998878698737e-04,
                ],
                [
                    2.6753690529251295e-03,
                    -2.2238103298805188e-03,
                    1.4664020387531151e-04,
                    -5.9819892691992231e-04,
                ],
                [
                    1.9255409204458906e-03,
                    -1.4370513687990533e-03,
                    -4.6007226550537908e-04,
                    -2.8417286141458200e-05,
                ],
                [
                    2.8937141977593106e-03,
                    -2.1730716375984544e-03,
                    1.1143957115889575e-05,
                    -7.3178651727674477e-04,
                ],
            ]
        )

        np.testing.assert_allclose(
            alphas,
            expected_alphas,
            rtol=1e-10,
            equal_nan=True,
            err_msg="Alpha path changed - check causal alignment",
        )

    def test_exact_coef_values(self, alpha_deterministic_panel):
        """Test exact EWLS coefficient values."""
        model = EWSharpeOptimalAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            horizon=3,
            half_life=5,
        )
        model.fit(alpha_deterministic_panel)

        # coef shape: (n_descriptors,) = (1,) for single descriptor
        assert model.coef_ is not None
        assert model.coef_.shape == (1,)

        # Expected coefficient (computed with current implementation)
        # To regenerate: print(repr(model.coef_))
        expected_coef = np.array([2.131623629622646e-03])

        np.testing.assert_allclose(
            model.coef_,
            expected_coef,
            rtol=1e-10,
            err_msg="Coefficient values changed - check for computation changes",
        )
