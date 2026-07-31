"""Unit tests for FixedWeightedAlpha."""

import numpy as np
import pytest

from skfolio._constants import _IDIO_VARIANCES
from skfolio.alpha import FixedWeightedAlpha, ForecastUnit
from skfolio.descriptor import Passthrough


class TestBasicFunctionality:
    """Test basic fixed-weighted alpha behavior."""

    def test_fit_returns_self(self, alpha_deterministic_panel):
        """fit returns self for method chaining."""
        model = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.01,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )

        assert model.fit(alpha_deterministic_panel) is model
        assert model.alpha_.shape == (alpha_deterministic_panel.n_assets,)

    def test_fit_transform_returns_alpha_path(self, alpha_deterministic_panel):
        """fit_transform returns one alpha row per observation."""
        model = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.01,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )

        alphas = model.fit_transform(alpha_deterministic_panel)

        assert alphas.shape == (
            alpha_deterministic_panel.n_observations,
            alpha_deterministic_panel.n_assets,
        )
        np.testing.assert_allclose(model.alpha_, alphas[-1])

    def test_partial_fit_transform_matches_batch(self, alpha_deterministic_panel):
        """Chunked fixed-weighted alpha path matches batch path."""
        model_batch = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.01,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )
        batch_path = model_batch.fit_transform(alpha_deterministic_panel)

        model_stream = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.01,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )
        stream_path = np.vstack(
            [
                model_stream.partial_fit_transform(alpha_deterministic_panel[:7]),
                model_stream.partial_fit_transform(alpha_deterministic_panel[7:]),
            ]
        )

        np.testing.assert_allclose(batch_path, stream_path, equal_nan=True)


class TestFixedWeights:
    """Test signed finite-aware descriptor weighting."""

    def test_signed_weights_use_absolute_normalization(self, alpha_deterministic_panel):
        """Signed weights are normalized by available absolute weight."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel["signal2"] = 2.0 * panel["signal"]

        model = FixedWeightedAlpha(
            descriptors=[
                ("signal", Passthrough("signal")),
                ("signal2", Passthrough("signal2")),
            ],
            weights=[2.0, -1.0],
            forecast_scale=0.01,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )

        alphas = model.fit_transform(panel)
        expected_composite = (2.0 * panel["signal"] - panel["signal2"]) / 3.0
        expected_alpha = 0.01 * expected_composite

        np.testing.assert_allclose(alphas, expected_alpha)
        np.testing.assert_allclose(model.composite_score_, expected_composite)

    def test_missing_scores_renormalize_available_abs_weight(
        self, alpha_deterministic_panel
    ):
        """Missing descriptor scores are excluded and remaining weights renormalize."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel["signal2"] = 2.0 * panel["signal"]
        panel["signal2"][:, 0] = np.nan

        model = FixedWeightedAlpha(
            descriptors=[
                ("signal", Passthrough("signal")),
                ("signal2", Passthrough("signal2")),
            ],
            weights=[2.0, -1.0],
            forecast_scale=1.0,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )

        alphas = model.fit_transform(panel)

        np.testing.assert_allclose(alphas[:, 0], panel["signal"][:, 0])
        expected_other = (2.0 * panel["signal"][:, 1:] - panel["signal2"][:, 1:]) / 3.0
        np.testing.assert_allclose(alphas[:, 1:], expected_other)

    def test_min_coverage_uses_absolute_weight(self, alpha_deterministic_panel):
        """min_coverage is based on available absolute descriptor weight."""
        panel = alpha_deterministic_panel.copy(deep=True)
        panel["signal2"] = 2.0 * panel["signal"]
        panel["signal2"][:, 0] = np.nan

        model = FixedWeightedAlpha(
            descriptors=[
                ("signal", Passthrough("signal")),
                ("signal2", Passthrough("signal2")),
            ],
            weights=[1.0, -3.0],
            forecast_scale=1.0,
            min_coverage=0.5,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )

        alphas = model.fit_transform(panel)

        assert np.isnan(alphas[:, 0]).all()
        assert np.isfinite(alphas[:, 1:]).all()


class TestForecastUnit:
    """Test forecast unit conversion."""

    def test_idio_return_forecast_unit(self, alpha_deterministic_panel):
        """idio_return forecasts are published directly as alpha."""
        model = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.01,
            forecast_unit=ForecastUnit.IDIO_RETURN,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )

        alphas = model.fit_transform(alpha_deterministic_panel)

        np.testing.assert_allclose(alphas, 0.01 * alpha_deterministic_panel["signal"])

    def test_idio_sharpe_forecast_unit(self, alpha_deterministic_panel):
        """idio_sharpe forecasts are converted to return units with idio vol."""
        model = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.01,
            forecast_unit=ForecastUnit.IDIO_SHARPE,
            outlier_transformer="passthrough",
            scoring_transformer="passthrough",
        )

        alphas = model.fit_transform(alpha_deterministic_panel)
        expected = (
            0.01
            * alpha_deterministic_panel["signal"]
            * np.sqrt(alpha_deterministic_panel[_IDIO_VARIANCES])
        )

        np.testing.assert_allclose(alphas, expected)


class TestValidation:
    """Test parameter validation."""

    def test_empty_descriptors_raises(self, alpha_deterministic_panel):
        """Empty descriptors should raise ValueError."""
        model = FixedWeightedAlpha(descriptors=[], forecast_scale=0.01)

        with pytest.raises(ValueError, match="descriptors cannot be empty"):
            model.fit(alpha_deterministic_panel)

    def test_invalid_forecast_unit_raises(self, alpha_deterministic_panel):
        """Unsupported forecast_unit should raise ValueError."""
        model = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.01,
            forecast_unit="bad",
        )

        with pytest.raises(TypeError, match="forecast_unit must be of type"):
            model.fit(alpha_deterministic_panel)

    def test_non_positive_forecast_scale_raises(self, alpha_deterministic_panel):
        """forecast_scale must be positive."""
        model = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            forecast_scale=0.0,
        )

        with pytest.raises(
            ValueError, match="forecast_scale must be a positive number"
        ):
            model.fit(alpha_deterministic_panel)

    def test_bad_weights_shape_raises(self, alpha_deterministic_panel):
        """weights length must match descriptors length."""
        model = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            weights=[1.0, -1.0],
            forecast_scale=0.01,
        )

        with pytest.raises(ValueError, match="same length as descriptors"):
            model.fit(alpha_deterministic_panel)

    def test_all_zero_weights_raises(self, alpha_deterministic_panel):
        """At least one absolute descriptor weight must be positive."""
        model = FixedWeightedAlpha(
            descriptors=[("signal", Passthrough("signal"))],
            weights=[0.0],
            forecast_scale=0.01,
        )

        with pytest.raises(ValueError, match="sum\\(abs\\(weights\\)\\)"):
            model.fit(alpha_deterministic_panel)
