"""Unit tests for alpha forecast evaluation."""

import numpy as np
import pandas as pd
import pytest

from skfolio._constants import _BENCHMARK_WEIGHTS, _EXPOSURES, _IDIO_RETURNS
from skfolio.alpha import (
    AlphaForecastComparison,
    FixedWeightedAlpha,
    alpha_forecast_evaluation,
)
from skfolio.containers import AssetPanel
from skfolio.descriptor import Passthrough
from skfolio.utils.stats import (
    CSWeighting,
    CorrelationMethod,
    _forward_mean_return,
    cs_pearson_correlation,
    cs_rank,
    cs_spearman_correlation,
    safe_divide,
)


def _fixed_signal_alpha() -> FixedWeightedAlpha:
    """Create a fixed alpha estimator from the `signal` field."""
    return FixedWeightedAlpha(
        descriptors=[("signal", Passthrough("signal"))],
        forecast_scale=1.0,
        outlier_transformer="passthrough",
        scoring_transformer="passthrough",
    )


def _perfect_forecast_panel() -> AssetPanel:
    """Create a panel where the signal is aligned with next-period returns."""
    n_obs = 12
    n_assets = 5
    observations = pd.bdate_range("2020-01-01", periods=n_obs).to_numpy()
    assets = np.array([f"asset_{i}" for i in range(n_assets)])
    base = np.linspace(-0.02, 0.02, n_assets)
    idio_returns = np.vstack([base + 0.001 * i for i in range(n_obs)])
    signal = np.vstack([idio_returns[1:], np.full((1, n_assets), np.nan)])
    benchmark_weights = np.ones((n_obs, n_assets)) / n_assets
    regression_weights = np.linspace(1.0, 2.0, n_assets)[None, :].repeat(n_obs, axis=0)

    panel = AssetPanel(
        fields={
            _IDIO_RETURNS: idio_returns,
            "signal": signal,
            _BENCHMARK_WEIGHTS: benchmark_weights,
            "custom_weights": regression_weights,
        },
        observations=observations,
        asset_names=assets,
    )
    return panel


def _decay_common_sample_panel() -> AssetPanel:
    """Create a panel that distinguishes common-sample decay diagnostics."""
    n_obs = 8
    n_assets = 4
    observations = pd.bdate_range("2020-01-01", periods=n_obs).to_numpy()
    assets = np.array([f"asset_{i}" for i in range(n_assets)])
    base = np.arange(n_assets, dtype=float)
    signal = np.tile(base, (n_obs, 1))
    idio_returns = np.vstack([base if i <= 5 else -base for i in range(n_obs)])

    return AssetPanel(
        fields={
            _IDIO_RETURNS: idio_returns,
            "signal": signal,
        },
        observations=observations,
        asset_names=assets,
    )


def _factor_correlation_panel() -> AssetPanel:
    """Create a panel with factor exposures aligned with the signal."""
    n_obs = 9
    n_assets = 6
    observations = pd.bdate_range("2020-01-01", periods=n_obs).to_numpy()
    assets = np.array([f"asset_{i}" for i in range(n_assets)])
    base = np.linspace(-1.0, 1.0, n_assets)
    signal = np.vstack([base + 0.01 * i for i in range(n_obs)])
    idio_returns = np.vstack([0.001 * i + base for i in range(n_obs)])
    custom_weights = np.linspace(1.0, 2.0, n_assets)[None, :].repeat(n_obs, axis=0)
    mixed = np.array([1.0, -1.0, 0.5, -0.5, 1.5, -1.5])
    exposures = np.stack(
        [
            signal,
            -signal,
            np.tile(mixed, (n_obs, 1)),
        ],
        axis=2,
    )

    panel = AssetPanel(
        fields={
            _IDIO_RETURNS: idio_returns,
            "signal": signal,
            "custom_weights": custom_weights,
        },
        observations=observations,
        asset_names=assets,
    )
    panel.add_3d_field(
        _EXPOSURES,
        exposures,
        third_axis_name="factors",
        third_axis_labels=["positive_factor", "negative_factor", "mixed_factor"],
        third_axis_groups=["style", "style", "risk"],
    )
    return panel


class TestAlphaForecastEvaluation:
    """Test alpha forecast evaluation diagnostics."""

    def test_ic_matches_stats_helpers(self, alpha_deterministic_panel):
        """Spearman and Pearson IC reuse cross-sectional stats helpers."""
        model = _fixed_signal_alpha()
        evaluation = alpha_forecast_evaluation(
            model,
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
            cs_weighting=CSWeighting.BENCHMARK,
        )

        alphas = model.fit_transform(alpha_deterministic_panel)
        target = _forward_mean_return(
            alpha_deterministic_panel[_IDIO_RETURNS], horizon=1, lag=1
        )
        eval_idx = np.arange(0, alpha_deterministic_panel.n_observations - 1)
        expected_spearman_ic = cs_spearman_correlation(
            alphas[eval_idx], target[eval_idx], axis=1
        )
        expected_pearson_ic = cs_pearson_correlation(
            alphas[eval_idx],
            target[eval_idx],
            weights=alpha_deterministic_panel[_BENCHMARK_WEIGHTS][eval_idx],
            axis=1,
        )

        np.testing.assert_allclose(
            evaluation.spearman_ic, expected_spearman_ic, equal_nan=True
        )
        np.testing.assert_allclose(
            evaluation.pearson_ic, expected_pearson_ic, equal_nan=True
        )

    def test_summary_methods_return_expected_labels(self, alpha_deterministic_panel):
        """Dedicated summary tables expose the core quant diagnostics."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
            n_forward_periods=3,
            quantiles=(0.25,),
        )

        assert list(evaluation.ic_summary().index) == ["spearman_ic", "pearson_ic"]
        assert list(evaluation.ic_summary().columns) == [
            "mean",
            "std",
            "icir",
            "t_stat",
            "hit_rate",
        ]
        assert list(evaluation.portfolio_summary().index) == [
            "rank_weighted_portfolio",
            "zscore_weighted_portfolio",
        ]
        assert list(evaluation.portfolio_summary().columns) == [
            "annualized_mean",
            "annualized_vol",
            "annualized_ir",
            "hit_rate",
            "mean_turnover",
        ]
        assert evaluation.quantile_summary().index.name == "quantile"
        assert list(evaluation.quantile_summary().index) == [0.25]
        assert list(evaluation.quantile_summary().columns) == [
            "annualized_mean",
            "annualized_vol",
            "annualized_ir",
            "hit_rate",
        ]
        assert list(evaluation.holding_period_summary().index) == [1, 2, 3]
        assert list(evaluation.decay_summary().index) == [1, 2, 3]
        assert evaluation.holding_period_summary().index.name == "holding_period"
        assert evaluation.decay_summary().index.name == "period"
        assert "spearman_mean_ic" in evaluation.holding_period_summary().columns
        assert "spearman_mean_ic" in evaluation.decay_summary().columns
        assert "spearman_icir" in evaluation.decay_summary().columns
        assert "spearman_ic_t_stat" in evaluation.decay_summary().columns
        assert list(evaluation.factor_correlation_summary().columns) == [
            "mean",
            "std",
            "ir",
            "t_stat",
            "hit_rate",
        ]
        assert evaluation.factor_correlation_summary().empty

    def test_ic_summary_t_stat(self, alpha_deterministic_panel):
        """IC t-stat is computed from finite evaluation dates."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
        )

        spearman_ic = evaluation.spearman_ic[np.isfinite(evaluation.spearman_ic)]
        expected = np.nanmean(spearman_ic) / (
            np.nanstd(spearman_ic, ddof=1) / np.sqrt(spearman_ic.size)
        )

        assert evaluation.ic_summary().loc["spearman_ic", "t_stat"] == pytest.approx(
            expected
        )

    def test_return_summaries_are_annualized(self, alpha_deterministic_panel):
        """Portfolio and quantile summaries annualize return statistics."""
        evaluation_period = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
            quantiles=(0.25,),
            annualization_factor=1.0,
        )
        evaluation_annualized = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
            quantiles=(0.25,),
            annualization_factor=4.0,
        )

        period_portfolio = evaluation_period.portfolio_summary()
        annualized_portfolio = evaluation_annualized.portfolio_summary()
        np.testing.assert_allclose(
            annualized_portfolio["annualized_mean"],
            4.0 * period_portfolio["annualized_mean"],
        )
        np.testing.assert_allclose(
            annualized_portfolio["annualized_vol"],
            2.0 * period_portfolio["annualized_vol"],
        )
        np.testing.assert_allclose(
            annualized_portfolio["annualized_ir"],
            2.0 * period_portfolio["annualized_ir"],
        )
        np.testing.assert_allclose(
            annualized_portfolio["mean_turnover"],
            period_portfolio["mean_turnover"],
        )

        period_quantile = evaluation_period.quantile_summary()
        annualized_quantile = evaluation_annualized.quantile_summary()
        np.testing.assert_allclose(
            annualized_quantile["annualized_mean"],
            4.0 * period_quantile["annualized_mean"],
        )
        np.testing.assert_allclose(
            annualized_quantile["annualized_vol"],
            2.0 * period_quantile["annualized_vol"],
        )
        np.testing.assert_allclose(
            annualized_quantile["annualized_ir"],
            2.0 * period_quantile["annualized_ir"],
        )

    def test_simple_portfolios_use_200_percent_gross(self):
        """Simple alpha portfolio returns use 100% long and 100% short exposure."""
        panel = _perfect_forecast_panel()
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            holding_period=1,
            signal_lag=1,
        )

        alpha = panel["signal"][:-1]
        target = _forward_mean_return(panel[_IDIO_RETURNS], horizon=1, lag=1)[:-1]

        ranks = cs_rank(alpha, axis=1)
        rank_centered = ranks - np.nanmean(ranks, axis=1, keepdims=True)
        rank_centered = np.where(np.isfinite(rank_centered), rank_centered, 0.0)
        rank_gross = np.sum(np.abs(rank_centered), axis=1, keepdims=True)
        rank_weights = 2.0 * safe_divide(rank_centered, rank_gross, fill_value=0.0)

        zscore_centered = alpha - np.nanmean(alpha, axis=1, keepdims=True)
        zscore_centered = np.where(np.isfinite(zscore_centered), zscore_centered, 0.0)
        zscore_gross = np.sum(np.abs(zscore_centered), axis=1, keepdims=True)
        zscore_weights = 2.0 * safe_divide(
            zscore_centered, zscore_gross, fill_value=0.0
        )

        np.testing.assert_allclose(np.sum(np.abs(rank_weights), axis=1), 2.0)
        np.testing.assert_allclose(np.sum(np.abs(zscore_weights), axis=1), 2.0)
        np.testing.assert_allclose(
            evaluation.rank_weighted_portfolio_return,
            np.sum(rank_weights * target, axis=1),
        )
        np.testing.assert_allclose(
            evaluation.zscore_weighted_portfolio_return,
            np.sum(zscore_weights * target, axis=1),
        )

    def test_return_plots_use_percent_axis(self, alpha_deterministic_panel):
        """Return plots follow the Portfolio percentage axis convention."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
            quantiles=(0.25,),
        )

        assert evaluation.plot_cumulative_returns().layout.yaxis.tickformat == ".2%"
        assert evaluation.plot_quantile_returns().layout.yaxis.tickformat == ".2%"

    def test_evaluation_step_subsamples_evaluation_dates(
        self, alpha_deterministic_panel
    ):
        """evaluation_step controls spacing between evaluated forecast dates."""
        evaluation_daily = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
            evaluation_step=1,
        )
        evaluation_weekly = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
            evaluation_step=5,
        )
        evaluation_default = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=3,
            signal_lag=1,
        )

        assert len(evaluation_weekly.observations) < len(evaluation_daily.observations)
        assert evaluation_weekly.evaluation_step == 5
        assert evaluation_default.evaluation_step == 3
        np.testing.assert_array_equal(
            evaluation_weekly.observations, evaluation_daily.observations[::5]
        )

    def test_custom_cs_weighting_field(self):
        """A string cs_weighting is interpreted as an AssetPanel field."""
        panel = _perfect_forecast_panel()
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            holding_period=1,
            signal_lag=1,
            cs_weighting="custom_weights",
        )
        target = _forward_mean_return(panel[_IDIO_RETURNS], horizon=1, lag=1)
        expected = cs_pearson_correlation(
            panel["signal"][:-1],
            target[:-1],
            weights=panel["custom_weights"][:-1],
            axis=1,
        )

        np.testing.assert_allclose(evaluation.pearson_ic, expected, equal_nan=True)

    def test_factor_correlation_uses_all_forecast_dates(self):
        """Factor correlations are contemporaneous alpha-exposure diagnostics."""
        panel = _factor_correlation_panel()
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            holding_period=3,
            signal_lag=1,
            evaluation_step=5,
            factor_correlation_method=CorrelationMethod.SPEARMAN,
        )
        alpha = panel["signal"]
        expected = cs_spearman_correlation(
            alpha[:, :, np.newaxis],
            panel[_EXPOSURES],
            axis=1,
        )

        assert len(evaluation.observations) < panel.n_observations
        assert evaluation.factor_correlation.shape == (
            panel.n_observations,
            3,
        )
        assert evaluation.factor_correlation_method is CorrelationMethod.SPEARMAN
        np.testing.assert_allclose(
            evaluation.factor_correlation, expected, equal_nan=True
        )

    def test_factor_correlation_summary_and_filters(self):
        """Factor correlation summary exposes factor rows and family filters."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            _factor_correlation_panel(),
            holding_period=1,
            signal_lag=1,
        )

        summary = evaluation.factor_correlation_summary()
        assert list(summary.index) == [
            "positive_factor",
            "negative_factor",
            "mixed_factor",
        ]
        assert summary.loc["positive_factor", "mean"] == pytest.approx(1.0)
        assert summary.loc["positive_factor", "hit_rate"] == pytest.approx(1.0)
        assert summary.loc["negative_factor", "mean"] == pytest.approx(-1.0)
        assert summary.loc["negative_factor", "hit_rate"] == pytest.approx(0.0)

        style_summary = evaluation.factor_correlation_summary(families="style")
        assert list(style_summary.index) == ["positive_factor", "negative_factor"]

        factor_summary = evaluation.factor_correlation_summary(factors=["mixed_factor"])
        assert list(factor_summary.index) == ["mixed_factor"]

    def test_invalid_factor_correlation_method_raises(self):
        """Factor correlation method must be Pearson, Spearman or None."""
        with pytest.raises(TypeError, match="factor_correlation_method"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                _factor_correlation_panel(),
                holding_period=1,
                signal_lag=1,
                factor_correlation_method="both",
            )

    def test_factor_correlation_uses_custom_weights_for_pearson(self):
        """Pearson factor correlations use the evaluation cross-sectional weights."""
        panel = _factor_correlation_panel()
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            holding_period=1,
            signal_lag=1,
            cs_weighting="custom_weights",
        )
        weights = panel["custom_weights"]
        expected = cs_pearson_correlation(
            panel["signal"][:, :, np.newaxis],
            panel[_EXPOSURES],
            weights=weights,
            axis=1,
        )

        np.testing.assert_allclose(
            evaluation.factor_correlation, expected, equal_nan=True
        )
        assert evaluation.factor_correlation_method is CorrelationMethod.PEARSON

    def test_factor_correlation_plot(self):
        """Factor correlation plot shows top mean correlations as bars."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            _factor_correlation_panel(),
            holding_period=1,
            signal_lag=1,
        )

        fig = evaluation.plot_factor_correlation(top_n=2)

        assert fig.data[0].orientation is None
        assert len(fig.data[0].x) == 2
        assert list(fig.data[0].x) == ["Positive Factor", "Negative Factor"]
        assert fig.layout.yaxis.range == (-1.0, 1.0)

    def test_perfect_forecast_calibration_slope(self):
        """A forecast already in target units has a calibration slope near one."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            _perfect_forecast_panel(),
            holding_period=1,
            signal_lag=1,
        )

        np.testing.assert_allclose(evaluation.calibration_slope, 1.0)
        assert np.nanmean(evaluation.spearman_ic) == pytest.approx(1.0)
        assert np.nanmean(evaluation.rank_weighted_portfolio_return) > 0

    def test_hit_rates_ignore_nan_diagnostics(self):
        """Hit rates are computed over finite diagnostics only."""
        panel = _perfect_forecast_panel()
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            holding_period=1,
            signal_lag=1,
            min_count=panel.n_assets + 1,
        )

        assert np.isnan(evaluation.ic_summary().loc["spearman_ic", "hit_rate"])
        assert np.isnan(evaluation.ic_summary().loc["spearman_ic", "t_stat"])
        assert np.isnan(
            evaluation.portfolio_summary().loc["rank_weighted_portfolio", "hit_rate"]
        )

    def test_decay_uses_common_evaluation_dates(self):
        """Decay periods are summarized on comparable evaluation dates."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            _decay_common_sample_panel(),
            holding_period=1,
            signal_lag=1,
            n_forward_periods=3,
        )

        decay = evaluation.decay_summary()
        assert list(decay.index) == [1, 2, 3]
        assert "target_start_offset" not in decay.columns
        assert "target_end_offset" not in decay.columns
        assert decay.loc[1, "spearman_mean_ic"] == pytest.approx(1.0)

    def test_missing_target_field_raises(self, alpha_deterministic_panel):
        """target must be a field in the AssetPanel."""
        with pytest.raises(ValueError, match="Required fields are missing"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                alpha_deterministic_panel,
                target="missing",
            )

    def test_missing_custom_factor_exposures_raises(self):
        """Custom factor exposure fields must exist."""
        with pytest.raises(ValueError, match="factor_exposures"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                _perfect_forecast_panel(),
                factor_exposures="missing_exposures",
            )

    def test_zero_annualization_factor_raises(self, alpha_deterministic_panel):
        """Zero annualization_factor should raise ValueError."""
        with pytest.raises(
            ValueError, match="annualization_factor must be a positive number"
        ):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                alpha_deterministic_panel,
                annualization_factor=0.0,
            )

    def test_comparison_concatenates_dedicated_summaries(
        self, alpha_deterministic_panel
    ):
        """Comparison wraps several evaluation results."""
        evaluation_1 = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
            name="alpha_1",
        )
        evaluation_2 = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=3,
            signal_lag=1,
            name="alpha_2",
        )

        comparison = AlphaForecastComparison([evaluation_1, evaluation_2])

        assert list(comparison.ic_summary().columns.levels[0]) == [
            "alpha_1",
            "alpha_2",
        ]
        assert list(comparison.portfolio_summary().columns.levels[0]) == [
            "alpha_1",
            "alpha_2",
        ]
        assert comparison.plot_cumulative_returns().layout.yaxis.tickformat == ".2%"
