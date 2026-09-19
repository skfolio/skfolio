"""Unit tests for alpha forecast evaluation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import sklearn.base as skb

from skfolio._constants import (
    _BENCHMARK_WEIGHTS,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
    _REGRESSION_WEIGHTS,
)
from skfolio.alpha import (
    AlphaForecastComparison,
    FixedWeightedAlpha,
    alpha_forecast_evaluation,
)
from skfolio.alpha._evaluation import (
    _calibration_curve,
    _compute_factor_correlation_diagnostics,
    _coverage_stats,
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


class _NoFitTransform(skb.BaseEstimator):
    """Minimal estimator that intentionally lacks fit_transform."""


class _WrongShapeAlpha(skb.BaseEstimator):
    """Minimal estimator returning a malformed alpha forecast."""

    def fit_transform(self, X, y=None):
        return np.zeros((X.n_observations, X.n_assets + 1))


@pytest.fixture
def evaluation_with_diagnostics():
    """Create a small evaluation with holding-period and decay diagnostics."""
    return alpha_forecast_evaluation(
        _fixed_signal_alpha(),
        _perfect_forecast_panel(),
        holding_period=1,
        signal_lag=1,
        n_forward_periods=3,
        quantiles=(0.25,),
    )


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

        calibration = evaluation.calibration_summary()
        assert calibration.name == "Calibration"
        assert calibration["calibration_slope"] == evaluation.calibration_slope
        assert calibration["n_bins"] == len(evaluation.calibration_curve)

        coverage = evaluation.coverage_summary()
        assert coverage.name == "Coverage"
        assert coverage["mean_coverage"] == pytest.approx(
            np.nanmean(evaluation.coverage)
        )
        assert coverage["min_n_valid_assets"] == np.nanmin(evaluation.n_valid_assets)

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

    def test_cumulative_ic_plot_preserves_acronym(self, alpha_deterministic_panel):
        """Cumulative IC legend labels preserve the uppercase acronym."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            alpha_deterministic_panel,
            holding_period=1,
            signal_lag=1,
        )

        fig = evaluation.plot_cumulative_ic()

        assert [trace.name for trace in fig.data] == ["Spearman IC", "Pearson IC"]

    def test_rolling_ic_plot(self, evaluation_with_diagnostics):
        """Rolling IC plot exposes both correlation series."""
        fig = evaluation_with_diagnostics.plot_rolling_ic(window=2, title="Rolling IC")

        assert fig.layout.title.text == "Rolling IC"
        expected = (
            pd.DataFrame(
                {
                    "spearman": evaluation_with_diagnostics.spearman_ic,
                    "pearson": evaluation_with_diagnostics.pearson_ic,
                },
                index=evaluation_with_diagnostics.observations,
            )
            .rolling(2)
            .mean()
            .iloc[1:]
        )
        np.testing.assert_allclose(fig.data[0].y, expected["spearman"], equal_nan=True)
        np.testing.assert_allclose(fig.data[1].y, expected["pearson"], equal_nan=True)
        np.testing.assert_array_equal(fig.data[0].x, expected.index)
        np.testing.assert_array_equal(fig.data[1].x, expected.index)

    def test_calibration_plot(self, evaluation_with_diagnostics):
        """Calibration plot compares observed buckets with the pooled slope."""
        fig = evaluation_with_diagnostics.plot_calibration()

        assert [trace.name for trace in fig.data] == ["Observed", "Pooled Slope"]
        np.testing.assert_allclose(
            fig.data[0].x,
            evaluation_with_diagnostics.calibration_curve["mean_forecast"],
        )
        np.testing.assert_allclose(
            fig.data[0].y,
            evaluation_with_diagnostics.calibration_curve["mean_target"],
        )
        pooled_x = np.asarray(fig.data[1].x)
        np.testing.assert_allclose(
            fig.data[1].y,
            evaluation_with_diagnostics.calibration_slope * pooled_x,
        )

    def test_ic_by_holding_period_plot(self, evaluation_with_diagnostics):
        """Holding-period IC plot uses all requested forward periods."""
        fig = evaluation_with_diagnostics.plot_ic_by_holding_period()
        diagnostics = evaluation_with_diagnostics.holding_period_diagnostics

        assert [trace.name for trace in fig.data] == ["Spearman IC", "Pearson IC"]
        np.testing.assert_array_equal(fig.data[0].x, diagnostics.index)
        np.testing.assert_array_equal(fig.data[1].x, diagnostics.index)
        np.testing.assert_allclose(fig.data[0].y, diagnostics["spearman_mean_ic"])
        np.testing.assert_allclose(fig.data[1].y, diagnostics["pearson_mean_ic"])

    def test_portfolio_by_holding_period_plot(self, evaluation_with_diagnostics):
        """Holding-period portfolio plot exposes both weighting schemes."""
        fig = evaluation_with_diagnostics.plot_portfolio_by_holding_period()
        diagnostics = evaluation_with_diagnostics.holding_period_diagnostics

        assert [trace.name for trace in fig.data] == [
            "Rank-Weighted Portfolio",
            "Z-Score-Weighted Portfolio",
        ]
        np.testing.assert_array_equal(fig.data[0].x, diagnostics.index)
        np.testing.assert_array_equal(fig.data[1].x, diagnostics.index)
        np.testing.assert_allclose(
            fig.data[0].y, diagnostics["rank_weighted_portfolio_ir"]
        )
        np.testing.assert_allclose(
            fig.data[1].y, diagnostics["zscore_weighted_portfolio_ir"]
        )

    def test_ic_decay_plot(self, evaluation_with_diagnostics):
        """IC decay plot uses all requested forward periods."""
        fig = evaluation_with_diagnostics.plot_ic_decay()
        decay = evaluation_with_diagnostics.decay

        assert [trace.name for trace in fig.data] == ["Spearman IC", "Pearson IC"]
        np.testing.assert_array_equal(fig.data[0].x, decay.index)
        np.testing.assert_array_equal(fig.data[1].x, decay.index)
        np.testing.assert_allclose(fig.data[0].y, decay["spearman_mean_ic"])
        np.testing.assert_allclose(fig.data[1].y, decay["pearson_mean_ic"])

    def test_portfolio_decay_plot(self, evaluation_with_diagnostics):
        """Portfolio decay plot exposes both weighting schemes."""
        fig = evaluation_with_diagnostics.plot_portfolio_decay()
        decay = evaluation_with_diagnostics.decay

        assert [trace.name for trace in fig.data] == [
            "Rank-Weighted Portfolio",
            "Z-Score-Weighted Portfolio",
        ]
        np.testing.assert_array_equal(fig.data[0].x, decay.index)
        np.testing.assert_array_equal(fig.data[1].x, decay.index)
        np.testing.assert_allclose(fig.data[0].y, decay["rank_weighted_portfolio_ir"])
        np.testing.assert_allclose(fig.data[1].y, decay["zscore_weighted_portfolio_ir"])

    @pytest.mark.parametrize(
        "method",
        [
            "plot_ic_by_holding_period",
            "plot_portfolio_by_holding_period",
            "plot_ic_decay",
            "plot_portfolio_decay",
            "plot_factor_correlation",
        ],
    )
    def test_unavailable_diagnostic_plot_raises(
        self, method, evaluation_with_diagnostics
    ):
        evaluation = replace(
            evaluation_with_diagnostics,
            holding_period_diagnostics=pd.DataFrame(),
            decay=pd.DataFrame(),
        )

        with pytest.raises(ValueError, match=r"No .* available"):
            getattr(evaluation, method)()

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

    @pytest.mark.parametrize("quantiles", [(), (0.0,), (0.6,), (np.nan,)])
    def test_invalid_quantiles_raise(self, quantiles):
        """Quantiles must contain finite tail probabilities in (0, 0.5]."""
        with pytest.raises(ValueError, match="quantiles"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                _perfect_forecast_panel(),
                quantiles=quantiles,
            )

    def test_invalid_cs_weighting_type_raises(self):
        """Cross-sectional weighting must be an enum value or field name."""
        with pytest.raises(TypeError, match="cs_weighting"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                _perfect_forecast_panel(),
                cs_weighting=1,
            )

    def test_negative_custom_cs_weights_raise(self):
        """Finite custom cross-sectional weights must be non-negative."""
        panel = _perfect_forecast_panel()
        panel["custom_weights"] = -np.ones((panel.n_observations, panel.n_assets))

        with pytest.raises(ValueError, match="weights must be non-negative"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                panel,
                cs_weighting="custom_weights",
            )

    def test_invalid_factor_exposures_type_raises(self):
        """Factor exposure selection must be a field name or None."""
        with pytest.raises(TypeError, match="factor_exposures"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                _perfect_forecast_panel(),
                factor_exposures=1,
            )

    def test_factor_exposures_field_must_be_3d(self):
        """A selected factor exposure field must retain its factor axis."""
        with pytest.raises(TypeError, match="not a Field3D"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                _perfect_forecast_panel(),
                factor_exposures="signal",
            )

    def test_estimator_must_expose_fit_transform(self):
        """Evaluation requires the estimator's historical forecast path."""
        with pytest.raises(TypeError, match="fit_transform"):
            alpha_forecast_evaluation(
                _NoFitTransform(),
                _perfect_forecast_panel(),
            )

    def test_estimator_forecast_shape_is_validated(self):
        """Forecasts must align with both panel axes."""
        with pytest.raises(ValueError, match="must return an array with shape"):
            alpha_forecast_evaluation(
                _WrongShapeAlpha(),
                _perfect_forecast_panel(),
            )

    def test_evaluation_requires_a_valid_date(self):
        """An entirely missing forecast path cannot produce diagnostics."""
        panel = _perfect_forecast_panel()
        panel["signal"] = np.full((panel.n_observations, panel.n_assets), np.nan)

        with pytest.raises(ValueError, match="No valid evaluation date"):
            alpha_forecast_evaluation(_fixed_signal_alpha(), panel)

    @pytest.mark.parametrize(
        ("weighting", "field"),
        [
            (CSWeighting.REGRESSION, _REGRESSION_WEIGHTS),
            (CSWeighting.INVERSE_IDIO_VARIANCE, _IDIO_VARIANCES),
        ],
    )
    def test_enum_weighting_fields_are_resolved(self, weighting, field):
        """Built-in weighting rules resolve their conventional panel fields."""
        panel = _perfect_forecast_panel()
        panel[field] = np.ones((panel.n_observations, panel.n_assets))

        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            cs_weighting=weighting,
        )

        assert evaluation.cs_weighting is weighting

    def test_constant_forecast_has_one_calibration_bucket(self):
        """A constant forecast produces a single meaningful calibration bucket."""
        panel = _perfect_forecast_panel()
        panel["signal"] = np.ones((panel.n_observations, panel.n_assets))

        evaluation = alpha_forecast_evaluation(_fixed_signal_alpha(), panel)

        assert len(evaluation.calibration_curve) == 1
        assert evaluation.calibration_curve.iloc[0]["mean_forecast"] == 1.0

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
        fig = comparison.plot_cumulative_ic()
        assert [trace.name for trace in fig.data] == [
            "Alpha 1",
            "Alpha 2",
        ]
        np.testing.assert_allclose(
            fig.data[0].y, np.nancumsum(evaluation_1.spearman_ic)
        )
        np.testing.assert_allclose(
            fig.data[1].y, np.nancumsum(evaluation_2.spearman_ic)
        )
        assert comparison.plot_cumulative_returns().layout.yaxis.tickformat == ".2%"

    def test_comparison_requires_at_least_one_evaluation(self):
        """A comparison without evaluations has no meaningful output."""
        with pytest.raises(ValueError, match="at least one"):
            AlphaForecastComparison([])

    def test_comparison_requires_one_name_per_evaluation(
        self, evaluation_with_diagnostics
    ):
        """Explicit comparison names must align one-to-one with evaluations."""
        with pytest.raises(ValueError, match="names has length"):
            AlphaForecastComparison([evaluation_with_diagnostics], names=[])

    def test_comparison_supplies_a_name_for_unnamed_evaluation(
        self, evaluation_with_diagnostics
    ):
        """Unnamed evaluations receive deterministic display names."""
        evaluation = replace(evaluation_with_diagnostics, name=None)

        comparison = AlphaForecastComparison([evaluation])

        assert comparison.plot_cumulative_ic().data[0].name == "Estimator 0"

    def test_comparison_uses_explicit_names(self, evaluation_with_diagnostics):
        """Explicit names override the evaluation names in every summary."""
        comparison = AlphaForecastComparison(
            [evaluation_with_diagnostics, evaluation_with_diagnostics],
            names=["first", "second"],
        )

        assert list(comparison.ic_summary().columns.levels[0]) == [
            "first",
            "second",
        ]
        assert [trace.name for trace in comparison.plot_cumulative_ic().data] == [
            "First",
            "Second",
        ]

    def test_three_dimensional_cs_weighting_field_raises(self):
        """Cross-sectional weights must be a 2D field."""
        panel = _factor_correlation_panel()

        with pytest.raises(ValueError, match="must have shape"):
            alpha_forecast_evaluation(
                _fixed_signal_alpha(),
                panel,
                cs_weighting=_EXPOSURES,
            )

    def test_factor_exposures_none_disables_factor_diagnostics(self):
        """`factor_exposures=None` skips factor correlations even with exposures."""
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            _factor_correlation_panel(),
            factor_exposures=None,
        )

        assert evaluation.factor_correlation is None
        assert evaluation.factor_correlation_method is None
        assert evaluation.factor_names.size == 0

    def test_factor_exposures_without_factors_gives_empty_correlations(self):
        """A 3D exposure field with no factors yields zero-width correlations."""
        panel = _factor_correlation_panel()
        panel.add_3d_field(
            "empty",
            np.zeros((panel.n_observations, panel.n_assets, 0)),
            third_axis_name="factors",
            third_axis_labels=[],
            third_axis_groups=[],
        )

        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            factor_exposures="empty",
        )

        assert evaluation.factor_correlation.shape == (panel.n_observations, 0)
        assert evaluation.factor_correlation_method is CorrelationMethod.PEARSON
        assert evaluation.factor_names.size == 0
        assert evaluation.factor_families.size == 0
        assert evaluation.factor_correlation_summary().empty

    def test_factor_correlation_diagnostics_validate_exposure_shape(self):
        """Exposures must be aligned with the alpha array on the first two axes."""
        panel = _factor_correlation_panel()
        alpha = panel["signal"][:-1]

        with pytest.raises(ValueError, match="first two axes matching alpha"):
            _compute_factor_correlation_diagnostics(
                alpha=alpha,
                exposures_field=panel.get_field(_EXPOSURES),
                estimation_mask=np.ones(alpha.shape, dtype=bool),
                cs_weights=None,
                method=CorrelationMethod.PEARSON,
                min_count=3,
            )

    def test_calibration_curve_without_valid_pairs_is_empty(self):
        """All-NaN forecasts produce an empty calibration curve with fixed columns."""
        curve = _calibration_curve(np.full((3, 4), np.nan), np.ones((3, 4)))

        assert curve.empty
        assert list(curve.columns) == [
            "bucket",
            "mean_forecast",
            "mean_target",
            "n_observations",
        ]

    def test_coverage_stats(self):
        """Coverage statistics summarize the coverage path and valid asset count."""
        stats = _coverage_stats(
            np.array([0.5, np.nan, 1.0]), np.array([2, 4, 6], dtype=int)
        )

        assert stats["mean"] == pytest.approx(0.75)
        assert stats["std"] == pytest.approx(np.std([0.5, 1.0], ddof=1))
        assert np.isnan(stats["ir"])
        assert np.isnan(stats["hit_rate"])
        assert stats["n_valid_assets"] == pytest.approx(4.0)
