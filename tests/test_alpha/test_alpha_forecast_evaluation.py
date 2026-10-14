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
    _correlation_stats,
    _coverage_stats,
    _mean_inference,
    _quadratic_spectral_basis,
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


def _noisy_forecast_panel(n_obs: int = 150, n_assets: int = 12) -> AssetPanel:
    """Create a panel where a noisy signal predicts next-period returns.

    Use a fixed seed. Signals on the last date are NaN because there is no
    next-period return.
    """
    rng = np.random.default_rng(7)
    observations = pd.bdate_range("2020-01-01", periods=n_obs).to_numpy()
    assets = np.array([f"asset_{i}" for i in range(n_assets)])
    idio_returns = 0.01 * rng.standard_normal((n_obs, n_assets))
    signal = np.vstack(
        [
            0.3 * idio_returns[1:] + 0.01 * rng.standard_normal((n_obs - 1, n_assets)),
            np.full((1, n_assets), np.nan),
        ]
    )
    return AssetPanel(
        fields={_IDIO_RETURNS: idio_returns, "signal": signal},
        observations=observations,
        asset_names=assets,
    )


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
            "p_value",
            "mean_ci_lower",
            "mean_ci_upper",
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
        for prefix in ("spearman", "pearson"):
            for suffix in (
                "ic_t_stat",
                "ic_p_value",
                "mean_ic_ci_lower",
                "mean_ic_ci_upper",
            ):
                assert f"{prefix}_{suffix}" in evaluation.decay_summary().columns
                assert (
                    f"{prefix}_{suffix}" in evaluation.holding_period_summary().columns
                )
        assert list(evaluation.factor_correlation_summary().columns) == [
            "mean",
            "std",
            "ir",
            "t_stat",
            "p_value",
            "mean_ci_lower",
            "mean_ci_upper",
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


class TestHacInference:
    """Check HAC statistics against direct kernel and bootstrap calculations."""

    @staticmethod
    def _overlapping_series(n=300, window=10, mean=0.1, seed=1):
        """Generate `n` overlapping Gaussian averages with the requested mean."""
        noise = np.random.default_rng(seed).normal(size=n + window - 1)
        return np.convolve(noise, np.ones(window) / window, mode="valid") + mean

    @staticmethod
    def _reference_kernel(positions, span):
        """Build the full quadratic spectral kernel matrix for comparison.

        Use bandwidth `0.1 * span`, with `positions` and `span` as defined in
        `_quadratic_spectral_basis`.
        """
        distance = np.abs(np.subtract.outer(positions, positions)) / (0.1 * span)
        kernel = np.ones_like(distance, dtype=float)
        nonzero = distance != 0
        z = 6 * np.pi * distance[nonzero] / 5
        kernel[nonzero] = 3 * (np.sin(z) / z - np.cos(z)) / z**2
        return kernel

    @classmethod
    def _reference_t_stat(cls, arr):
        """Compute the HAC t-statistic directly from the full kernel matrix.

        Keep the gaps left by non-finite values in `arr`.
        """
        positions = np.flatnonzero(np.isfinite(arr))
        positions -= positions[0]
        values = arr[np.isfinite(arr)]
        centered_values = values - values.mean()
        kernel = cls._reference_kernel(positions, positions[-1] + 1)
        return (
            values.mean()
            * values.size
            / np.sqrt(centered_values @ kernel @ centered_values)
        )

    @pytest.mark.parametrize("span", [20, 250, 5000])
    def test_quadrature_matches_exact_kernel(self, span):
        positions = np.unique(np.linspace(0, span - 1, min(span, 101), dtype=int))
        kernel_basis = _quadratic_spectral_basis(positions, span)
        np.testing.assert_allclose(
            kernel_basis @ kernel_basis.T,
            self._reference_kernel(positions, span),
            atol=1e-12,
            rtol=1e-10,
        )

    @pytest.mark.parametrize("gaps", [False, True])
    def test_t_stat_matches_quadratic_spectral_reference(self, gaps):
        arr = self._overlapping_series(n=200, window=5, seed=2)
        if gaps:
            arr[[40, 41, 150]] = np.nan
        stats = _correlation_stats(arr, ratio_name="icir")
        assert stats["t_stat"] == pytest.approx(self._reference_t_stat(arr), rel=1e-10)
        if gaps:
            compacted = _correlation_stats(arr[np.isfinite(arr)], ratio_name="icir")
            assert stats["t_stat"] != pytest.approx(compacted["t_stat"], rel=1e-9)

    @pytest.mark.parametrize("gaps", [False, True])
    @pytest.mark.parametrize("changing_variance", [False, True])
    def test_calibration_matches_direct_independent_wild_bootstrap(
        self, gaps, changing_variance
    ):
        # Build full bootstrap samples as an independent check of the shortcut.
        arr = self._overlapping_series(n=80, window=5, mean=0.12, seed=21)
        if changing_variance:
            arr *= np.where(np.arange(arr.size) < 55, 0.2, 1.0)
        if gaps:
            arr[[8, 9, 10, 34, 55, 56]] = np.nan
        positions = np.flatnonzero(np.isfinite(arr))
        values = arr[positions]
        n = values.size
        kernel = self._reference_kernel(positions, arr.size)
        centered_values = values - values.mean()
        standard_error = np.sqrt(centered_values @ kernel @ centered_values) / n
        t_stat = values.mean() / standard_error
        draws = np.random.default_rng(12).standard_normal((39_999, n)) * centered_values
        boot_mean = draws.mean(axis=1)
        draws -= boot_mean[:, None]
        boot_scale = np.sqrt(np.sum((draws @ kernel) * draws, axis=1)) / n
        abs_boot = np.abs(boot_mean / boot_scale)
        reference_p = (1 + np.count_nonzero(abs_boot >= abs(t_stat))) / 40_000
        critical = np.sort(abs_boot)[38_000]
        stats = _mean_inference(arr)
        assert stats["t_stat"] == pytest.approx(t_stat, rel=1e-10)
        # Different random draws produce slightly different p-values and intervals.
        assert stats["p_value"] == pytest.approx(reference_p, abs=0.02)
        half_width = (stats["mean_ci_upper"] - stats["mean_ci_lower"]) / 2
        assert half_width == pytest.approx(critical * standard_error, rel=0.06)

    def test_overlap_shrinks_t_stat_versus_independent_dates(self):
        arr = self._overlapping_series(n=2000, mean=0.1)
        t_stat = _mean_inference(arr)["t_stat"]
        iid = np.mean(arr) / (np.std(arr, ddof=1) / np.sqrt(arr.size))
        assert 0 < t_stat < iid

    @pytest.mark.parametrize("mean", [-0.3, 0.0, 0.3])
    def test_p_value_and_confidence_interval(self, mean):
        arr = self._overlapping_series(mean=0.0)
        arr = arr - arr.mean() + mean
        stats = _mean_inference(arr)
        assert stats["mean_ci_lower"] < mean < stats["mean_ci_upper"]
        excludes_zero = stats["mean_ci_lower"] > 0 or stats["mean_ci_upper"] < 0
        assert excludes_zero == (stats["p_value"] < 0.05) == (mean != 0)

    def test_confidence_interval_matches_strict_p_value_boundary(self, monkeypatch):
        arr = np.tile([0.0, 2.0], 20)
        statistic = _mean_inference(arr)["t_stat"]
        bootstrap = statistic * np.r_[np.full(95, 0.5), 1.0, np.full(3, 2.0)]
        monkeypatch.setattr(
            "skfolio.alpha._evaluation._wild_bootstrap_t_stats", lambda *_: bootstrap
        )
        stats = _mean_inference(arr)
        assert stats["p_value"] == 0.05
        assert stats["mean_ci_lower"] == pytest.approx(0.0, abs=1e-14)
        bootstrap *= 0.99
        stats = _mean_inference(arr)
        assert stats["p_value"] == 0.04
        assert stats["mean_ci_lower"] > 0

    def test_inference_is_reproducible_without_touching_global_random_state(self):
        arr = self._overlapping_series()
        state = np.random.get_state()
        first = _mean_inference(arr)
        _mean_inference(arr[::2])
        second = _mean_inference(arr.copy())
        assert first == second
        after = np.random.get_state()
        assert state[0] == after[0]
        np.testing.assert_array_equal(state[1], after[1])
        assert state[2:] == after[2:]

    @pytest.mark.parametrize("kind", ["iid", "overlap", "changing_variance"])
    def test_null_rejection_rate(self, monkeypatch, kind):
        monkeypatch.setattr("skfolio.alpha._evaluation._N_BOOTSTRAP_SAMPLES", 999)
        rng = np.random.default_rng(11)
        n_trials, rejections = 300, 0
        for _ in range(n_trials):
            noise = rng.standard_normal(250 + 9)
            arr = (
                3 * np.convolve(noise, np.ones(10) / 10, mode="valid")
                if kind == "overlap"
                else noise[:250]
            )
            if kind == "changing_variance":
                arr *= np.where(np.arange(250) < 200, 0.2, 1.0)
            rejections += _mean_inference(arr)["p_value"] < 0.05
        assert 0.02 <= rejections / n_trials <= 0.09

    @pytest.mark.parametrize("value", [0.0, 0.1, np.nan])
    def test_nan_with_degenerate_series(self, value):
        assert all(np.isnan(x) for x in _mean_inference(np.full(50, value)).values())

    def test_nan_with_fewer_than_20_valid_observations(self):
        arr = self._overlapping_series(n=30)
        arr[:11] = np.nan
        stats = _correlation_stats(arr, ratio_name="icir")
        assert np.isfinite(stats["mean"])
        for key in ("t_stat", "p_value", "mean_ci_lower", "mean_ci_upper"):
            assert np.isnan(stats[key])

    @pytest.mark.parametrize("scale", [1e-100, 1e-6, 1e6])
    def test_inference_is_scale_invariant(self, scale):
        arr = self._overlapping_series()
        reference = _mean_inference(arr)
        scaled = _mean_inference(scale * arr)
        assert scaled["t_stat"] == pytest.approx(reference["t_stat"])
        assert scaled["p_value"] == pytest.approx(reference["p_value"], abs=0.002)
        for key in ("mean_ci_lower", "mean_ci_upper"):
            assert scaled[key] / scale == pytest.approx(reference[key], rel=0.02)

    def test_small_dispersion_around_nonzero_mean_is_not_degenerate(self):
        arr = 5.0 + 1e-6 * self._overlapping_series(n=100, window=5, mean=0, seed=8)
        stats = _mean_inference(arr)
        assert all(np.isfinite(value) for value in stats.values())
        assert stats["mean_ci_lower"] < arr.mean() < stats["mean_ci_upper"]

    @pytest.mark.parametrize(
        "n,horizon,step,finite",
        [
            (60, 6, 1, True),
            (59, 6, 1, False),
            (60, 7, 1, False),
            (95, 10, 1, False),
            (100, 10, 1, True),
            (50, 10, 2, False),
            (51, 10, 2, True),
        ],
    )
    def test_history_boundary_ignores_empty_ends(self, n, horizon, step, finite):
        arr = self._overlapping_series(n=n, window=3)
        reference = _mean_inference(arr, horizon=horizon, evaluation_step=step)
        padded = _mean_inference(
            np.pad(arr, (40, 400), constant_values=np.nan),
            horizon=horizon,
            evaluation_step=step,
        )
        assert np.isfinite(reference["t_stat"]) == finite
        for key in reference:
            assert padded[key] == pytest.approx(reference[key], nan_ok=True)

    def test_explicit_positions_match_nan_gaps(self):
        arr = self._overlapping_series(n=200)
        arr[[35, 36, 81, 135]] = np.nan
        positions = np.flatnonzero(np.isfinite(arr))
        reference = _mean_inference(arr, horizon=10)
        compact = _mean_inference(arr[positions], positions=positions + 100, horizon=10)
        assert compact == reference

    @pytest.mark.parametrize("evaluation_step", [1, 2])
    def test_ic_summary_preserves_missing_dates_and_offset(self, evaluation_step):
        panel = _noisy_forecast_panel()
        target = panel[_IDIO_RETURNS].copy()
        target[:4] = np.nan
        target[60:62] = np.nan
        panel[_IDIO_RETURNS] = target
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            holding_period=2,
            evaluation_step=evaluation_step,
            n_forward_periods=2,
        )
        rows = np.searchsorted(panel.observations, evaluation.observations)
        positions = (rows - rows[0]) // evaluation_step
        assert rows[0] == 2
        summary = evaluation.ic_summary()
        for name in ("spearman_ic", "pearson_ic"):
            values = getattr(evaluation, name)
            grid = np.full(positions[-1] + 1, np.nan)
            grid[positions] = values
            expected = _mean_inference(grid, horizon=2, evaluation_step=evaluation_step)
            for key in expected:
                assert summary.loc[name, key] == pytest.approx(expected[key])
            assert summary.loc[name, "mean"] == pytest.approx(np.nanmean(values))
            assert summary.loc[name, "std"] == pytest.approx(np.nanstd(values, ddof=1))
            assert summary.loc[name, "hit_rate"] == pytest.approx(
                np.mean(values[np.isfinite(values)] > 0)
            )

    def test_manually_constructed_evaluation_defaults_to_consecutive_positions(self):
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            _noisy_forecast_panel(),
            n_forward_periods=2,
        )
        evaluation = replace(evaluation, _evaluation_positions=None)
        expected = _correlation_stats(
            evaluation.spearman_ic, ratio_name="icir", horizon=1
        )
        assert evaluation.ic_summary().loc["spearman_ic"].to_dict() == expected

    @pytest.mark.parametrize("evaluation_step", [1, 2])
    def test_window_inference_uses_each_target_horizon(self, evaluation_step):
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            _noisy_forecast_panel(),
            holding_period=5,
            evaluation_step=evaluation_step,
            n_forward_periods=3,
        )
        holding = evaluation.holding_period_summary()
        decay = evaluation.decay_summary()
        # Fewer than 150 observations remain, so a 15-period window is too long.
        # All decay windows have length 5 and need only 50 observations.
        for prefix in ("spearman", "pearson"):
            for suffix in (
                "ic_t_stat",
                "ic_p_value",
                "mean_ic_ci_lower",
                "mean_ic_ci_upper",
            ):
                column = f"{prefix}_{suffix}"
                assert holding[column].iloc[:2].notna().all()
                assert np.isnan(holding[column].iloc[2])
                assert decay[column].notna().all()

    def test_window_summaries_return_copies_of_stored_frames(self):
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            _noisy_forecast_panel(),
            holding_period=2,
            n_forward_periods=3,
        )
        for stored, method in (
            (evaluation.holding_period_diagnostics, evaluation.holding_period_summary),
            (evaluation.decay, evaluation.decay_summary),
        ):
            summary = method()
            pd.testing.assert_frame_equal(stored, summary)
            summary.iloc[0, 0] = 123.0
            assert stored.iloc[0, 0] != 123.0
            pd.testing.assert_frame_equal(stored, method())

    def test_summary_without_common_sample_keeps_schema(
        self, evaluation_with_diagnostics
    ):
        panel = _perfect_forecast_panel()
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            n_forward_periods=3,
            min_count=panel.n_assets + 1,
        )
        for method in ("holding_period_summary", "decay_summary"):
            summary = getattr(evaluation, method)()
            reference = getattr(evaluation_with_diagnostics, method)()
            pd.testing.assert_index_equal(summary.index, reference.index)
            pd.testing.assert_index_equal(summary.columns, reference.columns)
            assert summary.isna().all().all()

    @pytest.mark.parametrize("evaluation_step", [1, 3])
    @pytest.mark.parametrize(
        "summary_name", ["holding_period_summary", "decay_summary"]
    )
    def test_window_inference_preserves_common_sample_gaps(
        self, evaluation_step, summary_name
    ):
        panel = _noisy_forecast_panel()
        target = panel[_IDIO_RETURNS].copy()
        target[:4] = np.nan
        target[60:62] = np.nan
        target[98] = np.nan
        panel[_IDIO_RETURNS] = target
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            holding_period=2,
            evaluation_step=evaluation_step,
            n_forward_periods=3,
        )
        summary = getattr(evaluation, summary_name)()
        windows = (
            [(2 * period, 1) for period in range(1, 4)]
            if summary_name == "holding_period_summary"
            else [(2, 1 + 2 * delay) for delay in range(3)]
        )
        first_date = np.searchsorted(panel.observations, evaluation.observations[0])
        grid = np.arange(first_date, panel.n_observations, evaluation_step)
        forecasts = panel["signal"][grid]
        targets = [
            _forward_mean_return(target, horizon=horizon, lag=lag)[grid]
            for horizon, lag in windows
        ]
        common = np.logical_and.reduce(
            [
                np.sum(np.isfinite(forecasts) & np.isfinite(values), axis=1) >= 2
                for values in targets
            ]
        )
        assert not common.all()
        for i, ((horizon, _), values) in enumerate(zip(windows, targets, strict=True)):
            for prefix, correlation in (
                ("spearman", cs_spearman_correlation),
                ("pearson", cs_pearson_correlation),
            ):
                # Keep excluded dates as NaN in this reference calculation.
                ic = correlation(forecasts, values, axis=1)
                ic[~common] = np.nan
                expected = _mean_inference(
                    ic, horizon=horizon, evaluation_step=evaluation_step
                )
                row = summary.iloc[i]
                assert row[f"{prefix}_ic_t_stat"] == pytest.approx(
                    self._reference_t_stat(ic), rel=1e-10
                )
                assert row[f"{prefix}_ic_p_value"] == pytest.approx(expected["p_value"])
                for bound in ("lower", "upper"):
                    assert row[f"{prefix}_mean_ic_ci_{bound}"] == pytest.approx(
                        expected[f"mean_ci_{bound}"]
                    )
                assert row[f"{prefix}_mean_ic"] == pytest.approx(np.nanmean(ic))
                assert row[f"{prefix}_icir"] == pytest.approx(
                    np.nanmean(ic) / np.nanstd(ic, ddof=1)
                )

    @pytest.mark.parametrize("evaluation_step", [1, 2, 5])
    def test_factor_summary_preserves_all_dates_and_descriptive_statistics(
        self, evaluation_step
    ):
        panel = _noisy_forecast_panel()
        target = panel[_IDIO_RETURNS].copy()
        target[:4] = np.nan
        panel[_IDIO_RETURNS] = target
        exposures = np.random.default_rng(3).standard_normal((150, 12, 2))
        exposures[59] = np.nan
        panel.add_3d_field(
            _EXPOSURES,
            exposures,
            third_axis_name="factors",
            third_axis_labels=["f0", "f1"],
        )
        evaluation = alpha_forecast_evaluation(
            _fixed_signal_alpha(),
            panel,
            holding_period=20,
            evaluation_step=evaluation_step,
            n_forward_periods=2,
        )
        summary = evaluation.factor_correlation_summary()
        assert evaluation.ic_summary()["t_stat"].isna().all()
        for i in range(2):
            values = evaluation.factor_correlation[:, i]
            mean, std = np.nanmean(values), np.nanstd(values, ddof=1)
            assert summary.iloc[i]["mean"] == pytest.approx(mean)
            assert summary.iloc[i]["std"] == pytest.approx(std)
            assert summary.iloc[i]["ir"] == pytest.approx(mean / std)
            assert summary.iloc[i]["hit_rate"] == pytest.approx(
                np.mean(values[np.isfinite(values)] > 0)
            )
            for key, value in _mean_inference(values).items():
                assert summary.iloc[i][key] == pytest.approx(value)
