"""Test covariance_forecast_evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from skfolio.model_selection import (
    CovarianceForecastComparison,
    CovarianceForecastEvaluation,
    covariance_forecast_evaluation,
)
from skfolio.moments import EWCovariance, LedoitWolf


@pytest.fixture()
def X_array():
    """Synthetic return array (no index)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 5)) * 0.01


@pytest.fixture()
def evaluation_full():
    """Evaluation with all fields populated (single portfolio)."""
    rng = np.random.default_rng(42)
    n_steps = 200
    n_assets = 5
    return CovarianceForecastEvaluation(
        observations=pd.date_range("2020-01-01", periods=n_steps, freq="B"),
        horizon=1,
        squared_mahalanobis_distance=rng.chisquare(df=n_assets, size=n_steps),
        mahalanobis_calibration_ratio=rng.chisquare(df=n_assets, size=n_steps)
        / n_assets,
        diagonal_calibration_ratio=rng.chisquare(df=n_assets, size=n_steps) / n_assets,
        portfolio_standardized_return=rng.standard_normal((n_steps, 1)),
        portfolio_variance_qlike_loss=rng.standard_normal((n_steps, 1)) + 5.0,
        n_valid_assets=np.full(n_steps, n_assets, dtype=int),
        n_portfolios=1,
    )


@pytest.fixture()
def evaluation_integer_index():
    """Evaluation with integer index (no DatetimeIndex)."""
    rng = np.random.default_rng(42)
    n_steps = 100
    n_assets = 3
    return CovarianceForecastEvaluation(
        observations=np.arange(n_steps),
        horizon=5,
        squared_mahalanobis_distance=rng.chisquare(df=n_assets, size=n_steps),
        mahalanobis_calibration_ratio=rng.chisquare(df=n_assets, size=n_steps)
        / n_assets,
        diagonal_calibration_ratio=rng.chisquare(df=n_assets, size=n_steps) / n_assets,
        portfolio_standardized_return=rng.standard_normal((n_steps, 1)),
        portfolio_variance_qlike_loss=rng.standard_normal((n_steps, 1)) + 5.0,
        n_valid_assets=np.full(n_steps, n_assets, dtype=int),
        n_portfolios=1,
    )


def _make_eval(seed, n_steps=200, n_assets=5, n_portfolios=1):
    """Helper to create an evaluation with a given seed."""
    rng = np.random.default_rng(seed)
    return CovarianceForecastEvaluation(
        observations=pd.date_range("2020-01-01", periods=n_steps, freq="B"),
        horizon=1,
        squared_mahalanobis_distance=rng.chisquare(df=n_assets, size=n_steps),
        mahalanobis_calibration_ratio=rng.chisquare(df=n_assets, size=n_steps)
        / n_assets,
        diagonal_calibration_ratio=rng.chisquare(df=n_assets, size=n_steps) / n_assets,
        portfolio_standardized_return=rng.standard_normal((n_steps, n_portfolios)),
        portfolio_variance_qlike_loss=rng.standard_normal((n_steps, n_portfolios))
        + 5.0,
        n_valid_assets=np.full(n_steps, n_assets, dtype=int),
        n_portfolios=n_portfolios,
    )


@pytest.fixture()
def comparison():
    """Comparison of two single-portfolio evaluations."""
    return CovarianceForecastComparison(
        [_make_eval(42), _make_eval(123)],
        names=["EWCov(30)", "EWCov(60)"],
    )


@pytest.fixture()
def comparison_three():
    """Comparison of three evaluations."""
    return CovarianceForecastComparison(
        [_make_eval(42), _make_eval(123), _make_eval(456)],
        names=["EWCov(30)", "EWCov(60)", "LedoitWolf"],
    )


@pytest.fixture()
def comparison_multi_portfolio():
    """Comparison with multi-portfolio evaluations."""
    return CovarianceForecastComparison(
        [_make_eval(42, n_portfolios=5), _make_eval(123, n_portfolios=5)],
        names=["EWCov(30)", "EWCov(60)"],
    )


class TestCovarianceForecastEvaluation:
    def test_frozen(self, evaluation_integer_index):
        with pytest.raises(AttributeError):
            evaluation_integer_index.horizon = 10

    def test_fields(self, evaluation_full):
        ev = evaluation_full
        assert ev.horizon == 1
        assert len(ev.observations) == 200
        assert ev.squared_mahalanobis_distance.shape == (200,)
        assert ev.mahalanobis_calibration_ratio.shape == (200,)
        assert ev.diagonal_calibration_ratio.shape == (200,)
        assert ev.portfolio_standardized_return.shape == (200, 1)
        assert ev.portfolio_variance_qlike_loss.shape == (200, 1)
        assert ev.n_portfolios == 1


class TestBiasStatistic:
    def test_bias_statistic_array(self, evaluation_full):
        b = evaluation_full.bias_statistic
        assert isinstance(b, np.ndarray)
        assert b.shape == (1,)
        assert b[0] > 0

    def test_bias_statistic_value(self):
        arr = np.array([[0.5], [-0.5], [1.0], [-1.0], [0.0]])
        ev = CovarianceForecastEvaluation(
            observations=np.arange(5),
            horizon=1,
            squared_mahalanobis_distance=np.ones(5),
            mahalanobis_calibration_ratio=np.ones(5),
            diagonal_calibration_ratio=np.ones(5),
            portfolio_standardized_return=arr,
            portfolio_variance_qlike_loss=np.ones((5, 1)),
            n_valid_assets=np.full(5, 2, dtype=int),
            n_portfolios=1,
        )
        expected = float(np.std(arr[:, 0], ddof=1))
        assert abs(ev.bias_statistic[0] - expected) < 1e-12


class TestSummary:
    def test_summary_rows_and_columns(self, evaluation_full):
        df = evaluation_full.summary()
        assert isinstance(df, pd.DataFrame)
        expected_rows = {
            "Mahalanobis ratio",
            "Diagonal ratio",
            "Portfolio standardized returns",
            "Portfolio QLIKE",
        }
        assert set(df.index) == expected_rows
        assert list(df.columns) == [
            "mean",
            "median",
            "std",
            "p5",
            "p95",
            "mad_from_target",
            "target",
        ]

    def test_target_values(self, evaluation_full):
        df = evaluation_full.summary()
        assert df.loc["Mahalanobis ratio", "target"] == 1.0
        assert df.loc["Diagonal ratio", "target"] == 1.0
        assert df.loc["Portfolio standardized returns", "target"] == "mean=0, std=1"
        assert df.loc["Portfolio QLIKE", "target"] == "lower is better"

    def test_ratio_summary_values(self, evaluation_full):
        df = evaluation_full.summary()
        row = df.loc["Mahalanobis ratio"]
        assert row["mean"] > 0
        assert row["p5"] < row["p95"]
        assert row["mad_from_target"] >= 0

    def test_standardized_return_std_is_bias_statistic(self, evaluation_full):
        df = evaluation_full.summary()
        assert (
            abs(
                df.loc["Portfolio standardized returns", "std"]
                - evaluation_full.bias_statistic[0]
            )
            < 1e-12
        )

    def test_qlike_mad_is_nan(self, evaluation_full):
        df = evaluation_full.summary()
        assert np.isnan(df.loc["Portfolio QLIKE", "mad_from_target"])

    def test_centered_mad_is_mean_abs(self, evaluation_full):
        df = evaluation_full.summary()
        b = evaluation_full.portfolio_standardized_return[:, 0]
        expected = float(np.mean(np.abs(b)))
        assert (
            abs(df.loc["Portfolio standardized returns", "mad_from_target"] - expected)
            < 1e-12
        )


class TestExceedanceSummary:
    def test_default_levels(self, evaluation_full):
        df = evaluation_full.exceedance_summary()
        assert isinstance(df, pd.DataFrame)
        assert df.index.name == "confidence_level"
        assert list(df.index) == [0.95, 0.99]
        assert list(df.columns) == ["observed_rate", "deviation"]

    def test_custom_levels(self, evaluation_full):
        df = evaluation_full.exceedance_summary(confidence_levels=(0.90, 0.95))
        assert list(df.index) == [0.90, 0.95]

    def test_observed_rate_in_range(self, evaluation_full):
        df = evaluation_full.exceedance_summary()
        for rate in df["observed_rate"]:
            assert 0 <= rate <= 1

    def test_deviation_consistent(self, evaluation_full):
        df = evaluation_full.exceedance_summary(confidence_levels=(0.95,))
        rate = df.loc[0.95, "observed_rate"]
        deviation = df.loc[0.95, "deviation"]
        assert abs(deviation - (rate - 0.05)) < 1e-12


class TestPlots:
    def test_plot_calibration_default(self, evaluation_full):
        fig = evaluation_full.plot_calibration()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3

    def test_plot_calibration_single_metric(self, evaluation_full):
        fig = evaluation_full.plot_calibration(diagnostics=("mahalanobis",))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1

    def test_plot_calibration_two_metrics(self, evaluation_full):
        fig = evaluation_full.plot_calibration(
            diagnostics=("mahalanobis", "diagonal"),
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_calibration_invalid_metric(self, evaluation_full):
        with pytest.raises(ValueError, match="Unknown diagnostics"):
            evaluation_full.plot_calibration(diagnostics=("invalid",))

    def test_plot_calibration_integer_index(self, evaluation_integer_index):
        fig = evaluation_integer_index.plot_calibration()
        assert isinstance(fig, go.Figure)

    def test_plot_qlike_loss(self, evaluation_full):
        fig = evaluation_full.plot_qlike_loss()
        assert isinstance(fig, go.Figure)

    def test_plot_exceedance(self, evaluation_full):
        fig = evaluation_full.plot_exceedance()
        assert isinstance(fig, go.Figure)

    def test_plot_exceedance_custom_levels(self, evaluation_integer_index):
        fig = evaluation_integer_index.plot_exceedance(
            confidence_levels=(0.90, 0.95, 0.99),
        )
        assert isinstance(fig, go.Figure)

    def test_plot_exceedance_preserves_nan_rows(self):
        ev = CovarianceForecastEvaluation(
            observations=np.arange(3),
            horizon=1,
            squared_mahalanobis_distance=np.array([10.0, np.nan, 0.0]),
            mahalanobis_calibration_ratio=np.ones(3),
            diagonal_calibration_ratio=np.ones(3),
            portfolio_standardized_return=np.zeros((3, 1)),
            portfolio_variance_qlike_loss=np.zeros((3, 1)),
            n_valid_assets=np.ones(3, dtype=int),
            n_portfolios=1,
        )
        fig = ev.plot_exceedance(confidence_levels=(0.95,), window=1)
        assert isinstance(fig, go.Figure)
        y = np.asarray(fig.data[0].y, dtype=float)
        assert y[0] == 1.0
        assert np.isnan(y[1])
        assert y[2] == 0.0

    def test_custom_window(self, evaluation_full):
        fig = evaluation_full.plot_calibration(window=21)
        assert isinstance(fig, go.Figure)

    def test_custom_title(self, evaluation_full):
        fig = evaluation_full.plot_calibration(title="Custom Title")
        assert fig.layout.title.text == "Custom Title"


class TestCovarianceForecastComparison:
    def test_construction(self, comparison):
        assert len(comparison.evaluations) == 2
        assert comparison._names == ["EWCov(30)", "EWCov(60)"]

    def test_auto_names_from_evaluation(self):
        ev1 = _make_eval(42)
        ev2 = _make_eval(123)
        object.__setattr__(ev1, "name", "MyEstimator1")
        object.__setattr__(ev2, "name", "MyEstimator2")
        comp = CovarianceForecastComparison([ev1, ev2])
        assert comp._names == ["MyEstimator1", "MyEstimator2"]

    def test_auto_names_fallback(self):
        comp = CovarianceForecastComparison(
            [_make_eval(42), _make_eval(123)],
        )
        assert comp._names == ["Estimator 0", "Estimator 1"]

    def test_names_override(self):
        comp = CovarianceForecastComparison(
            [_make_eval(42), _make_eval(123)],
            names=["Short", "Long"],
        )
        assert comp._names == ["Short", "Long"]

    def test_names_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            CovarianceForecastComparison(
                [_make_eval(42), _make_eval(123)],
                names=["Only one"],
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            CovarianceForecastComparison([])

    def test_frozen(self, comparison):
        with pytest.raises(AttributeError):
            comparison.evaluations = []

    def test_summary(self, comparison):
        df = comparison.summary()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.columns, pd.MultiIndex)
        assert "estimator" in df.columns.names
        assert "EWCov(30)" in df.columns.get_level_values("estimator")
        assert "EWCov(60)" in df.columns.get_level_values("estimator")
        expected_rows = {
            "Mahalanobis ratio",
            "Diagonal ratio",
            "Portfolio standardized returns",
            "Portfolio QLIKE",
        }
        assert set(df.index) == expected_rows

    def test_summary_slice_by_estimator(self, comparison):
        df = comparison.summary()
        sub = df["EWCov(30)"]
        assert isinstance(sub, pd.DataFrame)
        assert "mean" in sub.columns
        assert "target" in sub.columns

    def test_bias_statistic_summary(self, comparison):
        df = comparison.bias_statistic_summary()
        assert isinstance(df, pd.DataFrame)
        assert list(df.index) == ["EWCov(30)", "EWCov(60)"]
        assert "median" in df.columns
        assert "n_portfolios" in df.columns

    def test_exceedance_summary(self, comparison):
        df = comparison.exceedance_summary()
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.columns, pd.MultiIndex)
        assert "estimator" in df.columns.names
        assert df.index.name == "confidence_level"

    def test_exceedance_summary_custom_levels(self, comparison):
        df = comparison.exceedance_summary(confidence_levels=(0.90,))
        assert list(df.index) == [0.90]

    def test_plot_calibration_default(self, comparison):
        fig = comparison.plot_calibration()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_plot_calibration_single_metric(self, comparison):
        fig = comparison.plot_calibration(diagnostics=("mahalanobis",))
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_calibration_all_metrics(self, comparison_three):
        fig = comparison_three.plot_calibration()
        assert isinstance(fig, go.Figure)
        # 3 estimators x 3 diagnostics = 9 traces
        assert len(fig.data) == 9

    def test_plot_calibration_invalid_metric(self, comparison):
        with pytest.raises(ValueError, match="Unknown diagnostics"):
            comparison.plot_calibration(diagnostics=("invalid",))

    def test_plot_qlike_loss(self, comparison):
        fig = comparison.plot_qlike_loss()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_exceedance(self, comparison):
        fig = comparison.plot_exceedance()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_plot_exceedance_custom_confidence_level(self, comparison):
        fig = comparison.plot_exceedance(confidence_level=0.99)
        assert isinstance(fig, go.Figure)

    def test_plot_exceedance_preserves_nan_rows(self):
        ev1 = CovarianceForecastEvaluation(
            observations=np.arange(3),
            horizon=1,
            squared_mahalanobis_distance=np.array([10.0, np.nan, 0.0]),
            mahalanobis_calibration_ratio=np.ones(3),
            diagonal_calibration_ratio=np.ones(3),
            portfolio_standardized_return=np.zeros((3, 1)),
            portfolio_variance_qlike_loss=np.zeros((3, 1)),
            n_valid_assets=np.ones(3, dtype=int),
            n_portfolios=1,
            name="A",
        )
        ev2 = CovarianceForecastEvaluation(
            observations=np.arange(3),
            horizon=1,
            squared_mahalanobis_distance=np.array([0.0, 10.0, 0.0]),
            mahalanobis_calibration_ratio=np.ones(3),
            diagonal_calibration_ratio=np.ones(3),
            portfolio_standardized_return=np.zeros((3, 1)),
            portfolio_variance_qlike_loss=np.zeros((3, 1)),
            n_valid_assets=np.ones(3, dtype=int),
            n_portfolios=1,
            name="B",
        )
        comp = CovarianceForecastComparison([ev1, ev2], names=["A", "B"])
        fig = comp.plot_exceedance(confidence_level=0.95, window=1)
        assert isinstance(fig, go.Figure)
        y = np.asarray(fig.data[0].y, dtype=float)
        assert y[0] == 1.0
        assert np.isnan(y[1])
        assert y[2] == 0.0

    def test_plot_calibration_multi_portfolio_band(self, comparison_multi_portfolio):
        """Bias bands appear in comparison with multi-portfolio evaluations."""
        fig = comparison_multi_portfolio.plot_calibration(
            diagnostics=("bias",),
        )
        assert isinstance(fig, go.Figure)
        # 2 estimators x (median + upper + lower) = 6 traces
        assert len(fig.data) == 6
        legend_names = [
            trace.name for trace in fig.data if trace.showlegend is not False
        ]
        assert not any("P5-P95" in name for name in legend_names)
        assert not any(trace.name and "P5-P95" in trace.name for trace in fig.data)

    def test_plot_qlike_multi_portfolio_band(self, comparison_multi_portfolio):
        """QLIKE bands appear in comparison with multi-portfolio evaluations."""
        fig = comparison_multi_portfolio.plot_qlike_loss()
        assert isinstance(fig, go.Figure)
        # 2 estimators x (median + upper + lower) = 6 traces
        assert len(fig.data) == 6
        legend_names = [
            trace.name for trace in fig.data if trace.showlegend is not False
        ]
        assert not any("P5-P95" in name for name in legend_names)
        assert not any(trace.name and "P5-P95" in trace.name for trace in fig.data)

    def test_custom_title(self, comparison):
        fig = comparison.plot_calibration(title="My Comparison")
        assert fig.layout.title.text == "My Comparison"

    def test_custom_window(self, comparison):
        fig = comparison.plot_calibration(window=21)
        assert isinstance(fig, go.Figure)


class TestBatchCovarianceForecastEvaluation:
    def test_basic_array(self, X_array):
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=1,
        )
        assert isinstance(ev, CovarianceForecastEvaluation)
        assert ev.horizon == 1
        assert ev.n_portfolios == 1
        assert ev.squared_mahalanobis_distance.shape[0] > 0

    def test_basic_dataframe(self, X_array):
        idx = pd.bdate_range("2020-01-01", periods=X_array.shape[0])
        X_df = pd.DataFrame(X_array, index=idx, columns=list("ABCDE"))
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_df,
            train_size=100,
            test_size=1,
        )
        assert isinstance(ev, CovarianceForecastEvaluation)
        assert isinstance(ev.observations[0], pd.Timestamp)

    def test_multi_step(self, X_array):
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=5,
        )
        assert ev.horizon == 5
        n_steps = ev.mahalanobis_calibration_ratio.shape[0]
        assert n_steps > 0

    def test_expand_train(self, X_array):
        ev_rolling = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=5,
            expand_train=False,
        )
        ev_expand = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=5,
            expand_train=True,
        )
        assert isinstance(ev_rolling, CovarianceForecastEvaluation)
        assert isinstance(ev_expand, CovarianceForecastEvaluation)
        assert (
            ev_rolling.mahalanobis_calibration_ratio.shape[0]
            == (ev_expand.mahalanobis_calibration_ratio.shape[0])
        )

    def test_custom_weights_1d(self, X_array):
        w = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=1,
            portfolio_weights=w,
        )
        n_steps = ev.mahalanobis_calibration_ratio.shape[0]
        assert ev.portfolio_standardized_return.shape == (n_steps, 1)
        assert ev.n_portfolios == 1

    def test_custom_weights_2d(self, X_array):
        rng = np.random.default_rng(99)
        W = rng.dirichlet(np.ones(5), size=4)
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=1,
            portfolio_weights=W,
        )
        n_steps = ev.mahalanobis_calibration_ratio.shape[0]
        assert ev.n_portfolios == 4
        assert ev.portfolio_standardized_return.shape == (n_steps, 4)
        assert ev.portfolio_variance_qlike_loss.shape == (n_steps, 4)
        assert ev.bias_statistic.shape == (4,)

    def test_summary_returns_dataframe(self, X_array):
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
        )
        df = ev.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Mahalanobis ratio" in df.index

    def test_calibration_values_reasonable(self, X_array):
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=1,
        )
        mean_mahal = float(np.mean(ev.mahalanobis_calibration_ratio))
        mean_diag = float(np.mean(ev.diagonal_calibration_ratio))
        assert 0.1 < mean_mahal < 10.0
        assert 0.1 < mean_diag < 10.0

    def test_purged_size(self, X_array):
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=1,
            purged_size=2,
        )
        assert isinstance(ev, CovarianceForecastEvaluation)
        assert ev.squared_mahalanobis_distance.shape[0] > 0

    def test_consistent_shapes(self, X_array):
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X_array,
            train_size=100,
            test_size=1,
        )
        n_steps = len(ev.observations)
        assert ev.mahalanobis_calibration_ratio.shape == (n_steps,)
        assert ev.diagonal_calibration_ratio.shape == (n_steps,)
        assert ev.squared_mahalanobis_distance.shape == (n_steps,)
        assert ev.portfolio_standardized_return.shape == (n_steps, 1)
        assert ev.portfolio_variance_qlike_loss.shape == (n_steps, 1)
        assert all(ev.mahalanobis_calibration_ratio > 0)
        assert all(ev.diagonal_calibration_ratio > 0)
        assert all(ev.squared_mahalanobis_distance > 0)

    def test_ewcovariance_works(self, X_array):
        ev = covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            train_size=100,
            test_size=1,
        )
        assert isinstance(ev, CovarianceForecastEvaluation)
        assert ev.n_valid_assets[0] == 5

    def test_partial_nan_block_without_complete_rows_is_kept(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((102, 5)) * 0.01
        X[100] = np.array([0.01, np.nan, 0.02, 0.03, 0.04])
        X[101] = np.array([0.05, 0.06, np.nan, 0.07, 0.08])
        ev = covariance_forecast_evaluation(
            LedoitWolf(),
            X,
            train_size=100,
            test_size=2,
        )
        assert isinstance(ev, CovarianceForecastEvaluation)
        assert len(ev.observations) == 1
        assert ev.n_valid_assets[0] == 5
        assert np.isfinite(ev.squared_mahalanobis_distance[0])
        assert np.isfinite(ev.mahalanobis_calibration_ratio[0])
        assert np.isfinite(ev.diagonal_calibration_ratio[0])
        assert np.all(np.isfinite(ev.portfolio_standardized_return[0]))
        assert np.all(np.isfinite(ev.portfolio_variance_qlike_loss[0]))
