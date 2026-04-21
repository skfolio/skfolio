"""Test online_covariance_forecast_evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn import config_context
from sklearn.exceptions import UnsetMetadataPassedError

from skfolio.model_selection import (
    CovarianceForecastEvaluation,
    online_covariance_forecast_evaluation,
)
from skfolio.moments import EWCovariance, EWMu
from skfolio.prior import EmpiricalPrior


@pytest.fixture()
def X_array():
    """Synthetic return array (no index)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((500, 5)) * 0.01


@pytest.fixture()
def X_df(X_array):
    """Synthetic return DataFrame with DatetimeIndex."""
    idx = pd.bdate_range("2020-01-01", periods=X_array.shape[0])
    return pd.DataFrame(X_array, index=idx, columns=list("ABCDE"))


@pytest.fixture()
def evaluation_multi_portfolio():
    """Evaluation with multiple test portfolios."""
    rng = np.random.default_rng(42)
    n_steps = 200
    n_assets = 5
    n_portfolios = 10
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


def _make_inactive_block(X, start: int, stop: int, assets: tuple[int, ...] = (0, 1)):
    """Create a NaN block with a matching active mask."""
    X_masked = X.copy()
    active_mask = np.ones(X.shape, dtype=bool)
    active_mask[start:stop, list(assets)] = False
    if hasattr(X_masked, "iloc"):
        X_masked.iloc[start:stop, list(assets)] = np.nan
    else:
        X_masked[start:stop, list(assets)] = np.nan
    return X_masked, active_mask


class TestOnlineCovarianceForecastEvaluation:
    def test_basic_array(self, X_array):
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
        )
        assert isinstance(ev, CovarianceForecastEvaluation)
        assert ev.horizon == 1
        assert ev.n_portfolios == 1
        assert ev.squared_mahalanobis_distance.shape[0] > 0
        assert ev.mahalanobis_calibration_ratio.shape[0] > 0
        assert ev.diagonal_calibration_ratio.shape[0] > 0
        n_steps = ev.mahalanobis_calibration_ratio.shape[0]
        assert ev.portfolio_standardized_return.shape == (n_steps, 1)
        assert ev.portfolio_variance_qlike_loss.shape == (n_steps, 1)

    def test_basic_dataframe(self, X_df):
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_df,
            warmup_size=100,
            test_size=1,
        )
        assert isinstance(ev, CovarianceForecastEvaluation)
        assert isinstance(ev.observations[0], pd.Timestamp)

    def test_default_inverse_vol_weights(self, X_array):
        """Default (None) uses dynamic inverse-volatility weights."""
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
        )
        assert ev.portfolio_standardized_return is not None
        assert ev.portfolio_variance_qlike_loss is not None
        bs = ev.bias_statistic
        assert bs.shape == (1,)
        assert isinstance(bs[0], np.floating)

    def test_custom_weights_1d(self, X_array):
        w = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
            portfolio_weights=w,
        )
        n_steps = ev.mahalanobis_calibration_ratio.shape[0]
        assert ev.portfolio_standardized_return.shape == (n_steps, 1)
        assert ev.n_portfolios == 1

    def test_custom_weights_2d(self, X_array):
        """2D weight matrix for multiple test portfolios."""
        rng = np.random.default_rng(99)
        W = rng.dirichlet(np.ones(5), size=4)
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
            portfolio_weights=W,
        )
        n_steps = ev.mahalanobis_calibration_ratio.shape[0]
        assert ev.n_portfolios == 4
        assert ev.portfolio_standardized_return.shape == (n_steps, 4)
        assert ev.portfolio_variance_qlike_loss.shape == (n_steps, 4)
        assert ev.bias_statistic.shape == (4,)

    def test_multi_step(self, X_array):
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=5,
        )
        assert ev.horizon == 5
        n_steps = ev.mahalanobis_calibration_ratio.shape[0]
        assert n_steps > 0
        assert n_steps < ev.horizon * 500

    def test_summary_returns_dataframe(self, X_array):
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
        )
        df = ev.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Mahalanobis ratio" in df.index
        assert "Diagonal ratio" in df.index
        assert "Portfolio standardized returns" in df.index
        assert "Portfolio QLIKE" in df.index
        assert "target" in df.columns

    def test_bias_statistic_reasonable(self, X_array):
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
        )
        bs = ev.bias_statistic
        assert all(0.1 < b < 10.0 for b in bs)

    def test_consistent_shapes(self, X_array):
        """Per-step values are positive and have expected shapes."""
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
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

    def test_plots_smoke(self, X_array):
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
        )
        assert isinstance(ev.plot_calibration(), go.Figure)
        assert isinstance(ev.plot_qlike_loss(), go.Figure)
        assert isinstance(ev.plot_exceedance(), go.Figure)

    def test_with_prior_estimator(self, X_array):
        ev = online_covariance_forecast_evaluation(
            EmpiricalPrior(
                mu_estimator=EWMu(half_life=30),
                covariance_estimator=EWCovariance(half_life=30),
            ),
            X_array,
            warmup_size=100,
            test_size=1,
        )
        assert isinstance(ev, CovarianceForecastEvaluation)
        assert ev.n_valid_assets[0] == 5

    def test_too_short_data_raises(self):
        X = np.random.default_rng(0).standard_normal((10, 3))
        with pytest.raises(ValueError, match="The sum of"):
            online_covariance_forecast_evaluation(
                EWCovariance(),
                X,
                warmup_size=20,
            )

    def test_no_partial_fit_raises(self):
        from skfolio.moments import EmpiricalCovariance

        X = np.random.default_rng(0).standard_normal((300, 3))
        with pytest.raises(TypeError, match="partial_fit"):
            online_covariance_forecast_evaluation(
                EmpiricalCovariance(),
                X,
                warmup_size=100,
            )

    def test_calibration_values_reasonable(self, X_array):
        """Check that calibration ratios are in a reasonable range."""
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
        )
        mean_mahal = float(np.mean(ev.mahalanobis_calibration_ratio))
        mean_diag = float(np.mean(ev.diagonal_calibration_ratio))
        assert 0.1 < mean_mahal < 10.0
        assert 0.1 < mean_diag < 10.0

    def test_partial_nan_block_without_complete_rows_is_kept(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((102, 5)) * 0.01
        X[100] = np.array([0.01, np.nan, 0.02, 0.03, 0.04])
        X[101] = np.array([0.05, 0.06, np.nan, 0.07, 0.08])
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X,
            warmup_size=100,
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

    def test_metadata_routing_active_mask(self, X_array):
        """online_covariance_forecast_evaluation routes active_mask metadata."""
        X_masked, active_mask = _make_inactive_block(X_array, start=120, stop=180)

        with config_context(enable_metadata_routing=True):
            ev_routed = online_covariance_forecast_evaluation(
                EWCovariance(half_life=30).set_partial_fit_request(active_mask=True),
                X_masked,
                warmup_size=100,
                test_size=5,
                params={"active_mask": active_mask},
            )
            ev_plain = online_covariance_forecast_evaluation(
                EWCovariance(half_life=30),
                X_masked,
                warmup_size=100,
                test_size=5,
            )

        assert ev_routed.mahalanobis_calibration_ratio.shape == (
            ev_plain.mahalanobis_calibration_ratio.shape
        )
        assert not np.allclose(
            ev_routed.mahalanobis_calibration_ratio,
            ev_plain.mahalanobis_calibration_ratio,
            equal_nan=True,
        )

    def test_metadata_routing_requires_request(self, X_array):
        """online_covariance_forecast_evaluation raises without a request."""
        X_masked, active_mask = _make_inactive_block(X_array, start=120, stop=180)

        with config_context(enable_metadata_routing=True):
            with pytest.raises(
                UnsetMetadataPassedError,
                match="online_covariance_forecast_evaluation",
            ):
                online_covariance_forecast_evaluation(
                    EWCovariance(half_life=30),
                    X_masked,
                    warmup_size=100,
                    test_size=5,
                    params={"active_mask": active_mask},
                )


class TestMultiPortfolioEvaluation:
    def test_multi_portfolio_online(self, X_array):
        """Multi-portfolio via 2D weights."""
        rng = np.random.default_rng(42)
        K = 8
        W = rng.dirichlet(np.ones(5), size=K)
        ev = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
            portfolio_weights=W,
        )
        assert ev.n_portfolios == K
        n_steps = len(ev.observations)
        assert ev.portfolio_standardized_return.shape == (n_steps, K)
        assert ev.portfolio_variance_qlike_loss.shape == (n_steps, K)

        bs = ev.bias_statistic
        assert bs.shape == (K,)
        assert all(b > 0 for b in bs)

    def test_bias_statistic_summary(self, evaluation_multi_portfolio):
        s = evaluation_multi_portfolio.bias_statistic_summary()
        assert isinstance(s, pd.Series)
        assert s.name == "Bias statistic"
        assert "p5" in s.index
        assert "p95" in s.index
        assert "median" in s.index
        assert "n_portfolios" in s.index
        assert s["n_portfolios"] == 10

    def test_summary_multi_portfolio(self, evaluation_multi_portfolio):
        """Summary uses median across portfolios for multi-portfolio."""
        df = evaluation_multi_portfolio.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Portfolio standardized returns" in df.index
        assert "Portfolio QLIKE" in df.index

    def test_summary_multi_portfolio_std_is_median_bias(self):
        """std column equals median of per-portfolio bias statistics."""
        rng = np.random.default_rng(99)
        n_steps = 99
        n_assets = 3
        n_portfolios = 3
        base = np.tile(np.array([-1.0, 0.0, 1.0]), n_steps // 3)
        std_ret = np.column_stack([0.5 * base, 1.0 * base, 2.0 * base])
        ev = CovarianceForecastEvaluation(
            observations=np.arange(n_steps),
            horizon=1,
            squared_mahalanobis_distance=rng.chisquare(df=n_assets, size=n_steps),
            mahalanobis_calibration_ratio=np.ones(n_steps),
            diagonal_calibration_ratio=np.ones(n_steps),
            portfolio_standardized_return=std_ret,
            portfolio_variance_qlike_loss=rng.standard_normal((n_steps, n_portfolios)),
            n_valid_assets=np.full(n_steps, n_assets, dtype=int),
            n_portfolios=n_portfolios,
        )
        df = ev.summary()
        reported_std = df.loc["Portfolio standardized returns", "std"]
        expected = float(np.median(ev.bias_statistic))
        non_expected = float(np.mean(ev.bias_statistic))
        assert abs(expected - non_expected) > 1e-12
        assert abs(reported_std - expected) < 1e-12

    def test_summary_invariant_std_matches_bias_statistic_summary(
        self, evaluation_multi_portfolio
    ):
        """std column equals bias_statistic_summary median for multi-portfolio."""
        df = evaluation_multi_portfolio.summary()
        reported_std = df.loc["Portfolio standardized returns", "std"]
        bias_median = evaluation_multi_portfolio.bias_statistic_summary()["median"]
        assert abs(reported_std - bias_median) < 1e-12

    def test_summary_multi_portfolio_qlike_mean_is_median_of_portfolio_means(self):
        """QLIKE mean column equals median of per-portfolio time-series means."""
        n_steps = 100
        n_assets = 3
        n_portfolios = 3
        rng = np.random.default_rng(42)
        qlike = np.column_stack(
            [
                np.full(n_steps, 3.0),
                np.full(n_steps, 5.0),
                np.full(n_steps, 9.0),
            ]
        )
        ev = CovarianceForecastEvaluation(
            observations=np.arange(n_steps),
            horizon=1,
            squared_mahalanobis_distance=rng.chisquare(df=n_assets, size=n_steps),
            mahalanobis_calibration_ratio=np.ones(n_steps),
            diagonal_calibration_ratio=np.ones(n_steps),
            portfolio_standardized_return=rng.standard_normal((n_steps, n_portfolios)),
            portfolio_variance_qlike_loss=qlike,
            n_valid_assets=np.full(n_steps, n_assets, dtype=int),
            n_portfolios=n_portfolios,
        )
        df = ev.summary()
        reported_mean = df.loc["Portfolio QLIKE", "mean"]
        per_portfolio_means = [float(np.mean(qlike[:, k])) for k in range(n_portfolios)]
        expected = float(np.median(per_portfolio_means))
        non_expected = float(np.mean(per_portfolio_means))
        assert abs(expected - non_expected) > 1e-12
        assert abs(reported_mean - expected) < 1e-12

    def test_plots_multi_portfolio_smoke(self, evaluation_multi_portfolio):
        """Plot methods work with multiple portfolios."""
        assert isinstance(
            evaluation_multi_portfolio.plot_calibration(diagnostics=("bias",)),
            go.Figure,
        )
        assert isinstance(evaluation_multi_portfolio.plot_qlike_loss(), go.Figure)

    def test_plot_calibration_multi_portfolio_band(self, evaluation_multi_portfolio):
        """Bias band traces present when n_portfolios > 1."""
        fig = evaluation_multi_portfolio.plot_calibration(
            diagnostics=("bias",),
        )
        assert isinstance(fig, go.Figure)
        # median line + upper bound (invisible) + lower bound (fill)
        assert len(fig.data) >= 3
        legend_names = [
            trace.name for trace in fig.data if trace.showlegend is not False
        ]
        assert not any("P5-P95" in name for name in legend_names)
        assert not any(trace.name and "P5-P95" in trace.name for trace in fig.data)

    def test_plot_qlike_multi_portfolio_band(self, evaluation_multi_portfolio):
        """QLIKE band traces present when n_portfolios > 1."""
        fig = evaluation_multi_portfolio.plot_qlike_loss()
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3
        legend_names = [
            trace.name for trace in fig.data if trace.showlegend is not False
        ]
        assert not any("P5-P95" in name for name in legend_names)
        assert not any(trace.name and "P5-P95" in trace.name for trace in fig.data)

    def test_single_matches_1d_input(self, X_array):
        """1D weight array produces same result as 2D with one row."""
        w_1d = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
        w_2d = w_1d[np.newaxis, :]

        ev_1d = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
            portfolio_weights=w_1d,
        )
        ev_2d = online_covariance_forecast_evaluation(
            EWCovariance(half_life=30),
            X_array,
            warmup_size=100,
            test_size=1,
            portfolio_weights=w_2d,
        )
        np.testing.assert_allclose(
            ev_1d.portfolio_standardized_return,
            ev_2d.portfolio_standardized_return,
        )
        np.testing.assert_allclose(
            ev_1d.portfolio_variance_qlike_loss,
            ev_2d.portfolio_variance_qlike_loss,
        )
