"""Alpha forecast evaluation."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sklearn as sk
import sklearn.base as skb
from sklearn.pipeline import Pipeline

from skfolio._constants import (
    _ANNUALIZATION_FACTOR_DEFAULT,
    _BENCHMARK_WEIGHTS,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
    _REGRESSION_WEIGHTS,
)
from skfolio.containers import AssetPanel, AssetPanelView, Field3D
from skfolio.model_selection._validation import _route_params
from skfolio.typing import FloatArray, IntArray, StrArray
from skfolio.utils._factor_tools import _resolve_factor_subset
from skfolio.utils.figure import format_plot_label, format_plot_labels
from skfolio.utils.stats import (
    CSWeighting,
    CorrelationMethod,
    _forward_mean_return,
    cs_pearson_correlation,
    cs_rank,
    cs_spearman_correlation,
    safe_divide,
)
from skfolio.utils.tools import (
    _validate_non_negative_integer,
    _validate_positive_integer,
    _validate_positive_real,
)
from skfolio.utils.validation import validate_asset_panel

__all__ = [
    "AlphaForecastComparison",
    "AlphaForecastEvaluation",
    "CorrelationMethod",
    "alpha_forecast_evaluation",
]

_CALIBRATION_BINS = 10
_LONG_SHORT_GROSS = 2.0


@dataclass(frozen=True, eq=False)
class AlphaForecastEvaluation:
    r"""Out-of-sample alpha forecast evaluation.

    Stores cross-sectional diagnostics produced by
    :func:`~skfolio.alpha.alpha_forecast_evaluation` and provides
    summary statistics and plots.

    The evaluation compares historical alpha forecasts observed at time :math:`t`
    with the forward mean of a target field over
    :math:`[t + \ell, t + \ell + h)`, where :math:`h` is `holding_period`
    and :math:`\ell` is `signal_lag`. The default target is `idio_returns`,
    which evaluates the alpha component not explained by the factor model.

    The core diagnostics are:

    * **IC**: cross-sectional correlation between alpha forecasts and future target
      returns. Spearman IC measures ordering quality. Pearson IC is the
      weighted Pearson correlation under `cs_weighting`.
    * **Simple alpha portfolios**: 200% gross rank-weighted and
      z-score-weighted long-short portfolios built directly from the forecast.
      They measure the realized target return of alpha-only portfolios before
      the alpha is passed to an optimizer.
    * **Quantile spreads**: top-minus-bottom target returns for forecast
      quantiles, equivalent to 200% gross long-short bucket returns. They
      measure whether realized returns are concentrated in the highest-scored
      and lowest-scored assets.
    * **Calibration**: scale multiplier from a weighted regression of realized
      target on forecast with zero intercept. A value near 1 indicates that the
      forecast is already scaled to realized target units.
    * **Factor correlations**: contemporaneous cross-sectional correlation
      between alpha forecasts and factor exposures. They help assess whether
      the alpha forecast is cross-sectionally neutral to existing factors.
    * **Holding-period summary**: the same forecasts evaluated against
      cumulative forward target windows.
    * **Decay**: the same forecasts evaluated against disjoint forward target
      windows.

    Parameters
    ----------
    observations : ndarray of shape (n_steps,)
        Observation labels for the evaluated forecast dates.

    holding_period : int
        Number of observations in the forward target window used for the main
        evaluation.

    n_forward_periods : int
        Number of consecutive forward periods used for holding-period and decay
        diagnostics.

    signal_lag : int
        Number of observations between the forecast date and the first target
        observation. For a forecast at date :math:`t`, the target window is
        :math:`[t + \ell, t + \ell + h)`, where :math:`\ell` is `signal_lag`
        and :math:`h` is `holding_period`.

    evaluation_step : int
        Spacing between evaluated forecast dates.

    annualization_factor : float
        Number of observations per year used to annualize return statistics in
        `portfolio_summary` and `quantile_summary`.

    target : str
        Name of the evaluated target field in the input `AssetPanel`.

    cs_weighting : CSWeighting or str
        Cross-sectional weighting rule used for Pearson IC and the calibration
        scale multiplier.

    spearman_ic : ndarray of shape (n_steps,)
        Spearman rank IC over time.

    pearson_ic : ndarray of shape (n_steps,)
        Pearson IC over time using `cs_weighting`. With
        `CSWeighting.IDENTITY`, this is equal-weighted Pearson IC.

    rank_weighted_portfolio_return : ndarray of shape (n_steps,)
        Forward target return of a centered-rank long-short portfolio with
        200% gross exposure.

    zscore_weighted_portfolio_return : ndarray of shape (n_steps,)
        Forward target return of a centered-forecast long-short portfolio with
        200% gross exposure.

    rank_weighted_turnover : ndarray of shape (n_steps,)
        Turnover of the rank-weighted portfolio. The first value is `NaN`.

    zscore_weighted_turnover : ndarray of shape (n_steps,)
        Turnover of the z-score-weighted portfolio. The first value is `NaN`.

    quantile_spread : ndarray of shape (n_steps, n_quantiles)
        Top-minus-bottom target return for each quantile in `quantiles`,
        equivalent to a 200% gross long-short bucket return.

    quantiles : tuple of float
        Quantiles evaluated in `quantile_spread`.

    n_valid_assets : ndarray of shape (n_steps,)
        Number of assets with finite forecast and target values.

    coverage : ndarray of shape (n_steps,)
        Fraction of eligible assets used at each evaluation date.

    calibration_slope : float
        Scale multiplier from a weighted regression of realized target on
        forecast with zero intercept.

    mean_forecast : float
        Mean evaluated alpha forecast.

    std_forecast : float
        Standard deviation of evaluated alpha forecasts.

    mean_target : float
        Mean evaluated forward target.

    std_target : float
        Standard deviation of evaluated forward targets.

    calibration_curve : DataFrame
        Forecast-bucket calibration table with average forecast and realized
        target values.

    factor_correlation : ndarray of shape (n_observations, n_factors), optional
        Contemporaneous correlation between alpha forecasts and factor exposures.
        Pearson correlations are weighted by the cross-sectional weights resolved
        from `cs_weighting`. `None` when factor correlation diagnostics were
        skipped.

    factor_correlation_method : CorrelationMethod, optional
        Factor correlation method computed from the exposure field. `None`
        when factor correlation diagnostics were skipped.

    factor_names : ndarray of shape (n_factors,)
        Factor names for `factor_correlation`.

    factor_families : ndarray of shape (n_factors,), optional
        Factor family label for each factor. `None` when the factor exposure
        field does not define groups.

    holding_period_diagnostics : DataFrame
        Summary statistics by cumulative holding period.

    decay : DataFrame
        Summary statistics by disjoint forward period.

    name : str, optional
        Display name for the evaluation.
    """

    observations: FloatArray
    holding_period: int
    n_forward_periods: int
    signal_lag: int
    evaluation_step: int
    annualization_factor: float
    target: str
    cs_weighting: CSWeighting | str
    spearman_ic: FloatArray
    pearson_ic: FloatArray
    rank_weighted_portfolio_return: FloatArray
    zscore_weighted_portfolio_return: FloatArray
    rank_weighted_turnover: FloatArray
    zscore_weighted_turnover: FloatArray
    quantile_spread: FloatArray
    quantiles: tuple[float, ...]
    n_valid_assets: IntArray
    coverage: FloatArray
    calibration_slope: float
    mean_forecast: float
    std_forecast: float
    mean_target: float
    std_target: float
    calibration_curve: pd.DataFrame
    factor_correlation: FloatArray | None
    factor_correlation_method: CorrelationMethod | None
    factor_names: StrArray
    factor_families: StrArray | None
    holding_period_diagnostics: pd.DataFrame
    decay: pd.DataFrame
    name: str | None = None

    def ic_summary(self) -> pd.DataFrame:
        r"""Information Coefficient summary.

        Returns one row for Spearman IC and one row for Pearson IC. The `icir`
        column is :math:`\bar{IC} / \sigma_{IC}`. The `t_stat` column is the
        date-level t-statistic of the mean IC.
        """
        return pd.DataFrame(
            {
                "spearman_ic": _correlation_stats(self.spearman_ic, ratio_name="icir"),
                "pearson_ic": _correlation_stats(self.pearson_ic, ratio_name="icir"),
            }
        ).T

    def portfolio_summary(self) -> pd.DataFrame:
        """Annualized 200% gross simple alpha portfolio summary."""
        return pd.DataFrame(
            {
                "rank_weighted_portfolio": _portfolio_stats(
                    self.rank_weighted_portfolio_return,
                    self.rank_weighted_turnover,
                    self.annualization_factor,
                ),
                "zscore_weighted_portfolio": _portfolio_stats(
                    self.zscore_weighted_portfolio_return,
                    self.zscore_weighted_turnover,
                    self.annualization_factor,
                ),
            }
        ).T

    def quantile_summary(self) -> pd.DataFrame:
        """Annualized top-minus-bottom quantile spread summary by tail quantile."""
        records = []
        for i, _ in enumerate(self.quantiles):
            records.append(
                _annualized_return_stats(
                    self.quantile_spread[:, i], self.annualization_factor
                )
            )
        return pd.DataFrame(records, index=pd.Index(self.quantiles, name="quantile"))

    def calibration_summary(self) -> pd.Series:
        """Forecast scale calibration summary."""
        return pd.Series(
            {
                "calibration_slope": self.calibration_slope,
                "mean_forecast": self.mean_forecast,
                "std_forecast": self.std_forecast,
                "mean_target": self.mean_target,
                "std_target": self.std_target,
                "n_bins": len(self.calibration_curve),
            },
            name="Calibration",
        )

    def coverage_summary(self) -> pd.Series:
        """Coverage summary over evaluated forecast dates."""
        return pd.Series(
            {
                "mean_coverage": float(np.nanmean(self.coverage)),
                "min_coverage": float(np.nanmin(self.coverage)),
                "mean_n_valid_assets": float(np.nanmean(self.n_valid_assets)),
                "min_n_valid_assets": int(np.nanmin(self.n_valid_assets)),
            },
            name="Coverage",
        )

    def factor_correlation_summary(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
    ) -> pd.DataFrame:
        r"""Alpha-factor correlation summary.

        Measures contemporaneous cross-sectional correlation between alpha
        forecasts and factor exposures. This helps assess whether the alpha
        forecast is cross-sectionally neutral to existing factors. The `ir`
        column is :math:`\bar{\rho} / \sigma_{\rho}`. The `t_stat` column is
        the date-level t-statistic of the mean correlation. Pearson
        correlations are weighted by the cross-sectional weights resolved from
        `cs_weighting`.

        Parameters
        ----------
        factors : list of str, optional
            Explicit factor names to include. Takes precedence over `families`.

        families : str, list of str, optional
            Factor families to include. `None` includes all factors.

        Returns
        -------
        summary : DataFrame
            Rows are factors and columns are `mean`, `std`, `ir`, `t_stat` and
            `hit_rate`.
        """
        columns = ["mean", "std", "ir", "t_stat", "hit_rate"]
        if self.factor_correlation is None or len(self.factor_names) == 0:
            return pd.DataFrame(columns=columns)

        factor_indices, factor_names = _resolve_factor_subset(
            factor_names=self.factor_names,
            factor_families=self.factor_families,
            factor_names_to_keep=factors,
            family_names_to_keep=families,
        )
        corr = self.factor_correlation[:, factor_indices]
        records = [
            _correlation_stats(corr[:, i], ratio_name="ir")
            for i in range(corr.shape[1])
        ]
        return pd.DataFrame(records, index=factor_names, columns=columns)

    def holding_period_summary(self) -> pd.DataFrame:
        """Alpha diagnostics by cumulative holding period."""
        return self.holding_period_diagnostics.copy()

    def decay_summary(self) -> pd.DataFrame:
        """Alpha decay summary by disjoint forward period."""
        return self.decay.copy()

    def plot_cumulative_ic(
        self, *, include_pearson: bool = True, title: str | None = None
    ) -> go.Figure:
        """Plot cumulative IC over time."""
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.observations,
                y=np.nancumsum(self.spearman_ic),
                mode="lines",
                name="Spearman IC",
            )
        )
        if include_pearson:
            fig.add_trace(
                go.Scatter(
                    x=self.observations,
                    y=np.nancumsum(self.pearson_ic),
                    mode="lines",
                    name="Pearson IC",
                )
            )
        fig.update_layout(
            title=title or "Cumulative Alpha IC", yaxis_title="Cumulative IC"
        )
        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="gray")
        return fig

    def plot_rolling_ic(self, window: int = 50, title: str | None = None) -> go.Figure:
        """Plot rolling mean IC over time."""
        _validate_positive_integer(window, "window")
        series = {
            "Spearman IC": _rolling(self.spearman_ic, self.observations, window),
            "Pearson IC": _rolling(self.pearson_ic, self.observations, window),
        }
        return _plot_lines(
            series,
            title=title or f"Rolling Alpha IC ({window} observations)",
            yaxis_title="IC",
            ref_value=0.0,
        )

    def plot_cumulative_returns(self, title: str | None = None) -> go.Figure:
        """Plot cumulative returns of 200% gross simple alpha portfolios."""
        series = {
            "Rank-Weighted Portfolio": pd.Series(
                np.nancumsum(self.rank_weighted_portfolio_return),
                index=self.observations,
            ),
            "Z-Score-Weighted Portfolio": pd.Series(
                np.nancumsum(self.zscore_weighted_portfolio_return),
                index=self.observations,
            ),
        }
        fig = _plot_lines(
            series,
            title=title or "Cumulative 200% Gross Alpha Portfolio Returns",
            yaxis_title="Cumulative Return",
            ref_value=0.0,
        )
        fig.update_yaxes(tickformat=".2%")
        return fig

    def plot_quantile_returns(self, title: str | None = None) -> go.Figure:
        """Plot cumulative top-minus-bottom quantile spreads."""
        series = {
            f"Quantile {q:g}": pd.Series(
                np.nancumsum(self.quantile_spread[:, i]), index=self.observations
            )
            for i, q in enumerate(self.quantiles)
        }
        fig = _plot_lines(
            series,
            title=title or "Cumulative Alpha Quantile Spreads",
            yaxis_title="Cumulative Spread",
            ref_value=0.0,
        )
        fig.update_yaxes(tickformat=".2%")
        return fig

    def plot_calibration(self, title: str | None = None) -> go.Figure:
        """Plot realized target by forecast bucket."""
        df = self.calibration_curve
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["mean_forecast"],
                y=df["mean_target"],
                mode="markers+lines",
                name="Observed",
            )
        )
        x = np.asarray(df["mean_forecast"], dtype=float)
        if np.isfinite(x).any() and np.isfinite(self.calibration_slope):
            x0 = float(np.nanmin(x))
            x1 = float(np.nanmax(x))
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[self.calibration_slope * x0, self.calibration_slope * x1],
                    mode="lines",
                    name="Pooled Slope",
                    line=dict(dash="dash"),
                )
            )
        fig.update_layout(
            title=title or "Alpha Forecast Calibration",
            xaxis_title="Mean Forecast",
            yaxis_title="Mean Realized Target",
        )
        return fig

    def plot_ic_by_holding_period(self, title: str | None = None) -> go.Figure:
        """Plot mean IC by cumulative holding period."""
        if self.holding_period_diagnostics.empty:
            raise ValueError("No holding-period diagnostics are available.")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.holding_period_diagnostics.index,
                y=self.holding_period_diagnostics["spearman_mean_ic"],
                mode="markers+lines",
                name="Spearman IC",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.holding_period_diagnostics.index,
                y=self.holding_period_diagnostics["pearson_mean_ic"],
                mode="markers+lines",
                name="Pearson IC",
            )
        )
        fig.update_layout(
            title=title or "Alpha IC by Holding Period",
            xaxis_title="Holding Period",
            yaxis_title="Mean IC",
        )
        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="gray")
        return fig

    def plot_portfolio_by_holding_period(self, title: str | None = None) -> go.Figure:
        """Plot simple portfolio IR by cumulative holding period."""
        if self.holding_period_diagnostics.empty:
            raise ValueError("No holding-period diagnostics are available.")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.holding_period_diagnostics.index,
                y=self.holding_period_diagnostics["rank_weighted_portfolio_ir"],
                mode="markers+lines",
                name="Rank-Weighted Portfolio",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.holding_period_diagnostics.index,
                y=self.holding_period_diagnostics["zscore_weighted_portfolio_ir"],
                mode="markers+lines",
                name="Z-Score-Weighted Portfolio",
            )
        )
        fig.update_layout(
            title=title or "Alpha Portfolio by Holding Period",
            xaxis_title="Holding Period",
            yaxis_title="IR",
        )
        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="gray")
        return fig

    def plot_ic_decay(self, title: str | None = None) -> go.Figure:
        """Plot mean IC by disjoint forward period."""
        if self.decay.empty:
            raise ValueError("No decay diagnostics are available.")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.decay.index,
                y=self.decay["spearman_mean_ic"],
                mode="markers+lines",
                name="Spearman IC",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.decay.index,
                y=self.decay["pearson_mean_ic"],
                mode="markers+lines",
                name="Pearson IC",
            )
        )
        fig.update_layout(
            title=title or "Alpha IC Decay",
            xaxis_title="Period",
            yaxis_title="Mean IC",
        )
        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="gray")
        return fig

    def plot_portfolio_decay(self, title: str | None = None) -> go.Figure:
        """Plot simple portfolio IR by disjoint forward period."""
        if self.decay.empty:
            raise ValueError("No decay diagnostics are available.")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.decay.index,
                y=self.decay["rank_weighted_portfolio_ir"],
                mode="markers+lines",
                name="Rank-Weighted Portfolio",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.decay.index,
                y=self.decay["zscore_weighted_portfolio_ir"],
                mode="markers+lines",
                name="Z-Score-Weighted Portfolio",
            )
        )
        fig.update_layout(
            title=title or "Alpha Portfolio Decay",
            xaxis_title="Period",
            yaxis_title="IR",
        )
        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="gray")
        return fig

    def plot_factor_correlation(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        top_n: int | None = 20,
        title: str | None = None,
    ) -> go.Figure:
        """Plot mean alpha-factor correlations."""
        if top_n is not None:
            _validate_positive_integer(top_n, "top_n")
        summary = self.factor_correlation_summary(factors=factors, families=families)
        if summary.empty:
            raise ValueError("No factor correlation diagnostics are available.")
        order = summary["mean"].abs().sort_values(ascending=False).index
        summary = summary.loc[order]
        if top_n is not None:
            summary = summary.iloc[:top_n]
        method = self.factor_correlation_method
        method_label = (
            format_plot_label(method.value) if method is not None else "Correlation"
        )

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=format_plot_labels(summary.index),
                y=summary["mean"],
                name="Mean Correlation",
            )
        )
        fig.update_layout(
            title=title or f"Alpha Factor Correlation ({method_label})",
            xaxis_title="Factor",
            yaxis_title="Mean Correlation",
        )
        fig.update_yaxes(range=[-1.0, 1.0])
        fig.add_hline(y=0.0, line_width=1, line_dash="dash", line_color="gray")
        return fig


@dataclass(frozen=True, eq=False)
class AlphaForecastComparison:
    """Side-by-side comparison of alpha forecast evaluations."""

    evaluations: list[AlphaForecastEvaluation]
    names: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise ValueError("evaluations must contain at least one entry.")
        if self.names is not None:
            if len(self.names) != len(self.evaluations):
                raise ValueError(
                    f"names has length {len(self.names)} but evaluations has "
                    f"length {len(self.evaluations)}."
                )
            names = list(self.names)
        else:
            names = [
                ev.name if ev.name is not None else f"Estimator {i}"
                for i, ev in enumerate(self.evaluations)
            ]
        object.__setattr__(self, "_names", names)

    def _named_evaluations(self):
        """Iterate over (name, evaluation) pairs."""
        return zip(self._names, self.evaluations, strict=True)

    def ic_summary(self) -> pd.DataFrame:
        """IC summary for all evaluations."""
        return pd.concat(
            {name: ev.ic_summary() for name, ev in self._named_evaluations()},
            axis=1,
            names=["estimator"],
        )

    def portfolio_summary(self) -> pd.DataFrame:
        """Simple portfolio summary for all evaluations."""
        return pd.concat(
            {name: ev.portfolio_summary() for name, ev in self._named_evaluations()},
            axis=1,
            names=["estimator"],
        )

    def plot_cumulative_ic(self, title: str | None = None) -> go.Figure:
        """Plot cumulative Spearman IC for all evaluations."""
        series = {
            name: pd.Series(np.nancumsum(ev.spearman_ic), index=ev.observations)
            for name, ev in self._named_evaluations()
        }
        return _plot_lines(
            series,
            title=title or "Cumulative Alpha IC (Spearman)",
            yaxis_title="Cumulative IC",
            ref_value=0.0,
        )

    def plot_cumulative_returns(self, title: str | None = None) -> go.Figure:
        """Plot cumulative 200% gross rank-weighted portfolio returns."""
        series = {
            name: pd.Series(
                np.nancumsum(ev.rank_weighted_portfolio_return),
                index=ev.observations,
            )
            for name, ev in self._named_evaluations()
        }
        fig = _plot_lines(
            series,
            title=title or "Cumulative 200% Gross Alpha Portfolio Return Comparison",
            yaxis_title="Cumulative Return",
            ref_value=0.0,
        )
        fig.update_yaxes(tickformat=".2%")
        return fig


def alpha_forecast_evaluation(
    estimator: skb.BaseEstimator | Pipeline,
    X: AssetPanel | AssetPanelView,
    *,
    target: str = _IDIO_RETURNS,
    holding_period: int = 1,
    signal_lag: int = 1,
    evaluation_step: int | None = None,
    n_forward_periods: int = 10,
    cs_weighting: CSWeighting | str = CSWeighting.IDENTITY,
    factor_exposures: str | None = _EXPOSURES,
    factor_correlation_method: CorrelationMethod | None = CorrelationMethod.PEARSON,
    quantiles: tuple[float, ...] = (0.1,),
    annualization_factor: float = _ANNUALIZATION_FACTOR_DEFAULT,
    min_count: int = 3,
    params: dict | None = None,
    name: str | None = None,
) -> AlphaForecastEvaluation:
    r"""Evaluate alpha forecast quality.

    The function fits `estimator` with `fit_transform`, obtains historical
    alpha forecasts, and compares them with a forward mean target built from an
    `AssetPanel` field. The default target is `idio_returns`, which evaluates
    the idiosyncratic component forecast by the alpha estimators used by
    :class:`~skfolio.prior.CharacteristicsFactorModel`.

    The diagnostics evaluate alpha forecasts before the alpha is passed to an
    optimizer. IC measures cross-sectional ordering and Pearson correlation.
    Simple rank-weighted and z-score-weighted portfolios measure the realized
    target return of 200% gross alpha-only long-short portfolios. The
    calibration slope estimates the scale multiplier needed to map forecast
    values to realized target units.
    Holding-period diagnostics evaluate the same forecasts against cumulative
    target windows. Decay diagnostics evaluate the same forecasts against
    disjoint future target windows.
    Diagnostics are computed on the final alpha forecast returned by the
    estimator. For rank-transformed forecasts, `pearson_ic` and
    `zscore_weighted_portfolio` evaluate the transformed rank scores, not raw
    descriptor magnitudes. If `factor_exposures` is available and
    `factor_correlation_method` is not `None`, the evaluation also measures
    contemporaneous correlation between the alpha forecast and factor exposures.
    Holding-period and decay diagnostics use the same evaluation dates as the
    main evaluation.

    For example, with `holding_period=5`, `signal_lag=1` and
    `n_forward_periods=3`, `decay_summary` computes IC on disjoint windows:
    :math:`corr(\alpha_t, \bar{y}_{t+1:t+5})`,
    :math:`corr(\alpha_t, \bar{y}_{t+6:t+10})` and
    :math:`corr(\alpha_t, \bar{y}_{t+11:t+15})`.
    `holding_period_summary` computes IC on cumulative windows:
    :math:`corr(\alpha_t, \bar{y}_{t+1:t+5})`,
    :math:`corr(\alpha_t, \bar{y}_{t+1:t+10})` and
    :math:`corr(\alpha_t, \bar{y}_{t+1:t+15})`.
    With `holding_period=5` and `n_forward_periods=79`, the last cumulative
    window is :math:`corr(\alpha_t, \bar{y}_{t+1:t+395})`.

    Parameters
    ----------
    estimator : BaseEstimator or Pipeline
        Alpha estimator exposing `fit_transform` and returning historical alpha
        forecasts with shape `(n_observations, n_assets)`.

    X : AssetPanel or AssetPanelView
        Point-in-time asset panel containing `target` and all fields required by
        `estimator`.

    target : str, default="idio_returns"
        Name of the 2D target field in `X`.

    holding_period : int, default=1
        Number of observations in the forward target window used for the main
        evaluation. For a forecast at date :math:`t`, the target is the mean
        value over :math:`[t + \ell, t + \ell + h)`, where :math:`\ell` is
        `signal_lag` and :math:`h` is `holding_period`.

    signal_lag : int, default=1
        Number of observations between the forecast date and the first target
        observation. `signal_lag=1` evaluates next-period targets and avoids
        same-period look-ahead when forecasts are observed after the current
        target is known. `signal_lag=0` evaluates same-period targets.

    evaluation_step : int, optional
        Spacing between evaluated forecast dates. The default `None` uses
        `holding_period`, which produces mostly non-overlapping target windows
        for the main evaluation. `evaluation_step=1` evaluates every valid
        forecast date, which is common for signal research and creates
        overlapping forward targets when `holding_period > 1`. Values greater
        than `holding_period` produce a sparse evaluation. When
        `evaluation_step < holding_period`, summary means remain descriptive
        diagnostics, but IC t-statistics and IR should be interpreted with the
        serial dependence from overlapping targets in mind.

    n_forward_periods : int, default=10
        Number of consecutive forward periods used for holding-period and decay
        diagnostics. `holding_period_summary` evaluates cumulative windows from
        :math:`1 \times h` to :math:`n \times h`. `decay_summary` evaluates
        :math:`n` disjoint forward windows of length :math:`h`, where
        :math:`h` is `holding_period` and :math:`n` is `n_forward_periods`.

    cs_weighting : CSWeighting or str, default=CSWeighting.IDENTITY
        Cross-sectional weighting for Pearson IC and the calibration scale
        multiplier. A string is interpreted as a 2D field name in `X`.
        Descriptive forecast, target and calibration-curve statistics are
        unweighted.

    factor_exposures : str, optional, default="exposures"
        Name of a 3D field in `AssetPanel` `X` containing factor exposures
        used to compute alpha-factor correlation diagnostics. If the default
        field is not present, factor correlation diagnostics are skipped.
        Passing `None` skips them explicitly.

    factor_correlation_method : CorrelationMethod, optional, default=CorrelationMethod.PEARSON
        Factor correlation method to compute. `PEARSON` measures linear tilt
        of forecast values to factor exposures and is weighted by
        `cs_weighting`. `SPEARMAN` measures monotonic alignment of forecast
        ordering with exposure ordering and is more expensive for large
        exposure tensors. Passing `None` skips factor correlation diagnostics.

    quantiles : tuple of float, default=(0.1,)
        Forecast quantiles for top-minus-bottom spread diagnostics. Each value
        must be in `(0, 0.5]`.

    annualization_factor : float, default=252.0
        Number of observations per year used to annualize return statistics in
        `portfolio_summary` and `quantile_summary`.

    min_count : int, default=3
        Minimum number of valid assets required for each cross-sectional
        diagnostic.

    params : dict, optional
        Parameters routed to `estimator.fit_transform`.

    name : str, optional
        Display name for the evaluation. Defaults to `str(estimator)`.

    Returns
    -------
    evaluation : AlphaForecastEvaluation
        Frozen dataclass with diagnostic series, summary statistics and plots.
    """
    _validate_positive_integer(min_count, "min_count")
    _validate_positive_integer(holding_period, "holding_period")
    _validate_positive_integer(n_forward_periods, "n_forward_periods")
    _validate_non_negative_integer(signal_lag, "signal_lag")
    _validate_positive_real(annualization_factor, "annualization_factor")

    estimator = sk.clone(estimator)
    holding_period = int(holding_period)
    n_forward_periods = int(n_forward_periods)
    signal_lag = int(signal_lag)
    if evaluation_step is None:
        evaluation_step = holding_period
    else:
        _validate_positive_integer(evaluation_step, "evaluation_step")
        evaluation_step = int(evaluation_step)
    annualization_factor = float(annualization_factor)
    quantiles = _validate_quantiles(quantiles)
    if factor_correlation_method is not None and not isinstance(
        factor_correlation_method, CorrelationMethod
    ):
        raise TypeError(
            "factor_correlation_method must be a `CorrelationMethod` or None."
        )

    required_fields = [target]
    weighting_field = _field_required_by_cs_weighting(cs_weighting)
    if weighting_field is not None and weighting_field not in required_fields:
        required_fields.append(weighting_field)
    if cs_weighting is CSWeighting.INVERSE_IDIO_VARIANCE and target != _IDIO_VARIANCES:
        required_fields.append(_IDIO_VARIANCES)
    factor_exposures_field = (
        None
        if factor_correlation_method is None
        else _resolve_factor_exposures_field(X, factor_exposures)
    )
    if factor_exposures_field is not None:
        required_fields.append(factor_exposures)

    validate_asset_panel(
        estimator,
        X,
        required_fields=required_fields,
        finite_or_nan=required_fields,
        strictly_positive_or_nan=(
            [_IDIO_VARIANCES]
            if cs_weighting is CSWeighting.INVERSE_IDIO_VARIANCE
            else []
        ),
        reset=True,
    )

    if not hasattr(estimator, "fit_transform"):
        raise TypeError(
            "`alpha_forecast_evaluation` requires an estimator exposing "
            "`fit_transform`."
        )

    routed_params = _route_params(
        estimator,
        params,
        owner="alpha_forecast_evaluation",
        callee="fit_transform",
    )
    alpha = estimator.fit_transform(X, **routed_params.estimator_params)
    alpha = np.asarray(alpha, dtype=float)
    expected_shape = (X.n_observations, X.n_assets)
    if alpha.shape != expected_shape:
        raise ValueError(
            "`estimator.fit_transform` must return an array with shape "
            f"{expected_shape}, got {alpha.shape}."
        )

    target_forward = _forward_mean_return(
        X[target], horizon=holding_period, lag=signal_lag
    )
    cs_weights = _resolve_cs_weights(X, cs_weighting)
    eval_idx = _evaluation_indices(
        alpha, target_forward, X.estimation_mask, evaluation_step
    )

    if eval_idx.size == 0:
        raise ValueError("No valid evaluation date is available.")

    diagnostics = _compute_diagnostics(
        alpha=alpha[eval_idx],
        target=target_forward[eval_idx],
        eligible_mask=X.estimation_mask[eval_idx],
        cs_weights=None if cs_weights is None else cs_weights[eval_idx],
        quantiles=quantiles,
        min_count=min_count,
    )
    factor_diagnostics = _compute_factor_correlation_diagnostics(
        alpha=alpha,
        exposures_field=factor_exposures_field,
        estimation_mask=X.estimation_mask,
        cs_weights=cs_weights,
        method=factor_correlation_method,
        min_count=min_count,
    )
    holding_period_diagnostics = _compute_holding_period_diagnostics(
        alpha=alpha,
        X=X,
        target=target,
        eval_idx=eval_idx,
        holding_period=holding_period,
        signal_lag=signal_lag,
        n_forward_periods=n_forward_periods,
        cs_weights=cs_weights,
        quantiles=quantiles,
        min_count=min_count,
    )
    decay = _compute_decay(
        alpha=alpha,
        X=X,
        target=target,
        eval_idx=eval_idx,
        holding_period=holding_period,
        signal_lag=signal_lag,
        n_forward_periods=n_forward_periods,
        cs_weights=cs_weights,
        quantiles=quantiles,
        min_count=min_count,
    )

    return AlphaForecastEvaluation(
        observations=np.asarray(X.observations)[eval_idx],
        holding_period=holding_period,
        n_forward_periods=n_forward_periods,
        signal_lag=signal_lag,
        evaluation_step=evaluation_step,
        annualization_factor=annualization_factor,
        target=target,
        cs_weighting=cs_weighting,
        quantiles=quantiles,
        holding_period_diagnostics=holding_period_diagnostics,
        decay=decay,
        name=name or str(estimator),
        **diagnostics,
        **factor_diagnostics,
    )


def _validate_quantiles(quantiles: tuple[float, ...]) -> tuple[float, ...]:
    """Validate quantile levels for symmetric spread diagnostics."""
    if len(quantiles) == 0:
        raise ValueError("quantiles must contain at least one value.")
    out = tuple(float(q) for q in quantiles)
    if any(not np.isfinite(q) or q <= 0.0 or q > 0.5 for q in out):
        raise ValueError("quantiles must contain finite values in (0, 0.5].")
    return out


def _field_required_by_cs_weighting(cs_weighting: CSWeighting | str) -> str | None:
    """Return the AssetPanel field required by a weighting rule."""
    if not isinstance(cs_weighting, CSWeighting):
        if isinstance(cs_weighting, str):
            return cs_weighting
        raise TypeError("cs_weighting must be a `CSWeighting` or a field name.")
    if cs_weighting is CSWeighting.BENCHMARK:
        return _BENCHMARK_WEIGHTS
    if cs_weighting is CSWeighting.REGRESSION:
        return _REGRESSION_WEIGHTS
    if cs_weighting is CSWeighting.INVERSE_IDIO_VARIANCE:
        return _IDIO_VARIANCES
    return None


def _resolve_cs_weights(
    X: AssetPanel | AssetPanelView, cs_weighting: CSWeighting | str
) -> FloatArray | None:
    """Return cross-sectional weights aligned with an AssetPanel."""
    field = _field_required_by_cs_weighting(cs_weighting)
    if field is None:
        return None
    if cs_weighting is CSWeighting.INVERSE_IDIO_VARIANCE:
        weights = safe_divide(1.0, X[field], fill_value=np.nan)
    else:
        weights = np.asarray(X[field], dtype=float)
    if weights.shape != (X.n_observations, X.n_assets):
        raise ValueError(
            f"`cs_weighting` field must have shape {(X.n_observations, X.n_assets)}, "
            f"got {weights.shape}."
        )
    bad = np.isfinite(weights) & (weights < 0.0)
    if np.any(bad):
        raise ValueError("Cross-sectional weights must be non-negative.")
    return weights


def _resolve_factor_exposures_field(
    X: AssetPanel | AssetPanelView, factor_exposures: str | None
) -> Field3D | None:
    """Return the optional factor exposure field used for correlation diagnostics."""
    if factor_exposures is None:
        return None
    if not isinstance(factor_exposures, str):
        raise TypeError("factor_exposures must be a string or None.")
    if factor_exposures not in X.keys():
        if factor_exposures == _EXPOSURES:
            return None
        raise ValueError(
            f"`factor_exposures` '{factor_exposures}' is not in the "
            f"AssetPanel. Available fields: {sorted(X.keys())}."
        )
    field = X.get_field(factor_exposures)
    if not isinstance(field, Field3D):
        raise TypeError(f"Field '{factor_exposures}' is not a Field3D.")
    return field


def _compute_factor_correlation_diagnostics(
    *,
    alpha: FloatArray,
    exposures_field: Field3D | None,
    estimation_mask: FloatArray,
    cs_weights: FloatArray | None,
    method: CorrelationMethod | None,
    min_count: int,
) -> dict[str, object]:
    """Compute alpha-factor correlation diagnostics."""
    if exposures_field is None or method is None:
        return _empty_factor_correlation_diagnostics()

    exposures = np.asarray(exposures_field.values, dtype=float)
    if exposures.shape[:2] != alpha.shape:
        raise ValueError(
            "`factor_exposures` must have first two axes matching alpha "
            f"shape {alpha.shape}, got {exposures.shape[:2]}."
        )

    factor_names = np.asarray(exposures_field.third_axis_labels, dtype=str)
    factor_families = (
        None
        if exposures_field.third_axis_groups is None
        else np.asarray(exposures_field.third_axis_groups, dtype=str)
    )
    if exposures.shape[2] == 0:
        return {
            "factor_correlation": np.empty((alpha.shape[0], 0), dtype=float),
            "factor_correlation_method": method,
            "factor_names": factor_names,
            "factor_families": factor_families,
        }

    alpha_eval = np.where(estimation_mask & np.isfinite(alpha), alpha, np.nan)
    if cs_weights is None:
        weights = None
    else:
        weights = np.where(
            estimation_mask & np.isfinite(cs_weights) & (cs_weights > 0.0),
            cs_weights,
            0.0,
        )

    if method is CorrelationMethod.SPEARMAN:
        factor_correlation = cs_spearman_correlation(
            alpha_eval[:, :, np.newaxis],
            exposures,
            axis=1,
            min_count=min_count,
        )
    else:
        factor_correlation = cs_pearson_correlation(
            alpha_eval,
            exposures,
            weights=weights,
            axis=1,
            min_count=min_count,
        )
    return {
        "factor_correlation": factor_correlation,
        "factor_correlation_method": method,
        "factor_names": factor_names,
        "factor_families": factor_families,
    }


def _empty_factor_correlation_diagnostics() -> dict[str, object]:
    """Return empty alpha-factor correlation diagnostics."""
    return {
        "factor_correlation": None,
        "factor_correlation_method": None,
        "factor_names": np.array([], dtype=str),
        "factor_families": None,
    }


def _evaluation_indices(
    alpha: FloatArray,
    target: FloatArray,
    eligible_mask: FloatArray,
    evaluation_step: int,
) -> IntArray:
    """Return forecast dates with at least one valid evaluation asset."""
    valid = np.isfinite(alpha) & np.isfinite(target) & eligible_mask
    has_valid = np.any(valid, axis=1)
    valid_indices = np.flatnonzero(has_valid)
    if valid_indices.size == 0:
        return np.array([], dtype=int)
    start = int(valid_indices[0])
    idx = np.arange(start, alpha.shape[0], evaluation_step)
    return idx[has_valid[idx]]


def _compute_diagnostics(
    *,
    alpha: FloatArray,
    target: FloatArray,
    eligible_mask: FloatArray,
    cs_weights: FloatArray | None,
    quantiles: tuple[float, ...],
    min_count: int,
) -> dict[str, object]:
    """Compute IC, portfolio, spread, coverage and calibration diagnostics."""
    valid = np.isfinite(alpha) & np.isfinite(target) & eligible_mask
    eligible_count = np.sum(eligible_mask, axis=1)
    n_valid_assets = np.sum(valid, axis=1)
    coverage = safe_divide(n_valid_assets, eligible_count, fill_value=np.nan)

    alpha_eval = np.where(valid, alpha, np.nan)
    target_eval = np.where(valid, target, np.nan)
    spearman_ic = cs_spearman_correlation(
        alpha_eval, target_eval, axis=1, min_count=min_count
    )

    if cs_weights is None:
        weighted_alpha = alpha_eval
        weighted_target = target_eval
        weights = None
    else:
        weights = np.where(
            np.isfinite(cs_weights) & (cs_weights > 0.0), cs_weights, 0.0
        )
        valid_weighted = valid & (weights > 0.0)
        weighted_alpha = np.where(valid_weighted, alpha, np.nan)
        weighted_target = np.where(valid_weighted, target, np.nan)

    pearson_ic = cs_pearson_correlation(
        weighted_alpha,
        weighted_target,
        weights=weights,
        axis=1,
        min_count=min_count,
    )
    rank_weighted_weights = _rank_weighted_portfolio_weights(alpha_eval)
    zscore_weighted_weights = _zscore_weighted_portfolio_weights(alpha_eval)
    rank_weighted_portfolio_return = np.sum(
        rank_weighted_weights * np.where(np.isfinite(target_eval), target_eval, 0.0),
        axis=1,
    )
    zscore_weighted_portfolio_return = np.sum(
        zscore_weighted_weights * np.where(np.isfinite(target_eval), target_eval, 0.0),
        axis=1,
    )
    rank_weighted_turnover = _turnover(rank_weighted_weights)
    zscore_weighted_turnover = _turnover(zscore_weighted_weights)
    quantile_spread = _quantile_spread(alpha_eval, target_eval, quantiles)
    calibration_slope = _calibration_slope(
        weighted_alpha, weighted_target, weights=weights
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_forecast = float(np.nanmean(alpha_eval))
        std_forecast = float(np.nanstd(alpha_eval, ddof=1))
        mean_target = float(np.nanmean(target_eval))
        std_target = float(np.nanstd(target_eval, ddof=1))
    calibration_curve = _calibration_curve(alpha_eval, target_eval)

    invalid_rows = n_valid_assets < min_count
    for arr in (
        rank_weighted_portfolio_return,
        zscore_weighted_portfolio_return,
        rank_weighted_turnover,
        zscore_weighted_turnover,
    ):
        arr[invalid_rows] = np.nan
    quantile_spread[invalid_rows] = np.nan

    return {
        "spearman_ic": spearman_ic,
        "pearson_ic": pearson_ic,
        "rank_weighted_portfolio_return": rank_weighted_portfolio_return,
        "zscore_weighted_portfolio_return": zscore_weighted_portfolio_return,
        "rank_weighted_turnover": rank_weighted_turnover,
        "zscore_weighted_turnover": zscore_weighted_turnover,
        "quantile_spread": quantile_spread,
        "n_valid_assets": n_valid_assets.astype(int),
        "coverage": coverage,
        "calibration_slope": calibration_slope,
        "mean_forecast": mean_forecast,
        "std_forecast": std_forecast,
        "mean_target": mean_target,
        "std_target": std_target,
        "calibration_curve": calibration_curve,
    }


def _rank_weighted_portfolio_weights(alpha: FloatArray) -> FloatArray:
    """Return 200% gross centered-rank long-short portfolio weights."""
    ranks = cs_rank(alpha, axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_rank = np.nanmean(ranks, axis=1, keepdims=True)
    centered = ranks - mean_rank
    centered = np.where(np.isfinite(centered), centered, 0.0)
    gross = np.sum(np.abs(centered), axis=1, keepdims=True)
    return _LONG_SHORT_GROSS * safe_divide(centered, gross, fill_value=0.0)


def _zscore_weighted_portfolio_weights(alpha: FloatArray) -> FloatArray:
    """Return 200% gross centered-forecast long-short portfolio weights."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_alpha = np.nanmean(alpha, axis=1, keepdims=True)
    centered = alpha - mean_alpha
    centered = np.where(np.isfinite(centered), centered, 0.0)
    gross = np.sum(np.abs(centered), axis=1, keepdims=True)
    return _LONG_SHORT_GROSS * safe_divide(centered, gross, fill_value=0.0)


def _turnover(weights: FloatArray) -> FloatArray:
    """Compute one-way weight turnover between consecutive observations."""
    out = np.full(weights.shape[0], np.nan, dtype=float)
    if weights.shape[0] > 1:
        out[1:] = np.sum(np.abs(np.diff(weights, axis=0)), axis=1)
    return out


def _quantile_spread(
    alpha: FloatArray, target: FloatArray, quantiles: tuple[float, ...]
) -> FloatArray:
    """Compute top-minus-bottom target returns by forecast quantile."""
    out = np.full((alpha.shape[0], len(quantiles)), np.nan, dtype=float)
    probs = np.array([*quantiles, *(1.0 - q for q in quantiles)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        thresholds = np.nanquantile(alpha, probs, axis=1)
    low_thresholds = thresholds[: len(quantiles)]
    high_thresholds = thresholds[len(quantiles) :]

    for i, (low, high) in enumerate(zip(low_thresholds, high_thresholds, strict=True)):
        low_mask = alpha <= low[:, None]
        high_mask = alpha >= high[:, None]
        low_mean = _masked_mean(target, low_mask)
        high_mean = _masked_mean(target, high_mask)
        out[:, i] = high_mean - low_mean
    return out


def _masked_mean(values: FloatArray, mask: FloatArray) -> FloatArray:
    """Compute row-wise means over finite masked values."""
    valid = mask & np.isfinite(values)
    total = np.sum(np.where(valid, values, 0.0), axis=1)
    count = np.sum(valid, axis=1)
    return safe_divide(total, count, fill_value=np.nan)


def _calibration_slope(
    alpha: FloatArray, target: FloatArray, weights: FloatArray | None
) -> float:
    r"""Compute the forecast scale multiplier :math:`\sum w \alpha y / \sum w \alpha^2`."""
    valid = np.isfinite(alpha) & np.isfinite(target)
    if weights is None:
        weights = valid.astype(float)
    else:
        weights = np.where(valid & np.isfinite(weights), weights, 0.0)
    numerator = float(np.sum(weights * np.where(valid, alpha * target, 0.0)))
    denominator = float(np.sum(weights * np.where(valid, alpha**2, 0.0)))
    return safe_divide(numerator, denominator, fill_value=np.nan)


def _calibration_curve(alpha: FloatArray, target: FloatArray) -> pd.DataFrame:
    """Compute forecast-bucket realized target means."""
    alpha_flat = alpha.ravel()
    target_flat = target.ravel()
    valid = np.isfinite(alpha_flat) & np.isfinite(target_flat)
    alpha_flat = alpha_flat[valid]
    target_flat = target_flat[valid]
    if alpha_flat.size == 0:
        return pd.DataFrame(
            columns=["bucket", "mean_forecast", "mean_target", "n_observations"]
        )

    quantiles = np.linspace(0.0, 1.0, _CALIBRATION_BINS + 1)
    edges = np.unique(np.nanquantile(alpha_flat, quantiles))
    if edges.size < 2:
        return pd.DataFrame(
            {
                "bucket": [0],
                "mean_forecast": [float(np.nanmean(alpha_flat))],
                "mean_target": [float(np.nanmean(target_flat))],
                "n_observations": [int(alpha_flat.size)],
            }
        )
    bucket = np.searchsorted(edges[1:-1], alpha_flat, side="right")
    records = []
    for idx in range(edges.size - 1):
        bucket_mask = bucket == idx
        if not np.any(bucket_mask):
            continue
        records.append(
            {
                "bucket": idx,
                "mean_forecast": float(np.mean(alpha_flat[bucket_mask])),
                "mean_target": float(np.mean(target_flat[bucket_mask])),
                "n_observations": int(np.sum(bucket_mask)),
            }
        )
    return pd.DataFrame.from_records(records)


def _compute_holding_period_diagnostics(
    *,
    alpha: FloatArray,
    X: AssetPanel | AssetPanelView,
    target: str,
    eval_idx: IntArray,
    holding_period: int,
    signal_lag: int,
    n_forward_periods: int,
    cs_weights: FloatArray | None,
    quantiles: tuple[float, ...],
    min_count: int,
) -> pd.DataFrame:
    """Compute common-sample diagnostics by cumulative holding period."""
    windows = tuple(
        (
            period * holding_period,
            period * holding_period,
            signal_lag,
        )
        for period in range(1, n_forward_periods + 1)
    )
    return _compute_forward_window_diagnostics(
        alpha=alpha,
        X=X,
        target=target,
        eval_idx=eval_idx,
        windows=windows,
        index_name="holding_period",
        cs_weights=cs_weights,
        quantiles=quantiles,
        min_count=min_count,
    )


def _compute_decay(
    *,
    alpha: FloatArray,
    X: AssetPanel | AssetPanelView,
    target: str,
    eval_idx: IntArray,
    holding_period: int,
    signal_lag: int,
    n_forward_periods: int,
    cs_weights: FloatArray | None,
    quantiles: tuple[float, ...],
    min_count: int,
) -> pd.DataFrame:
    """Compute common-sample diagnostics by disjoint forward period."""
    windows = tuple(
        (
            period,
            holding_period,
            signal_lag + (period - 1) * holding_period,
        )
        for period in range(1, n_forward_periods + 1)
    )
    return _compute_forward_window_diagnostics(
        alpha=alpha,
        X=X,
        target=target,
        eval_idx=eval_idx,
        windows=windows,
        index_name="period",
        cs_weights=cs_weights,
        quantiles=quantiles,
        min_count=min_count,
    )


def _compute_forward_window_diagnostics(
    *,
    alpha: FloatArray,
    X: AssetPanel | AssetPanelView,
    target: str,
    eval_idx: IntArray,
    windows: tuple[tuple[int, int, int], ...],
    index_name: str,
    cs_weights: FloatArray | None,
    quantiles: tuple[float, ...],
    min_count: int,
) -> pd.DataFrame:
    """Compute common-sample diagnostics for a set of forward target windows."""
    targets = [
        _forward_mean_return(X[target], horizon=horizon, lag=lag)
        for _, horizon, lag in windows
    ]
    common_eval_idx = eval_idx
    for target_forward in targets:
        valid = (
            np.isfinite(alpha[common_eval_idx])
            & np.isfinite(target_forward[common_eval_idx])
            & X.estimation_mask[common_eval_idx]
        )
        common_eval_idx = common_eval_idx[np.sum(valid, axis=1) >= min_count]

    records = []
    if common_eval_idx.size == 0:
        records = [
            _forward_window_record(index_name=index_name, index_value=index_value)
            for index_value, _, _ in windows
        ]
        return pd.DataFrame.from_records(records).set_index(index_name)

    for (index_value, _, _), target_forward in zip(windows, targets, strict=True):
        diagnostics = _compute_diagnostics(
            alpha=alpha[common_eval_idx],
            target=target_forward[common_eval_idx],
            eligible_mask=X.estimation_mask[common_eval_idx],
            cs_weights=None if cs_weights is None else cs_weights[common_eval_idx],
            quantiles=quantiles,
            min_count=min_count,
        )
        records.append(
            _forward_window_record(
                index_name=index_name,
                index_value=index_value,
                diagnostics=diagnostics,
            )
        )
    return pd.DataFrame.from_records(records).set_index(index_name)


def _forward_window_record(
    *,
    index_name: str,
    index_value: int,
    diagnostics: dict[str, object] | None = None,
) -> dict[str, float]:
    """Return one summary record for a forward target window."""
    record = {index_name: index_value}
    if diagnostics is None:
        record.update(
            {
                "spearman_mean_ic": np.nan,
                "spearman_icir": np.nan,
                "spearman_ic_t_stat": np.nan,
                "pearson_mean_ic": np.nan,
                "pearson_icir": np.nan,
                "pearson_ic_t_stat": np.nan,
                "rank_weighted_portfolio_mean": np.nan,
                "rank_weighted_portfolio_ir": np.nan,
                "zscore_weighted_portfolio_mean": np.nan,
                "zscore_weighted_portfolio_ir": np.nan,
                "mean_coverage": np.nan,
            }
        )
        return record

    spearman_ic = _correlation_stats(diagnostics["spearman_ic"], ratio_name="icir")
    pearson_ic = _correlation_stats(diagnostics["pearson_ic"], ratio_name="icir")
    rank_weighted_portfolio = _return_stats(
        diagnostics["rank_weighted_portfolio_return"]
    )
    zscore_weighted_portfolio = _return_stats(
        diagnostics["zscore_weighted_portfolio_return"]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_coverage = float(np.nanmean(diagnostics["coverage"]))
    record.update(
        {
            "spearman_mean_ic": spearman_ic["mean"],
            "spearman_icir": spearman_ic["icir"],
            "spearman_ic_t_stat": spearman_ic["t_stat"],
            "pearson_mean_ic": pearson_ic["mean"],
            "pearson_icir": pearson_ic["icir"],
            "pearson_ic_t_stat": pearson_ic["t_stat"],
            "rank_weighted_portfolio_mean": rank_weighted_portfolio["mean"],
            "rank_weighted_portfolio_ir": rank_weighted_portfolio["ir"],
            "zscore_weighted_portfolio_mean": zscore_weighted_portfolio["mean"],
            "zscore_weighted_portfolio_ir": zscore_weighted_portfolio["ir"],
            "mean_coverage": mean_coverage,
        }
    )
    return record


def _correlation_stats(arr: FloatArray, *, ratio_name: str) -> dict[str, float]:
    """Compute summary statistics for correlation values."""
    valid = np.isfinite(arr)
    n_observations = int(np.sum(valid))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = float(np.nanmean(arr))
        std = float(np.nanstd(arr, ddof=1))
    ratio = safe_divide(mean, std, fill_value=np.nan)
    return {
        "mean": mean,
        "std": std,
        ratio_name: ratio,
        "t_stat": ratio * np.sqrt(n_observations),
        "hit_rate": _hit_rate(arr),
    }


def _portfolio_stats(
    returns: FloatArray, turnover: FloatArray, annualization_factor: float = 1.0
) -> dict[str, float]:
    """Compute annualized return statistics with average turnover."""
    return_stats = _annualized_return_stats(returns, annualization_factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_turnover = float(np.nanmean(turnover))
    return {**return_stats, "mean_turnover": mean_turnover}


def _annualized_return_stats(
    returns: FloatArray, annualization_factor: float = 1.0
) -> dict[str, float]:
    """Compute annualized mean, volatility, IR and hit rate for returns."""
    return_stats = _return_stats(returns)
    annualized_mean = return_stats["mean"] * annualization_factor
    annualized_vol = return_stats["vol"] * np.sqrt(annualization_factor)
    return {
        "annualized_mean": annualized_mean,
        "annualized_vol": annualized_vol,
        "annualized_ir": safe_divide(
            annualized_mean, annualized_vol, fill_value=np.nan
        ),
        "hit_rate": return_stats["hit_rate"],
    }


def _return_stats(returns: FloatArray) -> dict[str, float]:
    """Compute mean, volatility, IR and hit rate for returns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = float(np.nanmean(returns))
        vol = float(np.nanstd(returns, ddof=1))
    return {
        "mean": mean,
        "vol": vol,
        "ir": safe_divide(mean, vol, fill_value=np.nan),
        "hit_rate": _hit_rate(returns),
    }


def _hit_rate(values: FloatArray) -> float:
    """Compute the positive fraction over finite values."""
    valid = np.isfinite(values)
    return safe_divide(np.sum(values[valid] > 0.0), np.sum(valid), fill_value=np.nan)


def _coverage_stats(coverage: FloatArray, n_valid_assets: IntArray) -> dict[str, float]:
    """Compute summary statistics for evaluation coverage."""
    return {
        "mean": float(np.nanmean(coverage)),
        "std": float(np.nanstd(coverage, ddof=1)),
        "ir": np.nan,
        "hit_rate": np.nan,
        "n_valid_assets": float(np.nanmean(n_valid_assets)),
    }


def _rolling(arr: FloatArray, observations: FloatArray, window: int) -> pd.Series:
    """Compute a rolling mean series indexed by observations."""
    return pd.Series(arr, index=observations).rolling(window).mean().iloc[window - 1 :]


def _plot_lines(
    series_map: dict[str, pd.Series],
    *,
    title: str,
    yaxis_title: str,
    ref_value: float | None = None,
) -> go.Figure:
    """Plot one or more time series as lines."""
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, (name, series) in enumerate(series_map.items()):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=format_plot_label(name),
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )
    if ref_value is not None:
        fig.add_hline(
            y=ref_value,
            line_width=1,
            line_dash="dash",
            line_color="gray",
        )
    fig.update_layout(title=title, yaxis_title=yaxis_title)
    return fig
