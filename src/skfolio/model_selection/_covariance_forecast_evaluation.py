"""Covariance forecast evaluation."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as sst
import sklearn as sk
import sklearn.base as skb
import sklearn.utils as sku
from sklearn.pipeline import Pipeline

from skfolio.metrics._covariance import (
    _aggregated_return_and_effective_covariance,
    _get_covariance,
    _prepare_active_subset,
)
from skfolio.model_selection._validation import _route_params
from skfolio.model_selection._walk_forward import WalkForward
from skfolio.typing import ArrayLike, FloatArray, IntArray
from skfolio.utils.stats import inverse_volatility_weights, squared_mahalanobis_dist
from skfolio.utils.tools import fit_single_estimator, safe_indexing

__all__ = [
    "CovarianceForecastComparison",
    "CovarianceForecastEvaluation",
    "covariance_forecast_evaluation",
]


_NUMERICAL_THRESHOLD = 1e-12

_VALID_DIAGNOSTICS = frozenset({"mahalanobis", "diagonal", "bias"})

_DIAGNOSTIC_LABELS = {
    "mahalanobis": "Mahalanobis ratio",
    "diagonal": "Diagonal ratio",
    "bias": "Bias statistic",
}


@dataclass(frozen=True, eq=False)
class CovarianceForecastEvaluation:
    r"""Out-of-sample covariance forecast evaluation.

    Stores per-step calibration diagnostics produced by
    :func:`~skfolio.model_selection.covariance_forecast_evaluation` or
    :func:`~skfolio.model_selection.online_covariance_forecast_evaluation`
    and provides summary statistics and plots.

    The four core diagnostics are:

    * **Mahalanobis calibration ratio**: tests whether the full covariance
      structure (all eigenvalue directions) is correctly specified. At each
      step, let :math:`r_t` be the one-period realized return vector and let
      :math:`R^{(h)}` be the aggregated return over the evaluation window of
      :math:`h` observations. The squared Mahalanobis distance
      :math:`d^2 = {R^{(h)}}^\top(h\,\Sigma)^{-1}R^{(h)}` yields the
      calibration ratio :math:`d^2 / n`, where :math:`n` is the number of
      active assets. The target is 1.0. A value above 1.0 indicates
      underestimated risk; below 1.0 indicates overestimated risk.

    * **Diagonal calibration ratio**: tests whether the individual asset
      variances are correctly specified, ignoring correlations. Computed as
      :math:`\frac{1}{n}\sum_i (R_i^{(h)})^2 / (h_i\,\sigma_i^2)` where
      :math:`h_i` is the number of finite returns for asset :math:`i` in the
      evaluation window. The target is 1.0. A value above 1.0 indicates
      underestimated volatilities; below 1.0 indicates overestimated
      volatilities.

    * **Portfolio standardized returns**: tests whether the covariance is
      well calibrated along one or more portfolio directions rather than
      across all directions. For a portfolio with weights :math:`w`, the
      realized portfolio return is standardized by the matching forecast
      portfolio volatility: :math:`b = r_p / \hat\sigma_p` with
      :math:`r_p = w^\top R^{(h)}` and
      :math:`\hat\sigma_p^{2} = w^\top(h\,\Sigma)w`.
      Under correct calibration :math:`b_t` has mean 0 and standard
      deviation 1. The bias statistic :math:`B = \mathrm{std}(b_t)`
      summarizes forecast quality: :math:`B \approx 1` is well calibrated,
      :math:`B > 1` indicates underestimated risk, :math:`B < 1` indicates
      overestimated risk.

    * **Portfolio QLIKE**: evaluates portfolio variance forecasts along one
      or more portfolio directions by comparing the forecast portfolio
      variance with the realized sum of squared portfolio returns over the
      evaluation window. Lower values indicate better portfolio variance
      forecasts.

    When `X_test` contains NaNs (e.g. holidays, pre-listing, or
    post-delisting periods), only finite observations contribute to the
    aggregated return. For portfolio diagnostics, NaN returns for active
    assets contribute zero to the realized portfolio return and the forecast
    covariance is scaled by the pairwise observation count matrix :math:`H`
    (Hadamard product :math:`H \odot \Sigma`) so that the realized portfolio
    variance and forecast variance follow the same missing-data convention.
    In skfolio, NaN diagonal entries in the forecast covariance mark inactive
    assets, which are excluded from the evaluation.

    When multiple test portfolios are provided, portfolio-level diagnostics
    are computed for each portfolio independently. The
    cross-portfolio distribution of bias statistics reveals anisotropic
    calibration errors that a single portfolio might miss.

    Parameters
    ----------
    observations : ndarray of shape (n_steps,)
        Time index labels for each evaluation step.

    horizon : int
        Number of observations per evaluation window. Every window has exactly
        this many observations.

    squared_mahalanobis_distance : ndarray of shape (n_steps,)
        Squared Mahalanobis distance
        :math:`d_t^2 = {R_t^{(h)}}^\top(h\,\Sigma_t)^{-1}R_t^{(h)}`.
        Under correct Gaussian calibration each value follows a
        :math:`\chi^2(n)` distribution, where :math:`n` is the number of
        active assets.

    mahalanobis_calibration_ratio : ndarray of shape (n_steps,)
        :math:`d_t^2 / n`, where :math:`n` is the number of active assets.
        Target is 1.0. Tests whether the full covariance structure (all
        eigenvalue directions) is correctly specified.

    diagonal_calibration_ratio : ndarray of shape (n_steps,)
        :math:`\frac{1}{n}\sum_i (R_{i,t}^{(h)})^2 / (h_{i,t}\,\sigma_{i,t}^2)`.
        Target is 1.0. Tests individual asset variances only.

    portfolio_standardized_return : ndarray of shape (n_steps, n_portfolios)
        :math:`b_t = r_{p,t} / \hat\sigma_{p,t}`.
        Target mean is 0.0 and target std is 1.0 (the bias statistic).

    portfolio_variance_qlike_loss : ndarray of shape (n_steps, n_portfolios)
        :math:`\log(\hat\sigma_{p,t}^{2}) +
        \sum_{j=1}^{h} r_{p,t,j}^{2} / \hat\sigma_{p,t}^{2}`.
        Compares the forecast portfolio variance with the realized sum of
        squared portfolio returns over the evaluation window. Lower values
        are better.

    n_valid_assets : ndarray of shape (n_steps,)
        Number of active assets used at each evaluation step.

    n_portfolios : int
        Number of test portfolios.

    name : str or None, default=None
        Display name for the evaluation.

    Examples
    --------
    >>> from skfolio.model_selection import online_covariance_forecast_evaluation
    >>> from skfolio.moments import EWCovariance
    >>>
    >>> evaluation = online_covariance_forecast_evaluation(  # doctest: +SKIP
    ...     EWCovariance(half_life=30),
    ...     X,
    ...     warmup_size=252,
    ... )
    >>> evaluation.summary()  # doctest: +SKIP
    >>> evaluation.plot_calibration()  # doctest: +SKIP
    """

    observations: FloatArray
    horizon: int
    squared_mahalanobis_distance: FloatArray
    mahalanobis_calibration_ratio: FloatArray
    diagonal_calibration_ratio: FloatArray
    portfolio_standardized_return: FloatArray
    portfolio_variance_qlike_loss: FloatArray
    n_valid_assets: IntArray
    n_portfolios: int
    name: str | None = None

    @property
    def bias_statistic(self) -> FloatArray:
        r"""Per-portfolio bias statistic.

        Computed as the sample standard deviation of the portfolio
        standardized returns :math:`B_k = \mathrm{std}(b_{k,t})` for each
        test portfolio :math:`k`.

        A value near 1.0 indicates well-calibrated risk forecasts.
        Values above 1.0 indicate underestimated risk; values below 1.0
        indicate overestimated risk.

        Returns
        -------
        bias : ndarray of shape (n_portfolios,)
        """
        return np.std(self.portfolio_standardized_return, axis=0, ddof=1)

    def bias_statistic_summary(self) -> pd.Series:
        r"""Cross-portfolio distribution of bias statistics.

        Computes percentiles of bias statistics across test portfolios. This
        is useful for evaluating covariance forecast quality using a set
        of representative portfolios.

        Under Gaussian returns with perfect forecasts, :math:`B^2(T-1)`
        follows a :math:`\chi^2(T-1)` distribution where :math:`T` is the
        number of evaluation steps. Reference bands can be derived from
        the appropriate chi-squared quantiles:
        :math:`B_{p} = \sqrt{\chi^2_{p}(T-1) / (T-1)}`. In financial return
        series, heavy tails widen these bands because the sampling variance of
        :math:`B` increases.

        Returns
        -------
        summary : Series
        """
        bias_values = self.bias_statistic
        return pd.Series(
            {
                "p5": float(np.percentile(bias_values, 5)),
                "p25": float(np.percentile(bias_values, 25)),
                "median": float(np.median(bias_values)),
                "p75": float(np.percentile(bias_values, 75)),
                "p95": float(np.percentile(bias_values, 95)),
                "mean": float(np.mean(bias_values)),
                "n_portfolios": self.n_portfolios,
            },
            name="Bias statistic",
        )

    def summary(self) -> pd.DataFrame:
        r"""Consolidated summary statistics.

        Returns a DataFrame with one row per metric and columns `mean`, `median`, `std`,
        `p5`, `p95`, `mad_from_target`, and `target`.

        * For calibration ratios, the target is `1.0`, so `mad_from_target` is the mean
          absolute deviation from `1.0`.

        * For portfolio standardized returns, the target mean is `0.0`, so
          `mad_from_target` is the mean absolute value. The `std` column corresponds to
          the bias statistic :math:`B = \mathrm{std}(b_t)`, whose target is `1.0`.
          Values near `1.0` indicate well-calibrated risk forecasts, values above `1.0`
          indicate underestimated risk, and values below `1.0` indicate overestimated
          risk.When only one portfolio is evaluated, the `std` column is exactly that
          portfolio's bias statistic. When multiple portfolios are evaluated,
          portfolio-level diagnostics are first computed separately for each portfolio
          and then aggregated by their median. In particular, the `std` column becomes
          the median of the per-portfolio bias statistics. See also
          :attr:`bias_statistic` and :meth:`bias_statistic_summary`.

        * For QLIKE loss, there is no fixed numeric target. Accordingly,
          `mad_from_target` is NaN and `target` is `"lower is better"`.

        Returns
        -------
        summary : DataFrame
        """
        if self.n_portfolios > 1:
            std_ret_stats = _median_portfolio_stats(
                self.portfolio_standardized_return, _centered_stats
            )
            qlike_stats = _median_portfolio_stats(
                self.portfolio_variance_qlike_loss, _loss_stats
            )
        else:
            std_ret_stats = _centered_stats(self.portfolio_standardized_return[:, 0])
            qlike_stats = _loss_stats(self.portfolio_variance_qlike_loss[:, 0])

        records: dict[str, dict[str, object]] = {
            "Mahalanobis ratio": _ratio_stats(self.mahalanobis_calibration_ratio),
            "Diagonal ratio": _ratio_stats(self.diagonal_calibration_ratio),
            "Portfolio standardized returns": std_ret_stats,
            "Portfolio QLIKE": qlike_stats,
        }
        return pd.DataFrame(records).T

    def exceedance_summary(
        self,
        confidence_levels: tuple[float, ...] = (0.95, 0.99),
    ) -> pd.DataFrame:
        r"""Exceedance rate summary.

        Compares squared Mahalanobis distances to :math:`\chi^2` thresholds.
        The rate is sensitive not only to covariance misspecification but also
        to heavy tails, regime shifts, and non-Gaussian standardized returns.
        It is best used as a comparative metric across estimators rather than
        as an absolute calibration test.

        Parameters
        ----------
        confidence_levels : tuple of float, default=(0.95, 0.99)
            Confidence levels used to define the upper chi-squared thresholds.

        Returns
        -------
        summary : DataFrame
            Indexed by `confidence_level` with columns `observed_rate` and
            `deviation`, where `deviation` is measured relative to the target
            exceedance rate :math:`1 - \text{confidence\_level}`.
        """
        sq_mahal_dist = self.squared_mahalanobis_distance
        records: list[dict[str, float]] = []
        for confidence_level in confidence_levels:
            indicators = _exceedance_indicators(
                sq_mahal_dist,
                n_valid_assets=self.n_valid_assets,
                confidence_level=confidence_level,
            )
            rate = (
                float(np.nanmean(indicators))
                if np.isfinite(indicators).any()
                else float("nan")
            )
            records.append(
                {
                    "confidence_level": confidence_level,
                    "observed_rate": rate,
                    "deviation": rate - (1 - confidence_level),
                }
            )
        return pd.DataFrame(records).set_index("confidence_level")

    def plot_calibration(
        self,
        diagnostics: tuple[str, ...] = ("mahalanobis", "diagonal", "bias"),
        window: int = 50,
        title: str | None = None,
    ) -> go.Figure:
        r"""Rolling calibration diagnostics over time.

        Plots rolling calibration diagnostics with a reference line at 1.0.
        By default all three diagnostics are shown: rolling mean of the
        Mahalanobis ratio, rolling mean of the diagonal ratio, and rolling
        standard deviation of the portfolio standardized return (bias
        statistic).

        For multiple portfolios, the bias statistic shows the median across
        portfolios with a P5-P95 shaded band.

        Parameters
        ----------
        diagnostics : tuple of str, default=("mahalanobis", "diagonal", "bias")
            Which diagnostics to include. Valid values are `"mahalanobis"`,
            `"diagonal"`, and `"bias"`.

        window : int, default=50
            Rolling window length.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        _validate_diagnostics(diagnostics)
        series_map, bands = _build_calibration_series(
            self,
            diagnostics,
            window,
        )
        if title is None:
            title = f"Rolling Calibration Diagnostics ({window}-observation window)"

        return _plot_lines(
            series_map,
            title=title,
            yaxis_title="Calibration",
            ref_value=1.0,
            bands=bands or None,
        )

    def plot_qlike_loss(
        self,
        window: int = 50,
        title: str | None = None,
    ) -> go.Figure:
        """Rolling portfolio QLIKE loss over time.

        The QLIKE loss compares the forecast portfolio variance with the
        realized sum of squared portfolio returns over the evaluation
        window. Lower values are better.

        For multiple portfolios, a shaded band shows the P5-P95 range across portfolios,
        with a line for the median.

        Parameters
        ----------
        window : int, default=50
            Rolling window length.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        series_map = {}
        bands = {}
        if self.n_portfolios > 1:
            median, p5, p95 = _rolling_portfolio_band(
                self.portfolio_variance_qlike_loss,
                self.observations,
                window,
                "mean",
            )
            series_map["QLIKE"] = median
            bands["QLIKE"] = (p5, p95)
        else:
            series_map["QLIKE"] = _rolling(
                self.portfolio_variance_qlike_loss[:, 0],
                self.observations,
                window,
            )
        if title is None:
            title = f"Rolling Portfolio QLIKE Loss ({window}-observation window)"

        return _plot_lines(
            series_map,
            title=title,
            yaxis_title="QLIKE Loss",
            bands=bands or None,
        )

    def plot_exceedance(
        self,
        confidence_levels: tuple[float, ...] = (0.95, 0.99),
        window: int = 50,
        title: str | None = None,
    ) -> go.Figure:
        r"""Rolling exceedance rates over time.

        Compares squared Mahalanobis distances to :math:`\chi^2` thresholds.
        The rate is sensitive not only to covariance misspecification but also
        to heavy tails, regime shifts, and non-Gaussian standardized returns.
        It is best used as a comparative metric across estimators rather than
        as an absolute calibration test.

        Parameters
        ----------
        confidence_levels : tuple of float, default=(0.95, 0.99)
            Confidence levels used to define the upper chi-squared thresholds.

        window : int, default=50
            Rolling window length.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        colors = px.colors.qualitative.Plotly
        sq_mahal_dist = self.squared_mahalanobis_distance

        series_map = {}
        for confidence_level in confidence_levels:
            indicators = _exceedance_indicators(
                sq_mahal_dist,
                n_valid_assets=self.n_valid_assets,
                confidence_level=confidence_level,
            )
            series_map[f"Rate (confidence_level={confidence_level})"] = _rolling(
                indicators,
                self.observations,
                window,
            )

        if title is None:
            title = f"Rolling Exceedance Rate ({window}-observation window)"

        fig = _plot_lines(
            series_map,
            title=title,
            yaxis_title="Exceedance Rate",
        )

        for i, confidence_level in enumerate(confidence_levels):
            color = colors[i % len(colors)]
            fig.add_hline(
                y=1 - confidence_level,
                line_width=1,
                line_dash="dash",
                line_color=color,
                annotation_text=f"target={(1 - confidence_level):.0%}",
                annotation_position=("top left" if i == 0 else "bottom left"),
            )
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        return fig


@dataclass(frozen=True, eq=False)
class CovarianceForecastComparison:
    r"""Side-by-side comparison of covariance forecast evaluations.

    Aggregates multiple :class:`CovarianceForecastEvaluation` instances and provides
    combined summary tables and overlay plots for comparing estimator performance.

    Parameters
    ----------
    evaluations : list of CovarianceForecastEvaluation
        Evaluation results to compare.

    names : list of str or None, default=None
        Override display names. When provided, must have the same length as
        `evaluations`. When `None`, defaults to each evaluation's
        :attr:`~CovarianceForecastEvaluation.name` (falling back to
        `"Estimator 0"`, `"Estimator 1"`, etc. when the name is unset).

    Examples
    --------
    >>> from skfolio.model_selection import (
    ...     CovarianceForecastComparison,
    ...     online_covariance_forecast_evaluation,
    ... )
    >>> from skfolio.moments import EWCovariance
    >>>
    >>> evaluatio_30 = online_covariance_forecast_evaluation(  # doctest: +SKIP
    ...     EWCovariance(half_life=30), X, warmup_size=252,
    ... )
    >>> evaluatio_60 = online_covariance_forecast_evaluation(  # doctest: +SKIP
    ...     EWCovariance(half_life=60), X, warmup_size=252,
    ... )
    >>> comparison = CovarianceForecastComparison(  # doctest: +SKIP
    ...     [evaluatio_30, evaluatio_60],
    ...     names=["EWCov(30)", "EWCov(60)"],
    ... )
    >>> comparison.summary()  # doctest: +SKIP
    >>> comparison.plot_calibration()  # doctest: +SKIP
    """

    evaluations: list[CovarianceForecastEvaluation]
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
            resolved = list(self.names)
        else:
            resolved = [
                ev.name if ev.name is not None else f"Estimator {i}"
                for i, ev in enumerate(self.evaluations)
            ]
        object.__setattr__(self, "_names", resolved)

    def _named_evaluations(self):
        """Iterate over (name, evaluation) pairs."""
        return zip(self._names, self.evaluations, strict=True)

    def summary(self) -> pd.DataFrame:
        r"""Consolidated summary statistics for all estimators.

        Returns a DataFrame with metrics as rows and a column-level
        MultiIndex `(estimator, stat)` where stat is one of `mean`,
        `median`, `std`, `p5`, `p95`, `mad_from_target`, `target`.

        Returns
        -------
        summary : DataFrame
        """
        return pd.concat(
            {name: ev.summary() for name, ev in self._named_evaluations()},
            axis=1,
            names=["estimator"],
        )

    def bias_statistic_summary(self) -> pd.DataFrame:
        r"""Cross-portfolio bias statistic distribution for all estimators.

        Returns a DataFrame indexed by estimator name with percentile
        columns and portfolio count.

        Returns
        -------
        summary : DataFrame
        """
        return pd.DataFrame(
            {
                name: ev.bias_statistic_summary()
                for name, ev in self._named_evaluations()
            }
        ).T

    def exceedance_summary(
        self,
        confidence_levels: tuple[float, ...] = (0.95, 0.99),
    ) -> pd.DataFrame:
        r"""Exceedance rate summary for all estimators.

        Returns a DataFrame with confidence levels as rows and a column-level
        MultiIndex `(estimator, stat)` where stat is `observed_rate` or
        `deviation`.

        Parameters
        ----------
        confidence_levels : tuple of float, default=(0.95, 0.99)
            Confidence levels used to define the upper chi-squared thresholds.

        Returns
        -------
        summary : DataFrame
        """
        return pd.concat(
            {
                name: ev.exceedance_summary(confidence_levels)
                for name, ev in self._named_evaluations()
            },
            axis=1,
            names=["estimator"],
        )

    def plot_calibration(
        self,
        diagnostics: tuple[str, ...] = ("mahalanobis", "diagonal", "bias"),
        window: int = 50,
        title: str | None = None,
    ) -> go.Figure:
        r"""Rolling calibration diagnostics comparison.

        Overlays calibration diagnostics from all estimators on one figure.
        Each `(estimator, diagnostic)` pair gets a distinct auto-assigned
        color.

        Parameters
        ----------
        diagnostics : tuple of str, default=("mahalanobis", "diagonal", "bias")
            Which diagnostics to include. Valid values are `"mahalanobis"`,
            `"diagonal"`, and `"bias"`.

        window : int, default=50
            Rolling window length.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        _validate_diagnostics(diagnostics)
        single_diag = len(diagnostics) == 1

        all_series = {}
        all_bands = {}
        for est_name, ev in self._named_evaluations():
            prefix = None if single_diag else est_name

            est_series, est_bands = _build_calibration_series(
                ev,
                diagnostics,
                window,
                prefix,
            )

            if single_diag:
                old_key = next(iter(est_series))
                est_series = {est_name: est_series[old_key]}
                if old_key in est_bands:
                    est_bands = {est_name: est_bands[old_key]}
                else:
                    est_bands = {}

            all_series.update(est_series)
            all_bands.update(est_bands)

        if title is None:
            title = f"Rolling Calibration Diagnostics ({window}-observation window)"

        return _plot_lines(
            all_series,
            title=title,
            yaxis_title="Calibration",
            ref_value=1.0,
            bands=all_bands or None,
        )

    def plot_qlike_loss(
        self,
        window: int = 50,
        title: str | None = None,
    ) -> go.Figure:
        """Rolling portfolio QLIKE loss comparison.

        Overlays QLIKE loss from all estimators on one figure. For
        evaluations with multiple portfolios, the median across portfolios
        is shown with a P5-P95 band.

        Parameters
        ----------
        window : int, default=50
            Rolling window length.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        series_map = {}
        bands = {}

        for name, ev in self._named_evaluations():
            if ev.n_portfolios > 1:
                median, p5, p95 = _rolling_portfolio_band(
                    ev.portfolio_variance_qlike_loss,
                    ev.observations,
                    window,
                    "mean",
                )
                series_map[name] = median
                bands[name] = (p5, p95)
            else:
                series_map[name] = _rolling(
                    ev.portfolio_variance_qlike_loss[:, 0],
                    ev.observations,
                    window,
                )

        if title is None:
            title = f"Rolling Portfolio QLIKE Loss ({window}-observation window)"

        return _plot_lines(
            series_map,
            title=title,
            yaxis_title="QLIKE Loss",
            bands=bands or None,
        )

    def plot_exceedance(
        self,
        confidence_level: float = 0.95,
        window: int = 50,
        title: str | None = None,
    ) -> go.Figure:
        r"""Rolling exceedance rate comparison at a fixed confidence level.

        Overlays exceedance rates from all estimators at a single
        confidence level on one figure.

        Parameters
        ----------
        confidence_level : float, default=0.95
            Confidence level used to define the upper chi-squared threshold.

        window : int, default=50
            Rolling window length.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        series_map = {}

        for name, ev in self._named_evaluations():
            indicators = _exceedance_indicators(
                ev.squared_mahalanobis_distance,
                n_valid_assets=ev.n_valid_assets,
                confidence_level=confidence_level,
            )
            series_map[name] = _rolling(
                indicators,
                ev.observations,
                window,
            )
        if title is None:
            title = (
                f"Rolling Exceedance Rate (confidence_level={confidence_level}, "
                f"{window}-observation window)"
            )
        fig = _plot_lines(
            series_map,
            title=title,
            yaxis_title="Exceedance Rate",
            ref_value=1 - confidence_level,
        )
        fig.update_yaxes(tickformat=".0%", rangemode="tozero")
        return fig


def covariance_forecast_evaluation(
    estimator: skb.BaseEstimator | Pipeline,
    X: ArrayLike,
    y: ArrayLike | None = None,
    train_size: int = 252,
    test_size: int = 1,
    expand_train: bool = False,
    portfolio_weights: ArrayLike | None = None,
    purged_size: int = 0,
    params: dict | None = None,
) -> CovarianceForecastEvaluation:
    r"""Evaluate out-of-sample covariance forecast quality using walk-forward
    cross-validation.

    At each fold the estimator is fitted from scratch on the training window
    and the fitted covariance is evaluated against the next `test_size`
    observations. This is the batch counterpart of
    :func:`~skfolio.model_selection.online_covariance_forecast_evaluation`,
    which instead updates the estimator incrementally via `partial_fit`.

    The walk-forward scheme is controlled by `train_size` and
    `expand_train`, mirroring the semantics of
    :class:`~skfolio.model_selection.WalkForward`:

    * `expand_train=False` (default): rolling window of fixed `train_size`.
    * `expand_train=True`: expanding window starting from the first `train_size`
      observations.

    Every evaluation window contains exactly `test_size` observations, ensuring that
    diagnostics (in particular QLIKE) are directly comparable across folds.

    Four core diagnostics are computed:

    * **Mahalanobis calibration ratio**: tests whether the full covariance
      structure (all eigenvalue directions) is correctly specified. The
      target is 1.0. A value above 1.0 indicates underestimated risk;
      below 1.0 indicates overestimated risk.
    * **Diagonal calibration ratio**: tests whether the individual asset
      variances are correctly specified, ignoring correlations. The target
      is 1.0. A value above 1.0 indicates underestimated volatilities;
      below 1.0 indicates overestimated volatilities.
    * **Portfolio standardized returns / bias statistic**: tests whether
      the covariance is well calibrated along one or more portfolio
      directions.
    * **Portfolio QLIKE**: evaluates portfolio variance forecasts along
      one or more portfolio directions by comparing the forecast portfolio
      variance with the realized sum of squared portfolio returns over the
      evaluation window. Lower values indicate better portfolio variance
      forecasts.

    When the test returns contain NaNs (e.g. holidays, pre-listing, or post-delisting
    periods), only finite observations contribute to the aggregated return. For
    portfolio diagnostics, NaN returns for active assets contribute zero to the realized
    portfolio return and the forecast covariance is scaled by the pairwise observation
    count matrix :math:`H` (Hadamard product :math:`H \odot \Sigma`) so that the
    realized portfolio variance and forecast variance follow the same missing-data
    convention. In skfolio, NaN diagonal entries in the forecast covariance mark
    inactive assets, which are excluded from the evaluation.

    Parameters
    ----------
    estimator : BaseEstimator or Pipeline
        Fitted estimator or Pipeline. Must expose `covariance_` or
        `return_distribution_.covariance` after fitting.

    X : array-like of shape (n_observations, n_assets)
        Asset returns.

    y : Ignored
        Present for scikit-learn API compatibility.

    train_size : int, default=252
        Number of observations in each training window (rolling or initial expanding
        window size).

    test_size : int, default=1
        Number of observations per evaluation window. All windows have exactly this many
        observations.

    expand_train : bool, default=False
        If `True`, each subsequent training window includes all past observations
        (expanding window). If `False`, a rolling window of fixed `train_size` is used.

    portfolio_weights : array-like of shape (n_assets,) or (n_portfolios, n_assets), optional
        Portfolio weights for portfolio-level diagnostics (bias statistic and QLIKE).

        If `None` (default), inverse-volatility weights are used, recomputed dynamically
        at each step from the forecast covariance. This neutralizes volatility
        dispersion so that high-volatility assets do not dominate the diagnostic.

        If a 1D array is provided, a single static portfolio is used.

        If a 2D array of shape `(n_portfolios, n_assets)` is provided, each row defines
        a test portfolio and diagnostics are computed independently for each.

        For equal-weight calibration, pass `portfolio_weights=np.ones(n_assets) / n_assets`.

    purged_size : int, default=0
        Number of observations to skip between training and test data.

    params : dict, optional
        Parameters routed to the estimator's `fit` via metadata routing.

    Returns
    -------
    evaluation : CovarianceForecastEvaluation
        Frozen dataclass with per-step calibration arrays, summary statistics, and
        plotting methods.

    Raises
    ------
    ValueError
        If the data is too short for at least one evaluation fold.

    See Also
    --------
    online_covariance_forecast_evaluation : Online counterpart that updates
        the estimator incrementally via `partial_fit`.
    CovarianceForecastEvaluation : Result dataclass with summary statistics
        and plotting methods.
    CovarianceForecastComparison : Compare multiple evaluation results side
        by side with combined summary tables and overlay plots.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.model_selection import covariance_forecast_evaluation
    >>> from skfolio.moments import LedoitWolf
    >>> from skfolio.preprocessing import prices_to_returns
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> evaluation = covariance_forecast_evaluation(  # doctest: +SKIP
    ...     LedoitWolf(),
    ...     X,
    ...     train_size=252,
    ...     test_size=5,
    ... )
    >>> evaluation.summary()  # doctest: +SKIP
    >>> evaluation.bias_statistic  # doctest: +SKIP
    >>> evaluation.plot_calibration()  # doctest: +SKIP
    """
    estimator = sk.clone(estimator)
    X, y = sku.indexable(X, y)
    observations = X.index if hasattr(X, "index") else np.arange(len(X))
    routed_params = _route_params(
        estimator,
        params,
        owner="covariance_forecast_evaluation",
        callee="fit",
    )

    portfolio_weights = _normalize_portfolio_weights(portfolio_weights)
    n_portfolios = portfolio_weights.shape[0] if portfolio_weights is not None else 1

    cv = WalkForward(
        test_size=test_size,
        train_size=train_size,
        expand_train=expand_train,
        purged_size=purged_size,
    )

    evaluation_observations: list = []
    squared_mahalanobis_distances: list[float] = []
    mahalanobis_calibration_ratios: list[float] = []
    diagonal_calibration_ratios: list[float] = []
    portfolio_standardized_returns: list[FloatArray] = []
    portfolio_variance_qlike_losses: list[FloatArray] = []
    active_asset_counts: list[int] = []

    for train_idx, test_idx in cv.split(X, y):
        est = sk.clone(estimator)
        fit_single_estimator(
            est,
            X,
            y,
            fit_params=routed_params.estimator_params,
            indices=train_idx,
        )

        covariance = _get_covariance(est)
        X_test = safe_indexing(X, indices=test_idx)

        step = _compute_step_diagnostics(
            covariance,
            X_test,
            portfolio_weights,
        )
        if step is None:
            continue

        (
            squared_mahalanobis_distance,
            mahalanobis_calibration_ratio,
            diagonal_calibration_ratio,
            standardized_portfolio_return,
            portfolio_variance_qlike_loss,
            n_active_assets,
        ) = step

        squared_mahalanobis_distances.append(squared_mahalanobis_distance)
        mahalanobis_calibration_ratios.append(mahalanobis_calibration_ratio)
        diagonal_calibration_ratios.append(diagonal_calibration_ratio)
        portfolio_standardized_returns.append(standardized_portfolio_return)
        portfolio_variance_qlike_losses.append(portfolio_variance_qlike_loss)
        active_asset_counts.append(n_active_assets)
        evaluation_observations.append(observations[test_idx[0]])

    return CovarianceForecastEvaluation(
        observations=np.array(evaluation_observations),
        horizon=test_size,
        squared_mahalanobis_distance=np.array(squared_mahalanobis_distances),
        mahalanobis_calibration_ratio=np.array(mahalanobis_calibration_ratios),
        diagonal_calibration_ratio=np.array(diagonal_calibration_ratios),
        portfolio_standardized_return=np.array(portfolio_standardized_returns),
        portfolio_variance_qlike_loss=np.array(portfolio_variance_qlike_losses),
        n_valid_assets=np.array(active_asset_counts, dtype=int),
        n_portfolios=n_portfolios,
        name=str(estimator),
    )


def _normalize_portfolio_weights(
    portfolio_weights: ArrayLike | None,
) -> FloatArray | None:
    """Normalize portfolio weights for portfolio-level diagnostics.

    Parameters
    ----------
    portfolio_weights : array-like of shape (n_assets,) or (n_portfolios, n_assets), optional
        User-provided portfolio weights.

    Returns
    -------
    normalized_portfolio_weights : ndarray of shape (n_portfolios, n_assets) or None
        Normalized weights, or `None` if not provided.
    """
    if portfolio_weights is None:
        return None

    weights = np.atleast_2d(np.asarray(portfolio_weights, dtype=np.float64))
    return weights / weights.sum(axis=1, keepdims=True)


def _compute_step_diagnostics(
    covariance: FloatArray, X_test: FloatArray, portfolio_weights: FloatArray | None
) -> tuple[float, float, float, FloatArray, FloatArray, int] | None:
    """Compute all per-step calibration diagnostics for one evaluation window.

    Parameters
    ----------
    covariance : ndarray of shape (n_assets, n_assets)
        Forecast covariance matrix. In skfolio, NaN diagonal entries mark
        inactive assets.

    X_test : ndarray of shape (n_obs, n_assets)
        Realized returns for the evaluation window.

    portfolio_weights : ndarray of shape (n_portfolios, n_assets) or None
        Portfolio weights (`None` for dynamic inverse-vol).

    Returns
    -------
    diagnostics : tuple of (
        float,
        float,
        float,
        ndarray of shape (n_portfolios,),
        ndarray of shape (n_portfolios,),
        int,
    ) or None
        Tuple of squared Mahalanobis distance, Mahalanobis calibration ratio,
        diagonal calibration ratio, standardized portfolio return,
        portfolio variance QLIKE loss, and the number of active assets.
        Returns `None` when no assets are active.
    """
    result = _prepare_active_subset(covariance, X_test)
    if result is None:
        return None

    active_cov, active_returns, active_asset_indices, n_active_assets = result
    active_mask = np.isfinite(active_returns)
    active_returns_filled = np.where(active_mask, active_returns, 0.0)
    aggregated_return, effective_cov = _aggregated_return_and_effective_covariance(
        active_returns, active_cov
    )
    squared_mahalanobis_distance = float(
        squared_mahalanobis_dist(aggregated_return, effective_cov)
    )
    mahalanobis_calibration_ratio = squared_mahalanobis_distance / n_active_assets
    effective_horizon_per_asset = active_mask.sum(axis=0)
    scaled_variances = effective_horizon_per_asset * np.maximum(
        np.diag(active_cov), _NUMERICAL_THRESHOLD
    )
    diagonal_calibration_ratio = float(
        np.sum(aggregated_return**2 / scaled_variances) / n_active_assets
    )

    if portfolio_weights is None:
        portfolio_weights = inverse_volatility_weights(active_cov)[np.newaxis, :]
    else:
        portfolio_weights = portfolio_weights[:, active_asset_indices]
        row_sums = portfolio_weights.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, _NUMERICAL_THRESHOLD)
        portfolio_weights = portfolio_weights / row_sums

    forecast_portfolio_variance = np.maximum(
        np.sum(
            portfolio_weights * (portfolio_weights @ effective_cov),
            axis=1,
        ),
        _NUMERICAL_THRESHOLD,
    )
    realized_portfolio_return = portfolio_weights @ aggregated_return
    standardized_portfolio_return = realized_portfolio_return / np.sqrt(
        forecast_portfolio_variance
    )
    realized_portfolio_returns = active_returns_filled @ portfolio_weights.T
    realized_portfolio_variance = np.sum(realized_portfolio_returns**2, axis=0)
    portfolio_variance_qlike_loss = (
        np.log(forecast_portfolio_variance)
        + realized_portfolio_variance / forecast_portfolio_variance
    )

    return (
        squared_mahalanobis_distance,
        mahalanobis_calibration_ratio,
        diagonal_calibration_ratio,
        standardized_portfolio_return,
        portfolio_variance_qlike_loss,
        n_active_assets,
    )


def _rolling(
    arr: FloatArray,
    observations: FloatArray,
    window: int,
    stats_type: str = "mean",
) -> pd.Series:
    """Compute a rolling statistic aligned on the observation axis.

    Parameters
    ----------
    arr : ndarray of shape (n_steps,)
        Input series.

    observations : ndarray of shape (n_steps,)
        Observation labels used as the pandas index.

    window : int
        Rolling window length.

    stats_type : {"mean", "std"}, default="mean"
        Rolling statistic to compute.

    Returns
    -------
    series : Series
        Rolling statistic with the warmup period removed.
    """
    series = pd.Series(arr, index=observations)
    if stats_type == "std":
        return series.rolling(window=window).std(ddof=1).iloc[window - 1 :]
    return series.rolling(window=window).mean().iloc[window - 1 :]


def _exceedance_indicators(
    squared_distances: FloatArray, n_valid_assets: IntArray, confidence_level: float
) -> FloatArray:
    """Compute exceedance indicators against chi-squared thresholds.

    Parameters
    ----------
    squared_distances : ndarray of shape (n_steps,)
        Squared Mahalanobis distances.

    n_valid_assets : ndarray of shape (n_steps,)
        Number of active assets used to determine the chi-squared degrees of
        freedom at each step.

    confidence_level : float
        Confidence level used to compute the upper chi-squared threshold.

    Returns
    -------
    indicators : ndarray of shape (n_steps,)
        Array equal to `1.0` when the squared distance exceeds the
        chi-squared threshold, `0.0` otherwise, with NaN preserved for invalid
        rows.
    """
    thresholds = sst.chi2.ppf(confidence_level, df=n_valid_assets)
    valid = np.isfinite(squared_distances)
    indicators = np.full(squared_distances.shape, np.nan, dtype=float)
    indicators[valid] = (squared_distances[valid] > thresholds[valid]).astype(float)
    return indicators


def _rolling_portfolio_band(
    arr: FloatArray, observations: FloatArray, window: int, stats_type: str = "mean"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute rolling portfolio statistics and cross-portfolio bands.

    Parameters
    ----------
    arr : ndarray of shape (n_steps, n_portfolios)
        Per-step portfolio values.

    observations : ndarray of shape (n_steps,)
        Observation labels used as the pandas index.

    window : int
        Rolling window length.

    stats_type : {"mean", "std"}, default="mean"
        Rolling statistic to compute for each portfolio.

    Returns
    -------
    median : Series
        Rolling median across portfolios.

    p5 : Series
        Rolling 5th percentile across portfolios.

    p95 : Series
        Rolling 95th percentile across portfolios.
    """
    df = pd.DataFrame(arr, index=observations)
    if stats_type == "std":
        rolled = df.rolling(window=window).std(ddof=1).iloc[window - 1 :]
    else:
        rolled = df.rolling(window=window).mean().iloc[window - 1 :]
    return (
        rolled.median(axis=1),
        rolled.quantile(0.05, axis=1),
        rolled.quantile(0.95, axis=1),
    )


def _median_portfolio_stats(arr: FloatArray, stats_func: Callable) -> dict[str, object]:
    """Aggregate per-portfolio summary statistics by their median.

    Computes `stats_type` independently for each portfolio column, then
    takes the median of each numeric statistic across portfolios.
    Non-numeric values (strings, NaN sentinels) are passed through from the
    first portfolio.

    Parameters
    ----------
    arr : ndarray of shape (n_steps, n_portfolios)
        Per-step portfolio metric values.

    stats_func : Callable
        Function returning a dictionary of summary statistics for a 1D array.

    Returns
    -------
    stats : dict[str, object]
        Median summary statistics across portfolios.
    """
    per_portfolio = [stats_func(arr[:, k]) for k in range(arr.shape[1])]
    result: dict[str, object] = {}
    for key in per_portfolio[0]:
        values = [d[key] for d in per_portfolio]
        first = values[0]
        if isinstance(first, str) or (isinstance(first, float) and np.isnan(first)):
            result[key] = first
        else:
            result[key] = float(np.median(values))
    return result


def _validate_diagnostics(diagnostics: tuple[str, ...]) -> None:
    """Validate requested diagnostic names.

    Parameters
    ----------
    diagnostics : tuple of str
        Diagnostic identifiers requested by the caller.

    Raises
    ------
    ValueError
        If at least one diagnostic name is unknown.
    """
    invalid = set(diagnostics) - _VALID_DIAGNOSTICS
    if invalid:
        raise ValueError(
            f"Unknown diagnostics {invalid}. "
            f"Valid values are {set(_VALID_DIAGNOSTICS)}."
        )


def _build_calibration_series(
    evaluation: CovarianceForecastEvaluation,
    diagnostics: tuple[str, ...],
    window: int,
    key_prefix: str | None = None,
) -> tuple[dict[str, pd.Series], dict[str, tuple[pd.Series, pd.Series]]]:
    """Build rolling calibration series and optional percentile bands.

    Parameters
    ----------
    evaluation : CovarianceForecastEvaluation
        Evaluation object containing per-step diagnostics.

    diagnostics : tuple of str
        Diagnostic identifiers to include.

    window : int
        Rolling window length.

    key_prefix : str, optional
        Prefix added to each display label.

    Returns
    -------
    series_map : dict[str, Series]
        Rolling diagnostic series keyed by display label.

    bands : dict[str, tuple[Series, Series]]
        Optional lower and upper bands for diagnostics that aggregate across
        multiple portfolios.
    """
    series_map = {}
    bands = {}
    for diag in diagnostics:
        label = _DIAGNOSTIC_LABELS[diag]
        key = f"{key_prefix} - {label}" if key_prefix else label

        if diag == "mahalanobis":
            series_map[key] = _rolling(
                evaluation.mahalanobis_calibration_ratio,
                evaluation.observations,
                window,
            )
        elif diag == "diagonal":
            series_map[key] = _rolling(
                evaluation.diagonal_calibration_ratio,
                evaluation.observations,
                window,
            )
        elif diag == "bias":
            if evaluation.n_portfolios > 1:
                median, p5, p95 = _rolling_portfolio_band(
                    evaluation.portfolio_standardized_return,
                    evaluation.observations,
                    window,
                    "std",
                )
                series_map[key] = median
                bands[key] = (p5, p95)
            else:
                series_map[key] = _rolling(
                    evaluation.portfolio_standardized_return[:, 0],
                    evaluation.observations,
                    window,
                    "std",
                )

    return series_map, bands


def _base_stats(arr: FloatArray) -> dict[str, object]:
    """Compute common summary statistics for a 1D array.

    Parameters
    ----------
    arr : ndarray of shape (n_samples,)
        Input values.

    Returns
    -------
    stats : dict[str, object]
        Dictionary with mean, median, standard deviation, and tail
        percentiles.
    """
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def _ratio_stats(arr: FloatArray) -> dict[str, object]:
    """Compute summary statistics for a calibration ratio.

    Parameters
    ----------
    arr : ndarray of shape (n_samples,)
        Calibration ratio values.

    Returns
    -------
    stats : dict[str, object]
        Summary statistics with target-specific fields for a ratio whose
        calibration target is `1.0`.
    """
    stats = _base_stats(arr)
    stats["mad_from_target"] = float(np.mean(np.abs(arr - 1.0)))
    stats["target"] = 1.0
    return stats


def _centered_stats(arr: FloatArray) -> dict[str, object]:
    """Compute summary statistics for a centered diagnostic.

    Parameters
    ----------
    arr : ndarray of shape (n_samples,)
        Centered diagnostic values.

    Returns
    -------
    stats : dict[str, object]
        Summary statistics with target-specific fields for a metric whose
        target is mean `0` and standard deviation `1`.
    """
    stats = _base_stats(arr)
    stats["mad_from_target"] = float(np.mean(np.abs(arr)))
    stats["target"] = "mean=0, std=1"
    return stats


def _loss_stats(arr: FloatArray) -> dict[str, object]:
    """Compute summary statistics for a loss function.

    Parameters
    ----------
    arr : ndarray of shape (n_samples,)
        Loss values.

    Returns
    -------
    stats : dict[str, object]
        Summary statistics with target metadata indicating that lower values
        are better.
    """
    stats = _base_stats(arr)
    stats["mad_from_target"] = np.nan
    stats["target"] = "lower is better"
    return stats


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a hex color code to an rgba CSS string.

    Parameters
    ----------
    hex_color : str
        Hexadecimal RGB color string such as `#1f77b4`.

    alpha : float
        Opacity between 0 and 1.

    Returns
    -------
    color : str
        RGBA color string usable in Plotly traces.
    """
    hex_digits = hex_color.lstrip("#")
    r, g, b = (
        int(hex_digits[0:2], 16),
        int(hex_digits[2:4], 16),
        int(hex_digits[4:6], 16),
    )
    return f"rgba({r}, {g}, {b}, {alpha})"


def _plot_lines(
    series_map: dict[str, pd.Series],
    title: str,
    yaxis_title: str,
    ref_value: float | None = None,
    bands: dict[str, tuple[pd.Series, pd.Series]] | None = None,
) -> go.Figure:
    """Plot pre-computed time series with optional percentile bands.

    Parameters
    ----------
    series_map : dict[str, Series]
        Mapping from display name to series to plot.

    title : str
        Figure title.

    yaxis_title : str
        Y-axis title.

    ref_value : float, optional
        Horizontal reference level added to the figure.

    bands : dict[str, tuple[Series, Series]], optional
        Optional lower and upper bands keyed by the same names as
        `series_map`.

    Returns
    -------
    fig : go.Figure
        Plotly figure.
    """
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()

    for i, (name, series) in enumerate(series_map.items()):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
                legendgroup=name,
                line=dict(color=color, width=2),
            )
        )

        if bands and name in bands:
            lower, upper = bands[name]
            fill_color = _hex_to_rgba(color, 0.15)
            fig.add_trace(
                go.Scatter(
                    x=upper.index,
                    y=upper.values,
                    mode="lines",
                    legendgroup=name,
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=lower.index,
                    y=lower.values,
                    mode="lines",
                    legendgroup=name,
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                    showlegend=False,
                )
            )

    if ref_value is not None:
        fig.add_hline(
            y=ref_value,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"target={ref_value}",
            annotation_position="top left",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Observation",
        yaxis_title=yaxis_title,
    )
    return fig
