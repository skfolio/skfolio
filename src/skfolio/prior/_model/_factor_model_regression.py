"""Cross-sectional regression diagnostics of the factor model."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from functools import cached_property
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from skfolio.prior._model._factor_model_plots import (
    _multi_line_plot,
    _plot_single_ts,
    _rolling_title,
)
from skfolio.utils.figure import format_plot_labels
from skfolio.utils.stats import (
    safe_divide,
)


class _CrossSectionalRegressionDiagnosticsMixin:
    """Cross-sectional regression diagnostics, mixed into
    :class:`~skfolio.prior.FactorModel`.

    These methods are prefixed with `cs_regression_`. They require point-in-time
    exposures, estimated factor returns and idiosyncratic returns, as in
    characteristics-based models, and are not available for time-series factor models
    that only store a static loading matrix.

    This mixin is not usable on its own: it reads the fields and private helpers of
    `FactorModel`.
    """

    @cached_property
    def cs_regression_scores(self) -> pd.DataFrame:
        r"""Fit diagnostics for each cross-sectional factor regression.

        This property is available when the model contains point-in-time
        exposures, estimated factor returns and idiosyncratic returns, as in
        characteristics-based cross-sectional factor models. It is not available
        for time-series factor models without point-in-time exposures.

        * `r2`: cross-sectional :math:`R^2`,

        .. math::

            R^2_t = 1 - \frac{\sum_i w_{ti}\,\varepsilon_{ti}^2}
                             {\sum_i w_{ti}\,(r_{ti} - \bar{r}_t)^2}

        * `adjusted_r2`: :math:`R^2` adjusted for the effective number of regressors
          :math:`k`,

        .. math::

            \bar{R}^2_t = 1 - (1 - R^2_t)\,\frac{n_t - 1}{n_t - k - 1}

        * `aic`: Akaike Information Criterion,

        .. math::

            \mathrm{AIC}_t = n_t \ln\!\left(\frac{\mathrm{RSS}_t}{n_t}\right) + 2k

        * `bic`: Bayesian Information Criterion,

        .. math::

            \mathrm{BIC}_t = n_t \ln\!\left(\frac{\mathrm{RSS}_t}{n_t}\right)
            + k \ln(n_t)

        Here :math:`n_t` is the number of valid samples at observation :math:`t` and
        :math:`k = \text{n\_regressors}` is the effective number of regressors (reduced
        dimension when family constraints are active). Lower AIC/BIC indicate a better
        fit-complexity trade-off; BIC penalises complexity more heavily than AIC for
        large cross-sections.

        Returns
        -------
        scores : DataFrame of shape (n_observations - exposure_lag, 4)
            Index aligned with the lagged regression observations.
            Columns: `r2`, `adjusted_r2`, `aic`, `bic`.
        """
        self._require(
            ["exposures", "factor_returns", "idio_returns"], "cs_regression_scores"
        )

        regression_data = self._regression_data
        lagged_exposures = regression_data.exposures
        factor_returns = regression_data.factor_returns
        idio_returns = regression_data.idio_returns
        regression_weights = regression_data.regression_weights

        systematic_returns = (
            lagged_exposures @ factor_returns[:, :, np.newaxis]
        ).squeeze(-1)
        asset_returns = systematic_returns + idio_returns
        finite_mask = np.isfinite(asset_returns)
        estimation_mask, estimation_weights = self._estimation_mask_and_weights(
            finite_mask, regression_weights
        )
        n_valid = estimation_mask.sum(axis=1)
        weight_sum = estimation_weights.sum(axis=1)
        normalized_weights = safe_divide(
            estimation_weights, weight_sum[:, None], fill_value=0.0
        )
        asset_returns = np.where(estimation_mask, asset_returns, 0.0)
        idio_returns = np.where(estimation_mask, idio_returns, 0.0)
        rss = (normalized_weights * idio_returns**2).sum(axis=1)
        mean = (normalized_weights * asset_returns).sum(axis=1)
        tss = (normalized_weights * (asset_returns - mean[:, None]) ** 2).sum(axis=1)
        r2 = 1.0 - safe_divide(rss, tss, fill_value=np.nan)

        k = self._n_regressors
        valid_aic = n_valid > k
        valid_adj = n_valid > k + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            log_msr = np.where(valid_aic, np.log(rss), np.nan)
            adjusted_r2 = np.where(
                valid_adj,
                1.0 - (1.0 - r2) * (n_valid - 1) / (n_valid - k - 1),
                np.nan,
            )
            aic = np.where(valid_aic, n_valid * log_msr + 2 * k, np.nan)
            bic = np.where(valid_aic, n_valid * log_msr + k * np.log(n_valid), np.nan)

        return pd.DataFrame(
            {"r2": r2, "adjusted_r2": adjusted_r2, "aic": aic, "bic": bic},
            index=self._aligned("observations"),
        )

    @property
    def cs_regression_t_stats(self) -> pd.DataFrame:
        r"""Cross-sectional regression coefficient t-statistics.

        .. math::

            t_{tj} = \frac{\hat{\beta}_{tj}}{\mathrm{SE}(\hat{\beta}_{tj})}

        where :math:`\hat{\beta}_{tj}` is the estimated coefficient of factor
        :math:`j` at observation :math:`t`. In a cross-sectional factor model,
        this coefficient is the per-observation factor return. The standard error is
        derived from :math:`\hat\sigma^2_t (X^\top W X)^{-1}`.

        A common rule of thumb is that :math:`|t| > 2` suggests significance at
        approximately the 5 % level.

        When :attr:`family_constraint_basis` is set, the design matrix and factor
        returns are projected into the reduced (full-rank) basis, so the columns are the
        reduced-basis factor names rather than the full `factor_names`.

        Returns
        -------
        cs_regression_t_stats : DataFrame
            Time-indexed t-statistics of shape
            `(n_observations - exposure_lag, n_reduced_factors)`.
        """
        return pd.DataFrame(
            self._gram_diagnostics.t_stats,
            index=self._aligned("observations"),
            columns=self._reduced_regression_factor_names,
        )

    def cs_regression_t_stat_exceedance_rate(self, threshold: float = 2.0) -> pd.Series:
        r"""Fraction of observations with significant cross-sectional regression t-statistics.

        The t-statistic exceedance rate measures how often a factor's cross-sectional
        t-statistic exceeds the absolute threshold: :math:`|t| > \text{threshold}`.
        With `threshold=2.0`, a factor whose true cross-sectional coefficient is zero
        and whose t-statistics are approximately Gaussian would exceed the threshold
        about 5 % of the time. Rates above this reference level indicate that the factor
        is repeatedly significant across observations.

        Parameters
        ----------
        threshold : float, default=2.0
            Absolute t-statistic threshold for significance.

        Returns
        -------
        cs_regression_t_stat_exceedance_rate : Series
            Shape `(n_reduced_factors,)`.
            Fraction of significant observations per factor.
        """
        t_stats = self._gram_diagnostics.t_stats
        significant = np.abs(t_stats) > threshold
        n_valid = np.sum(np.isfinite(t_stats), axis=0)
        rates = safe_divide(np.nansum(significant, axis=0), n_valid, fill_value=0.0)
        return pd.Series(
            rates,
            index=self._reduced_regression_factor_names,
            name="cs_regression_t_stat_exceedance_rate",
        )

    def plot_cs_regression_scores(
        self,
        score: Literal["adjusted_r2", "r2", "aic", "bic"] = "adjusted_r2",
        window: int = 30,
        title: str | None = None,
    ) -> go.Figure:
        """Plot a cross-sectional regression score over time.

        Draws the selected per-observation score as a faded line and overlays its
        rolling mean over `window` observations to highlight changes in fit quality.
        A horizontal line marks the full-sample average and is annotated with its
        numerical value.

        Parameters
        ----------
        score : str, default="adjusted_r2"
            Score to plot. Must be one of `"r2"`, `"adjusted_r2"`, `"aic"`, or `"bic"`.

        window : int, default=30
            Number of observations required for the rolling mean.

        title : str, optional
            Custom title.

        Returns
        -------
        fig : go.Figure
        """
        score_labels = {
            "r2": "R\u00b2",
            "adjusted_r2": "Adjusted R\u00b2",
            "aic": "AIC",
            "bic": "BIC",
        }

        if score not in score_labels:
            raise ValueError(
                f"`score` must be one of {list(score_labels)}, got {score!r}."
            )

        series = self.cs_regression_scores[score]
        label = score_labels[score]
        return _plot_single_ts(
            series.rename(label),
            title=title or _rolling_title(label, window),
            yaxis_title=label,
            window=window,
            show_raw=True,
            mean_fmt=".4f",
        )

    def plot_cs_regression_t_stats(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        window: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        r"""Plot absolute cross-sectional regression t-statistics over time per factor.

        When `window` is provided, plots the rolling mean of :math:`|t|` over
        `window` observations instead of the raw values. A horizontal reference line
        at :math:`|t| = 2` marks the conventional significance threshold.

        Parameters
        ----------
        factors : list of str, optional
            Subset of factor names to include.

        families : str, list of str, optional
            Factor families to include. Ignored when `factors` is given.

        window : int, optional
            If provided, plot the rolling mean of :math:`|t|`.

        title : str, optional
            Custom title.

        Returns
        -------
        fig : go.Figure
        """
        factor_indices, factor_names = self._resolve_factor_subset(
            factors, families, reduced_basis=True, regression_only=True
        )
        abs_t_stats = np.abs(self._gram_diagnostics.t_stats[:, factor_indices])
        df = pd.DataFrame(
            abs_t_stats, index=self._aligned("observations"), columns=factor_names
        )
        if window is not None:
            df = df.rolling(window=window).mean()

        default_title = (
            "|t|-statistic per Factor"
            if window is None
            else f"Rolling Mean |t|-statistic ({window} observations)"
        )

        fig = _multi_line_plot(df, title=title or default_title, yaxis_title="|t|")
        fig.add_hline(
            y=2,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="|t| = 2",
            annotation_position="top left",
        )
        return fig

    def plot_cs_regression_t_stat_exceedance_rate(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        threshold: float = 2.0,
        title: str | None = None,
    ) -> go.Figure:
        r"""Bar chart of the cross-sectional regression t-statistic exceedance rate.

        The t-statistic exceedance rate is the fraction of observations where
        :math:`|t| >` `threshold`. A vertical reference line at 5% marks the
        conventional null-rate benchmark used at `threshold = 2`; for other thresholds
        it is only an approximate guide and the exact Gaussian null rate is
        :math:`2\,\Phi(-\text{threshold})`.

        Parameters
        ----------
        factors : list of str, optional
            Subset of factor names to include. Takes precedence over `families` when
            specified.

        families : str, list of str, optional
            Factor families to include. Ignored when `factors` is given.

        threshold : float, default=2.0
            Absolute t-statistic threshold.

        title : str, optional
            Custom title.

        Returns
        -------
        fig : go.Figure
        """
        _, factor_names = self._resolve_factor_subset(
            factors, families, reduced_basis=True, regression_only=True
        )
        rates = self.cs_regression_t_stat_exceedance_rate(threshold=threshold)
        rates = rates.loc[factor_names]
        order = np.argsort(rates.values)
        sorted_values = rates.values[order]
        sorted_names = format_plot_labels([str(rates.index[i]) for i in order])

        fig = go.Figure(
            go.Bar(
                x=sorted_values,
                y=sorted_names,
                orientation="h",
                marker_color="rgb(31, 119, 180)",
            )
        )
        fig.add_vline(
            x=0.05,
            line_width=1,
            line_dash="dash",
            line_color="gray",
        )
        x_max = max(float(np.max(sorted_values)), 0.05)
        tick_step = 0.1
        tickvals = np.sort(
            np.unique(
                np.concatenate([np.arange(0, x_max + tick_step, tick_step), [0.05]])
            )
        )
        fig.update_layout(
            title=title
            or f"Cross-sectional Regression t-Statistic Exceedance Rate (|t| > {threshold})",
            xaxis_title="Exceedance Rate",
            yaxis_title="Factor",
        )
        fig.update_xaxes(tickmode="array", tickvals=tickvals, tickformat=".0%")
        return fig
