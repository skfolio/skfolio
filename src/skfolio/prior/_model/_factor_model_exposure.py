"""Exposure diagnostics of the factor model."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from skfolio.prior._model._factor_model_plots import (
    _add_family_outlines,
    _heatmap,
    _multi_line_plot,
    _plot_single_ts,
    _rolling_title,
)
from skfolio.prior._model._factor_model_utils import (
    _check_correlation_method,
)
from skfolio.typing import (
    FloatArray,
)
from skfolio.utils.figure import format_plot_label, format_plot_labels
from skfolio.utils.stats import (
    CSWeighting,
    CorrelationMethod,
    safe_divide,
)


class _ExposureDiagnosticsMixin:
    """Exposure diagnostics, mixed into :class:`~skfolio.prior.FactorModel`.

    These methods are prefixed with `exposure_`. They describe the point-in-time
    exposure panel or its regression design, and are available only when `exposures`
    is populated.

    This mixin is not usable on its own: it reads the fields and private helpers of
    `FactorModel`.
    """

    def exposure_correlation(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        cs_weighting: CSWeighting = CSWeighting.BENCHMARK,
    ) -> FloatArray:
        """Time-average pairwise correlation matrix of factor exposures.

        Highly correlated exposures indicate redundant factors. They are a
        cross-sectional analogue of multicollinearity diagnostics used in regression,
        where redundant predictors can inflate variance inflation factors (VIFs).

        Pairs involving a factor with degenerate cross-sectional variance (e.g. the
        constant global factor exposure) have an undefined correlation and are
        reported as zero by convention. When two factors are never finite on at
        least 3 common assets at any observation, their correlation cannot be
        estimated and is reported as NaN.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, optional
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        cs_weighting : CSWeighting, default=CSWeighting.BENCHMARK
            Cross-sectional weights for the correlation computation. Falls back
            to `CSWeighting.IDENTITY` with a warning when unavailable.

        Returns
        -------
        corr : ndarray of shape (n_selected_factors, n_selected_factors)
            Time-average correlation matrix.
        """
        self._require("exposures", "exposure_correlation")
        factor_indices, _ = self._resolve_factor_subset(factors, families)
        exposures = self.exposures[:, :, factor_indices]
        weights = self._resolve_cs_weighting(
            cs_weighting,
            latest=False,
            fallback_cs_weighting=CSWeighting.IDENTITY,
        )
        min_count = 3
        eps = 1e-12
        # The weighted variance is computed by cancellation of terms of the order
        # of the weighted square sum, so degenerate (constant) exposures must be
        # detected with a tolerance relative to that scale
        rel_tol = 1e-9
        finite = np.isfinite(exposures)
        mask = finite.astype(float)
        clean_exposures = np.where(finite, exposures, 0.0)
        mask_t = mask.transpose(0, 2, 1)
        clean_exposures_t = clean_exposures.transpose(0, 2, 1)

        n_valid = mask_t @ mask
        if weights is None:
            weight_sum = n_valid
            weighted_sum = clean_exposures_t @ mask
            weighted_square_sum = (clean_exposures**2).transpose(0, 2, 1) @ mask
            weighted_cross_sum = clean_exposures_t @ clean_exposures
        else:
            weights_3d = weights[:, :, np.newaxis]
            weighted_mask = weights_3d * mask
            weighted_exposures = weights_3d * clean_exposures

            weight_sum = weighted_mask.transpose(0, 2, 1) @ mask
            weighted_sum = weighted_exposures.transpose(0, 2, 1) @ mask
            weighted_square_sum = (weighted_exposures * clean_exposures).transpose(
                0, 2, 1
            ) @ mask
            weighted_cross_sum = weighted_exposures.transpose(0, 2, 1) @ clean_exposures

        weighted_sum_t = weighted_sum.swapaxes(1, 2)
        weighted_square_sum_t = weighted_square_sum.swapaxes(1, 2)
        covariance = weighted_cross_sum - safe_divide(
            weighted_sum * weighted_sum_t, weight_sum, fill_value=np.nan
        )
        variance = weighted_square_sum - safe_divide(
            weighted_sum**2, weight_sum, fill_value=np.nan
        )
        variance_t = weighted_square_sum_t - safe_divide(
            weighted_sum_t**2, weight_sum, fill_value=np.nan
        )
        variance = np.maximum(variance, 0.0)
        variance_t = np.maximum(variance_t, 0.0)

        denom = np.sqrt(variance * variance_t)
        pairwise_corr = safe_divide(covariance, denom, fill_value=np.nan, atol=eps)
        insufficient = n_valid < min_count
        degenerate = (variance <= eps + rel_tol * weighted_square_sum) | (
            variance_t <= eps + rel_tol * weighted_square_sum_t
        )
        pairwise_corr[insufficient | degenerate] = np.nan
        with warnings.catch_warnings():
            # All-NaN slices are expected for degenerate pairs and handled below
            warnings.simplefilter("ignore", category=RuntimeWarning)
            corr = np.nanmean(pairwise_corr, axis=0)
        # A pair that is degenerate at every observation with sufficient joint
        # coverage (e.g. any pair involving the constant global factor exposure)
        # has an undefined correlation and is reported as zero by convention
        zero_by_convention = np.isnan(corr) & (degenerate & ~insufficient).any(axis=0)
        corr[zero_by_convention] = 0.0
        np.fill_diagonal(corr, 1.0)
        return corr

    @property
    def exposure_vif(self) -> pd.DataFrame:
        r"""Variance Inflation Factor of the exposure design per observation.

        VIF measures how much the variance of a cross-sectional regression coefficient
        is inflated due to collinearity among factor exposures:

        .. math::

            \mathrm{VIF}_k = (X^\top W X)_{kk} \cdot
                             [(X^\top W X)^{-1}]_{kk}

        A VIF of 1 indicates no collinearity; values above 5-10 suggest problematic
        multicollinearity.

        When :attr:`family_constraint_basis` is set, VIFs are computed in the reduced
        (full-rank) basis.

        Returns
        -------
        exposure_vif : DataFrame
            Time-indexed VIF values of shape
            `(n_observations - exposure_lag, n_reduced_factors)`.
        """
        return pd.DataFrame(
            self._gram_diagnostics.vif,
            index=self._aligned("observations"),
            columns=self._reduced_regression_factor_names,
        )

    @property
    def exposure_condition_number(self) -> pd.Series:
        r"""Condition number of the exposure Gram matrix per observation.

        The condition number :math:`\kappa(X^\top W X)` is the ratio of the largest to
        smallest singular value. Large values indicate near-singular design matrices
        and numerically unstable coefficient estimates. When :attr:`family_constraint_basis`
        is set, the Gram matrix is built in the reduced (full-rank) basis.

        Returns
        -------
        exposure_condition_number : Series
            Time-indexed condition numbers of shape
            `(n_observations - exposure_lag,)`.
        """
        return pd.Series(
            self._gram_diagnostics.condition_number,
            index=self._aligned("observations"),
            name="exposure_condition_number",
        )

    def exposure_ic_summary(
        self,
        correlation_method: CorrelationMethod = CorrelationMethod.SPEARMAN,
        horizon: int = 1,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
    ) -> pd.DataFrame:
        r"""Summary statistics for exposure Information Coefficients (ICs).

        Measures the cross-sectional correlation between factor exposures at :math:`t`
        and the forward mean asset return from :math:`t + 1` to :math:`t + h`, where
        :math:`h` is the forecast *horizon*.

        .. note::

            The IC quantifies **return-predictive** power. In a **risk model**, factors
            are designed to explain covariance structure, not to predict expected
            returns. A factor can be an excellent risk factor even when
            :math:`\mathbb{E}[\text{IC}] \approx 0`. Do not discard a risk factor solely
            because its IC is low: use exposure stability, bias statistics, and variance
            contribution instead.

        Parameters
        ----------
        correlation_method : CorrelationMethod, default=CorrelationMethod.SPEARMAN
            Correlation method used for the exposure IC. `SPEARMAN` computes
            Spearman rank IC. `PEARSON` computes Pearson IC, weighted by
            `regression_weights` when available.

        horizon : int, default=1
            Forward window in number of observations. The mean return from
            :math:`t + 1` to :math:`t + h` is used.

        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, optional
            Factor families to include. `None` includes all factors.

        Returns
        -------
        summary : DataFrame of shape (n_selected_factors, 4)
            Columns: `mean_ic`, `std_ic`, `ic_ir`, `hit_rate`.
        """
        _check_correlation_method(correlation_method)
        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        ic, _ = self._ic(
            correlation_method=correlation_method,
            horizon=horizon,
            factor_indices=factor_indices,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_ic = np.nanmean(ic, axis=0)
            std_ic = np.nanstd(ic, axis=0, ddof=1)
            ic_ir = safe_divide(mean_ic, std_ic, fill_value=np.nan)
            hit_rate = np.nanmean(ic > 0, axis=0)

        return pd.DataFrame(
            {
                "mean_ic": mean_ic,
                "std_ic": std_ic,
                "ic_ir": ic_ir,
                "hit_rate": hit_rate,
            },
            index=factor_names,
        )

    def plot_exposure_vif(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        window: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        r"""Plot exposure Variance Inflation Factors over time per factor.

        When `window` is provided, plots the rolling mean over `window` observations
        instead of raw per-observation values. A horizontal reference line at VIF = 5
        marks the conventional collinearity threshold.

        Parameters
        ----------
        factors : list of str, optional
            Subset of factor names to include.

        families : str, list of str, optional
            Factor families to include. Ignored when `factors` is given.

        window : int, optional
            If provided, plot the rolling mean.

        title : str, optional
            Custom title.

        Returns
        -------
        fig : go.Figure
        """
        factor_indices, factor_names = self._resolve_factor_subset(
            factors, families, reduced_basis=True, regression_only=True
        )
        vif = self._gram_diagnostics.vif[:, factor_indices]
        df = pd.DataFrame(
            vif, index=self._aligned("observations"), columns=factor_names
        )

        if window is not None:
            df = df.rolling(window=window).mean()

        default_title = (
            "Exposure Variance Inflation Factor"
            if window is None
            else f"Rolling Mean Exposure VIF ({window} observations)"
        )

        fig = _multi_line_plot(df, title=title or default_title, yaxis_title="VIF")
        fig.add_hline(
            y=5,
            line_width=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="VIF = 5",
            annotation_position="top left",
        )
        return fig

    def plot_exposure_condition_number(
        self, window: int = 30, title: str | None = None
    ) -> go.Figure:
        """Plot the exposure Gram-matrix condition number over time.

        Draws the per-observation condition number as a faded line and overlays its
        rolling mean over `window` observations. Large values indicate near-collinear
        exposures and less stable coefficient estimates.

        Parameters
        ----------
        window : int, default=30
            Number of observations required for the rolling mean.

        title : str, optional
            Custom title.

        Returns
        -------
        fig : go.Figure
        """
        label = "Exposure Condition Number"
        return _plot_single_ts(
            self.exposure_condition_number.rename(label),
            title=title or _rolling_title(label, window),
            yaxis_title=label,
            window=window,
            show_raw=True,
            mean_fmt=".4f",
        )

    def plot_cumulative_exposure_ic(
        self,
        correlation_method: CorrelationMethod = CorrelationMethod.SPEARMAN,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        title: str | None = None,
    ) -> go.Figure:
        r"""Cumulative exposure Information Coefficient (IC) over time.

        Plots the cumulative sum of the single-period cross-sectional correlation
        between factor exposures at :math:`t` and asset returns at :math:`t + 1`.

        - A monotonically rising curve indicates persistent predictive power
          (positive alpha signal).
        - A flat curve means the factor carries no return-predictive information.
        - A declining curve indicates a contrarian signal (negative alpha).

        For IC decay analysis across different holding periods, use
        :meth:`exposure_ic_summary` with varying `horizon` values instead.

        .. note::

            The IC quantifies **return-predictive** power. In a **risk model**, factors
            are designed to explain covariance structure, not to predict expected
            returns. A factor can be an excellent risk factor even when
            :math:`\mathbb{E}[\text{IC}] \approx 0`.

        Parameters
        ----------
        correlation_method : CorrelationMethod, default=CorrelationMethod.SPEARMAN
            Correlation method used for the exposure IC. `SPEARMAN` computes
            Spearman rank IC. `PEARSON` computes Pearson IC, weighted by
            `regression_weights` when available.

        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, optional
            Factor families to include. `None` includes all factors.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        _check_correlation_method(correlation_method)
        factor_indices, factor_names = self._resolve_factor_subset(
            factors, families, reduced_basis=True
        )
        ic, obs_offset = self._ic(
            correlation_method=correlation_method,
            horizon=1,
            factor_indices=factor_indices,
            reduced_basis=True,
        )
        cum_ic = np.nancumsum(ic, axis=0)

        method_label = format_plot_label(correlation_method.value)
        default_title = f"Cumulative Exposure IC ({method_label})"

        df = pd.DataFrame(
            cum_ic,
            index=self.observations[obs_offset : obs_offset + cum_ic.shape[0]],
            columns=factor_names,
        )
        return _multi_line_plot(
            df, title=title or default_title, yaxis_title="Cumulative IC"
        )

    def plot_exposure_distribution(
        self,
        factor: str,
        observation_idx: int | None = None,
        n_bins: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Cross-sectional histogram of exposures for a single factor.

        When `observation` is `None` (default), all observations are pooled into one
        histogram showing the typical distribution. When an integer index is provided,
        only the exposures at that observation are plotted.

        Parameters
        ----------
        factor : str
            Name of the factor to plot.

        observation_idx : int , optional
            Observation index. `None` pools all dates, `-1` selects the last
            observation, `0` the first, etc.

        n_bins : int , optional
            Number of histogram bins. `None` lets Plotly choose automatically.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        self._require("exposures", "plot_exposure_distribution")
        factor_indices, _ = self._resolve_factor_subset(
            factor_names_to_keep=[factor], family_names_to_keep=None
        )
        factor_idx = factor_indices[0]

        if observation_idx is not None:
            obs_label = str(self.observations[observation_idx])
            values = self.exposures[observation_idx, :, factor_idx]
            default_title = (
                f"Exposure Distribution: {format_plot_label(factor)} ({obs_label})"
            )
        else:
            values = self.exposures[:, :, factor_idx].ravel()
            default_title = (
                f"Exposure Distribution: {format_plot_label(factor)} (all observations)"
            )

        values = values[np.isfinite(values)]

        fig = go.Figure(
            go.Histogram(
                x=values,
                nbinsx=n_bins,
                marker_color="rgb(31, 119, 180)",
                opacity=0.75,
            )
        )
        fig.update_layout(
            title=title or default_title,
            xaxis_title="Exposure",
            yaxis_title="Count",
        )
        return fig

    def plot_exposure_dispersion(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = "style",
        cs_weighting: CSWeighting = CSWeighting.BENCHMARK,
        title: str | None = None,
    ) -> go.Figure:
        """Cross-sectional standard deviation of exposures over time.

        The absolute level depends on how exposures were standardized upstream. When
        the model uses weighted-mean centering or a different variance normalization,
        the equal-weighted cross-sectional std computed here will not be 1.0. Focus on
        temporal stability rather than the absolute level: a collapse may signal
        data-feed issues and an explosion may indicate an outlier.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        cs_weighting : CSWeighting, default=CSWeighting.BENCHMARK
            Cross-sectional weights for the std computation. Falls back to
            `CSWeighting.IDENTITY` with a warning when unavailable.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        self._require("exposures", "plot_exposure_dispersion")

        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        selected_exposures = self.exposures[:, :, factor_indices]
        weights = self._resolve_cs_weighting(
            cs_weighting, latest=False, fallback_cs_weighting=CSWeighting.IDENTITY
        )

        finite_mask = np.isfinite(selected_exposures)
        if weights is None:
            weights_3d = finite_mask.astype(float)
        else:
            weights_3d = weights[:, :, None] * finite_mask.astype(float)

        weight_sum = np.sum(weights_3d, axis=1, keepdims=True)
        normalized_weights = safe_divide(weights_3d, weight_sum, fill_value=0.0)
        weighted_mean = np.sum(
            np.where(finite_mask, selected_exposures, 0.0) * normalized_weights,
            axis=1,
            keepdims=True,
        )
        centered_exposures = np.where(
            finite_mask, selected_exposures - weighted_mean, 0.0
        )
        cs_var = np.sum(normalized_weights * centered_exposures**2, axis=1)
        cs_std = np.sqrt(cs_var)
        cs_std[weight_sum[:, 0, 0] == 0] = np.nan

        df = pd.DataFrame(cs_std, index=self.observations, columns=factor_names)

        fig = _multi_line_plot(
            df, title=title or "Exposure Cross-Sectional Std", yaxis_title="Std"
        )
        fig.update_yaxes(rangemode="tozero")
        return fig

    def plot_exposure_stability(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = "style",
        step: int = 21,
        cs_weighting: CSWeighting = CSWeighting.BENCHMARK,
        title: str | None = None,
    ) -> go.Figure:
        r"""Weighted cross-sectional correlation of exposures between observation
        :math:`t` and :math:`t + \text{step}` over time.

        Measures whether the cross-sectional exposures are stable across the chosen
        horizon.

        The expected level depends on the factor's investment horizon. Slow-moving
        factors (e.g. value, size) should maintain high correlation at the default
        monthly step and values consistently below 0.80 may indicate noisy or poorly
        constructed exposures. Fast-turnover factors (e.g. reversal, short-term
        momentum) are designed to reshuffle quickly and will naturally show low monthly
        stability. For these factors, use a shorter `step` (e.g., 1-5 for daily data) to
        assess stability at the relevant horizon.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        step : int, default=21
            Number of observations between the two cross-sections being compared (e.g.,
            21 for approximately monthly stability with daily data).

        cs_weighting : CSWeighting, default=CSWeighting.BENCHMARK
            Cross-sectional weights for the correlation computation. Falls back
            to `CSWeighting.IDENTITY` with a warning when unavailable.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        self._require("exposures", "plot_exposure_stability")

        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        exposures = self.exposures[:, :, factor_indices]

        if exposures.shape[0] <= step:
            raise ValueError(
                f"Not enough observations ({exposures.shape[0]}) for "
                f"step={step}. Need at least {step + 1}."
            )

        stability = self._exposure_stability(
            exposures, step=step, cs_weighting=cs_weighting
        )
        df = pd.DataFrame(
            stability, index=self.observations[step:], columns=factor_names
        )

        fig = _multi_line_plot(
            df,
            title=title or "Exposure Stability (Weighted Cross-sectional Correlation)",
            yaxis_title="Correlation",
        )
        fig.update_yaxes(range=[-0.05, 1.05])
        return fig

    def plot_exposure_correlation(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        cs_weighting: CSWeighting = CSWeighting.BENCHMARK,
        title: str | None = None,
    ) -> go.Figure:
        """Time-average pairwise correlation heatmap of factor exposures.

        Highly correlated exposures indicate redundant factors and may
        inflate VIF.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        cs_weighting : CSWeighting, default=CSWeighting.BENCHMARK
            Cross-sectional weights for the correlation computation. Falls back
            to `CSWeighting.IDENTITY` with a warning when unavailable.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        corr_avg = self.exposure_correlation(
            factors=factors, families=families, cs_weighting=cs_weighting
        )
        fig = _heatmap(
            corr_avg,
            labels=format_plot_labels(factor_names),
            title=title or "Time-Average Exposure Correlation",
            zmin=-1,
            zmax=1,
        )
        _add_family_outlines(fig, self.factor_families, factor_indices)
        return fig
