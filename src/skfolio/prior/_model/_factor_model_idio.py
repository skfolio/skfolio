"""Idiosyncratic diagnostics of the factor model."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.stats as scs

from skfolio.prior._model._factor_model_plots import (
    _plot_single_ts,
    _rolling_title,
)
from skfolio.prior._model._factor_model_utils import (
    _cs_kurtosis,
    _cs_skewness,
)
from skfolio.utils.stats import (
    cs_spearman_correlation,
    safe_divide,
)


class _IdioDiagnosticsMixin:
    """Idiosyncratic diagnostics, mixed into :class:`~skfolio.prior.FactorModel`.

    These methods are prefixed with `idio_` and describe the quality of the
    idiosyncratic calibration. Those based on standardized idiosyncratic returns
    additionally require `idio_variances`.

    This mixin is not usable on its own: it reads the fields and private helpers of
    `FactorModel`.
    """

    def idio_calibration_summary(self) -> pd.Series:
        r"""Summary statistics for the calibration quality of standardized idiosyncratic
         returns.

        Computes time-aggregated statistics of the cross-sectional distribution of
        standardized idiosyncratic returns
        :math:`z_{it} = \epsilon_{it} / \hat\sigma_{i,t}`.

        Under a Gaussian assumption, the expected values are :math:`\text{std}(z) = 1`,
        excess kurtosis :math:`= 0`, skewness :math:`= 0`, and the 3-:math:`\sigma` tail
        rate :math:`\approx 0.27\%`. In practice, standardized idiosyncratic returns
        exhibit fat tails, so the tail rate is typically well above 0.27% (values
        around 1--3% are common for equity factor models).

        - `mean_cs_std` close to 1.0 indicates correctly scaled specific risk.
          Values persistently above 1 suggest underestimated risk; below 1 suggests
          overestimated risk.
        - `mean_tail_rate_3sigma` is expected to exceed the Gaussian reference due to
          fat tails.
        - `mean_cs_excess_kurtosis` > 0 (fat tails) and moderate `mean_cs_skewness` are
          typical.

        Returns
        -------
        summary : Series
            Index: `mean_cs_std`, `median_cs_std`, `mean_cs_excess_kurtosis`,
            `mean_cs_skewness`, `mean_tail_rate_3sigma`.
        """
        cs_std = self.idio_calibration.values
        return pd.Series(
            {
                "mean_cs_std": np.nanmean(cs_std),
                "median_cs_std": np.nanmedian(cs_std),
                "mean_cs_excess_kurtosis": np.nanmean(self.idio_kurtosis.values),
                "mean_cs_skewness": np.nanmean(self.idio_skewness.values),
                "mean_tail_rate_3sigma": np.nanmean(self.idio_tail_rate().values),
            },
            name="idio_calibration",
        )

    @cached_property
    def idio_vol_ic(self) -> pd.Series:
        r"""Information Coefficient of idiosyncratic volatility estimates.

        Computes the cross-sectional rank correlation (Spearman) between the predicted
        specific volatility :math:`\hat\sigma_{i,t}` and the next-period absolute
        idiosyncratic return :math:`|\epsilon_{i,t+1}|`.

        If the model captures the cross-sectional scale of idiosyncratic shocks, then
        assets with larger :math:`\hat\sigma_{i,t}` should tend to realize larger
        absolute moves at :math:`t + 1`.

        * High positive values indicate that the model ranks cross-sectional differences
          in idiosyncratic volatility well.
        * This diagnostic can also pick up broad cross-sectional scale effects such as
          size or liquidity, so it should be read together with
          :attr:`idio_vol_residual_dependence` which checks whether the standardized
          idiosyncratic return magnitude :math:`|z_{i,t+1}|` still depends on the
          predicted volatility level.
        """
        self._require(("idio_returns", "idio_variances"), "idio_vol_ic")
        predicted_vol = np.sqrt(np.maximum(self.idio_variances[:-1], 0.0))
        abs_idio_next = np.abs(self.idio_returns[1:])
        corr = cs_spearman_correlation(
            predicted_vol, abs_idio_next, axis=1, min_count=5
        )
        return pd.Series(
            corr, index=self.observations[1:], name="Idio Vol IC (Spearman)"
        )

    @cached_property
    def idio_vol_residual_dependence(self) -> pd.Series:
        r"""Residual dependence of standardized idiosyncratic returns on predicted
        idiosyncratic volatility.

        Computes the cross-sectional rank correlation (Spearman) between the predicted
        specific volatility :math:`\hat\sigma_{i,t}` and the next-period standardized
        absolute idiosyncratic return
        :math:`|\epsilon_{i,t+1}| / \hat\sigma_{i,t} = |z_{i,t+1}|`.
        If the volatility forecast is well calibrated, this standardized magnitude
        should be roughly independent of :math:`\hat\sigma_{i,t}`, so the correlation
        should be close to 0.

        Read together with :attr:`idio_vol_ic`, this diagnostic helps separate ranking
        power from calibration. A desirable pattern is a high :attr:`idio_vol_ic`
        combined with residual dependence near 0.
        """
        self._require(
            ("idio_returns", "idio_variances"), "idio_vol_residual_dependence"
        )
        predicted_vol = np.sqrt(np.maximum(self.idio_variances[:-1], 0.0))
        abs_idio_next = np.abs(self.idio_returns[1:])
        standardized_abs_idio_next = safe_divide(
            abs_idio_next, predicted_vol, fill_value=np.nan
        )
        corr = cs_spearman_correlation(
            predicted_vol, standardized_abs_idio_next, axis=1, min_count=5
        )
        return pd.Series(
            corr,
            index=self.observations[1:],
            name="Idio Vol Residual Dependence (Spearman)",
        )

    @cached_property
    def idio_calibration(self) -> pd.Series:
        """Cross-sectional std of standardized idiosyncratic returns."""
        z = self._standardized_idio_returns()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cs_std = np.nanstd(z, axis=1, ddof=1)
        return pd.Series(
            cs_std, index=self.observations, name="Standardised Idio Return Std"
        )

    def idio_tail_rate(self, threshold: float = 3.0) -> pd.Series:
        r"""Fraction of assets with extreme standardized idiosyncratic returns.

        For each observation, computes the cross-sectional fraction of available
        standardized idiosyncratic returns whose absolute value exceeds
        `threshold`:

        .. math::

            \frac{1}{n_t}\sum_i \mathbf{1}\{|z_{i,t}| > c\},

        where :math:`z_{i,t}` is the standardized idiosyncratic return, :math:`c`
        is `threshold`, and :math:`n_t` is the number of finite standardized
        idiosyncratic returns at observation :math:`t`.

        Under a Gaussian reference model, the expected rate is
        :math:`2\Phi(-c)`. Higher realized rates indicate that the standardized
        residuals have heavier tails than implied by the idiosyncratic volatility
        estimates. In equity factor models, standardized idiosyncratic returns
        are often fat-tailed, so rates above the Gaussian reference are common.

        Parameters
        ----------
        threshold : float, default=3.0
            Absolute standardized-return threshold :math:`c`.

        Returns
        -------
        tail_rate : Series of shape (n_observations,)
            Time series of cross-sectional tail exceedance rates, indexed by
            `observations`.
        """
        z = self._standardized_idio_returns()
        n_valid = np.sum(np.isfinite(z), axis=1)
        n_exceed = np.sum(np.abs(z) > threshold, axis=1)
        rate = safe_divide(n_exceed, n_valid, fill_value=np.nan)
        return pd.Series(rate, index=self.observations, name="Tail Rate")

    @cached_property
    def idio_kurtosis(self) -> pd.Series:
        """Cross-sectional excess kurtosis of standardized idiosyncratic returns."""
        z = self._standardized_idio_returns()
        cs_kurt = _cs_kurtosis(z)
        return pd.Series(cs_kurt, index=self.observations, name="Excess Kurtosis")

    @cached_property
    def idio_skewness(self) -> pd.Series:
        """Cross-sectional skewness of standardized idiosyncratic returns."""
        z = self._standardized_idio_returns()
        cs_skew = _cs_skewness(z)
        return pd.Series(cs_skew, index=self.observations, name="Skewness")

    def plot_idio_calibration(
        self, window: int | None = None, title: str | None = None
    ) -> go.Figure:
        r"""Cross-sectional std of standardized idiosyncratic returns over time.

        Under correct calibration, :math:`\text{std}(z_t) \approx 1`. Persistent
        deviations indicate mis-specified specific risk.

        Parameters
        ----------
        window : int, optional
            Rolling-mean smoothing window.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        return _plot_single_ts(
            self.idio_calibration,
            title=title or _rolling_title("Idiosyncratic Calibration", window),
            yaxis_title="Cross-Sectional Std of Standardised Idio Returns",
            window=window,
            ref_value=1.0,
            ref_label="Ideal = 1.0",
            mean_fmt=".3f",
        )

    def plot_idio_tail_rate(
        self,
        threshold: float = 3.0,
        window: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        r"""Plot the idiosyncratic tail exceedance rate over time.

        For each observation, the plotted value is the fraction of assets whose
        finite standardized idiosyncratic return satisfies
        :math:`|z_{i,t}| > \text{threshold}`. When `window` is provided, the rolling
        mean is plotted to smooth short-lived cross-sectional tail spikes.

        A dashed reference line shows the Gaussian rate
        :math:`2\,\Phi(-\text{threshold})`, which is about 0.27% when `threshold = 3`.
        Persistent values above this reference indicate heavier idiosyncratic residual
        tails than implied by the volatility estimates. In equity factor models,
        standardized idiosyncratic returns are often fat-tailed, so observed rates above
        the Gaussian reference are common.

        Parameters
        ----------
        threshold : float, default=3.0
            Absolute standardized-return threshold.

        window : int, optional
            Rolling-mean smoothing window.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        expected_rate = 2 * scs.norm.sf(threshold)
        return _plot_single_ts(
            self.idio_tail_rate(threshold=threshold),
            title=title
            or _rolling_title(
                "Idiosyncratic Tail Rate",
                window,
                context=f"threshold={threshold}",
            ),
            yaxis_title="Fraction of Assets",
            window=window,
            ref_value=expected_rate,
            ref_label=f"Gaussian: {expected_rate:.2%}",
            mean_fmt=".2%",
            tick_format=".2%",
        )

    def plot_idio_kurtosis(
        self,
        window: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        r"""Cross-sectional excess kurtosis of standardised idiosyncratic returns over
        time.

        Each point is the excess kurtosis of :math:`z_{it}` computed across assets at a
        single observation. The Gaussian reference is zero, but positive values are
        expected because standardised idiosyncratic returns typically have fat tails.

        Parameters
        ----------
        window : int, optional
            Rolling-mean smoothing window.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        return _plot_single_ts(
            self.idio_kurtosis,
            title=title or _rolling_title("Cross-Sectional Excess Kurtosis", window),
            yaxis_title="Excess Kurtosis",
            window=window,
            ref_value=0.0,
            ref_label="Gaussian: 0",
        )

    def plot_idio_skewness(
        self,
        window: int | None = None,
        title: str | None = None,
    ) -> go.Figure:
        r"""Cross-sectional skewness of standardised idiosyncratic returns over time.

        Each point is the skewness of :math:`z_{it}` computed across assets at a single
        observation. The Gaussian reference is zero. Mild negative skewness is common
        for equity factor models.

        Parameters
        ----------
        window : int, optional
            Rolling-mean smoothing window.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        return _plot_single_ts(
            self.idio_skewness,
            title=title or _rolling_title("Cross-Sectional Skewness", window),
            yaxis_title="Skewness",
            window=window,
            ref_value=0.0,
            ref_label="Gaussian: 0",
        )

    def plot_idio_vol_ic(
        self,
        window: int = 60,
        title: str | None = None,
    ) -> go.Figure:
        r"""Information Coefficient (IC) of idiosyncratic volatility estimates.

        Plots the cross-sectional rank correlation (Spearman) between the predicted
        specific volatility :math:`\hat\sigma_{i,t}` and the next-period absolute
        idiosyncratic return :math:`|\epsilon_{i,t+1}|`.

        This is a ranking diagnostic: do names predicted to have larger
        :math:`\hat\sigma_{i,t}` tend to realize larger raw absolute moves.

        - High positive values indicate that the model ranks cross-sectional differences
          in idiosyncratic volatility well.
        - This diagnostic can also pick up broad cross-sectional scale effects such as
          size or liquidity.

        This is a ranking diagnostic, not a calibration diagnostic. For the
        post-standardization check, see :meth:`plot_idio_vol_residual_dependence`.

        Parameters
        ----------
        window : int, default=60
            Rolling window for the smoothed mean.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        return _plot_single_ts(
            self.idio_vol_ic.rename("Rank Correlation (Spearman)"),
            title=title
            or _rolling_title(
                "Idiosyncratic Volatility IC",
                window,
                context="Spearman",
            ),
            yaxis_title="Rank Correlation",
            window=window,
            show_raw=True,
            show_mean=False,
            raw_trace_name="Rank Correlation",
        )

    def plot_idio_vol_residual_dependence(
        self,
        window: int = 60,
        title: str | None = None,
    ) -> go.Figure:
        r"""Residual dependence of standardized idiosyncratic returns on predicted
        idiosyncratic volatility.

        Plots the cross-sectional rank correlation (Spearman) between the predicted
        specific volatility :math:`\hat\sigma_{i,t}` and the next-period standardized
        absolute idiosyncratic return
        :math:`|\epsilon_{i,t+1}| / \hat\sigma_{i,t} = |z_{i,t+1}|`.
        If the volatility forecast is well calibrated, this standardized magnitude
        should be roughly independent of :math:`\hat\sigma_{i,t}`, so the correlation
        should be close to 0.

        Read together with :meth:`plot_idio_vol_ic`, this helps distinguish ranking
        power from calibration. A desirable pattern is high :meth:`plot_idio_vol_ic`
        together with residual dependence near 0.

        Parameters
        ----------
        window : int, default=60
            Rolling window for the smoothed mean.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        return _plot_single_ts(
            self.idio_vol_residual_dependence.rename("Residual Dependence (Spearman)"),
            title=title
            or _rolling_title(
                "Idiosyncratic Volatility Residual Dependence",
                window,
                context="Spearman",
            ),
            yaxis_title=(
                "Rank Correlation (Spearman, predicted idio vol vs "
                "|idio return| / predicted idio vol)"
            ),
            window=window,
            show_raw=True,
            show_mean=False,
            ref_value=0.0,
            raw_trace_name="Residual Dependence",
        )
