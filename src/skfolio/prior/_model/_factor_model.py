"""Factor Model Dataclass."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from skfolio._constants import (
    _CURRENCY,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
    _REGRESSION_WEIGHTS,
)
from skfolio.containers import AssetPanel, AssetPanelView, InactivePolicy
from skfolio.prior._model._covariance_sqrt import CovarianceSqrt
from skfolio.prior._model._factor_model_attribution import _AttributionMixin
from skfolio.prior._model._factor_model_exposure import _ExposureDiagnosticsMixin
from skfolio.prior._model._factor_model_idio import _IdioDiagnosticsMixin
from skfolio.prior._model._factor_model_plots import (
    _add_family_outlines,
    _heatmap,
    _multi_line_plot,
)
from skfolio.prior._model._factor_model_regression import (
    _CrossSectionalRegressionDiagnosticsMixin,
)
from skfolio.prior._model._factor_model_utils import (
    _GramDiagnostics,
    _RegressionData,
    _check_correlation_method,
    _lag1_autocorr,
    _positions_to_indexer,
    _selector_to_positions,
)
from skfolio.prior._model._family_constraint_basis import FamilyConstraintBasis
from skfolio.typing import (
    AnyArray,
    ArrayLike,
    BoolArray,
    FloatArray,
    StrArray,
)
from skfolio.utils._factor_tools import _resolve_factor_subset
from skfolio.utils.figure import format_plot_labels
from skfolio.utils.stats import (
    CSWeighting,
    CorrelationMethod,
    _forward_mean_return,
    cov_to_corr,
    cs_pearson_correlation,
    cs_spearman_correlation,
    safe_cholesky,
    safe_divide,
)

__all__ = ["CorrelationMethod", "FactorModel"]


@dataclass(frozen=True, eq=False)
class FactorModel(
    _IdioDiagnosticsMixin,
    _CrossSectionalRegressionDiagnosticsMixin,
    _ExposureDiagnosticsMixin,
    _AttributionMixin,
):
    r"""Factor model decomposition of asset returns.

    Holds the loading matrix, factor moments and idiosyncratic covariance, together with
    the optional time series of exposures, factor returns and idiosyncratic returns.
    Exposes a factor-structured covariance square root, plus cross-sectional regression
    diagnostics, idiosyncratic-calibration metrics and factor attribution when the
    relevant fields are populated.

    Produced by factor-model prior estimators:

        * :class:`~skfolio.prior.TimeSeriesFactorModel`,
        * :class:`~skfolio.prior.CharacteristicsFactorModel`

    and consumed downstream via :attr:`~skfolio.prior.ReturnDistribution.factor_model`.

    Method groups
    -------------
    Each diagnostic group below is implemented in its own module and mixed in, so the
    group a method belongs to is also where it lives: `_factor_model_idio.py`,
    `_factor_model_regression.py`, `_factor_model_exposure.py` and
    `_factor_model_attribution.py`.

    Factor structure and factor-return methods use the stored loading matrix,
    factor moments and, the factor return time series. They are
    available for both time-series and characteristics-based factor models.
    These include `factor_forecast_correlation`,
    `plot_factor_forecast_correlation`, `plot_factor_forecast_volatilities`,
    `plot_factor_cumulative_returns`, and `predicted_attribution`.

    Cross-sectional regression diagnostics are prefixed with `cs_regression_`.
    They require point-in-time exposures, estimated factor returns and
    idiosyncratic returns, as in characteristics-based models. These include
    `cs_regression_scores`, `cs_regression_t_stats`,
    `cs_regression_t_stat_exceedance_rate`, and their plotting methods.They
    are not available for time-series factor models that only store a static
    loading matrix.

    Exposure diagnostics are prefixed with `exposure_`. They describe the
    point-in-time exposure panel or its regression design and are available only
    when `exposures` is populated. These include `exposure_correlation`,
    `exposure_vif`, `exposure_condition_number`, `exposure_ic_summary`, and
    their plotting methods.

    Idiosyncratic diagnostics are prefixed with `idio_`. Diagnostics based on
    standardized idiosyncratic returns additionally require `idio_variances`.

    Attributes
    ----------
    observations : ndarray of shape (n_observations,)
        Time index labels.

    asset_names : ndarray of shape (n_assets,)
        Asset names.

    factor_names : ndarray of shape (n_factors,)
        Factor names (e.g. `"value"`, `"momentum"`).

    factor_families : ndarray of shape (n_factors,) or None
        Family label for each factor (e.g. `"style"`, `"industry"`).
        Populated by cross-sectional factor models.

    loading_matrix : ndarray of shape (n_assets, n_factors)
        Asset-by-factor loading (exposure) matrix. Time-invariant for time-series factor
        models; the most recent point-in-time loadings for cross-sectional factor models
        (full history in `exposures`).

    exposures : ndarray of shape (n_observations, n_assets, n_factors) or None
        Full historical time series of asset-by-factor exposure (loading) matrices
        following the as-of time-indexing convention. Populated for cross-sectional
        factor models. `None` for time-series factor models, which use the single
        time-invariant `loading_matrix`.

    factor_covariance : ndarray of shape (n_factors, n_factors)
        Factor return covariance matrix. Under family constraints this full-basis
        matrix is rank-deficient; use :attr:`effective_factor_covariance` (paired with
        :attr:`effective_loading_matrix`) for decompositions such as Cholesky.

    factor_mu : ndarray of shape (n_factors,)
        Expected factor returns.

    factor_returns : ndarray of shape (n_observations, n_factors) or None
        Per-period factor returns. For time-series factor models, this is the input
        factor return series; for cross-sectional factor models, this is the per-period
        factor returns estimated from the cross-sectional regression.

    idio_covariance : ndarray of shape (n_assets, n_assets) or (n_assets,)
        Idiosyncratic covariance (diagonal vector or full matrix).

    idio_mu : ndarray of shape (n_assets,) or None
        Factor-orthogonal expected return for each asset, also called orthogonal alpha.
        With the default weighted least-squares projection, it satisfies
        :math:`B^\top W\,\text{idio\_mu}=0`. Custom robust or regularized
        cross-sectional regressors may produce a component that is only approximately
        orthogonal. Distinct from the time-series mean of `idio_returns`, which is not
        enforced to be factor-orthogonal. Populated by cross-sectional factor models.

    idio_returns : ndarray of shape (n_observations, n_assets) or None
        Per-period idiosyncratic returns, obtained from the corresponding factor
        regression. For time-series factor models, these are
        :math:`r - a - Bf`, where :math:`a` is the vector of time-series regression
        intercepts. For cross-sectional factor models, these are
        :math:`R(t) - B(t-\ell)f(t)`.

    idio_variances : ndarray of shape (n_observations, n_assets) or None
        Time-varying per-asset predicted idiosyncratic variances
        :math:`\hat\sigma^2_{i,t}`. Populated by cross-sectional factor models.

    exposure_lag : int, default=1
        Lag applied to time-varying exposures under the as-of time-indexing convention.
        The default value of `1` aligns exposures at :math:`t-1` with returns over
        :math:`(t-1, t]`. Meaningful only when `exposures` is populated; ignored by
        time-series factor models, where the loading matrix is constant.

    regression_weights : ndarray of shape (n_observations, n_assets) or None
        Cross-sectional WLS regression weights. Non-negative. Assets with zero weight
        are excluded from the estimation universe. Row :math:`t` holds the weights
        used by the regression at date :math:`t`; like the lagged exposures, they are
        built from market caps at :math:`t - \text{lag}` and idiosyncratic variances
        estimated up to :math:`t - 1`. `None` for time-series factor models.

    benchmark_weights : ndarray of shape (n_observations, n_assets) or None
        Benchmark weights used for weighted cross-sectional diagnostics. Non-negative.
        `None` for time-series factor models.

    family_constraint_basis : FamilyConstraintBasis or None
        Compact basis encoding the family-constraint change of coordinates. Used by
        cross-sectional factor models with linear constraints across factor families
        (e.g. industry sum-to-zero). When present, diagnostics (t-statistics, VIF,
        condition number) and adjusted :math:`R^2` are computed in the reduced basis
        where constrained families are full-rank.
    """

    observations: StrArray  # (n_observations,)
    asset_names: StrArray  # (n_assets,)
    factor_names: StrArray  # (n_factors,)
    factor_families: StrArray | None  # (n_factors,)

    loading_matrix: FloatArray  # (n_assets, n_factors)
    exposures: FloatArray | None  # (n_observations, n_assets, n_factors)

    # Factors
    factor_covariance: FloatArray  # (n_factors, n_factors)
    factor_mu: FloatArray  # (n_factors,)
    factor_returns: FloatArray | None  # (n_observations, n_factors)

    # Idio
    idio_covariance: FloatArray  # (n_assets, n_assets) or (n_assets,)
    idio_mu: FloatArray | None  # (n_assets,)
    idio_returns: FloatArray | None  # (n_observations, n_assets)
    idio_variances: FloatArray | None  # (n_observations, n_assets)

    exposure_lag: int = 1
    regression_weights: FloatArray | None = None  # (n_observations, n_assets)
    benchmark_weights: FloatArray | None = None  # (n_observations, n_assets)
    family_constraint_basis: FamilyConstraintBasis | None = None

    def __post_init__(self) -> None:
        """Validate optional weight arrays."""
        self._validate_weights(self.regression_weights, name="regression_weights")
        self._validate_weights(self.benchmark_weights, name="benchmark_weights")

    # General utilities
    def summary(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        annualization_factor: float = 252.0,
        stability_step: int = 21,
        stability_cs_weighting: CSWeighting = CSWeighting.BENCHMARK,
        t_stat_threshold: float = 2.0,
    ) -> pd.DataFrame:
        r"""Summary statistics for the factor model.

        Combines factor-return statistics, Gram-matrix diagnostics, and exposure-quality
        metrics:

            * `annualized_mean`: factor annualized mean return.
            * `annualized_vol`: factor annualized volatility.
            * `annualized_sharpe`: factor annualized Sharpe ratio.
            * `autocorrelation`: factor return lag-1 autocorrelation.
            * `mean_abs_t_stat`: factor mean absolute cross-sectional t-statistic.
            * `t_stat_exceedance_rate`: fraction of observations where :math:`|t| > \text{threshold}`.
            * `mean_vif`: factor mean Variance Inflation Factor.
            * `stability`: factor median exposure stability coefficient over the chosen step.
            * `coverage`: average fraction of estimation-universe assets (positive
              regression weight) with non-missing factor exposure.

        For characteristics-based models, `annualized_mean` and `annualized_vol` are
        computed from model-native factor returns: the cross-sectional regression
        coefficients per one unit of exposure. Equivalently, each factor return is the
        WLS factor-mimicking portfolio return with unit exposure to that factor and zero
        exposure to the other regression factors, without additional rescaling to fixed
        gross exposure or volatility. The sign follows the exposure convention; for
        example, a size factor built from log market capitalization is large-minus-small,
        the opposite sign of the Fama-French SMB convention.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, optional
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        annualization_factor : float, default=252.0
            Number of observations per year (e.g., 252 for daily data) to annualize
            mean, volatility and sharpe ratio.

        stability_step : int, default=21
            Number of observations between the two cross-sections used for the exposure
            stability coefficient (e.g. 21 for approximately monthly stability with
            daily data).

        stability_cs_weighting : CSWeighting, default=CSWeighting.BENCHMARK
            Cross-sectional weights for the stability computation. Falls back
            to `CSWeighting.IDENTITY` with a warning when unavailable.

        t_stat_threshold : float, default=2.0
            Absolute t-statistic threshold for the exceedance rate.
        """
        self._require("factor_returns", "summary")
        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        factor_returns = self.factor_returns[:, factor_indices]
        n_observations = factor_returns.shape[0]
        n_selected = len(factor_names)

        # Factor-return statistics
        mean = np.nanmean(factor_returns, axis=0) * annualization_factor
        vol = np.nanstd(factor_returns, axis=0, ddof=1) * np.sqrt(annualization_factor)
        sharpe = safe_divide(mean, vol, fill_value=np.nan)
        if n_observations > 1:
            autocorr = _lag1_autocorr(factor_returns)
        else:
            autocorr = np.full(n_selected, np.nan)

        data = {
            "annualized_mean": mean,
            "annualized_vol": vol,
            "annualized_sharpe": sharpe,
            "autocorrelation": autocorr,
        }

        # Gram-matrix diagnostics (reduced-basis aware)
        if (
            self.exposures is None
            or self.factor_returns is None
            or self.idio_returns is None
        ):
            for metric_name in (
                "mean_abs_t_stat",
                "t_stat_exceedance_rate",
                "mean_vif",
            ):
                data[metric_name] = np.full(n_selected, np.nan)
        else:
            diagnostics = self._gram_diagnostics
            t_stat_exceedance_rate = self.cs_regression_t_stat_exceedance_rate(
                threshold=t_stat_threshold
            ).values
            reduced_names = list(self._reduced_regression_factor_names)
            reduced_idx = {name: i for i, name in enumerate(reduced_names)}

            def _map_to_selected(values: FloatArray) -> FloatArray:
                mapped = np.full(n_selected, np.nan)
                for i, name in enumerate(factor_names):
                    j = reduced_idx.get(name)
                    if j is not None:
                        mapped[i] = values[j]
                return mapped

            data["mean_abs_t_stat"] = _map_to_selected(
                np.nanmean(np.abs(diagnostics.t_stats), axis=0)
            )
            data["t_stat_exceedance_rate"] = _map_to_selected(t_stat_exceedance_rate)
            data["mean_vif"] = _map_to_selected(np.nanmean(diagnostics.vif, axis=0))

        # Exposure diagnostics
        if self.exposures is None:
            for metric_name in ("stability", "coverage"):
                data[metric_name] = np.full(n_selected, np.nan)
        else:
            if self.regression_weights is not None:
                in_universe = self.regression_weights > 0
            else:
                in_universe = np.ones(self.exposures.shape[:2], dtype=bool)

            n_eligible = in_universe.sum(axis=1, keepdims=True)
            n_covered = (np.isfinite(self.exposures) & in_universe[..., None]).sum(
                axis=1
            )
            data["coverage"] = np.nanmean(
                safe_divide(n_covered, n_eligible, fill_value=0.0), axis=0
            )[factor_indices]

            exposures = self.exposures[:, :, factor_indices]
            cs_var = np.nanvar(exposures, axis=1)
            is_constant = np.nanmax(cs_var, axis=0) < 1e-12
            if n_observations > stability_step:
                stability_ts = self._exposure_stability(
                    exposures, step=stability_step, cs_weighting=stability_cs_weighting
                )
                stability_ts[:, is_constant] = 1.0
                stability = np.nanmedian(stability_ts, axis=0)
            else:
                stability = np.where(is_constant, 1.0, np.nan)

            data["stability"] = stability

        return pd.DataFrame(data, index=factor_names)

    @property
    def factor_returns_df(self) -> pd.DataFrame:
        """Factor returns DataFrame of shape (n_observations, n_factors)."""
        self._require("factor_returns", "factor_returns_df")
        return pd.DataFrame(
            self.factor_returns, index=self.observations, columns=self.factor_names
        )

    @property
    def idio_returns_df(self) -> pd.DataFrame:
        """Idiosyncratic returns DataFrame of shape (n_observations, n_assets)."""
        self._require("idio_returns", "idio_returns_df")
        return pd.DataFrame(
            self.idio_returns, index=self.observations, columns=self.asset_names
        )

    @property
    def exposures_df(self) -> pd.DataFrame:
        """Exposures as a MultiIndex DataFrame of shape
        (n_observations, n_factors * n_assets).
        """
        self._require("exposures", "exposures_df")
        cols = pd.MultiIndex.from_product(
            (self.factor_names, self.asset_names), names=["factor", "asset"]
        )
        exposures = self.exposures.transpose(0, 2, 1).reshape(
            len(self.observations), -1
        )
        return pd.DataFrame(exposures, index=self.observations, columns=cols)

    def factor_forecast_correlation(
        self, factors: list[str] | None = None, families: str | list[str] | None = None
    ) -> FloatArray:
        """Factor return correlation forecast from :attr:`factor_covariance`.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over
            `families` when specified.

        families : str, list of str, optional
            Factor families to include. `None` includes all factors.
            Ignored when `factors` is given or when
            `factor_families` is `None`.

        Returns
        -------
        corr : ndarray of shape (n_selected_factors, n_selected_factors)
            Symmetric factor return correlation matrix with diagonal entries
            fixed to 1.
        """
        factor_indices, _ = self._resolve_factor_subset(factors, families)
        if factor_indices == slice(None):
            cov = self.factor_covariance
        else:
            cov = self.factor_covariance[np.ix_(factor_indices, factor_indices)]
        corr, _ = cov_to_corr(cov)
        return corr

    @property
    def effective_loading_matrix(self) -> FloatArray:
        r"""Full-rank loading matrix, reduced when family constraints are present.

        When the factor model uses family constraints, the full-basis loading matrix is
        rank-deficient because constrained factor families introduce linear dependencies
        among columns. This property converts it to the reduced (full-rank) basis so
        that downstream computations (e.g. orthogonal projectors) correctly identify the
        factor span.

        When :attr:`family_constraint_basis` is `None`, the loading matrix is returned
        unchanged.

        Returns
        -------
        loading : ndarray of shape (n_assets, n_reduced_factors)
            Full-rank loading matrix.
        """
        if self.family_constraint_basis is None:
            return self.loading_matrix
        return self.family_constraint_basis.reduce_loading_matrix(self.loading_matrix)

    @property
    def effective_exposures(self) -> FloatArray:
        r"""Full-rank historical exposures, reduced when family constraints are present.

        When the factor model uses family constraints, the full-basis exposure tensor is
        rank-deficient because constrained factor families introduce linear dependencies
        among columns. This property converts the historical exposures to the same
        reduced full-rank basis as :attr:`effective_loading_matrix`.

        When :attr:`family_constraint_basis` is `None`, the historical exposures are
        returned unchanged.

        Returns
        -------
        exposures : ndarray of shape (n_observations, n_assets, n_reduced_factors)
            Historical full-rank exposure tensor.
        """
        self._require("exposures", "effective_exposures")
        if self.family_constraint_basis is None:
            return self.exposures
        return self.family_constraint_basis.reduce_exposures(self.exposures)

    @property
    def effective_factor_names(self) -> StrArray:
        """Factor names aligned with the effective reduced basis."""
        if self.family_constraint_basis is None:
            return self.factor_names
        return self.family_constraint_basis.reduced_factor_names(self.factor_names)

    @property
    def effective_factor_families(self) -> StrArray | None:
        """Factor families aligned with the effective reduced basis."""
        if self.factor_families is None:
            return None
        if self.family_constraint_basis is None:
            return self.factor_families
        return self.family_constraint_basis.reduced_factor_names(self.factor_families)

    @property
    def effective_factor_covariance(self) -> FloatArray:
        r"""Full-rank factor covariance, reduced when family constraints are present.

        When the factor model uses family constraints, the full-basis factor covariance
        :math:`R\,\Sigma_f^{\mathrm{red}}\,R^\top` is rank-deficient. This property
        returns the reduced full-rank covariance aligned with
        :attr:`effective_loading_matrix`, so that decompositions (e.g. Cholesky) and
        SOC-based optimizers operate on a positive definite matrix.

        When :attr:`family_constraint_basis` is `None`, the covariance is returned
        unchanged.

        Returns
        -------
        factor_covariance : ndarray of shape (n_reduced_factors, n_reduced_factors)
            Full-rank factor covariance.
        """
        if self.family_constraint_basis is None:
            return self.factor_covariance
        return self.family_constraint_basis.reduce_factor_covariance(
            self.factor_covariance
        )

    @cached_property
    def covariance_sqrt(self) -> CovarianceSqrt:
        r"""Covariance square root exploiting the factor structure.

        Decomposes the asset covariance :math:`\Sigma = B\,\Sigma_f\,B^\top + D` into a
        :class:`~skfolio.prior.CovarianceSqrt` that separates the systematic and
        idiosyncratic contributions, allowing SOC-based optimizers to work with smaller
        matrices.

        When idiosyncratic covariance is diagonal, the decomposition avoids an
        :math:`(n \times n)` Cholesky entirely and represents the idiosyncratic part as
        an element-wise multiply.

        When family constraints are present, the full-basis factor covariance
        :math:`R\,\Sigma_f^{\mathrm{red}}\,R^\top` is rank-deficient. The systematic
        square root is then built from the full-rank :attr:`effective_loading_matrix`
        and :attr:`effective_factor_covariance`, which keeps the Cholesky exact and the
        systematic component minimal.

        Returns
        -------
        CovarianceSqrt
        """
        systematic = self.effective_loading_matrix @ safe_cholesky(
            self.effective_factor_covariance
        )

        if self.idio_covariance.ndim == 1:
            return CovarianceSqrt(
                components=(systematic,),
                diagonal=np.sqrt(self.idio_covariance),
            )
        return CovarianceSqrt(
            components=(systematic, safe_cholesky(self.idio_covariance)),
        )

    def enrich_asset_panel(
        self, panel: AssetPanel | AssetPanelView, copy: bool = True
    ) -> AssetPanel | AssetPanelView:
        """Add factor-model fields to an :class:`~skfolio.containers.AssetPanel`.

        The returned panel contains the fields required by alpha estimators:
        `idio_returns`, `idio_variances`, `regression_weights` and `exposures`.
        Observations and assets are aligned by label. Panel observations that are not
        present in the factor model are kept and filled with missing values, except
        `regression_weights`, which is filled with zero. The asset set must match
        exactly, although the order may differ. If `panel` is an
        :class:`~skfolio.containers.AssetPanelView`, enriched fields are added as
        view-local fields.

        When family constraints are present, `exposures` are added in the reduced
        full-rank basis used by the cross-sectional regression and factor covariance
        estimator.

        Parameters
        ----------
        panel : AssetPanel or AssetPanelView
            Panel or observation view to enrich.

        copy : bool, default=True
            If `True`, enrich a shallow copy of `panel`. If `False`, mutate `panel`.

        Returns
        -------
        enriched_panel : AssetPanel or AssetPanelView
            Panel or view containing the factor-model fields.

        Raises
        ------
        TypeError
            If `panel` is not an :class:`~skfolio.containers.AssetPanel` or
            :class:`~skfolio.containers.AssetPanelView`.

        ValueError
            If required factor-model histories are unavailable, if labels cannot be
            aligned, or if any target field already exists.
        """
        if not isinstance(panel, (AssetPanel, AssetPanelView)):
            raise TypeError(
                "`panel` must be an AssetPanel or AssetPanelView, "
                f"got {type(panel).__name__!r}."
            )

        field_names = {
            _IDIO_RETURNS,
            _IDIO_VARIANCES,
            _REGRESSION_WEIGHTS,
            _EXPOSURES,
        }
        existing = field_names.intersection(panel.fields)
        if existing:
            raise ValueError(
                "Cannot enrich AssetPanel because it already contains "
                f"{sorted(existing)}."
            )

        obs_idx = pd.Index(self.observations).get_indexer(panel.observations)
        valid_obs = obs_idx >= 0
        if not np.any(valid_obs):
            raise ValueError(
                "The FactorModel and AssetPanel observations do not overlap."
            )

        asset_idx = pd.Index(self.asset_names).get_indexer(panel.asset_names)
        if np.any(asset_idx < 0) or len(panel.asset_names) != len(self.asset_names):
            raise ValueError(
                "FactorModel asset names must match AssetPanel asset names exactly."
            )

        self._require(
            ("idio_returns", "idio_variances", "regression_weights", "exposures"),
            "enrich_asset_panel",
        )

        if copy:
            if isinstance(panel, AssetPanelView):
                enriched_panel = panel.copy(deep=False, copy_owner=False)
            else:
                enriched_panel = panel.copy(deep=False)
        else:
            enriched_panel = panel

        def align_2d(values: FloatArray, fill_value: float) -> FloatArray:
            out = np.full(
                (panel.n_observations, panel.n_assets), fill_value, dtype=float
            )
            out[valid_obs] = np.asarray(values, dtype=float)[obs_idx[valid_obs]][
                :, asset_idx
            ]
            return out

        def align_3d(values: FloatArray) -> FloatArray:
            values = np.asarray(values, dtype=float)
            out = np.full(
                (panel.n_observations, panel.n_assets, values.shape[2]),
                np.nan,
                dtype=float,
            )
            out[valid_obs] = values[obs_idx[valid_obs]][:, asset_idx]
            return out

        enriched_panel[_IDIO_RETURNS] = align_2d(self.idio_returns, np.nan)
        enriched_panel[_IDIO_VARIANCES] = align_2d(self.idio_variances, np.nan)
        enriched_panel.add_2d_field(
            name=_REGRESSION_WEIGHTS,
            values=align_2d(self.regression_weights, 0.0),
            inactive_policy=InactivePolicy.ZERO,
        )
        enriched_panel.add_3d_field(
            name=_EXPOSURES,
            values=align_3d(self.effective_exposures),
            third_axis_name="factors",
            third_axis_labels=self.effective_factor_names,
            third_axis_groups=self.effective_factor_families,
        )
        return enriched_panel

    def select_assets(
        self, assets: ArrayLike | slice | None = None, slim: bool = False
    ) -> FactorModel:
        """Return a new `FactorModel` restricted to selected assets.

        Per-asset fields (`asset_names`, `loading_matrix`, `exposures`,
        `idio_covariance`, `idio_mu`, `idio_returns`, `idio_variances`,
        `regression_weights`, `benchmark_weights`) are subsetted along the asset axis.
        Per-factor and time-only fields (`factor_names`, `factor_families`,
        `factor_covariance`, `factor_mu`, `factor_returns`, `observations`) and
        `family_constraint_basis` are passed through by reference. When `assets` keeps
        every asset in order and `slim` is `False`, `self` is returned directly.

        Parameters
        ----------
        assets : array-like, slice , optional
            Assets to keep. Boolean arrays are treated as masks, integer arrays and
            slices are positional selectors and other arrays are matched against
            `asset_names`. The selection must be duplicate-free. If `None`, keep all
            assets.

        slim : bool, default=False
            When `True`, heavy time-series fields not used by downstream portfolio
            optimization (`exposures`, `idio_returns`, `idio_variances`,
            `benchmark_weights`) are set to `None` to save memory.

        Returns
        -------
        subset : FactorModel
        """
        all_assets = assets is None
        if all_assets and not slim:
            return self

        if all_assets:
            positions = None
            asset_indexer = None
        else:
            positions = _selector_to_positions(
                assets, self.asset_names, axis_name="assets"
            )
            if len(np.unique(positions)) != len(positions):
                raise ValueError("`assets` must be a duplicate-free selector.")
            if len(positions) == len(self.asset_names) and np.array_equal(
                positions, np.arange(len(self.asset_names))
            ):
                if not slim:
                    return self
                asset_indexer = None
            else:
                asset_indexer = _positions_to_indexer(positions)

        def _subset(arr: AnyArray | None, axis: int = 0) -> AnyArray | None:
            if arr is None:
                return None
            if asset_indexer is None:
                return arr
            indexer = [slice(None)] * arr.ndim
            indexer[axis] = asset_indexer
            return arr[tuple(indexer)]

        idio_cov = self.idio_covariance
        if idio_cov is not None and asset_indexer is not None:
            if idio_cov.ndim == 1:
                idio_cov = idio_cov[asset_indexer]
            else:
                idio_cov = idio_cov[np.ix_(positions, positions)]

        if slim:
            exposures = None
            idio_returns = None
            idio_variances = None
            benchmark_weights = None
        else:
            exposures = _subset(self.exposures, axis=1)
            idio_returns = _subset(self.idio_returns, axis=1)
            idio_variances = _subset(self.idio_variances, axis=1)
            benchmark_weights = _subset(self.benchmark_weights, axis=1)

        return FactorModel(
            observations=self.observations,
            asset_names=_subset(self.asset_names),
            factor_names=self.factor_names,
            factor_families=self.factor_families,
            loading_matrix=_subset(self.loading_matrix),
            exposures=exposures,
            factor_covariance=self.factor_covariance,
            factor_mu=self.factor_mu,
            factor_returns=self.factor_returns,
            idio_covariance=idio_cov,
            idio_mu=_subset(self.idio_mu),
            idio_returns=idio_returns,
            idio_variances=idio_variances,
            exposure_lag=self.exposure_lag,
            regression_weights=_subset(self.regression_weights, axis=1),
            benchmark_weights=benchmark_weights,
            family_constraint_basis=self.family_constraint_basis,
        )

    def select_observations(self, observations: ArrayLike | slice) -> FactorModel:
        r"""Return a new `FactorModel` restricted to selected observations.

        Slices all time-varying fields (`factor_returns`, `exposures`, `idio_returns`,
        `idio_variances`, `regression_weights`, `benchmark_weights`) to match
        `observations` while passing through all static fields (`loading_matrix`,
        `factor_covariance`, `idio_covariance`, `factor_mu`, `idio_mu`) unchanged.

        When the target observations map to a contiguous range inside the model's
        observation axis, numpy views are used to avoid copies.

        .. note::

            Static fields are shared by reference.  In particular, `loading_matrix`
            is **not** updated to `exposures[-1]` of the sliced model. It retains the
            value set by the estimator that produced this `FactorModel`.

        Parameters
        ----------
        observations : array-like or slice
            Observations to keep. Boolean arrays are treated as masks, integer arrays
            and slices are positional selectors, and other arrays are matched against
            `self.observations`. The selection must be duplicate-free and preserve the
            original observation order.

        Returns
        -------
        subset : FactorModel
            A `FactorModel` whose time-varying arrays cover only the requested
            observations. If `observations` already matches `self.observations`, `self`
            is returned directly (zero-cost no-op).

        Raises
        ------
        ValueError
            If any element of `observations` is not found in `self.observations`, or if
            the requested labels are repeated or not in increasing order relative to
            `self.observations`.
        """
        indices = _selector_to_positions(
            observations, self.observations, axis_name="observations"
        )
        if len(indices) > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError(
                "`observations` must be a duplicate-free subset of "
                "`self.observations` in the same relative order."
            )

        if len(indices) == len(self.observations) and np.array_equal(
            indices, np.arange(len(self.observations))
        ):
            return self

        observation_indexer = _positions_to_indexer(indices)

        def _slice(arr: AnyArray | None) -> AnyArray | None:
            return arr[observation_indexer] if arr is not None else None

        return FactorModel(
            observations=_slice(self.observations),
            asset_names=self.asset_names,
            factor_names=self.factor_names,
            factor_families=self.factor_families,
            loading_matrix=self.loading_matrix,
            exposures=_slice(self.exposures),
            factor_covariance=self.factor_covariance,
            factor_mu=self.factor_mu,
            factor_returns=_slice(self.factor_returns),
            idio_covariance=self.idio_covariance,
            idio_mu=self.idio_mu,
            idio_returns=_slice(self.idio_returns),
            idio_variances=_slice(self.idio_variances),
            exposure_lag=self.exposure_lag,
            regression_weights=_slice(self.regression_weights),
            benchmark_weights=_slice(self.benchmark_weights),
            family_constraint_basis=_slice(self.family_constraint_basis),
        )

    def plot_factor_forecast_correlation(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Factor return correlation forecast heatmap from :attr:`factor_covariance`.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        corr = self.factor_forecast_correlation(factors=factors, families=families)
        fig = _heatmap(
            corr,
            labels=format_plot_labels(factor_names),
            title=title or "Factor Forecast Correlation",
            zmin=-1,
            zmax=1,
        )
        _add_family_outlines(fig, self.factor_families, factor_indices)
        return fig

    def plot_factor_forecast_volatilities(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        annualization_factor: float = 252.0,
        title: str | None = None,
    ) -> go.Figure:
        r"""Bar chart of annualized factor volatility forecasts.

        Computes annualized volatility as
        :math:`\sqrt{\mathrm{diag}(\Sigma_F) \cdot \text{annualization\_factor}}`
        from :attr:`factor_covariance`. Distinct from realized historical volatility
        in :meth:`summary`.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        annualization_factor : float, default=252.0
            Number of observations per year.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        factor_indices, factor_names = self._resolve_factor_subset(factors, families)

        factor_vols = np.sqrt(
            np.diag(self.factor_covariance)[factor_indices] * annualization_factor
        )
        sort_order = np.argsort(factor_vols)
        factor_vols = factor_vols[sort_order]
        sorted_factor_names = format_plot_labels(
            [str(factor_names[index]) for index in sort_order]
        )

        fig = go.Figure(
            go.Bar(
                x=factor_vols,
                y=sorted_factor_names,
                orientation="h",
                marker_color="rgb(31, 119, 180)",
            )
        )
        fig.update_xaxes(tickformat=".2%")
        fig.update_layout(
            title=title or "Factor Forecast Volatility",
            xaxis_title="Annualized Volatility",
            yaxis_title="Factor",
        )
        return fig

    def plot_factor_cumulative_returns(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        title: str | None = None,
    ) -> go.Figure:
        r"""Cumulative (non-compounded) factor returns over time.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        self._require("factor_returns", "plot_factor_cumulative_returns")

        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        cum_ret = np.nancumsum(self.factor_returns[:, factor_indices], axis=0)
        df = pd.DataFrame(cum_ret, index=self.observations, columns=factor_names)

        fig = _multi_line_plot(
            df,
            title=title or "Factor Cumulative Returns (Non-Compounded)",
            yaxis_title="Cumulative Return",
        )
        fig.update_yaxes(tickformat=".2%")
        return fig

    # Private helpers
    @cached_property
    def _regression_data(self) -> _RegressionData:
        """Lag-aligned data for cross-sectional regression diagnostics."""
        self._require(
            ("exposures", "factor_returns", "idio_returns"), "_regression_data"
        )
        (
            lagged_exposures,
            factor_returns,
            idio_returns,
            regression_weights,
        ) = self._aligned(
            ["exposures", "factor_returns", "idio_returns", "regression_weights"]
        )

        family_basis = self.family_constraint_basis
        if family_basis is not None:
            exposure_basis = (
                family_basis[: -self.exposure_lag]
                if self.exposure_lag > 0
                else family_basis
            )
            lagged_exposures = exposure_basis.reduce_exposures(lagged_exposures)
            factor_returns = family_basis.reduce_factor_returns(factor_returns)
            factor_names = self._reduced_factor_names
            regression_factor_mask = ~self._reduced_factor_is_currency
        else:
            factor_names = self.factor_names
            regression_factor_mask = ~self._factor_is_currency

        if not np.all(regression_factor_mask):
            lagged_exposures = lagged_exposures[:, :, regression_factor_mask]
            factor_returns = factor_returns[:, regression_factor_mask]
            factor_names = factor_names[regression_factor_mask]

        if len(factor_names) == 0:
            raise ValueError(
                "No cross-sectional regression factors are available. Currency factors "
                "are direct factors and do not have regression diagnostics."
            )

        return _RegressionData(
            exposures=lagged_exposures,
            factor_returns=factor_returns,
            idio_returns=idio_returns,
            regression_weights=regression_weights,
            factor_names=factor_names,
        )

    @cached_property
    def _gram_diagnostics(self) -> _GramDiagnostics:
        r"""Per-observation t-statistics, VIF, and condition number.

        When a :attr:`family_constraint_basis` is present, the design matrix
        and factor returns are projected into the reduced (full-rank)
        basis before building the Gram matrix.  This avoids the
        rank-deficient :math:`X^\top W X` that arises from collinear
        constrained families (e.g. industry dummies).

        The three outputs share a single weighted least-squares pass and are
        memoised together so any downstream property pays the cost only once.
        """
        self._require(
            ("exposures", "factor_returns", "idio_returns"), "_gram_diagnostics"
        )
        regression_data = self._regression_data
        lagged_exposures = regression_data.exposures
        factor_returns = regression_data.factor_returns
        idio_returns = regression_data.idio_returns
        regression_weights = regression_data.regression_weights

        n_observations, _, n_factors = lagged_exposures.shape

        finite_mask = np.isfinite(idio_returns) & np.all(
            np.isfinite(lagged_exposures), axis=2
        )
        estimation_mask, estimation_weights = self._estimation_mask_and_weights(
            finite_mask, regression_weights
        )

        n_valid = estimation_mask.sum(axis=1)
        idio_returns = np.where(estimation_mask, idio_returns, 0.0)
        rss = (estimation_weights * idio_returns**2).sum(axis=1)

        weighted_exposures = np.where(estimation_mask[..., None], lagged_exposures, 0.0)
        weighted_exposures *= np.sqrt(estimation_weights)[..., None]
        gram_matrices = weighted_exposures.transpose(0, 2, 1) @ weighted_exposures

        degrees_of_freedom = (n_valid - n_factors).astype(float)
        valid_design = degrees_of_freedom > 0
        valid_t_stats = valid_design & np.all(np.isfinite(factor_returns), axis=1)

        t_stats = np.full((n_observations, n_factors), np.nan)
        vif = np.full((n_observations, n_factors), np.nan)
        condition_numbers = np.full(n_observations, np.nan)

        if np.any(valid_design):
            gram_matrices = gram_matrices[valid_design]
            identity = np.broadcast_to(np.eye(n_factors), gram_matrices.shape).copy()
            try:
                gram_inverse = np.linalg.solve(gram_matrices, identity)
            except np.linalg.LinAlgError:
                gram_inverse = np.linalg.pinv(gram_matrices)

            gram_diagonal = np.einsum("tii->ti", gram_matrices, optimize=True)
            gram_inverse_diagonal = np.einsum("tii->ti", gram_inverse, optimize=True)
            vif[valid_design] = gram_diagonal * gram_inverse_diagonal
            condition_numbers[valid_design] = np.linalg.cond(gram_matrices)

            if np.any(valid_t_stats):
                t_stats_mask = valid_t_stats[valid_design]
                residual_variance = (
                    rss[valid_t_stats] / degrees_of_freedom[valid_t_stats]
                )
                standard_error = np.sqrt(
                    np.maximum(
                        residual_variance[:, None]
                        * gram_inverse_diagonal[t_stats_mask],
                        0.0,
                    )
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    t_stats[valid_t_stats] = np.where(
                        standard_error > 0,
                        factor_returns[valid_t_stats] / standard_error,
                        np.nan,
                    )

        return _GramDiagnostics(
            t_stats=t_stats, vif=vif, condition_number=condition_numbers
        )

    @property
    def _reduced_factor_names(self) -> StrArray:
        """Factor names aligned with reduced-basis computations.

        Returns the reduced-basis names when a family-constraint basis is present,
        otherwise the full factor names.
        """
        family_basis = self.family_constraint_basis
        if family_basis is not None:
            return family_basis.reduced_factor_names(self.factor_names)
        return self.factor_names

    @property
    def _factor_is_currency(self) -> BoolArray:
        """Mask of direct currency factors in the full factor basis."""
        if self.factor_families is None:
            return np.zeros(len(self.factor_names), dtype=bool)
        return self.factor_families == _CURRENCY

    @property
    def _reduced_factor_families(self) -> StrArray | None:
        """Factor families aligned with reduced-basis computations."""
        if self.factor_families is None:
            return None
        family_basis = self.family_constraint_basis
        if family_basis is not None:
            return family_basis.reduced_factor_names(self.factor_families)
        return self.factor_families

    @property
    def _reduced_factor_is_currency(self) -> BoolArray:
        """Mask of direct currency factors in the reduced factor basis."""
        factor_families = self._reduced_factor_families
        if factor_families is None:
            return np.zeros(len(self._reduced_factor_names), dtype=bool)
        return factor_families == _CURRENCY

    @property
    def _reduced_regression_factor_names(self) -> StrArray:
        """Reduced-basis factor names estimated by cross-sectional regression."""
        return self._reduced_factor_names[~self._reduced_factor_is_currency]

    @property
    def _n_regressors(self) -> int:
        """Effective number of independent regressors."""
        return len(self._reduced_regression_factor_names)

    def _resolve_cs_weighting(
        self,
        cs_weighting: CSWeighting,
        *,
        latest: bool,
        fallback_cs_weighting: CSWeighting | None = None,
    ) -> FloatArray | None:
        """Return cross-sectional weights, or `None` for equal weights."""
        if not isinstance(cs_weighting, CSWeighting):
            raise TypeError("`cs_weighting` must be a `CSWeighting`.")
        if fallback_cs_weighting is not None and not isinstance(
            fallback_cs_weighting, CSWeighting
        ):
            raise TypeError("`fallback_cs_weighting` must be a `CSWeighting`.")
        if fallback_cs_weighting == cs_weighting:
            fallback_cs_weighting = None

        if cs_weighting == CSWeighting.IDENTITY:
            return None

        if cs_weighting == CSWeighting.BENCHMARK:
            weights = self.benchmark_weights
            name = "benchmark_weights"
        elif cs_weighting == CSWeighting.REGRESSION:
            weights = self.regression_weights
            name = "regression_weights"
        else:
            if latest:
                if self.idio_covariance.ndim == 1:
                    idio_variance = self.idio_covariance
                else:
                    idio_variance = np.diag(self.idio_covariance)
            else:
                idio_variance = self.idio_variances
                if idio_variance is None:
                    raise ValueError(
                        "`cs_weighting=CSWeighting.INVERSE_IDIO_VARIANCE` with "
                        "`latest=False` requires `idio_variances`."
                    )
            if np.any(idio_variance <= 0):
                raise ValueError(
                    "Idiosyncratic variances must be positive for "
                    "`cs_weighting=CSWeighting.INVERSE_IDIO_VARIANCE`."
                )
            weights = 1.0 / idio_variance
            name = "idio_covariance" if latest else "idio_variances"

        if weights is None:
            if fallback_cs_weighting is not None:
                warnings.warn(
                    f"`cs_weighting=CSWeighting.{cs_weighting.name}` requires "
                    f"`{name}`, which is not available. Falling back to "
                    f"`CSWeighting.{fallback_cs_weighting.name}`.",
                    stacklevel=2,
                )
                return self._resolve_cs_weighting(fallback_cs_weighting, latest=latest)
            raise ValueError(
                f"`cs_weighting=CSWeighting.{cs_weighting.name}` requires `{name}`."
            )

        weights = np.asarray(weights, dtype=float)
        if latest and weights.ndim == 2:
            weights = weights[-1]
        elif not latest and weights.ndim == 1:
            weights = np.broadcast_to(
                weights, (len(self.observations), len(self.asset_names))
            )

        return weights

    def _ic(
        self,
        correlation_method: CorrelationMethod = CorrelationMethod.SPEARMAN,
        horizon: int = 1,
        factor_indices: slice | list[int] = slice(None),
        reduced_basis: bool = False,
    ) -> tuple[FloatArray, int]:
        r"""Predictive Information Coefficient per factor over time.

        Computes the cross-sectional correlation between exposures at :math:`t` and the
        cumulative asset return from :math:`t + 1` to :math:`t + h`, where :math:`h` is
        the forecast *horizon*.

        Parameters
        ----------
        correlation_method : CorrelationMethod, default=CorrelationMethod.SPEARMAN
            Correlation method used for the exposure IC. `SPEARMAN` computes
            Spearman rank IC. `PEARSON` computes Pearson IC, weighted by
            `regression_weights` when available.

        horizon : int, default=1
            Forward window in number of observations. The cumulative return from
            :math:`t + 1` to :math:`t + h` is used.

        factor_indices : slice or list of int, default=slice(None)
            Factor columns to evaluate. Use `slice(None)` to compute IC for all factors
            without copying the exposure tensor.

        reduced_basis : bool, default=False
            If `True`, compute ICs on reduced-basis exposures when a
            `family_constraint_basis` is present.

        Returns
        -------
        ic : ndarray of shape (n_pairs, n_factors)
        obs_offset : int
            Starting index into `self.observations` for the IC values.
            `ic[i]` corresponds to `observations[obs_offset + i]`.
        """
        _check_correlation_method(correlation_method)
        self._require(("exposures", "factor_returns", "idio_returns"), "_ic")

        if horizon < 1:
            raise ValueError("`horizon` must be >= 1.")

        lagged_exposures, factor_returns, idio_returns, _ = self._aligned(
            ["exposures", "factor_returns", "idio_returns", "regression_weights"]
        )
        systematic_returns = (
            lagged_exposures @ factor_returns[:, :, np.newaxis]
        ).squeeze(-1)
        asset_returns = systematic_returns + idio_returns
        n_observations = self.exposures.shape[0]

        start_t = max(0, self.exposure_lag - 1)
        n_pairs = n_observations - horizon - start_t
        if n_pairs < 1:
            raise ValueError(
                f"Not enough observations ({n_observations}) for horizon={horizon} "
                f"with exposure_lag={self.exposure_lag}."
            )

        forward_returns = _forward_mean_return(
            asset_returns, horizon=horizon, lag=max(1 - self.exposure_lag, 0)
        )[:n_pairs]

        exposures = self.exposures
        if reduced_basis and self.family_constraint_basis is not None:
            exposures = self.family_constraint_basis.reduce_exposures(exposures)
        exposure_window = exposures[start_t : start_t + n_pairs, :, factor_indices]

        if correlation_method is CorrelationMethod.SPEARMAN:
            ic = cs_spearman_correlation(
                forward_returns[:, :, np.newaxis],
                exposure_window,
                axis=1,
            )
        else:
            regression_weights = self.regression_weights
            if regression_weights is not None:
                regression_weights = regression_weights[start_t : start_t + n_pairs]
            ic = cs_pearson_correlation(
                forward_returns,
                exposure_window,
                weights=regression_weights,
                axis=1,
            )
        return ic, start_t

    def _exposure_stability(
        self,
        exposures: FloatArray,
        step: int = 21,
        cs_weighting: CSWeighting = CSWeighting.BENCHMARK,
    ) -> FloatArray:
        r"""Weighted cross-sectional correlation of exposures between observation
        :math:`t` and :math:`t + \text{step}`.

        Parameters
        ----------
        exposures : ndarray of shape (n_observations, n_assets, n_selected_factors)
            Subset of exposures already sliced to the desired factors.

        step : int, default=21
            Number of observations between the two cross-sections.

        cs_weighting : CSWeighting, default=CSWeighting.BENCHMARK
            Cross-sectional weights passed to :func:`cs_pearson_correlation`.
            Falls back to `CSWeighting.IDENTITY` with a warning when unavailable.

        Returns
        -------
        stability : ndarray of shape (n_observations - step, n_selected_factors)
        """
        weights = self._resolve_cs_weighting(
            cs_weighting, latest=False, fallback_cs_weighting=CSWeighting.IDENTITY
        )
        if weights is not None:
            weights = weights[:-step]
        return cs_pearson_correlation(
            exposures[:-step], exposures[step:], weights=weights, axis=1
        )

    def _attribution_inputs(
        self, name: str, compute_uncertainty: bool
    ) -> tuple[FloatArray | None, FloatArray | None]:
        """Validate inputs for realized/rolling attribution."""
        self._require(("factor_returns", "exposures", "idio_returns"), name)
        if not compute_uncertainty:
            return None, None
        if self.regression_weights is None or self.idio_variances is None:
            raise ValueError(
                "`compute_uncertainty=True` requires both "
                "`regression_weights` and `idio_variances` to be "
                "available in the factor model."
            )
        return self.regression_weights, self.idio_variances

    def _require(self, fields: str | list[str] | tuple[str, ...], name: str) -> None:
        """Validate that one or more `FactorModel` attributes are populated."""
        if isinstance(fields, str):
            fields = (fields,)
        missing = [f for f in fields if getattr(self, f) is None]
        if not missing:
            return
        joined = ", ".join(f"`{f}`" for f in missing)
        raise ValueError(
            f"`{name}` requires {joined} which is not available in this "
            f"FactorModel. The prior estimator used to fit this model does "
            f"not populate {joined}. Check that your estimator supports "
            f"these attributes."
        )

    def _aligned(
        self, fields: str | list[str]
    ) -> AnyArray | list[AnyArray | None] | None:
        r"""Apply `exposure_lag` to one or several time-indexed fields.

        The `exposures` field is the predictor side of the cross-sectional regression
        and is trimmed at the tail. Return-like fields are trimmed at the head so that
        predetermined exposures :math:`B_{t-\ell}` align row-wise with returns at
        :math:`t`.

        With `exposure_lag = 0` the underlying array is returned unchanged. Returns
        `None` when the requested attribute is `None`. When a list of field names is
        provided, the result is a tuple matching the requested order.
        """
        lag = self.exposure_lag

        def align(field: str) -> AnyArray | None:
            arr = getattr(self, field)
            if arr is None:
                return None
            if lag == 0:
                return arr
            return arr[:-lag] if field == "exposures" else arr[lag:]

        if isinstance(fields, str):
            return align(fields)
        return tuple(align(field) for field in fields)

    @staticmethod
    def _estimation_mask_and_weights(
        finite_mask: BoolArray, regression_weights: FloatArray | None
    ) -> tuple[BoolArray, FloatArray]:
        """Combine a finite-data mask with optional regression weights.

        When `regression_weights` is `None`, every finite observation gets unit weight.
        Otherwise, observations are kept only where the weight is strictly positive and
        zeroed out elsewhere.
        """
        if regression_weights is None:
            return finite_mask, finite_mask.astype(float)
        estimation_mask = finite_mask & (regression_weights > 0)
        estimation_weights = np.where(estimation_mask, regression_weights, 0.0)
        return estimation_mask, estimation_weights

    def _resolve_factor_subset(
        self,
        factor_names_to_keep: list[str] | None,
        family_names_to_keep: str | list[str] | None,
        reduced_basis: bool = False,
        regression_only: bool = False,
    ) -> tuple[slice | list[int], StrArray | list[str]]:
        """Resolve a factor subset from explicit names or family labels.

        Parameters
        ----------
        factor_names_to_keep : list of str or None
            Explicit factor names to keep.

        family_names_to_keep : str, list of str, or None
            Family labels to keep. Ignored when `factor_names_to_keep` is given.

        reduced_basis : bool, default=False
            Use True for diagnostics computed in the reduced basis.

        regression_only : bool, default=False
            Restrict available factors to factors estimated by cross-sectional
            regression, excluding direct currency factors.

        Returns
        -------
        indices : slice or list of int
            Column selector into *factor_names*. The unfiltered case returns
            `slice(None)` so callers can index arrays without copying.

        names : list of str
            Selected factor names in the same order as `indices`.
        """
        factor_names = (
            self._reduced_factor_names if reduced_basis else self.factor_names
        )
        factor_families = (
            self._reduced_factor_families if reduced_basis else self.factor_families
        )
        if regression_only:
            currency_mask = (
                self._reduced_factor_is_currency
                if reduced_basis
                else self._factor_is_currency
            )
            factor_names = factor_names[~currency_mask]
            if factor_families is not None:
                factor_families = factor_families[~currency_mask]

        return _resolve_factor_subset(
            factor_names=factor_names,
            factor_families=factor_families,
            factor_names_to_keep=factor_names_to_keep,
            family_names_to_keep=family_names_to_keep,
        )

    def _standardized_idio_returns(self) -> FloatArray:
        r"""Compute :math:`z_{it} = \epsilon_{it} / \hat\sigma_{i,t}`."""
        self._require(("idio_returns", "idio_variances"), "standardized_idio_returns")
        idio_vol = np.sqrt(np.maximum(self.idio_variances, 0.0))
        return safe_divide(self.idio_returns, idio_vol, fill_value=np.nan)

    def _validate_weights(self, weights: FloatArray | None, name: str) -> None:
        """Return validated optional weights."""
        if weights is None:
            return None
        expected_shape = (len(self.observations), len(self.asset_names))
        if weights.shape != expected_shape:
            raise ValueError(
                f"`{name}` must have shape {expected_shape}, got {weights.shape}."
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError(f"`{name}` must contain only finite values.")
        if np.any(weights < 0):
            raise ValueError(f"`{name}` must be non-negative.")
