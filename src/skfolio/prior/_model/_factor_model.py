"""Factor Model Dataclass."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.linalg as sc_linalg
from scipy import stats as sp_stats

from skfolio.factor_model._family_constraint_basis import FamilyConstraintBasis
from skfolio.factor_model.attribution import (
    Attribution,
    predicted_factor_attribution,
    realized_factor_attribution,
    rolling_realized_factor_attribution,
)
from skfolio.prior._model._covariance_sqrt import CovarianceSqrt
from skfolio.typing import (
    AnyArray,
    ArrayLike,
    BoolArray,
    FloatArray,
    IntArray,
    StrArray,
)
from skfolio.utils.stats import (
    cov_to_corr,
    cs_rank_correlation,
    cs_weighted_correlation,
    safe_cholesky,
    safe_divide,
)

__all__ = ["FactorModel"]


@dataclass(frozen=True, eq=False)
class FactorModel:
    r"""Factor model decomposition of asset returns.

    Holds the loading matrix, factor moments and idiosyncratic covariance,
    together with the optional time series of exposures, factor returns
    and idiosyncratic returns. Exposes a factor-structured covariance
    square root, the factor-orthogonal residual covariance and basis,
    plus cross-sectional regression diagnostics, idiosyncratic-calibration
    metrics and factor attribution when the relevant fields are populated.

    Produced by factor-model prior estimators:

        * :class:`~skfolio.prior.TimeSeriesFactorModel`,
        * :class:`~skfolio.prior.CharacteristicsFactorModel`

    and consumed downstream via :attr:`~skfolio.prior.ReturnDistribution.factor_model`.

    Method groups
    -------------
    Factor structure and factor-return methods use the stored loading matrix,
    factor moments and, the factor return time series. They are
    available for both time-series and characteristics-based factor models.
    These include `factor_correlation`,
    `plot_factor_correlation`, `plot_factor_volatilities`,
    `plot_factor_cumulative_returns`, and `predicted_attribution`.

    Cross-sectional regression diagnostics are prefixed with `cs_regression_`.
    They require point-in-time exposures, estimated factor returns and
    idiosyncratic returns, as in characteristics-based models. They are not
    available for time-series factor models that only store a static loading
    matrix. These include `cs_regression_scores`, `cs_regression_t_stats`,
    `cs_regression_t_stat_exceedance_rate`, and their plotting methods.

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
        following the as-of convention. Populated for cross-sectional factor models.
        `None` for time-series factor models, which use the single time-invariant
        `loading_matrix`.

    factor_covariance : ndarray of shape (n_factors, n_factors)
        Factor return covariance matrix.

    factor_mu : ndarray of shape (n_factors,)
        Factor expected returns.

    factor_returns : ndarray of shape (n_observations, n_factors) or None
        Per-period factor returns. For time-series factor models, this is the input
        factor return series; for cross-sectional factor models, this is the per-period
        factor returns estimated from the cross-sectional regression.

    idio_covariance : ndarray of shape (n_assets, n_assets) or (n_assets,)
        Idiosyncratic covariance (diagonal vector or full matrix).

    idio_mu : ndarray of shape (n_assets,) or None
        Per-asset expected idiosyncratic return (alpha), constrained to be
        :math:`W`-orthogonal to the factor loadings (:math:`B^\top W \alpha = 0`).
        Distinct from the time-series mean of `idio_returns`, which is not enforced to
        be factor-orthogonal. Populated by cross-sectional factor models.

    idio_returns : ndarray of shape (n_observations, n_assets) or None
        Per-period idiosyncratic returns. For time-series factor models, this is the
        regression residuals :math:`r - B f`; for cross-sectional factor models, this is
        the per-period residuals from the cross-sectional regression.

    idio_variances : ndarray of shape (n_observations, n_assets) or None
        Time-varying per-asset predicted idiosyncratic variances
        :math:`\hat\sigma^2_{i,t}`. Populated by cross-sectional factor models.

    exposure_lag : int, default=1
        Lag applied to time-varying exposures under the as-of convention. The default
        value of `1` aligns exposures known at the end of observation :math:`t-1` with
        returns over :math:`(t-1, t]`, explicitly encoding data availability in the API
        and guarding against look-ahead bias. Meaningful only when `exposures` is
        populated; ignored by time-series factor models, where the loading matrix is
        constant.

    regression_weights : ndarray of shape (n_observations, n_assets) or None
        Cross-sectional WLS regression weights. Non-negative. Assets with zero weight
        are excluded from the estimation universe. `None` for time-series factor models.

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
        stability_weighting: Literal["benchmark", "regression"] | None = "benchmark",
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

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default=None
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        annualization_factor : float, default=252.0
            Number of observations per year (e.g., 252 for daily data) to annualize
            mean, volatility and sharpe ratio.

        stability_step : int, default=21
            Number of observations between the two cross-sections used for the exposure
            stability coefficient (e.g., 21 for approximately monthly stability with
            daily data).

        stability_weighting : "benchmark", "regression", or None, default="benchmark"
            Cross-sectional weights for the stability computation.

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
            reduced_names = list(self._reduced_factor_names)
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
                    exposures, step=stability_step, weighting=stability_weighting
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

    def factor_correlation(
        self, factors: list[str] | None = None, families: str | list[str] | None = None
    ) -> FloatArray:
        """Factor return correlation matrix from the estimated covariance.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over
            `families` when specified.

        families : str, list of str, or None, default=None
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
        loading : ndarray of shape (n_assets, n_factors_reduced)
            Full-rank loading matrix.
        """
        if self.family_constraint_basis is None:
            return self.loading_matrix
        return self.family_constraint_basis.to_reduced_loading_matrix(
            self.loading_matrix
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

        Returns
        -------
        CovarianceSqrt
        """
        systematic = self.loading_matrix @ safe_cholesky(self.factor_covariance)

        if self.idio_covariance.ndim == 1:
            return CovarianceSqrt(
                components=(systematic,),
                diagonal=np.sqrt(self.idio_covariance),
            )
        return CovarianceSqrt(
            components=(systematic, safe_cholesky(self.idio_covariance)),
        )

    @cached_property
    def orthogonal_inflation(self) -> FloatArray:
        r"""Factor-orthogonal covariance inflation matrix.

        Symmetric positive semi-definite matrix used to inflate the asset covariance in
        directions the factor model cannot reach,
        :math:`\Sigma \mapsto \Sigma + \tau\, M`, with closed form

        .. math::

            M \;=\; W^{-1} \;-\; B\,(B^\top W B)^{-1} B^\top
                \;=\; W^{-1/2}\,(I - P)\,W^{-1/2},

        where :math:`B` is the loading matrix, :math:`P` is the orthogonal projector
        onto the whitened loading :math:`W^{1/2} B`, and :math:`W` is the regression
        weighting matrix: the last observation of `regression_weights` when available,
        otherwise the inverse idiosyncratic variance :math:`D^{-1}`.

        For any portfolio :math:`w`, :math:`w^\top M w` is the squared content of
        :math:`w` in the factor-orthogonal subspace: zero on factor-mimicking portfolios
        :math:`w = W B \alpha`, positive otherwise. With the default :math:`W = D^{-1}`,
        :math:`M = D - B(B^\top D^{-1} B)^{-1} B^\top`.

        Returns
        -------
        M : ndarray of shape (n_assets, n_assets)
            Positive semi-definite matrix with rank at most
            :math:`n_{\mathrm{assets}} - n_{\mathrm{factors\_effective}}`.
        """
        weight_inv_sqrt, projector = self._orthogonal_projector
        m = weight_inv_sqrt @ projector @ weight_inv_sqrt
        m = (m + m.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(m)
        eigvals = np.maximum(eigvals, 0.0)
        m = eigvecs @ np.diag(eigvals) @ eigvecs.T
        return m

    @cached_property
    def orthogonal_basis(self) -> FloatArray:
        r"""Orthonormal basis of the orthogonal complement of the factor span.

        Returns a matrix :math:`G \in \mathbb{R}^{n_{\mathrm{assets}} \times r}` whose
        columns form an orthonormal basis (:math:`G^\top G = I`) of the subspace of
        portfolios that are :math:`W`-orthogonal to the factor loadings:

        .. math::

            \{\, w \in \mathbb{R}^{n_{\mathrm{assets}}} \;:\;
                B^\top W w = 0 \,\},

        where :math:`B` is the loading matrix and :math:`W` is the regression weighting
        matrix: the last observation of `regression_weights` when available, otherwise
        the inverse idiosyncratic variance :math:`D^{-1}`.

        Equivalently, every column of :math:`G` is :math:`W`-orthogonal to every column
        of :math:`B`, so portfolios of the form :math:`G\beta` carry no factor exposure
        under the regression weighting.

        Returns
        -------
        G : ndarray of shape (n_assets, rank)
            Orthonormal columns spanning the orthogonal complement of the factor span,
            with `rank <= n_assets - n_factors_reduced`.
        """
        weight_inv_sqrt, projector = self._orthogonal_projector
        n_assets = weight_inv_sqrt.shape[0]

        ortho = weight_inv_sqrt @ projector
        gram = ortho.T @ ortho
        eigvals, eigvecs = np.linalg.eigh(gram)

        abs_tol = n_assets * np.finfo(float).eps
        rel_tol = n_assets * np.max(np.abs(eigvals)) * np.finfo(float).eps
        keep = eigvals > max(abs_tol, rel_tol)

        if not np.any(keep):
            basis = np.zeros((n_assets, 0))
        else:
            basis = ortho @ eigvecs[:, keep]
            basis, _ = np.linalg.qr(basis, mode="reduced")

        return basis

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
        assets : array-like, slice or None, default=None
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

    def plot_factor_correlation(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        title: str | None = None,
    ) -> go.Figure:
        """Factor return correlation heatmap from the estimated covariance.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names.  Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include.  `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        corr = self.factor_correlation(factors=factors, families=families)
        fig = _heatmap(
            corr,
            labels=factor_names,
            title=title or "Factor Return Correlation",
            zmin=-1,
            zmax=1,
        )
        _add_family_outlines(fig, self.factor_families, factor_indices)
        return fig

    def plot_factor_volatilities(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        annualization_factor: float = 252.0,
        title: str | None = None,
    ) -> go.Figure:
        """Bar chart of annualized factor returns volatilities.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names.  Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include.  `None` includes all factors. Ignored when
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
        sorted_factor_names = [str(factor_names[index]) for index in sort_order]

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
            title=title or "annualized Factor Volatility",
            xaxis_title="Volatility",
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
            Explicit subset of factor names.  Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include.  `None` includes all factors. Ignored when
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

    # Idiosyncratic diagnostics
    def idio_calibration_summary(self) -> pd.Series:
        r"""Summary statistics for the calibration quality of standardized idiosyncratic
         returns.

        Computes time-aggregated statistics of the cross-sectional distribution of
        standardized idiosyncratic returns :math:`z_{it} = u_{it} / \hat\sigma_{i,t}`.

        Under a Gaussian assumption, the expected values are :math:`\text{std}(z) = 1`,
        excess kurtosis :math:`= 0`, skewness :math:`= 0`, and the 3-:math:`\sigma` tail
        rate :math:`\approx 0.27\%`. In practice, standardized idiosyncratic returns
        exhibit fat tails, so the tail rate is typically well above 0.27 % (values
        around 1--3 % are common for equity factor models).

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
        idiosyncratic return :math:`|u_{i,t+1}|`.

        If the model captures the cross-sectional scale of idiosyncratic shocks, then
        assets with larger :math:`\hat\sigma_{i,t}` should tend to realize larger
        absolute moves at :math:`t + 1`.

        * High positive values indicate that the model ranks cross-sectional differences
          in idiosyncratic volatility well.
        * This diagnostic can also pick up broad cross-sectional scale effects such as
          size or liquidity, so it should be read together with
          :attr:`idio_vol_residual_dependence` which checks whether the standardized
          residual magnitude :math:`|u_{i,t+1}| / \hat\sigma_{i,t}` still depends on the
          predicted volatility level.
        """
        self._require(("idio_returns", "idio_variances"), "idio_vol_ic")
        predicted_vol = np.sqrt(np.maximum(self.idio_variances[:-1], 0.0))
        abs_idio_next = np.abs(self.idio_returns[1:])
        corr = cs_rank_correlation(predicted_vol, abs_idio_next, axis=1, min_count=5)
        return pd.Series(
            corr, index=self.observations[1:], name="Idio Vol IC (Spearman)"
        )

    @cached_property
    def idio_vol_residual_dependence(self) -> pd.Series:
        r"""Residual dependence of standardized idiosyncratic returns on predicted
        idiosyncratic volatility.

        Computes the cross-sectional rank correlation (Spearman) between the predicted
        specific volatility :math:`\hat\sigma_{i,t}` and the next-period standardized
        absolute idiosyncratic return :math:`|u_{i,t+1}| / \hat\sigma_{i,t}`.

        In the stylized relation :math:`u_{i,t+1} = \hat\sigma_{i,t}\,\varepsilon_{i,t+1}`,
        dividing by :math:`\hat\sigma_{i,t}` gives
        :math:`|u_{i,t+1}| / \hat\sigma_{i,t} = |\varepsilon_{i,t+1}|`.
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
        corr = cs_rank_correlation(
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
            title=title or "Idiosyncratic Calibration",
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
        :math:`2\,\Phi(-\text{threshold})`, which is about 0.27 % when `threshold = 3`.
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
        expected_rate = 2 * sp_stats.norm.sf(threshold)
        return _plot_single_ts(
            self.idio_tail_rate(threshold=threshold),
            title=title or f"abs(standardized idio returns) > {threshold}",
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
        single observation.  The Gaussian reference is zero, but positive values are
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
            title=title
            or "Cross-Sectional Excess Kurtosis of Standardised Idio Returns",
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
            title=title or "Cross-Sectional Skewness of Standardised Idio Returns",
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
        idiosyncratic return :math:`|u_{i,t+1}|`.

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
            title=title or "Idiosyncratic Volatility IC (Spearman)",
            yaxis_title=(
                "Rank Correlation (Spearman, predicted idio vol vs |idio return|)"
            ),
            window=window,
            show_raw=True,
            show_mean=False,
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
        absolute idiosyncratic return :math:`|u_{i,t+1}| / \hat\sigma_{i,t}`.

        In the stylized relation :math:`u_{i,t+1} = \hat\sigma_{i,t}\,\varepsilon_{i,t+1}`,
        dividing by :math:`\hat\sigma_{i,t}` gives
        :math:`|u_{i,t+1}| / \hat\sigma_{i,t} = |\varepsilon_{i,t+1}|`.
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
            title=title or "Idiosyncratic Volatility Residual Dependence (Spearman)",
            yaxis_title=(
                "Rank Correlation (Spearman, predicted idio vol vs "
                "|idio return| / predicted idio vol)"
            ),
            window=window,
            show_raw=True,
            show_mean=False,
            ref_value=0.0,
        )

    # Cross-sectional regression diagnostics
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

        (
            lagged_exposures,
            factor_returns,
            idio_returns,
            regression_weights,
        ) = self._aligned(
            ["exposures", "factor_returns", "idio_returns", "regression_weights"]
        )

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
        r"""T-statistics of cross-sectional regression coefficients.

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
            `(n_observations - exposure_lag, n_factors_reduced)`.
        """
        return pd.DataFrame(
            self._gram_diagnostics.t_stats,
            index=self._aligned("observations"),
            columns=self._reduced_factor_names,
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
            Shape `(n_factors_reduced,)`.
            Fraction of significant observations per factor.
        """
        t_stats = self._gram_diagnostics.t_stats
        significant = np.abs(t_stats) > threshold
        n_valid = np.sum(np.isfinite(t_stats), axis=0)
        rates = safe_divide(np.nansum(significant, axis=0), n_valid, fill_value=0.0)
        return pd.Series(
            rates,
            index=self._reduced_factor_names,
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
            Number of observations for the rolling mean.

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
            title=title or f"Rolling {label} \u2014 {window}-observation window",
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

        When `window` is provided, plots the rolling mean of :math:`|t|` instead of the
        raw values. A horizontal reference line at :math:`|t| = 2` marks the
        conventional significance threshold.

        Parameters
        ----------
        factors : list of str, optional
            Subset of factor names to include.

        families : str, list of str, or None, default=None
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
            factors, families, reduced_basis=True
        )
        abs_t_stats = np.abs(self._gram_diagnostics.t_stats[:, factor_indices])
        df = pd.DataFrame(
            abs_t_stats, index=self._aligned("observations"), columns=factor_names
        )
        if window is not None:
            df = df.rolling(window=window, min_periods=1).mean()

        default_title = (
            "|t|-statistic per Factor"
            if window is None
            else f"Rolling Mean |t|-statistic \u2014 {window}-observation window"
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
        :math:`|t| >` `threshold`. A vertical reference line at 5 % marks the
        conventional null-rate benchmark used at `threshold = 2`; for other thresholds
        it is only an approximate guide and the exact Gaussian null rate is
        :math:`2\,\Phi(-\text{threshold})`.

        Parameters
        ----------
        factors : list of str, optional
            Subset of factor names to include. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default=None
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
            factors, families, reduced_basis=True
        )
        rates = self.cs_regression_t_stat_exceedance_rate(threshold=threshold)
        rates = rates.loc[factor_names]
        order = np.argsort(rates.values)
        sorted_values = rates.values[order]
        sorted_names = [str(rates.index[i]) for i in order]

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
            annotation_text="5 % (null)",
            annotation_position="top right",
        )
        fig.update_layout(
            title=title
            or f"Cross-sectional Regression t-Statistic Exceedance Rate (|t| > {threshold})",
            xaxis_title="Exceedance Rate",
            yaxis_title="Factor",
        )
        fig.update_xaxes(tickformat=".0%")
        return fig

    # Exposure diagnostics
    def exposure_correlation(
        self,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
        weighting: Literal["benchmark", "regression"] | None = "benchmark",
    ) -> FloatArray:
        """Time-average pairwise correlation matrix of factor exposures.

        Highly correlated exposures indicate redundant factors. They are a
        cross-sectional analogue of multicollinearity diagnostics used in regression,
        where redundant predictors can inflate variance inflation factors (VIFs).

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, optional
            Factor families to include. `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        weighting : "benchmark", "regression", or None, default="benchmark"
            Cross-sectional weights for the correlation computation. `"benchmark"` uses
            benchmark weights, `"regression"`  uses regression weights, and `None` uses
            equal weights. Falls back to equal weights when the requested weights are
            not stored.

        Returns
        -------
        corr : ndarray of shape (n_selected_factors, n_selected_factors)
            Time-average correlation matrix.
        """
        self._require("exposures", "exposure_correlation")
        factor_indices, _ = self._resolve_factor_subset(factors, families)
        exposures = self.exposures[:, :, factor_indices]
        weights = self._resolve_weighting(weighting)
        n_observations, _, n_factors = exposures.shape

        pairwise_corr = np.full((n_observations, n_factors, n_factors), np.nan)
        for factor_i in range(n_factors):
            pairwise_corr[:, factor_i, factor_i] = 1.0
            exposures_i = exposures[:, :, factor_i]
            for factor_j in range(factor_i + 1, n_factors):
                exposures_j = exposures[:, :, factor_j]
                corr_ij = cs_weighted_correlation(
                    exposures_i, exposures_j, weights=weights, axis=1
                )
                pairwise_corr[:, factor_i, factor_j] = corr_ij
                pairwise_corr[:, factor_j, factor_i] = corr_ij
        corr = np.nanmean(pairwise_corr, axis=0)
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
            `(n_observations - exposure_lag, n_factors_reduced)`.
        """
        return pd.DataFrame(
            self._gram_diagnostics.vif,
            index=self._aligned("observations"),
            columns=self._reduced_factor_names,
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
        rank: bool = True,
        horizon: int = 1,
        factors: list[str] | None = None,
        families: str | list[str] | None = None,
    ) -> pd.DataFrame:
        r"""Summary statistics for exposure Information Coefficients (ICs).

        Measures the cross-sectional correlation between factor exposures at :math:`t`
        and the cumulative asset return from :math:`t + 1` to :math:`t + h`, where
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
        rank : bool, default=True
            If `True`, compute Spearman (rank) IC. If `False`, compute Pearson IC
            (weighted by`regression_weights` when available).

        horizon : int, default=1
            Forward window in number of observations. The cumulative return from
            :math:`t + 1` to :math:`t + h` is used.

        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default=None
            Factor families to include. `None` includes all factors.

        Returns
        -------
        summary : DataFrame of shape (n_selected_factors, 4)
            Columns: `mean_ic`, `std_ic`, `ic_ir`, `hit_rate`.
        """
        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        ic, _ = self._ic(rank=rank, horizon=horizon, factor_indices=factor_indices)

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

        When `window` is provided, plots the rolling mean instead of raw per-observation
        values. A horizontal reference line at VIF = 5 marks the conventional
        collinearity threshold.

        Parameters
        ----------
        factors : list of str, optional
            Subset of factor names to include.

        families : str, list of str, or None, default=None
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
            factors, families, reduced_basis=True
        )
        vif = self._gram_diagnostics.vif[:, factor_indices]
        df = pd.DataFrame(
            vif, index=self._aligned("observations"), columns=factor_names
        )

        if window is not None:
            df = df.rolling(window=window, min_periods=1).mean()

        default_title = (
            "Exposure Variance Inflation Factor"
            if window is None
            else f"Rolling Mean Exposure VIF \u2014 {window}-observation window"
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
            Number of observations for the rolling mean.

        title : str, optional
            Custom title.

        Returns
        -------
        fig : go.Figure
        """
        label = "Exposure Condition Number"
        return _plot_single_ts(
            self.exposure_condition_number.rename(label),
            title=title or f"Rolling {label} \u2014 {window}-observation window",
            yaxis_title=label,
            window=window,
            show_raw=True,
            mean_fmt=".4f",
        )

    def plot_cumulative_exposure_ic(
        self,
        rank: bool = True,
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
        rank : bool, default=True
            If `True`, compute Spearman (rank) IC. If `False`, compute Pearson IC
            (weighted by `regression_weights` when available).

        factors : list of str, optional
            Explicit subset of factor names. Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default=None
            Factor families to include. `None` includes all factors.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        factor_indices, factor_names = self._resolve_factor_subset(
            factors, families, reduced_basis=True
        )
        ic, obs_offset = self._ic(
            rank=rank, horizon=1, factor_indices=factor_indices, reduced_basis=True
        )
        cum_ic = np.nancumsum(ic, axis=0)

        default_title = f"Cumulative {'Rank IC' if rank else 'IC'}"

        df = pd.DataFrame(
            cum_ic,
            index=self.observations[obs_offset : obs_offset + cum_ic.shape[0]],
            columns=factor_names,
        )
        return _multi_line_plot(
            df, title=title or default_title, yaxis_title=default_title
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

        observation_idx : int or None, default=None
            Observation index. `None` pools all dates, `-1` selects the last
            observation, `0` the first, etc.

        n_bins : int or None, default=None
            Number of histogram bins. `None` lets Plotly choose automatically.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        self._require("exposures", "plot_exposure_distribution")
        factor_idx = _factor_indices(
            factors=[factor], available=list(self.factor_names)
        )[0]

        if observation_idx is not None:
            obs_label = str(self.observations[observation_idx])
            values = self.exposures[observation_idx, :, factor_idx]
            default_title = f"Exposure Distribution: {factor} ({obs_label})"
        else:
            values = self.exposures[:, :, factor_idx].ravel()
            default_title = f"Exposure Distribution: {factor} (all observations)"

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
        weighting: Literal["benchmark", "regression"] | None = "benchmark",
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

        weighting : "benchmark", "regression", or None, default="benchmark"
            Cross-sectional weights for the std computation. `"benchmark"` uses
            benchmark weights, `"regression"` uses regression weights, and `None` uses
            equal weights. Falls back to equal weights when the requested weights are
            not stored.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        self._require("exposures", "plot_exposure_dispersion")

        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        selected_exposures = self.exposures[:, :, factor_indices]
        weights = self._resolve_weighting(weighting)

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
        weighting: Literal["benchmark", "regression"] | None = "benchmark",
        title: str | None = None,
    ) -> go.Figure:
        r"""Weighted cross-sectional correlation of exposures between observation
        :math:`t` and :math:`t + \text{step}` over time.

        Measures whether the cross-sectional exposures is stable across the chosen
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
            Explicit subset of factor names.  Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include.  `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        step : int, default=21
            Number of observations between the two cross-sections being compared (e.g.,
            21 for approximately monthly stability with daily data).

        weighting : "benchmark", "regression", or None, default="benchmark"
            Cross-sectional weights for the correlation computation. `"benchmark"` uses
            benchmark weights, `"regression"` uses regression weights, and `None` uses
            equal weights. Falls back to equal weights when the requested weights are
            not stored.

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

        stability = self._exposure_stability(exposures, step=step, weighting=weighting)
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
        weighting: Literal["benchmark", "regression"] | None = "benchmark",
        title: str | None = None,
    ) -> go.Figure:
        """Time-average pairwise correlation heatmap of factor exposures.

        Highly correlated exposures indicate redundant factors and may
        inflate VIF.

        Parameters
        ----------
        factors : list of str, optional
            Explicit subset of factor names.  Takes precedence over `families` when
            specified.

        families : str, list of str, or None, default="style"
            Factor families to include.  `None` includes all factors. Ignored when
            `factors` is given or when `factor_families` is `None`.

        weighting : "benchmark", "regression", or None, default="benchmark"
            Cross-sectional weights for the correlation computation. `"benchmark"` uses
            benchmark weights, `"regression"` uses regression weights, and `None` uses
            equal weights. Falls back to equal weights when the requested weights are
            not stored.

        title : str, optional
            Custom figure title.

        Returns
        -------
        fig : go.Figure
        """
        factor_indices, factor_names = self._resolve_factor_subset(factors, families)
        corr_avg = self.exposure_correlation(
            factors=factors, families=families, weighting=weighting
        )
        fig = _heatmap(
            corr_avg,
            labels=factor_names,
            title=title or "Time-Average Exposure Correlation",
            zmin=-1,
            zmax=1,
        )
        _add_family_outlines(fig, self.factor_families, factor_indices)
        return fig

    # Attribution
    def predicted_attribution(
        self,
        weights: ArrayLike,
        annualized_factor: float = 252.0,
        compute_asset_breakdowns: bool = True,
    ) -> Attribution:
        r"""Compute ex-ante (predicted) factor volatility and return attribution.

        Decomposes portfolio volatility using the exposure-volatility-correlation
        framework (:math:`x`-:math:`\sigma`-:math:`\rho`) and, when
        `factor_mu` is available, decomposes expected return into spanned
        and orthogonal components.

        See :func:`~skfolio.factor_model.attribution.predicted_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        weights : array-like of shape (n_assets,)
            Portfolio weights vector.

        annualized_factor : float, default=252.0
            Annualization factor applied to variances and expected returns
            (volatilities are scaled by :math:`\sqrt{\text{annualized\_factor}}`).
            Use 1.0 to disable annualization.

        compute_asset_breakdowns : bool, default=True
            If `True`, compute per-asset systematic/idiosyncratic
            decomposition. Set to `False` for faster computation when
            only portfolio-level results are needed.

        Returns
        -------
        attribution : Attribution
            Component-level, factor-level, and optionally asset-level
            attribution results.
        """
        return predicted_factor_attribution(
            weights=weights,
            loading_matrix=self.loading_matrix,
            factor_covariance=self.factor_covariance,
            idio_covariance=self.idio_covariance,
            asset_names=self.asset_names,
            factor_names=self.factor_names,
            factor_families=self.factor_families,
            factor_mu=self.factor_mu,
            idio_mu=self.idio_mu,
            annualized_factor=annualized_factor,
            compute_asset_breakdowns=compute_asset_breakdowns,
        )

    def realized_attribution(
        self,
        weights: ArrayLike,
        portfolio_returns: ArrayLike,
        annualized_factor: float = 252.0,
        compute_asset_breakdowns: bool = True,
        compute_uncertainty: bool = True,
    ) -> Attribution:
        r"""Compute realized (ex-post) factor volatility and return attribution.

        Decomposes realized portfolio risk and return into contributions
        from individual factors and idiosyncratic sources using actual
        historical data rather than model-predicted covariances.

        See :func:`~skfolio.factor_model.attribution.realized_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        weights : array-like of shape (n_assets,) or (n_observations, n_assets)
            Portfolio weights. If 1D, the same weights are used for all
            observations. If 2D, time-varying weights are used.

        portfolio_returns : array-like of shape (n_observations,)
            Portfolio return time series.

        annualized_factor : float, default=252.0
            Annualization factor applied to variances and mean returns
            (volatilities are scaled by :math:`\sqrt{\text{annualized\_factor}}`).
            Use 1.0 to disable annualization.

        compute_asset_breakdowns : bool, default=True
            If `True`, compute per-asset attribution breakdowns. Set to `False`
            for faster computation when only portfolio-level results are
            needed.

        compute_uncertainty : bool, default=True
            If `True`, compute attribution uncertainty (standard errors on
            the factor/idiosyncratic PnL split). Requires both
            `regression_weights` and `idio_variances` to be available
            in this factor model; raises `ValueError` otherwise.

        Returns
        -------
        attribution : Attribution
            Component-level, factor-level, and optionally asset-level
            attribution results.

        Raises
        ------
        ValueError
            If `factor_returns`, `exposures`, or `idio_returns` is
            not available, or if `compute_uncertainty=True` but
            `regression_weights` or `idio_variances` is missing.
        """
        regression_weights, idio_variances = self._attribution_inputs(
            "realized_attribution", compute_uncertainty
        )

        return realized_factor_attribution(
            factor_returns=self.factor_returns,
            portfolio_returns=portfolio_returns,
            exposures=self.exposures,
            weights=weights,
            idio_returns=self.idio_returns,
            asset_names=self.asset_names,
            factor_names=self.factor_names,
            factor_families=self.factor_families,
            annualized_factor=annualized_factor,
            compute_asset_breakdowns=compute_asset_breakdowns,
            exposure_lag=self.exposure_lag,
            regression_weights=regression_weights,
            idio_variances=idio_variances,
            compute_uncertainty=regression_weights is not None,
            family_constraint_basis=self.family_constraint_basis,
        )

    def rolling_realized_attribution(
        self,
        weights: ArrayLike,
        portfolio_returns: ArrayLike,
        annualized_factor: float = 252.0,
        window_size: int = 60,
        step: int = 21,
        compute_asset_breakdowns: bool = True,
        compute_asset_factor_contribs: bool = False,
        compute_uncertainty: bool = True,
    ) -> Attribution:
        r"""Compute rolling realized (ex-post) factor attribution.

        Runs :func:`~skfolio.factor_model.attribution.rolling_realized_factor_attribution`
        over rolling windows of the factor model's time-varying data.

        See :func:`~skfolio.factor_model.attribution.rolling_realized_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        weights : array-like of shape (n_assets,) or (n_observations, n_assets)
            Portfolio weights. If 1D, the same weights are used for all
            observations. If 2D, time-varying weights are used.

        portfolio_returns : array-like of shape (n_observations,)
            Portfolio return time series.

        annualized_factor : float, default=252.0
            Annualization factor applied to variances and mean returns
            (volatilities are scaled by :math:`\sqrt{\text{annualized\_factor}}`).
            Use 1.0 to disable annualization.

        window_size : int, default=60
            Number of effective return periods in each rolling window.

        step : int, default=21
            Number of observations to advance between consecutive windows.
            The default of 21 produces approximately monthly output for
            daily data.

        compute_asset_breakdowns : bool, default=True
            If `True`, compute per-asset attribution breakdowns for each window.

        compute_asset_factor_contribs : bool, default=False
            If `True`, compute asset-by-factor contributions for each window.

        compute_uncertainty : bool, default=True
            If `True`, compute per-window attribution uncertainty
            (standard errors on the factor/idiosyncratic PnL split).
            Requires both `regression_weights` and `idio_variances`
            to be available in this factor model; raises `ValueError`
            otherwise.

        Returns
        -------
        attribution : Attribution
            Rolling attribution results with an additional leading dimension
            for the number of windows.

        Raises
        ------
        ValueError
            If `factor_returns`, `exposures`, or `idio_returns` are
            not available, or if `window_size` exceeds `n_observations`.
        """
        regression_weights, idio_variances = self._attribution_inputs(
            "rolling_realized_attribution", compute_uncertainty
        )

        return rolling_realized_factor_attribution(
            factor_returns=self.factor_returns,
            portfolio_returns=portfolio_returns,
            exposures=self.exposures,
            weights=weights,
            idio_returns=self.idio_returns,
            factor_names=self.factor_names,
            asset_names=self.asset_names,
            observations=self.observations,
            factor_families=self.factor_families,
            annualized_factor=annualized_factor,
            window_size=window_size,
            step=step,
            compute_asset_breakdowns=compute_asset_breakdowns,
            compute_asset_factor_contribs=compute_asset_factor_contribs,
            exposure_lag=self.exposure_lag,
            regression_weights=regression_weights,
            idio_variances=idio_variances,
            compute_uncertainty=regression_weights is not None,
            family_constraint_basis=self.family_constraint_basis,
        )

    # Private helpers
    @cached_property
    def _orthogonal_projector(self) -> tuple[FloatArray, FloatArray]:
        r"""Whitened projector onto the orthogonal complement of the factor span.

        Builds the shared block :math:`(W^{-1/2},\; I - P)` consumed by
        :attr:`orthogonal_inflation` and :attr:`orthogonal_basis`.

        The whitened loading matrix is :math:`\tilde B = W^{1/2} B`, where :math:`W` is
        the regression weighting matrix:

        * When `regression_weights` is provided, :math:`W` is the diagonal matrix built
          from its last observation (most recent cross-sectional calibration).
        * Otherwise, :math:`W = D^{-1}` is the inverse idiosyncratic variance: diagonal
          when `idio_covariance` is stored as a vector,
          and :math:`(\Sigma_u^{1/2})^{-1}` via the matrix square root when stored as a
          full matrix.

        :math:`P` is the orthogonal projector (in the standard inner product) onto
        :math:`\mathrm{col}(\tilde B)`, computed from :math:`\tilde B`. :math:`I - P`
        therefore projects onto the orthogonal complement of the whitened factor span.
        Mapping back through :math:`W^{-1/2}` recovers the :math:`W`-orthogonal
        complement in original asset coordinates.

        Returns
        -------
        weight_inv_sqrt : ndarray of shape (n_assets, n_assets)
            :math:`W^{-1/2}`. Maps quantities from whitened back to original asset
            coordinates.

        projector : ndarray of shape (n_assets, n_assets)
            :math:`I - P`.
        """
        loading = self.effective_loading_matrix
        n_assets = loading.shape[0]

        regression_weights = (
            None if self.regression_weights is None else self.regression_weights[-1]
        )

        if regression_weights is not None:
            weight_inv_sqrt = np.diag(1.0 / np.sqrt(regression_weights))
            whitened_loading = np.diag(np.sqrt(regression_weights)) @ loading
        elif self.idio_covariance.ndim == 1:
            weight_inv_sqrt = np.diag(np.sqrt(self.idio_covariance))
            whitened_loading = np.diag(1.0 / np.sqrt(self.idio_covariance)) @ loading
        else:
            idio_sqrt = sc_linalg.sqrtm(self.idio_covariance).real
            weight_inv_sqrt = idio_sqrt
            whitened_loading = sc_linalg.inv(idio_sqrt) @ loading

        orthonormal_factor_basis, _ = np.linalg.qr(whitened_loading, mode="reduced")
        projector = (
            np.eye(n_assets) - orthonormal_factor_basis @ orthonormal_factor_basis.T
        )

        return weight_inv_sqrt, projector

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
            if self.exposure_lag > 0:
                family_basis = self.family_constraint_basis[: -self.exposure_lag]
            lagged_exposures = family_basis.to_reduced_exposures(lagged_exposures)
            factor_returns = self.family_constraint_basis.to_reduced_factor_returns(
                factor_returns
            )

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
    def _n_regressors(self) -> int:
        """Effective number of independent regressors."""
        if self.family_constraint_basis is not None:
            return self.family_constraint_basis.n_factors_reduced
        return len(self.factor_names)

    def _resolve_weighting(
        self, weighting: Literal["benchmark", "regression"] | None
    ) -> FloatArray | None:
        """Return the requested weight array, or `None` for equal weights."""
        if weighting == "benchmark":
            return self.benchmark_weights
        if weighting == "regression":
            return self.regression_weights
        return None

    def _ic(
        self,
        rank: bool = True,
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
        rank : bool, default=True
            If `True`, compute Spearman (rank) IC. If `False`, compute Pearson IC
            (weighted by `regression_weights` when available).

        horizon : int, default=1
            Forward window in number of observations.  The cumulative return from
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

        # Cumulative return from obs t+1 to t+h for each t.
        # asset_returns[j] = return at obs j + lag, so the slice
        # starting at base aligns obs t+1 with ret[base + (t - start_t)].
        base = max(1 - self.exposure_lag, 0)
        if horizon == 1:
            cumulative_returns = asset_returns[base : base + n_pairs]
        else:
            windows = np.lib.stride_tricks.sliding_window_view(
                asset_returns[base:], horizon, axis=0
            )
            cumulative_returns = windows[:n_pairs].sum(axis=-1)

        exposures = self.exposures
        if reduced_basis and self.family_constraint_basis is not None:
            exposures = self.family_constraint_basis.to_reduced_exposures(exposures)
        exposure_window = exposures[start_t : start_t + n_pairs, :, factor_indices]

        if rank:
            ic = cs_rank_correlation(
                exposure_window, cumulative_returns[:, :, np.newaxis], axis=1
            )
        else:
            regression_weights = self.regression_weights
            if regression_weights is not None:
                regression_weights = regression_weights[start_t : start_t + n_pairs]
            ic = cs_weighted_correlation(
                exposure_window,
                cumulative_returns[:, :, np.newaxis],
                weights=regression_weights,
                axis=1,
            )
        return ic, start_t

    def _exposure_stability(
        self,
        exposures: FloatArray,
        step: int = 21,
        weighting: Literal["benchmark", "regression"] | None = "benchmark",
    ) -> FloatArray:
        r"""Weighted cross-sectional correlation of exposures between observation
        :math:`t` and :math:`t + \text{step}`.

        Parameters
        ----------
        exposures : ndarray of shape (n_observations, n_assets, n_selected_factors)
            Subset of exposures already sliced to the desired factors.

        step : int, default=21
            Number of observations between the two cross-sections.

        weighting : "benchmark", "regression", or None, default="benchmark"
            Cross-sectional weights passed to :func:`cs_weighted_correlation`.

        Returns
        -------
        stability : ndarray of shape (n_observations - step, n_selected_factors)
        """
        weights = self._resolve_weighting(weighting)
        if weights is not None:
            weights = weights[:-step]
        return cs_weighted_correlation(
            exposures[:-step],
            exposures[step:],
            weights=weights,
            axis=1,
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
        factors: list[str] | None,
        families: str | list[str] | None,
        reduced_basis: bool = False,
    ) -> tuple[slice | list[int], StrArray | list[str]]:
        """Resolve a factor subset from explicit names or family labels.

        Parameters
        ----------
        factors : list of str or None
            Explicit factor names to keep.

        families : str, list of str, or None
            Family labels to keep. Ignored when `factors` is given.

        reduced_basis : bool, default=False
            Use True for diagnostics computed in the reduced basis.

        Returns
        -------
        indices : slice or list of int
            Column selector into *factor_names*. The unfiltered case returns
            `slice(None)` so callers can index arrays without copying.

        names : list of str
            Selected factor names in the same order as `indices`.
        """
        all_names = self._reduced_factor_names if reduced_basis else self.factor_names
        all_names = list(all_names)

        if factors is None and families is None:
            return slice(None), all_names

        if factors is not None:
            return _factor_indices(factors, all_names), factors

        if families is not None:
            if self.factor_families is None:
                raise ValueError(
                    "`families` was specified but `factor_families` is None."
                )
            if isinstance(families, str):
                families = [families]
            available_families = set(self.factor_families)
            unknown = set(families) - available_families
            if unknown:
                raise ValueError(
                    f"Unknown family/families: {sorted(unknown)}. "
                    f"Available families: {sorted(available_families)}."
                )
            name_to_family = dict(
                zip(self.factor_names, self.factor_families, strict=True)
            )
            indices = [
                index
                for index, factor_name in enumerate(all_names)
                if name_to_family.get(factor_name) in families
            ]
            return indices, [all_names[index] for index in indices]

        return list(range(len(all_names))), all_names

    def _standardized_idio_returns(self) -> FloatArray:
        r"""Compute :math:`z_{it} = u_{it} / \hat\sigma_{i,t}`."""
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


class _GramDiagnostics(NamedTuple):
    """Per-observation Gram-matrix diagnostics."""

    t_stats: FloatArray  # (n_observations - exposure_lag, n_factors_reduced)
    vif: FloatArray  # (n_observations - exposure_lag, n_factors_reduced)
    condition_number: FloatArray  # (n_observations - exposure_lag,)


def _factor_indices(factors: list[str], available: list[str]) -> list[int]:
    """Resolve factor names to column indices, raising on unknown."""
    missing = set(factors) - set(available)
    if missing:
        raise ValueError(
            f"Unknown factor(s): {missing}. Available: {[str(x) for x in available]}."
        )
    return [available.index(f) for f in factors]


def _multi_line_plot(df: pd.DataFrame, title: str, yaxis_title: str) -> go.Figure:
    """Create a multi-line time series plot with legend toggling."""
    colors = px.colors.qualitative.Plotly
    fig = go.Figure()
    for i, col in enumerate(df.columns):
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col].values,
                mode="lines",
                name=str(col),
                line=dict(color=colors[i % len(colors)], width=1.5),
                visible="legendonly",
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Observation",
        yaxis_title=yaxis_title,
    )
    return fig


def _plot_single_ts(
    series: pd.Series,
    title: str,
    yaxis_title: str,
    *,
    window: int | None = None,
    show_raw: bool = False,
    show_mean: bool = True,
    ref_value: float | None = None,
    ref_label: str | None = None,
    mean_fmt: str = ".2f",
    tick_format: str | None = None,
) -> go.Figure:
    """Single time-series plot with optional rolling mean and reference lines.

    Parameters
    ----------
    series : pd.Series
        Raw time series (DatetimeIndex).

    title : str
        Figure title.

    yaxis_title : str
        Y-axis label.

    window : int, optional
        If given, smooth with a rolling mean before plotting.

    show_raw : bool, default=False
        When `True` and `window` is set, also plot the raw series as
        a faded line behind the smoothed one. The raw trace is added
        first (`fig.data[0]`) so callers and tests can rely on its
        position.

    show_mean : bool, default=True
        Whether to overlay a horizontal line at the full-sample mean
        with a side annotation.

    ref_value : float, optional
        Y-value for a dashed reference line (e.g. Gaussian expectation).

    ref_label : str, optional
        Annotation text for the reference line.

    mean_fmt : str, default=".2f"
        Format string for the mean annotation value.

    tick_format : str, optional
        Y-axis tick format (e.g. `".2%"`).
    """
    raw = series.values.copy()
    smoothed = (
        series.rolling(window=window, min_periods=1).mean()
        if window is not None
        else series
    )

    mean_val = np.nanmean(raw)

    fig = go.Figure()
    if show_raw and window is not None:
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=raw,
                mode="lines",
                name=series.name,
                line=dict(color="rgba(31, 119, 180, 0.35)", width=1),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed.values,
                mode="lines",
                name=f"Rolling Mean ({window} obs)",
                line=dict(color="rgb(31, 119, 180)", width=2),
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=smoothed.index,
                y=smoothed.values,
                mode="lines",
                name=series.name,
                line=dict(color="rgb(31, 119, 180)", width=1.5),
            )
        )
    if ref_value is not None:
        fig.add_hline(
            y=ref_value,
            line_width=1,
            line_dash="dash",
            line_color="gray",
        )
        if ref_label is not None:
            fig.add_annotation(
                xref="paper",
                yref="y",
                x=1.0,
                y=ref_value,
                text=ref_label,
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                xshift=10,
            )
    if show_mean:
        fig.add_hline(
            y=mean_val,
            line_width=1,
            line_dash="dot",
            line_color="rgb(255, 127, 14)",
        )
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.0,
            y=mean_val,
            text=f"Mean: {mean_val:{mean_fmt}}",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            xshift=10,
        )
    fig.update_layout(
        title=title,
        xaxis_title="Observation",
        yaxis_title=yaxis_title,
        margin=dict(r=180),
    )
    if tick_format is not None:
        fig.update_yaxes(tickformat=tick_format)
    return fig


def _heatmap(
    matrix: FloatArray,
    labels: list[str],
    title: str,
    zmin: float | None = None,
    zmax: float | None = None,
) -> go.Figure:
    """Create an annotated correlation-style heatmap."""
    text = [
        [f"{matrix[i, j]:.2f}" for j in range(len(labels))] for i in range(len(labels))
    ]
    fig = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )
    return fig


def _add_family_outlines(
    fig: go.Figure,
    families: StrArray | None,
    idx: slice | list[int],
) -> None:
    """Draw rectangles around contiguous family blocks on a heatmap."""
    if families is None:
        return
    if idx == slice(None):
        fam_labels = [str(family) for family in families]
    else:
        fam_labels = [str(families[i]) for i in idx]
    unique_families = dict.fromkeys(fam_labels)
    if len(unique_families) <= 1:
        return
    blocks = []
    start = 0
    for end in range(1, len(fam_labels) + 1):
        if end == len(fam_labels) or fam_labels[end] != fam_labels[start]:
            blocks.append((start, end))
            start = end
    if len(blocks) > len(unique_families):
        return
    for start, end in blocks:
        fig.add_shape(
            type="rect",
            x0=start - 0.5,
            y0=start - 0.5,
            x1=end - 0.5,
            y1=end - 0.5,
            line=dict(color="gold", width=2),
        )


def _cs_kurtosis(z: FloatArray) -> FloatArray:
    """Cross-sectional excess kurtosis per observation.

    Uses the bias-corrected (Fisher) estimator matching scipy
    defaults.  Observations with fewer than 4 valid assets return
    NaN.

    Parameters
    ----------
    z : ndarray of shape (n_observations, n_assets)

    Returns
    -------
    kurtosis : ndarray of shape (n_observations,)
    """
    n = np.sum(np.isfinite(z), axis=1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = z - np.nanmean(z, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = np.nansum(d**2, axis=1) / n
        m4 = np.nansum(d**4, axis=1) / n
        raw_kurt = m4 / m2**2 - 3.0

    kurt = np.full(z.shape[0], np.nan)
    ok = n >= 4
    if np.any(ok):
        adj = (n[ok] - 1) / ((n[ok] - 2) * (n[ok] - 3))
        kurt[ok] = ((n[ok] + 1) * raw_kurt[ok] + 6) * adj
    return kurt


def _cs_skewness(z: FloatArray) -> FloatArray:
    """Cross-sectional skewness per observation.

    Uses the bias-corrected (Fisher) estimator matching scipy
    defaults.  Observations with fewer than 3 valid assets return
    NaN.

    Parameters
    ----------
    z : ndarray of shape (n_observations, n_assets)

    Returns
    -------
    skewness : ndarray of shape (n_observations,)
    """
    n = np.sum(np.isfinite(z), axis=1).astype(float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        d = z - np.nanmean(z, axis=1, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        m2 = np.nansum(d**2, axis=1) / n
        m3 = np.nansum(d**3, axis=1) / n
        raw_skew = m3 / m2**1.5

    skew = np.full(z.shape[0], np.nan)
    ok = n >= 3
    if np.any(ok):
        skew[ok] = raw_skew[ok] * np.sqrt(n[ok] * (n[ok] - 1)) / (n[ok] - 2)
    return skew


def _exceedance_agg(threshold: float):
    """Return an aggregation function for t-stat exceedance rate."""

    def _agg(raw_t: FloatArray) -> FloatArray:
        significant = np.abs(raw_t) > threshold
        n_valid = np.sum(np.isfinite(raw_t), axis=0)
        return safe_divide(np.nansum(significant, axis=0), n_valid, fill_value=0.0)

    return _agg


def _lag1_autocorr(x: FloatArray) -> FloatArray:
    """Column-wise lag-1 Pearson autocorrelation."""
    a, b = x[:-1], x[1:]
    a = a - a.mean(0)
    b = b - b.mean(0)
    return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))


def _selector_to_positions(
    selector: ArrayLike | slice | None, labels: StrArray, *, axis_name: str
) -> IntArray:
    """Resolve an axis selector to positional indices.

    Parameters
    ----------
    selector : array-like, slice or None
        Axis selector. Boolean arrays are interpreted as masks, integer arrays
        as positional selectors, slices as positional slices, and other arrays
        as labels matched against `labels`. Negative integer positions follow
        NumPy indexing rules. If `None`, all positions are kept.

    labels : ndarray of shape (n_labels,)
        Labels available on the selected axis.

    axis_name : str
        Axis name used in validation error messages.

    Returns
    -------
    positions : ndarray of shape (n_selected,)
        Positional indices corresponding to `selector`.

    Raises
    ------
    ValueError
        If `selector` is not one-dimensional, if a boolean mask has the wrong
        length, if an integer selector is out of bounds, or if a label is not
        present in `labels`.
    """
    n_labels = len(labels)
    if selector is None:
        return np.arange(n_labels, dtype=np.intp)

    if isinstance(selector, slice):
        start, stop, step = selector.indices(n_labels)
        return np.arange(start, stop, step, dtype=np.intp)

    arr = np.asarray(selector)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError(f"`{axis_name}` must be a 1D selector.")

    if np.issubdtype(arr.dtype, np.bool_):
        if arr.shape[0] != n_labels:
            raise ValueError(
                f"Boolean `{axis_name}` selector must have length {n_labels}, "
                f"got {arr.shape[0]}."
            )
        return np.flatnonzero(arr).astype(np.intp, copy=False)

    if np.issubdtype(arr.dtype, np.integer):
        positions = arr.astype(np.intp, copy=False)
        if np.any((positions < -n_labels) | (positions >= n_labels)):
            raise ValueError(
                f"Integer `{axis_name}` selector contains out-of-bounds positions."
            )
        if np.any(positions < 0):
            positions = positions.copy()
            positions[positions < 0] += n_labels
        return positions

    missing_sentinel = object()
    label_lookup = {label: idx for idx, label in enumerate(labels)}
    positions = np.empty(arr.shape[0], dtype=np.intp)
    missing = []
    for target_index, label in enumerate(arr):
        position = label_lookup.get(label, missing_sentinel)
        if position is missing_sentinel:
            missing.append(label)
        else:
            positions[target_index] = position

    if missing:
        raise ValueError(
            f"{len(missing)} {axis_name} label(s) not found in FactorModel. "
            f"First five: {missing[:5]}"
        )
    return positions


def _positions_to_indexer(positions: IntArray) -> IntArray | slice:
    """Convert contiguous positional indices to an indexer.

    Parameters
    ----------
    positions : ndarray of shape (n_selected,)
        Positional indices on a single axis.

    Returns
    -------
    indexer : ndarray or slice
        A `slice` when `positions` is contiguous, otherwise the input integer
        positions. The slice form lets NumPy return views for contiguous
        selections.
    """
    if len(positions) > 0 and (len(positions) == 1 or np.all(np.diff(positions) == 1)):
        return slice(int(positions[0]), int(positions[-1]) + 1)
    return positions
