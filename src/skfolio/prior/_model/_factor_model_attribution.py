"""Factor attribution methods of the factor model."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from skfolio.attribution import (
    Attribution,
    predicted_factor_attribution,
    realized_factor_attribution,
    rolling_realized_factor_attribution,
)
from skfolio.typing import (
    ArrayLike,
)


class _AttributionMixin:
    """Factor attribution, mixed into :class:`~skfolio.prior.FactorModel`.

    These methods decompose portfolio risk and return over the factors, predicted from
    the factor moments or realized from the factor return time series.

    This mixin is not usable on its own: it reads the fields and private helpers of
    `FactorModel`.
    """

    def predicted_attribution(
        self,
        weights: ArrayLike,
        annualization_factor: float = 252.0,
        compute_asset_breakdowns: bool = True,
    ) -> Attribution:
        r"""Compute ex-ante (predicted) factor volatility and return attribution.

        Decomposes portfolio volatility using the exposure-volatility-correlation
        framework (:math:`x`-:math:`\sigma`-:math:`\rho`) and, when
        `factor_mu` is available, decomposes expected return into factor-spanned
        and factor-orthogonal components.

        See :func:`~skfolio.attribution.predicted_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        weights : array-like of shape (n_assets,)
            Portfolio weights vector.

        annualization_factor : float, default=252.0
            Annualization factor applied to variances and expected returns
            (volatilities are scaled by :math:`\sqrt{\text{annualization\_factor}}`).
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
            annualization_factor=annualization_factor,
            compute_asset_breakdowns=compute_asset_breakdowns,
        )

    def realized_attribution(
        self,
        weights: ArrayLike,
        portfolio_returns: ArrayLike,
        annualization_factor: float = 252.0,
        compute_asset_breakdowns: bool = True,
        compute_uncertainty: bool = True,
    ) -> Attribution:
        r"""Compute realized (ex-post) factor volatility and return attribution.

        Decomposes realized portfolio risk and return into contributions
        from individual factors and idiosyncratic sources using actual
        historical data rather than model-predicted covariances.

        See :func:`~skfolio.attribution.realized_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        weights : array-like of shape (n_assets,) or (n_observations, n_assets)
            Portfolio weights. If 1D, the same weights are used for all
            observations. If 2D, time-varying weights are used.

        portfolio_returns : array-like of shape (n_observations,)
            Portfolio return time series.

        annualization_factor : float, default=252.0
            Annualization factor applied to variances and mean returns
            (volatilities are scaled by :math:`\sqrt{\text{annualization\_factor}}`).
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
            annualization_factor=annualization_factor,
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
        annualization_factor: float = 252.0,
        window_size: int = 60,
        step: int = 21,
        compute_asset_breakdowns: bool = True,
        compute_asset_factor_contribs: bool = False,
        compute_uncertainty: bool = True,
    ) -> Attribution:
        r"""Compute rolling realized (ex-post) factor attribution.

        Runs :func:`~skfolio.attribution.rolling_realized_factor_attribution`
        over rolling windows of the factor model's time-varying data.

        See :func:`~skfolio.attribution.rolling_realized_factor_attribution`
        for the full mathematical description.

        Parameters
        ----------
        weights : array-like of shape (n_assets,) or (n_observations, n_assets)
            Portfolio weights. If 1D, the same weights are used for all
            observations. If 2D, time-varying weights are used.

        portfolio_returns : array-like of shape (n_observations,)
            Portfolio return time series.

        annualization_factor : float, default=252.0
            Annualization factor applied to variances and mean returns
            (volatilities are scaled by :math:`\sqrt{\text{annualization\_factor}}`).
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
            annualization_factor=annualization_factor,
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
