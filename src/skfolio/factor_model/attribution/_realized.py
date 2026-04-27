"""Realized (ex-post) factor model attribution."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np

from skfolio.factor_model._family_constraint_basis import FamilyConstraintBasis
from skfolio.factor_model.attribution._model import (
    AssetBreakdown,
    AssetByFactorContribution,
    Attribution,
    Component,
    FactorBreakdown,
    FamilyBreakdown,
)
from skfolio.factor_model.attribution._utils import _cov_with_centered, _validate_no_nan
from skfolio.typing import ArrayLike, FloatArray, ObjArray
from skfolio.utils.stats import safe_divide

__all__ = ["realized_factor_attribution", "rolling_realized_factor_attribution"]


def realized_factor_attribution(
    *,
    asset_names: ArrayLike,
    factor_names: ArrayLike,
    factor_families: ArrayLike | None = None,
    weights: ArrayLike,
    factor_returns: ArrayLike,
    portfolio_returns: ArrayLike,
    exposures: ArrayLike,
    exposure_lag: int = 1,
    idio_returns: ArrayLike,
    idio_variances: ArrayLike | None = None,
    regression_weights: ArrayLike | None = None,
    family_constraint_basis: FamilyConstraintBasis | None = None,
    annualized_factor: float = 252.0,
    compute_asset_breakdowns: bool = True,
    compute_uncertainty: bool = False,
) -> Attribution:
    r"""Compute realized (ex-post) factor volatility and return attribution.

    This function decomposes realized portfolio volatility and return into systematic
    (factors), idiosyncratic and unexplained contributions.

    **Time convention (as-of indexing):**

    All time-varying inputs follow the as-of convention: a value at observation
    :math:`t` reflects information available up to and including the end of period
    :math:`t`. Asset and factor returns span :math:`(t-1, t]`, while exposures
    :math:`B_t` are measured at the end of period :math:`t`.

    For time-varying exposures, returns are explained by predetermined exposures to
    avoid look-ahead bias. When `exposure_lag > 0`, the function aligns
    :math:`B_{t-\ell}` with returns at :math:`t`; the first :math:`\ell` return
    observations are discarded. For 2D static exposures, no trimming is needed.

    .. math::
        R_{P,t} =
        \sum_{k=1}^{K} x_{k,t} f_{k,t}
        + \varepsilon_{P,t}
        + \eta_{P,t}

    where :math:`x_{k,t} = B_{:,k,t-\ell}^\top w_t`, :math:`\varepsilon_{P,t}` is the
    portfolio idiosyncratic return, :math:`\eta_{P,t}` is the unexplained portfolio
    return, and :math:`\ell` is `exposure_lag`.

    **Volatility Attribution (Variance Decomposition):**

    Using the covariance identity, the total portfolio variance decomposes as:

    .. math::
        \operatorname{Var}(R_P) =
        \sum_{k=1}^{K} \operatorname{Cov}(x_k f_k, R_P)
        + \operatorname{Cov}(\varepsilon_P, R_P)
        + \operatorname{Cov}(\eta_P, R_P)

    Each factor's variance contribution is :math:`\operatorname{Cov}(x_k f_k, R_P)`,
    which captures both the exposure magnitude and the factor's correlation with
    portfolio returns. These contributions are additive and sum exactly to total
    variance.

    **Volatility Contribution:**

    The volatility contribution divides the variance contribution by portfolio
    volatility:

    .. math::
        \operatorname{VolContrib}_k =
        \frac{\operatorname{Cov}(x_k f_k, R_P)}{\sigma_P}

    This also satisfies the :math:`\sigma \cdot \rho` identity:

    .. math::
        \operatorname{VolContrib}_k =
        \operatorname{std}(x_k f_k) \cdot
        \operatorname{corr}(x_k f_k, R_P)

    **Return Attribution:**

    The mean return contribution of each factor is the average of the exposure-weighted
    factor returns:

    .. math::
        \operatorname{MuContrib}_k = \overline{x_k f_k}

    Parameters
    ----------
    asset_names : array-like of shape (n_assets,)
        Names for each asset (e.g., ["AAPL", "GOOGL", "MSFT"]).

    factor_names : array-like of shape (n_factors,)
        Names for each factor (e.g., ["Momentum", "Value", "Size"]).

    factor_families : array-like of shape (n_factors,), optional
        Family/category for each factor (e.g., "Style", "Industry").  If provided,
        enables family-level aggregation in the output.

    weights : array-like of shape (n_assets,) or (n_observations, n_assets)
        Portfolio weights. If 1D, the same weights are used for all observations
        (static). If 2D, time-varying weights are used.

    factor_returns : array-like of shape (n_observations, n_factors)
        Factor return time series.

    portfolio_returns : array-like of shape (n_observations,)
        Portfolio return time series.

    exposures : array-like of shape (n_assets, n_factors) or (n_observations, n_assets, n_factors)
        Asset-by-factor exposure (loading) values. If 2D, this is the static loading
        matrix used for all observations. If 3D, this is a time series of loading
        matrices following the as-of convention (the function applies `exposure_lag`
        internally and trims the returns and weights series accordingly).

    exposure_lag : int, default=1
        Lag applied to time-varying exposures under the as-of convention. The default
        value of `1` aligns exposures known at the end of observation :math:`t-1` with
        returns over :math:`(t-1, t]`, explicitly encoding data availability in the API
        and guarding against look-ahead bias. Only affects 3D (time-varying) exposures.

    idio_returns : array-like of shape (n_observations, n_assets)
        Idiosyncratic returns from the factor model regression. These are the residuals
         :math:`\varepsilon_{i,t}` from the cross-sectional regression.

    idio_variances : array-like of shape (n_observations, n_assets) or None, optional
        Per-asset idiosyncratic (specific) variances :math:`\sigma^2_{\varepsilon,i,t}`.
        Required when`compute_uncertainty=True`.

    regression_weights : array-like of shape (n_observations, n_assets) or None, optional
        Per-asset cross-sectional regression weights :math:`q_{i,t}` used when
        estimating factor returns. Required when `compute_uncertainty=True`. Must not
        contain NaN.

    family_constraint_basis : FamilyConstraintBasis or None, optional
        When provided, the uncertainty estimator is computed in the reduced (full-rank)
        basis defined by the family-constraint change of coordinates. This avoids the
        singular Gram matrix that arises from collinear constrained families and
        produces well-conditioned standard errors. Only used when
        `compute_uncertainty=True`.

    annualized_factor : float, default=252.0
        Used to annualize expected returns, variances and volatilities. Use 1.0 to
        disable annualization. Common values: 252 for daily data, 12 for monthly data.

    compute_asset_breakdowns : bool, default=True
        If True, compute asset-level attribution (systematic/idiosyncratic
        decomposition). Set to False to skip asset attribution for faster computation.

    compute_uncertainty : bool, default=False
        If `True`, compute attribution uncertainty (standard errors on
        the factor/idiosyncratic return split). Requires both `regression_weights` and
        `idio_variances`; raises `ValueError` if either is missing. If `False`
        (default), uncertainty is not computed.

    Returns
    -------
    attribution : Attribution
        The :class:`Attribution` dataclass containing component-level,  factor-level
        and optionally asset-level attribution results.

    See Also
    --------
    predicted_factor_attribution : Predicted (ex-ante) factor model attribution.

    Notes
    -----
    When exposures are time-varying, `vol_contrib` cannot be exactly
    reproduced as `exposure_mean * sigma(f) * rho(f, R_P)` because the actual
    contribution is computed from the covariance of the exposure-weighted factor
    return series. The displayed statistics provide intuitive factor-level
    information while the contributions reflect the true realized attribution.

    **NaN handling:**

    `exposures` and `idio_returns` may contain NaN entries for assets that are inactive
    at a given date (delistings, not-yet-listed securities, trading holidays). These NaN
    values are replaced with 0 before any computation: portfolio weight for an inactive
    asset is zero, so its return contribution is economically zero.

    `factor_returns`, `portfolio_returns`, and `weights` must not contain NaN; a
    `ValueError` is raised otherwise.

    Examples
    --------
    >>> from skfolio.factor_model.attribution import realized_factor_attribution
    >>> import numpy as np
    >>>
    >>> # Static exposures and weights
    >>> attribution = realized_factor_attribution(
    ...     factor_returns=factor_returns,  # (252, 3)
    ...     portfolio_returns=portfolio_returns,  # (252,)
    ...     exposures=loading_matrix,  # (10, 3)
    ...     weights=weights,  # (10,)
    ...     idio_returns=residuals,  # (252, 10)
    ...     factor_names=["Momentum", "Value", "Size"],
    ... )
    >>> print(f"Total volatility: {attribution.total.vol:.2%}")
    >>> print(f"Factor contributions: {attribution.factors.vol_contrib}")
    >>>
    >>> # Time-varying weights (e.g., from rebalancing)
    >>> attribution = realized_factor_attribution(
    ...     factor_returns=factor_returns,
    ...     portfolio_returns=portfolio_returns,
    ...     exposures=loading_matrix,
    ...     weights=daily_weights,  # (252, 10)
    ...     idio_returns=residuals,
    ...     factor_names=["Momentum", "Value", "Size"],
    ... )
    >>> print(f"Exposure std (shows position dynamism): {attribution.factors.exposure_std}")
    """
    factor_returns = np.asarray(factor_returns, dtype=float)
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    weights = np.asarray(weights, dtype=float)
    idio_returns = np.asarray(idio_returns, dtype=float)
    factor_names = np.asarray(factor_names)
    asset_names = np.asarray(asset_names)

    _validate_no_nan(factor_returns, "factor_returns")
    _validate_no_nan(portfolio_returns, "portfolio_returns")
    _validate_no_nan(weights, "weights")

    if factor_families is not None:
        factor_families = np.asarray(factor_families)

    if compute_uncertainty:
        if regression_weights is None or idio_variances is None:
            raise ValueError(
                "`compute_uncertainty=True` requires both `regression_weights` "
                "and `idio_variances` to be provided."
            )
        regression_weights = np.asarray(regression_weights, dtype=float)
        idio_variances = np.asarray(idio_variances, dtype=float)
        _validate_no_nan(regression_weights, "regression_weights")
    else:
        regression_weights = None
        idio_variances = None

    _validate_attribution_inputs(
        factor_returns=factor_returns,
        portfolio_returns=portfolio_returns,
        exposures=exposures,
        weights=weights,
        idio_returns=idio_returns,
        factor_names=factor_names,
        asset_names=asset_names,
        factor_families=factor_families,
    )

    # Apply exposure lag for 3D (time-varying) exposures.
    # Under the as-of convention, exposures[t] = B_t and returns[t] = R_t.
    # The cross-sectional identity is R_t = B_{t-l} f_t + epsilon_t, so we
    # align by keeping exposures[:-lag] with returns[lag:].
    exposure_is_static = exposures.ndim == 2
    if not exposure_is_static and exposure_lag > 0:
        exposures = exposures[:-exposure_lag]
        factor_returns = factor_returns[exposure_lag:]
        portfolio_returns = portfolio_returns[exposure_lag:]
        idio_returns = idio_returns[exposure_lag:]
        if weights.ndim == 2:
            weights = weights[exposure_lag:]
        if regression_weights is not None:
            regression_weights = regression_weights[exposure_lag:]
        if idio_variances is not None:
            idio_variances = idio_variances[exposure_lag:]
        if family_constraint_basis is not None:
            family_constraint_basis = family_constraint_basis[:-exposure_lag]

    # NaN in exposures or idio_returns marks inactive (date, asset) entries:
    # delistings, not-yet-listed securities or trading holidays. Both arrays are zeroed
    # at these positions so that systematic and idiosyncratic contributions are zero for
    # inactive entries, preserving the additive identity (systematic + idiosyncratic +
    # unexplained = total) at every time step. This runs after lag alignment so that
    # each exposure row is paired with its corresponding idio row.
    exposures, idio_returns, inactive_mask = _zero_inactive_entries(
        exposures, idio_returns, exposure_is_static
    )

    # Zero NaN in idio_variances at inactive entries.
    if idio_variances is not None:
        idio_variances = np.where(inactive_mask, 0.0, idio_variances)
        _validate_no_nan(idio_variances, "idio_variances")

    return _realized_factor_attribution_core(
        factor_returns=factor_returns,
        portfolio_returns=portfolio_returns,
        exposures=exposures,
        weights=weights,
        idio_returns=idio_returns,
        asset_names=asset_names,
        factor_names=factor_names,
        factor_families=factor_families,
        annualized_factor=annualized_factor,
        compute_asset_breakdowns=compute_asset_breakdowns,
        regression_weights=regression_weights,
        idio_variances=idio_variances,
        family_constraint_basis=family_constraint_basis,
    )


def rolling_realized_factor_attribution(
    *,
    observations: ArrayLike,
    window_size: int = 60,
    step: int = 21,
    asset_names: ArrayLike,
    factor_names: ArrayLike,
    factor_families: ArrayLike | None = None,
    weights: ArrayLike,
    factor_returns: ArrayLike,
    portfolio_returns: ArrayLike,
    exposures: ArrayLike,
    exposure_lag: int = 1,
    idio_returns: ArrayLike,
    idio_variances: ArrayLike | None = None,
    regression_weights: ArrayLike | None = None,
    family_constraint_basis: FamilyConstraintBasis | None = None,
    annualized_factor: float = 252.0,
    compute_asset_breakdowns: bool = True,
    compute_asset_factor_contribs: bool = False,
    compute_uncertainty: bool = False,
) -> Attribution:
    r"""Compute rolling realized (ex-post) factor volatility and return attribution.

    This function computes :func:`realized_factor_attribution` over rolling windows,
    returning an :class:`Attribution` object where all numeric fields are arrays
    with an additional leading dimension corresponding to the number of windows.

    Parameters
    ----------
    observations : array-like of shape (n_observations,)
        Observation labels (e.g., dates) corresponding to each row of the input
        data. The output `Attribution.observations` will contain the labels
        for the last observation of each window.

    window_size : int, default=60
        Number of observations in each rolling window.

    step : int, default=21
        Number of observations to advance between consecutive windows. The default of
        21 produces approximately monthly output for daily data. Use `step=1` for fully
        overlapping windows (daily updates), or `step=window_size` for non-overlapping
        windows.

    asset_names : array-like of shape (n_assets,)
        Names for each asset (e.g., ["AAPL", "GOOGL", "MSFT"]).

    factor_names : array-like of shape (n_factors,)
        Names for each factor (e.g., ["Momentum", "Value", "Size"]).

    factor_families : array-like of shape (n_factors,), optional
        Family/category for each factor (e.g., "Style", "Industry").  If provided,
        enables family-level aggregation in the output.

    weights : array-like of shape (n_assets,) or (n_observations, n_assets)
        Portfolio weights. If 1D, the same weights are used for all observations
        (static). If 2D, time-varying weights are used.

    factor_returns : array-like of shape (n_observations, n_factors)
        Factor return time series.

    portfolio_returns : array-like of shape (n_observations,)
        Portfolio return time series.

    exposures : array-like of shape (n_assets, n_factors) or (n_observations, n_assets, n_factors)
        Asset-by-factor exposure (loading) values. If 2D, this is the static loading
        matrix used for all observations. If 3D, this is a time series of loading
        matrices following the as-of convention (the function applies `exposure_lag`
        internally and trims the returns and weights series accordingly).

    exposure_lag : int, default=1
        Lag applied to time-varying exposures under the as-of convention. The default
        value of `1` aligns exposures known at the end of observation :math:`t-1` with
        returns over :math:`(t-1, t]`, explicitly encoding data availability in the API
        and guarding against look-ahead bias. Only affects 3D (time-varying) exposures.

    idio_returns : array-like of shape (n_observations, n_assets)
        Idiosyncratic returns from the factor model regression. These are the residuals
        :math:`\varepsilon_{i,t}` from the cross-sectional regression.

    idio_variances : array-like of shape (n_observations, n_assets) or None, optional
        Per-asset idiosyncratic (specific) variances :math:`\sigma^2_{\varepsilon,i,t}`.
        Required when`compute_uncertainty=True`.

    regression_weights : array-like of shape (n_observations, n_assets) or None, optional
        Per-asset cross-sectional regression weights :math:`q_{i,t}` used when
        estimating factor returns. Required when `compute_uncertainty=True`. Must not
        contain NaN.

    family_constraint_basis : FamilyConstraintBasis or None, optional
        When provided, the uncertainty estimator is computed in the reduced (full-rank)
        basis defined by the family-constraint change of coordinates. This avoids the
        singular Gram matrix that arises from collinear constrained families and
        produces well-conditioned standard errors. Only used when
        `compute_uncertainty=True`.

    annualized_factor : float, default=252.0
        Used to annualize expected returns, variances and volatilities. Use 1.0 to
        disable annualization. Common values: 252 for daily data, 12 for monthly data.

    compute_asset_breakdowns : bool, default=True
        If True, compute asset-level attribution for each window.
        Results in 2D arrays of shape `(n_windows, n_assets)` in AssetBreakdown.
        Set to False to skip asset attribution for faster computation.

    compute_asset_factor_contribs : bool, default=False
        If True, compute asset-by-factor contributions for each window.
        Results in 3D arrays of shape `(n_windows, n_assets, n_factors)`.
        Disabled by default for faster computation.

    compute_uncertainty : bool, default=False
        If `True`, compute per-window attribution uncertainty (standard errors on the
        factor/idiosyncratic return split). Requires both `regression_weights` and
        `idio_variances`; raises `ValueError` if either is missing. If `False`
        (default), uncertainty is not computed.

    Returns
    -------
    attribution : Attribution
        The :class:`Attribution` dataclass with rolling results. All numeric fields in
        :class:`Component` are 1D arrays of shape `(n_windows,)`. All numeric fields in
        :class:`Breakdown` are 2D arrays of shape `(n_windows, n_factors)` or
        `(n_windows, n_families)`. If `compute_asset_breakdowns=True`, asset attribution
        has shape `(n_windows, n_assets)`. The `observations` field contains the window
        end labels.

    See Also
    --------
    realized_factor_attribution : Single-point realized factor attribution.

    Examples
    --------
    >>> from skfolio.factor_model.attribution import rolling_realized_factor_attribution
    >>> import numpy as np
    >>> import pandas as pd
    >>>
    >>> # Rolling attribution with 60-day windows, advancing 21 days (monthly)
    >>> dates = pd.bdate_range("2023-01-01", periods=252)
    >>> attribution = rolling_realized_factor_attribution(
    ...     factor_returns=factor_returns,  # (252, 3)
    ...     portfolio_returns=portfolio_returns,  # (252,)
    ...     exposures=loading_matrix,  # (10, 3)
    ...     weights=weights,  # (10,)
    ...     idio_returns=residuals,  # (252, 10)
    ...     factor_names=["Momentum", "Value", "Size"],
    ...     observations=dates,
    ...     window_size=60,
    ...     step=21,
    ... )
    >>> print(f"Number of windows: {len(attribution.observations)}")
    >>> print(f"Total vol over time: {attribution.total.vol}")
    >>>
    >>> # Get MultiIndex DataFrame of factor attribution over time
    >>> df = attribution.factors_df(formatted=False)
    >>> print(df.head())
    """
    factor_returns = np.asarray(factor_returns, dtype=float)
    portfolio_returns = np.asarray(portfolio_returns, dtype=float)
    exposures = np.asarray(exposures, dtype=float)
    weights = np.asarray(weights, dtype=float)
    idio_returns = np.asarray(idio_returns, dtype=float)
    factor_names = np.asarray(factor_names)
    asset_names = np.asarray(asset_names)
    observations = np.asarray(observations)

    if factor_families is not None:
        factor_families = np.asarray(factor_families)

    if compute_uncertainty:
        if regression_weights is None or idio_variances is None:
            raise ValueError(
                "`compute_uncertainty=True` requires both `regression_weights` "
                "and `idio_variances` to be provided."
            )
        regression_weights = np.asarray(regression_weights, dtype=float)
        idio_variances = np.asarray(idio_variances, dtype=float)
        _validate_no_nan(regression_weights, "regression_weights")
    else:
        regression_weights = None
        idio_variances = None

    _validate_no_nan(factor_returns, "factor_returns")
    _validate_no_nan(portfolio_returns, "portfolio_returns")
    _validate_no_nan(weights, "weights")

    if factor_returns.ndim != 2:
        raise ValueError(
            f"`factor_returns` must be 2D (n_observations, n_factors), got {factor_returns.ndim}D."
        )

    # Apply exposure lag globally before windowing so that window_size consistently
    # refers to the number of effective return periods.
    exposure_is_static = exposures.ndim == 2
    if not exposure_is_static and exposure_lag > 0:
        exposures = exposures[:-exposure_lag]
        factor_returns = factor_returns[exposure_lag:]
        portfolio_returns = portfolio_returns[exposure_lag:]
        idio_returns = idio_returns[exposure_lag:]
        observations = observations[exposure_lag:]
        if weights.ndim == 2:
            weights = weights[exposure_lag:]
        if regression_weights is not None:
            regression_weights = regression_weights[exposure_lag:]
        if idio_variances is not None:
            idio_variances = idio_variances[exposure_lag:]
        if family_constraint_basis is not None:
            family_constraint_basis = family_constraint_basis[:-exposure_lag]

    exposures, idio_returns, inactive_mask = _zero_inactive_entries(
        exposures, idio_returns, exposure_is_static
    )

    if idio_variances is not None:
        idio_variances = np.where(inactive_mask, 0.0, idio_variances)
        _validate_no_nan(idio_variances, "idio_variances")

    n_observations, _ = factor_returns.shape

    if len(observations) != n_observations:
        raise ValueError(
            f"`observations` length {len(observations)} does not match n_observations={n_observations}."
        )

    if window_size > n_observations:
        raise ValueError(
            f"`window_size` ({window_size}) exceeds n_observations ({n_observations})."
        )

    if window_size < 2:
        raise ValueError(f"`window_size` must be >= 2, got {window_size}.")

    if step < 1:
        raise ValueError(f"`step` must be >= 1, got {step}.")

    # Compute rolling windows: each window spans [start, start + window_size)
    window_starts = np.arange(0, n_observations - window_size + 1, step)
    n_windows = len(window_starts)

    if n_windows == 0:
        raise ValueError(
            f"No valid windows: n_observations={n_observations}, window_size={window_size}, step={step}."
        )

    # Label each window by the last observation it contains
    window_labels = observations[window_starts + window_size - 1]

    weights_is_static = weights.ndim == 1

    has_uncertainty = regression_weights is not None

    # Compute attribution for each window
    results = []
    for start in window_starts:
        end = start + window_size
        attr = _realized_factor_attribution_core(
            factor_returns=factor_returns[start:end],
            portfolio_returns=portfolio_returns[start:end],
            exposures=exposures if exposure_is_static else exposures[start:end],
            weights=weights if weights_is_static else weights[start:end],
            idio_returns=idio_returns[start:end],
            factor_names=factor_names,
            asset_names=asset_names,
            factor_families=factor_families,
            annualized_factor=annualized_factor,
            compute_asset_breakdowns=compute_asset_breakdowns,
            regression_weights=(
                regression_weights[start:end] if has_uncertainty else None
            ),
            idio_variances=(idio_variances[start:end] if has_uncertainty else None),
            family_constraint_basis=(
                family_constraint_basis[start:end]
                if family_constraint_basis is not None
                else None
            ),
        )
        results.append(attr)

    systematic = _stack_dataclass([r.systematic for r in results])
    idio = _stack_dataclass([r.idio for r in results])
    unexplained = _stack_dataclass([r.unexplained for r in results])
    total = _stack_dataclass([r.total for r in results])
    factors = _stack_dataclass([r.factors for r in results])

    if factor_families is not None:
        families = _stack_dataclass([r.families for r in results])
    else:
        families = None

    # Stack asset breakdowns
    if compute_asset_breakdowns and results[0].assets is not None:
        assets = _stack_dataclass([r.assets for r in results])
    else:
        assets = None

    # Stack asset-by-factor contributions.
    if compute_asset_factor_contribs and results[0].asset_by_factor_contrib is not None:
        asset_factor_contribs = _stack_dataclass(
            [r.asset_by_factor_contrib for r in results]
        )
    else:
        asset_factor_contribs = None

    return Attribution(
        systematic=systematic,
        idio=idio,
        unexplained=unexplained,
        total=total,
        factors=factors,
        families=families,
        assets=assets,
        asset_by_factor_contrib=asset_factor_contribs,
        is_realized=True,
        observations=window_labels,
    )


def _validate_attribution_inputs(
    factor_returns: FloatArray,
    portfolio_returns: FloatArray,
    exposures: FloatArray,
    weights: FloatArray,
    idio_returns: FloatArray,
    factor_names: ObjArray,
    asset_names: ObjArray,
    factor_families: ObjArray | None,
) -> None:
    """Validate shapes and consistency of attribution inputs."""
    if factor_returns.ndim != 2:
        raise ValueError(
            f"`factor_returns` must be 2D (n_observations, n_factors), got {factor_returns.ndim}D."
        )
    n_observations, n_factors = factor_returns.shape

    if portfolio_returns.ndim != 1:
        raise ValueError(
            f"`portfolio_returns` must be 1D (n_observations,), got {portfolio_returns.ndim}D."
        )
    if portfolio_returns.shape[0] != n_observations:
        raise ValueError(
            f"`portfolio_returns` length {portfolio_returns.shape[0]} does not match "
            f"n_observations={n_observations} from factor_returns."
        )

    if exposures.ndim == 2:
        n_assets = exposures.shape[0]
        if exposures.shape[1] != n_factors:
            raise ValueError(
                f"`exposures` has {exposures.shape[1]} factors, expected {n_factors}."
            )
    elif exposures.ndim == 3:
        if exposures.shape[0] != n_observations:
            raise ValueError(
                f"`exposures` has {exposures.shape[0]} observations, expected {n_observations}."
            )
        n_assets = exposures.shape[1]
        if exposures.shape[2] != n_factors:
            raise ValueError(
                f"`exposures` has {exposures.shape[2]} factors, expected {n_factors}."
            )
    else:
        raise ValueError(
            f"`exposures` must be 2D (n_assets, n_factors) or "
            f"3D (n_observations, n_assets, n_factors), got {exposures.ndim}D."
        )

    if weights.ndim == 1:
        if weights.shape[0] != n_assets:
            raise ValueError(
                f"`weights` length {weights.shape[0]} does not match n_assets={n_assets}."
            )
    elif weights.ndim == 2:
        if weights.shape[0] != n_observations:
            raise ValueError(
                f"`weights` has {weights.shape[0]} observations, expected {n_observations}."
            )
        if weights.shape[1] != n_assets:
            raise ValueError(
                f"`weights` has {weights.shape[1]} assets, expected {n_assets}."
            )
    else:
        raise ValueError(
            f"`weights` must be 1D (n_assets,) or 2D (n_observations, n_assets), "
            f"got {weights.ndim}D."
        )

    if idio_returns.ndim != 2:
        raise ValueError(
            f"`idio_returns` must be 2D (n_observations, n_assets), got {idio_returns.ndim}D."
        )
    if idio_returns.shape[0] != n_observations:
        raise ValueError(
            f"`idio_returns` has {idio_returns.shape[0]} observations, expected {n_observations}."
        )
    if idio_returns.shape[1] != n_assets:
        raise ValueError(
            f"`idio_returns` has {idio_returns.shape[1]} assets, expected {n_assets}."
        )

    if factor_names.shape[0] != n_factors:
        raise ValueError(
            f"`factor_names` length {factor_names.shape[0]} does not match "
            f"n_factors={n_factors}."
        )

    if asset_names.shape[0] != n_assets:
        raise ValueError(
            f"`asset_names` length {asset_names.shape[0]} does not match "
            f"n_assets={n_assets}."
        )

    if factor_families is not None and factor_families.shape[0] != n_factors:
        raise ValueError(
            f"`factor_families` length {factor_families.shape[0]} does not match "
            f"n_factors={n_factors}."
        )


def _zero_inactive_entries(
    exposures: FloatArray, idio_returns: FloatArray, exposure_is_static: bool
) -> tuple[FloatArray, FloatArray, np.ndarray]:
    """Replace NaN with zero in exposures and idio_returns.

    An entry is considered inactive when either its idiosyncratic return or its
    exposure is NaN. Both are zeroed at inactive positions to preserve the
    additive identity across systematic, idiosyncratic, and unexplained
    components.
    """
    nan_idio = np.isnan(idio_returns)
    nan_exposures = np.isnan(exposures)

    if exposure_is_static:
        inactive_assets = nan_exposures.any(axis=1)
        inactive_mask = nan_idio | inactive_assets[np.newaxis, :]
        exposure_inactive = inactive_assets[:, np.newaxis]
    else:
        inactive_mask = nan_idio | nan_exposures.any(axis=2)
        exposure_inactive = inactive_mask[:, :, np.newaxis]

    if inactive_mask.any():
        idio_returns = np.where(inactive_mask, 0.0, idio_returns)
        exposures = np.where(exposure_inactive, 0.0, exposures)

    return exposures, idio_returns, inactive_mask


def _realized_factor_attribution_core(
    factor_returns: FloatArray,
    portfolio_returns: FloatArray,
    exposures: FloatArray,
    weights: FloatArray,
    idio_returns: FloatArray,
    asset_names: ObjArray,
    factor_names: ObjArray,
    factor_families: ObjArray | None,
    annualized_factor: float,
    compute_asset_breakdowns: bool,
    regression_weights: FloatArray | None = None,
    idio_variances: FloatArray | None = None,
    family_constraint_basis: FamilyConstraintBasis | None = None,
) -> Attribution:
    """Core computation for realized factor attribution.

    Assumes all inputs are validated numpy arrays with NaN already zeroed and exposure
    lag already applied.
    """
    _, n_factors = factor_returns.shape
    exposure_is_static = exposures.ndim == 2
    weights_is_static = weights.ndim == 1

    has_uncertainty = regression_weights is not None and idio_variances is not None

    # Portfolio
    total_mu = float(np.mean(portfolio_returns))
    total_vol = float(np.std(portfolio_returns, ddof=1))
    if total_vol <= 0:
        raise ValueError(
            f"Non-positive total volatility ({total_vol:.2e}). Check Portfolio Returns."
        )
    ptf_ret_centered = portfolio_returns - np.mean(portfolio_returns)

    # Pure factor statistics
    factor_mu = np.mean(factor_returns, axis=0)
    factor_vol = np.std(factor_returns, axis=0, ddof=1)
    factor_cov_with_ptf = _cov_with_centered(factor_returns, ptf_ret_centered)
    factor_corr_with_ptf = np.full(n_factors, np.nan)
    valid_corr = factor_vol > 0
    factor_corr_with_ptf[valid_corr] = factor_cov_with_ptf[valid_corr] / (
        factor_vol[valid_corr] * total_vol
    )

    # Portfolio factor exposures
    if exposure_is_static and weights_is_static:
        ptf_factor = weights @ exposures
        exposure_mean = ptf_factor
        exposure_std = np.zeros(n_factors)
    else:
        if exposure_is_static:
            ptf_factor = weights @ exposures
        elif weights_is_static:
            ptf_factor = exposures.transpose(0, 2, 1) @ weights
        else:
            ptf_factor = (weights[:, np.newaxis, :] @ exposures).squeeze(1)
        exposure_mean = np.mean(ptf_factor, axis=0)
        exposure_std = np.std(ptf_factor, axis=0, ddof=1)

    # Attribution uncertainty
    if has_uncertainty:
        systematic_uncertainty, per_factor_uncertainty, per_family_uncertainty = (
            _compute_attribution_uncertainty(
                exposures=exposures,
                ptf_factor=ptf_factor,
                regression_weights=regression_weights,
                idio_variances=idio_variances,
                factor_families=factor_families,
                annualized_factor=annualized_factor,
                family_constraint_basis=family_constraint_basis,
            )
        )
        idio_uncertainty = systematic_uncertainty
    else:
        systematic_uncertainty = None
        idio_uncertainty = None
        per_factor_uncertainty = None
        per_family_uncertainty = None

    # Factor contributions
    factor_pnl = ptf_factor * factor_returns
    factor_mu_contrib = np.mean(factor_pnl, axis=0)
    factor_pct_total_mu = safe_divide(factor_mu_contrib, total_mu, np.nan, atol=1e-12)
    factor_var_contrib = _cov_with_centered(factor_pnl, ptf_ret_centered)
    factor_vol_contrib = factor_var_contrib / total_vol
    factor_pct_total_variance = factor_vol_contrib / total_vol

    # Systematic
    systematic_pnl = np.sum(factor_pnl, axis=1)
    systematic_mu = float(np.sum(factor_mu_contrib))
    systematic_pct_total_mu = safe_divide(systematic_mu, total_mu, np.nan, atol=1e-12)
    systematic_vol = float(np.std(systematic_pnl, ddof=1))
    systematic_cov = float(_cov_with_centered(systematic_pnl, ptf_ret_centered))
    systematic_corr = safe_divide(
        systematic_cov, systematic_vol * total_vol, np.nan, atol=1e-12
    )
    systematic_variance = float(np.sum(factor_var_contrib))
    systematic_vol_contrib = systematic_variance / total_vol
    systematic_pct_total_variance = systematic_vol_contrib / total_vol

    # Idiosyncratic
    idio_pnl = np.sum(weights * idio_returns, axis=1)
    idio_mu = float(np.mean(idio_pnl))
    idio_mu_pct_total_mu = safe_divide(idio_mu, total_mu, np.nan, atol=1e-12)
    idio_vol = float(np.std(idio_pnl, ddof=1))
    idio_cov = float(_cov_with_centered(idio_pnl, ptf_ret_centered))
    idio_corr = safe_divide(idio_cov, idio_vol * total_vol, np.nan, atol=1e-12)
    idio_vol_contrib = idio_cov / total_vol
    idio_pct_total_variance = idio_vol_contrib / total_vol

    # Unexplained: fees, cash, slippage, model misspecification
    unexplained_pnl = portfolio_returns - systematic_pnl - idio_pnl
    unexplained_mu = float(np.mean(unexplained_pnl))
    unexplained_pct_total_mu = safe_divide(unexplained_mu, total_mu, np.nan, atol=1e-12)
    unexplained_vol = float(np.std(unexplained_pnl, ddof=1))
    unexplained_cov = float(_cov_with_centered(unexplained_pnl, ptf_ret_centered))
    unexplained_corr = safe_divide(
        unexplained_cov, unexplained_vol * total_vol, np.nan, atol=1e-12
    )
    unexplained_vol_contrib = unexplained_cov / total_vol
    unexplained_pct_total_variance = unexplained_vol_contrib / total_vol

    ann_sqrt = math.sqrt(annualized_factor)

    factors = FactorBreakdown(
        names=factor_names,
        family=factor_families,
        exposure=exposure_mean,
        vol=factor_vol * ann_sqrt,
        corr_with_ptf=factor_corr_with_ptf,
        vol_contrib=factor_vol_contrib * ann_sqrt,
        pct_total_variance=factor_pct_total_variance,
        mu=factor_mu * annualized_factor,
        mu_contrib=factor_mu_contrib * annualized_factor,
        pct_total_mu=factor_pct_total_mu,
        exposure_std=exposure_std,
        mu_contrib_uncertainty=per_factor_uncertainty,
    )

    # Family breakdown
    if factor_families is not None:
        families = _compute_realized_family_breakdown(
            factors=factors,
            factor_families=factor_families,
            ptf_factor=ptf_factor,
            is_static=exposure_is_static and weights_is_static,
            per_family_uncertainty=per_family_uncertainty,
        )
    else:
        families = None

    # Asset-level attribution
    if compute_asset_breakdowns:
        assets, asset_factor_contribs = _compute_realized_assets(
            weights=weights,
            exposures=exposures,
            factor_returns=factor_returns,
            idio_returns=idio_returns,
            ptf_ret_centered=ptf_ret_centered,
            total_vol=total_vol,
            total_mu=total_mu,
            asset_names=asset_names,
            factor_names=factor_names,
            exposure_is_static=exposure_is_static,
            weights_is_static=weights_is_static,
            annualized_factor=annualized_factor,
        )
    else:
        assets = None
        asset_factor_contribs = None

    return Attribution(
        systematic=Component(
            vol=systematic_vol * ann_sqrt,
            vol_contrib=systematic_vol_contrib * ann_sqrt,
            pct_total_variance=systematic_pct_total_variance,
            mu_contrib=systematic_mu * annualized_factor,
            pct_total_mu=systematic_pct_total_mu,
            corr_with_ptf=systematic_corr,
            mu_uncertainty=systematic_uncertainty,
        ),
        idio=Component(
            vol=idio_vol * ann_sqrt,
            vol_contrib=idio_vol_contrib * ann_sqrt,
            pct_total_variance=idio_pct_total_variance,
            mu_contrib=idio_mu * annualized_factor,
            pct_total_mu=idio_mu_pct_total_mu,
            corr_with_ptf=idio_corr,
            mu_uncertainty=idio_uncertainty,
        ),
        unexplained=Component(
            vol=unexplained_vol * ann_sqrt,
            vol_contrib=unexplained_vol_contrib * ann_sqrt,
            pct_total_variance=unexplained_pct_total_variance,
            mu_contrib=unexplained_mu * annualized_factor,
            pct_total_mu=unexplained_pct_total_mu,
            corr_with_ptf=unexplained_corr,
        ),
        total=Component(
            vol=total_vol * ann_sqrt,
            vol_contrib=total_vol * ann_sqrt,
            pct_total_variance=1.0,
            mu_contrib=total_mu * annualized_factor,
            pct_total_mu=1.0,
            corr_with_ptf=1.0,
        ),
        factors=factors,
        families=families,
        assets=assets,
        asset_by_factor_contrib=asset_factor_contribs,
        is_realized=True,
    )


def _compute_realized_family_breakdown(
    factors: FactorBreakdown,
    factor_families: ObjArray,
    ptf_factor: FloatArray,
    is_static: bool,
    per_family_uncertainty: FloatArray | None = None,
) -> FamilyBreakdown:
    """Compute family-level breakdown for realized attribution."""
    unique_families, _ = np.unique(factor_families, return_inverse=True)
    n_families = len(unique_families)

    exposure_mean = np.zeros(n_families)
    exposure_std = np.zeros(n_families)
    vol_contrib = np.zeros(n_families)
    pct_total_variance = np.zeros(n_families)
    mu_contrib = np.zeros(n_families)
    pct_total_mu = np.zeros(n_families)

    for i, family in enumerate(unique_families):
        indices = np.where(factor_families == family)[0]
        exposure_mean[i] = factors.exposure[indices].sum()

        # Family exposure std
        if is_static:
            exposure_std[i] = 0.0
        else:
            family_exposure = np.sum(ptf_factor[:, indices], axis=1)
            exposure_std[i] = np.std(family_exposure, ddof=1)

        vol_contrib[i] = factors.vol_contrib[indices].sum()
        pct_total_variance[i] = factors.pct_total_variance[indices].sum()
        mu_contrib[i] = factors.mu_contrib[indices].sum()
        pct_total_mu[i] = np.nansum(factors.pct_total_mu[indices])

    # Sort by absolute pct_total_variance (descending)
    sort_order = np.argsort(-np.abs(pct_total_variance))

    return FamilyBreakdown(
        names=unique_families[sort_order],
        exposure=exposure_mean[sort_order],
        exposure_std=exposure_std[sort_order],
        vol_contrib=vol_contrib[sort_order],
        pct_total_variance=pct_total_variance[sort_order],
        mu_contrib=mu_contrib[sort_order],
        pct_total_mu=pct_total_mu[sort_order],
        mu_contrib_uncertainty=(
            per_family_uncertainty[sort_order]
            if per_family_uncertainty is not None
            else None
        ),
    )


def _compute_realized_assets(
    weights: FloatArray,
    exposures: FloatArray,
    factor_returns: FloatArray,
    idio_returns: FloatArray,
    ptf_ret_centered: FloatArray,
    total_vol: float,
    total_mu: float,
    asset_names: ObjArray,
    factor_names: ObjArray,
    exposure_is_static: bool,
    weights_is_static: bool,
    annualized_factor: float,
) -> tuple[AssetBreakdown, AssetByFactorContribution]:
    """Compute asset-level attribution for realized attribution."""
    ann_sqrt = math.sqrt(annualized_factor)
    n_observations = factor_returns.shape[0]
    n_assets = idio_returns.shape[1]

    # Asset systematic returns: (n_observations, n_assets)
    if exposure_is_static:
        systematic_returns = factor_returns @ exposures.T
    else:
        systematic_returns = (exposures @ factor_returns[:, :, np.newaxis]).squeeze(-1)

    # Asset returns (model): (n_observations, n_assets)
    asset_returns = systematic_returns + idio_returns

    # Standalone asset volatility and mean return
    vol = np.std(asset_returns, axis=0, ddof=1)
    mu = np.mean(asset_returns, axis=0)

    # Compute systematic and idio covariances. Derive total by linearity
    # Cov(systematic + idio, R_P) = Cov(systematic, R_P) + Cov(idio, R_P)
    if weights_is_static:
        weight_mean = weights
        weight_std = np.zeros(n_assets)
        systematic_cov = _cov_with_centered(systematic_returns, ptf_ret_centered)
        idio_cov = _cov_with_centered(idio_returns, ptf_ret_centered)
        cov_with_ptf = systematic_cov + idio_cov
        systematic_vol_contrib = weights * systematic_cov / total_vol
        idio_vol_contrib = weights * idio_cov / total_vol
        total_vol_contrib = systematic_vol_contrib + idio_vol_contrib
        systematic_mu_contrib = weights * np.mean(systematic_returns, axis=0)
        idio_mu_contrib = weights * np.mean(idio_returns, axis=0)
    else:
        weight_mean = np.mean(weights, axis=0)
        weight_std = np.std(weights, axis=0)
        systematic_pnl = weights * systematic_returns
        idio_pnl = weights * idio_returns
        systematic_cov = _cov_with_centered(systematic_returns, ptf_ret_centered)
        idio_cov = _cov_with_centered(idio_returns, ptf_ret_centered)
        cov_with_ptf = systematic_cov + idio_cov
        systematic_vol_contrib = (
            _cov_with_centered(systematic_pnl, ptf_ret_centered) / total_vol
        )
        idio_vol_contrib = _cov_with_centered(idio_pnl, ptf_ret_centered) / total_vol
        total_vol_contrib = systematic_vol_contrib + idio_vol_contrib
        systematic_mu_contrib = np.mean(systematic_pnl, axis=0)
        idio_mu_contrib = np.mean(idio_pnl, axis=0)

    total_mu_contrib = systematic_mu_contrib + idio_mu_contrib

    # Asset correlation with portfolio (uses total covariance from linearity)
    corr_with_ptf = np.full(n_assets, np.nan)
    valid_vol = vol > 0
    corr_with_ptf[valid_vol] = cov_with_ptf[valid_vol] / (vol[valid_vol] * total_vol)

    # Percentage
    pct_total_variance = total_vol_contrib / total_vol
    pct_total_mu = safe_divide(total_mu_contrib, total_mu, np.nan, atol=1e-12)

    assets = AssetBreakdown(
        names=asset_names,
        weight=weight_mean,
        weight_std=weight_std,
        vol=vol * ann_sqrt,
        mu=mu * annualized_factor,
        corr_with_ptf=corr_with_ptf,
        systematic_vol_contrib=systematic_vol_contrib * ann_sqrt,
        idio_vol_contrib=idio_vol_contrib * ann_sqrt,
        vol_contrib=total_vol_contrib * ann_sqrt,
        pct_total_variance=pct_total_variance,
        systematic_mu_contrib=systematic_mu_contrib * annualized_factor,
        idio_mu_contrib=idio_mu_contrib * annualized_factor,
        mu_contrib=total_mu_contrib * annualized_factor,
        pct_total_mu=pct_total_mu,
    )

    # Asset-factor contributions (vectorized over factors)
    if exposure_is_static and weights_is_static:
        factor_pnl = weights[:, None] * exposures * factor_returns[:, None, :]
    elif exposure_is_static:
        factor_pnl = (
            weights[:, :, None] * exposures[None, :, :] * factor_returns[:, None, :]
        )
    elif weights_is_static:
        factor_pnl = weights[None, :, None] * exposures * factor_returns[:, None, :]
    else:
        factor_pnl = weights[:, :, None] * exposures * factor_returns[:, None, :]

    factor_mu_contrib = np.mean(factor_pnl, axis=0)

    # Vectorized covariance: center once
    factor_pnl_centered = factor_pnl - np.mean(factor_pnl, axis=0, keepdims=True)
    factor_vol_contrib = (factor_pnl_centered.transpose(1, 2, 0) @ ptf_ret_centered) / (
        (n_observations - 1) * total_vol
    )

    asset_factor_contribs = AssetByFactorContribution(
        asset_names=asset_names,
        factor_names=factor_names,
        vol_contrib=factor_vol_contrib * ann_sqrt,
        mu_contrib=factor_mu_contrib * annualized_factor,
    )

    return assets, asset_factor_contribs


def _compute_attribution_uncertainty(
    exposures: FloatArray,
    ptf_factor: FloatArray,
    regression_weights: FloatArray,
    idio_variances: FloatArray,
    factor_families: ObjArray | None,
    annualized_factor: float,
    family_constraint_basis: FamilyConstraintBasis | None = None,
) -> tuple[float, FloatArray, FloatArray | None]:
    r"""Compute standard errors for realized factor attribution.

    Propagate cross-sectional factor-return estimation uncertainty to mean return
    contributions. At each observation :math:`t`, the estimated factor return vector
    :math:`\hat{f}_t` has sandwich covariance

    .. math::

        \operatorname{Var}(\hat{f}_t) =
            (B_t^\top W_t B_t)^{-1} \;
            B_t^\top W_t \, \Omega_{\varepsilon,t} \, W_t B_t \;
            (B_t^\top W_t B_t)^{-1}

    where :math:`W_t = \operatorname{diag}(q_{i,t})` and
    :math:`\Omega_{\varepsilon,t} =
    \operatorname{diag}(\sigma^2_{\varepsilon,i,t})`.

    When `family_constraint_basis` is provided, the sandwich covariance is computed in
    the reduced full-rank basis and mapped back to the full factor basis for per-factor
    and per-family reporting.

    The systematic mean return standard error is

    .. math::

        \operatorname{SE} = \frac{\text{ann}}{T}
            \sqrt{\sum_{t=1}^{T} g_t^\top
            \operatorname{Var}(\hat{f}_t) \, g_t}

    where :math:`g_t = B_t^\top w_t` is the portfolio factor exposure vector.

    Parameters
    ----------
    exposures : ndarray of shape (n_assets, n_factors) or (n_observations, n_assets, n_factors)
        Factor exposures in the full basis (already lag-aligned and NaN-zeroed).

    ptf_factor : ndarray of shape (n_factors,) or (n_observations, n_factors)
        Portfolio factor exposure :math:`g_t = B_t^\top w_t` in the full basis.

    regression_weights : ndarray of shape (n_observations, n_assets)
        Per-asset cross-sectional regression weights :math:`q_{i,t}`.

    idio_variances : ndarray of shape (n_observations, n_assets)
        Per-asset idiosyncratic return variances :math:`\sigma^2_{\varepsilon,i,t}`.

    factor_families : ndarray of shape (n_factors,) or None
        Family labels for per-family uncertainty.

    annualized_factor : float
        Annualization factor.

    family_constraint_basis : FamilyConstraintBasis or None
        Reduced full-rank basis used for constrained factor families.

    Returns
    -------
    systematic_uncertainty : float
        Standard error (SE) of the systematic (factor) mean return attribution. Equals
        the idiosyncratic SE because their estimation errors sum to zero (total
        portfolio return is observed).

    per_factor_uncertainty : ndarray of shape (n_factors,)
        Per-factor attribution standard errors.

    per_family_uncertainty : ndarray of shape (n_families,) or None
        Per-family attribution standard errors or None if `factor_families` is not
        provided.
    """
    n_observations, n_assets = regression_weights.shape
    if idio_variances.shape != regression_weights.shape:
        raise ValueError(
            "`idio_variances` must have the same shape as `regression_weights`."
        )

    if exposures.ndim == 2:
        full_exposures = np.broadcast_to(
            exposures[np.newaxis], (n_observations, n_assets, exposures.shape[1])
        )
    else:
        full_exposures = exposures

    full_portfolio_exposure = ptf_factor
    if full_portfolio_exposure.ndim == 1:
        full_portfolio_exposure = np.broadcast_to(
            full_portfolio_exposure[np.newaxis],
            (n_observations, full_portfolio_exposure.shape[0]),
        )

    reg_weights = regression_weights
    idio_var = idio_variances

    # When a family-constraint basis is present, work in the reduced (full-rank) space
    # to avoid the singular Gram matrix caused by collinear constrained families.
    if family_constraint_basis is not None:
        regression_exposures = family_constraint_basis.to_reduced_exposures(
            full_exposures
        )
        regression_portfolio_exposure = (
            family_constraint_basis.to_reduced_factor_coordinates(
                full_portfolio_exposure
            )
        )
    else:
        regression_exposures = full_exposures
        regression_portfolio_exposure = full_portfolio_exposure

    # Gram matrix
    weighted_exposures = regression_exposures * reg_weights[:, :, np.newaxis]
    gram = weighted_exposures.transpose(0, 2, 1) @ regression_exposures

    # Middle term
    variance_weighted_exposures = (
        regression_exposures * (reg_weights * reg_weights * idio_var)[:, :, np.newaxis]
    )
    sandwich_middle_term = (
        variance_weighted_exposures.transpose(0, 2, 1) @ regression_exposures
    )

    # Covariance via solve (avoids explicit inverse):
    try:
        left_solved_term = np.linalg.solve(gram, sandwich_middle_term)
        factor_covariance_regression = np.linalg.solve(
            gram, left_solved_term.transpose(0, 2, 1)
        ).transpose(0, 2, 1)
    except np.linalg.LinAlgError:
        gram_inv = np.linalg.pinv(gram)
        factor_covariance_regression = gram_inv @ sandwich_middle_term @ gram_inv

    scale = annualized_factor / n_observations

    # Total systematic SE (invariant under basis change: g' Var g is the same
    # in full or reduced basis).
    covariance_times_exposure = (
        factor_covariance_regression @ regression_portfolio_exposure[:, :, np.newaxis]
    )
    per_observation_variance = (
        regression_portfolio_exposure[:, np.newaxis, :] @ covariance_times_exposure
    ).ravel()
    systematic_uncertainty = scale * math.sqrt(
        max(0.0, float(np.sum(per_observation_variance)))
    )

    # Map covariance back to full basis for per-factor and per-family SEs.
    if family_constraint_basis is not None:
        factor_covariance = family_constraint_basis.to_full_factor_covariance(
            factor_covariance_regression
        )
    else:
        factor_covariance = factor_covariance_regression

    # Per-factor SE (diagonal elements only)
    factor_variances = np.diagonal(factor_covariance, axis1=1, axis2=2)
    per_factor_uncertainty = scale * np.sqrt(
        np.maximum(0.0, np.sum(full_portfolio_exposure**2 * factor_variances, axis=0))
    )

    # Per-family SE (full covariance sub-block per family)
    if factor_families is not None:
        unique_families = np.unique(factor_families)
        n_families = len(unique_families)
        per_family_uncertainty = np.empty(n_families)
        for i, fam in enumerate(unique_families):
            family_idx = np.where(factor_families == fam)[0]
            family_exposure = full_portfolio_exposure[:, family_idx]
            family_covariance = factor_covariance[:, family_idx][:, :, family_idx]
            family_cov_exposure = family_covariance @ family_exposure[:, :, np.newaxis]
            family_variance = (
                family_exposure[:, np.newaxis, :] @ family_cov_exposure
            ).ravel()
            per_family_uncertainty[i] = scale * math.sqrt(
                max(0.0, float(np.sum(family_variance)))
            )
    else:
        per_family_uncertainty = None

    return systematic_uncertainty, per_factor_uncertainty, per_family_uncertainty


def _stack_dataclass(items: list):
    """Stack a list of dataclass instances into one with arrays stacked along axis 0."""
    first_item = items[0]
    cls = first_item.__class__
    kwargs = {}
    for field in fields(cls):
        fname = field.name
        first_val = getattr(first_item, fname)

        if first_val is None:
            kwargs[fname] = None
        elif isinstance(first_val, (int, float)):
            # Scalar float: stack into 1D array
            kwargs[fname] = np.array([getattr(item, fname) for item in items])
        elif isinstance(first_val, np.ndarray) and np.issubdtype(
            first_val.dtype, np.floating
        ):
            if first_val.ndim == 1:
                # 1D FloatArray: stack into 2D array
                kwargs[fname] = np.vstack([getattr(item, fname) for item in items])
            else:
                # 2D FloatArray: stack into 3D array
                kwargs[fname] = np.stack(
                    [getattr(item, fname) for item in items], axis=0
                )
        else:
            # ObjArray (names, family) or other: take from first element
            kwargs[fname] = first_val

    return cls(**kwargs)
