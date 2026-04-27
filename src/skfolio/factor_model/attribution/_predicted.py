"""Predicted (ex-ante) factor model attribution."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import numpy as np

from skfolio.factor_model.attribution._model import (
    AssetBreakdown,
    AssetByFactorContribution,
    Attribution,
    Component,
    FactorBreakdown,
    FamilyBreakdown,
)
from skfolio.factor_model.attribution._utils import _validate_no_nan
from skfolio.typing import ArrayLike, FloatArray, ObjArray
from skfolio.utils.stats import assert_is_square, safe_divide

__all__ = ["predicted_factor_attribution"]


def predicted_factor_attribution(
    weights: ArrayLike,
    loading_matrix: ArrayLike,
    factor_covariance: ArrayLike,
    idio_covariance: ArrayLike,
    asset_names: ArrayLike,
    factor_names: ArrayLike,
    factor_families: ArrayLike | None = None,
    factor_mu: ArrayLike | None = None,
    idio_mu: ArrayLike | None = None,
    annualized_factor: float = 252.0,
    compute_asset_breakdowns: bool = True,
) -> Attribution:
    r"""Compute predicted (ex-ante) factor volatility and return attribution.

    The volatility attribution follows the exposure-volatility-correlation framework
    (also called :math:`x-\sigma-\rho`). It decomposes portfolio volatility into
    systematic (factor) and idiosyncratic (specific) contributions.

    The return attribution decomposes portfolio expected return into spanned
    (factor-explained) and orthogonal expected return contributions.

    **Factor Model:**

    The asset covariance matrix is modeled as:

    .. math::
        \Sigma = B F B^\top + D

    where :math:`B` is the asset-by-factor loading matrix, :math:`F` is the factor
    covariance matrix and :math:`D` is the idiosyncratic covariance matrix.

    The expected return vector is modeled as:

    .. math::
        \mu = B \lambda + \mu_\perp

    where :math:`\lambda` is the factor expected return (factor premia),
    :math:`B \lambda` is the spanned expected return (spanned alpha) and
    :math:`\mu_\perp` is the orthogonal expected return (orthogonal alpha).

    **Portfolio Variance Decomposition:**

    Let :math:`w` be portfolio weights and :math:`b = B^\top w` be the portfolio factor
     exposure vector. Then:

    .. math::

        \sigma_P^2 = w^\top \Sigma w = b^\top F b + w^\top D w.

    **Portfolio Expected Return Decomposition:**

    .. math::

        \mu_P = w^\top \mu = b^\top \lambda + w^\top \mu_\perp.

    **Volatility Contributions:**

    The contribution of factor :math:`k` to portfolio volatility is defined as:

    .. math::

        \operatorname{VolContrib}_k =
        \frac{b_k (F b)_k}{\sigma_P}.

    where :math:`\sigma_P = \sqrt{w^\top \Sigma w}` is total portfolio volatility.

    These contributions are additive: they sum to the systematic component of
    volatility.

    .. math::

        \sum_k \operatorname{VolContrib}_k = \frac{b^\top F b}{\sigma_P}.

    The systematic vs. idiosyncratic vs. total component contributions are:

    .. math::

        \operatorname{VolContrib}_{\mathrm{sys}} = \frac{b^\top F b}{\sigma_P},
        \qquad
        \operatorname{VolContrib}_{\mathrm{idio}} = \frac{w^\top D w}{\sigma_P},
        \qquad
        \operatorname{VolContrib}_{\mathrm{total}} = \sigma_P.

    and sum exactly:
    :math:`\operatorname{VolContrib}_{\mathrm{sys}} + \operatorname{VolContrib}_{\mathrm{idio}} = \sigma_P`.

    **Expected Return Contributions:**

    The contribution of each factor to spanned expected return is:

    .. math::

        \operatorname{MuContrib}_k = b_k \lambda_k.

    These are also additive:

    .. math::

        \sum_k \operatorname{MuContrib}_k = b^\top \lambda.

    **Correlation (x-sigma-rho framework):**

    Let :math:`\sigma_k = \sqrt{F_{kk}}` be factor :math:`k` standalone
    volatility. The correlation of factor :math:`k` with the portfolio return is:

    .. math::

        \rho_{k,P} = \frac{(F b)_k}{\sigma_k \sigma_P}.

    The factor volatility contribution can then be written as:

    .. math::

        \operatorname{VolContrib}_k = b_k \sigma_k \rho_{k,P}.

    **Percentage of Total:**

    The percentage-of-total variance and expected return are computed as:

    .. math::

        \operatorname{PctTotalVariance}_k =
        \frac{\operatorname{VolContrib}_k}{\sigma_P},
        \qquad
        \operatorname{PctTotalMu}_k =
        \frac{\operatorname{MuContrib}_k}{\mu_P}.

    **NaN handling**:

    `loading_matrix`, `idio_covariance` and `idio_mu` may contain NaN for non-investable
    assets (delisted, not-yet-listed, warm-up). For `idio_covariance`, inactive assets
    are identified by NaN diagonal entries, following the covariance estimator
    convention. When a non-zero weight falls on such an asset a warning is emitted and
    the asset's contribution is effectively zeroed out. `weights`, `factor_covariance`
    and `factor_mu` must be finite.

    Parameters
    ----------
    weights : array-like of shape (n_assets,)
        Portfolio weights vector.

    loading_matrix : array-like of shape (n_assets, n_factors)
        Asset-by-factor loading (exposure) matrix.. NaN entries for non-investable
        assets are filled with 0 (requires corresponding weights to be zero).

    factor_covariance : array-like of shape (n_factors, n_factors)
        Covariance matrix of the factors. Must be per-period (e.g., daily covariance if
        using daily data). Use `annualized_factor` to scale to annualized values.

    idio_covariance : array-like of shape (n_assets,) or (n_assets, n_assets)
        Idiosyncratic (specific) covariance. If 1D, treated as diagonal variances.
        If 2D, used as full covariance matrix. Must be per-period, same as
        `factor_covariance`. NaN entries for non-investable assets are filled with 0.

    asset_names : array-like of shape (n_assets,)
        Names for each asset (e.g., ["AAPL", "GOOGL", "MSFT"]).

    factor_names : array-like of shape (n_factors,)
        Names for each factor (e.g., ["Momentum", "Value", "Size"]).

    factor_families : array-like of shape (n_factors,), optional
        Family/category for each factor (e.g., "Style", "Industry"). If provided,
        enables family-level aggregation in DataFrame output.

    factor_mu : array-like of shape (n_factors,), optional
        Expected returns of each factor (factor premia), :math:`\lambda`. Defaults to
        zeros if not provided. Must be per-period (e.g., daily expected returns if using
        daily data). All inputs (`factor_covariance`, `idio_covariance`, `factor_mu`,
        `idio_mu`) must share the same periodicity. Use `annualized_factor` to scale
        outputs to annualized values.

    idio_mu : array-like of shape (n_assets,), optional
        Per-asset expected idiosyncratic return, :math:`\mu_\perp`, constrained to be
        orthogonal to the factor loadings. It is distinct from the time-series mean of
        `idio_returns`, which is not enforced to be factor-orthogonal. Defaults to zeros
        if not provided. Must be per-period, same as `factor_mu`. NaN entries for
        non-investable assets are filled with 0 (requires corresponding weights to be
        zero).

        .. note::

            This vector **must already be orthogonal** to the column span of the
            loading matrix :math:`B` (with respect to your chosen regression metric,
            e.g., OLS or GLS). This function does **not** perform any orthogonalization.
            It assumes `idio_mu` satisfies the decomposition
            :math:`\mu = B \lambda + \mu_\perp` where :math:`B^\top \mu_\perp = 0` (or
            the appropriate weighted inner product equals zero for GLS). Typically, this
            is the residual vector from regressing asset expected returns onto the factor
            loadings.

    annualized_factor : float, default=252.0
        Used to annualize expected returns, variances and volatilities. Use 1.0 to
        disable annualization. Common values: 252 for daily data, 12 for monthly data.

    compute_asset_breakdowns : bool, default=True
        If True, compute asset-level attribution (systematic/idiosyncratic
        decomposition). Set to False to skip asset attribution for faster computation.

    Returns
    -------
    attribution : Attribution
        The :class:`Attribution` dataclass containing component-level, factor-level and
        optionally asset-level attribution results. Use `attribution.summary_df()`,
        `attribution.factors_df()`, `attribution.assets_df()` to convert to pandas
        DataFrames.

    Raises
    ------
    ValueError
        If input dimensions are inconsistent, if `weights`,`factor_covariance` or
        `factor_mu` contain NaN, or if total variance is non-positive.

    Examples
    --------
    >>> from skfolio.factor_model.attribution import predicted_factor_attribution
    >>> import numpy as np
    >>>
    >>> # Volatility attribution only
    >>> attribution = predicted_factor_attribution(
    ...     weights=np.array([0.4, 0.3, 0.3]),
    ...     loading_matrix=loading_matrix,
    ...     factor_covariance=factor_cov,
    ...     idio_covariance=idio_cov,
    ...     factor_names=["Momentum", "Value", "Size"],
    ... )
    >>> print(f"Total volatility: {attribution.total.vol:.2%}")
    >>> print(f"Factor exposures: {attribution.factors.exposure}")
    >>>
    >>> # With families
    >>> attribution = predicted_factor_attribution(
    ...     weights=np.array([0.4, 0.3, 0.3]),
    ...     loading_matrix=loading_matrix,
    ...     factor_covariance=factor_cov,
    ...     idio_covariance=idio_cov,
    ...     factor_names=["Momentum", "Value", "Size"],
    ...     factor_families=["Style", "Style", "Size"],
    ... )
    >>> print(f"Family names: {attribution.families.names}")
    >>> print(f"Family vol contribs: {attribution.families.vol_contrib}")
    >>>
    >>> # Volatility and return attribution
    >>> attribution = predicted_factor_attribution(
    ...     weights=np.array([0.4, 0.3, 0.3]),
    ...     loading_matrix=loading_matrix,
    ...     factor_covariance=factor_cov,
    ...     idio_covariance=idio_cov,
    ...     factor_names=["Momentum", "Value", "Size"],
    ...     factor_mu=np.array([0.05, 0.03, 0.02]),
    ... )
    >>> attribution.summary_df()
    """
    # Validate weights
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError(f"`weights` must be 1D, got {weights.ndim}D array.")
    n_assets = weights.shape[0]

    # Validate loading matrix
    loadings = np.asarray(loading_matrix, dtype=float)
    if loadings.ndim != 2:
        raise ValueError(f"`loading_matrix` must be 2D, got {loadings.ndim}D array.")
    if loadings.shape[0] != n_assets:
        raise ValueError(
            f"`loading_matrix` must have {n_assets} rows (n_assets), "
            f"got {loadings.shape[0]}."
        )
    n_factors = loadings.shape[1]

    # Validate factor covariance
    factor_cov = np.asarray(factor_covariance, dtype=float)
    assert_is_square(factor_cov)
    if factor_cov.shape[0] != n_factors:
        raise ValueError(
            f"`factor_covariance` shape {factor_cov.shape} does not match "
            f"n_factors={n_factors} from loading_matrix."
        )

    # Validate idiosyncratic covariance (1D or 2D)
    idio_cov = np.asarray(idio_covariance, dtype=float)
    if idio_cov.ndim == 1:
        if idio_cov.shape[0] != n_assets:
            raise ValueError(
                f"`idio_covariance` 1D length {idio_cov.shape[0]} "
                f"does not match n_assets={n_assets}."
            )
        # Convert diagonal variances to full covariance matrix
        idio_cov = np.diag(idio_cov)
    elif idio_cov.ndim == 2:
        assert_is_square(idio_cov)
        if idio_cov.shape[0] != n_assets:
            raise ValueError(
                f"`idio_covariance` shape {idio_cov.shape} "
                f"does not match n_assets={n_assets}."
            )
    else:
        raise ValueError(f"`idio_covariance` must be 1D or 2D, got {idio_cov.ndim}D.")

    # Validate factor names
    factor_names = np.asarray(factor_names)
    if factor_names.shape[0] != n_factors:
        raise ValueError(
            f"`factor_names` length {factor_names.shape[0]} "
            f"does not match n_factors={n_factors}."
        )

    # Validate asset names
    asset_names = np.asarray(asset_names)
    if asset_names.shape[0] != n_assets:
        raise ValueError(
            f"`asset_names` length {asset_names.shape[0]} "
            f"does not match n_assets={n_assets}."
        )

    # Validate factor families
    if factor_families is not None:
        factor_families = np.asarray(factor_families)
        if factor_families.shape[0] != n_factors:
            raise ValueError(
                f"`factor_families` length {factor_families.shape[0]} "
                f"does not match n_factors={n_factors}."
            )

    # Default factor_mu to zeros if not provided
    if factor_mu is None:
        factor_mu = np.zeros(n_factors)
    else:
        factor_mu = np.asarray(factor_mu)
        if factor_mu.ndim != 1:
            raise ValueError(f"`factor_mu` must be 1D, got {factor_mu.ndim}D array.")
        if factor_mu.shape[0] != n_factors:
            raise ValueError(
                f"`factor_mu` length {factor_mu.shape[0]} "
                f"does not match n_factors={n_factors}."
            )

    # Default idio_mu to zeros if not provided
    if idio_mu is None:
        idio_mu = np.zeros(n_assets)
    else:
        idio_mu = np.asarray(idio_mu)
        if idio_mu.ndim != 1:
            raise ValueError(f"`idio_mu` must be 1D, got {idio_mu.ndim}D array.")
        if idio_mu.shape[0] != n_assets:
            raise ValueError(
                f"`idio_mu` length {idio_mu.shape[0]} "
                f"does not match n_assets={n_assets}."
            )

    # NaN handling
    # factor_covariance and factor_mu are per-factor quantities and must be finite.
    _validate_no_nan(weights, "weights")
    _validate_no_nan(factor_cov, "factor_covariance")
    _validate_no_nan(factor_mu, "factor_mu")

    # loading_matrix, idio_covariance and idio_mu may contain NaN for non-investable
    # assets (delistings, warm-up, not-yet-listed). A non-zero weight on such an asset
    # triggers a warning. NaN entries are then filled with 0 so that 0*NaN do not
    # propagate.
    non_investable = np.isnan(loadings).any(axis=1)
    non_investable |= np.isnan(np.diag(idio_cov))
    non_investable |= np.isnan(idio_mu)
    if non_investable.any():
        conflict = non_investable & (np.abs(weights) > 0)
        if conflict.any():
            names = asset_names[conflict].tolist()
            warnings.warn(
                f"{int(conflict.sum())} asset(s) have non-zero weight but "
                f"NaN model estimates (delisted or non-investable): {names}. "
                f"Their contributions are zeroed out in the decomposition.",
                stacklevel=2,
            )
        loadings = np.where(np.isnan(loadings), 0.0, loadings)
        idio_cov = np.where(np.isnan(idio_cov), 0.0, idio_cov)
        idio_mu = np.where(np.isnan(idio_mu), 0.0, idio_mu)

    # Apply annualization
    factor_cov = factor_cov * annualized_factor
    idio_cov = idio_cov * annualized_factor
    factor_mu = factor_mu * annualized_factor
    idio_mu = idio_mu * annualized_factor

    # Portfolio factor exposures
    factor = loadings.T @ weights

    # Variance decomposition
    factor_cov_with_ptf = factor_cov @ factor
    systematic_var = float(factor.T @ factor_cov_with_ptf)
    idio_var = float(weights.T @ idio_cov @ weights)
    total_var = systematic_var + idio_var

    if total_var <= 0:
        raise ValueError(
            f"Non-positive total variance ({total_var:.2e}). "
            "Check inputs for alignment and unit consistency."
        )

    total_vol = float(np.sqrt(total_var))
    systematic_vol = float(np.sqrt(systematic_var))
    idio_vol = float(np.sqrt(idio_var))

    # Volatility contributions (additive, sum to total_vol)
    systematic_vol_contrib = systematic_var / total_vol
    idio_vol_contrib = idio_var / total_vol

    systematic_pct_total_variance = systematic_var / total_var
    idio_pct_total_variance = idio_var / total_var

    # Component correlations with portfolio
    # Under factor model: Cov(component, ptf) = Var(component) since components
    # are uncorrelated. Therefore: Corr = Var / (vol * total_vol) = vol / total_vol
    systematic_corr_with_ptf = systematic_vol / total_vol
    idio_corr_with_ptf = idio_vol / total_vol

    # Per-factor risk statistics
    factor_vol = np.sqrt(np.maximum(np.diag(factor_cov), 0.0))

    # Correlation of each factor with total portfolio
    factor_corr_with_ptf = np.full(n_factors, np.nan, dtype=float)
    nonzero_vol = factor_vol > 0
    factor_corr_with_ptf[nonzero_vol] = factor_cov_with_ptf[nonzero_vol] / (
        factor_vol[nonzero_vol] * total_vol
    )

    # Volatility contribution per factor
    factor_vol_contrib = factor * factor_cov_with_ptf / total_vol
    factor_pct_total_variance = factor_vol_contrib / total_vol

    # Per-factor expected return contribution
    factor_mu_contrib = factor * factor_mu

    # Spanned expected return
    systematic_mu = float(np.sum(factor_mu_contrib))

    # Orthogonal expected return
    orthogonal_mu = float(weights.T @ idio_mu)

    # Total expected return
    total_mu = systematic_mu + orthogonal_mu

    # Percentage contributions
    spanned_pct_total_mu = safe_divide(systematic_mu, total_mu, np.nan, atol=1e-12)
    orthogonal_pct_total_mu = safe_divide(orthogonal_mu, total_mu, np.nan, atol=1e-12)
    factor_pct_total_mu = safe_divide(factor_mu_contrib, total_mu, np.nan, atol=1e-12)

    # Factors breakdown
    factors = FactorBreakdown(
        names=factor_names,
        family=factor_families,
        exposure=factor,
        exposure_std=None,
        vol=factor_vol,
        corr_with_ptf=factor_corr_with_ptf,
        vol_contrib=factor_vol_contrib,
        pct_total_variance=factor_pct_total_variance,
        mu=factor_mu,
        mu_contrib=factor_mu_contrib,
        pct_total_mu=factor_pct_total_mu,
    )

    # Family breakdown
    if factor_families is not None:
        families = _compute_predicted_family_breakdown(
            factors=factors,
            factor_families=factor_families,
        )
    else:
        families = None

    # Asset-level attribution
    if compute_asset_breakdowns:
        assets, asset_factor_contribs = _compute_predicted_assets(
            weights=weights,
            loadings=loadings,
            factor_cov=factor_cov,
            idio_cov=idio_cov,
            factor_mu=factor_mu,
            idio_mu=idio_mu,
            factor_cov_with_ptf=factor_cov_with_ptf,
            total_vol=total_vol,
            total_mu=total_mu,
            asset_names=asset_names,
            factor_names=factor_names,
        )
    else:
        assets = None
        asset_factor_contribs = None

    return Attribution(
        systematic=Component(
            vol=systematic_vol,
            vol_contrib=systematic_vol_contrib,
            pct_total_variance=systematic_pct_total_variance,
            mu_contrib=systematic_mu,
            pct_total_mu=spanned_pct_total_mu,
            corr_with_ptf=systematic_corr_with_ptf,
        ),
        idio=Component(
            vol=idio_vol,
            vol_contrib=idio_vol_contrib,
            pct_total_variance=idio_pct_total_variance,
            mu_contrib=orthogonal_mu,
            pct_total_mu=orthogonal_pct_total_mu,
            corr_with_ptf=idio_corr_with_ptf,
        ),
        total=Component(
            vol=total_vol,
            vol_contrib=total_vol,
            pct_total_variance=1.0,
            mu_contrib=total_mu,
            pct_total_mu=1.0,
            corr_with_ptf=1.0,
        ),
        unexplained=None,
        factors=factors,
        families=families,
        assets=assets,
        asset_by_factor_contrib=asset_factor_contribs,
    )


def _compute_predicted_family_breakdown(
    factors: FactorBreakdown, factor_families: ObjArray
) -> FamilyBreakdown:
    """Compute family-level breakdown for predicted attribution."""
    unique_families, _ = np.unique(factor_families, return_inverse=True)
    n_families = len(unique_families)

    exposure = np.zeros(n_families)
    vol_contrib = np.zeros(n_families)
    pct_total_variance = np.zeros(n_families)
    mu_contrib = np.zeros(n_families)
    pct_total_mu = np.zeros(n_families)

    for i, family in enumerate(unique_families):
        indices = np.where(factor_families == family)[0]
        exposure[i] = factors.exposure[indices].sum()
        vol_contrib[i] = factors.vol_contrib[indices].sum()
        pct_total_variance[i] = factors.pct_total_variance[indices].sum()
        mu_contrib[i] = factors.mu_contrib[indices].sum()
        pct_total_mu[i] = np.nansum(factors.pct_total_mu[indices])

    # Sort by absolute pct_total_variance (descending)
    sort_order = np.argsort(-np.abs(pct_total_variance))

    return FamilyBreakdown(
        names=unique_families[sort_order],
        exposure=exposure[sort_order],
        exposure_std=None,
        vol_contrib=vol_contrib[sort_order],
        pct_total_variance=pct_total_variance[sort_order],
        mu_contrib=mu_contrib[sort_order],
        pct_total_mu=pct_total_mu[sort_order],
    )


def _compute_predicted_assets(
    weights: FloatArray,
    loadings: FloatArray,
    factor_cov: FloatArray,
    idio_cov: FloatArray,
    factor_mu: FloatArray,
    idio_mu: FloatArray,
    factor_cov_with_ptf: FloatArray,
    total_vol: float,
    total_mu: float,
    asset_names: ObjArray,
    factor_names: ObjArray,
) -> tuple[AssetBreakdown, AssetByFactorContribution]:
    """Compute asset-level attribution for predicted attribution."""
    n_assets = len(weights)

    # Full covariance for asset-level stats
    systematic_cov = loadings @ factor_cov @ loadings.T
    full_cov = systematic_cov + idio_cov

    # Asset standalone volatility and expected return
    vol = np.sqrt(np.maximum(np.diag(full_cov), 0.0))
    mu = loadings @ factor_mu + idio_mu

    # Asset covariance with portfolio
    cov_with_ptf = full_cov @ weights

    # Asset correlation with portfolio
    corr_with_ptf = np.full(n_assets, np.nan)
    valid_vol = vol > 0
    corr_with_ptf[valid_vol] = cov_with_ptf[valid_vol] / (vol[valid_vol] * total_vol)

    # Asset total volatility contributions
    total_vol_contrib = weights * cov_with_ptf / total_vol

    # Systematic volatility contributions
    systematic_vol_contrib = weights * (systematic_cov @ weights) / total_vol

    # Idiosyncratic volatility contributions
    idio_vol_contrib = weights * (idio_cov @ weights) / total_vol

    # Percentage of total variance
    pct_total_variance = total_vol_contrib / total_vol

    # Asset return contributions
    systematic_mu_contrib = weights * (loadings @ factor_mu)
    idio_mu_contrib = weights * idio_mu
    total_mu_contrib = systematic_mu_contrib + idio_mu_contrib
    pct_total_mu = safe_divide(total_mu_contrib, total_mu, np.nan, atol=1e-12)

    assets = AssetBreakdown(
        names=asset_names,
        weight=weights,
        weight_std=None,
        vol=vol,
        mu=mu,
        corr_with_ptf=corr_with_ptf,
        systematic_vol_contrib=systematic_vol_contrib,
        idio_vol_contrib=idio_vol_contrib,
        vol_contrib=total_vol_contrib,
        pct_total_variance=pct_total_variance,
        systematic_mu_contrib=systematic_mu_contrib,
        idio_mu_contrib=idio_mu_contrib,
        mu_contrib=total_mu_contrib,
        pct_total_mu=pct_total_mu,
    )

    # Asset-factor contribs
    factor_vol_contrib = np.outer(weights, factor_cov_with_ptf) * loadings / total_vol
    factor_mu_contrib = np.outer(weights, factor_mu) * loadings

    asset_factor_contrib = AssetByFactorContribution(
        asset_names=asset_names,
        factor_names=factor_names,
        vol_contrib=factor_vol_contrib,
        mu_contrib=factor_mu_contrib,
    )

    return assets, asset_factor_contrib
