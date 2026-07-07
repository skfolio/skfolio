"""Characteristics Factor Model."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from graphlib import TopologicalSorter
from typing import ClassVar

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.utils as sku
import sklearn.utils.metadata_routing as skm
import sklearn.utils.parallel as skp
import sklearn.utils.validation as skv

from skfolio._constants import (
    _BENCHMARK_WEIGHTS,
    _CURRENCY,
    _EXPOSURES,
    _IDIO_RETURNS,
    _IDIO_VARIANCES,
    _MARKET_CAP,
    _REGRESSION_WEIGHTS,
    _RETURNS,
)
from skfolio.alpha import BaseAlpha
from skfolio.base import BaseComposition
from skfolio.containers import AssetPanel, InactivePolicy
from skfolio.factor_exposure import BaseFactorExposure, DerivedFactor
from skfolio.linear_model import BaseCSLinearModel, CSLinearRegression
from skfolio.moments import (
    BaseCovariance,
    EWCovariance,
    EWMu,
    RegimeAdjustedEWCovariance,
)
from skfolio.moments.variance import BaseVariance, RegimeAdjustedEWVariance
from skfolio.preprocessing import CSWinsorizer
from skfolio.prior._base import BasePrior
from skfolio.prior._empirical import EmpiricalPrior
from skfolio.prior._model import FactorModel, ReturnDistribution
from skfolio.prior._model._family_constraint_basis import (
    FamilyConstraintBasis,
    compute_family_constraint_basis,
)
from skfolio.typing import AnyArray, BoolArray, FloatArray, StrArray
from skfolio.utils._array_buffer import _ArrayBuffer, _update_buffer
from skfolio.utils._factor_tools import _neutralize_exposures
from skfolio.utils.equations import _validate_factor_names_and_families
from skfolio.utils.stats import corr_to_cov, cov_nearest, cov_to_corr, safe_divide
from skfolio.utils.tools import (
    _filter_supported_params,
    _validate_non_negative_real,
    _validate_positive_integer,
    _validate_positive_real,
    _validate_unit_interval,
    call_asset_panel_transform,
    check_estimator,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "factor_model_"


class CharacteristicsFactorModel(BasePrior, BaseComposition):
    r"""Characteristics-based cross-sectional factor model.

    `CharacteristicsFactorModel` estimates a point-in-time, cross-sectional equity
    factor model from asset characteristics stored in an
    :class:`~skfolio.containers.AssetPanel` [1]_.

    The model is fitted as follows:

    1. Start from point-in-time asset characteristics stored as panel fields (e.g.,
       `returns`, `market_cap`, `book_equity`, `industry`, `country`).
    2. Compute descriptor values from these fields, or pass through existing fields
       unchanged, using descriptor estimators (e.g.
       :class:`~skfolio.descriptor.BookToPrice`,
       :class:`~skfolio.descriptor.EWMomentum`,
       :class:`~skfolio.descriptor.Passthrough`).
    3. Build factor exposures. Style factors are typically formed by combining one or
       more descriptors and applying cross-sectional transformation such as
       winsorization and z-scoring, for example with
       :class:`~skfolio.factor_exposure.FixedWeightedFactor`. Categorical
       factors (e.g. industry, country, currency) are represented by one-hot exposures
       with :class:`~skfolio.factor_exposure.OneHotCategoricalFactors`.
    4. Orthogonalize selected exposures against other factors or families when
       `neutralize_against` is provided.
    5. Reparameterize constrained families when `constrained_families` is
       provided. This enforces the benchmark-weighted zero-sum constraint on factor
       returns within each constrained family and produces a full-rank basis for
       factor-level estimators, such as the factor covariance estimator.
    6. Lag exposures by `exposure_lag` periods and estimate realized factor returns
       with `cs_regressor` on the estimation universe defined by the panel's
       `estimation_mask`. By default, regression weights are based on market
       capitalization through `regression_mcap_power`. When
       `inv_idio_variance_weight_shrinkage > 0`, a two-pass procedure blends those
       weights with inverse-idiosyncratic-variance weights estimated from first-pass
       residuals.
    7. Estimate the factor return distribution with `factor_prior_estimator`,
       including expected factor returns (factor premia), factor covariance, and factor
       return scenarios. This step can introduce factor covariance shrinkage,
       short-term volatility updating, or Newey-West HAC correction.
    8. Estimate idiosyncratic variances with `idio_variance_estimator`, then form
       the idiosyncratic covariance as a diagonal matrix or, when
       `idio_corr_threshold > 0`, as a sparse covariance using correlation thresholding.
    9. If provided, fit `alpha_estimator` to produce alpha forecast. Decompose it
       into factor-spanned and orthogonal alphas, blend the spanned alpha with the
       expected factor returns using `spanned_alpha_shrinkage`, shrink the orthogonal
       alpha with `orthogonal_alpha_confidence` and assemble the final :math:`\mu`,
       :math:`\Sigma` and asset return scenarios on the investment universe.

    For asset :math:`i` and observation :math:`t`, factor returns are estimated from the
    cross-sectional regression

    .. math::

        R_i(t) = B_i(t - \ell)\,f(t) + \epsilon_i(t)

    where :math:`R_i(t)` is the local excess return, :math:`B_i(t-\ell)` denotes
    asset :math:`i`'s factor exposure vector, :math:`f(t)` is the vector of realized
    factor returns, :math:`\ell` is the exposure lag and :math:`\epsilon_i(t)` is the
    idiosyncratic return.

    The estimator follows skfolio's as-of time-indexing convention: all time-varying
    inputs at observation :math:`t` reflect information available up to and including
    the end of period :math:`t`. Point-in-time fields and derived values store the
    latest available value for observation :math:`t`. Returns stored at observation
    :math:`t` cover the period ending at :math:`t`, namely :math:`(t-1, t]`.

    Factor-return regressions estimate the factor returns realized over
    :math:`(t-1, t]`. The exposure matrix must therefore describe the assets before
    that return interval begins. `exposure_lag` selects that exposure date.

    The fitted covariance uses the latest available exposures :math:`B(T)`:

    .. math::

        \Sigma = B(T)\,F\,B(T)^\top + D

    where :math:`F` is the factor return covariance and :math:`D` is the idiosyncratic
    covariance.

    The fitted expected-return vector combines expected factor returns with optional
    alpha forecast. When an alpha estimator is provided, its contribution is controlled
    by `spanned_alpha_shrinkage` for the factor-spanned alpha and
    `orthogonal_alpha_confidence` for the orthogonal alpha. Direct currency expected
    returns are added when currency factors are present.

    Asset return scenarios in `return_distribution_.returns` are built by mapping
    factor-prior scenarios through the latest loading matrix and adding idiosyncratic
    scenarios calibrated to the latest idiosyncratic-risk forecast. They therefore
    include both factor and idiosyncratic return components for downstream optimizers
    that use scenario-based risk measures such as CVaR.

    The estimator distinguishes the coverage, estimation and investment universes:

    * The coverage universe is the set of assets stored in `characteristics`. The
      panel's `active_mask` identifies which asset-observation pairs are active
      within that universe. If `active_mask=True` and a value is NaN, the observation
      is treated as missing data (e.g., holiday or missing quote). If
      `active_mask=False`, the asset is inactive at that observation (e.g.,
      pre-listing or post-delisting period). `AssetPanel` applies each field's
      `inactive_policy` outside `active_mask`, commonly NaN for numeric fields and
      `MISSING=-1` for categorical fields.

    * The estimation universe is defined by the panel's `estimation_mask`, which is
      enforced as a subset of `active_mask`. It selects the active pairs used to fit
      cross-sectional statistics, factor-return regressions, benchmark and regression
      weights, alpha estimators and regime statistics. Other active pairs can still
      receive transformed values, exposures and forecasts, but they do not contribute to
      those fitted statistics.

    * The investment universe is defined by `X.columns` when `X` is provided. `X` is
      the skfolio API input for asset returns and is used by downstream workflows for
      validation, cross-validation, prediction, and scoring. If `X` is `None`, the
      investment universe is the full coverage universe.

    Fitting is performed on the coverage universe to use all available cross-sectional
    information, then the outputs are reduced to the investment universe. Within that
    investment universe, NaNs in fitted moments mark assets that are currently
    unavailable or not yet warmed up. Compatible downstream optimizers infer the
    investable subset from finite moments, solve on that subset and expand weights back
    with zero weight for unavailable assets.

    For the complete factor-model user guide, see
    :ref:`Factor Models <factor_models>`. For more details on the input panel format,
    see :ref:`Asset Data Representation <asset_data_representation>`.

    Parameters
    ----------
    factors : list of (str, BaseFactorExposure) tuples
        Named factor exposure estimators. Each tuple `(name, estimator)` defines a
        factor whose exposure is computed from the :class:`~skfolio.containers.AssetPanel`.
        Every estimator must have a `family` attribute (e.g., `"market"`, `"style"`,
        `"industry"`, `"country"`). Factor families are used for neutralization,
        zero-sum constraints and reporting.

    currency_factor : BaseFactorExposure, optional
        Optional factor exposure estimator for a multi-currency universe. This is
        expected to be a `OneHotCategoricalFactors` estimator on the point-in-time asset
        currency field, or an equivalent estimator returning one-hot currency exposures.

        For asset :math:`i` with primary currency :math:`C_i(t)` at observation
        :math:`t`, the exposure to currency factor :math:`c` is:

        .. math::

            x^{ccy}_{i,c}(t) =
            \begin{cases}
                1, & C_i(t) = c, \\
                0, & C_i(t) \ne c.
            \end{cases}

        Currency factor returns are supplied through `currency_excess_returns`. The
        base-currency excess return is:

        .. math::

            R^{excess,base}_i(t)
            = R^{excess,local}_i(t) + R^{ccy}_{C_i(t)}(t)

        where:

        .. math::

            R^{ccy}_{C_i(t)}(t)
            = R^{FX}_{C_i(t)}(t) + r^{cash}_{C_i(t)}(t)
            - r^{cash}_{base}(t) + R^{local}_i(t) R^{FX}_{C_i(t)}(t).

        Non-currency factor returns are estimated from local excess returns by
        cross-sectional regression. Currency factor returns are not estimated by that
        regression because they are observed FX series in the investor's numeraire.
        They are appended directly to the factor return distribution with family
        `"currency"`. The family name `"currency"` is reserved for this estimator.

    exposure_lag : int, default=1
        Number of periods by which factor exposures are lagged in the cross-sectional
        regression. Must be >= 1.

        The estimator follows skfolio's as-of time-indexing convention: all
        time-varying inputs at observation :math:`t` reflect information available up
        to and including the end of period :math:`t`. Point-in-time fields and derived
        values store the latest available value for observation :math:`t`. Returns
        stored at observation :math:`t` cover the period ending at :math:`t`, namely
        :math:`(t-1, t]`.

        Factor-return regressions estimate the factor returns realized over
        :math:`(t-1, t]`. The exposure matrix must therefore describe the assets before
        that return interval begins. `exposure_lag` selects that exposure date. The
        regression model is:

        .. math::

            R(t) = B(t - \ell)\,f(t) + \epsilon(t)

        where :math:`\ell` is `exposure_lag`. With the default :math:`\ell = 1`,
        returns over :math:`(t-1, t]` are regressed on exposures measured at
        :math:`t-1`.

    cs_regressor : BaseCSLinearModel, optional
        Cross-sectional regression estimator used to estimate factor returns from asset
        returns and lagged exposures. Must have `fit_intercept=False`. To model an
        intercept, include a :class:`~skfolio.factor_exposure.GlobalFactor`
        in `factors`. The default (`None`) is to use
        :class:`~skfolio.linear_model.CSLinearRegression`.

        .. note::
            Unlike factor exposures, which are typically winsorized and standardized
            by the exposure estimators, asset returns enter the cross-sectional
            regression unadjusted. Cleaning return data errors is an upstream
            responsibility, since the same returns drive benchmark weights, realized
            performance and downstream optimization. Winsorizing legitimate extreme
            returns would break the reconciliation of
            :math:`R = B\,f + \epsilon` and understate idiosyncratic risk for
            heavy-tailed assets; outlier influence is instead limited through
            exposure winsorization, `regression_mcap_power` and
            `inv_idio_variance_weight_shrinkage`. For bounded-influence estimation
            of factor returns, supply a robust `cs_regressor`.

    neutralize_against : dict of {str: list[str]}, optional
        Keys are factor names or family names to neutralize and values are lists of
        factor names or family names to neutralize against. When a key is a family name,
        every factor in that family is neutralized independently against the same
        targets. Entries are processed in insertion order: later entries see exposures
        already modified by earlier ones.

        For example:

        * `{"volatility": ["beta"]}`: orthogonalizes the volatility factor exposure
          against the beta factor exposure.
        * `{"momentum": ["industry"]}`: orthogonalizes the momentum factor exposure
          against all industry factor exposures.
        * `{"style": ["industry"]}`: orthogonalizes every style factor exposure against
          all industry factor exposures.

        .. note::
            When industry factors are one-hot encoded, neutralizing against industry is
            equivalent to applying within-industry demeaning. Using
            `FixedWeightedFactor` with `transform_by_group="industry"` and
            cross-sectional scoring that support group demeaning (e.g.
            `CSStandardScaler`) achieves the same result and is preferred for
            performance.

    constrained_families : list[tuple[str, str | None]], optional
        Zero-sum constraints applied within factor families. This is useful for one-hot
        families such as industry or country, whose exposures are collinear with the
        market factor (intercept) when all categories are included.

        Economically, the market factor captures the benchmark portfolio return.
        Constrained family factors capture relative effects around it. For example,
        constrained industry factors measure industry effects whose benchmark-weighted
        average is zero.

        Each tuple `(family, factor_to_drop)` specifies a family to reparameterize.
        The model removes one redundant factor and rewrites the family exposures in
        an equivalent full-rank basis. If `factor_to_drop` is `None`, the redundant
        factor is selected automatically to improve numerical conditioning.

        The same full-rank basis is used by downstream factor-level estimators, such
        as the factor prior and covariance estimator.

    benchmark_mcap_power : float, default=1.0
        Exponent applied to market capitalization to define the model benchmark weights:

        .. math::

            w_i \propto \mathrm{mcap}_i^p

        where :math:`p` is `benchmark_mcap_power`. These weights are used for weighted
        cross-sectional centering, zero-sum family constraints and the reference
        portfolio associated with the global factor.

        Typical choices:

        * `0.0`: equal-weighted
        * `0.5`: square-root market-cap-weighted
        * `1.0`: market-cap-weighted (default)

    regression_mcap_power : float, default=0.5
        Exponent applied to market capitalization to define the initial weights used in
        the cross-sectional regression:

        .. math::

            w_i \propto \mathrm{mcap}_i^p

        where :math:`p` is `regression_mcap_power`. These weights are used for
        regression estimation only and do not affect the model benchmark weights.
        When `inv_idio_variance_weight_shrinkage > 0`, these market-cap-based weights
        are blended with inverse-idiosyncratic-variance weights before the second-pass
        cross-sectional regression.

        Like the exposures, market caps are lagged by `exposure_lag`. Regression
        weights at observation :math:`t` use market caps from the selected exposure
        date, so the weights are fixed before the return interval being regressed.

        Typical choices:

        * `0.0`: equal-weighted
        * `0.5`: square-root market-cap-weighted (default)
        * `1.0`: market-cap-weighted

    inv_idio_variance_weight_shrinkage : float, default=0.0
        Shrinkage toward inverse-idiosyncratic-variance regression weights.
        When nonzero, the initial market-cap-based regression weights are blended with
        inverse idiosyncratic-variance weights in a two-pass WLS procedure. This
        approximates GLS by using estimated idiosyncratic variances as regression
        weights.

        The blended regression weight for each asset is:

        .. math::

            w_i = \lambda\,w_i^{\text{inv-var}} + (1 - \lambda)\,w_i^{\text{cap}}

        where :math:`\lambda` is `inv_idio_variance_weight_shrinkage`. Larger values
        give more weight to inverse idiosyncratic variance, while `0.0` uses only the
        market-cap-based regression weights. Must satisfy
        `0 <= inv_idio_variance_weight_shrinkage <= 1`.

        This is a standard two-step feasible GLS: the variances feeding the weights
        are estimated from the cap-weighted first-pass residuals, so the regression
        weights never depend on their own output, avoiding the feedback loop of
        recursive weighting schemes where a low estimated variance increases an
        asset's weight and, in turn, its influence on later residuals. The weights
        used at date :math:`t` are estimated from residuals up to :math:`t - 1` only.

    inv_idio_variance_max_weight_ratio : float, default=20
        Maximum ratio between any inverse-idiosyncratic-variance weight and the
        cross-sectional median inverse-idiosyncratic-variance weight. This caps extreme
        regression weights for assets with very low estimated idiosyncratic variance.

    factor_prior_estimator : BasePrior, optional
        Prior estimator for the factor return distribution: expected returns (factor
        premia), covariance and factor return scenarios. It is fitted on the estimated
        factor return time series.

        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior` with
        :class:`~skfolio.moments.EWMu` and
        :class:`~skfolio.moments.RegimeAdjustedEWCovariance`.

    alpha_estimator : BaseAlpha, optional
        Estimator producing asset-level expected returns (i.e. alpha forecast), from
        idiosyncratic returns and alpha signals computed from `AssetPanel` fields,
        before decomposition into factor-spanned alpha and orthogonal alpha.

        If `None` (default), expected asset returns are determined entirely by the
        expected factor returns estimated by `factor_prior_estimator` (factor premia).

        The alpha should be expressed in expected-return units when it is combined with
        expected factor returns or used by optimizers that trade it off against
        realized-return quantities (e.g., transaction costs, market impact, turnover
        constraints or return targets). Unitless cross-sectional scores are appropriate
        only when the downstream objective treats them purely as ordinal signals.

    spanned_alpha_shrinkage : float, default=1.0
        Shrinkage applied to spanned alpha. Alpha from `alpha_estimator` is split
        into spanned alpha and orthogonal alpha. The spanned alpha is blended with
        the expected factor returns:

        .. math::

            \mu^{\text{span}} =
            (1 - \lambda)\,\mu^{\text{span}}_{\text{alpha}}
            + \lambda\,\mu^{\text{span}}_{\text{factor}}

        where :math:`\lambda` is `spanned_alpha_shrinkage`.

        * `0`: use only the spanned alpha from `alpha_estimator`. When
          `alpha_estimator=None`, this sets the non-currency spanned expected return
          to zero.
        * `1` (default): use only the expected factor returns.
        * Between 0 and 1: partially shrink the spanned alpha toward the expected
          factor returns.

        Must satisfy `0 <= spanned_alpha_shrinkage <= 1`.

    orthogonal_alpha_confidence : float, default=1.0
        Confidence weight applied to orthogonal alpha. After alpha is decomposed into
        spanned alpha and orthogonal alpha, the orthogonal alpha is shrunk toward
        zero:

        .. math::

            \mu = \mu^{\text{span}} + c\,\mu^{\perp}

        where :math:`c` is `orthogonal_alpha_confidence`.

        * `0`: discard the orthogonal alpha.
        * `1` (default): use the orthogonal alpha as-is.
        * Between 0 and 1: partially shrink the orthogonal alpha toward zero.

        Must satisfy `0 <= orthogonal_alpha_confidence <= 1`.

        As an alternative to shrinking the point estimate, orthogonal uncertainty can be
        handled at the optimizer level with
        :class:`~skfolio.uncertainty_set.OrthogonalMuUncertaintySet` or
        :class:`~skfolio.uncertainty_set.OrthogonalCovarianceUncertaintySet`.

    idio_variance_estimator : BaseVariance, optional
        Variance estimator for idiosyncratic returns. It must support `partial_fit` so
        the model can recover per-asset variance estimates at each observation. These
        estimates are stored in `factor_model_.idio_variances`. The default (`None`) is
        :class:`~skfolio.moments.RegimeAdjustedEWVariance`.

    idio_corr_estimator : BaseCovariance, optional
        Estimator for idiosyncratic correlation thresholding. Although this parameter
        accepts a :class:`~skfolio.moments.BaseCovariance` estimator, only the
        correlation component of its output is retained.

        The estimator is fitted on idiosyncratic returns standardized by their
        contemporaneous idiosyncratic volatility from `idio_variance_estimator`. The
        resulting covariance matrix is converted to a correlation matrix, thresholded
        with `idio_corr_threshold` and recombined with the latest per-asset
        idiosyncratic variances to produce the final idiosyncratic covariance.

        By construction, idiosyncratic returns should be nearly uncorrelated after
        removing the factor structure, so the idiosyncratic covariance is diagonal by
        default. Correlation thresholding addresses cases where this assumption can
        break down, such as linked securities, multiple share classes, ADRs versus
        ordinary shares, or dual listings. Without it, optimizers may treat highly
        related securities as diversified sources of idiosyncratic risk.

        Variances and correlations are estimated separately because a single full
        covariance estimator would mix per-asset variance estimation with off-diagonal
        correlation noise. This keeps per-asset variances driven by
        `idio_variance_estimator` and applies correlation thresholding only where
        residual correlations are large enough to retain.

        The default (`None`) is :class:`~skfolio.moments.EWCovariance`. Correlation
        thresholding is used only when `idio_corr_threshold > 0`.

    idio_corr_threshold : float, default=0.0
        Absolute correlation threshold :math:`\tau` used for idiosyncratic correlation
        thresholding. Off-diagonal correlations with :math:`|\rho_{ij}| \le \tau` are
        set to zero. If `0` (default), correlation thresholding is disabled and the
        idiosyncratic covariance is diagonal.

    max_history : int, optional
        Maximum number of fitted observations to retain in time-series outputs and asset
        return scenarios. This applies to `return_distribution_.returns` and to fitted
        `factor_model_` histories such as `factor_returns`, `idio_returns`,
        `idio_variances`, `exposures`, and `regression_weights`.

        In incremental learning, setting `max_history` limits memory usage and keeps
        optimization with scenario-based risk measures, such as CVaR, computed on a
        rolling window of recent scenarios.

        * If `None` (default), all fitted observations are retained.
        * If an integer, only the last `max_history` fitted observations are retained.

    min_regression_assets : int, optional
        Minimum number of assets that must have all factor exposures finite and belong
        to the estimation universe at every post-warmup observation. If any observation
        falls below this threshold, a ValueError is raised. If `None` (default), the
        threshold is set automatically to `max(2 * n_factors, 30)` to reduce the risk
        of underspecified regressions and unstable factor-return estimates.

    n_jobs : int, default=1
        Number of parallel jobs used to compute factor exposures. Factors in the same
        dependency layer are computed in parallel using threads, avoiding copies of the
        underlying `AssetPanel`. Set to `-1` to use all available processors.

    Attributes
    ----------
    return_distribution_ : ReturnDistribution
        Fitted :class:`~skfolio.prior.ReturnDistribution` containing the expected
        asset returns, covariance matrix, asset return scenarios and reference to
        the fitted :class:`~skfolio.prior.FactorModel`.

    factor_model_ : FactorModel
        Fitted :class:`~skfolio.prior.FactorModel` containing factor exposures, factor
        returns, factor covariance, idiosyncratic returns, idiosyncratic variances and
        idiosyncratic covariance. Stored `exposures` follow the as-of time-indexing
        convention for each observation.

    cs_regressor_ : BaseCSLinearModel
        Fitted cross-sectional regression estimator.

    factor_prior_estimator_ : BasePrior
        Fitted factor prior estimator.

    alpha_estimator_ : BaseAlpha or None
        Fitted alpha estimator or `None` if no alpha estimator was provided.

    idio_variance_estimator_ : BaseVariance
        Fitted idiosyncratic variance estimator.

    idio_corr_estimator_ : BaseCovariance
        Fitted idiosyncratic correlation estimator.

    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets_,)
        Asset names in the coverage universe.

    n_features_in_ : int
        Number of assets in the investment universe.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Asset names in the investment universe. When `X` is `None`, equals
        `asset_names_`.

    Notes
    -----
    When the factor list includes a
    :class:`~skfolio.factor_exposure.GlobalFactor`, that factor acts as
    the regression intercept because its exposure is one for every asset. Its estimated
    return is close to the return of the market (defined as the benchmark-weighted
    portfolio on the estimation universe).

    When remaining factor exposures are centered so their benchmark-weighted average is
    zero, as produced by :class:`~skfolio.preprocessing.CSStandardScaler` and family
    constraints, they satisfy:

    .. math::

        \sum_i w_i^{\text{bench}}\,B_{ij} = 0 \quad \forall\; j \neq 0

    This centers these factors around the market, so the global factor captures the
    market return. If regression weights differ from benchmark weights, small tilts
    can remain:

    .. math::

        \sum_i w_i^{\text{reg}} B_{ij}

    These tilts explain why the global factor return may differ slightly from the
    exact market return.

    When `regression_mcap_power == benchmark_mcap_power` and
    `inv_idio_variance_weight_shrinkage == 0`, regression weights are proportional to
    benchmark weights, the tilts vanish and the identity becomes exact:

    .. math::

        \hat{f}_0(t) = \sum_i \hat{w}_i^{\text{bench}} R_i(t)

    References
    ----------
    .. [1] "The Elements of Quantitative Investing", Wiley Finance,
        Giuseppe A. Paleologo (2025).

    .. [2] "Active Portfolio Management: A Quantitative Approach for Producing Superior
       Returns and Controlling Risk", McGraw-Hill, Grinold & Kahn (1999).

    .. [3] "Multivariate Exponentially Weighted Moving Covariance Matrix",
        Technometrics, Hawkins & Maboudou-Tchao (2008).

    Examples
    --------
    Build a characteristics factor model from market, industry and style exposures:

    >>> from skfolio.datasets import make_synthetic_characteristics
    >>> from skfolio.descriptor import (
    ...     BookToPrice,
    ...     EWMomentum,
    ...     EWMarketBeta,
    ...     ForwardEarningsToPrice,
    ...     LogMarketCap,
    ... )
    >>> from skfolio.factor_exposure import (
    ...     DerivedFactor,
    ...     FixedWeightedFactor,
    ...     GlobalFactor,
    ...     OneHotCategoricalFactors,
    ... )
    >>> from skfolio.moments import EWMu, RegimeAdjustedEWCovariance
    >>> from skfolio.prior import CharacteristicsFactorModel, EmpiricalPrior
    >>>
    >>> characteristics = make_synthetic_characteristics()
    >>>
    >>> # Market and industry factors.
    >>> market_factor = GlobalFactor()
    >>> industry_factors = OneHotCategoricalFactors(
    ...     category="industry",
    ...     family="industry",
    ... )
    >>>
    >>> # Style factors built from descriptors.
    >>> beta_factor = FixedWeightedFactor(
    ...     descriptors=[("market_beta", EWMarketBeta(half_life=63))],
    ...     family="style",
    ...     transform_by_group="industry",
    ... )
    >>> momentum_factor = FixedWeightedFactor(
    ...     descriptors=[("momentum", EWMomentum(half_life=126, skip=21))],
    ...     family="style",
    ...     transform_by_group="industry",
    ... )
    >>> size_factor = FixedWeightedFactor(
    ...     descriptors=[("log_market_cap", LogMarketCap())],
    ...     family="style",
    ...     transform_by_group="industry",
    ... )
    >>> earnings_yield_factor = FixedWeightedFactor(
    ...     descriptors=[
    ...         ("book_to_price", BookToPrice()),
    ...         ("forward_earnings_to_price", ForwardEarningsToPrice()),
    ...     ],
    ...     weights=[0.5, 0.5],
    ...     family="style",
    ...     transform_by_group="industry",
    ... )
    >>>
    >>> # Style exposure derived from an existing factor.
    >>> non_linear_size_factor = DerivedFactor(
    ...     source="size",
    ...     func=lambda x: x**3,
    ...     transform_by_group="industry",
    ... )
    >>>
    >>> model = CharacteristicsFactorModel(
    ...     factors=[
    ...         ("market", market_factor),
    ...         ("industry", industry_factors),
    ...         ("beta", beta_factor),
    ...         ("momentum", momentum_factor),
    ...         ("size", size_factor),
    ...         ("earnings_yield", earnings_yield_factor),
    ...         ("non_linear_size", non_linear_size_factor),
    ...     ],
    ...     neutralize_against={
    ...         "momentum": ["industry"],
    ...         "non_linear_size": ["size"],
    ...     },
    ...     constrained_families=[("industry", None)],
    ...     factor_prior_estimator=EmpiricalPrior(
    ...         mu_estimator=EWMu(),
    ...         covariance_estimator=RegimeAdjustedEWCovariance(
    ...             half_life=40,
    ...             corr_half_life=60,
    ...         ),
    ...     ),
    ...     inv_idio_variance_weight_shrinkage=0.5,
    ...     n_jobs=-1,
    ... )
    >>> model.fit(characteristics=characteristics)
    >>>
    >>> # Inspect the fitted factor model and diagnostics.
    >>> fm = model.factor_model_
    >>> fm.summary()
    >>> fm.idio_calibration_summary()
    >>> fm.idio_vol_ic
    >>> fm.idio_tail_rate()
    >>> fm.factor_returns_df
    >>> fm.exposures_df

    Use :meth:`partial_fit` for online updates:

    >>> model.fit(characteristics=characteristics[:400])
    >>> for start in range(400, len(characteristics), 5):
    ...     model.partial_fit(characteristics=characteristics[start : start + 5])
    """

    # Request `characteristics` by default when this estimator is used inside a sklearn
    # metadata router so child meta-estimators do not need to call `set_fit_request`
    # and `set_partial_fit_request` explicitly.
    __metadata_request__fit: ClassVar[dict[str, bool]] = {"characteristics": True}
    __metadata_request__partial_fit: ClassVar[dict[str, bool]] = {
        "characteristics": True
    }

    factor_model_: FactorModel
    return_distribution_: ReturnDistribution
    factor_estimators_: dict[str, BaseFactorExposure]
    currency_factor_estimator_: BaseFactorExposure | None

    # Coverage universe from `characteristics`
    n_assets_: int
    asset_names_: StrArray

    # Investment universe from `X`, following scikit-learn API
    n_features_in_: int
    feature_names_in_: StrArray

    def __init__(
        self,
        *,
        factors: list[tuple[str, BaseFactorExposure]],
        currency_factor: BaseFactorExposure | None = None,
        exposure_lag: int = 1,
        cs_regressor: BaseCSLinearModel | None = None,
        neutralize_against: dict[str, list[str]] | None = None,
        constrained_families: list[tuple[str, str | None]] | None = None,
        benchmark_mcap_power: float = 1.0,
        regression_mcap_power: float = 0.5,
        inv_idio_variance_weight_shrinkage: float = 0.0,
        inv_idio_variance_max_weight_ratio: float = 20.0,
        factor_prior_estimator: BasePrior | None = None,
        alpha_estimator: BaseAlpha | None = None,
        spanned_alpha_shrinkage: float = 1.0,
        orthogonal_alpha_confidence: float = 1.0,
        idio_variance_estimator: BaseVariance | None = None,
        idio_corr_estimator: BaseCovariance | None = None,
        idio_corr_threshold: float = 0.0,
        max_history: int | None = None,
        min_regression_assets: int | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.factors = factors
        self.currency_factor = currency_factor
        self.exposure_lag = exposure_lag
        self.cs_regressor = cs_regressor
        self.neutralize_against = neutralize_against
        self.constrained_families = constrained_families
        self.benchmark_mcap_power = benchmark_mcap_power
        self.regression_mcap_power = regression_mcap_power
        self.inv_idio_variance_weight_shrinkage = inv_idio_variance_weight_shrinkage
        self.inv_idio_variance_max_weight_ratio = inv_idio_variance_max_weight_ratio
        self.factor_prior_estimator = factor_prior_estimator
        self.alpha_estimator = alpha_estimator
        self.spanned_alpha_shrinkage = spanned_alpha_shrinkage
        self.orthogonal_alpha_confidence = orthogonal_alpha_confidence
        self.idio_variance_estimator = idio_variance_estimator
        self.idio_corr_estimator = idio_corr_estimator
        self.idio_corr_threshold = idio_corr_threshold
        self.max_history = max_history
        self.min_regression_assets = min_regression_assets
        self.n_jobs = n_jobs

    def fit(
        self,
        X: pd.DataFrame | None = None,
        y=None,
        *,
        characteristics: AssetPanel,
        currency_excess_returns: pd.DataFrame | None = None,
        **fit_params,
    ) -> CharacteristicsFactorModel:
        """Fit the characteristics factor model.

        Resets all fitted state and estimates the full model pipeline on the provided
        data. For incremental updates, use :meth:`partial_fit`.

        Parameters
        ----------
        X : DataFrame of shape (n_observations, n_assets), optional
            Asset returns whose columns define the investment universe, following the
            standard skfolio estimator API. In this estimator, `X.columns` define the
            assets returned in `return_distribution_` and `factor_model_`.

            Factor estimation uses the `"returns"` field of `characteristics`, which can
            cover a broader point-in-time universe than `X`. This keeps the estimator
            compatible with skfolio pipelines, cross-validation, prediction and scoring,
            while allowing the factor model to use a wider coverage universe for
            estimation.

        y : Ignored
            Not used, present for API consistency by convention.

        characteristics : AssetPanel
            Point-in-time :class:`~skfolio.containers.AssetPanel` for the coverage
            universe. Must include `"returns"` and, when market-cap weighting is used,
            `"market_cap"`. The panel's `active_mask` identifies active
            asset-observation pairs within the coverage universe and `estimation_mask`
            identifies which active pairs contribute to estimation. For more details see
            :ref:`Asset Data Representation <asset_data_representation>`.

        currency_excess_returns : DataFrame, optional
            Currency excess returns. Required only when `currency_factor` is set.
            Columns must contain the unique currency factor names produced by
            `currency_factor`. Assets are mapped to these columns through the one-hot
            currency exposures.

        **fit_params : dict
            Parameters passed to underlying estimators. Only available when
            `enable_metadata_routing=True`, set with
            `sklearn.set_config(enable_metadata_routing=True)`. See
            :ref:`Metadata Routing User Guide <metadata_routing>` for more details.

        Returns
        -------
        self : CharacteristicsFactorModel
            Fitted estimator.

        Raises
        ------
        ValueError
            If the data does not contain enough observations to estimate the model after
            warmup and exposure-lag trimming.
        """
        self._reset()
        return self._fit(
            X,
            y,
            characteristics=characteristics,
            currency_excess_returns=currency_excess_returns,
            method="fit",
            **fit_params,
        )

    def partial_fit(
        self,
        X: pd.DataFrame | None = None,
        y=None,
        *,
        characteristics: AssetPanel,
        currency_excess_returns: pd.DataFrame | None = None,
        **fit_params,
    ) -> CharacteristicsFactorModel:
        """Incrementally fit the characteristics factor model.

        This method allows for streaming/online updates. Each call updates the internal
        state with new observations without resetting previously accumulated state. All
        sub-estimators (`factor_prior_estimator`, `idio_variance_estimator`, etc.)
        must implement `partial_fit` for this method to work.

        Parameters
        ----------
        X : DataFrame of shape (n_observations, n_assets), optional
            Asset returns whose columns define the investment universe, following the
            standard skfolio estimator API. In this estimator, `X.columns` define the
            assets returned in `return_distribution_` and `factor_model_`.

            Factor estimation uses the `"returns"` field of `characteristics`, which can
            cover a broader point-in-time universe than `X`. This keeps the estimator
            compatible with skfolio pipelines, cross-validation, prediction and scoring,
            while allowing the factor model to use a wider coverage universe for
            estimation.

        y : Ignored
            Not used, present for API consistency by convention.

        characteristics : AssetPanel
            Point-in-time :class:`~skfolio.containers.AssetPanel` for the coverage
            universe. Must include `"returns"` and, when market-cap weighting is used,
            `"market_cap"`. The panel's `active_mask` identifies active
            asset-observation pairs within the coverage universe and `estimation_mask`
            identifies which active pairs contribute to estimation. For more details see
            :ref:`Asset Data Representation <asset_data_representation>`.

        currency_excess_returns : DataFrame, optional
            Currency excess returns. Required only when `currency_factor` is set.
            Columns must contain the unique currency factor names produced by
            `currency_factor`. Assets are mapped to these columns through the one-hot
            currency exposures.

        **fit_params : dict
            Parameters passed to underlying estimators. Only available when
            `enable_metadata_routing=True`, set with
            `sklearn.set_config(enable_metadata_routing=True)`. See
            :ref:`Metadata Routing User Guide <metadata_routing>` for more details.

        Returns
        -------
        self : CharacteristicsFactorModel
            Fitted estimator.
        """
        return self._fit(
            X,
            y,
            characteristics=characteristics,
            currency_excess_returns=currency_excess_returns,
            method="partial_fit",
            **fit_params,
        )

    def _fit(
        self,
        X: pd.DataFrame | None = None,
        y=None,
        *,
        characteristics: AssetPanel,
        currency_excess_returns: pd.DataFrame | None = None,
        method: str,
        **fit_params,
    ) -> CharacteristicsFactorModel:
        """Core fitting logic shared by fit and partial_fit."""
        routed_params = skm.process_routing(self, method, **fit_params)

        first_call = not hasattr(self, _FITTED_ATTR)

        characteristics, currency_excess_returns = self._validate_data(
            X=X,
            characteristics=characteristics,
            currency_excess_returns=currency_excess_returns,
            first_call=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        self._attach_benchmark_weights(characteristics=characteristics)

        asset_returns = characteristics[_RETURNS]
        estimation_mask = characteristics.estimation_mask
        active_mask = characteristics.active_mask
        observations = characteristics.observations
        benchmark_weights = characteristics[_BENCHMARK_WEIGHTS]
        market_cap = characteristics[_MARKET_CAP] if self._need_market_cap else None

        if self.currency_factor is not None:
            ccy_exposures, ccy_factor_names, ccy_factor_families = (
                self._compute_currency_exposure(
                    characteristics=characteristics,
                    routed_params=routed_params,
                    method=method,
                )
            )
            currency_excess_returns = _validate_currency_excess_returns(
                currency_excess_returns=currency_excess_returns,
                observations=observations,
                currency_factor_names=ccy_factor_names,
            )
        else:
            ccy_exposures = None
            ccy_factor_names = None
            ccy_factor_families = None

        exposures, factor_names, factor_families = self._compute_factor_exposures(
            characteristics=characteristics, routed_params=routed_params, method=method
        )
        inactive_mask = ~active_mask
        if inactive_mask.any():
            exposures[inactive_mask] = np.nan
            if ccy_exposures is not None:
                ccy_exposures[inactive_mask] = np.nan

        if first_call:
            _validate_factor_names_and_families(
                factor_names=factor_names,
                factor_families=factor_families,
                name="factor model",
            )

        warmup_end = self._validate_exposure_warmup(
            asset_returns=asset_returns,
            exposures=exposures,
            estimation_mask=estimation_mask,
        )

        if warmup_end > 0:
            observations = observations[warmup_end:]
            exposures = exposures[warmup_end:]
            asset_returns = asset_returns[warmup_end:]
            estimation_mask = estimation_mask[warmup_end:]
            active_mask = active_mask[warmup_end:]
            benchmark_weights = benchmark_weights[warmup_end:]
            market_cap = market_cap[warmup_end:] if market_cap is not None else None
            ccy_exposures = (
                ccy_exposures[warmup_end:] if ccy_exposures is not None else None
            )
            currency_excess_returns = (
                currency_excess_returns.iloc[warmup_end:]
                if currency_excess_returns is not None
                else None
            )

        # Neutralize style exposures in-place against other factors/families. Runs
        # before the family-constraint basis change so that one-hot families are
        # neutralized against the full exposure set.
        if self.neutralize_against:
            _neutralize_exposures(
                cs_regressor=sk.clone(self.cs_regressor_),
                neutralize_against=self.neutralize_against,
                exposures=exposures,
                benchmark_weights=benchmark_weights,
                factor_names=factor_names,
                factor_families=factor_families,
            )

        # Apply zero-sum constraints within factor families, dropping one factor per
        # family so that the benchmark-weighted average factor return is zero.
        if self.constrained_families is not None:
            # Unless provided by the user, the factor to drop inside a family constraint
            # is determined by compute_family_constraint_basis based on a methodology
            # that improves numerical conditioning. When calling partial_fit, the chosen
            # factor needs to remain the same between all incremental calls so we cache it
            basis, self._constrained_families = compute_family_constraint_basis(
                constrained_families=(
                    self._constrained_families
                    if self._constrained_families is not None
                    else self.constrained_families
                ),
                factor_exposures=exposures,
                benchmark_weights=benchmark_weights,
                factor_names=factor_names,
                factor_families=factor_families,
            )
            self._family_constraint_basis = basis
            factor_names_reduced = basis.reduced_factor_names(factor_names)
            factor_families_reduced = basis.reduced_factor_names(factor_families)
            exposures_reduced = basis.reduce_exposures(exposures)
        else:
            basis = None
            factor_names_reduced = factor_names
            factor_families_reduced = factor_families
            exposures_reduced = exposures

        # Apply the exposure lag used in cross-sectional regression. Buffers carry
        # the last `exposure_lag` rows across `partial_fit` calls so no return
        # observations are lost at batch boundaries.
        lagged_exposures_reduced, lag_trim, self._buffer_exposures_reduced = (
            _lag_with_buffer(
                exposures_reduced, self._buffer_exposures_reduced, self.exposure_lag
            )
        )

        # The regression coefficients at observation t are coordinates in the reduced
        # basis built from the lagged exposures, so reconstructing full-basis factor
        # returns must use the ratios c(t - lag).
        if basis is not None:
            lagged_constraint_ratios, _, self._buffer_constraint_ratios = (
                _lag_with_buffer(
                    basis.constraint_ratios,
                    self._buffer_constraint_ratios,
                    self.exposure_lag,
                )
            )
            lagged_basis = basis.with_constraint_ratios(lagged_constraint_ratios)
        else:
            lagged_basis = None

        # The market cap used for regression weighting is lagged like the exposures.
        # The current observation cap is endogenous to the regressand so weighting
        # observation t by mcap(t) embeds the return being regressed, correlating
        # weights with residuals (same-observation winners get up-weighted, losers
        # down-weighted, inflating R2).
        if self.regression_mcap_power != 0:
            lagged_market_cap, _, self._buffer_market_cap = _lag_with_buffer(
                market_cap, self._buffer_market_cap, self.exposure_lag
            )
        else:
            lagged_market_cap = None

        if lagged_exposures_reduced.shape[0] == 0:
            prefix = (
                "The first `partial_fit` call must contain enough data to estimate "
                "at least one regression observation. "
                if first_call and method == "partial_fit"
                else ""
            )
            raise ValueError(
                f"{prefix}Not enough observations to estimate the factor model after "
                f"exposure lag. `exposure_lag={self.exposure_lag}` requires at least "
                f"{self.exposure_lag + 1} post-warmup observations. Provide more "
                "observations in the first batch, reduce descriptor warmup parameters, "
                "or reduce `exposure_lag`."
            )

        # On the first call the lag consumes `exposure_lag` leading rows.
        # Trim all parallel arrays so everything stays aligned.
        if lag_trim > 0:
            observations = observations[lag_trim:]
            asset_returns = asset_returns[lag_trim:]
            estimation_mask = estimation_mask[lag_trim:]
            active_mask = active_mask[lag_trim:]
            exposures = exposures[lag_trim:]
            exposures_reduced = exposures_reduced[lag_trim:]
            benchmark_weights = benchmark_weights[lag_trim:]
            ccy_exposures = (
                ccy_exposures[lag_trim:] if ccy_exposures is not None else None
            )
            currency_excess_returns = (
                currency_excess_returns.iloc[lag_trim:]
                if currency_excess_returns is not None
                else None
            )
            basis = basis[lag_trim:] if basis is not None else None

        regression_eligible_mask = (
            np.isfinite(asset_returns)
            & estimation_mask
            & np.all(np.isfinite(lagged_exposures_reduced), axis=2)
        )
        self._validate_regression_coverage(
            regression_eligible_mask=regression_eligible_mask,
            asset_returns=asset_returns,
            lagged_exposures=lagged_exposures_reduced,
            estimation_mask=estimation_mask,
            observations=observations,
            factor_names=factor_names_reduced,
        )

        # Cross-sectional regression
        idio_returns, factor_returns_reduced, regression_weights = (
            self._cross_sectional_regression(
                lagged_exposures=lagged_exposures_reduced,
                asset_returns=asset_returns,
                lagged_market_cap=lagged_market_cap,
                regression_eligible_mask=regression_eligible_mask,
                estimation_mask=estimation_mask,
                active_mask=active_mask,
                routed_params=routed_params.cs_regressor.fit,
            )
        )

        # Combine regression-estimated local factors with directly observed currency
        # factors. The result stays in the reduced basis when constraints are active.
        _, n_reduced_factors = factor_returns_reduced.shape

        if ccy_exposures is not None:
            ccy_factor_returns = currency_excess_returns.loc[
                observations, ccy_factor_names
            ].to_numpy(dtype=float, copy=False)
            if not np.all(np.isfinite(ccy_factor_returns)):
                raise ValueError(
                    "`currency_excess_returns` must contain only finite values "
                    "for the fitted observations and currency factors."
                )
            factor_returns_reduced_with_ccy = np.concatenate(
                [factor_returns_reduced, ccy_factor_returns], axis=1
            )
            factor_names_reduced_with_ccy = np.concatenate(
                [factor_names_reduced, ccy_factor_names]
            )
            loading_matrix_reduced_with_ccy = np.concatenate(
                [exposures_reduced[-1], ccy_exposures[-1]], axis=1
            )
            basis_with_ccy = (
                basis.append_passthrough_factors(len(ccy_factor_names))
                if basis is not None
                else None
            )
            lagged_basis_with_ccy = (
                lagged_basis.append_passthrough_factors(len(ccy_factor_names))
                if lagged_basis is not None
                else None
            )
        else:
            loading_matrix_reduced_with_ccy = exposures_reduced[-1]
            factor_returns_reduced_with_ccy = factor_returns_reduced
            factor_names_reduced_with_ccy = factor_names_reduced
            basis_with_ccy = basis
            lagged_basis_with_ccy = lagged_basis

        # Estimate factor return distribution: mu, cov and return scenarios
        factor_dist_reduced_with_ccy = self._compute_factor_returns_dist(
            factor_returns=factor_returns_reduced_with_ccy,
            factor_names=factor_names_reduced_with_ccy,
            observations=observations,
            routed_params=routed_params,
            first_call=first_call,
        )

        # Per-observation idiosyncratic variance estimates (n_observations, n_assets)
        idio_variances = self._compute_idio_variances(
            idio_returns=idio_returns,
            estimation_mask=estimation_mask,
            active_mask=active_mask,
            routed_params=routed_params,
        )

        # Latest idiosyncratic covariance estimate. If `idio_corr_threshold == 0`, uses
        # the latest per-asset variances and returns the diagonal of shape (n_assets,).
        # Otherwise, estimates a sparse idio covariance via correlation thresholding and
        # returns the full matrix of shape (n_assets, n_assets).
        idio_cov = self._compute_idio_covariance(
            idio_returns=idio_returns,
            idio_variances=idio_variances,
            estimation_mask=estimation_mask,
            active_mask=active_mask,
            routed_params=routed_params,
            first_call=first_call,
        )

        # Alpha forecast from user provided `alpha_estimator`. Default is zeros.
        alpha = self._compute_alpha(
            characteristics=characteristics,
            idio_returns=idio_returns,
            idio_variances=idio_variances,
            regression_weights=regression_weights,
            exposures=exposures_reduced,
            factor_names=factor_names_reduced,
            factor_families=factor_families_reduced,
            routed_params=routed_params,
            first_call=first_call,
        )

        # Decompose alpha into expected factor returns and an orthogonal residual.
        # The spanned asset alpha is rebuilt later after shrinkage.
        factor_mu_reduced, orthogonal_alpha = self._decompose_alpha(
            alpha=alpha,
            exposure=exposures_reduced[[-1]],
            regression_weights=regression_weights[[-1]],
            routed_params=routed_params.cs_regressor.fit,
        )

        # Reduce to the investment universe
        idx = self._investment_idx_in_coverage
        if idx is not None:
            orthogonal_alpha = orthogonal_alpha[idx]
            exposures_reduced = exposures_reduced[:, idx]
            exposures = exposures[:, idx]
            loading_matrix_reduced_with_ccy = loading_matrix_reduced_with_ccy[idx]
            idio_returns = idio_returns[:, idx]
            idio_variances = idio_variances[:, idx]
            regression_weights = regression_weights[:, idx]
            benchmark_weights = benchmark_weights[:, idx]
            active_mask = active_mask[:, idx]
            idio_cov = (
                idio_cov[idx] if idio_cov.ndim == 1 else idio_cov[np.ix_(idx, idx)]
            )
            ccy_exposures = ccy_exposures[:, idx] if ccy_exposures is not None else None

        _validate_covariance_readiness(
            factor_covariance=factor_dist_reduced_with_ccy.covariance,
            latest_idio_variances=idio_variances[-1],
        )

        # Shrink spanned mu towards the factor prior projection
        factor_mu_prior_reduced = factor_dist_reduced_with_ccy.mu[:n_reduced_factors]
        factor_mu_reduced = (
            factor_mu_prior_reduced * self.spanned_alpha_shrinkage
            + factor_mu_reduced * (1 - self.spanned_alpha_shrinkage)
        )
        factor_mu_reduced_with_ccy = factor_dist_reduced_with_ccy.mu.copy()
        factor_mu_reduced_with_ccy[:n_reduced_factors] = factor_mu_reduced

        spanned_alpha = exposures_reduced[-1] @ factor_mu_reduced

        # Shrink orthogonal alpha towards 0 as a function of user confidence
        orthogonal_alpha *= self.orthogonal_alpha_confidence

        # Reassemble both spanned and orthogonal into the assets' expected returns vector
        alpha = spanned_alpha + orthogonal_alpha
        if ccy_exposures is not None:
            alpha += ccy_exposures[-1] @ factor_mu_reduced_with_ccy[n_reduced_factors:]

        # Asset covariance (n_assets, n_assets)
        factor_cov_reduced_with_ccy = factor_dist_reduced_with_ccy.covariance
        asset_cov = loading_matrix_reduced_with_ccy @ (
            factor_cov_reduced_with_ccy @ loading_matrix_reduced_with_ccy.T
        )
        if idio_cov.ndim == 1:
            asset_cov[np.diag_indices_from(asset_cov)] += idio_cov
        else:
            asset_cov += idio_cov

        if ccy_exposures is not None:
            exposures = np.concatenate([exposures, ccy_exposures], axis=2)
            factor_names = np.concatenate([factor_names, ccy_factor_names])
            factor_families = np.concatenate([factor_families, ccy_factor_families])
            if first_call:
                _validate_factor_names_and_families(
                    factor_names=factor_names,
                    factor_families=factor_families,
                    name="factor model",
                )

        if basis_with_ccy is not None:
            # The regression at observation t uses exposures from t - lag, so the
            # dropped factor's return must be recovered with the constraint ratios from
            # t - lag as well (lagged basis). This guarantees that exposures(t - lag)
            # @ factor_returns(t) + idio_returns(t) reproduces asset returns exactly.
            # Factor mu and covariance describe the next period (forcast) and use the
            # current loading matrix with the current (unlagged) ratios.
            factor_returns = lagged_basis_with_ccy.expand_factor_returns(
                factor_returns_reduced_with_ccy
            )
            factor_mu = basis_with_ccy.expand_factor_mu(factor_mu_reduced_with_ccy)
            factor_cov = basis_with_ccy.expand_factor_covariance(
                factor_cov_reduced_with_ccy
            )
        else:
            factor_returns = factor_returns_reduced_with_ccy
            factor_mu = factor_mu_reduced_with_ccy
            factor_cov = factor_cov_reduced_with_ccy

        standardized_idio_returns = _compute_standardized_idio_returns(
            idio_returns=idio_returns,
            idio_variances=idio_variances,
            active_mask=active_mask,
        )

        history_arrays = dict(
            observations=observations,
            factor_returns=factor_returns,
            idio_returns=idio_returns,
            idio_variances=idio_variances,
            standardized_idio_returns=standardized_idio_returns,
            exposures=exposures,
            regression_weights=regression_weights,
            benchmark_weights=benchmark_weights,
            active_mask=active_mask,
        )
        if basis is not None:
            history_arrays["family_constraint_ratios"] = basis.constraint_ratios

        self._accumulate_history(**history_arrays)
        history = self._get_history()

        if self.constrained_families is not None:
            accumulated_basis = self._family_constraint_basis.with_constraint_ratios(
                history["family_constraint_ratios"]
            )
            if ccy_exposures is not None:
                accumulated_basis = accumulated_basis.append_passthrough_factors(
                    len(ccy_factor_names)
                )
        else:
            accumulated_basis = None

        # `factor_dist_reduced_with_ccy.returns` contains the factor-return scenarios
        # produced by the factor prior estimator. These scenarios are mapped through the
        # latest loading matrix, then combined with calibrated idiosyncratic scenarios
        # to build `ReturnDistribution.returns` for downstream optimizers that use
        # scenario-based risk measures such as CVaR. By contrast,
        # `FactorModel.factor_returns` stores the realized historical factor returns
        # estimated by the cross-sectional regressions. They match
        # `factor_dist_reduced_with_ccy.returns` only when the factor prior estimator
        # preserves historical scenarios (e.g., `EmpiricalPrior`) and there is no family
        # constraint.
        asset_return_scenarios, sample_weight = _assemble_asset_return_scenarios(
            factor_return_scenarios=factor_dist_reduced_with_ccy.returns,
            loading_matrix=loading_matrix_reduced_with_ccy,
            standardized_idio_returns=history["standardized_idio_returns"],
            latest_active_mask=active_mask[-1],
            latest_idio_variances=idio_variances[-1],
            sample_weight=factor_dist_reduced_with_ccy.sample_weight,
        )

        self.factor_model_ = FactorModel(
            observations=history["observations"],
            asset_names=self.feature_names_in_,
            factor_names=factor_names,
            factor_families=factor_families,
            loading_matrix=exposures[-1],
            exposures=history["exposures"],
            factor_returns=history["factor_returns"],
            factor_mu=factor_mu,
            factor_covariance=factor_cov,
            idio_returns=history["idio_returns"],
            idio_variances=history["idio_variances"],
            idio_mu=orthogonal_alpha,
            idio_covariance=idio_cov,
            exposure_lag=self.exposure_lag,
            regression_weights=history["regression_weights"],
            benchmark_weights=history["benchmark_weights"],
            family_constraint_basis=accumulated_basis,
        )

        # Assets
        self.return_distribution_ = ReturnDistribution(
            mu=alpha,
            covariance=asset_cov,
            returns=asset_return_scenarios,
            sample_weight=sample_weight,
            factor_model=self.factor_model_,
        )

        return self

    @property
    def named_factors(self) -> sku.Bunch:
        """Dictionary for accessing factors by name.

        Returns
        -------
        :class:`~sklearn.utils.Bunch`
        """
        return sku.Bunch(**dict(self.factors))

    def set_params(self, **params) -> CharacteristicsFactorModel:
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with `get_params()`. Note that you
        can directly set the parameters of the factor estimators contained in
        `factors`.

        Parameters
        ----------
        **params : keyword arguments
            Specific parameters using e.g.
            `set_params(parameter_name=new_value)`. In addition to setting the
            parameters of the estimator, the individual factor estimators can
            also be set, or can be removed by setting them to 'drop'.

        Returns
        -------
        self : object
            Estimator instance.
        """
        super()._set_params("factors", **params)
        return self

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Get the parameters of this estimator.

        Returns the parameters given in the constructor as well as the
        factor estimators contained within the `factors` parameter.

        Parameters
        ----------
        deep : bool, default=True
            Setting it to True gets the various estimators and the parameters
            of the estimators as well.

        Returns
        -------
        params : dict
            Parameter and estimator names mapped to their values or parameter
            names mapped to their values.
        """
        return super()._get_params("factors", deep=deep)

    def get_metadata_routing(self) -> skm.MetadataRouter:
        """Get metadata routing for this estimator.

        Returns
        -------
        routing : MetadataRouter
            Metadata routing configuration.
        """
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add_self_request(self)
            .add(
                cs_regressor=self.cs_regressor,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="fit"),
            )
            .add(
                factor_prior_estimator=self.factor_prior_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
            .add(
                alpha_estimator=self.alpha_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
            .add(
                idio_variance_estimator=self.idio_variance_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="partial_fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
            .add(
                idio_corr_estimator=self.idio_corr_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        )
        if self.currency_factor is not None:
            router.add(
                currency_factor=self.currency_factor,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
        for name, factor in self.factors:
            router.add(
                **{name: factor},
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )

        return router

    @property
    def _need_market_cap(self) -> bool:
        """Whether market capitalization is required for weighting.

        Market cap is needed only when at least one weighting scheme uses a non-zero
        power. With both powers at zero, weighting is equal-weighted and the
        `market_cap` field is not required.
        """
        return self.regression_mcap_power != 0 or self.benchmark_mcap_power != 0

    def _validate_data(
        self,
        X: pd.DataFrame | None,
        characteristics: AssetPanel,
        currency_excess_returns: pd.DataFrame | None,
        first_call: bool,
    ) -> tuple[AssetPanel, pd.DataFrame | None]:
        """Validate and prepare input data.

        Validates the :class:`~skfolio.containers.AssetPanel` and optional `X` input.
        On the first call, this method also sets the investment-universe attributes
        used by the standard skfolio estimator API, following scikit-learn convention:

        * `n_features_in_`: number of assets in the investment universe.
        * `feature_names_in_`: asset names in the investment universe.
        * `_investment_idx_in_coverage`: integer positions mapping `X.columns` to
          `characteristics.asset_names`, or `None` when `X` is not provided.

        When `X` is provided, `X.columns` define both the assets and the order of the
        investment universe. Final outputs are reduced and reordered accordingly. When
        `X` is `None`, the investment universe is the full coverage universe.

        Parameters
        ----------
        X : DataFrame or None
            Asset returns for the investment universe, or `None` to use the full
            coverage universe.

        characteristics : AssetPanel
            Point-in-time panel for the coverage universe.

        currency_excess_returns : DataFrame or None
            Currency excess returns, if applicable.

        first_call : bool
            Whether this is the first `fit` or `partial_fit` call. Investment-universe
            attributes are initialized only on the first call and validated by
            `sklearn.utils.validation.validate_data` on subsequent calls.

        Returns
        -------
        characteristics : AssetPanel
            Validated shallow copy of the input panel.

        currency_excess_returns : DataFrame or None
            Validated currency excess returns, or `None` when currency factors are not
            used.

        Raises
        ------
        ValueError
            If `X` is not a DataFrame, required fields are missing, required panel
            values are invalid, all estimation returns are missing for any observation,
            or currency inputs are inconsistent.
        """
        if X is not None and not isinstance(X, pd.DataFrame):
            raise ValueError("`X` must be a pd.DataFrame or None.")

        required_fields = [_RETURNS]
        finite_fields = None
        strictly_positive_fields = None
        if self._need_market_cap:
            required_fields.append(_MARKET_CAP)
            finite_fields = [_MARKET_CAP]
            strictly_positive_fields = [_MARKET_CAP]

        characteristics = validate_asset_panel(
            self,
            asset_panel=characteristics,
            required_fields=required_fields,
            reserved_fields=[
                _BENCHMARK_WEIGHTS,
                _EXPOSURES,
                _IDIO_RETURNS,
                _IDIO_VARIANCES,
                _REGRESSION_WEIGHTS,
            ],
            finite_when_active=finite_fields,
            strictly_positive_when_active=strictly_positive_fields,
            reset=first_call,
            copy=True,  # shallow copy
        )

        # Validate that no observation has all non-finite returns in the estimation
        # universe (e.g. common holidays across all estimation assets), which would make
        # cross-sectional regression impossible.
        estimation_finite = (
            np.isfinite(characteristics[_RETURNS]) & characteristics.estimation_mask
        )
        all_nan_obs = ~estimation_finite.any(axis=1)
        if all_nan_obs.any():
            bad_obs = characteristics.observations[all_nan_obs]
            raise ValueError(
                f"Found {int(all_nan_obs.sum())} observation(s) where all returns in"
                f" the estimation universe are non-finite (e.g. common holidays), which"
                f" prevents cross-sectional regression: {bad_obs.tolist()}"
            )

        if X is not None:
            skv.validate_data(self, X, reset=first_call, ensure_all_finite=False)
            if first_call:
                asset_idx = {v: i for i, v in enumerate(self.asset_names_)}
                try:
                    self._investment_idx_in_coverage = np.array(
                        [asset_idx[x] for x in X.columns], dtype=int
                    )
                except KeyError as e:
                    raise ValueError(
                        f"Asset {e.args[0]!r} from `X` is missing from"
                        f" `characteristics`."
                    ) from e
        else:
            if first_call:
                self.n_features_in_ = characteristics.n_assets
                self.feature_names_in_ = np.asarray(characteristics.asset_names)
                self._investment_idx_in_coverage = None

        if self.currency_factor is None:
            if currency_excess_returns is not None:
                raise ValueError(
                    "`currency_excess_returns` can only be provided when "
                    "`currency_factor` is set."
                )
        else:
            if currency_excess_returns is None:
                raise ValueError(
                    "`currency_excess_returns` is required when "
                    "`currency_factor` is set."
                )
            if not isinstance(currency_excess_returns, pd.DataFrame):
                raise ValueError("`currency_excess_returns` must be a pd.DataFrame.")
            if currency_excess_returns.shape[0] != characteristics.shape[0]:
                raise ValueError(
                    "`characteristics` and `currency_excess_returns` must have the "
                    "same number of observations."
                )
            if not np.array_equal(
                np.asarray(currency_excess_returns.index),
                characteristics.observations,
            ):
                raise ValueError(
                    "`currency_excess_returns.index` must match "
                    "`characteristics.observations`."
                )

        return characteristics, currency_excess_returns

    def _validate_factors(self) -> tuple[list[str], list[BaseFactorExposure]]:
        """Validate the `factors` parameter.

        Returns
        -------
        names : list[str]
            The list of factor names.

        estimators : list[BaseFactorExposure]
            The list of factor estimators.
        """
        if self.factors is None or len(self.factors) == 0:
            raise ValueError(
                "Invalid 'factors' attribute, 'factors' should be a list"
                " of (string, factor) tuples."
            )
        names, factors = zip(*self.factors, strict=True)

        self._validate_names(names)

        for name, factor in zip(names, factors, strict=True):
            if not hasattr(factor, "family") or not isinstance(factor.family, str):
                raise ValueError(
                    f"Factor '{name}' is missing the required 'family' parameter. "
                    f"All factor estimators must define a 'family' string "
                    f"(e.g., 'market', 'style', 'industry')."
                )
            if factor.family == _CURRENCY:
                raise ValueError(
                    f"Factor family {_CURRENCY!r} is reserved for `currency_factor`."
                )

        return names, factors

    def _validate_exposure_warmup(
        self,
        asset_returns: FloatArray,
        exposures: FloatArray,
        estimation_mask: BoolArray,
    ) -> int:
        """Validate descriptor warmup and return the number of leading cold
        observations.

        An observation is cold when no asset has finite same-date exposures, finite
        returns and membership in the estimation universe.

        Returns
        -------
        warmup_end : int
            Number of leading "cold" observations to trim.

        Raises
        ------
        ValueError
            If all observations are cold.
        """
        n_eligible_assets = (
            np.all(np.isfinite(exposures), axis=2)
            & np.isfinite(asset_returns)
            & estimation_mask
        ).sum(axis=1)

        is_cold = n_eligible_assets == 0
        if is_cold.all():
            raise ValueError(
                "Not enough observations to estimate the factor model. All observations"
                " were consumed by descriptor warmup and/or exposure lag. Provide more"
                " observations, reduce descriptor warmup parameters, or reduce"
                " `exposure_lag`."
            )

        warmup_end = int((~is_cold).argmax())
        return warmup_end

    def _validate_regression_coverage(
        self,
        regression_eligible_mask: BoolArray,
        asset_returns: FloatArray,
        lagged_exposures: FloatArray,
        estimation_mask: BoolArray,
        observations: AnyArray,
        factor_names: StrArray,
    ) -> None:
        """Validate lagged regression coverage for every fitted observation."""
        _, _, n_factors = lagged_exposures.shape

        min_regression_assets = (
            self.min_regression_assets
            if self.min_regression_assets is not None
            else max(2 * n_factors, 30)
        )

        n_eligible_assets = regression_eligible_mask.sum(axis=1)
        insufficient = n_eligible_assets < min_regression_assets

        if not insufficient.any():
            return

        failing_obs = observations[insufficient]
        failing_counts = n_eligible_assets[insufficient]
        failing_returns = asset_returns[insufficient]
        failing_exposures = lagged_exposures[insufficient]
        failing_estimation_mask = estimation_mask[insufficient]

        n_estimation_assets = failing_estimation_mask.sum(axis=1)

        n_nan_returns = (~np.isfinite(failing_returns) & failing_estimation_mask).sum(
            axis=1
        )

        n_nan_exposures_by_factor = (
            ~np.isfinite(failing_exposures) & failing_estimation_mask[:, :, np.newaxis]
        ).sum(axis=1)

        avg_n_estimation_assets = n_estimation_assets.mean()
        avg_n_nan_returns = n_nan_returns.mean()
        avg_n_nan_exposures_by_factor = n_nan_exposures_by_factor.mean(axis=0)

        worst_factor_idx = np.argsort(avg_n_nan_exposures_by_factor)[::-1][:5]
        worst_factor_parts = [
            f"{factor_names[i]} ({avg_n_nan_exposures_by_factor[i]:.1f})"
            for i in worst_factor_idx
            if avg_n_nan_exposures_by_factor[i] > 0
        ]

        diagnostics_msg = (
            f"Avg across failing observations: "
            f"{avg_n_estimation_assets:.0f} estimation-universe assets, "
            f"{avg_n_nan_returns:.1f} with NaN returns"
        )

        if worst_factor_parts:
            diagnostics_msg += (
                ", top NaN-exposure factors by avg count: "
                f"[{', '.join(worst_factor_parts)}]"
            )

        n_fail = int(insufficient.sum())
        min_count = int(failing_counts.min())

        max_examples = 10
        examples = dict(
            zip(
                failing_obs[:max_examples].tolist(),
                failing_counts[:max_examples].tolist(),
                strict=True,
            )
        )
        n_omitted = n_fail - len(examples)

        examples_msg = f"{examples}"
        if n_omitted > 0:
            examples_msg += f" ({n_omitted} more omitted)"

        raise ValueError(
            f"{n_fail} observation(s) after warmup have fewer than "
            f"min_regression_assets={min_regression_assets} regression-eligible "
            f"assets. Minimum eligible assets observed: {min_count}. "
            f"Examples: {examples_msg}. "
            f"{diagnostics_msg}. "
            f"Consider lowering `min_coverage` in factor exposure estimators, "
            f"adding descriptors with broader coverage, or checking data quality "
            f"in the AssetPanel."
        )

    def _validate_params(self) -> None:
        """Validate hyperparameters."""
        _validate_positive_integer(self.exposure_lag, "exposure_lag")
        _validate_non_negative_real(self.benchmark_mcap_power, "benchmark_mcap_power")
        _validate_non_negative_real(self.regression_mcap_power, "regression_mcap_power")
        _validate_unit_interval(
            self.inv_idio_variance_weight_shrinkage,
            "inv_idio_variance_weight_shrinkage",
        )
        _validate_unit_interval(self.spanned_alpha_shrinkage, "spanned_alpha_shrinkage")
        _validate_unit_interval(
            self.orthogonal_alpha_confidence, "orthogonal_alpha_confidence"
        )
        _validate_unit_interval(self.idio_corr_threshold, "idio_corr_threshold")
        _validate_positive_real(
            self.inv_idio_variance_max_weight_ratio,
            "inv_idio_variance_max_weight_ratio",
        )

        if self.min_regression_assets is not None:
            _validate_positive_integer(
                self.min_regression_assets, "min_regression_assets"
            )

        if self.max_history is not None:
            _validate_positive_integer(self.max_history, "max_history")

        if self.constrained_families is not None:
            constrained_families = {x[0] for x in self.constrained_families}
            if _CURRENCY in constrained_families:
                raise ValueError(
                    f"constrained_families cannot target the reserved "
                    f"{_CURRENCY!r} family."
                )

    def _initialize(self) -> None:
        """Initialize internal state on the first call.

        Validates and clones all sub-estimators and sets up internal buffers. Called
        once during the first `fit` or `partial_fit` invocation.
        """
        self.cs_regressor_ = check_estimator(
            self.cs_regressor,
            default=CSLinearRegression(),
            check_type=BaseCSLinearModel,
        )
        if getattr(self.cs_regressor_, "fit_intercept", False):
            raise ValueError("`cs_regressor.fit_intercept` must be set to `False`.")

        self.factor_prior_estimator_ = check_estimator(
            self.factor_prior_estimator,
            default=EmpiricalPrior(
                mu_estimator=EWMu(),
                covariance_estimator=RegimeAdjustedEWCovariance(),
            ),
            check_type=BasePrior,
        )

        self.alpha_estimator_ = check_estimator(
            self.alpha_estimator, default=None, check_type=BaseAlpha
        )

        self.idio_variance_estimator_ = check_estimator(
            self.idio_variance_estimator,
            default=RegimeAdjustedEWVariance(),
            check_type=BaseVariance,
        )
        if not hasattr(self.idio_variance_estimator_, "partial_fit"):
            raise TypeError(
                f"idio_variance_estimator "
                f"({type(self.idio_variance_estimator_).__name__}) does not "
                f"implement `partial_fit`. Use an estimator with incremental "
                f"learning support (e.g., RegimeAdjustedEWVariance)."
            )

        # Independent clone for inverse-variance regression weights
        if self.inv_idio_variance_weight_shrinkage != 0:
            self._idio_variance_estimator_inv_var = sk.clone(
                self.idio_variance_estimator_
            )

        self.idio_corr_estimator_ = check_estimator(
            self.idio_corr_estimator,
            # Nearest cov is handled in _compute_idio_covariance
            default=EWCovariance(nearest=False),
            check_type=BaseCovariance,
        )

        names, factors = self._validate_factors()

        self.named_factor_estimators_ = {
            name: sk.clone(estimator)
            for name, estimator in zip(names, factors, strict=True)
        }

        if self.currency_factor is not None:
            self.currency_factor_estimator_ = sk.clone(self.currency_factor)

        # Buffers carrying the last `exposure_lag` rows across partial_fit calls,
        # used to lag exposures, family-constraint ratios and market cap at batch
        # boundaries
        self._buffer_exposures_reduced = None
        self._buffer_constraint_ratios = None
        self._buffer_market_cap = None

        # Resolved constrained_families (caches heuristic factor_to_drop choices from
        # the first call so subsequent partial_fit calls use the same basis)
        self._constrained_families = None

        # Resolved basis from the first constrained call.
        self._family_constraint_basis: FamilyConstraintBasis | None = None

        # Accumulator for time series across partial_fit calls. On the first call,
        # stores raw arrays (zero-copy for batch fit). On the second call, promotes to
        # _ArrayBuffer buffers for amortized O(1) appends (avoids O(N^2) np.concatenate)
        self._history: dict[str, AnyArray | _ArrayBuffer] | None = None

    def _attach_benchmark_weights(self, characteristics: AssetPanel) -> None:
        """Compute benchmark_weights and add it as a field in characteristics for
        sub-estimators.
        """
        benchmark_weight_mask = (
            np.isfinite(characteristics[_RETURNS]) & characteristics.estimation_mask
        )
        benchmark_weights = _cap_weights_from_mask(
            self.benchmark_mcap_power,
            market_cap=characteristics[_MARKET_CAP] if self._need_market_cap else None,
            weight_mask=benchmark_weight_mask,
        )
        # Attach benchmark weights for sub_estimators. The panel is a shallow copy from
        # _validate_data, so the user-provided AssetPanel is not mutated.
        characteristics.add_2d_field(
            name=_BENCHMARK_WEIGHTS,
            values=benchmark_weights,
            inactive_policy=InactivePolicy.ZERO,
        )

    def _accumulate_history(self, **arrays: AnyArray) -> None:
        """Accumulate time-series arrays across `partial_fit` calls.

        First call stores raw references (zero-copy, optimal for batch `fit`). Second
        call promotes to :class:`_ArrayBuffer` buffers for amortized O(1) appends,
        avoiding the O(N^2) cost of repeated `np.concatenate`.
        """
        max_history = self.max_history
        if self._history is None:
            if max_history is not None:
                self._history = {k: v[-max_history:].copy() for k, v in arrays.items()}
            else:
                self._history = arrays
        else:
            history = self._history
            if not isinstance(next(iter(history.values())), _ArrayBuffer):
                history = {k: _ArrayBuffer(v) for k, v in history.items()}
                self._history = history

            if max_history is not None:
                arrays = {
                    k: v[-max_history:] if v.shape[0] > max_history else v
                    for k, v in arrays.items()
                }

            for k, v in arrays.items():
                history[k].append(v)

            if max_history is not None:
                for buf in history.values():
                    buf.truncate_to_last(max_history)

    def _get_history(self) -> dict[str, AnyArray]:
        """Return the accumulated history arrays."""
        if self._history is None:
            raise AttributeError("History has not been initialized.")
        if isinstance(next(iter(self._history.values())), _ArrayBuffer):
            return {k: v.array for k, v in self._history.items()}
        return self._history

    def _get_dependency_layers(self) -> list[list[str]]:
        """Return factors grouped by dependency layer for ordered fitting when the
        factor estimator list contains a `DerivedFactor`.

        Each layer contains factors that can be computed in parallel (no dependencies
        within the layer). Layers are ordered so that all dependencies are satisfied.

        Returns
        -------
        layers : list[list[str]]
            List of layers, where each layer is a list of factor names.

        Raises
        ------
        ValueError
            If a DerivedFactor references an undefined source factor.

        CycleError
            If circular dependencies are detected (raised by TopologicalSorter).
        """
        factor_estimator_names = list(self.named_factor_estimators_.keys())
        graph = {name: set() for name in factor_estimator_names}
        for name, factor_estimator in self.named_factor_estimators_.items():
            if isinstance(factor_estimator, DerivedFactor):
                if factor_estimator.source not in self.named_factor_estimators_:
                    raise ValueError(
                        f"DerivedFactor '{name}'s source '{factor_estimator.source}'"
                        f" must be a defined factor. Available factors: "
                        f"{factor_estimator_names}"
                    )
                graph[name].add(factor_estimator.source)

        sorter = TopologicalSorter(graph)
        sorter.prepare()
        layers = []
        while sorter.is_active():
            layer = list(sorter.get_ready())
            layers.append(layer)
            sorter.done(*layer)

        return layers

    def _compute_factor_exposures(
        self, characteristics: AssetPanel, routed_params: dict, method: str
    ) -> tuple[FloatArray, StrArray, StrArray]:
        """Compute factor exposures from asset characteristics.

        Factors are computed in topological order (respecting
            :class:`~skfolio.factor_exposure.DerivedFactor` dependencies). Independent
        factors within the same dependency layer are computed in parallel via threads.

        Parameters
        ----------
        characteristics : AssetPanel
            Panel data for the coverage universe.

        routed_params : dict
            Metadata-routed parameters for each factor estimator.

        Returns
        -------
        exposures : ndarray of shape (n_observations, n_assets, n_factors)
            Stacked factor exposures.

        factor_names : ndarray of shape (n_factors,)
            Ordered factor names.

        factor_families : ndarray of shape (n_factors,)
            Factor family for each factor.

        Raises
        ------
        ValueError
            If a :class:`DerivedFactor` references an undefined source, or if an
            estimator returns an unexpected number of dimensions.
        """
        results_dict = {}
        for layer in self._get_dependency_layers():
            tasks = []
            for name in layer:
                factor_estimator = self.named_factor_estimators_[name]
                if isinstance(factor_estimator, DerivedFactor):
                    try:
                        source_exposure, _, _ = results_dict[factor_estimator.source]
                    except KeyError:
                        raise ValueError(
                            f"DerivedFactor '{name}' depends on"
                            f" '{factor_estimator.source}' which was not found."
                            f" Available factors: {list(results_dict.keys())}"
                        ) from None
                else:
                    source_exposure = None

                tasks.append(
                    (factor_estimator, routed_params[name][method], source_exposure)
                )

            # Threading avoids copying the (potentially large) AssetPanel to each
            # worker. Workers only read from it, so shared memory is safe.
            # Factor/descriptor computations are NumPy-dominated and release the GIL,
            # giving true parallelism with threads.
            results = skp.Parallel(n_jobs=self.n_jobs, prefer="threads")(
                skp.delayed(_compute_single_factor_estimator)(
                    factor_estimator=factor_estimator,
                    characteristics=characteristics,
                    fit_params=fit_params,
                    source_exposure=source_exposure,
                    method=f"{method}_transform",
                )
                for factor_estimator, fit_params, source_exposure in tasks
            )

            for name, res in zip(layer, results, strict=True):
                results_dict[name] = res

        # Assemble exposures, names and families in original factor order
        exposures = []
        factor_names = []
        factor_families = []

        for name in self.named_factor_estimators_.keys():
            exposure, multi_factor_names, factor_family = results_dict[name]

            if exposure.ndim == 2:
                exposures.append(np.expand_dims(exposure, axis=-1))
                factor_names.append(name)
                factor_families.append(factor_family)
            elif exposure.ndim == 3:
                n_multi_factors = exposure.shape[-1]
                exposures.append(exposure)
                factor_names.extend(multi_factor_names)
                factor_families.extend([factor_family] * n_multi_factors)
            else:
                raise ValueError(
                    f"Factor estimator '{name}' returned exposure with {exposure.ndim} "
                    f"dimensions; expected 2 (single factor) or 3 (multi-factor)."
                )

        factor_names = np.array(factor_names)
        factor_families = np.array(factor_families)
        exposures = np.concatenate(exposures, axis=2)

        return exposures, factor_names, factor_families

    def _compute_currency_exposure(
        self, characteristics: AssetPanel, routed_params: dict, method: str
    ) -> tuple[FloatArray, StrArray, StrArray]:
        """Compute one-hot currency factor exposures.

        Currency exposures identify the point-in-time primary currency of each asset.
        They are computed separately from the regression factors because their factor
        returns come directly from `currency_excess_returns` rather than from
        cross-sectional regression on local excess returns.

        Parameters
        ----------
        characteristics : AssetPanel
            Panel data for the coverage universe.

        routed_params : dict
            Metadata-routed parameters for the currency factor estimator.

        Returns
        -------
        exposures : ndarray of shape (n_observations, n_assets, n_currency_factors)
            One-hot currency factor exposures.

        factor_names : ndarray of shape (n_currency_factors,)
            Currency factor names.

        factor_families : ndarray of shape (n_currency_factors,)
            Factor family labels (all `"currency"`).
        """
        exposures = call_asset_panel_transform(
            self.currency_factor_estimator_,
            X=characteristics,
            fit_params=routed_params["currency_factor"][method],
            method=f"{method}_transform",
        )
        factor_names = self.currency_factor_estimator_.factor_names_
        factor_families = np.array([_CURRENCY] * len(factor_names))

        return exposures, factor_names, factor_families

    def _cross_sectional_regression(
        self,
        lagged_exposures: FloatArray,
        asset_returns: FloatArray,
        lagged_market_cap: FloatArray | None,
        regression_eligible_mask: BoolArray,
        estimation_mask: BoolArray,
        active_mask: BoolArray,
        routed_params: dict,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Run cross-sectional regression.

        Parameters
        ----------
        lagged_exposures : ndarray of shape (n_observations, n_assets, n_factors)
            Lagged exposures.

        asset_returns : ndarray of shape (n_observations, n_assets)
            Asset returns aligned with `lagged_exposures`.

        lagged_market_cap : ndarray of shape (n_observations, n_assets) or None
            Optional market cap for regression weighting, lagged by `exposure_lag`
            and aligned with `lagged_exposures`. `None` when
            `regression_mcap_power == 0`.

        regression_eligible_mask : ndarray of shape (n_observations, n_assets)
            Mask indicating assets with finite returns, finite lagged exposures and
            membership in the estimation universe.

        estimation_mask : ndarray of shape (n_observations, n_assets)
            Mask indicating estimation assets.

        active_mask : ndarray of shape (n_observations, n_assets)
            Mask indicating active assets.

        routed_params : dict
            Metadata-routed parameters for the cross-sectional regressor.

        Returns
        -------
        idio_returns : ndarray of shape (n_observations, n_assets)
            Idiosyncratic returns.

        factor_returns : ndarray of shape (n_observations, n_factors)
            Factor returns.

        regression_weights : ndarray of shape (n_observations, n_assets)
            Regression weights: 0 for non-eligible assets, positive otherwise.
        """
        n_observations, n_assets = asset_returns.shape

        weights_mcap = _cap_weights_from_mask(
            self.regression_mcap_power,
            market_cap=lagged_market_cap,
            weight_mask=regression_eligible_mask,
        )

        # First pass regression
        self.cs_regressor_.fit(
            X=lagged_exposures,
            y=asset_returns,
            cs_weights=weights_mcap,
            **routed_params,
        )
        idio_returns = asset_returns - self.cs_regressor_.predict(lagged_exposures)

        if self.inv_idio_variance_weight_shrinkage == 0:
            return idio_returns, self.cs_regressor_.coef_, weights_mcap

        # Two-step feasible GLS: estimate idio variances from the cap-weighted
        # first-pass residuals, then refit with weights blended toward inverse idio
        # variance.
        idio_variance_weights = []
        for t in range(n_observations):
            # Read the variance state before updating it with the date-t residual.
            # A variance that already includes the date-t squared residual embeds
            # the return being regressed: an asset with a large date-t shock would
            # be down-weighted at date t itself, correlating weights with residuals.
            # On the very first observation no variance exists yet so the NaN row
            # falls back to cap weights below.
            if hasattr(self._idio_variance_estimator_inv_var, "variance_"):
                idio_variance_weights.append(
                    1 / self._idio_variance_estimator_inv_var.variance_
                )
            else:
                idio_variance_weights.append(np.full(n_assets, np.nan))
            self._idio_variance_estimator_inv_var.partial_fit(
                idio_returns[[t]],
                **_filter_supported_params(
                    self._idio_variance_estimator_inv_var,
                    method="partial_fit",
                    estimation_mask=estimation_mask[[t]],
                    active_mask=active_mask[[t]],
                ),
            )

        # Assets in variance warmup (e.g. half-life warm-up of EWVariance) are NaN.
        idio_variance_weights = np.stack(idio_variance_weights)

        # Restrict to the regression-eligible set before the median cap and the
        # normalization below. Both blend components are then normalized over the
        # same universe, so the realized blend matches the nominal shrinkage.
        idio_variance_weights[~regression_eligible_mask] = np.nan

        # Protect against very small variance, which would create huge weights.
        idio_variance_weights = CSWinsorizer(low=0.025, high=0.975).fit_transform(
            idio_variance_weights, cs_weights=weights_mcap
        )

        # Cap idio variance weights.
        any_finite = np.isfinite(idio_variance_weights).any(axis=1)
        w_cap = np.full(n_observations, np.nan, dtype=float)
        w_cap[any_finite] = (
            np.nanmedian(idio_variance_weights[any_finite], axis=1)
            * self.inv_idio_variance_max_weight_ratio
        )
        idio_variance_weights = np.minimum(idio_variance_weights, w_cap[:, None])
        idio_variance_weights = (
            idio_variance_weights / np.nansum(idio_variance_weights, axis=1)[:, None]
        )

        # Normalize cap weights to the same scale before blending.
        weights_mcap = weights_mcap / np.nansum(weights_mcap, axis=1)[:, None]

        # Missing inverse-variance weights do not contribute to the inverse-variance
        # component. Rows with no ready variance estimates fall back to cap weights.
        np.nan_to_num(idio_variance_weights, nan=0.0, copy=False)
        idio_variance_weights[~any_finite] = weights_mcap[~any_finite]

        regression_weights = (
            self.inv_idio_variance_weight_shrinkage * idio_variance_weights
            + (1 - self.inv_idio_variance_weight_shrinkage) * weights_mcap
        )
        regression_weights = np.where(regression_eligible_mask, regression_weights, 0.0)

        # Second-pass regression with blended weights.
        self.cs_regressor_.fit(
            X=lagged_exposures,
            y=asset_returns,
            cs_weights=regression_weights,
            **routed_params,
        )
        idio_returns = asset_returns - self.cs_regressor_.predict(lagged_exposures)

        return idio_returns, self.cs_regressor_.coef_, regression_weights

    def _compute_factor_returns_dist(
        self,
        factor_returns: FloatArray,
        factor_names: StrArray,
        observations: AnyArray,
        routed_params: dict,
        first_call: bool,
    ) -> ReturnDistribution:
        """Estimate the factor return distribution.

        Fits (or partially fits) the `factor_prior_estimator` on the estimated factor
        return time series, producing expected factor returns and covariance.
        The factor returns must be in the family-constraint basis to enforce constraints
        in moments estimation and ensure full-rank covariance.

        Parameters
        ----------
        factor_returns : ndarray of shape (n_observations, n_factors)
            Estimated factor returns from the cross-sectional regression.

        factor_names : ndarray of shape (n_factors,)
            Factor names (used as DataFrame column names).

        observations : ndarray of shape (n_observations,)
            Observation labels used as the factor-return DataFrame index.

        routed_params : dict
            Metadata-routed parameters for the factor prior estimator.

        first_call : bool
            Whether this is the first `fit`/`partial_fit` call. On the first call the
            estimator is fitted, otherwise it is updated incrementally via `partial_fit`.

        Returns
        -------
        ReturnDistribution
            Fitted factor return distribution.

        Raises
        ------
        ValueError
            If `partial_fit` is called but the factor prior estimator
            does not implement `partial_fit`.
        """
        # Convert to Dataframe so that the factor prior estimator have access to factor
        # names if needed.
        factor_returns = pd.DataFrame(
            factor_returns, index=observations, columns=factor_names, copy=False
        )

        if first_call:
            self.factor_prior_estimator_.fit(
                factor_returns, **routed_params.factor_prior_estimator.fit
            )
        else:
            if not hasattr(self.factor_prior_estimator_, "partial_fit"):
                raise ValueError(
                    "When calling CharacteristicsFactorModel partial_fit, you must "
                    "provide a factor_prior_estimator that also implements `partial_fit`"
                )
            self.factor_prior_estimator_.partial_fit(
                factor_returns, **routed_params.factor_prior_estimator.partial_fit
            )

        return self.factor_prior_estimator_.return_distribution_

    def _compute_idio_variances(
        self,
        idio_returns: FloatArray,
        estimation_mask: BoolArray,
        active_mask: BoolArray,
        routed_params: dict,
    ) -> FloatArray:
        """Estimate idiosyncratic variances.

        Updates the `idio_variance_estimator_` incrementally for each observation via
        `partial_fit`, and returns the per-observation variance estimates for each
        asset.

        Parameters
        ----------
        idio_returns : ndarray of shape (n_observations, n_assets)
            Idiosyncratic returns.

        estimation_mask : ndarray of shape (n_observations, n_assets)
            Boolean mask indicating estimation assets. This is used by some variance
            estimators (e.g. `RegimeAdjustedEWVariance`) for estimating cross-sectional
            statistics.

        active_mask : ndarray of shape (n_observations, n_assets)
            Boolean mask indicating active assets.

        routed_params : dict
            Metadata-routed parameters for the variance estimator.

        Returns
        -------
        idio_variances : ndarray of shape (n_observations, n_assets)
            Per-observation idiosyncratic variance estimates.
        """
        idio_variances = np.empty_like(idio_returns)
        for t in range(len(idio_returns)):
            self.idio_variance_estimator_.partial_fit(
                idio_returns[[t]],
                **routed_params.idio_variance_estimator.partial_fit,
                **_filter_supported_params(
                    self.idio_variance_estimator_,
                    method="partial_fit",
                    estimation_mask=estimation_mask[[t]],
                    active_mask=active_mask[[t]],
                ),
            )
            idio_variances[t] = self.idio_variance_estimator_.variance_
        return idio_variances

    def _compute_idio_covariance(
        self,
        idio_returns: FloatArray,
        idio_variances: FloatArray,
        estimation_mask: BoolArray,
        active_mask: BoolArray,
        routed_params: dict,
        first_call: bool,
    ) -> FloatArray:
        """Compute the idiosyncratic covariance matrix.

        By construction, idiosyncratic returns are expected to be nearly
        uncorrelated after removing the factor structure, so the idiosyncratic
        covariance is diagonal by default. In that case, this method returns the
        latest per-asset idiosyncratic variances estimated by
        `idio_variance_estimator`.

        When `idio_corr_threshold > 0`, this method allows for sparse residual
        correlation between related securities, such as multiple share classes,
        ADRs versus ordinary shares, dual listings, or otherwise linked assets. It
        standardizes idiosyncratic returns by their contemporaneous idiosyncratic
        volatility, fits `idio_corr_estimator` on the standardized residuals,
        converts the result to a correlation matrix, thresholds small correlations,
        and recombines the retained correlations with the latest idiosyncratic
        variances.

        Variances and correlations are estimated separately so that per-asset risk
        remains driven by `idio_variance_estimator` while off-diagonal covariance
        is introduced only where residual correlations are large enough to retain.
        The final matrix is projected to the nearest positive semi-definite matrix
        when correlation thresholding is used.

        This estimator is fitted on the coverage universe, not the investment subset.
        This is to maximize information.

        Parameters
        ----------
        idio_returns : ndarray of shape (n_observations, n_assets)
            Idiosyncratic returns.

        idio_variances : ndarray of shape (n_observations, n_assets)
            Per-observation idiosyncratic variance estimates.

        estimation_mask : ndarray of shape (n_observations, n_assets)
            Boolean mask indicating estimation assets.

        active_mask : ndarray of shape (n_observations, n_assets)
            Boolean mask indicating active assets.

        routed_params : dict
            Metadata-routed parameters for the correlation estimator.

        Returns
        -------
        idio_cov : ndarray of shape (n_assets,) or (n_assets, n_assets)
            Idiosyncratic variance vector (diagonal case) or full covariance matrix
            (when correlation overlay is applied).
        """
        latest_idio_variances = idio_variances[-1]

        if self.idio_corr_threshold == 0:
            return latest_idio_variances.copy()

        # Standardize idio returns. Guard against non-positive or warmup-NaN variances
        # so non-finite values do not propagate into the correlation estimator.
        idio_returns_standardized = safe_divide(
            idio_returns, np.sqrt(idio_variances), fill_value=np.nan
        )

        corr_kwargs = _filter_supported_params(
            self.idio_corr_estimator_,
            method="fit",
            estimation_mask=estimation_mask,
            active_mask=active_mask,
        )

        # Covariance of standardized residuals
        if first_call:
            self.idio_corr_estimator_.fit(
                idio_returns_standardized,
                **corr_kwargs,
                **routed_params.idio_corr_estimator.fit,
            )
        else:
            if not hasattr(self.idio_corr_estimator_, "partial_fit"):
                raise ValueError(
                    "When calling CharacteristicsFactorModel partial_fit, you must provide "
                    "an idio_corr_estimator that also implements `partial_fit`"
                )
            self.idio_corr_estimator_.partial_fit(
                idio_returns_standardized,
                **corr_kwargs,
                **routed_params.idio_corr_estimator.partial_fit,
            )

        cov = self.idio_corr_estimator_.covariance_

        # Convert to correlation
        corr, _ = cov_to_corr(cov)

        # Sparsify via thresholding (preserve diagonal ones). `np.where` also zeroes
        # non-finite correlations (asset pairs within the correlation estimator
        # warmup), which would otherwise survive thresholding as NaN.
        keep = np.abs(corr) > self.idio_corr_threshold
        np.fill_diagonal(keep, True)
        corr = np.where(keep, corr, 0.0)

        # Build idio covariance
        cov = corr_to_cov(corr, std=np.sqrt(latest_idio_variances))

        # Enforce SPD on the sub-block of assets with finite variances. Assets within
        # the variance estimator warmup keep NaN rows and columns, mirroring the NaN
        # propagation of the diagonal path.
        finite = np.isfinite(latest_idio_variances)
        if finite.all():
            return cov_nearest(cov)
        finite_idx = np.flatnonzero(finite)
        if finite_idx.size:
            sub_block = np.ix_(finite_idx, finite_idx)
            cov[sub_block] = cov_nearest(cov[sub_block])
        return cov

    def _compute_alpha(
        self,
        characteristics: AssetPanel,
        idio_returns: FloatArray,
        idio_variances: FloatArray,
        regression_weights: FloatArray,
        exposures: FloatArray,
        factor_names: StrArray,
        factor_families: StrArray,
        routed_params: dict,
        first_call: bool,
    ) -> FloatArray:
        """Compute asset-level alpha from the alpha estimator.

        Alpha estimators may require the idiosyncratic returns, idiosyncratic variances,
        factor exposures and regression weights. We enrich the
        :class:`~skfolio.containers.AssetPanel` with these before delegating to the
        `alpha_estimator_`. These added fields can contain `NaN` values at the beginning
        corresponding to the warmup period of the descriptors, factor and variance
        estimators. Alpha estimators also have warmup periods from their descriptor
        estimators. In order to maximize data availability and avoid stacking warmup
        periods, we pass the full characteristics AssetPanel to the alpha estimators.
        This means that idiosyncratic returns, idiosyncratic variances, factor exposures
        and regression weights retain their warmup `NaN` values instead of truncating
        the entire AssetPanel. The skfolio alpha estimators handle this gracefully.

        This estimator is fitted on the coverage universe, not the investment subset.
        This is to maximize information.

        Parameters
        ----------
        characteristics : AssetPanel
            Panel data (enriched with additional fields).

        idio_returns : ndarray of shape (n_valid_obs, n_assets)
            Idiosyncratic returns.

        idio_variances : ndarray of shape (n_valid_obs, n_assets)
            Per-observation idiosyncratic variances.

        regression_weights : ndarray of shape (n_valid_obs, n_assets)
            Cross-sectional regression weights.

        exposures : ndarray of shape (n_valid_obs, n_assets, n_factors)
            Factor exposures (in reduced basis if applicable).

        factor_names : ndarray of shape (n_factors,)
            Factor names.

        factor_families : ndarray of shape (n_factors,)
            Factor family labels.

        routed_params : dict
            Metadata-routed parameters for the alpha estimator.

        Returns
        -------
        alpha : ndarray of shape (n_assets,)
            Asset-level expected returns. Zero vector if no alpha estimator is
            provided or while the alpha estimator is in its warmup period.
        """
        if self.alpha_estimator_ is None:
            return np.zeros(self.n_assets_)

        # Inject idio returns, variances, regression weights and exposures into the
        # panel so the alpha estimator can access them.
        n_obs = characteristics.n_observations
        n_post_warmup_obs, _ = idio_returns.shape
        n_warmup_obs = n_obs - n_post_warmup_obs
        if n_warmup_obs > 0:
            # idio returns
            full_idio_returns = np.full((n_obs, self.n_assets_), np.nan)
            full_idio_returns[n_warmup_obs:] = idio_returns

            # idio variances
            full_idio_variances = np.full((n_obs, self.n_assets_), np.nan)
            full_idio_variances[n_warmup_obs:] = idio_variances

            # regression weights
            full_regression_weights = np.zeros((n_obs, self.n_assets_))
            full_regression_weights[n_warmup_obs:] = regression_weights

            # exposure
            n_factors = len(factor_names)
            full_exposures = np.full((n_obs, self.n_assets_, n_factors), np.nan)
            full_exposures[n_warmup_obs:] = exposures
        else:
            full_idio_returns = idio_returns
            full_idio_variances = idio_variances
            full_regression_weights = regression_weights
            full_exposures = exposures

        characteristics[_IDIO_RETURNS] = full_idio_returns
        characteristics[_IDIO_VARIANCES] = full_idio_variances
        characteristics.add_2d_field(
            name=_REGRESSION_WEIGHTS,
            values=full_regression_weights,
            inactive_policy=InactivePolicy.ZERO,
        )
        characteristics.add_3d_field(
            name=_EXPOSURES,
            values=full_exposures,
            third_axis_name="factors",
            third_axis_labels=factor_names,
            third_axis_groups=factor_families,
        )

        if first_call:
            self.alpha_estimator_.fit(
                characteristics, **routed_params.alpha_estimator.fit
            )
        else:
            if not hasattr(self.alpha_estimator_, "partial_fit"):
                raise ValueError(
                    "When calling CharacteristicsFactorModel partial_fit, you must "
                    "provide an alpha_estimator that also implements `partial_fit`"
                )
            self.alpha_estimator_.partial_fit(
                characteristics, **routed_params.alpha_estimator.partial_fit
            )

        alpha = self.alpha_estimator_.alpha_
        if alpha is None:
            # Alpha estimators publish `alpha_ = None` while still in warmup
            # (e.g. PredictorAlpha, EWSharpeOptimalAlpha during their first batches).
            return np.zeros(self.n_assets_)
        return alpha

    def _decompose_alpha(
        self,
        alpha: FloatArray,
        exposure: FloatArray,
        regression_weights: FloatArray,
        routed_params: dict,
    ) -> tuple[FloatArray, FloatArray]:
        """Decompose asset-level alpha into factor-spanned and orthogonal components.

        Projects the asset-level alpha vector onto the column space of the latest
        factor exposure matrix using weighted cross-sectional regression. The fitted
        component is the part of alpha that can be explained by factor exposures and
        is represented as factor-level expected returns. The residual component is
        the asset-specific alpha left unexplained by the factor exposure space.


        Parameters
        ----------
        alpha : ndarray of shape (n_assets,)
            Asset-level alpha from the alpha estimator.

        exposure : ndarray of shape (1, n_assets, n_factors)
            Latest factor exposures (single observation).

        regression_weights : ndarray of shape (1, n_assets)
            Latest regression weights.

        routed_params : dict
            Metadata-routed parameters for the cross-sectional regressor.

        Returns
        -------
        factor_mu : ndarray of shape (n_factors,)
            Factor-level expected returns implied by the spanned alpha.

        orthogonal_alpha : ndarray of shape (n_assets,)
            Asset-specific alpha orthogonal to the factor exposure space. Missing
            alpha or exposure values propagate to the corresponding entries.
        """
        _, n_assets, n_factors = exposure.shape

        if np.allclose(alpha, 0):
            return np.zeros(n_factors), np.zeros(n_assets)

        # Exclude pairs with missing alpha (e.g. assets within the alpha estimator
        # warmup) or missing exposures by zeroing their regression weights, mirroring
        # `_cross_sectional_neutralize`. The regression weights derive from the lagged
        # exposures and returns, so they can be positive where the current alpha or
        # exposure is non-finite, which the regressor rejects.
        valid = np.isfinite(alpha)[np.newaxis, :] & np.all(
            np.isfinite(exposure), axis=2
        )
        regression_weights = np.where(valid, regression_weights, 0.0)
        if not (regression_weights > 0).any():
            return np.zeros(n_factors), np.zeros(n_assets)

        reg = sk.clone(self.cs_regressor_)
        reg.fit(
            X=exposure,
            y=alpha.reshape(1, -1),
            cs_weights=regression_weights,
            **routed_params,
        )

        factor_mu = reg.coef_[0]
        spanned_alpha = reg.predict(exposure).reshape(-1)
        orthogonal_alpha = alpha - spanned_alpha

        return factor_mu, orthogonal_alpha

    def _reset(self) -> None:
        """Reset the fitted state so the next call behaves like a fresh fit."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)


def _compute_single_factor_estimator(
    factor_estimator: BaseFactorExposure,
    characteristics: AssetPanel,
    fit_params: dict[str, object],
    source_exposure: FloatArray | None,
    method: str,
) -> tuple[FloatArray, StrArray | None, str]:
    """Compute a single factor's exposure from characteristics.

    Defined as a standalone function (not a bound method) so that joblib only needs to
    pickle the individual factor estimator, not the entire model.

    Parameters
    ----------
    factor_estimator : BaseFactorExposure
        Factor exposure estimator.

    characteristics : AssetPanel
        Panel data for the coverage universe (shared across parallel workers).

    fit_params : dict
        Metadata-routed fit parameters for this factor.

    source_exposure : ndarray or None
        For :class:`DerivedFactor`, the pre-computed exposure of the source factor.
        `None` for non-derived factors.

    method : str
        Asset-panel transform method called on `factor_estimator`.

    Returns
    -------
    exposure : ndarray of shape (n_observations, n_assets) or (n_observations, n_assets, n_factors)
        Factor exposure(s).

    multi_names : ndarray of shape (n_factors,) or None
        Factor names for multi-factor estimators (e.g. `OneHotCategoricalFactors`).
        `None` for single-factor estimators.

    family : str
        Factor family associated with `factor_estimator`.
    """
    if source_exposure is not None:
        fit_params = {**fit_params, "source_exposure": source_exposure}

    exposure = call_asset_panel_transform(
        factor_estimator,
        X=characteristics,
        fit_params=fit_params,
        method=method,
    )
    factor_names = getattr(factor_estimator, "factor_names_", None)
    return exposure, factor_names, factor_estimator.family


def _lag_with_buffer(
    values: AnyArray, buffer: AnyArray | None, lag: int
) -> tuple[AnyArray, int, AnyArray]:
    """Lag `values` by `lag` rows along axis 0, carrying the last `lag` rows of
    each batch across successive calls through `buffer`.

    On the first call (`buffer=None`), the lag consumes the `lag` leading rows and
    `trim` reports the number of rows consumed. On subsequent calls, the leading
    lagged rows come from `buffer` and `trim` is 0, so no rows are lost at batch
    boundaries during incremental learning.

    Parameters
    ----------
    values : ndarray of shape (n_observations, ...)
        Values to lag along axis 0.

    buffer : ndarray of shape (lag, ...) or None
        Last `lag` rows from previous calls, or `None` on the first call.

    lag : int
        Number of rows to lag by. Must be >= 1.

    Returns
    -------
    lagged : ndarray of shape (n_observations - trim, ...)
        Lagged values.

    trim : int
        Number of leading rows consumed by the lag (nonzero only on the first call).

    buffer : ndarray of shape (lag, ...)
        Updated buffer holding the last `lag` rows for the next call.
    """
    if buffer is None:
        return values[:-lag], lag, values[-lag:].copy()

    # `buffer` holds the last `lag` rows from previous calls. The first rows of
    # this batch use those buffered rows and later rows use this batch shifted by
    # `lag`. This avoids concatenating the full current batch.
    n_observations = values.shape[0]
    lagged = np.empty_like(values)
    n_from_buffer = min(lag, n_observations)
    lagged[:n_from_buffer] = buffer[:n_from_buffer]
    if n_observations > lag:
        lagged[lag:] = values[:-lag]
    _update_buffer(buffer, values, lag)
    return lagged, 0, buffer


def _cap_weights_from_mask(
    power: float, market_cap: FloatArray | None, weight_mask: BoolArray
) -> FloatArray:
    r"""Build `market_cap ** power` weights on a positive-weight mask.

    `power` controls the strength of cap weighting: `0` gives equal positive weights,
    `1` gives raw market-cap weights and values between `0` and `1` shrink cap
    concentration. Assets outside `weight_mask`  receive zero weight.

    The result follows the scikit-learn `sample_weight` convention: weights are relative
    and need not sum to one.

    Parameters
    ----------
    power : float
        Exponent applied to market capitalizations.

    market_cap : FloatArray or None
        Market capitalizations. Required when `power != 0`.

    weight_mask : BoolArray
        Mask identifying asset-observation pairs with positive weights.

    Returns
    -------
    FloatArray
        Weights with `market_cap ** power` inside `weight_mask` and zero outside it.
    """
    if power == 0:
        return weight_mask.astype(float)

    if market_cap is None:
        raise ValueError("market_cap must be provided when power != 0")

    weights = np.zeros_like(market_cap, dtype=float)

    if power == 1:
        weights[weight_mask] = market_cap[weight_mask]
    else:
        weights[weight_mask] = np.power(market_cap[weight_mask], power)

    return weights


def _validate_currency_excess_returns(
    currency_excess_returns: pd.DataFrame,
    observations: AnyArray,
    currency_factor_names: StrArray,
) -> pd.DataFrame:
    """Validate and select direct currency factor returns."""
    missing = set(currency_factor_names) - set(currency_excess_returns.columns)
    if missing:
        raise ValueError(
            "`currency_excess_returns` is missing currency factor columns: "
            f"{sorted(missing)}."
        )
    currency_excess_returns = currency_excess_returns.loc[
        observations, currency_factor_names
    ]
    if not np.all(np.isfinite(currency_excess_returns.to_numpy(dtype=float))):
        raise ValueError("`currency_excess_returns` must contain only finite values.")
    return currency_excess_returns


def _validate_covariance_readiness(
    factor_covariance: FloatArray, latest_idio_variances: FloatArray
) -> None:
    """Validate that covariance estimators have enough usable data.

    Assets may have non-finite idiosyncratic variance, for example newly listed assets
    or assets that recently entered the investment universe before the idiosyncratic
    variance estimator has warmed up. Those assets are intentionally left with NaN
    covariance/scenario entries so downstream optimizers can exclude them from the
    investable subset through their standard finite-data filtering.

    The model fails only when the factor covariance is non-finite, or when no current
    investment asset has finite idiosyncratic risk.
    """
    if not np.all(np.isfinite(factor_covariance)):
        raise ValueError(
            "Not enough observations to estimate a finite factor covariance. "
            "The factor prior estimator produced non-finite covariance values. "
            "Provide more observations or lower the covariance estimator "
            "`min_observations` warmup."
        )

    if not np.isfinite(latest_idio_variances).any():
        raise ValueError(
            "Not enough observations to estimate finite idiosyncratic covariance "
            "for the current investment universe. Provide more observations or "
            "lower the idiosyncratic variance estimator `min_observations` warmup."
        )


def _compute_standardized_idio_returns(
    idio_returns: FloatArray, idio_variances: FloatArray, active_mask: BoolArray
) -> FloatArray:
    """Compute standardized idiosyncratic returns.

    Idiosyncratic returns are divided by their contemporaneous idiosyncratic volatility.
    Missing active observations are imputed with the same-observation cross-sectional
    mean standardized idiosyncratic return so sparse active histories do not reduce
    the scenario set. Inactive asset-observation pairs remain NaN.
    """
    standardized_idio_returns = safe_divide(
        idio_returns, np.sqrt(idio_variances), fill_value=np.nan
    )

    standardized_idio_returns[~active_mask] = np.nan
    valid = np.isfinite(standardized_idio_returns)
    missing_active = active_mask & ~valid

    # Fill missing active standardized returns with the same-observation average
    # standardized returns so sparse asset histories do not shorten the scenario set.
    mean_standardized_idio_return = safe_divide(
        np.nansum(standardized_idio_returns, axis=1),
        valid.sum(axis=1),
        fill_value=0.0,
    )

    np.copyto(
        standardized_idio_returns,
        mean_standardized_idio_return[:, None],
        where=missing_active,
    )

    return standardized_idio_returns


def _assemble_asset_return_scenarios(
    factor_return_scenarios: FloatArray,
    loading_matrix: FloatArray,
    standardized_idio_returns: FloatArray,
    latest_active_mask: BoolArray,
    latest_idio_variances: FloatArray,
    sample_weight: FloatArray | None,
) -> tuple[FloatArray, FloatArray | None]:
    """Assemble asset-level return scenarios.

    Factor return scenarios are mapped to assets through the latest loading matrix to
    produce systematic returns. Historical standardized idiosyncratic returns are then
    rescaled by the latest idiosyncratic volatilities and added to the systematic
    component so the asset scenarios reflect both current factor risk and current
    idiosyncratic risk.
    """
    n_scenarios = min(
        factor_return_scenarios.shape[0],
        standardized_idio_returns.shape[0],
    )

    factor_return_scenarios = factor_return_scenarios[-n_scenarios:]
    standardized_idio_returns = standardized_idio_returns[-n_scenarios:]

    systematic_returns = factor_return_scenarios @ loading_matrix.T

    idio_return_scenarios = standardized_idio_returns * np.sqrt(latest_idio_variances)

    asset_return_scenarios = systematic_returns + idio_return_scenarios
    asset_return_scenarios[:, ~latest_active_mask] = np.nan

    if sample_weight is not None:
        sample_weight = sample_weight[-n_scenarios:].copy()

    return asset_return_scenarios, sample_weight
