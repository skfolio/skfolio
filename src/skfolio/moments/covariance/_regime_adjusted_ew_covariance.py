"""Regime Adjusted Exponentially Weighted Covariance Estimators."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers
import warnings
from collections import deque
from enum import auto

import numpy as np
import numpy.typing as npt
import scipy.special as scs
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.utils.stats import (
    corr_to_cov,
    inverse_volatility_weights,
    squared_mahalanobis_dist,
    squared_standardized_euclidean_dist,
    symmetrize,
)
from skfolio.utils.tools import AutoEnum, _validate_mask, half_life_to_decay_factor

_NUMERICAL_THRESHOLD = 1e-12


class RegimeAdjustmentTarget(AutoEnum):
    r"""Target dimension used to calibrate the short-term volatility update (STVU).

    Determines what statistic is computed to detect volatility regime changes.

    The STVU uses a statistic :math:`d^2` that measures the discrepancy between predicted
    and realized risk. The target determines which aspect of the covariance
    matrix is calibrated.

    .. list-table::
       :header-rows: 1
       :widths: 20 35 45

       * - Target
         - Formula
         - What it calibrates
       * - PORTFOLIO
         - :math:`d^2 = n \cdot (w^T r)^2 / (w^T \Sigma w)`
         - Portfolio variance along a single aggregated direction
       * - DIAGONAL
         - :math:`d^2 = \sum_i (r_i / \sigma_i)^2`
         - Individual asset volatilities across the universe (ignores correlations)
       * - MAHALANOBIS
         - :math:`d^2 = r^T \Sigma^{-1} r`
         - Full covariance structure (all eigenvalue directions)

    .. note::

        `PORTFOLIO` (the default) calibrates the covariance along economically
        relevant directions. `MAHALANOBIS` weights all eigenvector directions
        equally, including the smallest-eigenvalue directions whose estimates
        are typically the least stable. In practice this can make the regime
        multiplier sensitive to returns along poorly estimated directions
        that carry little portfolio relevance.

    References
    ----------
    .. [1] "The Elements of Quantitative Investing", Wiley Finance,
        Giuseppe Paleologo (2025).
    """

    MAHALANOBIS = auto()
    DIAGONAL = auto()
    PORTFOLIO = auto()


class RegimeAdjustmentMethod(AutoEnum):
    r"""Transformation used to map the STVU statistic to the volatility multiplier.

    Determines how the raw STVU statistic :math:`d^2` is transformed into the
    regime multiplier :math:`\phi` applied by the estimator.

    .. list-table::
       :header-rows: 1
       :widths: 15 40 45

       * - Method
         - Multiplier :math:`\phi`
         - Characteristics
       * - LOG
         - :math:`\phi = \exp(\text{EWMA}(\log d^2 - \kappa)/2)` where :math:`\kappa = E[\log d^2]`
         - Robust to outliers (log compresses extremes).
       * - FIRST_MOMENT
         - :math:`\phi = \text{EWMA}(d / \mathbb{E}[d])`
         - Calibrates the first moment of the standardized risk statistic.
           More robust than RMS, less robust than LOG.
       * - RMS
         - :math:`\phi = \sqrt{\text{EWMA}(d^2/n)}`
         - :math:`\chi^2` calibration. Sensitive to outliers (RMS ≥ mean).

    References
    ----------
    .. [1] "The Elements of Quantitative Investing", Wiley Finance,
        Giuseppe Paleologo (2025).
    """

    LOG = auto()
    FIRST_MOMENT = auto()
    RMS = auto()


_FITTED_ATTR = "covariance_"

# Minimum active assets required for STVU statistic computation per target
_MIN_ACTIVE_FOR_REGIME = {
    RegimeAdjustmentTarget.MAHALANOBIS: 2,
    RegimeAdjustmentTarget.DIAGONAL: 1,
    RegimeAdjustmentTarget.PORTFOLIO: 1,
}


class RegimeAdjustedEWCovariance(BaseCovariance):
    r"""Exponentially weighted covariance estimator with regime adjustment via the
    Short-Term Volatility Update (STVU) [1]_.

    This estimator computes an exponentially weighted covariance and applies a scalar
    multiplier :math:`\phi_t` to improve risk calibration when volatility regimes change
    more quickly than a plain EWMA can track.

    This estimator also supports separate half life for variance and correlation.
    Lower half life for variance allows the model to adapt faster to volatility shifts,
    while higher half life for correlation enables more stable estimation of
    co-movements, which typically require more data for reliable inference and reduces
    estimation noise. This choice also aligns with empirical evidence that volatility
    tends to mean-revert faster than correlation. Using a lower (more responsive) decay
    factor for variance can capture this behavior.

    Additionally, this estimator supports optional Newey-West HAC (Heteroskedasticity
    and Autocorrelation Consistent) correction via the `hac_lags` parameter. This
    adjusts for serial correlation in returns.

    The STVU is configured by two parameters:

    - :class:`RegimeAdjustmentTarget`: determines the statistic used to detect volatility regime
      changes (see the enum docstring for details and formulae).
    - :class:`RegimeAdjustmentMethod`: determines how the raw statistic is transformed into the
      regime multiplier :math:`\phi` (see the enum docstring for details and formulae).

    **NaN handling:**

    The estimator handles missing data (NaN returns) caused by late listings,
    delistings, and holidays using EWMA updates together with `active_mask`. An asset
    with `active_mask=True` is treated as active at time :math:`t`. If its return is
    finite, the EWMA is updated normally. If its return is NaN, the observation is
    treated as a holiday and covariance entries involving this asset are kept unchanged.
    An asset with `active_mask=False` is treated as inactive, for example during
    pre-listing or post-delisting periods, and covariance entries involving this asset
    are set to NaN.

    * **Active with valid return**: Normal EWMA update.
    * **Active with NaN return (holiday)**: Freeze; covariance entries involving this
      asset are kept unchanged.
    * **Inactive** (`active_mask=False`): Covariance entries involving this asset are
      set to NaN.

    When `active_mask` is not provided, trailing NaN returns are ambiguous: they could
    correspond either to holidays, in which case covariance is frozen, or to inactive
    periods, in which case covariance is set to NaN.

    The `min_observations` parameter controls a warm-up period: an asset's
    covariance entries remain NaN in the output until it has accumulated enough valid
    observations for a reliable estimate.

    **Late-listing bias correction:**

    The EWMA recursion is initialized at zero for every asset. This guarantees that
    the internal covariance state remains positive semi-definite at every step, but
    introduces a transient downward scale bias: after :math:`n_i` observations, the
    raw EWMA for asset :math:`i` is damped by a factor :math:`(1 - \lambda^{n_i})`.
    At output time, a per-asset correction removes this bias:

    .. math::

        \hat{\Sigma}_{ij} = \frac{S_{ij}}{\sqrt{(1 - \lambda^{n_i})(1 - \lambda^{n_j})}}

    where :math:`S` is the raw internal EWMA. This is a congruence transform
    :math:`D S D` with :math:`D = \text{diag}(1 / \sqrt{1 - \lambda^{n_i}})`,
    which preserves positive semi-definiteness while restoring the correct variance
    scale.

    When `corr_half_life` is provided, the same bias correction is applied independently
    to the variance state (using :math:`\lambda`) and the correlation state (using
    :math:`\lambda_c`), then the covariance is reconstructed from the corrected
    components. The correlation bias correction uses pairwise co-observation counts
    rather than per-asset counts, so asynchronous late listings, holidays, and
    delistings are corrected at the pair level.

    **Estimation universe for STVU:**

    An optional `estimation_mask` defines the estimation universe used for the STVU
    regime multiplier without affecting pairwise covariance EWMA updates. The STVU
    is computed in a one-step-ahead manner: the return observed at time :math:`t`
    is standardized by the bias-corrected covariance estimate available at time
    :math:`t-1`, and only assets that were already above `min_observations`
    before time :math:`t` contribute to the regime signal. This is important
    because the STVU statistic is sensitive to poorly-estimated assets. Noisy
    or illiquid assets with unreliable covariance estimates can inflate or
    deflate the distance, distorting the regime multiplier for the entire
    covariance matrix.

    For standard exponentially weighted covariance without regime adjustment,
    see :class:`EWCovariance`.

    Parameters
    ----------
    half_life : float, default=40
        Half-life of the exponential weights for variance estimation, in number of
        observations.

        When `corr_half_life` is None (default), this also controls the correlation
        estimation, resulting in standard EWMA covariance:
        :math:`\Sigma_t = \lambda \Sigma_{t-1} + (1-\lambda) r_t r_t^\top`

        When `corr_half_life` is provided, variance and correlation are updated
        with different half-lives to capture their different dynamics.

        The half-life controls how quickly older observations lose their influence:

        * **Larger half-life**: More stable estimates, slower to adapt (robust to noise)
        * **Smaller half-life**: More responsive estimates, faster to adapt (sensitive to noise)

        The decay factor :math:`\lambda` is computed as:
        :math:`\lambda = 2^{-1/\text{half-life}}`

        For example:
            * half-life = 40: :math:`\lambda \approx 0.983`
            * half-life = 23: :math:`\lambda \approx 0.970`
            * half-life = 11: :math:`\lambda \approx 0.939`
            * half-life = 6: :math:`\lambda \approx 0.891`

        .. note::
            For portfolio optimization, larger half-lives (>= 20) are generally
            preferred to avoid excessive turnover from estimation noise.

    corr_half_life : float, optional
        Half-life for correlation estimation, in number of observations.

        If None (default), the same `half_life` is used for both variance and
        correlation, resulting in standard EWMA covariance.

        If provided, enables separate half-lives: `half_life` governs variance
        and `corr_half_life` governs correlation. This is useful because volatility
        typically mean-reverts faster than correlation, so using a smaller (more
        responsive) half-life for variance can better capture regime changes.

    hac_lags : int, optional
        Number of lags for Newey-West HAC (Heteroskedasticity and Autocorrelation
        Consistent) correction. If None (default), no HAC correction is applied.

        When enabled, the covariance update uses HAC-adjusted cross-products instead
        of simple outer products, accounting for autocorrelation in returns:

        .. math::

            r_t r_t^T + \sum_{j=1}^{L} w_j (r_t r_{t-j}^T + r_{t-j} r_t^T)

        where :math:`w_j = 1 - j/(L+1)` is the Bartlett kernel weight.

        Typical values:
            * Daily equity data: 3-5 lags (weak autocorrelation from microstructure)
            * High-frequency data: 5-10 lags (stronger autocorrelation)
            * Monthly data: 1-2 lags

        Must be a positive integer if specified.

    regime_half_life : float, optional
        Half-life for smoothing the volatility regime signal, in number of
        observations.

        The regime signal is built from one-step-ahead standardized risk
        statistics and then transformed into the multiplier :math:`\phi`
        according to `regime_target` and `regime_method`. A shorter
        `regime_half_life` makes the multiplier react faster to abrupt
        changes in realized risk; a longer one produces a smoother, slower
        moving adjustment.

        If None (default), it is automatically calibrated as:
        :math:`\text{regime-half-life} = 0.5 \times \text{half-life}`

        This makes the STVU more responsive (shorter half-life) than the covariance,
        allowing it to quickly rescale risk when realized volatility deviates from
        the slower EWMA estimate.

    regime_target : RegimeAdjustmentTarget, default=RegimeAdjustmentTarget.PORTFOLIO
        Target dimension used to calibrate the short-term volatility update:

        - `PORTFOLIO`: Portfolio variance :math:`((w^T r)^2/(w^T \Sigma w))`
        - `DIAGONAL`: Individual volatilities :math:`(\sum_i (r_i/\sigma_i)^2)`
        - `MAHALANOBIS`: Full covariance :math:`(r^T \Sigma^{-1} r)`

    regime_method : RegimeAdjustmentMethod, default=RegimeAdjustmentMethod.FIRST_MOMENT
        Method used to transform the update statistic into the volatility multiplier :math:`\phi`:

        - `LOG`: Robust to outliers (log compresses extremes)
        - `FIRST_MOMENT`: Calibrates the first moment of the standardized risk
          statistic
        - `RMS`: :math:`\chi^2` calibration (sensitive to extremes)

    regime_portfolio_weights : array-like of shape (n_assets,) or (n_portfolios, n_assets) or None, default=None
        Portfolio weights used by the STVU `PORTFOLIO` target. Only used when
        `regime_target=RegimeAdjustmentTarget.PORTFOLIO`.

        If None (default), uses inverse-volatility weights, which neutralizes asset
        volatility dispersion so high-volatility assets don't dominate the calibration
        statistic. These weights are recomputed dynamically as variances evolve.

        If a 1D array is provided, a single static portfolio is used. If a 2D array of
        shape `(n_portfolios, n_assets)` is provided, the STVU statistic is computed
        independently for each portfolio, transformed, and then averaged into a single
        regime signal. This calibrates the covariance along multiple traded directions
        without being affected by noise in uninvestable eigenvector directions (unlike
        `MAHALANOBIS`).

        Weights are automatically normalized so each row sums to 1.

        For equal-weight calibration, pass
        `regime_portfolio_weights=np.ones(n_assets)/n_assets`.

    regime_multiplier_clip : tuple[float, float] or None, default=(0.7, 1.6)
        Clip :math:`\phi` to avoid extreme swings in the regime multiplier.
        Set to None to disable clipping. The multiplier is applied to the covariance
        as :math:`\phi^2 \Sigma`. With the default bounds, the covariance scale remains
        between :math:`0.7^2 = 0.49` and :math:`1.6^2 = 2.56`.

    regime_min_observations : int, optional
        Minimum number of one-step-ahead comparisons before enabling STVU.
        If insufficient data, STVU defaults to 1.0 (no adjustment).

        If None (default), it is automatically set to `int(regime_half_life)`,
        ensuring the STVU EWMA has seen roughly one half-life of data before being
        applied.

    min_observations : int, optional
        Minimum number of valid observations per asset before its covariance entries
        are considered reliable and exposed in the output `covariance_`. Until this
        threshold is reached, the asset's covariance entries remain NaN.

        This warm-up prevents noisy estimates from a few initial observations from being
        used by downstream optimizers.

        The default (`None`) uses `int(max(half_life, corr_half_life))` as the
        threshold when `corr_half_life` is set, or `int(half_life)` otherwise. This
        ensures both variance and correlation bias-correction factors have decayed to
        at most 50%. Set to 1 to disable warm-up entirely.

    assume_centered : bool, default=True
        If True (default), the EWMA update uses raw returns without demeaning. This
        is the standard convention for EWMA covariance estimation in finance.
        If False, returns are demeaned using an EWMA mean estimate before computing
        the covariance update, and `location_` tracks the EWMA mean.

    nearest : bool, default=True
        If this is set to True, the covariance is replaced by the nearest covariance
        matrix that is positive definite and with a Cholesky decomposition that can be
        computed. The variance is left unchanged.
        The default is `True`.

    higham : bool, default=False
        If this is set to True, the Higham (2002) algorithm is used to find the
        nearest PD covariance, otherwise the eigenvalues are clipped to a threshold
        above zeros (1e-13). The default is `False` and uses the clipping method as
        the Higham algorithm can be slow for large datasets.

    higham_max_iteration : int, default=100
        Maximum number of iterations of the Higham (2002) algorithm.
        The default value is `100`.

    Attributes
    ----------
    covariance_ : ndarray of shape (n_assets, n_assets)
        Estimated covariance matrix. Contains NaN for assets that are inactive
        or have not yet accumulated `min_observations` valid observations.

    regime_multiplier_ : float
        The volatility regime adjustment factor applied.
        Equal to 1.0 if insufficient data.

    location_ : ndarray of shape (n_assets,)
        Estimated location (mean). If `assume_centered=True`, this is zeros.
        Otherwise, it tracks the EWMA mean of returns. Contains NaN for inactive
        assets.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X`
        has feature names that are all strings.

    Notes
    -----
    The STVU compares predicted versus realized risk using a one-step-ahead
    standardized statistic :math:`d^2_{t+1}` computed from the covariance
    estimate at time :math:`t` and the return observed at time :math:`t+1`.
    The exact statistic depends on `regime_target`:

    * `PORTFOLIO` calibrates covariance along one or more portfolio
      directions.
    * `DIAGONAL` calibrates the diagonal risk scale and ignores
      correlations.
    * `MAHALANOBIS` calibrates the full covariance structure.

    Under correct calibration, the transformed statistic has unit scale in
    expectation. Persistent values above that level imply realized risk is
    higher than predicted, so :math:`\phi > 1` scales the covariance up.
    Persistent values below that level imply over-prediction, so
    :math:`\phi < 1` scales it down.

    This approach is related to volatility updating in multivariate GARCH
    models, but implemented here as a multiplicative adjustment on top of an
    EWMA covariance estimator.

    References
    ----------
    .. [1] "The Elements of Quantitative Investing", Wiley Finance,
        Giuseppe Paleologo (2025).

    .. [2] "Multivariate exponentially weighted moving covariance matrix",
        Technometrics, Hawkins & Maboudou-Tchao (2008).

    .. [3] "Dynamic conditional correlation: A simple class of multivariate GARCH
        models", Journal of Business & Economic Statistics, Engle (2002).

    .. [4] "Computing the nearest correlation matrix - A problem from finance",
        IMA Journal of Numerical Analysis, Higham (2002)

    .. [5] "An Introduction to Multivariate Statistical Analysis", Wiley,
        Anderson (2003).

    See Also
    --------
    :ref:`sphx_glr_auto_examples_online_learning_plot_1_online_covariance_forecast_evaluation.py`
        Online covariance forecast evaluation with `EWCovariance` and
        `RegimeAdjustedEWCovariance`.
    :ref:`sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py`
        Online covariance hyperparameter tuning with
        `RegimeAdjustedEWCovariance`.
    :ref:`sphx_glr_auto_examples_online_learning_plot_3_online_portfolio_optimization_evaluation.py`
        Online evaluation of portfolio optimization using `MeanRisk` with
        exponentially weighted moments.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.moments import RegimeAdjustedEWCovariance, RegimeAdjustmentTarget, RegimeAdjustmentMethod
    >>> from skfolio.preprocessing import prices_to_returns
    >>> import numpy as np
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>>
    >>> # Portfolio target with inverse-vol weights and FIRST_MOMENT method (default)
    >>> model = RegimeAdjustedEWCovariance(half_life=23)
    >>> model.fit(X)
    >>> print(model.regime_multiplier_)
    >>>
    >>> # DIAGONAL target (individual asset volatilities)
    >>> model2 = RegimeAdjustedEWCovariance(
    ...     regime_target=RegimeAdjustmentTarget.DIAGONAL,
    ...     regime_method=RegimeAdjustmentMethod.RMS,
    ... )
    >>> model2.fit(X)
    >>>
    >>> # Mahalanobis target (full covariance structure)
    >>> model_maha = RegimeAdjustedEWCovariance(
    ...     regime_target=RegimeAdjustmentTarget.MAHALANOBIS,
    ...     regime_method=RegimeAdjustmentMethod.FIRST_MOMENT,
    ... )
    >>> model_maha.fit(X)
    >>>
    >>> # Portfolio target with equal weights
    >>> n_assets = X.shape[1]
    >>> model_equal = RegimeAdjustedEWCovariance(
    ...     regime_target=RegimeAdjustmentTarget.PORTFOLIO,
    ...     regime_portfolio_weights=np.ones(n_assets) / n_assets,
    ... )
    >>> model_equal.fit(X)
    >>>
    >>> # With Newey-West HAC correction
    >>> model_hac = RegimeAdjustedEWCovariance(
    ...     half_life=23,
    ...     hac_lags=5
    ... )
    >>> model_hac.fit(X)
    """

    regime_multiplier_: float

    def __init__(
        self,
        half_life: float = 40,
        corr_half_life: float | None = None,
        hac_lags: int | None = None,
        regime_half_life: float | None = None,
        regime_target: RegimeAdjustmentTarget = RegimeAdjustmentTarget.PORTFOLIO,
        regime_method: RegimeAdjustmentMethod = RegimeAdjustmentMethod.FIRST_MOMENT,
        regime_portfolio_weights: npt.ArrayLike | None = None,
        regime_multiplier_clip: tuple[float, float] | None = (0.7, 1.6),
        regime_min_observations: int | None = None,
        min_observations: int | None = None,
        assume_centered: bool = True,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ) -> None:
        super().__init__(
            assume_centered=assume_centered,
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.half_life = half_life
        self.corr_half_life = corr_half_life
        self.hac_lags = hac_lags
        self.regime_target = regime_target
        self.regime_method = regime_method
        self.regime_portfolio_weights = regime_portfolio_weights
        self.regime_half_life = regime_half_life
        self.regime_multiplier_clip = regime_multiplier_clip
        self.regime_min_observations = regime_min_observations
        self.min_observations = min_observations

    def fit(
        self,
        X: npt.ArrayLike,
        y=None,
        *,
        active_mask: npt.ArrayLike | None = None,
        estimation_mask: npt.ArrayLike | None = None,
    ) -> RegimeAdjustedEWCovariance:
        """Fit the Regime-Adjusted Exponentially Weighted Covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets. May contain NaN for missing data
            (holidays, late listings, delistings).

        y : Ignored
            Not used, present for API consistency by convention.

        active_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating whether each asset is structurally active at
            each observation. Use this to distinguish between holidays
            (`active_mask=True` and NaN return: covariance is frozen) and
            inactive periods such as pre-listing or post-delisting
            (`active_mask=False`: covariance is set to NaN). If `None`
            (default), all pairs are assumed active.

        estimation_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating which active assets should belong to the
            estimation universe for the STVU statistic computation on each day.

            - If None (default), all active assets with finite returns and finite
              covariance estimates are used.
            - If provided, only assets where the mask is True contribute to the
              regime multiplier calculation.

            Pairwise covariance EWMA updates still use all active assets with valid
            observations; this mask only affects the STVU regime multiplier calculation.

            This is important because the STVU statistic is sensitive to
            poorly-estimated assets. Noisy or illiquid assets with unreliable covariance
            estimates can inflate or deflate the distance, distorting the regime
            multiplier for the entire covariance matrix.

            Use cases:
                * Focus on liquid assets to reduce noise in regime detection
                * Exclude recently-listed assets whose covariance is still poorly
                  estimated
                * Match the estimation universe used in a factor model

        Returns
        -------
        self : RegimeAdjustedEWCovariance
            Fitted estimator.
        """
        self._reset()
        self.partial_fit(X, y, active_mask=active_mask, estimation_mask=estimation_mask)
        return self

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y=None,
        *,
        active_mask: npt.ArrayLike | None = None,
        estimation_mask: npt.ArrayLike | None = None,
    ) -> RegimeAdjustedEWCovariance:
        """Incrementally fit the Regime-Adjusted EW Covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets. May contain NaN for missing data (holidays,
            late listings, delistings).

        y : Ignored
            Not used, present for API consistency by convention.

        active_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating whether each asset is structurally active at each
            observation. Use this to distinguish between holidays (`active_mask=True`
            and NaN return: covariance is frozen) and inactive periods such as
            pre-listing or post-delisting (`active_mask=False`: covariance is set to
            NaN). If `None` (default), all pairs are assumed active.

        estimation_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating which active assets should belong to the
            estimation universe for the STVU statistic computation on each day.
            See `fit` for details.

        Returns
        -------
        self : RegimeAdjustedEWCovariance
            Fitted estimator.
        """
        first_call = not hasattr(self, _FITTED_ATTR)
        X = skv.validate_data(
            self, X, reset=first_call, dtype=float, ensure_all_finite="allow-nan"
        )

        if first_call:
            self._validate_params()
            self._initialize()

        active_mask = _validate_mask(X=X, mask=active_mask, name="active_mask")
        estimation_mask = _validate_mask(
            X=X, mask=estimation_mask, name="estimation_mask"
        )

        all_active = np.ones(self.n_features_in_, dtype=bool)
        for t, returns in enumerate(X):
            active_row = active_mask[t] if active_mask is not None else all_active
            est_row = estimation_mask[t] if estimation_mask is not None else None
            self._process_return_row(returns, active_row, est_row)

        # Produce output covariance with bias correction
        covariance = self._bias_correct_covariance()

        # NaN-mask assets below min_observations threshold
        not_ready = self._obs_count < self._min_observations
        if np.any(not_ready):
            covariance[not_ready, :] = np.nan
            covariance[:, not_ready] = np.nan

        if not self.assume_centered:
            inactive = self._obs_count < 1
            if np.any(inactive):
                self.location_[inactive] = np.nan

        # Re-symmetrize active submatrix to prevent floating-point drift
        symmetrize(covariance, where=np.isfinite(np.diag(covariance)))

        # Compute regime multiplier
        if self._n_regime_observations < self._regime_min_observations:
            regime_multiplier = 1.0
        else:
            match self.regime_method:
                case RegimeAdjustmentMethod.RMS:
                    regime_multiplier = np.sqrt(max(self._regime_state, 0.0))
                case RegimeAdjustmentMethod.FIRST_MOMENT:
                    regime_multiplier = self._regime_state
                case RegimeAdjustmentMethod.LOG:
                    regime_multiplier = np.exp(0.5 * self._regime_state)
            if self.regime_multiplier_clip is not None:
                lo, hi = self.regime_multiplier_clip
                regime_multiplier = np.clip(regime_multiplier, lo, hi)

        self.regime_multiplier_ = regime_multiplier
        covariance = regime_multiplier**2 * covariance

        self._set_covariance(covariance)
        return self

    def _validate_params(self) -> None:
        """Validate parameters."""
        if not isinstance(self.regime_target, RegimeAdjustmentTarget):
            raise ValueError(
                f"regime_target must be a RegimeAdjustmentTarget, got "
                f"{self.regime_target!r}"
            )

        if not isinstance(self.regime_method, RegimeAdjustmentMethod):
            raise ValueError(
                f"regime_method must be a RegimeAdjustmentMethod, got "
                f"{self.regime_method!r}"
            )

        if self.regime_portfolio_weights is not None:
            if self.regime_target != RegimeAdjustmentTarget.PORTFOLIO:
                raise ValueError(
                    f"regime_portfolio_weights can only be used with "
                    f"regime_target=RegimeAdjustmentTarget.PORTFOLIO, got "
                    f"regime_target={self.regime_target}"
                )
            weights = np.atleast_2d(np.asarray(self.regime_portfolio_weights))
            if weights.ndim != 2:
                raise ValueError(
                    f"regime_portfolio_weights must be 1D or 2D, "
                    f"got shape {np.asarray(self.regime_portfolio_weights).shape}"
                )
            if weights.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"regime_portfolio_weights must have n_assets={self.n_features_in_} "
                    f"columns, got {weights.shape[1]}"
                )
            if np.any(weights < 0):
                raise ValueError("regime_portfolio_weights must be non-negative")
            row_sums = np.sum(weights, axis=1)
            if np.any(row_sums <= 0):
                raise ValueError(
                    "Each row of regime_portfolio_weights must have positive sum"
                )

        if (
            self.regime_min_observations is not None
            and self.regime_min_observations < 1
        ):
            raise ValueError(
                f"regime_min_observations must be >= 1 (got {self.regime_min_observations})"
            )

        if self.regime_multiplier_clip is not None:
            if len(self.regime_multiplier_clip) != 2:
                raise ValueError(
                    f"regime_multiplier_clip must be a tuple of length 2, got "
                    f"{len(self.regime_multiplier_clip)}"
                )
            lo, hi = self.regime_multiplier_clip
            if not (0 < lo < hi):
                raise ValueError(
                    f"regime_multiplier_clip must satisfy 0 < lo < hi, got ({lo}, {hi})"
                )

        if self.half_life <= 0:
            raise ValueError(
                f"half_life must be positive (got {self.half_life}). "
                f"Typical values: 10-100 observations."
            )

        if self.corr_half_life is not None and self.corr_half_life <= 0:
            raise ValueError(
                f"corr_half_life must be positive (got {self.corr_half_life}). "
                f"Typical values: 10-100 observations."
            )

        if self.hac_lags is not None:
            if not isinstance(self.hac_lags, numbers.Integral) or self.hac_lags < 1:
                raise ValueError(
                    f"hac_lags must be a positive integer, got {self.hac_lags}"
                )

        if self.regime_half_life is not None:
            if self.regime_half_life <= 0:
                raise ValueError(
                    f"regime_half_life must be positive (got {self.regime_half_life})."
                )
            if self.regime_half_life > 138:
                warnings.warn(
                    f"regime_half_life = {self.regime_half_life} produces a "
                    f"very slow-moving STVU that may desynchronize "
                    f"from covariance evolution and cause erratic behavior. "
                    f"Consider using auto-calibration (regime_half_life=None) or "
                    f"a value <= 138 for more stable results.",
                    UserWarning,
                    stacklevel=2,
                )

        effective_hl = self.half_life
        if self.corr_half_life is not None:
            effective_hl = max(self.half_life, self.corr_half_life)

        if self.min_observations is None:
            self._min_observations = max(1, int(effective_hl))
        else:
            if self.min_observations < 1:
                raise ValueError(
                    f"min_observations must be >= 1, got {self.min_observations}"
                )
            self._min_observations = self.min_observations

    def _initialize(self) -> None:
        """Initialize internal state with zero-seeded accumulators."""
        n_assets = self.n_features_in_
        self._n_regime_observations = 0
        self._obs_count = np.zeros(n_assets, dtype=int)
        self._is_active = np.ones(n_assets, dtype=bool)

        if self.regime_portfolio_weights is not None:
            w = np.atleast_2d(
                np.asarray(self.regime_portfolio_weights, dtype=np.float64)
            )
            self._regime_portfolio_weights = w / w.sum(axis=1, keepdims=True)
        else:
            self._regime_portfolio_weights = None

        self._regime_state = None

        self._decay = half_life_to_decay_factor(self.half_life)

        if self.corr_half_life is not None and not np.isclose(
            self.corr_half_life, self.half_life
        ):
            self._separate_var_corr = True
            self._corr_decay = half_life_to_decay_factor(self.corr_half_life)
        else:
            self._separate_var_corr = False
            self._corr_decay = self._decay

        # Zero-seeded accumulators
        self._cov = np.zeros((n_assets, n_assets))
        if self._separate_var_corr:
            self._var = np.zeros(n_assets)
            self._corr_state = np.zeros((n_assets, n_assets))
            self._pair_obs_count = np.zeros((n_assets, n_assets), dtype=np.uint32)

        if self.assume_centered:
            self.location_ = np.zeros(n_assets)
        else:
            self.location_ = np.full(n_assets, np.nan)

        if self.regime_half_life is None:
            self._regime_half_life = 0.5 * self.half_life
        else:
            self._regime_half_life = self.regime_half_life
        self._regime_decay = half_life_to_decay_factor(self._regime_half_life)

        if self.regime_min_observations is None:
            self._regime_min_observations = max(1, int(self._regime_half_life))
        else:
            self._regime_min_observations = self.regime_min_observations

        if self.hac_lags is not None:
            self._return_buffer = deque(maxlen=self.hac_lags)
        else:
            self._return_buffer = None

    def _bias_correct_covariance(self) -> np.ndarray:
        """Return a bias-corrected output covariance.

        For the direct path, applies the congruence transform to the raw
        accumulator. For the separate var/corr path, bias-corrects variance
        and correlation independently, then reconstructs covariance.
        """
        if not self._separate_var_corr:
            covariance = self._cov.copy()
            bc = np.where(
                self._obs_count > 0,
                1.0 / np.sqrt(np.maximum(1.0 - self._decay**self._obs_count, 1e-15)),
                1.0,
            )
            covariance *= np.outer(bc, bc)
            # NaN out inactive assets
            inactive = ~self._is_active
            if np.any(inactive):
                covariance[inactive, :] = np.nan
                covariance[:, inactive] = np.nan
            return covariance

        # Separate var/corr path: bias-correct each independently
        n_assets = self.n_features_in_
        covariance = np.full((n_assets, n_assets), np.nan)

        active = self._is_active & (self._obs_count > 0)
        if not np.any(active):
            return covariance

        active_idx = np.where(active)[0]

        # Bias-correct variance
        var_bc = self._var[active_idx].copy()
        var_bc_factor = 1.0 / np.maximum(
            1.0 - self._decay ** self._obs_count[active_idx], 1e-15
        )
        var_bc *= var_bc_factor
        std_bc = np.sqrt(np.maximum(var_bc, _NUMERICAL_THRESHOLD))

        # Bias-correct correlation state using pairwise co-observation counts.
        ix = np.ix_(active_idx, active_idx)
        corr_raw = self._corr_state[ix].copy()
        corr_counts = self._pair_obs_count[ix]
        corr_bc = 1.0 / np.maximum(1.0 - self._corr_decay**corr_counts, 1e-15)
        corr_raw *= corr_bc

        # Normalize to proper correlation matrix
        diag_sqrt = np.sqrt(np.clip(np.diag(corr_raw), _NUMERICAL_THRESHOLD, None))
        inv_diag_sqrt = 1.0 / diag_sqrt
        corr = corr_raw * (inv_diag_sqrt[:, None] * inv_diag_sqrt[None, :])
        corr = np.clip(corr, -1.0, 1.0)
        corr = 0.5 * (corr + corr.T)
        np.fill_diagonal(corr, 1.0)

        covariance[ix] = corr_to_cov(corr, std_bc)
        return covariance

    def _get_bias_corrected_cov_for_regime(self, regime_idx: np.ndarray) -> np.ndarray:
        """Return a bias-corrected covariance submatrix for STVU computation.

        This helper is used only for STVU calibration, not for the final
        `covariance_` output.

        It applies exact per-asset variance bias correction to the internally
        maintained `_cov` matrix. In the DCC path, the normalized correlation
        already embedded in `_cov` is reused without pairwise pair-count
        correction. This keeps STVU fast; exact pairwise correction is still
        applied in :meth:`_bias_correct_covariance` for the final output.

        For synchronous data, this is exact because the scalar correlation bias
        cancels during normalization. For asynchronous data, off-diagonal
        correlations are mildly shrunk towards zero for uneven pair histories,
        affecting only transient STVU calibration.

        Parameters
        ----------
        regime_idx : ndarray
            Indices of regime-eligible assets.
        """
        cov_raw = self._cov[np.ix_(regime_idx, regime_idx)].copy()
        bc = 1.0 / np.sqrt(
            np.maximum(1.0 - self._decay ** self._obs_count[regime_idx], 1e-15)
        )
        cov_raw *= np.outer(bc, bc)
        return cov_raw

    def _process_return_row(
        self,
        returns: np.ndarray,
        active_row: np.ndarray,
        estimation_row: np.ndarray | None = None,
    ) -> None:
        r"""Update internal states with a single observation.

        Parameters
        ----------
        returns : ndarray of shape (n_assets,)
            Single observation of asset returns. May contain NaN.

        active_row : ndarray of shape (n_assets,)
            Boolean mask indicating which assets are structurally active.

        estimation_row : ndarray of shape (n_assets,) or None
            Boolean mask indicating which assets should contribute to STVU.
        """
        valid = ~np.isnan(returns) & active_row

        # Replace NaN with 0 so arithmetic doesn't propagate NaN
        filled_returns = np.where(valid, returns, 0.0)

        # Compute demeaned returns and update mean
        if self.assume_centered:
            ret = filled_returns
        else:
            # Fill NaN location entries with 0 (new assets start from 0 mean)
            filled_location = np.where(np.isnan(self.location_), 0.0, self.location_)
            # Deviation from LAGGED mean to avoid systematic downward bias
            ret = filled_returns - filled_location
            # Update mean for valid assets only (holiday: freeze)
            self.location_ = np.where(
                valid,
                self._decay * filled_location + (1.0 - self._decay) * filled_returns,
                self.location_,
            )

        ret = np.where(valid, ret, 0.0)

        # STVU update (one-step-ahead: uses previous state before covariance update)
        self._update_regime(ret, valid, estimation_row)

        # Compute outer product (with HAC correction if enabled)
        outer_product = self._compute_hac_outer(ret)

        pair_valid = np.outer(valid, valid)

        if self._separate_var_corr:
            self._update_var_corr(outer_product, valid, pair_valid)
        else:
            self._cov = np.where(
                pair_valid,
                self._decay * self._cov + (1.0 - self._decay) * outer_product,
                self._cov,
            )
        self._obs_count[valid] += 1

        # Store returns in HAC buffer
        if self._return_buffer is not None:
            buffered_ret = ret.copy()
            buffered_ret[~valid] = np.nan
            self._return_buffer.append(buffered_ret)

        # Track active/inactive transitions
        newly_inactive = self._is_active & ~active_row
        if np.any(newly_inactive):
            self._cov[newly_inactive, :] = 0.0
            self._cov[:, newly_inactive] = 0.0
            self._obs_count[newly_inactive] = 0
            if self._separate_var_corr:
                self._var[newly_inactive] = 0.0
                self._corr_state[newly_inactive, :] = 0.0
                self._corr_state[:, newly_inactive] = 0.0
                self._pair_obs_count[newly_inactive, :] = 0
                self._pair_obs_count[:, newly_inactive] = 0
            if not self.assume_centered:
                self.location_[newly_inactive] = np.nan
        self._is_active[:] = active_row

    def _update_regime(
        self,
        ret: np.ndarray,
        valid: np.ndarray,
        estimation_row: np.ndarray | None = None,
    ) -> None:
        """Compute STVU statistic using bias-corrected covariance and update
        regime state. Only ready assets participate.

        Parameters
        ----------
        ret : ndarray of shape (n_assets,)
            Demeaned (or raw) return vector, zeroed for invalid assets.

        valid : ndarray of shape (n_assets,)
            Boolean mask of assets with valid returns and active status.

        estimation_row : ndarray of shape (n_assets,) or None
            Boolean mask restricting which assets contribute to STVU.
        """
        ready = self._is_active & (self._obs_count >= self._min_observations)
        regime_valid = valid & ready
        if estimation_row is not None:
            regime_valid &= estimation_row
        n_active = int(np.sum(regime_valid))

        min_active = _MIN_ACTIVE_FOR_REGIME[self.regime_target]
        if n_active < min_active:
            return

        active_idx = np.where(regime_valid)[0]
        ret_active = ret[active_idx]

        # Use bias-corrected covariance for calibrated STVU
        cov_active = self._get_bias_corrected_cov_for_regime(active_idx)

        try:
            match self.regime_target:
                case RegimeAdjustmentTarget.MAHALANOBIS:
                    regime_statistic = squared_mahalanobis_dist(
                        X=ret_active, covariance=cov_active
                    )
                case RegimeAdjustmentTarget.DIAGONAL:
                    regime_statistic = squared_standardized_euclidean_dist(
                        returns=ret_active, covariance=cov_active
                    )
                case RegimeAdjustmentTarget.PORTFOLIO:
                    if self._regime_portfolio_weights is not None:
                        w = self._regime_portfolio_weights[:, active_idx]
                        row_sums = w.sum(axis=1)
                        keep = row_sums > 0
                        if not np.any(keep):
                            return
                        w = w[keep]
                        w = w / w.sum(axis=1, keepdims=True)
                    else:
                        w = inverse_volatility_weights(cov_active)[np.newaxis, :]
                    r_ptf = w @ ret_active
                    var_ptf = np.sum((w @ cov_active) * w, axis=1)
                    var_ptf = np.maximum(var_ptf, _NUMERICAL_THRESHOLD)
                    regime_statistic = n_active * (r_ptf**2) / var_ptf
        except (ValueError, np.linalg.LinAlgError):
            # Submatrix not yet PD (late-listed asset); skip STVU update
            return

        # Dynamic kappa based on n_active
        match self.regime_target:
            case RegimeAdjustmentTarget.MAHALANOBIS | RegimeAdjustmentTarget.DIAGONAL:
                kappa = scs.digamma(0.5 * n_active) + np.log(2.0)
            case RegimeAdjustmentTarget.PORTFOLIO:
                kappa = np.log(n_active) + scs.digamma(0.5) + np.log(2.0)

        # Transform STVU statistic for regime state EWMA.
        # For PORTFOLIO with multiple portfolios, each per-portfolio statistic
        # is transformed independently and then averaged. This preserves the
        # correct expectation because E[mean(f(X_k))] = f_target when each
        # X_k has the same marginal distribution.
        stats = np.atleast_1d(regime_statistic)
        match self.regime_method:
            case RegimeAdjustmentMethod.RMS:
                transformed_values = stats / n_active
            case RegimeAdjustmentMethod.FIRST_MOMENT:
                sqrt_stats = np.sqrt(np.maximum(stats, 0.0))
                match self.regime_target:
                    case RegimeAdjustmentTarget.PORTFOLIO:
                        denom = np.sqrt(n_active) * np.sqrt(2.0 / np.pi)
                    case RegimeAdjustmentTarget.MAHALANOBIS:
                        denom = np.sqrt(2.0) * np.exp(
                            scs.gammaln(0.5 * (n_active + 1))
                            - scs.gammaln(0.5 * n_active)
                        )
                    case RegimeAdjustmentTarget.DIAGONAL:
                        # Diagonal STVU ignores correlations, so only the second
                        # moment calibration is exact in general. Keep the
                        # first-moment normalization as the diagonal-risk proxy.
                        denom = np.sqrt(n_active)
                transformed_values = sqrt_stats / denom
            case RegimeAdjustmentMethod.LOG:
                transformed_values = (
                    np.log(np.maximum(stats, _NUMERICAL_THRESHOLD)) - kappa
                )
        transformed = float(np.mean(transformed_values))

        if self._regime_state is None:
            self._regime_state = transformed
        else:
            self._regime_state = (
                self._regime_decay * self._regime_state
                + (1.0 - self._regime_decay) * transformed
            )
        self._n_regime_observations += 1

    def _update_var_corr(
        self,
        outer_product: np.ndarray,
        valid: np.ndarray,
        pair_valid: np.ndarray,
    ) -> None:
        """Pairwise update of separate variance and correlation (DCC-style).

        Parameters
        ----------
        outer_product : ndarray of shape (n_assets, n_assets)
            HAC-adjusted outer product (or simple outer product).

        valid : ndarray of shape (n_assets,)
            Boolean mask of assets with valid returns and active status.

        pair_valid : ndarray of shape (n_assets, n_assets)
            Outer product of valid mask.
        """
        # Update variance (zero-seeded)
        hac_var = np.maximum(np.diag(outer_product), 0.0)
        self._var = np.where(
            valid,
            self._decay * self._var + (1.0 - self._decay) * hac_var,
            self._var,
        )

        # Standardize outer product for correlation update
        has_positive_var = valid & (self._var > _NUMERICAL_THRESHOLD)
        guarded_var = np.where(has_positive_var, self._var, 1.0)
        inv_std = np.where(has_positive_var, 1.0 / np.sqrt(guarded_var), 0.0)
        outer_std = outer_product * (inv_std[:, None] * inv_std[None, :])

        # Update correlation state (zero-seeded)
        self._corr_state = np.where(
            pair_valid,
            self._corr_decay * self._corr_state + (1.0 - self._corr_decay) * outer_std,
            self._corr_state,
        )
        self._pair_obs_count[pair_valid] += 1

        # Reconstruct _cov from raw (uncorrected) state for internal bookkeeping.
        # This is used only for symmetrization; output uses _bias_correct_covariance.
        active = self._is_active & (self._var > 0)
        if np.any(active):
            active_idx = np.where(active)[0]
            ix = np.ix_(active_idx, active_idx)

            std_active = np.sqrt(self._var[active_idx])
            corr_raw = self._corr_state[ix]

            diag_vals = np.diag(corr_raw)
            if np.all(diag_vals > _NUMERICAL_THRESHOLD):
                diag_sqrt = np.sqrt(np.clip(diag_vals, _NUMERICAL_THRESHOLD, None))
                inv_diag_sqrt = 1.0 / diag_sqrt
                corr = corr_raw * (inv_diag_sqrt[:, None] * inv_diag_sqrt[None, :])
                corr = np.clip(corr, -1.0, 1.0)
                corr = 0.5 * (corr + corr.T)
                np.fill_diagonal(corr, 1.0)
                self._cov[ix] = corr_to_cov(corr, std_active)

    def _compute_hac_outer(self, ret: np.ndarray) -> np.ndarray:
        """Compute HAC-adjusted outer product using Newey-West (Bartlett kernel).

        Parameters
        ----------
        ret : ndarray of shape (n_assets,)
            Current (possibly demeaned) return vector, zeroed for invalid assets.

        Returns
        -------
        ndarray of shape (n_assets, n_assets)
            HAC-adjusted outer product. If hac_lags is None, returns simple
            outer product.
        """
        outer = np.outer(ret, ret)

        if self._return_buffer is None or len(self._return_buffer) == 0:
            return outer

        for j, past_ret in enumerate(reversed(self._return_buffer), start=1):
            w_j = 1.0 - j / (self.hac_lags + 1)
            # Zero out NaN entries in lagged returns for pairwise HAC
            filled_past_ret = np.where(np.isnan(past_ret), 0.0, past_ret)
            cross = np.outer(ret, filled_past_ret)
            outer += w_j * (cross + cross.T)

        return outer

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)
