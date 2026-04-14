"""Regime Adjusted Exponentially Weighted Variance Estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numbers
import warnings
from collections import deque

import numpy as np
import numpy.typing as npt
import scipy.special as scs
import sklearn.utils.validation as skv

from skfolio.moments.covariance import RegimeAdjustmentMethod
from skfolio.moments.variance._base import BaseVariance
from skfolio.utils.tools import _validate_mask, half_life_to_decay_factor

_NUMERICAL_THRESHOLD = 1e-12
_FITTED_ATTR = "variance_"


class RegimeAdjustedEWVariance(BaseVariance):
    r"""Exponentially weighted variance estimator with regime adjustment via the
    Short-Term Volatility Update (STVU) [1]_.

    This is the variance-only counterpart of
    :class:`~skfolio.moments.covariance.RegimeAdjustedEWCovariance`, assuming zero
    correlation. This is appropriate when:

    * Estimating **idiosyncratic (specific) risk** in factor models, where residual
      returns are uncorrelated by construction
    * Working with **orthogonalized** or **uncorrelated** return series
    * The full covariance structure is not needed or is constructed separately

    This estimator computes per-asset exponentially weighted variances and applies a
    scalar multiplier :math:`\phi_t` to improve risk calibration when volatility
    regimes change more quickly than a plain EWMA can track.

    Additionally, this estimator supports optional Newey-West HAC (Heteroskedasticity
    and Autocorrelation Consistent) correction via the `hac_lags` parameter. This
    adjusts for serial correlation in returns.

    **NaN handling:**

    The estimator handles missing data (NaN returns) caused by late listings,
    delistings, and holidays using EWMA updates together with `active_mask`.
    An asset with `active_mask=True` is treated as active at time :math:`t`.
    If its return is finite, the EWMA is updated normally. If its return is
    NaN, the observation is treated as a holiday and the previous variance is
    kept. An asset with `active_mask=False` is treated as inactive, for
    example during pre-listing or post-delisting periods, and its variance is
    set to NaN.

    * **Active with valid return**: Normal EWMA update.
    * **Active with NaN return (holiday)**: Freeze; the previous variance is
      kept.
    * **Inactive** (`active_mask=False`): Variance is set to NaN.

    When `active_mask` is not provided, trailing NaN returns are treated as
    holidays and the variance is frozen. When an asset becomes active again
    after an inactive period, its variance restarts from a zero prior and
    receives per-asset bias correction at output time.

    **Late-listing bias correction:**

    The EWMA recursion is initialized at zero for every asset. This
    zero-initialization introduces a transient downward scale bias: after
    :math:`n_i` valid observations, the raw EWMA weights sum to
    :math:`(1 - \lambda^{n_i})` instead of 1. At output time, a per-asset
    correction removes this bias:

    .. math::

        \hat{\sigma}^2_i = \frac{S_i}{1 - \lambda^{n_i}}

    where :math:`S_i` is the raw internal EWMA accumulator. For assets with a
    long history, the correction is negligible (:math:`\lambda^{n_i} \to 0`).

    The ``min_observations`` parameter controls a warm-up period: an asset's
    variance estimate remains NaN in the output until it has accumulated enough
    valid observations for a reliable estimate.

    **Estimation universe for STVU:**

    An optional `estimation_mask` defines the estimation universe used for the
    cross-sectional STVU statistic without affecting per-asset EWMA variance
    updates. The STVU is computed in a one-step-ahead manner: the return
    observed at time :math:`t` is standardized by the bias-corrected variance
    estimate available at time :math:`t-1`, and only assets that were already
    above `min_observations` before time :math:`t` contribute to the regime
    signal. This is important because the STVU multiplier is derived from a
    cross-sectional average of standardized squared returns: noisy or illiquid
    assets with unreliable variance estimates can inflate or deflate the
    statistic, distorting the regime multiplier applied to all variances.

    Parameters
    ----------
    half_life : float, default=40
        Half-life of the exponential weights in number of observations.

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

    hac_lags : int, optional
        Number of lags for Newey-West HAC (Heteroskedasticity and Autocorrelation
        Consistent) correction. If None (default), no HAC correction is applied.

        When enabled, the variance update uses HAC-adjusted squared returns instead
        of simple squared returns, accounting for autocorrelation:

        .. math::

            \text{hac\_var}_i = r_{i,t}^2 + 2 \sum_{j=1}^{L} w_j \cdot r_{i,t} \cdot r_{i,t-j}

        where :math:`w_j = 1 - j/(L+1)` is the Bartlett kernel weight.

        Typical values:
            * Daily equity data: 3-5 lags (weak autocorrelation from microstructure)
            * High-frequency data: 5-10 lags (stronger autocorrelation)
            * Monthly data: 1-2 lags

        Must be a positive integer if specified.

    regime_method : RegimeAdjustmentMethod, default=RegimeAdjustmentMethod.FIRST_MOMENT
        Method used to transform the update statistic into the volatility multiplier :math:`\phi`:

        - `LOG`: Robust to outliers (log compresses extremes)
        - `FIRST_MOMENT`: Calibrates the first moment of the standardized risk
          statistic
        - `RMS`: :math:`\chi^2` calibration (sensitive to extremes)

    regime_half_life : float, optional
        Half-life for smoothing the volatility regime signal, in number of
        observations.

        The regime signal is built from one-step-ahead standardized returns
        and then transformed into the multiplier :math:`\phi` according to
        `regime_method`. A shorter `regime_half_life` makes the multiplier
        react faster to abrupt changes in realized risk. A longer one
        produces a smoother, slower moving adjustment.

        If None (default), it is automatically calibrated as:
        :math:`\text{regime-half-life} = 0.5 \times \text{half-life}`

        This makes the STVU more responsive (shorter half-life) than the variance,
        allowing it to quickly rescale risk when realized volatility deviates from
        the slower EWMA estimate.

    regime_multiplier_clip : tuple[float, float] or None, default=(0.7, 1.6)
        Clip to avoid extreme swings in the regime multiplier.
        Set to None to disable clipping. The multiplier is applied to the covariance
        as :math:`\phi^2 \Sigma`.

        Default bounds rationale:
            * Lower bound (0.7): Limits volatility reduction to 30%, equivalent to a
              minimum variance scale of :math:`0.7^2 = 0.49`
            * Upper bound (1.6): Limits volatility increase to 60%, equivalent to a
              maximum variance scale of :math:`1.6^2 = 2.56`

    regime_min_observations : int, optional
        Minimum number of one-step-ahead comparisons before enabling STVU.
        If insufficient data, STVU defaults to 1.0 (no adjustment).

        If None (default), it is automatically set to
        `int(regime_half_life)`, ensuring the STVU EWMA has seen roughly one
        half-life of data before being applied.

    min_observations : int, optional
        Minimum number of valid observations per asset before its variance estimate
        is considered reliable and exposed in the output `variance_`. Until this
        threshold is reached, the asset's variance estimate remains NaN.

        The default (`None`) uses `int(half_life)` as the threshold, ensuring
        the late-listing initialization bias has decayed to at most 50%. Set to
        1 to disable warm-up entirely.

    assume_centered : bool, default=True
        If True (default), the EWMA update uses raw returns without demeaning. This
        is the standard convention for EWMA variance estimation in finance.
        If False, returns are demeaned using an EWMA mean estimate before computing
        the variance update, and `location_` tracks the EWMA mean.

        .. note::
            For factor model residuals, centering is typically not needed as residuals
            should already have zero mean by construction. Set to False only if
            residuals exhibit persistent non-zero means.

    Attributes
    ----------
    variance_ : ndarray of shape (n_assets,)
        Estimated regime-adjusted variances.

    regime_multiplier_ : float
        The volatility regime adjustment factor applied.
        Equal to 1.0 if insufficient data or no regime adjustment needed.

    location_ : ndarray of shape (n_assets,)
        Estimated location, i.e. the estimated mean.
        When `assume_centered=True`, this is zero.
        When `assume_centered=False`, this is the EWMA mean estimate.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X`
        has feature names that are all strings.

    References
    ----------
    .. [1] "The Elements of Quantitative Investing", Wiley Finance,
        Giuseppe Paleologo (2025).

    .. [2] "Multivariate exponentially weighted moving covariance matrix",
        Technometrics, Hawkins & Maboudou-Tchao (2008).

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.moments import RegimeAdjustedEWVariance, RegimeAdjustmentMethod
    >>> from skfolio.preprocessing import prices_to_returns
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> # Standard EWMA with STVU
    >>> model = RegimeAdjustedEWVariance(half_life=23)
    >>> model.fit(X)
    >>> print(model.regime_multiplier_)
    >>>
    >>> # With LOG method for robustness to outliers
    >>> model2 = RegimeAdjustedEWVariance(
    ...     half_life=11,
    ...     regime_method=RegimeAdjustmentMethod.LOG
    ... )
    >>> model2.fit(X)
    >>>
    >>> # With Newey-West HAC correction for autocorrelation
    >>> model3 = RegimeAdjustedEWVariance(
    ...     half_life=23,
    ...     hac_lags=5    # 5-lag Newey-West correction
    ... )
    >>> model3.fit(X)
    >>>
    >>> # With an estimation universe focused on specific assets
    >>> estimation_mask = np.ones((len(X), X.shape[1]), dtype=bool)
    >>> estimation_mask[:, :5] = False  # Exclude first 5 assets from STVU
    >>> model4 = RegimeAdjustedEWVariance(half_life=23)
    >>> model4.fit(X, estimation_mask=estimation_mask)
    """

    regime_multiplier_: float

    def __init__(
        self,
        half_life: float = 40,
        hac_lags: int | None = None,
        regime_method: RegimeAdjustmentMethod = RegimeAdjustmentMethod.FIRST_MOMENT,
        regime_half_life: float | None = None,
        regime_multiplier_clip: tuple[float, float] | None = (0.7, 1.6),
        regime_min_observations: int | None = None,
        min_observations: int | None = None,
        assume_centered: bool = True,
    ) -> None:
        super().__init__(assume_centered=assume_centered)
        self.half_life = half_life
        self.hac_lags = hac_lags
        self.regime_method = regime_method
        self.regime_half_life = regime_half_life
        self.regime_multiplier_clip = regime_multiplier_clip
        self.regime_min_observations = regime_min_observations
        self.min_observations = min_observations

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        *,
        estimation_mask: npt.ArrayLike | None = None,
        active_mask: npt.ArrayLike | None = None,
    ) -> RegimeAdjustedEWVariance:
        """Fit the Regime-Adjusted Exponentially Weighted Variance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Idiosyncratic (specific) residual returns per asset, typically obtained
            from a factor model regression. NaN values are allowed and handled robustly.

        y : Ignored
            Not used, present for API consistency by convention.

        estimation_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating which active assets should belong to the
            estimation universe for the cross-sectional STVU statistic on each day.

            - If None (default), all active assets with finite returns are used.
            - If provided, only assets where the mask is True contribute to the
              regime multiplier calculation on that day.

            Per-asset EWMA variance updates still use all active assets with finite
            returns; this parameter only affects the cross-sectional regime adjustment
            calculation.

            Use cases:
                * Focus on liquid assets to reduce noise from thinly traded securities
                * Exclude assets with suspected data quality issues
                * Match the estimation universe used in downstream models

        active_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating whether each asset is structurally active at each
            observation. Use this to distinguish between holidays (`active_mask=True`
            and NaN return: variance is frozen) and inactive periods such as pre-listing
            or post-delisting (`active_mask=False`: variance is set to NaN). If `None`
            (default), all pairs are assumed active and NaN returns are treated as
            holidays (variance frozen).

            When an asset becomes active again after an inactive period, its variance
            restarts from a zero prior with per-asset bias correction.

        Returns
        -------
        self : RegimeAdjustedEWVariance
            Fitted estimator.
        """
        self._reset()
        self.partial_fit(X, y, estimation_mask=estimation_mask, active_mask=active_mask)
        return self

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        *,
        estimation_mask: npt.ArrayLike | None = None,
        active_mask: npt.ArrayLike | None = None,
    ) -> RegimeAdjustedEWVariance:
        """Incrementally fit the estimator with new observations.

        This method allows online/streaming updates to the variance estimates.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Idiosyncratic (specific) residual returns per asset.
            NaN values are allowed and handled robustly.

        y : Ignored
            Not used, present for API consistency by convention.

        estimation_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating which active assets belong to the estimation
            universe for the cross-sectional STVU statistic on each day. See
            `fit` for details.

        active_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating whether each asset is structurally active at
            each observation. See `fit` for details.

        Returns
        -------
        self : RegimeAdjustedEWVariance
            Fitted estimator.
        """
        first_call = not hasattr(self, _FITTED_ATTR)
        X = skv.validate_data(
            self, X, reset=first_call, dtype=float, ensure_all_finite="allow-nan"
        )
        estimation_mask = _validate_mask(
            X=X,
            mask=estimation_mask,
            name="estimation_mask",
        )
        active_mask = _validate_mask(X=X, mask=active_mask, name="active_mask")

        if first_call:
            self._validate_params()
            self._initialize()

        # Process each observation
        for t, returns in enumerate(X):
            est_mask = estimation_mask[t] if estimation_mask is not None else None
            active_row = active_mask[t] if active_mask is not None else None
            self._process_return_row(returns, est_mask, active_row)

        # Bias correction
        variance = self._var.copy()
        correction = np.where(
            self._obs_count > 0,
            1.0 / np.maximum(1.0 - self._decay**self._obs_count, 1e-15),
            1.0,
        )
        variance *= correction

        # NaN-mask assets below min_observations threshold or inactive
        not_ready = ~self._is_active | (self._obs_count < self._min_observations)
        if np.any(not_ready):
            variance[not_ready] = np.nan

        if not self.assume_centered and np.any(~self._is_active):
            self.location_[~self._is_active] = np.nan

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
        self.variance_ = regime_multiplier**2 * variance
        return self

    def _validate_params(self):
        if not isinstance(self.regime_method, RegimeAdjustmentMethod):
            raise ValueError(
                f"regime_method must be a RegimeAdjustmentMethod, got "
                f"{self.regime_method!r}"
            )

        if self.min_observations is None:
            self._min_observations = max(1, int(self.half_life))
        else:
            if self.min_observations < 1:
                raise ValueError(
                    f"min_observations must be >= 1, got {self.min_observations}"
                )
            self._min_observations = self.min_observations

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
                    f"regime_half_life = {self.regime_half_life} leads to "
                    f"excessive memory that may desynchronize "
                    f"STVU from variance evolution and cause erratic behavior. "
                    f"Consider using auto-calibration (regime_half_life=None) or "
                    f"a value <= 138 for more stable results.",
                    UserWarning,
                    stacklevel=2,
                )

    def _initialize(self):
        n_assets = self.n_features_in_
        self._decay = half_life_to_decay_factor(self.half_life)
        self._var = np.zeros(n_assets)
        self._is_active = np.ones(n_assets, dtype=bool)
        self._obs_count = np.zeros(n_assets, dtype=int)
        self._kappa = scs.digamma(0.5) + np.log(2.0)
        self._expected_abs_z = np.sqrt(2.0 / np.pi)
        self._regime_state = None
        self._n_regime_observations = 0

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

    def _process_return_row(
        self,
        returns: np.ndarray,
        estimation_mask: np.ndarray | None,
        active_row: np.ndarray | None,
    ):
        """Process a single row of returns.

        Parameters
        ----------
        returns : ndarray of shape (n_assets,)
            Return vector for this observation.

        estimation_mask : ndarray of shape (n_assets,) or None
            Boolean mask indicating which assets should contribute to STVU.

        active_row : ndarray of shape (n_assets,) or None
            Boolean mask indicating which assets are structurally active.
        """
        finite_mask = np.isfinite(returns)
        valid = finite_mask if active_row is None else (finite_mask & active_row)
        prev_obs_count = self._obs_count.copy()

        # Track active/inactive transitions
        if active_row is not None:
            newly_inactive = self._is_active & ~active_row
            if np.any(newly_inactive):
                self._var[newly_inactive] = 0.0
                self._obs_count[newly_inactive] = 0
                if not self.assume_centered:
                    self.location_[newly_inactive] = np.nan
            self._is_active[:] = active_row
        else:
            self._is_active[:] = True

        if not np.any(valid):
            return

        # Compute demeaned returns (or raw if assume_centered).
        # Deviation from LAGGED mean for correct one-step-ahead calibration.
        if self.assume_centered:
            ret = returns
        else:
            loc = np.where(np.isnan(self.location_), 0.0, self.location_)
            ret = returns - loc
            self.location_[valid] = (
                self._decay * loc[valid] + (1.0 - self._decay) * returns[valid]
            )

        # STVU: compute z^2 using the pre-update bias-corrected variance.
        # Only assets that were already ready before this observation
        # contribute to the one-step-ahead regime signal.
        ready = valid & (prev_obs_count >= self._min_observations)
        z2 = np.full_like(returns, np.nan)
        has_var = ready & (self._var > _NUMERICAL_THRESHOLD)
        if np.any(has_var):
            bc = 1.0 / np.maximum(1.0 - self._decay ** prev_obs_count[has_var], 1e-15)
            var_corrected = self._var[has_var] * bc
            z2[has_var] = ret[has_var] ** 2 / var_corrected

        # EWMA variance update
        squared_returns = self._compute_hac_squared(ret, valid)
        valid_idx = np.flatnonzero(valid)
        self._var[valid_idx] = (
            self._decay * self._var[valid_idx]
            + (1.0 - self._decay) * squared_returns[valid_idx]
        )
        self._obs_count[valid] += 1

        if self._return_buffer is not None:
            buffered_ret = ret.copy()
            buffered_ret[~valid] = np.nan
            self._return_buffer.append(buffered_ret)

        # STVU update on regime-eligible assets
        regime_mask = ready.copy()
        if estimation_mask is not None:
            regime_mask &= estimation_mask

        if not np.any(regime_mask):
            return

        z2_valid = z2[regime_mask]
        if not np.any(np.isfinite(z2_valid)):
            return

        match self.regime_method:
            case RegimeAdjustmentMethod.RMS:
                transformed = np.nanmean(z2_valid)
            case RegimeAdjustmentMethod.FIRST_MOMENT:
                transformed = (
                    np.nanmean(np.sqrt(np.maximum(z2_valid, 0.0)))
                    / self._expected_abs_z
                )
            case RegimeAdjustmentMethod.LOG:
                log_z2 = np.log(np.maximum(z2_valid, _NUMERICAL_THRESHOLD))
                transformed = np.nanmean(log_z2) - self._kappa

        if self._regime_state is None:
            self._regime_state = transformed
        else:
            self._regime_state = (
                self._regime_decay * self._regime_state
                + (1.0 - self._regime_decay) * transformed
            )
        self._n_regime_observations += 1

    def _compute_hac_squared(
        self,
        ret: np.ndarray,
        finite_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute HAC-adjusted squared returns using Newey-West (Bartlett kernel).

        Parameters
        ----------
        ret : ndarray of shape (n_assets,)
            Current (possibly demeaned) return vector (may contain NaNs).

        finite_mask : ndarray of shape (n_assets,)
            Boolean mask indicating which returns are finite.

        Returns
        -------
        ndarray of shape (n_assets,)
            HAC-adjusted squared returns. If hac_lags is None, returns simple
            squared returns. Assets with NaN in current returns remain NaN.
        """
        squared = ret**2

        if self._return_buffer is None or len(self._return_buffer) == 0:
            return squared

        # Add lagged cross-products with Bartlett kernel weights
        # Treat NaN in past returns as 0 (no contribution from missing lagged values)
        for j, past_ret in enumerate(reversed(self._return_buffer), start=1):
            w_j = 1.0 - j / (self.hac_lags + 1)
            past_ret_clean = np.nan_to_num(past_ret, nan=0.0)
            squared += 2.0 * w_j * ret * past_ret_clean

        # Clip to non-negative for finite values (HAC terms can cause negative values)
        result = np.where(finite_mask, np.maximum(squared, 0.0), squared)
        return result

    def _reset(self) -> None:
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)
