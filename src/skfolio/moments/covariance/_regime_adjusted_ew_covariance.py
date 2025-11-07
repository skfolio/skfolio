"""Regime Adjusted Exponentially Weighted Covariance Estimators."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv

from skfolio.moments.covariance._base import BaseCovariance
from skfolio.utils.stats import corr_to_cov, cov_to_corr

_WARMUP_MIN_PERIODS = 60
_NUMERICAL_THRESHOLD = 1e-12
_RIDGE_ESCALATION_FACTOR = 10.0
_MAX_RIDGE_TRIES = 3


class RegimeAdjustedEWCovariance(BaseCovariance):
    r"""Exponentially Weighted Covariance with Volatility Regime Adjustment (VRA).

    Computes an exponentially weighted covariance and applies a volatility regime
    adjustment based on one-step-ahead Mahalanobis distances. This adjustment scales the
    covariance by a single factor to better align predicted and realized risk when
    volatility regimes change more quickly than a plain EWMA can track, improving
    calibration and stabilizing optimizers while preserving the correlation structure.

    The estimator also supports separate decay factors for variance and correlation.
    Lower decay for variance allows the model to adapt faster to volatility shifts,
    while higher decay for correlation enables more stable estimation of co-movements,
    which typically require more data for reliable inference and reduce estimation
    noise. This choice also aligns with empirical evidence that volatility tends to
    mean-revert faster than correlation. Using a lower (more responsive) decay factor
    for variance can capture this behavior.

    For standard exponentially weighted covariance without bias adjustment,
    see :class:`EWCovariance`.

    Parameters
    ----------
    window_size : int, optional
        Use only the last `window_size` observations. The default (`None`) is to
        use all the data.

    corr_decay_factor : float, default=0.97
        EWMA decay factor (:math:`\lambda`) for correlation estimation.

        When `var_decay_factor` is None (default), this also controls the variance
        (diagonal) estimation:
        :math:`\Sigma_t = \lambda \Sigma_{t-1} + (1-\lambda) r_t r_t^\top`

        When `var_decay_factor` is provided, the variance and correlation are updated
        with different decay factors to capture their different dynamics.

        Higher values produce more stable (robust) estimates and lower values are
        more responsive (adaptive) but noisier:

        * :math:`\lambda \to 1`: Very stable and slow to adapt (robust to noise)
        * :math:`\lambda \to 0`: Very responsive and fast to adapt (sensitive to noise)

        **Relationship to half-life:**

        The half-life is the number of observations for the weight to decay to 50%.
        :math:`\text{half-life} = -\ln(2) / \ln(\lambda)`

        **For example**:

        * :math:`\lambda = 0.97`: 23-day half-life
        * :math:`\lambda = 0.94`: 11-day half-life
        * :math:`\lambda = 0.90`: 6-day half-life
        * :math:`\lambda = 0.80`: 3-day half-life

        **Note:** For portfolio optimization, more stable values (≥ 0.94) are generally
        preferred to avoid excessive turnover from estimation noise.

        Must satisfy :math:`0 < \lambda < 1`.

    var_decay_factor : float, optional
        EWMA decay factor (:math:`\lambda_{var}`) for variance (diagonal) estimation.

        If None (default), the same `corr_decay_factor` is used for both variance and
        correlation, resulting in standard EWMA covariance.

        If provided, enables separate decay factors for variance and correlation.
        Must satisfy :math:`0 < \lambda_{var} < 1` if specified.

    center : bool, default=False
        If True, maintain an EWMA mean :math:`\mu` and update with
        :math:`\mu_t = \lambda \mu_{t-1} + (1-\lambda) r_t` using the same
        `corr_decay_factor` as for covariance. The covariance update then uses
        demeaned returns :math:`r - \mu`.

    regime_decay_factor : float, optional
        EWMA decay factor for the volatility regime adjustment (VRA).

        The VRA is computed as:
        :math:`\phi = \sqrt{\text{EWMA}_\lambda(z/N)}`
        where :math:`z` are one-step-ahead Mahalanobis distances.

        If None (default) it is automatically calibrated as:
        :math:`\lambda_{VRA} = \lambda_{cov}^{2/3}`

        This makes the VRA approximately 1.5x more stable (longer half-life) than
        the covariance, which prevents over-reaction to short-term volatility spikes.

        **Auto-calibration examples:**

        * :math:`\lambda_{cov} = 0.94` → :math:`\lambda_{VRA} \approx 0.96` (17-day half-life)
        * :math:`\lambda_{cov} = 0.97` → :math:`\lambda_{VRA} \approx 0.98` (34-day half-life)
        * :math:`\lambda_{cov} = 0.99` → :math:`\lambda_{VRA} \approx 0.993` (99-day half-life)

        Must satisfy :math:`0 < \lambda < 1` if specified.

    regime_min_obs : int, default=20
        Minimum number of one-step-ahead comparisons before enabling VRA.
        If insufficient data, VRA defaults to 1.0 (no adjustment).

    regime_clip : tuple[float, float] or None, default=(0.7, 1.6)
        Clip :math:`\phi` to `[regime_clip[0], regime_clip[1]]` to avoid extreme swings.
        Set to None to disable clipping.

    regime_cap_quantile : float, default=0.995
        Cap per-step Mahalanobis distance at the given empirical quantile
        to guard against rare spikes. For example, 0.995 means distances above
        the 99.5th percentile are capped at that value.
        Set to 0.0 to disable capping.

    warm_length : int, optional
        Number of initial observations to use for warm-up (initialization) of the
        covariance matrix before starting EWMA updates. If None (default), it is
        automatically determined as :math:`\min(60, \max(10, n_{assets} + 1))` but
        capped at :math:`n_{observations} / 3` to leave sufficient data for EWMA.

        **Automatic warm-up examples:**

        * 100 observations, 20 assets: warm_length = 21 (leaves 79 for EWMA)
        * 200 observations, 50 assets: warm_length = 51 (leaves 149 for EWMA)
        * 500 observations, 10 assets: warm_length = 60 (leaves 440 for EWMA)

        Set explicitly to override automatic determination. Must be at least 2 and
        less than `n_observations`.

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
        Estimated covariance matrix.

    regime_multiplier_ : float
        The volatility regime adjustment factor applied.
        Equal to 1.0 if insufficient data.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during `fit`. Defined only when `X`
        has feature names that are all strings.

    Notes
    -----
    The VRA works by comparing predicted versus realized risk through one-step-ahead
    Mahalanobis distances:

    1. At each step t, compute: :math:`z^2_{t+1} = r_{t+1}^T \Sigma_t^{-1} r_{t+1}`

    2. Under correct calibration (when returns follow the predicted distribution),
       :math:`E[z^2 / N] = 1` where N = n_assets.

    3. If realized volatility exceeds predicted, :math:`z^2/N > 1`, so the multiplier
       :math:`\phi = \sqrt{\text{EWMA}(z^2/N)} > 1` scales up the covariance.

    4. The EWMA smoothing prevents overreaction to single-period spikes and provides
       a stable adjustment factor.

    This approach is related to GARCH volatility updating but applied at the
    portfolio level rather than to univariate series.

    Use `RegimeAdjustedEWCovariance` when:

    - You expect volatility regimes to change more quickly than correlations
    - You need better calibration between predicted and realized risk
    - You want to reduce bias in portfolio risk forecasts during regime transitions

    Use `EWCovariance` when:

    - You have relatively stable market conditions and don't expect significant regime changes
    - Computational speed is critical (this estimator is ~30% slower)
    - You prefer simpler, more interpretable models

    References
    ----------
    .. [1] "Multivariate exponentially weighted moving covariance matrix",
        Technometrics, Hawkins & Maboudou-Tchao (2008).

    .. [2] "Dynamic conditional correlation: A simple class of multivariate GARCH
        models", Journal of Business & Economic Statistics, Engle (2002).

    .. [3] "Computing the nearest correlation matrix - A problem from finance",
        IMA Journal of Numerical Analysis, Higham (2002)

    .. [4] "An Introduction to Multivariate Statistical Analysis", Wiley,
        Anderson (2003).

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.moments import RegimeAdjustedEWCovariance
    >>> from skfolio.preprocessing import prices_to_returns
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> # Standard EWMA with bias adjustment
    >>> model = RegimeAdjustedEWCovariance(corr_decay_factor=0.97)
    >>> model.fit(X)
    >>> print(model.regime_multiplier_)
    >>>
    >>> # Separate decay factors for variance and correlation
    >>> # (more responsive variance, more stable correlation)
    >>> model2 = RegimeAdjustedEWCovariance(
    ...     corr_decay_factor=0.97,   # correlation: 23-day half-life
    ...     var_decay_factor=0.94     # variance: 11-day half-life
    ... )
    >>> model2.fit(X)
    """

    regime_multiplier_: float

    def __init__(
        self,
        window_size: int | None = None,
        corr_decay_factor: float = 0.97,
        var_decay_factor: float | None = None,
        center: bool = False,
        regime_decay_factor: float | None = None,
        regime_min_obs: int = 20,
        regime_clip: tuple[float, float] | None = (0.7, 1.6),
        regime_cap_quantile: float = 0.995,
        warm_length: int | None = None,
        nearest: bool = True,
        higham: bool = False,
        higham_max_iteration: int = 100,
    ) -> None:
        super().__init__(
            nearest=nearest,
            higham=higham,
            higham_max_iteration=higham_max_iteration,
        )
        self.window_size = window_size
        self.corr_decay_factor = corr_decay_factor
        self.var_decay_factor = var_decay_factor
        self.center = center
        self.regime_decay_factor = regime_decay_factor
        self.regime_min_obs = regime_min_obs
        self.regime_clip = regime_clip
        self.regime_cap_quantile = regime_cap_quantile
        self.warm_length = warm_length

    def fit(self, X: npt.ArrayLike, y=None) -> RegimeAdjustedEWCovariance:
        """Fit the Regime-Adjusted Exponentially Weighted Covariance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : RegimeAdjustedEWCovariance
            Fitted estimator.
        """
        # Validate and prepare data
        X = skv.validate_data(self, X)

        if self.window_size is not None:
            X = X[-int(self.window_size) :]

        n_observations, n_assets = X.shape
        if n_observations < 2:
            raise ValueError(f"Need at least 2 observations, got {n_observations}")

        if self.regime_min_obs < 1:
            raise ValueError(f"regime_min_obs must be >= 1 (got {self.regime_min_obs})")

        if not (0.0 <= self.regime_cap_quantile <= 1.0):
            raise ValueError(
                f"regime_cap_quantile must be in [0, 1] (got {self.regime_cap_quantile})"
            )

        if self.regime_clip is not None:
            if len(self.regime_clip) != 2:
                raise ValueError(
                    f"regime_clip must be a tuple of length 2, got "
                    f"{len(self.regime_clip)}"
                )
            lo, hi = self.regime_clip
            if not (0 < lo < hi):
                raise ValueError(
                    f"regime_clip must satisfy 0 < lo < hi, got ({lo}, {hi})"
                )

        if not (0.0 < self.corr_decay_factor < 1.0):
            raise ValueError(
                f"corr_decay_factor must satisfy 0 < λ < 1 (got "
                f"{self.corr_decay_factor}). Valid range: (0, 1), typical values: "
                f"0.90-0.99."
            )

        # Balance between stable initialization and leaving enough data for EWMA
        if self.warm_length is not None:
            if not isinstance(self.warm_length, int) or self.warm_length < 2:
                raise ValueError(
                    f"warm_length must be an integer >= 2, got {self.warm_length}"
                )
            if self.warm_length >= n_observations:
                raise ValueError(
                    f"warm_length ({self.warm_length}) must be less than "
                    f"n_observations ({n_observations})"
                )
            warm_length = self.warm_length
        else:
            # Ideal warm-up for stability: at least 10 or n_assets+1, but capped at 60
            ideal_warm = min(_WARMUP_MIN_PERIODS, max(10, n_assets + 1))
            # Never use more than 1/3 of available data for warm-up (leave 2/3 for EWMA)
            max_warm = max(2, n_observations // 3)
            warm_length = min(ideal_warm, max_warm)

            if n_observations < ideal_warm + 10:
                warnings.warn(
                    f"Only {n_observations} observations available. "
                    f"Recommended: at least {ideal_warm + 10} for {n_assets} assets. "
                    f"Results may be unstable.",
                    UserWarning,
                    stacklevel=2,
                )

        # Initialize covariance matrix with sample covariance if possible
        if warm_length >= n_assets + 1:
            cov = np.cov(X[:warm_length], rowvar=False, ddof=1)
        else:
            # Use diagonal with individual variances
            variances = np.var(X[:warm_length], axis=0, ddof=1)
            if np.any(variances < 1e-15):
                warnings.warn(
                    f"Near-zero variance detected during initialization "
                    f"(min var={variances.min():.2e}). Check for constant assets "
                    f"or data quality issues.",
                    UserWarning,
                    stacklevel=2,
                )
            cov = np.diag(np.maximum(variances, 1e-15))

        if self.var_decay_factor is not None:
            if not (0.0 < self.var_decay_factor < 1.0):
                raise ValueError(
                    f"var_decay_factor must satisfy 0 < λ < 1 "
                    f"(got {self.var_decay_factor}). Valid range: (0, 1), typical "
                    f"values: 0.90-0.99."
                )
            separate_var_corr = True
            var = np.diag(cov)
            unnormalized_corr, _ = cov_to_corr(cov)
        else:
            separate_var_corr = False
            var = None
            unnormalized_corr = None

        if self.regime_decay_factor is None:
            regime_decay_factor = self.corr_decay_factor ** (2.0 / 3.0)
        else:
            regime_decay_factor = self.regime_decay_factor
            if not (0.0 < regime_decay_factor < 1.0):
                raise ValueError(
                    f"regime_decay_factor must satisfy 0 < λ < 1 (got "
                    f"{regime_decay_factor}). A value of 1.0 means infinite memory with "
                    f"no updates."
                )
            if regime_decay_factor >= 0.995:
                warnings.warn(
                    f"regime_decay_factor = {regime_decay_factor} leads to "
                    f" excessive memory (half-life > 138 days) that may desynchronize "
                    f"VRA from covariance evolution and cause erratic behavior. "
                    f"Consider using auto-calibration (regime_decay_factor=None) or "
                    f"a value < 0.995 for more stable results.",
                    UserWarning,
                    stacklevel=2,
                )

        mu = X[:warm_length].mean(axis=0) if self.center else None

        dists = []
        for t in range(warm_length - 1, n_observations - 1):
            r_next = X[t + 1]

            # Compute Mahalanobis distance between the next return vector and the
            # covariance at the prior step
            dists.append(_squared_mahalanobis(returns=r_next, covariance=cov))

            # De-mean
            if self.center:
                mu = (
                    self.corr_decay_factor * mu
                    + (1.0 - self.corr_decay_factor) * r_next
                )
                ret = r_next - mu
            else:
                ret = r_next

            # Update covariance
            if separate_var_corr:
                # Update variance with its own decay factor
                var = self.var_decay_factor * var + (1.0 - self.var_decay_factor) * (
                    ret**2
                )
                # Standardize returns using current variance estimate
                std = np.sqrt(var)
                if np.any(std < _NUMERICAL_THRESHOLD):
                    warnings.warn(
                        f"Near-zero standard deviation detected (min std={std.min():.2e}). "
                        "This may indicate constant assets or data quality issues.",
                        UserWarning,
                        stacklevel=2,
                    )
                std = np.where(std > _NUMERICAL_THRESHOLD, std, _NUMERICAL_THRESHOLD)
                r_std = ret / std
                # Update unnormalized corr matrix (DCC-style GARCH approach)
                unnormalized_corr = self.corr_decay_factor * unnormalized_corr + (
                    1.0 - self.corr_decay_factor
                ) * np.outer(r_std, r_std)
                # Normalize unnormalized corr to proper correlation (guarantees diag=1)
                d = np.sqrt(
                    np.clip(np.diag(unnormalized_corr), _NUMERICAL_THRESHOLD, None)
                )
                d_inv = np.diag(1.0 / d)
                corr = d_inv @ unnormalized_corr @ d_inv
                # Recombine correlation with current variance to get covariance
                cov = corr_to_cov(corr, std)
            else:
                # Standard EWMA
                cov = self.corr_decay_factor * cov + (
                    1.0 - self.corr_decay_factor
                ) * np.outer(ret, ret)

            # Re-symmetrize to prevent floating-point drift in iterative updates
            # EWMA updates can introduce small asymmetries due to numerical precision
            cov = 0.5 * (cov + cov.T)

        # Compute VRA from Mahalanobis distances
        if len(dists) < max(self.regime_min_obs, 1):
            regime_multiplier = 1.0
        else:
            dists = np.asarray(dists, dtype=float)
            if self.regime_cap_quantile > 0:
                dists = np.minimum(
                    dists, np.percentile(dists, self.regime_cap_quantile * 100)
                )
            # Compute EWMA of normalized Mahalanobis distances
            dists_normalized = dists / n_assets
            bias_ratio = dists_normalized[0]
            for dist in dists_normalized[1:]:
                bias_ratio = (
                    regime_decay_factor * bias_ratio
                    + (1.0 - regime_decay_factor) * dist
                )
            regime_multiplier = np.sqrt(bias_ratio)

        if self.regime_clip is not None and np.isfinite(regime_multiplier):
            lo, hi = self.regime_clip
            regime_multiplier = np.clip(regime_multiplier, lo, hi)

        self.regime_multiplier_ = float(regime_multiplier)
        self._set_covariance(regime_multiplier**2 * cov)
        return self


def _squared_mahalanobis(
    returns: np.ndarray,
    covariance: np.ndarray,
    ridge_scale: float = _NUMERICAL_THRESHOLD,
    max_tries: int = _MAX_RIDGE_TRIES,
) -> float:
    """
    Compute z = r^T Sigma^{-1} r using a Cholesky-based solve.

    Parameters
    ----------
    returns : ndarray of shape (n_assets,)
        Asset return vector.

    covariance : ndarray of shape (n_assets, n_assets)
        Covariance matrix.

    ridge_scale : float, default=1e-12
        Relative ridge size, as a fraction of the average covariance diagonal.

    max_tries : int, default=3
        Maximum number of ridge escalations.

    Returns
    -------
    float
        Squared Mahalanobis distance.

    Raises
    ------
    ValueError
        If shapes are inconsistent, inputs have non-finite values, or Cholesky fails
        after retries.
    """
    returns = np.asarray(returns, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)

    # Symmetrize defensively
    covariance = 0.5 * (covariance + covariance.T)

    # Scale ridge to matrix level
    ridge = ridge_scale * np.mean(np.diag(covariance))

    for _ in range(max_tries):
        covariance[np.diag_indices_from(covariance)] += ridge
        try:
            chol = np.linalg.cholesky(covariance)
            x = np.linalg.solve(chol.T, np.linalg.solve(chol, returns))
            # Guard against tiny negative due to round-off
            return max(0.0, float(returns @ x))
        except np.linalg.LinAlgError:
            ridge *= _RIDGE_ESCALATION_FACTOR

    raise ValueError(
        f"Cholesky failed after {max_tries} attempts; last "
        f"ridge={ridge / _RIDGE_ESCALATION_FACTOR:.3e}."
    )
