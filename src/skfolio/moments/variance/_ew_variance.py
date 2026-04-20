"""Exponentially Weighted Variance Estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import sklearn.utils.validation as skv

from skfolio.moments.variance._base import BaseVariance
from skfolio.typing import ArrayLike, BoolArray, FloatArray
from skfolio.utils.tools import (
    _validate_mask,
    apply_window_size,
    half_life_to_decay_factor,
)

_FITTED_ATTR = "variance_"


class EWVariance(BaseVariance):
    r"""Exponentially Weighted Variance estimator.

    This is the variance-only counterpart of
    :class:`~skfolio.moments.covariance.EWCovariance`, computing only the diagonal
    elements (variances) and assuming zero correlation. This is appropriate when:

    * Estimating **idiosyncratic (specific) risk** in factor models, where residual
      returns are uncorrelated by construction
    * Working with **orthogonalized** or **uncorrelated** return series
    * The full covariance structure is not needed or is constructed separately

    This estimator uses the recursive EWMA formula:

    .. math::

        \sigma^2_{i,t} = \lambda \sigma^2_{i,t-1} + (1-\lambda) r_{i,t}^2

    where :math:`\lambda` is the decay factor, which determines how much weight is
    given to past observations. It is computed from the half-life parameter:

    .. math::

        \lambda = 2^{-1/\text{half-life}}

    The half-life is the number of observations for the weight to decay to 50%.

    This estimator supports both batch fitting via :meth:`fit` and incremental
    updates via :meth:`partial_fit`, making it suitable for online learning
    scenarios.

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

    When `active_mask` is not provided, trailing NaN returns are ambiguous:
    they could correspond either to holidays, in which case the variance is
    frozen, or to inactive periods, in which case the variance is set to NaN.

    **Late-listing bias correction:**

    When an asset becomes active (late listing), the EWMA recursion is
    initialized at zero rather than at the first squared return. This
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

    assume_centered : bool, default=True
        If True (default), the EWMA update uses raw returns without demeaning. This
        is the standard convention for EWMA variance estimation in finance.
        If False, returns are demeaned using an EWMA mean estimate before computing
        the variance update, and `location_` tracks the EWMA mean.

    min_observations : int, optional
        Minimum number of valid observations per asset before its variance estimate
        is considered reliable and exposed in the output ``variance_``. Until this
        threshold is reached, the asset's variance estimate remains NaN.

        The default (``None``) uses ``int(half_life)`` as the threshold, ensuring
        the late-listing initialization bias has decayed to at most 50%. Set to
        1 to disable warm-up entirely.

    window_size : int, optional
        Window size to truncate data to the last `window_size` observations before
        fitting. Only applies to the initial :meth:`fit` call (or equivalently, the
        first :meth:`partial_fit` call); subsequent :meth:`partial_fit` calls use
        all provided data.

        This is a computational optimization for very long time series. Due to
        exponential decay, observations far in the past contribute negligibly to
        the current estimate. For example, with half-life = 23 (:math:`\lambda = 0.97`),
        observations beyond ~150 periods contribute less than 1% to the estimate.
        Truncating to a reasonable window (e.g., 252 trading days) speeds up
        computation without materially affecting results.

        The default (``None``) uses all available data.

    Attributes
    ----------
    variance_ : ndarray of shape (n_assets,)
        Estimated variance vector. Contains NaN for assets that are inactive
        or that have not yet accumulated ``min_observations`` valid observations.

    location_ : ndarray of shape (n_assets,)
        Estimated location (mean). If ``assume_centered=True``, this is zeros.
        Otherwise, it tracks the EWMA mean of returns. Contains NaN for inactive
        assets when ``assume_centered=False``.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    Examples
    --------
    >>> import numpy as np
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.moments import EWVariance
    >>> from skfolio.preprocessing import prices_to_returns
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>>
    >>> # Batch fitting
    >>> model = EWVariance(half_life=40)
    >>> model.fit(X)
    >>> print(model.variance_.shape)
    >>>
    >>> # Streaming updates with partial_fit
    >>> model2 = EWVariance(half_life=20)
    >>> model2.partial_fit(X[:100])  # Initial fit
    >>> model2.partial_fit(X[100:200])  # Update with new data
    >>> model2.partial_fit(X[200:])  # Continue updating
    >>>
    >>> # NaN-aware fitting with active_mask
    >>> # Asset 2 is listed starting from observation 50
    >>> active_mask = np.ones(X.shape, dtype=bool)
    >>> active_mask[:50, 2] = False
    >>> X_nan = X.copy()
    >>> X_nan[:50, 2] = np.nan
    >>> model3 = EWVariance(half_life=40)
    >>> model3.fit(X_nan, active_mask=active_mask)
    """

    def __init__(
        self,
        half_life: float = 40,
        assume_centered: bool = True,
        min_observations: int | None = None,
        window_size: int | None = None,
    ) -> None:
        super().__init__(assume_centered=assume_centered)
        self.half_life = half_life
        self.min_observations = min_observations
        self.window_size = window_size

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        active_mask: ArrayLike | None = None,
    ) -> EWVariance:
        """Fit the Exponentially Weighted Variance estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets. NaN values are allowed and handled
            robustly.

        y : Ignored
            Not used, present for API consistency by convention.

        active_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating whether each asset is structurally active at
            each observation. Use this to distinguish between holidays
            (``active_mask=True`` and NaN return: variance is frozen) and
            inactive periods such as pre-listing or post-delisting
            (``active_mask=False``: variance is set to NaN). If ``None``
            (default), all assets are assumed active.

        Returns
        -------
        self : EWVariance
            Fitted estimator.
        """
        self._reset()
        return self.partial_fit(X, y, active_mask=active_mask)

    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike | None = None,
        *,
        active_mask: ArrayLike | None = None,
    ) -> EWVariance:
        """Incrementally fit the Exponentially Weighted Variance estimator.

        This method allows for streaming/online updates to the variance estimate.
        Each call updates the internal state with new observations.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets. NaN values are allowed and handled
            robustly.

        y : Ignored
            Not used, present for API consistency by convention.

        active_mask : array-like of shape (n_observations, n_assets), optional
            Boolean mask indicating whether each asset is structurally active at
            each observation. See `fit` for details.

        Returns
        -------
        self : EWVariance
            Fitted estimator.
        """
        first_call = not hasattr(self, _FITTED_ATTR)
        X = skv.validate_data(
            self, X, reset=first_call, dtype=float, ensure_all_finite="allow-nan"
        )
        active_mask = _validate_mask(X=X, mask=active_mask, name="active_mask")

        if first_call:
            if self.window_size is not None:
                X = apply_window_size(X, window_size=self.window_size)
                if active_mask is not None:
                    active_mask = apply_window_size(
                        active_mask, window_size=self.window_size
                    )
            self._validate_params()
            self._initialize()

        if active_mask is not None:
            for returns, active_row in zip(X, active_mask, strict=True):
                self._process_return_row(returns, active_row)
        elif self.assume_centered and not np.isnan(X).any():
            self._process_batch_no_nan(X)
        else:
            all_active = np.ones(self.n_features_in_, dtype=bool)
            for returns in X:
                self._process_return_row(returns, all_active)

        variance = self._bias_correct_variance()

        nan_mask = ~self._is_active | (self._obs_count < self._min_observations)
        if np.any(nan_mask):
            variance[nan_mask] = np.nan

        if not self.assume_centered and np.any(~self._is_active):
            self.location_[~self._is_active] = np.nan

        self.variance_ = variance
        return self

    def _validate_params(self) -> None:
        """Validate parameters."""
        if self.half_life <= 0:
            raise ValueError(
                f"half_life must be positive (got {self.half_life}). "
                f"Typical values: 10-100 observations."
            )

        if self.window_size is not None and self.window_size < 1:
            raise ValueError(
                f"window_size must be a positive integer, got {self.window_size}"
            )

        if self.min_observations is None:
            self._min_observations = max(1, int(self.half_life))
        else:
            if self.min_observations < 1:
                raise ValueError(
                    f"min_observations must be >= 1, got {self.min_observations}"
                )
            self._min_observations = self.min_observations

    def _initialize(self) -> None:
        r"""Initialize internal state.

        ``_var`` is zero-initialized (never NaN) so that EWMA arithmetic needs
        no NaN-fill step. Active state is tracked separately; NaN is applied
        only at output time.
        """
        n_assets = self.n_features_in_
        self._decay = half_life_to_decay_factor(self.half_life)
        self._var = np.zeros(n_assets)
        self._is_active = np.ones(n_assets, dtype=bool)
        self._obs_count = np.zeros(n_assets, dtype=int)
        if self.assume_centered:
            self.location_ = np.zeros(n_assets)
        else:
            self.location_ = np.full(n_assets, np.nan)

    def _process_return_row(self, returns: FloatArray, active_row: BoolArray) -> None:
        """Update internal EWMA state with a single observation.

        Only assets with valid returns are updated; all others are frozen.
        When an asset becomes inactive, its state is reset so that bias
        correction restarts if it becomes active again.

        Parameters
        ----------
        returns : ndarray of shape (n_assets,)
            Single observation of asset returns. May contain NaN.

        active_row : ndarray of shape (n_assets,)
            Boolean mask indicating which assets are structurally active.
        """
        valid = ~np.isnan(returns) & active_row
        self._obs_count[valid] += 1

        newly_left = self._is_active & ~active_row
        if np.any(newly_left):
            self._var[newly_left] = 0.0
            self._obs_count[newly_left] = 0
            if not self.assume_centered:
                self.location_[newly_left] = np.nan
        self._is_active[:] = active_row

        valid_idx = np.flatnonzero(valid)
        n_valid = valid_idx.size
        if n_valid == 0:
            return

        ret = returns[valid_idx]
        if not self.assume_centered:
            loc = np.nan_to_num(self.location_[valid_idx], nan=0.0)
            ret = ret - loc
            self.location_[valid_idx] = (
                self._decay * loc + (1.0 - self._decay) * returns[valid_idx]
            )

        squared = ret**2
        if n_valid == self.n_features_in_:
            self._var *= self._decay
            self._var += (1.0 - self._decay) * squared
        else:
            self._var[valid_idx] = (
                self._decay * self._var[valid_idx] + (1.0 - self._decay) * squared
            )

    def _process_batch_no_nan(self, X: FloatArray) -> None:
        """Vectorized EWMA update for the common case: no NaN, no
        active_mask, and assume_centered=True.

        Computes the weighted sum in one matrix-vector multiply instead of
        iterating row by row.
        """
        n_obs = X.shape[0]
        decay_powers = self._decay ** np.arange(n_obs - 1, -1, -1)
        weights = (1.0 - self._decay) * decay_powers

        self._var *= self._decay**n_obs
        self._var += weights @ (X**2)
        self._obs_count += n_obs
        self._is_active[:] = True

    def _bias_correct_variance(self) -> FloatArray:
        r"""Return a bias-corrected copy of the internal EWMA state.

        Divides each asset's raw accumulator by :math:`(1 - \lambda^{n_i})`
        to normalize the exponential weights to sum to 1.
        """
        variance = self._var.copy()
        correction = np.where(
            self._obs_count > 0,
            1.0 / np.maximum(1.0 - self._decay**self._obs_count, 1e-15),
            1.0,
        )
        variance *= correction
        return variance

    def _reset(self) -> None:
        """Reset fitted state."""
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)
