"""Exponentially weighted market beta descriptor."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import numpy as np

from skfolio.containers import MISSING_CATEGORY_CODE, AssetPanel
from skfolio.descriptor._base import BaseDescriptor
from skfolio.typing import FloatArray, IntArray
from skfolio.utils.stats import _market_returns
from skfolio.utils.tools import (
    _validate_positive_integer,
    _validate_positive_real,
    half_life_to_decay_factor,
)
from skfolio.utils.validation import validate_asset_panel

_FITTED_ATTR = "market_beta_"


class EWMarketBeta(BaseDescriptor):
    r"""Exponentially weighted market beta descriptor.

    Measures each asset's sensitivity to the market portfolio using exponentially
    weighted covariance and variance estimates:

    .. math::

        \beta_i = \frac{\text{Cov}(r_i, r_m)}{\text{Var}(r_m)}

    where :math:`r_i` is the return of asset :math:`i`, :math:`r_m` is the cap-weighted
    market return computed on the estimation universe and the EWMA uses decay
    :math:`\lambda = \exp(-\ln(2) / \text{half_life})`.

    Parameters
    ----------
    half_life : float, default=60.0
        EWMA half-life in observations after aggregation. For example, with
        `aggregation_period=5` and `half_life=60`, the EWMA decays over 60 aggregated
        periods, equivalent to 300 raw observations.

    aggregation_period : int, default=1
        Number of consecutive observations to aggregate before updating EWMA statistics.
        Aggregation can reduce desynchronization effects. Returns are aggregated with
        the mean of finite values. If an asset has no finite returns in an aggregation
        window, its state is unchanged.

    min_periods : int, optional
        Minimum number of market observations and valid asset returns required before
        computing market betas. Until both counts reach this value, the asset's output
        is NaN. This warm-up period avoids exposing early EWMA values before the beta
        estimate has sufficiently converged from its zero initialization. If `None`,
        defaults to :math:`\lceil\text{half_life}\rceil`, with a minimum of 1.

    shrinkage_group : str, optional
        Name of a categorical field containing group labels (e.g., `"industry"`) for
        Bayesian shrinkage. When provided, raw betas are shrunk toward the cap-weighted
        group mean using an empirical Bayes approach:

        .. math::

            \beta_i^{\text{shrunk}} = w_i \cdot \beta_i^{\text{raw}}
            + (1 - w_i) \cdot \mu_g

        where :math:`w_i = \tau_g^2 / (\tau_g^2 + \sigma_i^2)`, :math:`\mu_g` is the
        cap-weighted group mean, :math:`\tau_g^2` is the prior variance (cross-sectional
        variance minus noise) and :math:`\sigma_i^2` is the estimation error variance.

        Missing category codes are excluded from shrinkage. If `None` (default), no
        shrinkage is applied.

    min_group_size : int, default=5
        Minimum number of assets in a group to compute group-specific statistics.
        Groups with fewer assets fall back to global (cross-sectional) statistics.
        Only used when `shrinkage_group` is provided.

    shrinkage_bounds : tuple of float, default=(0.0, 1.0)
        Lower and upper bounds `(w_min, w_max)` for the raw-beta weight
        :math:`w_i`. Lower values apply more shrinkage toward the group mean, while
        higher values keep more of the raw beta. The coefficient is clipped to this
        range after estimation.

        For example, `(0.1, 0.9)` keeps at least 10% weight on the raw beta and at
        least 10% weight on the group mean. Only used when `shrinkage_group` is
        provided.

    eps : float, default=1e-12
        Small constant for numerical stability in
        :math:`1 / \text{Var}(\text{market})`.

    Attributes
    ----------
    n_assets_ : int
        Number of assets seen during fitting.

    asset_names_ : ndarray of shape (n_assets,)
        Asset names seen during fitting.

    market_beta_ : ndarray of shape (n_assets,)
        Last fitted market beta value for each asset.

    Notes
    -----
    NaNs are allowed as missing observations. Non-missing `returns` values must be
    finite. The market variance is updated at every observation. Asset covariances are
    updated only for assets with valid returns and each asset's valid-observation count
    controls when its output starts. This avoids emitting initialized values for
    late-listed or sparsely observed assets. The `active_mask` property of
    :class:`AssetPanel` distinguishes holidays from delistings.

    Market returns are computed from the estimation universe (`estimation_mask` of
    :class:`AssetPanel`). If no estimable asset has both finite returns and finite
    `market_cap` at an observation, the market return is undefined and a `ValueError`
    is raised.

    References
    ----------
    .. [1] "Capital asset prices: A theory of market equilibrium under conditions of
        risk". The Journal of Finance. Sharpe, W. F. (1964).
    """

    market_beta_: FloatArray

    def __init__(
        self,
        half_life: float = 60.0,
        aggregation_period: int = 1,
        min_periods: int | None = None,
        shrinkage_group: str | None = None,
        min_group_size: int = 5,
        shrinkage_bounds: tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-12,
    ):
        self.half_life = half_life
        self.aggregation_period = aggregation_period
        self.min_periods = min_periods
        self.shrinkage_group = shrinkage_group
        self.min_group_size = min_group_size
        self.shrinkage_bounds = shrinkage_bounds
        self.eps = eps

    def fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Compute exponentially weighted market betas.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing `returns` and `market_cap`, and when
            `shrinkage_group` is set, that group characteristic.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        betas : ndarray of shape (n_observations, n_assets)
            Market beta for each observation and asset.
        """
        self._reset()
        return self.partial_fit_transform(X, y, **fit_params)

    def partial_fit_transform(self, X: AssetPanel, y=None, **fit_params) -> FloatArray:
        """Update EWMA state on X and return betas for this batch.

        This method supports online updates by continuing from the current fitted state.
        Use `fit_transform` to start from a clean state.

        Parameters
        ----------
        X : AssetPanel
            Input panel containing at least `"returns"` and `"market_cap"`.
            If shrinkage is enabled, it must also contain the field specified
            by `shrinkage_group`.

        y : None
            Ignored. Present for compatibility with scikit-learn's API.

        **fit_params : dict
            Additional fit parameters. Ignored.

        Returns
        -------
        betas : ndarray of shape (n_observations, n_assets)
            Market beta for each observation and asset. Outputs are NaN until
            the global market state and the asset-specific valid-return count
            both reach `min_periods`.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        required_characteristics = ["returns", "market_cap"]
        if self.shrinkage_group is not None:
            required_characteristics.append(self.shrinkage_group)

        validate_asset_panel(
            self,
            X,
            required_fields=required_characteristics,
            finite_or_nan=["returns"],
            finite_when_active=["market_cap"],
            reset=first_call,
        )

        if first_call:
            self._validate_params()
            self._initialize()

        n_observations, n_assets = X.n_observations, X.n_assets

        asset_rets = X["returns"]
        market_caps = X["market_cap"]

        market_rets = _market_returns(
            asset_returns=asset_rets,
            weights=market_caps,
            estimation_mask=X.estimation_mask,
        )

        betas = np.empty((n_observations, n_assets), dtype=float)

        group_labels = None
        if self._shrinkage_enabled:
            group_field = X.get_field(self.shrinkage_group)
            if not group_field.is_categorical:
                raise ValueError(
                    f"Field '{self.shrinkage_group}' must be a CategoricalField."
                )
            group_labels = group_field.values

        if self._aggregation_period == 1:
            for i in range(n_observations):
                self._update_ewma(ret_assets=asset_rets[i], ret_market=market_rets[i])
                if self._shrinkage_enabled and self._t >= self._min_periods:
                    # Shrinkage applied every observation since betas update every time
                    self._shrunk_betas = self._apply_shrinkage(
                        self._betas, group_labels[i], market_caps[i]
                    )
                    betas[i] = self._shrunk_betas
                else:
                    betas[i] = self._betas
        else:
            # Aggregation path: buffer returns until window is complete
            for i in range(n_observations):
                self._buffer_assets[self._buffer_idx] = asset_rets[i]
                self._buffer_market[self._buffer_idx] = market_rets[i]
                self._buffer_idx += 1

                if self._buffer_idx >= self._aggregation_period:
                    self._update_from_buffer()
                    # Apply shrinkage only on flush (when betas update)
                    if self._shrinkage_enabled and self._t >= self._min_periods:
                        self._shrunk_betas = self._apply_shrinkage(
                            self._betas, group_labels[i], market_caps[i]
                        )

                # Output: shrunk betas if available, else raw betas
                if self._shrinkage_enabled and self._t >= self._min_periods:
                    betas[i] = self._shrunk_betas
                else:
                    betas[i] = self._betas

        # Mask output for out-of-universe assets (delisted / not yet listed).
        # Internal EWMA state is preserved (frozen) so that if an asset
        # re-enters the universe, estimation resumes from its last state.
        betas = np.where(X.active_mask, betas, np.nan)

        # Store only the last row for fitted checks and state access.
        self.market_beta_ = betas[-1].copy() if n_observations > 1 else betas[-1]

        return betas

    def _reset(self):
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    def _validate_params(self) -> None:
        """Validate constructor parameters."""
        _validate_positive_real(self.half_life, "half_life")
        _validate_positive_integer(self.aggregation_period, "aggregation_period")
        if self.min_periods is not None:
            _validate_positive_integer(self.min_periods, "min_periods")
        _validate_positive_integer(self.min_group_size, "min_group_size")
        if (
            len(self.shrinkage_bounds) != 2
            or self.shrinkage_bounds[0] < 0
            or self.shrinkage_bounds[1] > 1
            or self.shrinkage_bounds[0] > self.shrinkage_bounds[1]
        ):
            raise ValueError(
                f"shrinkage_bounds must be (w_min, w_max) with "
                f"0 <= w_min <= w_max <= 1, got {self.shrinkage_bounds}"
            )
        _validate_positive_real(self.eps, "eps")

    def _initialize(self) -> None:
        """Initialize EWMA state and aggregation buffers."""
        # Minimum valid aggregated observations before output.
        if self.min_periods is None:
            self._min_periods = max(1, int(np.ceil(self.half_life)))
        else:
            self._min_periods = int(self.min_periods)

        # Count of aggregated periods processed
        self._t = 0
        self._decay = half_life_to_decay_factor(self.half_life)
        self._mu_market = 0.0
        self._var_market = 0.0

        n_assets = self.n_assets_
        self._mu_assets = np.zeros(n_assets)
        self._cov_assets = np.zeros(n_assets)

        # Last computed betas (for holding during incomplete windows)
        self._betas = np.full(n_assets, np.nan, dtype=float)
        self._n_valid_assets = np.zeros(n_assets, dtype=int)

        # Residual variance tracking for shrinkage (EWMA of residual^2)
        self._shrinkage_enabled = self.shrinkage_group is not None
        if self._shrinkage_enabled:
            self._var_residual = np.zeros(n_assets)
            # Cache shrunk betas (updated only when raw betas update)
            self._shrunk_betas = np.full(n_assets, np.nan, dtype=float)

        # Aggregation buffers (pre-allocated for performance)
        self._aggregation_period = int(self.aggregation_period)
        if self._aggregation_period > 1:
            self._buffer_assets = np.empty(
                (self._aggregation_period, n_assets), dtype=float
            )
            self._buffer_market = np.empty(self._aggregation_period, dtype=float)
            self._buffer_idx = 0

    def _update_ewma(self, ret_assets: FloatArray, ret_market: float) -> None:
        """Update EWMA statistics with a single observation or aggregated values.

        Uses centered EWMA with lagged-mean deviations: deviations are computed from the
        previous step's means, then means are updated. This is the EWMA analogue of
        Welford's online algorithm and avoids the systematic downward bias of the naive
        approach (which computes deviations from the already-updated mean).

        The EWMA update formula for any statistic S is:
            S_t = decay * S_{t-1} + (1 - decay) * x_t

        where decay = exp(-ln(2) / half_life), so half the weight is on the  most recent
        `half_life` observations.
        """
        # Track residuals BEFORE updating beta (uses previous period's beta)
        # This gives E[epsilon^2] for standard error estimation
        if self._shrinkage_enabled and self._t >= self._min_periods:
            valid_for_residual = np.isfinite(ret_assets) & np.isfinite(self._betas)
            residuals = ret_assets - self._betas * ret_market
            self._var_residual[valid_for_residual] = (
                self._decay * self._var_residual[valid_for_residual]
                + (1 - self._decay) * residuals[valid_for_residual] ** 2
            )

        self._t += 1

        # Deviations from lagged means.
        market_deviation = ret_market - self._mu_market
        valid = np.isfinite(ret_assets)
        self._n_valid_assets[valid] += 1
        asset_deviations = ret_assets[valid] - self._mu_assets[valid]

        # Update EWMA means.
        self._mu_market = self._decay * self._mu_market + (1 - self._decay) * ret_market
        self._mu_assets[valid] = (
            self._decay * self._mu_assets[valid] + (1 - self._decay) * ret_assets[valid]
        )

        # Update EWMA variance and covariance.
        self._var_market = self._decay * self._var_market + (1 - self._decay) * (
            market_deviation * market_deviation
        )
        self._cov_assets[valid] = (
            self._decay * self._cov_assets[valid]
            + (1 - self._decay) * asset_deviations * market_deviation
        )

        # Beta
        if self._t >= self._min_periods:
            asset_ready = valid & (self._n_valid_assets >= self._min_periods)
            self._betas[asset_ready] = self._cov_assets[asset_ready] / (
                self._var_market + self.eps
            )

    def _update_from_buffer(self) -> None:
        """Aggregate buffered returns and update EWMA statistics."""
        # Compute aggregated asset returns, skipping missing values per asset.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Mean of empty slice", category=RuntimeWarning
            )
            agg_assets = np.nanmean(self._buffer_assets, axis=0)
        agg_market = np.mean(self._buffer_market)

        # Reset buffer index
        self._buffer_idx = 0

        # Update EWMA with aggregated values
        self._update_ewma(agg_assets, agg_market)

    def _apply_shrinkage(
        self, raw_betas: FloatArray, group_labels: IntArray, market_cap: FloatArray
    ) -> FloatArray:
        """Apply Bayesian shrinkage to raw betas using group priors.

        Uses Empirical Bayes (James-Stein) shrinkage where each asset's beta is pulled
        toward the cap-weighted group mean. The shrinkage intensity depends on the ratio
        of true cross-sectional variance to estimation noise.

        Parameters
        ----------
        raw_betas : darray of shape (n_assets,)
            Raw EWMA betas.

        group_labels : ndarray of shape (n_assets,)
            Group label for each asset (e.g., industry).

        market_cap : ndarray of shape (n_assets,)
            Market capitalization for each asset.

        Returns
        -------
        shrunk_betas : ndarray of shape (n_assets,)
            Shrunk betas.
        """
        shrunk_betas = raw_betas.copy()

        # Estimation error variance: SE(beta)^2 = Var(residual) / (effective_n * Var(market))
        # For EWMA, effective_n ~ 2 * half_life (effective sample size)
        effective_n = 2 * self.half_life
        beta_error_variance = self._var_residual / (
            effective_n * (self._var_market + self.eps)
        )

        # Identify valid assets (have valid beta, group, and market cap)
        valid_beta = ~np.isnan(raw_betas)
        valid_group = group_labels != MISSING_CATEGORY_CODE
        valid_mcap = np.isfinite(market_cap) & (market_cap > 0)
        valid = valid_beta & valid_group & valid_mcap

        if not valid.any():
            return shrunk_betas

        # Compute global statistics (used as fallback for small groups)
        global_beta_mean = np.average(raw_betas[valid], weights=market_cap[valid])
        global_observed_variance = np.average(
            (raw_betas[valid] - global_beta_mean) ** 2, weights=market_cap[valid]
        )
        global_mean_error_variance = np.mean(beta_error_variance[valid])
        # Prior variance = observed variance - estimation noise (floored at 0)
        global_prior_variance = max(
            global_observed_variance - global_mean_error_variance, 0.0
        )

        # Process each group
        unique_groups = np.unique(group_labels[valid])

        for group in unique_groups:
            group_mask = valid & (group_labels == group)
            group_size = group_mask.sum()

            if group_size < self.min_group_size:
                # Small group: fall back to global prior
                group_beta_mean = global_beta_mean
                group_prior_variance = global_prior_variance
            else:
                # Compute group-specific prior (cap-weighted)
                group_beta_mean = np.average(
                    raw_betas[group_mask], weights=market_cap[group_mask]
                )
                group_observed_variance = np.average(
                    (raw_betas[group_mask] - group_beta_mean) ** 2,
                    weights=market_cap[group_mask],
                )
                mean_error_variance = np.mean(beta_error_variance[group_mask])
                group_prior_variance = max(
                    group_observed_variance - mean_error_variance, 0.0
                )

            # Higher weight keeps more of the raw beta; lower weight shrinks more
            # toward the group mean.
            raw_beta_weight = group_prior_variance / (
                group_prior_variance + beta_error_variance[group_mask] + self.eps
            )

            # Clip to bounds for robustness
            w_min, w_max = self.shrinkage_bounds
            raw_beta_weight = np.clip(raw_beta_weight, w_min, w_max)

            # Apply shrinkage
            shrunk_betas[group_mask] = (
                raw_beta_weight * raw_betas[group_mask]
                + (1 - raw_beta_weight) * group_beta_mean
            )

        return shrunk_betas
