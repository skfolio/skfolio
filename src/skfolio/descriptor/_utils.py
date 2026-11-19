"""Shared averaging and regression helpers for streaming descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import inspect
import warnings

import numpy as np

from skfolio.typing import BoolArray, FloatArray
from skfolio.utils.tools import _validate_positive_real

# Minimum return spread relative to its absolute mean to exceed rounding noise.
_MIN_RELATIVE_SPREAD = 32 * np.finfo(float).eps
# Reject macro sensitivity if the supplied series' market correlation is near +/-1.
_MIN_VARIANCE_RATIO = 1e-8
_ROUNDING_TOL = 64 * np.finfo(float).eps


class _EWRegression:
    """Exponentially weighted regression of asset returns on market returns.

    Optionally includes macro returns, such as FX or commodity returns, to estimate each
    asset's sensitivity to them after removing market exposure. Each asset is fitted
    separately with an intercept, using the same dates for the asset, market and any
    macro returns.

    Regressions are updated one observation at a time.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    decay : float
        Multiplier in [0, 1) applied to existing observation weights when `update` adds
        an observation for an asset (`valid_mask=True`). The new observation receives
        weight `1 - decay`.

    include_macro_returns : bool, default=False
        Include macro returns, supplied to `EWMacroSensitivity` as `reference_returns`.
        If `True`, `beta` returns sensitivity to those returns after removing market
        exposure. Otherwise, it returns market beta.

    estimate_beta_variance : bool, default=False
        Store the additional statistics needed by `beta_variance` to estimate the
        sampling variance of market beta. Only available when
        `include_macro_returns=False`.

    Raises
    ------
    ValueError
        If both `include_macro_returns` and `estimate_beta_variance` are `True`.
    """

    def __init__(
        self,
        *,
        n_assets: int,
        decay: float,
        include_macro_returns: bool = False,
        estimate_beta_variance: bool = False,
    ) -> None:
        if include_macro_returns and estimate_beta_variance:
            raise ValueError(
                "Beta variance estimation requires include_macro_returns=False."
            )

        self.decay = decay
        self.include_macro_returns = include_macro_returns
        self.estimate_beta_variance = estimate_beta_variance

        self._moments = _EWMoments(
            n_variables=3 if include_macro_returns else 2,
            n_assets=n_assets,
            decay=decay,
        )

        if include_macro_returns:
            # Remember past variation even if its weighted variance decays to zero.
            self._constant_market = np.ones(n_assets, dtype=bool)

        if estimate_beta_variance:
            # Beta variance uses the same market history with squared weights.
            self._squared_weight_moments = _EWMoments(
                n_variables=1,
                n_assets=n_assets,
                decay=decay,
                squared_observation_weights=True,
            )

    def reset(self, inactive_mask: BoolArray) -> None:
        """Clear regression history for assets outside the AssetPanel universe.

        Parameters
        ----------
        inactive_mask : ndarray of bool of shape (n_assets,)
            Boolean mask selecting assets to reset. Descriptors pass `~X.active_mask[t]`
            for observation `t`. A `True` entry clears that asset's observation count
            and accumulated statistics.
        """
        self._moments.reset(inactive_mask)
        if self.include_macro_returns:
            self._constant_market[inactive_mask] = True
        if self.estimate_beta_variance:
            self._squared_weight_moments.reset(inactive_mask)

    def update(self, returns: FloatArray, valid_mask: BoolArray) -> None:
        """Update each asset's regression with one new observation.

        Parameters
        ----------
        returns : ndarray of shape (n_variables, n_assets)
            Returns for one observation and all assets. `returns[0]` contains market
            returns and `returns[-1]` contains asset returns. When
            `include_macro_returns=True`, `returns[1]` contains the macro returns. All
            returns for assets selected by `valid_mask` must be finite.

        valid_mask : ndarray of bool of shape (n_assets,)
            Boolean mask selecting assets with `AssetPanel.active_mask=True` and finite
            returns for the market, asset and any macro input (neither NaN nor
            infinite). A `False` entry leaves that asset's observation count and
            statistics unchanged, without decay.

        Raises
        ------
        ValueError
            If the weighted variance or covariance sums become non-finite.
        """
        moments = self._moments
        if self.include_macro_returns:
            self._constant_market &= (
                ~valid_mask
                | (moments.n_observations == 0)
                | (returns[0] == moments.last_returns[0])
            )
        moments.update(returns=returns, valid_mask=valid_mask)
        if self.estimate_beta_variance:
            self._squared_weight_moments.update(
                returns=returns[:1], valid_mask=valid_mask
            )

        # Finite inputs can overflow during moment updates and invalidate later fits.
        if not np.isfinite(moments.covariance_numerator).all():
            raise ValueError(
                "Non-finite EW regression moments. Rescale the input returns."
            )

    def beta(self, min_periods: int = 2) -> FloatArray:
        """Compute market betas or sensitivities to macro returns.

        Parameters
        ----------
        min_periods : int, default=2
            Minimum number of observations added with `valid_mask=True` per asset.
            Market beta requires at least two. With `include_macro_returns=True`, at
            least three are required unless market returns are constant, in which case
            two suffice.

        Returns
        -------
        beta : ndarray of shape (n_assets,)
            Market beta for each asset. With `include_macro_returns=True`, returns
            sensitivity to macro returns after removing market exposure. Contains NaN
            when there are too few observations or too little return variation to
            compute the sensitivity.

        Notes
        -----
        All return series use the same observations and weights, so the weight sums
        cancel in the covariance-to-variance ratios. With `include_macro_returns=True`,
        the Frisch-Waugh formula removes market exposure. Calculations use standard
        deviations to avoid products of very large or very small variances.
        """
        moments = self._moments
        market_variance_numerator = moments.covariance_numerator[0, 0]
        market_covariance = moments.covariance_numerator[0, -1]
        result = np.full(moments.n_observations.shape, np.nan)
        market_has_variation = self._has_sufficient_variation(0)

        if not self.include_macro_returns:
            ready = market_has_variation & (
                moments.n_observations >= max(2, min_periods)
            )
            np.divide(
                market_covariance, market_variance_numerator, out=result, where=ready
            )
        else:
            macro_variance_numerator = moments.covariance_numerator[1, 1]
            market_macro_covariance_numerator = moments.covariance_numerator[0, 1]
            macro_covariance = moments.covariance_numerator[1, -1]
            macro_has_variation = self._has_sufficient_variation(1)

            # An exactly constant market is absorbed by the intercept. Numerical
            # rejection of a varying market must not trigger this reduction.
            macro_only = (
                self._constant_market
                & macro_has_variation
                & (moments.n_observations >= max(2, min_periods))
            )
            np.divide(
                macro_covariance,
                macro_variance_numerator,
                out=result,
                where=macro_only,
            )

            ready = (
                ~self._constant_market
                & market_has_variation
                & macro_has_variation
                & (moments.n_observations >= max(3, min_periods))
            )
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                market_scale = np.sqrt(market_variance_numerator)
                macro_scale = np.sqrt(macro_variance_numerator)
                correlation = (
                    market_macro_covariance_numerator / market_scale
                ) / macro_scale

                # 1 - rho**2 is the macro variance left after market projection.
                residual_fraction = (1 - np.abs(correlation)) * (
                    1 + np.abs(correlation)
                )

                # Partial out the market in standardized predictor coordinates.
                market_component = correlation * (market_covariance / market_scale)
                residual_covariance = macro_covariance / macro_scale - market_component
                beta = (residual_covariance / residual_fraction) / macro_scale
            np.copyto(
                result,
                beta,
                where=ready & (residual_fraction >= _MIN_VARIANCE_RATIO),
            )

        # A ratio of finite moments can overflow when predictor variance is tiny.
        result[~np.isfinite(result)] = np.nan
        return result

    def beta_variance(self, beta: FloatArray) -> FloatArray:
        """Estimate the sampling variance of each asset's market beta.

        Requires `estimate_beta_variance=True`. Assumes independent regression errors
        with constant variance.

        Parameters
        ----------
        beta : ndarray of shape (n_assets,)
            Market betas returned by `beta` for the current state, before shrinkage.

        Returns
        -------
        variance : ndarray of shape (n_assets,)
            Estimated sampling variance of each market beta. Contains NaN when beta or
            its sampling variance cannot be estimated.

        Notes
        -----
        Squared observation weights account for the exponential weighting when
        estimating regression error variance and beta variance. The calculation also
        accounts for estimating the intercept and beta from a finite number of
        observations.
        """
        moments = self._moments
        squared_weight_moments = self._squared_weight_moments
        market_variance_numerator = moments.covariance_numerator[0, 0]
        market_covariance = moments.covariance_numerator[0, -1]
        asset_variance_numerator = moments.covariance_numerator[-1, -1]

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            # Recenter the squared-weight moments at the ordinary weighted market mean.
            leverage = (
                squared_weight_moments.covariance_numerator[0, 0]
                + squared_weight_moments.observation_weight_sum
                * (squared_weight_moments.mean_offset[0] - moments.mean_offset[0]) ** 2
            )
            relative_leverage = leverage / market_variance_numerator

            # Account for estimating both the intercept and slope from these data.
            residual_fraction = (
                1
                - squared_weight_moments.observation_weight_sum
                / moments.observation_weight_sum**2
                - relative_leverage / moments.observation_weight_sum
            )

            explained_sum_squares = beta * market_covariance
            residual_sum_squares = asset_variance_numerator - explained_sum_squares
            # Clamp small rounding errors. Larger negative sums must be rejected below
            # to avoid reporting zero uncertainty for an invalid fit.
            rounding = _ROUNDING_TOL * (
                asset_variance_numerator + np.abs(explained_sum_squares)
            )
            residual_sum_squares = np.where(
                (residual_sum_squares < 0) & (residual_sum_squares >= -rounding),
                0,
                residual_sum_squares,
            )

            variance = (
                residual_sum_squares
                / (moments.observation_weight_sum * residual_fraction)
                * relative_leverage
                / market_variance_numerator
            )

        # Shrinkage requires a valid uncertainty estimate with residual degrees of freedom.
        ready = (
            np.isfinite(beta)
            & (moments.n_observations >= 3)
            & (residual_fraction > _ROUNDING_TOL)
            & (residual_sum_squares >= 0)
            & np.isfinite(variance)
            & (variance >= 0)
        )
        return np.where(ready, variance, np.nan)

    def _has_sufficient_variation(self, variable: int) -> BoolArray:
        """Check whether market or macro returns vary beyond rounding noise.

        The weighted standard deviation must exceed `_MIN_RELATIVE_SPREAD` times the
        absolute weighted mean.

        Parameters
        ----------
        variable : int
            Return series to check, 0 for market returns or 1 for macro returns when
            `include_macro_returns=True`.

        Returns
        -------
        has_variation : ndarray of bool of shape (n_assets,)
            Boolean mask identifying assets whose return variation exceeds the rounding
            threshold.
        """
        moments = self._moments
        mean = moments.last_returns[variable] + moments.mean_offset[variable]
        minimum_spread = _MIN_RELATIVE_SPREAD * np.abs(mean)

        # Compare on the same weight scale without squaring the mean or dividing by zero.
        return np.sqrt(moments.covariance_numerator[variable, variable]) > (
            np.sqrt(moments.observation_weight_sum) * minimum_spread
        )


class _ReturnAverager:
    """Average market, asset and external returns over fixed windows.

    For each asset, average returns over `aggregation_period` consecutive observations.
    Include only observations where `active_mask=True` and all supplied returns are
    finite.

    Each call to `update` processes one observation. Averages are returned every
    `aggregation_period` calls. When `active_mask=False` for an asset, discard its
    earlier returns from the current averaging window. With `aggregation_period=1`,
    return the current returns without averaging.

    Parameters
    ----------
    n_assets : int
        Number of assets.

    n_variables : int
        Number of return series averaged together for each asset.

    aggregation_period : int
        Positive number of update calls per window, including calls where returns are
        NaN or `active_mask=False`.
    """

    def __init__(
        self, *, n_assets: int, n_variables: int, aggregation_period: int
    ) -> None:
        self.aggregation_period = aggregation_period
        self.window_position = 0
        self.n_observations = np.zeros(n_assets, dtype=int)
        self.last_returns = np.zeros((n_variables, n_assets))
        self.mean_offset = np.zeros_like(self.last_returns)

    def update(
        self, returns: FloatArray, active_mask: BoolArray
    ) -> tuple[FloatArray, BoolArray] | None:
        """Update return averages and return them every `aggregation_period` calls.

        Parameters
        ----------
        returns : ndarray of shape (n_variables, n_assets)
            Return series for one observation, with one column per asset. Only columns
            where `active_mask=True` and every value is finite contribute.

        active_mask : ndarray of bool of shape (n_assets,)
            The panel's `active_mask` for the current observation. A `False` entry
            clears that asset's pending window. A `True` entry with a NaN return
            preserves the pending window without adding an observation.

        Returns
        -------
        aggregated : tuple of (ndarray, ndarray) or None
            Every `aggregation_period` calls, returns the means of shape (n_variables,
            n_assets) and a boolean mask of shape (n_assets,). The mask is `True` where
            at least one observation with all finite returns contributed. Other columns
            must be ignored. Returns `None` on all other calls. For
            `aggregation_period=1`, returns the input array and a mask selecting columns
            with `active_mask=True` and all finite returns.

        Raises
        ------
        ValueError
            If averaging produces non-finite mean offsets.
        """
        valid_mask = active_mask & np.isfinite(returns).all(axis=0)
        if self.aggregation_period == 1:
            return returns, valid_mask

        self.n_observations[~active_mask] = 0
        self.last_returns[:, ~active_mask] = 0
        self.mean_offset[:, ~active_mask] = 0

        fraction = self.n_observations / (self.n_observations + 1)

        # Keep the mean relative to the latest tuple to preserve small variation.
        with np.errstate(over="ignore", invalid="ignore"):
            delta = (returns - self.last_returns) - self.mean_offset
            np.copyto(self.mean_offset, -fraction * delta, where=valid_mask)

        np.copyto(self.last_returns, returns, where=valid_mask)

        self.n_observations += valid_mask

        # Subtracting finite returns can overflow and invalidate the window's mean.
        if not np.isfinite(self.mean_offset).all():
            raise ValueError(
                "Non-finite aggregated returns. Rescale the input returns."
            )

        # Count every call toward the window, including NaNs and active_mask=False.
        self.window_position += 1

        if self.window_position < self.aggregation_period:
            return None

        means = self.last_returns + self.mean_offset
        valid_mask = self.n_observations > 0
        self.n_observations[:] = 0
        self.last_returns[:] = 0
        self.mean_offset[:] = 0
        self.window_position = 0

        return means, valid_mask


class _EWMoments:
    """Accumulate return statistics for exponentially weighted regressions.

    Market beta uses two return series for each asset: its returns and market returns.
    Macro sensitivity adds an external series, such as FX returns. This class tracks the
    average and variation of each series and how the series move together, separately
    for each asset.

    Parameters
    ----------
    n_variables : int
        Number of return series tracked together for each asset. Asset and market
        returns require 2, or 3 when an external series is also included. With one
        series, only its mean and variance are tracked.

    n_assets : int
        Number of assets.

    decay : float
        Factor in [0, 1) applied to earlier observation weights when adding new returns
        for an asset. Smaller values give less influence to older returns.

    squared_observation_weights : bool, default=False
        Whether to square the observation weights for beta uncertainty. If `True`, the
        decay becomes `decay**2` and each new observation receives weight
        `(1 - decay)**2`. Otherwise, the decay is unchanged and each new observation
        receives weight `1 - decay`.

    Notes
    -----
    The covariance matrix for each asset is
    `covariance_numerator / observation_weight_sum`. Diagonal entries contain the
    variance of each return series. Other entries contain covariances between series,
    such as asset and market returns. Means are stored as `last_returns + mean_offset`
    to preserve small changes in returns.
    """

    def __init__(
        self,
        *,
        n_variables: int,
        n_assets: int,
        decay: float,
        squared_observation_weights: bool = False,
    ) -> None:
        self.decay = decay
        self.observation_weight = 1 - decay
        if squared_observation_weights:
            self.decay **= 2
            self.observation_weight **= 2

        self.n_observations = np.zeros(n_assets, dtype=int)
        self.observation_weight_sum = np.zeros(n_assets)
        self.last_returns = np.zeros((n_variables, n_assets))
        self.mean_offset = np.zeros_like(self.last_returns)
        self.covariance_numerator = np.zeros((n_variables, n_variables, n_assets))

    def reset(self, inactive_mask: BoolArray) -> None:
        """Clear statistics for the selected assets.

        Parameters
        ----------
        inactive_mask : ndarray of bool of shape (n_assets,)
            Boolean mask selecting assets whose observation counts and accumulated
            statistics are set to zero. Other assets retain their statistics.
        """
        index = np.flatnonzero(inactive_mask & (self.n_observations > 0))
        if index.size:
            self.n_observations[index] = 0
            self.observation_weight_sum[index] = 0
            self.last_returns[:, index] = 0
            self.mean_offset[:, index] = 0
            self.covariance_numerator[:, :, index] = 0

    def update(self, returns: FloatArray, valid_mask: BoolArray) -> None:
        """Add one observation for the assets selected by `valid_mask`.

        Parameters
        ----------
        returns : ndarray of shape (n_variables, n_assets)
            Returns for one observation, with one column per asset. All values in
            columns selected by `valid_mask` must be finite (neither NaN nor infinite).

        valid_mask : ndarray of bool of shape (n_assets,)
            Boolean mask selecting assets to update. A `False` entry preserves that
            asset's statistics and observation count without decay.

        Notes
        -----
        Uses the weighted Welford algorithm to update means, variances and covariances.
        The change in the weighted mean is included in the variance and covariance
        updates, including during warm-up. The first observation selected by
        `valid_mask` sets the means and leaves the variance and covariance sums at zero.
        """
        old_weight_sum = self.decay * self.observation_weight_sum
        weight_sum = old_weight_sum + self.observation_weight
        fraction = old_weight_sum / weight_sum

        with np.errstate(over="ignore", invalid="ignore"):
            delta = (returns - self.last_returns) - self.mean_offset
            np.copyto(
                self.covariance_numerator,
                self.decay * self.covariance_numerator
                + self.observation_weight
                * fraction
                * delta[:, None, :]
                * delta[None, :, :],
                where=valid_mask,
            )
            # Store the new mean relative to the current returns.
            np.copyto(self.mean_offset, -fraction * delta, where=valid_mask)

        np.copyto(self.last_returns, returns, where=valid_mask)
        np.copyto(self.observation_weight_sum, weight_sum, where=valid_mask)
        self.n_observations += valid_mask


class _EWMean:
    """Exponentially weighted averages computed separately for each asset.

    Computes averages of quantities such as turnover, trading volume or squared returns,
    giving more influence to recent observations.

    It applies bias correction separately for each asset to prevent zero initialization
    from pulling averages toward zero for assets with short histories or when
    `AssetPanel.active_mask` differs across assets, as with late listings or assets that
    become active again.

    Parameters
    ----------
    n_assets : int
        Number of assets whose averages are tracked.

    decay : float
        Factor in [0, 1) applied to earlier observation weights when a new observation
        is added for an asset. Smaller values give more influence to recent
        observations.
    """

    def __init__(self, *, n_assets: int, decay: float) -> None:
        self.decay = decay
        self.n_observations = np.zeros(n_assets, dtype=int)
        self.weighted_sum = np.zeros(n_assets)

    def reset(self, inactive_mask: BoolArray) -> None:
        """Clear the estimation history for the selected assets.

        Parameters
        ----------
        inactive_mask : ndarray of bool of shape (n_assets,)
            Boolean mask selecting assets whose weighted sums and observation counts
            are set to zero.
        """
        self.weighted_sum[inactive_mask] = 0
        self.n_observations[inactive_mask] = 0

    def update(self, values: FloatArray, valid_mask: BoolArray) -> None:
        """Add one observation for the selected assets.

        Parameters
        ----------
        values : ndarray of shape (n_assets,)
            Input values for one observation. Values selected by `valid_mask` must be
            finite.

        valid_mask : ndarray of bool of shape (n_assets,)
            Boolean mask selecting assets to update. Other assets retain their
            statistics and observation counts without decay.
        """
        np.copyto(
            self.weighted_sum,
            self.decay * self.weighted_sum + (1 - self.decay) * values,
            where=valid_mask,
        )
        self.n_observations += valid_mask

    def mean(self, min_periods: int = 1) -> FloatArray:
        """Return the mean for each asset after normalizing observation weights.

        Parameters
        ----------
        min_periods : int, default=1
            Minimum number of observations required per asset. Must be at least one.

        Returns
        -------
        mean : ndarray of shape (n_assets,)
            Exponentially weighted means, with NaN for assets below `min_periods`.
        """
        result = np.full(self.weighted_sum.shape, np.nan)
        weight_sum = 1 - self.decay**self.n_observations
        np.divide(
            self.weighted_sum,
            weight_sum,
            out=result,
            where=self.n_observations >= min_periods,
        )
        return result


# TODO remove deprecated eps parameter and warning helper in v2.0
def _warn_deprecated_eps(eps: float | None) -> None:
    """Validate the unused legacy parameter and warn when it is supplied.

    Parameters
    ----------
    eps : float or None
        Legacy regularization parameter. A supplied value must be finite and positive.
        `None` does not emit a warning. Call once during estimator initialization.

    Raises
    ------
    ValueError
        If `eps` is supplied but is not finite and positive.

    Warns
    -----
    FutureWarning
        If `eps` is supplied, with the warning pointing to the caller's code.
    """
    if eps is not None:
        _validate_positive_real(eps, "eps")

        # Skip estimator methods and scikit-learn output wrappers on either fit path.
        frame = inspect.currentframe()
        stacklevel = 1
        while frame is not None and frame.f_globals.get("__name__", "").startswith(
            ("skfolio.", "sklearn.")
        ):
            stacklevel += 1
            frame = frame.f_back
        del frame
        warnings.warn(
            "eps is deprecated and has no effect. It will be removed in version 2.0.",
            FutureWarning,
            stacklevel=stacklevel,
        )
