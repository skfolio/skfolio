"""Module that includes all Measures functions used across `skfolio`."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Gini mean difference and OWA GMD weights features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

from __future__ import annotations

import warnings

import numpy as np
import scipy.optimize as sco

from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.stats import safe_divide


def mean(
    returns: ArrayLike, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the mean.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        The computed mean.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    if sample_weight is None:
        # Ignore NaNs and suppress warnings for all-NaN slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(returns, axis=0)
    returns = np.asarray(returns, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    if returns.shape[0] == 0:
        return np.full(returns.shape[1:], np.nan)[()]
    result = sample_weight @ returns
    # Scan returns for NaNs only if the weighted mean contains NaN.
    if not np.isnan(result).any():
        return result
    returns, weights = _prepare_weighted_returns(returns, weights=sample_weight)
    return _weighted_sum(returns, weights=weights)


def mean_absolute_deviation(
    returns: ArrayLike,
    min_acceptable_return: float | FloatArray | None = None,
    sample_weight: FloatArray | None = None,
) -> float | FloatArray:
    """Compute the mean absolute deviation (MAD).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    min_acceptable_return : float or ndarray of shape (n_assets,) optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns. The default (`None`) is to use the returns' mean.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Mean absolute deviation.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    if min_acceptable_return is None:
        min_acceptable_return = mean(returns, sample_weight=sample_weight)

    absolute_deviations = np.abs(returns - min_acceptable_return)

    return mean(absolute_deviations, sample_weight=sample_weight)


def first_lower_partial_moment(
    returns: ArrayLike,
    min_acceptable_return: float | FloatArray | None = None,
    sample_weight: FloatArray | None = None,
) -> float | FloatArray:
    """Compute the first lower partial moment.

    The first lower partial moment is the mean of the returns below a minimum
    acceptable return.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    min_acceptable_return : float or ndarray of shape (n_assets,) optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns. The default (`None`) is to use the returns' mean.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        First lower partial moment.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    if min_acceptable_return is None:
        min_acceptable_return = mean(returns, sample_weight=sample_weight)

    deviations = np.maximum(0, min_acceptable_return - returns)

    return mean(deviations, sample_weight=sample_weight)


def variance(
    returns: ArrayLike,
    biased: bool = False,
    sample_weight: FloatArray | None = None,
) -> float | FloatArray:
    """Compute the variance (second moment).

    Parameters
    ----------
     returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
         Array of return values.

    biased : bool, default=False
         If False (default), computes the sample variance (unbiased); otherwise,
         computes the population variance (biased).

    sample_weight : ndarray of shape (n_observations,), optional
         Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
     value : float or ndarray of shape (n_assets,)
         Variance.
         If `returns` is a 1D-array, the result is a float.
         If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    if sample_weight is None:
        # Ignore NaNs and suppress warnings for all-NaN slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanvar(returns, ddof=0 if biased else 1, axis=0)

    return _weighted_variance(returns, sample_weight=sample_weight, biased=biased)


def semi_variance(
    returns: ArrayLike,
    min_acceptable_return: float | FloatArray | None = None,
    sample_weight: FloatArray | None = None,
    biased: bool = False,
) -> float | FloatArray:
    """Compute the semi-variance (second lower partial moment).

    The semi-variance is the variance of the returns below a minimum acceptable return.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    min_acceptable_return : float or ndarray of shape (n_assets,) optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns. The default (`None`) is to use the returns' mean.

    biased : bool, default=False
        If False (default), computes the sample semi-variance (unbiased); otherwise,
        computes the population semi-variance (biased).

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Semi-variance.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    if sample_weight is not None:
        return _weighted_variance(
            returns,
            sample_weight=sample_weight,
            biased=biased,
            min_acceptable_return=min_acceptable_return,
            downside=True,
        )
    if min_acceptable_return is None:
        min_acceptable_return = mean(returns, sample_weight=sample_weight)

    biased_semi_var = mean(
        np.maximum(0, min_acceptable_return - returns) ** 2, sample_weight=sample_weight
    )
    if biased:
        return biased_semi_var

    # Apply the Bessel correction using each column's non-NaN count.
    returns = np.asarray(returns, dtype=float)
    n_observations = np.count_nonzero(~np.isnan(returns), axis=0)
    correction = safe_divide(n_observations, n_observations - 1, fill_value=np.nan)
    return biased_semi_var * correction


def standard_deviation(
    returns: ArrayLike,
    sample_weight: FloatArray | None = None,
    biased: bool = False,
) -> float | FloatArray:
    """Compute the standard-deviation (square root of the second moment).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    biased : bool, default=False
        If False (default), computes the sample standard-deviation (unbiased);
        otherwise, computes the population standard-deviation (biased).

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Standard-deviation.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return np.sqrt(variance(returns, sample_weight=sample_weight, biased=biased))


def semi_deviation(
    returns: ArrayLike,
    min_acceptable_return: float | FloatArray | None = None,
    sample_weight: FloatArray | None = None,
    biased: bool = False,
) -> float | FloatArray:
    """Compute the semi-deviation (square root of the second lower partial moment).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    min_acceptable_return : float or ndarray of shape (n_assets,) optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns. The default (`None`) is to use the returns' mean.

    biased : bool, default=False
        If False (default), computes the sample semi-deviation (unbiased); otherwise,
        computes the population semi-seviation (biased).

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Semi-deviation.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return np.sqrt(
        semi_variance(
            returns,
            min_acceptable_return=min_acceptable_return,
            biased=biased,
            sample_weight=sample_weight,
        )
    )


def third_central_moment(
    returns: ArrayLike, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the third central moment.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Third central moment.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return mean(
        (returns - mean(returns, sample_weight=sample_weight)) ** 3,
        sample_weight=sample_weight,
    )


def skew(
    returns: ArrayLike, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the Skew.

    The Skew is a measure of the lopsidedness of the distribution.
    A symmetric distribution have a Skew of zero.
    Higher Skew corresponds to longer right tail.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Skew.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return (
        third_central_moment(returns, sample_weight)
        / variance(returns, sample_weight=sample_weight, biased=True) ** 1.5
    )


def fourth_central_moment(
    returns: ArrayLike, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the Fourth central moment.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Fourth central moment.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return mean(
        (returns - mean(returns, sample_weight=sample_weight)) ** 4,
        sample_weight=sample_weight,
    )


def kurtosis(
    returns: ArrayLike, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the Kurtosis.

    The Kurtosis is a measure of the heaviness of the tail of the distribution.
    Higher Kurtosis corresponds to greater extremity of deviations (fat tails).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Kurtosis.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return (
        fourth_central_moment(returns, sample_weight=sample_weight)
        / variance(returns, sample_weight=sample_weight, biased=True) ** 2
    )


def fourth_lower_partial_moment(
    returns: ArrayLike, min_acceptable_return: float | None = None
) -> float | FloatArray:
    """Compute the fourth lower partial moment.

    The Fourth Lower Partial Moment is a measure of the heaviness of the downside tail
    of the returns below a minimum acceptable return.
    Higher Fourth Lower Partial Moment corresponds to greater extremity of downside
    deviations (downside fat tail).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    min_acceptable_return : float, optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns.
        The default (`None`) is to use the returns mean.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Fourth lower partial moment.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. The result is
    NaN if no observations remain.
    """
    if min_acceptable_return is None:
        min_acceptable_return = mean(returns)
    return mean(np.maximum(0, min_acceptable_return - returns) ** 4)


def worst_realization(returns: ArrayLike) -> float | FloatArray:
    """Compute the worst realization (worst return).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Worst realization.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. The result is
    NaN if no observations remain.
    """
    with warnings.catch_warnings():
        # all-NaN slice warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return -np.nanmin(returns, axis=0)


def value_at_risk(
    returns: ArrayLike, beta: float = 0.95, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the historical value at risk (VaR).
    The VaR is the maximum loss at a given confidence level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    beta : float, default=0.95
        The VaR confidence level (return on the worst (1-beta)% observation).
        Must be between 0 and 1. The endpoints select the best and worst
        usable returns, respectively; zero-weight observations are excluded.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Value at Risk.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return _tail_risk(
        returns, beta=beta, sample_weight=sample_weight, conditional=False
    )


def cvar(
    returns: ArrayLike, beta: float = 0.95, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the historical CVaR (conditional value at risk).

    The CVaR (or Tail VaR) represents the mean shortfall at a specified confidence
    level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    beta : float, default=0.95
        The CVaR confidence level (expected VaR on the worst (1-beta)% observations).
        Must be greater than or equal to 0 and less than 1. At 0, CVaR is
        the negative mean of the usable returns.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
     value : float or ndarray of shape (n_assets,)
        CVaR.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return _tail_risk(returns, beta=beta, sample_weight=sample_weight, conditional=True)


def entropic_risk_measure(
    returns: ArrayLike,
    theta: float = 1,
    beta: float = 0.95,
    sample_weight: FloatArray | None = None,
) -> float | FloatArray:
    """Compute the entropic risk measure.

    The entropic risk measure is a risk measure which depends on the risk aversion
    defined by the investor (theta) through the exponential utility function at a given
    confidence level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    theta : float, default=1.0
        Risk aversion.

    beta : float, default=0.95
         Confidence level.

    sample_weight : ndarray of shape (n_observations,), optional
        Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Entropic risk measure.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    return theta * np.log(
        mean(np.exp(-returns / theta), sample_weight=sample_weight) / (1 - beta)
    )


def evar(returns: ArrayLike, beta: float = 0.95) -> float:
    """Compute the EVaR (entropic value at risk) and its associated risk aversion.

    The EVaR is a coherent risk measure which is an upper bound for the VaR and the
    CVaR, obtained from the Chernoff inequality. The EVaR can be represented by using
    the concept of relative entropy.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    beta : float, default=0.95
        The EVaR confidence level.

    Returns
    -------
    value : float
        EVaR.
    """
    if np.isnan(returns).all():
        return np.nan

    def func(x: float) -> float:
        return entropic_risk_measure(returns=returns, theta=x, beta=beta)

    # The lower bound is chosen to avoid exp overflow
    lower_bound = np.nanmax(-returns) / 100
    result = sco.minimize(
        func,
        x0=np.array([lower_bound * 2]),
        method="SLSQP",
        bounds=[(lower_bound, np.inf)],
        tol=1e-10,
    )
    return result.fun


def get_cumulative_returns(
    returns: ArrayLike, compounded: bool = False, base: float = 1.0
) -> FloatArray:
    """Compute the cumulative returns from a series of returns.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    compounded : bool, default=False
        If True, compute compounded (geometric) cumulative returns as a wealth index
        starting at `base`. If False, compute non-compounded (arithmetic) cumulative
        returns starting at 0. Default is False.

    base : float, default=1.0
        Starting value for compounded cumulative returns, expressed as a wealth index.
        For example, use 1.0 for a "wealth index" representing $1 invested, or 100.0
        for index-style rebasing.

    Returns
    -------
    values: ndarray of shape (n_observations,) or (n_observations, n_assets)
        Cumulative returns.

    Notes
    -----
    NaN handling:
    Missing values (NaNs) remain at their original locations in the output and are
    treated as neutral elements during accumulation, so they do not propagate to
    subsequent values.
    """
    returns = np.asarray(returns, dtype=float)

    if np.isnan(returns).all():
        return np.full(returns.shape, np.nan, dtype=float)

    if np.isnan(returns).any():
        mask = np.isnan(returns)
        returns_clean = np.nan_to_num(returns, nan=0.0)
    else:
        mask = None
        returns_clean = returns

    if compounded:
        cumulative_returns = base * np.cumprod(1 + returns_clean, axis=0)
    else:
        cumulative_returns = np.cumsum(returns_clean, axis=0)

    if mask is not None:
        cumulative_returns[mask] = np.nan

    return cumulative_returns


def get_drawdowns(returns: ArrayLike, compounded: bool = False) -> FloatArray:
    """Compute the drawdowns' series from the returns.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    compounded : bool, default=False
       If this is set to True, the cumulative returns are compounded otherwise they
       are uncompounded.

    Returns
    -------
    values: ndarray of shape (n_observations,) or (n_observations, n_assets)
       Drawdowns.

    Notes
    -----
    NaN handling:
    Missing values (NaNs) remain at their original locations in the output and are
    treated as neutral elements during accumulation, so they do not propagate to
    subsequent values.
    """
    if np.isnan(returns).all():
        return np.full(returns.shape, np.nan, dtype=float)

    cumulative_returns = get_cumulative_returns(returns=returns, compounded=compounded)

    if np.isnan(cumulative_returns).any():
        mask = np.isnan(cumulative_returns)
        cum_clean = np.nan_to_num(cumulative_returns, nan=-np.inf)
    else:
        mask = None
        cum_clean = cumulative_returns

    peak = np.maximum.accumulate(cum_clean, axis=0)
    # Identify -Inf positions due to NaN at the start and replace with baseline
    peak = np.where(peak == -np.inf, 1.0 if compounded else 0.0, peak)

    if compounded:
        drawdowns = cum_clean / peak - 1
    else:
        drawdowns = cum_clean - peak

    if mask is not None:
        drawdowns[mask] = np.nan

    return drawdowns


def drawdown_at_risk(drawdowns: FloatArray, beta: float = 0.95) -> float | FloatArray:
    """Compute the Drawdown at risk.

    The Drawdown at risk is the maximum drawdown at a given confidence level (beta).

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Vector of drawdowns.

    beta : float, default = 0.95
        The DaR confidence level (drawdown on the worst (1-beta)% observations).

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Drawdown at risk.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return value_at_risk(returns=drawdowns, beta=beta)


def max_drawdown(drawdowns: FloatArray) -> float | FloatArray:
    """Compute the maximum drawdown.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Vector of drawdowns.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Maximum drawdown.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return drawdown_at_risk(drawdowns=drawdowns, beta=1)


def average_drawdown(drawdowns: FloatArray) -> float | FloatArray:
    """Compute the average drawdown.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Vector of drawdowns.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Average drawdown.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return cdar(drawdowns=drawdowns, beta=0)


def cdar(drawdowns: FloatArray, beta: float = 0.95) -> float | FloatArray:
    """Compute the historical CDaR (conditional drawdown at risk).

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Vector of drawdowns.

    beta : float, default = 0.95
        The CDaR confidence level (expected drawdown on the worst
        (1-beta)% observations).

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        CDaR.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return cvar(returns=drawdowns, beta=beta)


def edar(drawdowns: FloatArray, beta: float = 0.95) -> float:
    """Compute the EDaR (entropic drawdown at risk).

    The EDaR is a coherent risk measure which is an upper bound for the DaR and the
    CDaR, obtained from the Chernoff inequality. The EDaR can be represented by using
    the concept of relative entropy.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,)
        Vector of drawdowns.

    beta : float, default=0.95
      The EDaR confidence level.

    Returns
    -------
    value : float
        EDaR.
    """
    return evar(returns=drawdowns, beta=beta)


def ulcer_index(drawdowns: FloatArray) -> float | FloatArray:
    """Compute the Ulcer index.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Vector of drawdowns.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Ulcer Index.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return np.sqrt(mean(np.power(drawdowns, 2)))


def owa_gmd_weights(n_observations: int) -> FloatArray:
    """Compute the OWA weights used for the Gini mean difference (GMD) computation.

    Parameters
    ----------
    n_observations : int
        Number of observations.

    Returns
    -------
    value : float
        OWA GMD weights.
    """
    return (4 * np.arange(1, n_observations + 1) - 2 * (n_observations + 1)) / (
        n_observations * (n_observations - 1)
    )


def gini_mean_difference(returns: ArrayLike) -> float | FloatArray:
    """Compute the Gini mean difference (GMD).

    The GMD is the expected absolute difference between two realisations.
    The GMD is a superior measure of variability  for non-normal distribution than the
    variance.
    It can be used to form necessary conditions for second-degree stochastic dominance,
    while the variance cannot.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Gini mean difference.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    returns = np.asarray(returns, dtype=float)

    # No NaNs
    if not np.isnan(returns).any():
        w = owa_gmd_weights(returns.shape[0])
        return w @ np.sort(returns, axis=0)

    # 1D with NaN
    if returns.ndim == 1:
        v = returns[~np.isnan(returns)]
        if v.size == 0:
            return np.nan
        w = owa_gmd_weights(v.size)
        return w @ np.sort(v)

    # 2D with NaNs
    n_assets = returns.shape[1]
    out = np.full(n_assets, np.nan, dtype=float)
    isnan = np.isnan(returns)
    for j in range(n_assets):
        col = returns[:, j]
        v = col[~isnan[:, j]]
        if v.size == 0:
            continue  # leave NaN
        w = owa_gmd_weights(v.size)
        out[j] = w @ np.sort(v)
    return out


def effective_number_assets(weights: FloatArray) -> float:
    r"""Compute the effective number of assets, defined as the inverse of the
    Herfindahl index.

    .. math:: N_{eff} = \frac{1}{\Vert w \Vert_{2}^{2}}

    It quantifies portfolio concentration, with a higher value indicating a more
    diversified portfolio.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Weights of the assets.

    Returns
    -------
    value : float
        Effective number of assets.

    References
    ----------
    .. [1] "Banking and Financial Institutions Law in a Nutshell".
        Lovett, William Anthony (1988)
    """
    return 1.0 / (np.power(weights, 2).sum())


def correlation(X: ArrayLike, sample_weight: FloatArray | None = None) -> FloatArray:
    """Compute the correlation matrix.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_assets)
       Array of values.

    sample_weight : ndarray of shape (n_observations,), optional
       Sample weights for each observation. If None, equal weights are assumed.

    Returns
    -------
    corr : ndarray of shape (n_assets,)
       The correlation matrix.
    """
    cov = np.cov(X, rowvar=False, aweights=sample_weight)
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


def _weighted_sum(values: FloatArray, weights: FloatArray) -> float | FloatArray:
    """Sum over observations, using column-specific weights when needed.

    Parameters
    ----------
    values : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Values to sum along the observation axis.

    weights : ndarray of shape (n_observations,) or (n_observations, n_assets)
        A 1D weight vector shared by all columns, or a 2D array matching
        `values` that assigns weights separately for each column.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Weighted sum. A scalar for 1D values, or one result per column for
        2D values.
    """
    if weights.ndim == 1:
        return weights @ values
    # Avoid allocating a full product array for column-specific weights.
    return np.einsum("ij,ij->j", weights, values)


def _prepare_weighted_returns(
    returns: FloatArray, weights: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Prepare returns and weights for calculations that exclude NaNs.

    Inputs must be floating-point arrays and are not modified.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Return values, possibly containing NaNs.

    weights : ndarray of shape (n_observations,)
        Normalized, non-negative observation weights.

    Returns
    -------
    returns : ndarray with the same shape as the input
        Returns with NaN entries replaced by zero as an arithmetic placeholder.
        These zeros do not represent observed returns.

    weights : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Missing returns receive zero weight, and the remaining weights are
        rescaled to sum to one. If no positive weight remains, the normalized
        weights are NaN.
        Weights stay 1D for 1D returns or inputs without NaNs. For 2D returns
        containing NaNs, each column gets its own normalized weight vector.
    """
    missing = np.isnan(returns)
    if missing.any():
        weights = np.where(
            missing, 0.0, weights[:, None] if returns.ndim == 2 else weights
        )
        # This is a new array: normalize in place, with 0/0 marking empty columns.
        with np.errstate(invalid="ignore"):
            weights /= weights.sum(axis=0)
        returns = np.where(missing, 0.0, returns)
    return returns, weights


def _weighted_variance(
    returns: FloatArray,
    sample_weight: FloatArray,
    biased: bool,
    min_acceptable_return: float | FloatArray | None = None,
    downside: bool = False,
) -> float | FloatArray:
    """Compute weighted variance or semi-variance, excluding NaN returns.

    The remaining weights are rescaled to sum to one separately for each column.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Return values, possibly containing NaNs.

    sample_weight : ndarray of shape (n_observations,)
        Normalized, non-negative observation weights.

    biased : bool
        If True, return the population second moment. If False, divide it by
        ``1 - sum(weights**2)`` using the weights after excluding NaN returns.

    min_acceptable_return : float or ndarray of shape (n_assets,), optional
        Reference return for computing deviations. If None, use each column's
        weighted mean. When `downside` is True, this is the minimum acceptable
        return.

    downside : bool, default=False
        If True, use only deviations below the reference return to compute
        semi-variance. All remaining observations contribute to the weight
        normalization and unbiased correction.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Weighted variance or semi-variance. A scalar for 1D returns, or one
        result per column for 2D returns. Empty inputs and columns with no
        remaining positive weight produce NaN. The unbiased result is also
        NaN when its correction is zero.
    """
    returns, weights = _prepare_weighted_returns(
        np.asarray(returns, dtype=float), weights=np.asarray(sample_weight, dtype=float)
    )
    if returns.shape[0] == 0:
        return np.full(returns.shape[1:], np.nan)[()]
    if min_acceptable_return is None:
        min_acceptable_return = _weighted_sum(returns, weights=weights)
    deviations = returns - min_acceptable_return
    if downside:
        np.minimum(deviations, 0.0, out=deviations)
    np.square(deviations, out=deviations)
    result = _weighted_sum(deviations, weights=weights)
    if biased:
        return result
    correction = 1.0 - _weighted_sum(weights, weights=weights)
    return result / np.where(correction == 0, np.nan, correction)


def _tail_risk(
    returns: ArrayLike,
    beta: float,
    sample_weight: FloatArray | None,
    conditional: bool,
) -> float | FloatArray:
    """Compute VaR or CVaR over usable observations, optionally weighted.

    NaN returns are excluded separately for each column. When weights are
    supplied, zero-weight observations are also excluded and the remaining
    weights are rescaled to sum to one.

    Parameters
    ----------
    returns : array-like of shape (n_observations,) or (n_observations, n_assets)
        Return values, possibly containing NaNs.

    beta : float
        Confidence level in [0, 1] for VaR and [0, 1) for CVaR.

    sample_weight : ndarray of shape (n_observations,) or None
        Normalized, non-negative observation weights. If None, the remaining
        observations have equal weight.

    conditional : bool
        If True, return CVaR, the average loss in the lower return tail.
        If False, return VaR, the loss at the tail boundary.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        VaR or CVaR. A scalar for 1D returns, or one result per column for
        2D returns. The result is NaN if no observations or no positive
        weight remain.
    """
    returns = np.asarray(returns, dtype=float)

    def _unweighted(values):
        """Compute the unweighted tail measure using the enclosing settings."""
        size = values.shape[0]
        if size == 0:
            return np.nan
        k = (1.0 - beta) * size
        i = max(0, int(np.ceil(k) - 1))
        # Partition keeps the unweighted calculation linear in the sample size.
        part = np.partition(values, i, axis=0)
        if conditional:
            return -np.sum(part[:i], axis=0) / k + part[i] * (i / k - 1.0)
        return -part[i]

    if sample_weight is None:
        missing = np.isnan(returns)
        if missing.all():
            return np.full(returns.shape[1:], np.nan)[()]
        if not missing.any():
            return _unweighted(returns)
        if returns.ndim == 1:
            return _unweighted(returns[~missing])
        return np.array(
            [_unweighted(returns[~missing[:, j], j]) for j in range(returns.shape[1])]
        )

    weights = np.asarray(sample_weight, dtype=float)
    positive = weights > 0

    def _weighted(column):
        """Compute the weighted tail measure for one return column."""
        valid = ~np.isnan(column) & positive
        values, probs = column[valid], weights[valid]
        if len(probs) == 0:
            return np.nan
        probs /= probs.sum()
        if beta == 0:
            return -probs @ values if conditional else -values.max()
        if beta == 1:
            return -values.min()
        if np.all(probs == probs[0]):
            # Preserve the unweighted empirical rank convention exactly.
            return _unweighted(values)
        order = np.argsort(values)
        values, probs = values[order], probs[order]
        cumulative = np.cumsum(probs)
        tail_mass = (1.0 - beta) * cumulative[-1]
        i = np.searchsorted(cumulative, tail_mass)
        if not conditional or i == 0:
            return -values[i]
        return (
            -(probs[:i] @ values[:i] + values[i] * (tail_mass - cumulative[i - 1]))
            / tail_mass
        )

    if returns.ndim == 1:
        return _weighted(returns)
    return np.array([_weighted(column) for column in returns.T])
