"""Module that includes all Measures functions used across `skfolio`."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Gini mean difference and OWA GMD weights features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

import warnings

import numpy as np
import numpy.typing as npt
import scipy.optimize as sco


def mean(
    returns: npt.ArrayLike, sample_weight: np.ndarray | None = None
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    if sample_weight is None:
        # Ignore NaNs and suppress warnings for all-NaN slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(returns, axis=0)
    return sample_weight @ returns


def mean_absolute_deviation(
    returns: npt.ArrayLike,
    min_acceptable_return: float | np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    if min_acceptable_return is None:
        min_acceptable_return = mean(returns, sample_weight=sample_weight)

    absolute_deviations = np.abs(returns - min_acceptable_return)

    return mean(absolute_deviations, sample_weight=sample_weight)


def first_lower_partial_moment(
    returns: npt.ArrayLike,
    min_acceptable_return: float | np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    if min_acceptable_return is None:
        min_acceptable_return = mean(returns, sample_weight=sample_weight)

    deviations = np.maximum(0, min_acceptable_return - returns)

    return mean(deviations, sample_weight=sample_weight)


def variance(
    returns: npt.ArrayLike,
    biased: bool = False,
    sample_weight: np.ndarray | None = None,
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    if sample_weight is None:
        # Ignore NaNs and suppress warnings for all-NaN slices
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanvar(returns, ddof=0 if biased else 1, axis=0)

    biased_var = (
        sample_weight @ (returns - mean(returns, sample_weight=sample_weight)) ** 2
    )
    if biased:
        return biased_var
    n_eff = 1 / np.sum(sample_weight**2)
    return biased_var * n_eff / (n_eff - 1)


def semi_variance(
    returns: npt.ArrayLike,
    min_acceptable_return: float | np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    biased: bool = False,
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    if min_acceptable_return is None:
        min_acceptable_return = mean(returns, sample_weight=sample_weight)

    biased_semi_var = mean(
        np.maximum(0, min_acceptable_return - returns) ** 2, sample_weight=sample_weight
    )
    if biased:
        return biased_semi_var

    n_observations = len(returns)
    if sample_weight is None:
        correction = n_observations / (n_observations - 1)
    else:
        correction = 1.0 / (1.0 - np.sum(sample_weight**2))
    return biased_semi_var * correction


def standard_deviation(
    returns: npt.ArrayLike,
    sample_weight: np.ndarray | None = None,
    biased: bool = False,
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    return np.sqrt(variance(returns, sample_weight=sample_weight, biased=biased))


def semi_deviation(
    returns: npt.ArrayLike,
    min_acceptable_return: float | np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    biased: bool = False,
) -> float | np.ndarray:
    """Compute the semi deviation (square root of the second lower partial moment).

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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
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
    returns: npt.ArrayLike, sample_weight: np.ndarray | None = None
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    return mean(
        (returns - mean(returns, sample_weight=sample_weight)) ** 3,
        sample_weight=sample_weight,
    )


def skew(
    returns: npt.ArrayLike, sample_weight: np.ndarray | None = None
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    return (
        third_central_moment(returns, sample_weight)
        / variance(returns, sample_weight=sample_weight, biased=True) ** 1.5
    )


def fourth_central_moment(
    returns: npt.ArrayLike, sample_weight: np.ndarray | None = None
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    return mean(
        (returns - mean(returns, sample_weight=sample_weight)) ** 4,
        sample_weight=sample_weight,
    )


def kurtosis(
    returns: npt.ArrayLike, sample_weight: np.ndarray | None = None
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    return (
        fourth_central_moment(returns, sample_weight=sample_weight)
        / variance(returns, sample_weight=sample_weight, biased=True) ** 2
    )


def fourth_lower_partial_moment(
    returns: npt.ArrayLike, min_acceptable_return: float | None = None
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    if min_acceptable_return is None:
        min_acceptable_return = mean(returns)
    return mean(np.maximum(0, min_acceptable_return - returns) ** 4)


def worst_realization(returns: npt.ArrayLike) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    with warnings.catch_warnings():
        # all-NaN slice warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return -np.nanmin(returns, axis=0)


def value_at_risk(
    returns: npt.ArrayLike, beta: float = 0.95, sample_weight: np.ndarray | None = None
) -> float | np.ndarray:
    """Compute the historical value at risk (VaR).
    The VaR is the maximum loss at a given confidence level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    beta : float, default=0.95
        The VaR confidence level (return on the worst (1-beta)% observation).

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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    returns = np.asarray(returns, dtype=float)

    if np.isnan(returns).all():
        return (
            np.nan
            if returns.ndim == 1
            else np.full(returns.shape[1], np.nan, dtype=float)
        )

    def _func(arr: np.ndarray) -> float:
        size = arr.shape[0]
        if size == 0:
            return np.nan
        k = (1.0 - beta) * size
        ik = max(0, int(np.ceil(k) - 1))
        # We only need the first k elements, `partition` (~O(n) avg) beats
        # `sort` (O(n log n))
        part = np.partition(arr, ik, axis=0)
        return -part[ik]

    if sample_weight is None:
        if not np.isnan(returns).any():
            return _func(returns)

        # Contains NaNs and returns is 1D
        if returns.ndim == 1:
            return _func(returns[~np.isnan(returns)])

        # Contains NaNs and returns is 2D
        n_assets = returns.shape[1]
        return np.array(
            [_func(returns[~np.isnan(returns[:, j]), j]) for j in range(n_assets)],
            dtype=float,
        )

    # With sample weights
    sorted_idx = np.argsort(returns, axis=0)
    cum_weights = np.cumsum(sample_weight[sorted_idx], axis=0)
    i = np.apply_along_axis(
        np.searchsorted, axis=0, arr=cum_weights, v=1 - beta, side="left"
    )
    # Returns is 1D
    if returns.ndim == 1:
        return -returns[sorted_idx][i]
    # Returns is 2D
    return -np.diag(np.take_along_axis(returns, sorted_idx, axis=0)[i])


def cvar(
    returns: npt.ArrayLike, beta: float = 0.95, sample_weight: np.ndarray | None = None
) -> float | np.ndarray:
    """Compute the historical CVaR (conditional value at risk).

    The CVaR (or Tail VaR) represents the mean shortfall at a specified confidence
    level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    beta : float, default=0.95
        The CVaR confidence level (expected VaR on the worst (1-beta)% observations).

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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    returns = np.asarray(returns, dtype=float)

    if np.isnan(returns).all():
        return (
            np.nan
            if returns.ndim == 1
            else np.full(returns.shape[1], np.nan, dtype=float)
        )

    def _func(arr: np.ndarray) -> float:
        size = arr.shape[0]
        if size == 0:
            return np.nan
        k = (1.0 - beta) * size
        ik = max(0, int(np.ceil(k) - 1))
        # We only need the first k elements, `partition` (~O(n) avg) beats
        # `sort` (O(n log n))
        part = np.partition(arr, ik, axis=0)
        return -np.sum(part[:ik], axis=0) / k + part[ik] * (ik / k - 1.0)

    if sample_weight is None:
        if not np.isnan(returns).any():
            return _func(returns)

        # Contains NaNs and returns is 1D
        if returns.ndim == 1:
            return _func(returns[~np.isnan(returns)])

        # Contains NaNs and returns is 2D
        n_assets = returns.shape[1]
        return np.array(
            [_func(returns[~np.isnan(returns[:, j]), j]) for j in range(n_assets)],
            dtype=float,
        )

    # With sample weight
    order = np.argsort(returns, axis=0)
    sorted_returns = np.take_along_axis(returns, order, axis=0)
    sorted_w = sample_weight[order]
    cum_w = np.cumsum(sorted_w, axis=0)
    idx = np.apply_along_axis(
        np.searchsorted, axis=0, arr=cum_w, v=1 - beta, side="left"
    )

    def _func(_idx, _sorted_returns, _sorted_w, _cum_w) -> float:
        if _idx == 0:
            return _sorted_returns[0]
        return (
            _sorted_returns[:_idx] @ _sorted_w[:_idx]
            + _sorted_returns[_idx] * (1 - beta - _cum_w[_idx - 1])
        ) / (1 - beta)

    # Returns is 1D
    if returns.ndim == 1:
        return -_func(idx, sorted_returns, sorted_w, cum_w)

    # Returns is 2D
    n_assets = returns.shape[1]
    return -np.array(
        [
            _func(idx[i], sorted_returns[:, i], sorted_w[:, i], cum_w[:, i])
            for i in range(n_assets)
        ]
    )


def entropic_risk_measure(
    returns: npt.ArrayLike,
    theta: float = 1,
    beta: float = 0.95,
    sample_weight: np.ndarray | None = None,
) -> float | np.ndarray:
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
    - Unweighted: NaNs are ignored; all-NaN inputs yield NaN.
    - Weighted: NaNs propagate.
    """
    return theta * np.log(
        mean(np.exp(-returns / theta), sample_weight=sample_weight) / (1 - beta)
    )


def evar(returns: npt.ArrayLike, beta: float = 0.95) -> float:
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
    returns: npt.ArrayLike, compounded: bool = False, base: float = 1.0
) -> np.ndarray:
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


def get_drawdowns(returns: npt.ArrayLike, compounded: bool = False) -> np.ndarray:
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


def drawdown_at_risk(drawdowns: np.ndarray, beta: float = 0.95) -> float | np.ndarray:
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


def max_drawdown(drawdowns: np.ndarray) -> float | np.ndarray:
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


def average_drawdown(drawdowns: np.ndarray) -> float | np.ndarray:
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


def cdar(drawdowns: np.ndarray, beta: float = 0.95) -> float | np.ndarray:
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


def edar(drawdowns: np.ndarray, beta: float = 0.95) -> float:
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


def ulcer_index(drawdowns: np.ndarray) -> float | np.ndarray:
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


def owa_gmd_weights(n_observations: int) -> np.ndarray:
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


def gini_mean_difference(returns: npt.ArrayLike) -> float | np.ndarray:
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


def effective_number_assets(weights: np.ndarray) -> float:
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


def correlation(X: np.ndarray, sample_weight: np.ndarray | None = None) -> np.ndarray:
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
