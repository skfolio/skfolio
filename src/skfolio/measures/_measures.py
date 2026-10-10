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
import scipy.special as scs

from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.stats import safe_divide
from skfolio.utils.tools import _normalize_sample_weight, _validate_sample_weight


def mean(
    returns: ArrayLike, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the mean.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
    returns = np.asarray(returns, dtype=float)
    if sample_weight is None:
        return _mean(returns)
    n_observations = returns.shape[0]
    sample_weight = _validate_sample_weight(
        sample_weight, n_observations=n_observations
    )
    if n_observations == 0:
        return np.full(returns.shape[1:], np.nan)[()]
    result = _normalize_sample_weight(sample_weight) @ returns
    if not np.isnan(result).any():
        return result
    returns, sample_weight = _prepare_weighted_returns(
        returns, sample_weight=sample_weight
    )
    return _mean(returns, sample_weight=sample_weight)


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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
    deviations, sample_weight, _ = _prepare_deviations(
        returns, sample_weight=sample_weight, target=min_acceptable_return
    )
    return _mean(np.abs(deviations), sample_weight=sample_weight)


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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
    deviations, sample_weight, _ = _prepare_deviations(
        returns, sample_weight=sample_weight, target=min_acceptable_return
    )
    return _mean(np.maximum(-deviations, 0), sample_weight=sample_weight)


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
         Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
    returns = np.asarray(returns)
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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
    returns = np.asarray(returns, dtype=float)
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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
    deviations, sample_weight, _ = _prepare_deviations(
        returns, sample_weight=sample_weight
    )
    return _mean(deviations**3, sample_weight=sample_weight)


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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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

    Nearly constant returns produce NaN when the second central moment is at
    most `(eps * mean)**2`. Here `eps` is machine precision and `mean` uses the
    same sample weights.
    """
    return _standardized_moment(returns, order=3, sample_weight=sample_weight)


def fourth_central_moment(
    returns: ArrayLike, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the Fourth central moment.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
    deviations, sample_weight, _ = _prepare_deviations(
        returns, sample_weight=sample_weight
    )
    return _mean(deviations**4, sample_weight=sample_weight)


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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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

    Nearly constant returns produce NaN when the second central moment is at
    most `(eps * mean)**2`. Here `eps` is machine precision and `mean` uses the
    same sample weights.
    """
    return _standardized_moment(returns, order=4, sample_weight=sample_weight)


def fourth_lower_partial_moment(
    returns: ArrayLike,
    min_acceptable_return: float | FloatArray | None = None,
    sample_weight: FloatArray | None = None,
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

    min_acceptable_return : float or ndarray of shape (n_assets,), optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns.
        The default (`None`) is to use the returns mean.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Fourth lower partial moment.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.
    """
    deviations, sample_weight, _ = _prepare_deviations(
        returns, sample_weight=sample_weight, target=min_acceptable_return
    )
    return _mean(np.minimum(deviations, 0) ** 4, sample_weight=sample_weight)


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
    returns = np.asarray(returns)
    with warnings.catch_warnings():
        # all-NaN slice warning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return -np.nanmin(returns, axis=0)


def value_at_risk(
    returns: ArrayLike, beta: float = 0.95, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    r"""Compute the value at risk (VaR).

    The VaR is the smallest loss exceeded with probability at most
    :math:`1 - \beta`. It is the lower :math:`\beta`-quantile of the empirical loss
    distribution :math:`F_L`:

    .. math:: \mathrm{VaR}_{\beta} = \inf\{\ell \in \mathbb{R} : F_L(\ell) \geq \beta\}

    With `sample_weight`, :math:`F_L` is the weighted empirical distribution
    function.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    beta : float, default=0.95
        The VaR confidence level. Must be between 0 and 1. The endpoints select the
        best and worst usable returns, respectively; zero-weight observations are
        excluded.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Value at Risk.
        If `returns` is a 1D-array, the result is a float.
        If `returns` is a 2D-array, the result is a ndarray of shape (n_assets,).

    Notes
    -----
    With :math:`n` equally weighted observations and an integer
    :math:`k = (1 - \beta) n`, the VaR is the :math:`(k+1)`-th largest loss and the
    CVaR is the mean of the :math:`k` largest losses. For example, with 100
    observations and `beta=0.95`, the VaR is the sixth largest loss.

    At a boundary between two loss values, VaR selects the lower loss. Rounding
    differences in the confidence level and in probabilities are treated as ties
    at that boundary.

    NaN handling:
    NaN returns are excluded from each column's calculation. Remaining sample
    weights are rescaled to sum to one. The result is NaN if no observations
    or no positive weight remain.

    References
    ----------
    .. [1] "Conditional value-at-risk for general loss distributions",
        Journal of Banking & Finance, Rockafellar & Uryasev (2002)

    .. [2] "Quantitative Risk Management: Concepts, Techniques and Tools",
        Princeton University Press, McNeil, Frey & Embrechts (2015)
    """
    return _tail_risk(
        returns, beta=beta, sample_weight=sample_weight, conditional=False
    )


def cvar(
    returns: ArrayLike, beta: float = 0.95, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the CVaR (conditional value at risk).

    The CVaR (or Tail VaR) represents the mean shortfall at a specified confidence
    level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Array of return values.

    beta : float, default=0.95
        Confidence level between 0 and 1, inclusive. CVaR averages the worst
        `1 - beta` probability mass. At 0, CVaR is the negative mean
        of the usable returns. At 1, it is the largest loss with positive weight.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

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
    returns = np.asarray(returns)
    return theta * np.log(
        mean(np.exp(-returns / theta), sample_weight=sample_weight) / (1 - beta)
    )


def evar(
    returns: ArrayLike, beta: float = 0.95, sample_weight: FloatArray | None = None
) -> float:
    r"""Compute the EVaR (entropic value at risk).

    The EVaR is a coherent risk measure which is an upper bound for the VaR and the
    CVaR, obtained from the Chernoff inequality. The EVaR can be represented by using
    the concept of relative entropy.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    beta : float, default=0.95
        The EVaR confidence level. Must be between 0 and 1.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

    Returns
    -------
    value : float
        EVaR.

    Notes
    -----
    The EVaR of the returns :math:`X` at confidence level :math:`\beta` is

    .. math::
        \text{EVaR}_{\beta}(X) = \inf_{\theta > 0} \theta \log \left(
        \frac{\mathbb{E}\left[e^{-X / \theta}\right]}{1 - \beta} \right)

    It lies between the CVaR and the largest loss with positive weight. It equals
    that largest loss when its total probability is at least :math:`1-\beta`,
    and the weighted mean loss when `beta=0`. With equal weights, this condition
    is :math:`n (1-\beta) \le k`, where :math:`k` observations share the largest loss.

    NaN handling:
    NaN returns and their matching weights are excluded. The remaining weights
    are normalized. The result is NaN if no positive weight remains.

    References
    ----------
    .. [1] "Entropic Value-at-Risk: A New Coherent Risk Measure",
        Journal of Optimization Theory and Applications, Ahmadi-Javid (2012)
    """
    returns = np.asarray(returns, dtype=float)
    valid = ~np.isnan(returns)
    log_probabilities = None
    if sample_weight is not None:
        sample_weight = _validate_sample_weight(
            sample_weight, n_observations=returns.shape[0]
        )
        valid &= sample_weight > 0
        sample_weight = sample_weight[valid]
        # Equal weights use the unweighted calculation. For unequal weights,
        # normalize in log space so tiny positive probabilities do not become zero.
        if sample_weight.size and not np.all(sample_weight == sample_weight[0]):
            log_probabilities = np.log(sample_weight)
            log_probabilities -= scs.logsumexp(log_probabilities)
    losses = -returns[valid]
    if losses.size == 0:
        return np.nan

    max_loss = losses.max()
    spread = max_loss - losses.min()
    if beta == 0:
        value = (
            losses.mean()
            if sample_weight is None
            else _normalize_sample_weight(sample_weight) @ losses
        )
    elif beta == 1 or spread == 0:
        value = max_loss
    else:
        # The EVaR is translation equivariant and positively homogeneous.
        scaled_losses = (losses - max_loss) / spread
        if log_probabilities is None:
            scaled_evar = _unweighted_evar(scaled_losses, beta)
        else:
            scaled_evar = _weighted_evar(scaled_losses, beta, log_probabilities)
        value = max_loss + spread * scaled_evar
    # Adding 0.0 maps a negative zero to 0.0.
    return value + 0.0


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
    The running peak starts at the initial wealth (0 for uncompounded and 1 for
    compounded cumulative returns), so a loss on the first observation is a drawdown.

    NaN handling:
    Missing values (NaNs) remain at their original locations in the output and are
    treated as neutral elements during accumulation, so they do not propagate to
    subsequent values.
    """
    returns = np.asarray(returns)
    if np.isnan(returns).all():
        return np.full(returns.shape, np.nan, dtype=float)

    cumulative_returns = get_cumulative_returns(returns=returns, compounded=compounded)

    if np.isnan(cumulative_returns).any():
        mask = np.isnan(cumulative_returns)
        cum_clean = np.nan_to_num(cumulative_returns, nan=-np.inf)
    else:
        mask = None
        cum_clean = cumulative_returns

    # The starting wealth (0 uncompounded, 1.0 compounded) is the first peak, so a loss
    # on the first observation is a drawdown. It also replaces the -Inf left by leading
    # NaNs.
    peak = np.maximum.accumulate(cum_clean, axis=0)
    np.maximum(peak, 1.0 if compounded else 0.0, out=peak)

    if compounded:
        drawdowns = cum_clean / peak - 1
    else:
        drawdowns = cum_clean - peak

    if mask is not None:
        drawdowns[mask] = np.nan

    return drawdowns


def drawdown_at_risk(
    drawdowns: FloatArray, beta: float = 0.95, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    r"""Compute the Drawdown at risk.

    The Drawdown at risk (DaR) is the smallest drawdown exceeded with probability at
    most :math:`1 - \beta`. It is the value at risk of the drawdowns, see
    :func:`value_at_risk`.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Nonpositive drawdowns computed from the full return path, including
        dates with zero sample weight.

    beta : float, default = 0.95
        The DaR confidence level, between 0 and 1, inclusive.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed. NaN drawdowns and their weights are
        excluded, and the remaining weights are rescaled to sum to one.
        The result is NaN if no observations or no positive weight remain.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Drawdown at risk.
        If `drawdowns` is a 1D-array, the result is a float.
        If `drawdowns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return value_at_risk(returns=drawdowns, beta=beta, sample_weight=sample_weight)


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


def average_drawdown(
    drawdowns: FloatArray, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the average drawdown.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Nonpositive drawdowns computed from the full return path, including
        dates with zero sample weight.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed. NaN drawdowns and their weights are
        excluded, and the remaining weights are rescaled to sum to one.
        The result is NaN if no observations or no positive weight remain.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Average drawdown.
        If `drawdowns` is a 1D-array, the result is a float.
        If `drawdowns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return -mean(drawdowns, sample_weight=sample_weight)


def cdar(
    drawdowns: FloatArray, beta: float = 0.95, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the CDaR (conditional drawdown at risk).

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Nonpositive drawdowns computed from the full return path, including
        dates with zero sample weight.

    beta : float, default = 0.95
        The CDaR confidence level, between 0 and 1, inclusive.
        The result averages the worst drawdowns over probability `1 - beta`.
        At 1, it is the largest drawdown magnitude with positive weight.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed. NaN drawdowns and their weights are
        excluded, and the remaining weights are rescaled to sum to one.
        The result is NaN if no observations or no positive weight remain.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        CDaR.
        If `drawdowns` is a 1D-array, the result is a float.
        If `drawdowns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return cvar(returns=drawdowns, beta=beta, sample_weight=sample_weight)


def edar(
    drawdowns: FloatArray, beta: float = 0.95, sample_weight: FloatArray | None = None
) -> float:
    """Compute the EDaR (entropic drawdown at risk).

    The EDaR is a coherent risk measure which is an upper bound for the DaR and the
    CDaR, obtained from the Chernoff inequality. The EDaR can be represented by using
    the concept of relative entropy.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,)
        Nonpositive drawdowns computed from the full return path, including
        dates with zero sample weight.

    beta : float, default=0.95
        The EDaR confidence level, between 0 and 1, inclusive.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed. NaN drawdowns and their weights are
        excluded, and the remaining weights are rescaled to sum to one.
        The result is NaN if no observations or no positive weight remain.

    Returns
    -------
    value : float
        EDaR.
    """
    return evar(returns=drawdowns, beta=beta, sample_weight=sample_weight)


def ulcer_index(
    drawdowns: FloatArray, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the Ulcer index.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Nonpositive drawdowns computed from the full return path, including
        dates with zero sample weight.

    sample_weight : ndarray of shape (n_observations,), optional
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed. NaN drawdowns and their weights are
        excluded, and the remaining weights are rescaled to sum to one.
        The result is NaN if no observations or no positive weight remain.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Ulcer Index.
        If `drawdowns` is a 1D-array, the result is a float.
        If `drawdowns` is a 2D-array, the result is a ndarray of shape (n_assets,).
    """
    return np.sqrt(mean(np.square(drawdowns), sample_weight=sample_weight))


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
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

    Returns
    -------
    corr : ndarray of shape (n_assets, n_assets)
       The correlation matrix.
    """
    X = np.asarray(X)
    n_observations, n_assets = X.shape
    if sample_weight is not None:
        sample_weight = _validate_sample_weight(
            sample_weight, n_observations=n_observations
        )
        if not sample_weight.any():
            return np.full((n_assets, n_assets), np.nan)
        sample_weight = _normalize_sample_weight(sample_weight)
    cov = np.cov(X, rowvar=False, aweights=sample_weight)
    std = np.sqrt(np.diag(cov))
    return cov / np.outer(std, std)


def _prepare_weighted_returns(
    returns: FloatArray, *, sample_weight: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Prepare returns and weights for calculations that exclude NaNs.

    Inputs must be NumPy arrays of floats and are not modified.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Return values, possibly containing NaNs.

    sample_weight : ndarray of shape (n_observations,)
        Observation weights already checked to be finite and nonnegative.

    Returns
    -------
    returns : ndarray
        Returns after removing zero-weight rows and replacing NaNs with zero.
        Each replacement also receives zero weight, so it does not contribute
        to the calculation. If all input weights are zero, no rows remain.

    sample_weight : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Missing returns receive zero weight, and the remaining weights are
        rescaled to sum to one. A nonempty column with no remaining positive
        weight produces NaN weights.
        Weights stay 1D for 1D returns or inputs without NaNs. For 2D returns
        containing NaNs, each column gets its own normalized weight vector.
    """
    positive = sample_weight > 0
    if not positive.all():
        returns, sample_weight = returns[positive], sample_weight[positive]
    missing = np.isnan(returns)
    if missing.any():
        sample_weight = np.where(
            missing, 0.0, sample_weight[:, None] if returns.ndim == 2 else sample_weight
        )
        returns = np.where(missing, 0.0, returns)
    return returns, _normalize_sample_weight(sample_weight)


def _prepare_deviations(
    returns: ArrayLike,
    *,
    sample_weight: FloatArray | None,
    target: float | FloatArray | None = None,
) -> tuple[FloatArray, FloatArray | None, float | FloatArray]:
    """Prepare returns and weights, then subtract a target return.

    Parameters
    ----------
    returns : array-like of shape (n_observations,) or (n_observations, n_assets)
        Return values, possibly containing NaNs.

    sample_weight : ndarray of shape (n_observations,) or None
        Relative observation weights. Must be finite and nonnegative. If None,
        equal weights are assumed.

    target : float or ndarray of shape (n_assets,), optional
        Return to subtract. If None, use the mean of each column with the
        supplied weights, excluding NaNs.

    Returns
    -------
    deviations : ndarray
        Prepared returns minus the target. With weights, zero-weight rows are
        removed and missing returns are replaced before subtraction. Their
        deviations must be excluded using the returned weights. Without
        weights, NaNs remain in place.

    sample_weight : ndarray or None
        Weights aligned with the deviations and normalized after excluding NaN
        returns, separately for each column when needed. None if no weights
        were supplied.

    target : float or ndarray of shape (n_assets,)
        Target subtracted from the returns. When no target was supplied, this is
        the mean computed using the prepared returns and weights.
    """
    returns = np.asarray(returns, dtype=float)
    if sample_weight is not None:
        sample_weight = _validate_sample_weight(
            sample_weight, n_observations=returns.shape[0]
        )
        returns, sample_weight = _prepare_weighted_returns(
            returns, sample_weight=sample_weight
        )
    if target is None:
        target = _mean(returns, sample_weight=sample_weight)
    return returns - target, sample_weight, target


def _standardized_moment(
    returns: ArrayLike,
    *,
    order: int,
    sample_weight: FloatArray | None,
) -> float | FloatArray:
    """Compute skew or kurtosis from centered returns.

    Parameters
    ----------
    returns : array-like of shape (n_observations,) or (n_observations, n_assets)
        Return values, possibly containing NaNs.

    order : {3, 4}
        Moment order. Three gives skew and four gives kurtosis.

    sample_weight : ndarray of shape (n_observations,) or None
        Relative observation weights. None gives equal weights.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Standardized moment. NaN if no usable observations remain, the second
        moment is at most `(eps * mean)**2`, or the quotient is non-finite.
    """
    deviations, sample_weight, mean_return = _prepare_deviations(
        returns, sample_weight=sample_weight
    )
    # Remove the residual offset left by rounding the mean.
    deviations -= _mean(deviations, sample_weight=sample_weight)
    moment = _mean(deviations**order, sample_weight=sample_weight)
    second_moment = _mean(deviations**2, sample_weight=sample_weight)
    undefined = second_moment <= (np.finfo(float).eps * mean_return) ** 2
    denominator = np.where(undefined, np.nan, second_moment ** (order / 2))
    return safe_divide(moment, denominator, fill_value=np.nan)


def _mean(
    values: FloatArray, *, sample_weight: FloatArray | None = None
) -> float | FloatArray:
    """Compute the mean from prepared values and optional normalized weights.

    Parameters
    ----------
    values : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Values to average. Without weights, NaNs are excluded. With weights,
        missing values must have been handled by `_prepare_weighted_returns`
        before computing these values.

    sample_weight : ndarray of shape (n_observations,) or (n_observations, n_assets), optional
        Normalized weights from `_prepare_weighted_returns`, or None for an
        unweighted mean. This function does not validate or normalize weights.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Mean, or one mean per column. Empty inputs and columns with no usable
        observations produce NaN.
    """
    if values.shape[0] == 0:
        return np.full(values.shape[1:], np.nan)[()]
    if sample_weight is None:
        if np.isfinite(values).all():
            return values.mean(axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(values, axis=0)
    return _weighted_sum(values, sample_weight=sample_weight)


def _weighted_sum(
    values: FloatArray, *, sample_weight: FloatArray
) -> float | FloatArray:
    """Sum over observations, using column-specific weights when needed.

    Parameters
    ----------
    values : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Values to sum along the observation axis.

    sample_weight : ndarray of shape (n_observations,) or (n_observations, n_assets)
        A 1D weight vector shared by all columns, or a 2D array matching
        `values` that assigns weights separately for each column.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        Weighted sum. A scalar for 1D values, or one result per column for
        2D values.
    """
    if sample_weight.ndim == 1:
        return sample_weight @ values
    # Avoid allocating a full product array for column-specific weights.
    return np.einsum("ij,ij->j", sample_weight, values)


def _weighted_variance(
    returns: FloatArray,
    *,
    sample_weight: FloatArray,
    biased: bool,
    min_acceptable_return: float | FloatArray | None = None,
    downside: bool = False,
) -> float | FloatArray:
    r"""Compute weighted variance or semi-variance, excluding NaN returns.

    The remaining weights are rescaled to sum to one separately for each column.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,) or (n_observations, n_assets)
        Return values, possibly containing NaNs.

    sample_weight : ndarray of shape (n_observations,)
        Non-negative observation weights.

    biased : bool
        If True, return the population second moment. If False, divide it by
        :math:`1 - \sum_i w_i^2`, where :math:`w_i` are the weights after excluding
        NaN returns and rescaling.

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
    returns = np.asarray(returns, dtype=float)
    n_observations = returns.shape[0]
    sample_weight = _validate_sample_weight(
        sample_weight, n_observations=n_observations
    )
    returns, sample_weight = _prepare_weighted_returns(
        returns, sample_weight=sample_weight
    )
    if len(returns) == 0:
        return np.full(returns.shape[1:], np.nan)[()]
    if min_acceptable_return is None:
        min_acceptable_return = _weighted_sum(returns, sample_weight=sample_weight)
    deviations = returns - min_acceptable_return
    if downside:
        np.minimum(deviations, 0.0, out=deviations)
    np.square(deviations, out=deviations)
    result = _weighted_sum(deviations, sample_weight=sample_weight)
    if biased:
        return result
    correction = 1.0 - _weighted_sum(sample_weight, sample_weight=sample_weight)
    return result / np.where(correction == 0, np.nan, correction)


def _tail_risk(
    returns: ArrayLike,
    *,
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
        Confidence level in [0, 1].

    sample_weight : ndarray of shape (n_observations,) or None
        Non-negative observation weights. If None, the remaining observations
        have equal weight.

    conditional : bool
        If True, return CVaR, the average loss in the lower return tail.
        If False, return VaR, the lower `beta`-quantile of the loss.

    Returns
    -------
    value : float or ndarray of shape (n_assets,)
        VaR or CVaR. A scalar for 1D returns, or one result per column for
        2D returns. The result is NaN if no observations or no positive
        weight remain.
    """
    returns = np.asarray(returns, dtype=float)
    eps = np.finfo(float).eps

    def _unweighted(values: FloatArray) -> float | FloatArray:
        """Compute the unweighted tail measure using the enclosing settings."""
        size = values.shape[0]
        if size == 0:
            return np.nan
        if beta == 1:
            return -values.min(axis=0)
        k = (1.0 - beta) * size
        if conditional:
            i = max(0, int(np.ceil(k) - 1))
        else:
            # The tolerance absorbs the floating-point error of `k`, so that an
            # integer tail size such as `(1 - 0.9) * 10` is not rounded down.
            i = min(size - 1, int(np.floor(k + 4 * eps * size)))
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

    sample_weight = _validate_sample_weight(
        sample_weight, n_observations=returns.shape[0]
    )
    positive = sample_weight > 0

    def _weighted(column: FloatArray) -> float:
        """Compute the weighted tail measure for one return column."""
        valid = ~np.isnan(column) & positive
        values, probs = column[valid], sample_weight[valid]
        if len(probs) == 0:
            return np.nan
        probs = _normalize_sample_weight(probs)
        if beta == 0:
            return -probs @ values if conditional else -values.max()
        if beta == 1:
            return -values.min()
        if np.all(probs == probs[0]):
            # Preserve the unweighted empirical rank convention exactly.
            return float(_unweighted(values))
        order = np.argsort(values)
        values, probs = values[order], probs[order]
        cumulative = np.cumsum(probs)
        tail_mass = (1.0 - beta) * cumulative[-1]
        if not conditional:
            # Allow for rounding in beta and in the accumulated weights.
            # Scale the weight tolerance by the tail mass: even a tiny probability
            # can cover the whole tail when beta is close to one.
            beta_tolerance = 0.5 * np.spacing(beta) * cumulative[-1]
            weight_tolerance = 4 * eps * len(values) * tail_mass
            i = np.searchsorted(
                cumulative,
                tail_mass + (beta_tolerance + weight_tolerance),
                side="right",
            )
            return -values[min(i, len(values) - 1)]
        i = np.searchsorted(cumulative, tail_mass)
        if i == 0:
            return -values[i]
        return (
            -(probs[:i] @ values[:i] + values[i] * (tail_mass - cumulative[i - 1]))
            / tail_mass
        )

    if returns.ndim == 1:
        return _weighted(returns)
    return np.array([_weighted(column) for column in returns.T])


def _unweighted_evar(losses: FloatArray, beta: float) -> float:
    r"""Compute the EVaR of losses standardized to [-1, 0] with a maximum of 0.

    With :math:`t = 1 / \theta` and :math:`c = \log(n (1 - \beta))`, the EVaR is the
    infimum over :math:`t > 0` of :math:`f(t) = (\log \sum_i e^{t x_i} - c) / t`. The
    exponents are non-positive and the largest is 0, so the sum cannot overflow or
    underflow. The derivative of :math:`f` has the sign of
    :math:`g(t) = t \, \mathbb{E}_w[x] - \log \sum_i e^{t x_i} + c`, with
    :math:`w_i \propto e^{t x_i}`, which increases from :math:`\log(1 - \beta)` at
    :math:`t = 0`. The minimizer is the root of :math:`g`.

    Parameters
    ----------
    losses : ndarray of shape (n_observations,)
        Standardized losses.

    beta : float
        Confidence level in (0, 1).

    Returns
    -------
    value : float
        EVaR of the standardized losses, between -1 and 0.
    """
    c = np.log(losses.size) + np.log1p(-beta)

    def objective(log_t: float) -> float:
        """Compute :math:`f(t)`, where `log_t` is the log of :math:`t`."""
        t = np.exp(log_t)
        return (np.log(np.exp(t * losses).sum()) - c) / t

    def gradient_sign(log_t: float) -> float:
        """Compute :math:`g(t)`, where `log_t` is the log of :math:`t`."""
        t = np.exp(log_t)
        exp_losses = np.exp(t * losses)
        total = exp_losses.sum()
        return t * (exp_losses @ losses) / total - np.log(total) + c

    # The variance of the losses under w is at most 1/4, so
    # g(t) <= log(1 - beta) + t**2 / 8, which is negative at `lower`.
    lower = 0.5 * np.log(-2.0 * np.log1p(-beta))
    upper = 20.0
    if gradient_sign(upper) <= 0:
        # The infimum is reached as theta tends to 0, and f is within 1e-7 of its
        # limit 0 past `upper`.
        return 0.0
    return objective(sco.brentq(gradient_sign, lower, upper))


def _weighted_evar(
    losses: FloatArray, beta: float, log_probabilities: FloatArray
) -> float:
    """Compute weighted EVaR for losses scaled to [-1, 0], with maximum 0.

    Parameters
    ----------
    losses : ndarray of shape (n_observations,)
        Finite losses scaled so the minimum is -1 and the maximum is 0.
        Observations with zero probability must already have been removed.

    beta : float
        Confidence level strictly between 0 and 1.

    log_probabilities : ndarray of shape (n_observations,)
        Natural logarithms of the positive observation probabilities, aligned
        with `losses`. These probabilities must sum to one. Keeping them in
        logarithmic form preserves very small probabilities in the calculation.

    Returns
    -------
    value : float
        EVaR of the scaled losses, between their weighted mean and zero.

    Notes
    -----
    Find the minimum of the EVaR objective by locating where its derivative
    changes sign. The search uses the logarithm of the inverse temperature,
    where temperature is the positive scale parameter in the EVaR formula.

    Near confidence zero, use the weighted mean if Hoeffding's bound puts it
    within 1e-12 of EVaR. At a large inverse temperature, use the maximum loss
    if the objective is within 1e-12 of it. This second error is bounded by
    `log((1 - beta) / probability_at_maximum) / inverse_temperature`.
    Both bounds apply on the scaled loss range [-1, 0].
    """
    log_tail = np.log1p(-beta)
    log_maximum_mass = scs.logsumexp(log_probabilities[losses == 0])
    if log_maximum_mass >= log_tail:
        return 0.0
    probabilities = _normalize_sample_weight(np.exp(log_probabilities))
    risk_tolerance = 1e-12
    if np.sqrt(-0.5 * log_tail) <= risk_tolerance:
        return probabilities @ losses

    def evaluate(log_inverse_temperature: float) -> tuple[float, float]:
        """Return the EVaR objective and a value with its derivative's sign."""
        inverse_temperature = np.exp(log_inverse_temperature)
        scaled_losses = inverse_temperature * losses
        if inverse_temperature < 1:
            # At small confidence levels the exponential average is close to one.
            # Preserve its small change instead of subtracting rounded logarithms.
            log_average = np.log1p(probabilities @ np.expm1(scaled_losses))
            tilted_weights = probabilities * np.exp(scaled_losses - log_average)
        else:
            log_average = scs.logsumexp(log_probabilities + scaled_losses)
            tilted_weights = np.exp(log_probabilities + scaled_losses - log_average)
        objective = (log_average - log_tail) / inverse_temperature
        gradient = tilted_weights @ scaled_losses - log_average + log_tail
        return objective, gradient

    def gradient_sign(log_inverse_temperature: float) -> float:
        """Return a value with the same sign as the objective's derivative."""
        return evaluate(log_inverse_temperature)[1]

    lower = 0.5 * np.log(-2.0 * log_tail)
    upper = 20.0
    while gradient_sign(upper) <= 0:
        if (log_tail - log_maximum_mass) * np.exp(-upper) <= risk_tolerance:
            return 0.0
        upper += 4.0
    root = sco.brentq(gradient_sign, lower, upper)
    return evaluate(root)[0]
