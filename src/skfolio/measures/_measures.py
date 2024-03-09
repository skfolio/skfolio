"""Module that includes all Measures functions used across `skfolio`."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Gini mean difference and OWA GMD weights features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

import numpy as np
import scipy.optimize as sco


def mean(returns: np.ndarray) -> float:
    """Compute the mean.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Mean
    """
    return returns.mean()


def mean_absolute_deviation(
    returns: np.ndarray, min_acceptable_return: float | None = None
) -> float:
    """Compute the mean absolute deviation (MAD).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    min_acceptable_return : float, optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns.
        The default (`None`) is to use the returns' mean.

    Returns
    -------
    value : float
        Mean absolute deviation.
    """
    if min_acceptable_return is None:
        min_acceptable_return = np.mean(returns, axis=0)
    return float(np.mean(np.abs(returns - min_acceptable_return)))


def first_lower_partial_moment(
    returns: np.ndarray, min_acceptable_return: float | None = None
) -> float:
    """Compute the first lower partial moment.

    The first lower partial moment is the mean of the returns below a minimum
    acceptable return.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns

    min_acceptable_return : float, optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns.
        The default (`None`) is to use the mean.

    Returns
    -------
    value : float
        First lower partial moment.
    """
    if min_acceptable_return is None:
        min_acceptable_return = np.mean(returns, axis=0)
    return -np.sum(np.minimum(0, returns - min_acceptable_return)) / len(returns)


def variance(returns: np.ndarray) -> float:
    """Compute the variance (second moment).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Variance.
    """
    return returns.var(ddof=1)


def semi_variance(
    returns: np.ndarray, min_acceptable_return: float | None = None
) -> float:
    """Compute the semi-variance (second lower partial moment).

    The semi-variance is the variance of the returns below a minimum acceptable return.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns

    min_acceptable_return : float, optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns.
        The default (`None`) is to use the mean.

    Returns
    -------
    value : float
        Semi-variance.
    """
    if min_acceptable_return is None:
        min_acceptable_return = np.mean(returns, axis=0)
    return np.sum(np.power(np.minimum(0, returns - min_acceptable_return), 2)) / (
        len(returns) - 1
    )


def standard_deviation(returns: np.ndarray) -> float:
    """Compute the standard-deviation (square root of the second moment).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Standard-deviation.
    """
    return np.sqrt(variance(returns=returns))


def semi_deviation(
    returns: np.ndarray, min_acceptable_return: float | None = None
) -> float:
    """Compute the semi standard-deviation (semi-deviation) (square root of the second lower
    partial moment).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    min_acceptable_return : float, optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns.
        The default (`None`) is to use the returns mean.

    Returns
    -------
    value : float
        Semi-standard-deviation.
    """
    return np.sqrt(
        semi_variance(returns=returns, min_acceptable_return=min_acceptable_return)
    )


def third_central_moment(returns: np.ndarray) -> float:
    """Compute the third central moment.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Third central moment.
    """

    return np.sum(np.power(returns - np.mean(returns, axis=0), 3)) / len(returns)


def skew(returns: np.ndarray) -> float:
    """Compute the Skew.

    The Skew is a measure of the lopsidedness of the distribution.
    A symmetric distribution have a Skew of zero.
    Higher Skew corresponds to longer right tail.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Skew.
    """

    return third_central_moment(returns) / standard_deviation(returns) ** 3


def fourth_central_moment(returns: np.ndarray) -> float:
    """Compute the Fourth central moment.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Fourth central moment.
    """
    return np.sum(np.power(returns - np.mean(returns, axis=0), 4)) / len(returns)


def kurtosis(returns: np.ndarray) -> float:
    """Compute the Kurtosis.

    The Kurtosis is a measure of the heaviness of the tail of the distribution.
    Higher Kurtosis corresponds to greater extremity of deviations (fat tails).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Kurtosis.
    """

    return fourth_central_moment(returns) / standard_deviation(returns) ** 4


def fourth_lower_partial_moment(
    returns: np.ndarray, min_acceptable_return: float | None = None
) -> float:
    """Compute the fourth lower partial moment.

    The Fourth Lower Partial Moment is a measure of the heaviness of the downside tail
    of the returns below a minimum acceptable return.
    Higher Fourth Lower Partial Moment corresponds to greater extremity of downside
    deviations (downside fat tail).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns

    min_acceptable_return : float, optional
        Minimum acceptable return. It is the return target to distinguish "downside" and
        "upside" returns.
        The default (`None`) is to use the returns mean.

    Returns
    -------
    value : float
        Fourth lower partial moment.
    """
    if min_acceptable_return is None:
        min_acceptable_return = np.mean(returns, axis=0)
    return np.sum(np.power(np.minimum(0, returns - min_acceptable_return), 4)) / len(
        returns
    )


def worst_realization(returns: np.ndarray) -> float:
    """Compute the worst realization (worst return).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Worst realization.
    """
    return -min(returns)


def value_at_risk(returns: np.ndarray, beta: float = 0.95) -> float:
    """Compute the historical value at risk (VaR).

    The VaR is the maximum loss at a given confidence level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    beta : float, default=0.95
        The VaR confidence level (return on the worst (1-beta)% observation).

    Returns
    -------
    value : float
        Value at Risk.
    """
    k = (1 - beta) * len(returns)
    ik = max(0, int(np.ceil(k) - 1))
    # We only need the first k elements so using `partition` O(n log(n)) is faster
    # than `sort` O(n).
    ret = np.partition(returns, ik)
    return -ret[ik]


def cvar(returns: np.ndarray, beta: float = 0.95) -> float:
    """Compute the historical CVaR (conditional value at risk).

    The CVaR (or Tail VaR) represents the mean shortfall at a specified confidence
    level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    beta : float, default=0.95
        The CVaR confidence level (expected VaR on the worst (1-beta)% observations).

    Returns
    -------
    value : float
        CVaR.
    """
    k = (1 - beta) * len(returns)
    ik = max(0, int(np.ceil(k) - 1))
    # We only need the first k elements so using `partition` O(n log(n)) is faster
    # than `sort` O(n).
    ret = np.partition(returns, ik)
    return -np.sum(ret[:ik]) / k + ret[ik] * (ik / k - 1)


def entropic_risk_measure(
    returns: np.ndarray, theta: float = 1, beta: float = 0.95
) -> float:
    """Compute the entropic risk measure.

    The entropic risk measure is a risk measure which depends on the risk aversion
    defined by the investor (theat) through the exponential utility function at a given
    confidence level (beta).

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    theta : float, default=1.0
        Risk aversion.

    beta : float, default=0.95
         Confidence level.

    Returns
    -------
    value : float
        Entropic risk measure.
    """
    return theta * np.log(np.mean(np.exp(-returns / theta)) / (1 - beta))


def evar(returns: np.ndarray, beta: float = 0.95) -> float:
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

    def func(x: float) -> float:
        return entropic_risk_measure(returns=returns, theta=x, beta=beta)

    # The lower bound is chosen to avoid exp overflow
    lower_bound = np.max(-returns) / 100
    result = sco.minimize(
        func,
        x0=np.array([lower_bound * 2]),
        method="SLSQP",
        bounds=[(lower_bound, np.inf)],
        tol=1e-10,
    )
    return result.fun


def get_cumulative_returns(returns: np.ndarray, compounded: bool = False) -> np.ndarray:
    """Compute the cumulative returns from the returns.
    Non-compounded cumulative returns start at 0.
    Compounded cumulative returns are rescaled to start at 1000.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    compounded : bool, default=False
        If this is set to True, the cumulative returns are compounded otherwise they
        are uncompounded.

    Returns
    -------
    values: ndarray of shape (n_observations,)
        Cumulative returns.
    """
    if compounded:
        cumulative_returns = 1000 * np.cumprod(1 + returns)  # Rescaled to start at 1000
    else:
        cumulative_returns = np.cumsum(returns)
    return cumulative_returns


def get_drawdowns(returns: np.ndarray, compounded: bool = False) -> np.ndarray:
    """Compute the drawdowns' series from the returns.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
       Vector of returns.

    compounded : bool, default=False
       If this is set to True, the cumulative returns are compounded otherwise they
       are uncompounded.

    Returns
    -------
    values: ndarray of shape (n_observations,)
       Drawdowns.
    """
    cumulative_returns = get_cumulative_returns(returns=returns, compounded=compounded)
    if compounded:
        drawdowns = cumulative_returns / np.maximum.accumulate(cumulative_returns) - 1
    else:
        drawdowns = cumulative_returns - np.maximum.accumulate(cumulative_returns)
    return drawdowns


def drawdown_at_risk(drawdowns: np.ndarray, beta: float = 0.95) -> float:
    """Compute the Drawdown at risk.

    The Drawdown at risk is the maximum drawdown at a given confidence level (beta).

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,)
        Vector of drawdowns.

    beta : float, default = 0.95
        The DaR confidence level (drawdown on the worst (1-beta)% observations).

    Returns
    -------
    value : float
       Drawdown at risk.
    """
    return value_at_risk(returns=drawdowns, beta=beta)


def max_drawdown(drawdowns: np.ndarray) -> float:
    """Compute the maximum drawdown.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,)
        Vector of drawdowns.

    Returns
    -------
    value : float
        Maximum drawdown.
    """
    return drawdown_at_risk(drawdowns=drawdowns, beta=1)


def average_drawdown(drawdowns: np.ndarray) -> float:
    """Compute the average drawdown.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,)
        Vector of drawdowns.

    Returns
    -------
    value : float
        Average drawdown.
    """
    return cdar(drawdowns=drawdowns, beta=0)


def cdar(drawdowns: np.ndarray, beta: float = 0.95) -> float:
    """Compute the historical CDaR (conditional drawdown at risk).

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,)
        Vector of drawdowns.

    beta : float, default = 0.95
        The CDaR confidence level (expected drawdown on the worst
        (1-beta)% observations).

    Returns
    -------
    value : float
        CDaR.
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


def ulcer_index(drawdowns: np.ndarray) -> float:
    """Compute the Ulcer index.

    Parameters
    ----------
    drawdowns : ndarray of shape (n_observations,)
        Vector of drawdowns.

    Returns
    -------
    value : float
        Ulcer index.
    """
    return np.sqrt(np.sum(np.power(drawdowns, 2)) / len(drawdowns))


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


def gini_mean_difference(returns: np.ndarray) -> float:
    """Compute the Gini mean difference (GMD).

    The GMD is the expected absolute difference between two realisations.
    The GMD is a superior measure of variability  for non-normal distribution than the
    variance.
    It can be used to form necessary conditions for second-degree stochastic dominance,
    while the variance cannot.

    Parameters
    ----------
    returns : ndarray of shape (n_observations,)
        Vector of returns.

    Returns
    -------
    value : float
        Gini mean difference.
    """
    w = owa_gmd_weights(len(returns))
    return float(w @ np.sort(returns, axis=0))


def effective_number_assets(weights: np.ndarray) -> float:
    r"""Computes the effective number of assets, defined as the inverse of the
    Herfindahl index [1]_:

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
