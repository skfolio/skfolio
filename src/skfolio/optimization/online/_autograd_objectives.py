"""Loss functions and gradients for online portfolio optimization.

The internal objective catalogue provides analytical gradients and optional
Autograd derivatives for portfolio return and risk measures. Individual functions
specify their data windows, smoothing parameters and return conventions.
Supported derivatives do not by themselves establish an algorithm's regret bound;
that also depends on the loss assumptions, feasible set and update rule.
"""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from importlib import import_module
from typing import Any, NoReturn, cast

import numpy as np

try:
    # Autograd builds its NumPy namespace dynamically and is an optional backend.
    anp = import_module("autograd.numpy")
    grad = import_module("autograd").grad
except ImportError:
    # Analytical objectives must work without a runtime autograd dependency.
    anp = np

    def grad(*args: Any, **kwargs: Any) -> NoReturn:
        raise ImportError(
            "autograd is required only for use_autograd=True; "
            "use analytical gradients or install autograd for verification."
        )


from skfolio.measures import (
    cdar,
    cvar,
    edar,
    evar,
    first_lower_partial_moment,
    get_cumulative_returns,
    gini_mean_difference,
    log_wealth,
    mean,
    mean_absolute_deviation,
    owa_gmd_weights,
    semi_deviation,
    semi_variance,
    standard_deviation,
    variance,
    worst_realization,
)
from skfolio.measures._enums import BaseMeasure, PerfMeasure, RiskMeasure
from skfolio.optimization.online._utils import CLIP_EPSILON


class BaseObjective(ABC):
    """Base class for all objective functions in online portfolio optimization.

    Parameters
    ----------
    use_autograd : bool, default=False
        If True, use automatic differentiation to compute gradients.
        If False, use analytical gradients (when available).
    """

    def __init__(self, use_autograd: bool = False) -> None:
        self.use_autograd = use_autograd

    @abstractmethod
    def loss(
        self, weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
    ) -> float:
        """Compute the loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        net_returns : ndarray of shape (n_assets,) or (n_observations, n_assets)
            Net returns of assets.
        **kwargs
            Additional measure-specific parameters (e.g., beta for CVaR).

        Returns
        -------
        float
            Loss value.
        """
        pass

    @abstractmethod
    def grad(
        self, weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Compute the gradient of the loss.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        net_returns : ndarray of shape (n_assets,) or (n_observations, n_assets)
            Net returns of assets.
        **kwargs
            Additional measure-specific parameters (e.g., beta for CVaR).

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        pass


def _sample_covariance(net_returns: np.ndarray) -> np.ndarray:
    """Compute unbiased sample covariance matrix from net returns.

    Parameters
    ----------
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.

    Returns
    -------
    ndarray of shape (n_assets, n_assets)
        Unbiased sample covariance matrix (divides by T-1).
    """
    T = net_returns.shape[0]
    mean_returns = np.mean(net_returns, axis=0, keepdims=True)
    centered = net_returns - mean_returns
    # Use unbiased estimator (divide by T-1, not T)
    return (centered.T @ centered) / (T - 1) if T > 1 else (centered.T @ centered)


def _analytical_log_wealth_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of log-wealth: ∇_w (-log(1 + w^T r)) = -r / (1 + w^T r).

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_assets,)
        Net returns of assets (single observation). LOG_WEALTH is a per-observation
        measure and is excluded from automatic 2D reshaping.
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    portfolio_net_return = np.dot(weights, net_returns)
    gross_return = 1.0 + portfolio_net_return
    if gross_return <= 0:
        warnings.warn(
            "Non-positive portfolio return, returning zero gradient.",
            UserWarning,
            stacklevel=2,
        )
        return np.zeros_like(weights)
    return -net_returns / gross_return


def _analytical_mean_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of mean (expected return).

    Mean gradient: ∇_w mean = mean(X)
    where X is the returns matrix.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights (unused, included for signature consistency).
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    # Gradient of mean(w^T r) w.r.t. w is mean(r)
    return np.mean(net_returns, axis=0)


def _analytical_variance_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of variance: ∇_w var = 2Σw.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    T = net_returns.shape[0]
    if T == 1:
        # Single observation: use proxy (w^T r)^2, gradient = 2(w^T r)r
        portfolio_return = net_returns[0] @ weights
        return 2.0 * portfolio_return * net_returns[0]

    Sigma = _sample_covariance(net_returns)
    return 2.0 * (weights @ Sigma)


def _analytical_semi_variance_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of semi-variance (unbiased estimator).

    When min_acceptable_return is None (defaults to mean), the gradient includes
    the chain rule through the mean:
    ∇_w SV = (2T/(T-1)) * (1/T) Σ_{t: r_t^T w < μ(w)} (r_t^T w - μ(w)) * (r_t - ∇μ)
    where μ(w) = mean(w^T r) and ∇μ = mean(r).

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Additional parameters (min_acceptable_return).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.

    Notes
    -----
    When min_acceptable_return is an external constant, the gradient simplifies.
    When it depends on weights (via mean), we must apply the chain rule properly.
    """
    T = net_returns.shape[0]
    portfolio_returns = net_returns @ weights

    # Check if min_acceptable_return is provided or defaults to mean
    min_acceptable_return_arg = kwargs.get("min_acceptable_return", None)
    is_mean_dependent = min_acceptable_return_arg is None

    if is_mean_dependent:
        min_acceptable_return = np.mean(portfolio_returns)
        mean_gradient = np.mean(net_returns, axis=0)  # ∇_w mean(portfolio_returns)
    else:
        min_acceptable_return = min_acceptable_return_arg
        mean_gradient = np.zeros(net_returns.shape[1])  # No dependency on w

    # Indicator for returns below threshold
    below_threshold = portfolio_returns < min_acceptable_return

    grad = np.zeros_like(weights)
    if np.any(below_threshold):
        bad_returns = net_returns[below_threshold]
        bad_portfolio_returns = portfolio_returns[below_threshold]
        deviations = bad_portfolio_returns - min_acceptable_return

        # Gradient with chain rule: 2 * Σ(deviation) * (r_t - ∇τ)
        # where ∇τ = mean_gradient if threshold is mean-dependent, else 0
        grad_terms = bad_returns - mean_gradient[np.newaxis, :]
        grad = 2.0 * np.sum(deviations[:, np.newaxis] * grad_terms, axis=0) / T

        # Apply unbiased correction
        if T > 1:
            grad = grad * T / (T - 1)

    return grad


def _analytical_standard_deviation_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of Standard Deviation.

    SD = sqrt(var(r^T w)) with unbiased variance
    ∇_w SD = (1 / ((T-1) * SD)) Σ_t (r_t^T w - μ)(r_t - mean(r))

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    T = net_returns.shape[0]
    if T == 1:
        # Single observation proxy: |w^T r|
        portfolio_return = net_returns[0] @ weights
        return np.sign(portfolio_return) * net_returns[0]

    portfolio_returns = net_returns @ weights
    mean_portfolio = np.mean(portfolio_returns)
    mean_asset = np.mean(net_returns, axis=0)
    deviations = portfolio_returns - mean_portfolio
    # Use unbiased variance (divide by T-1)
    variance = np.sum(deviations**2) / (T - 1)
    std = np.sqrt(variance + 1e-10)  # avoid division by zero

    # Gradient: (1 / ((T-1) * std)) * Σ (r_t^p - μ)(r_t - mean(r))
    grad = ((deviations[:, None]) * (net_returns - mean_asset)).sum(axis=0) / (
        (T - 1) * std
    )
    return grad


def _analytical_semi_deviation_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of Semi-Deviation (unbiased estimator).

    SemiDev = sqrt(unbiased_semi_variance)
    Uses chain rule: ∇_w SemiDev = (1 / (2 * SemiDev)) * ∇_w SemiVar

    When min_acceptable_return is mean-dependent, accounts for chain rule.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Additional parameters (min_acceptable_return).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    # Compute gradient of semi-variance (handles mean-dependent threshold)
    grad_semi_var = _analytical_semi_variance_gradient(weights, net_returns, **kwargs)

    # Compute semi-deviation value for chain rule
    T = net_returns.shape[0]
    portfolio_returns = net_returns @ weights
    min_acceptable_return_arg = kwargs.get("min_acceptable_return", None)

    if min_acceptable_return_arg is None:
        min_acceptable_return = np.mean(portfolio_returns)
    else:
        min_acceptable_return = min_acceptable_return_arg

    downside = np.maximum(0, min_acceptable_return - portfolio_returns)
    biased_semi_var = np.mean(downside**2)
    semi_var = biased_semi_var * T / (T - 1) if T > 1 else biased_semi_var
    semi_dev = np.sqrt(semi_var + CLIP_EPSILON)

    # Chain rule: ∇SemiDev = (1 / (2*SemiDev)) * ∇SemiVar
    grad = grad_semi_var / (2.0 * semi_dev)
    return grad


def _analytical_mean_absolute_deviation_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of Mean Absolute Deviation.

    MAD gradient: ∇_w MAD = (1/T) Σ_t sign(r_t^T w - μ) r_t
    where μ = mean(r^T w)

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    T = net_returns.shape[0]
    if T == 1:
        # Single observation: use proxy |w^T r - mar|, gradient = sign(w^T r - mar) * r
        min_acceptable_return = kwargs.get("min_acceptable_return", 0.0)
        portfolio_return = net_returns[0] @ weights
        sign = np.sign(portfolio_return - min_acceptable_return)
        return sign * net_returns[0]

    portfolio_returns = net_returns @ weights
    mean_return = np.mean(portfolio_returns)
    deviations = portfolio_returns - mean_return
    signs = np.sign(deviations)
    # Chain rule: d/dw |p_t - μ| = sign(p_t - μ) * (r_t - mean(r))
    mean_asset = np.mean(net_returns, axis=0)
    centered = net_returns - mean_asset
    grad = (centered.T @ signs) / T
    return grad


def _analytical_first_lower_partial_moment_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of First Lower Partial Moment.

    FLPM = mean(max(0, τ - w^T r))

    When τ = mean(w^T r), the gradient includes chain rule through the mean:
    ∇_w FLPM = (1/T) Σ_{t: r_t^T w < μ} (∇μ - r_t)
    where μ(w) = mean(w^T r) and ∇μ = mean(r).

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Additional parameters (min_acceptable_return).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    T = net_returns.shape[0]
    portfolio_returns = net_returns @ weights

    # Check if min_acceptable_return is provided or defaults to mean
    min_acceptable_return_arg = kwargs.get("min_acceptable_return", None)
    is_mean_dependent = min_acceptable_return_arg is None

    if is_mean_dependent:
        min_acceptable_return = np.mean(portfolio_returns)
        mean_gradient = np.mean(net_returns, axis=0)  # ∇_w mean(portfolio_returns)
    else:
        min_acceptable_return = min_acceptable_return_arg
        mean_gradient = np.zeros(net_returns.shape[1])  # No dependency on w

    below_threshold = portfolio_returns < min_acceptable_return
    grad = np.zeros_like(weights)
    if np.any(below_threshold):
        bad_returns = net_returns[below_threshold]
        # Gradient with chain rule: Σ(∇τ - r_t) where threshold depends on w
        # FLPM = mean(max(0, τ - r^T w)), so ∇FLPM = mean((∇τ - r) * indicator)
        grad_terms = mean_gradient[np.newaxis, :] - bad_returns
        grad = np.sum(grad_terms, axis=0) / T
    return grad


def _analytical_worst_realization_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of Worst Realization (negative minimum return).

    Worst gradient is one-hot at the index of the worst portfolio return.
    ∇_w (-min r^T w) = -r_{worst}

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector (one-hot at worst observation).
    """
    portfolio_returns = net_returns @ weights
    worst_idx = np.argmin(portfolio_returns)
    # Gradient of -min is negative of the return vector at worst time
    return -net_returns[worst_idx]


def _analytical_gini_mean_difference_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of Gini Mean Difference.

    GMD gradient uses OWA (Ordered Weighted Averaging) weights.
    ∇_w GMD = Σ_t w_π(t) r_t
    where π is the sorting permutation and w_k = (4k - 2(T+1)) / (T(T-1))
    This matches skfolio.measures.owa_gmd_weights.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Unused, included for signature consistency.

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    T = net_returns.shape[0]
    if T == 1:
        # Single observation: use proxy |w^T r|, gradient = sign(w^T r) * r
        portfolio_return = net_returns[0] @ weights
        sign = np.sign(portfolio_return)
        return sign * net_returns[0]

    portfolio_returns = net_returns @ weights
    # Sort returns and get permutation
    sort_idx = np.argsort(portfolio_returns)
    owa_weights = owa_gmd_weights(T)
    # Apply OWA weights to sorted returns
    sorted_returns = net_returns[sort_idx]
    grad = sorted_returns.T @ owa_weights
    return grad


# ============================================================================
# Helper functions for entropic risk measures (EVaR, EDaR, CDaR with drawdowns)
# ============================================================================


def _find_theta_star_empirical(
    x: np.ndarray, beta: float = 0.95, tol: float = 1e-8, maxiter: int = 200
) -> float:
    """Find theta > 0 minimizing F(theta) = (1/theta)(log(mean(exp(theta*x))) - log(1-beta)).

    Uses Newton's method with safe bracketing fallback.

    Parameters
    ----------
    x : ndarray of shape (N,)
        Loss or drawdown values (nonnegative).
    beta : float, default=0.95
        Confidence level.
    tol : float, default=1e-8
        Convergence tolerance for derivative.
    maxiter : int, default=200
        Maximum iterations.

    Returns
    -------
    float
        Optimal theta value.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return 1.0

    # If all equal, any theta works
    if np.allclose(x, x[0]):
        return 1.0

    log1mbeta = np.log(1.0 - beta)

    def F_and_dF(theta: float) -> tuple[float, float]:
        """Compute F(theta) and dF/dtheta with numerical stability."""
        a = theta * x
        a_max = np.max(a)
        shifted = np.exp(a - a_max)
        M = shifted.mean() * np.exp(a_max)
        M1 = (x * shifted).mean() * np.exp(a_max)
        F = (np.log(M) - log1mbeta) / theta
        dF = -(np.log(M) - log1mbeta) / (theta * theta) + (M1 / M) / theta
        return F, dF

    # Initialize bracket
    theta = 1.0
    _F0, dF0 = F_and_dF(theta)
    low = 1e-12
    high = None
    sign0 = np.sign(dF0)

    # Find bracket where dF changes sign
    for it in range(80):
        theta_try = theta * (2.0**it)
        _F_try, dF_try = F_and_dF(theta_try)
        if np.sign(dF_try) != sign0:
            low = min(theta, theta_try)
            high = max(theta, theta_try)
            break

    if high is None:
        low = 1e-12
        high = max(1.0, theta * (2.0**8))

    # Newton with backtracking
    theta_k = max(1e-8, min(1.0, (low + high) / 2.0))
    for _ in range(maxiter):
        _Fk, dFk = F_and_dF(theta_k)
        if abs(dFk) < tol:
            return float(theta_k)

        # Numerical second derivative
        eps = 1e-6 * max(1.0, theta_k)
        _, dF_plus = F_and_dF(theta_k + eps)
        _, dF_minus = F_and_dF(theta_k - eps)
        d2 = (dF_plus - dF_minus) / (2 * eps)

        if d2 == 0 or not np.isfinite(d2):
            theta_k = 0.5 * (low + high)
            continue

        delta = dFk / d2
        theta_new = theta_k - delta

        # Clamp to bracket
        if theta_new <= low or theta_new >= high or not np.isfinite(theta_new):
            theta_new = 0.5 * (theta_k + (high if dFk > 0 else low))

        # Update bracket
        _F_new, dF_new = F_and_dF(theta_new)
        if abs(dF_new) < tol:
            return float(theta_new)

        if np.sign(dF_new) == np.sign(dFk):
            if dFk > 0:
                high = theta_new
            else:
                low = theta_new
        else:
            if dFk > 0:
                low = theta_new
            else:
                high = theta_new
        theta_k = theta_new

    return float(theta_k)


def _wealth_and_dwealth(
    net_returns: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wealth S_t and derivatives dS_t/dw.

    Parameters
    ----------
    net_returns : ndarray of shape (T, d)
        Asset net returns.
    w : ndarray of shape (d,)
        Portfolio weights.

    Returns
    -------
    S : ndarray of shape (T,)
        Wealth levels S_t (assumes S_0 = 1).
    dS : ndarray of shape (T, d)
        Derivatives dS_t/dw.
    """
    net_returns = np.asarray(net_returns, dtype=float)
    w = np.asarray(w, dtype=float)
    T, d = net_returns.shape
    one_plus = 1.0 + net_returns @ w  # shape (T,)

    if np.any(one_plus <= 0):
        # Fallback to direct product derivative (less stable)
        S = np.ones(T, dtype=float)
        dS = np.zeros((T, d), dtype=float)
        prod = 1.0
        for t in range(T):
            prod *= one_plus[t]
            S[t] = prod
            deriv = np.zeros(d, dtype=float)
            for s in range(t + 1):
                excl = 1.0
                for j in range(t + 1):
                    if j == s:
                        continue
                    excl *= one_plus[j]
                deriv += excl * net_returns[s]
            dS[t] = deriv
        return S, dS

    # Use stable log-derivative formula
    log_terms = np.log(one_plus)
    S = np.exp(np.cumsum(log_terms))
    inv_one_plus = 1.0 / one_plus
    accum = np.zeros((d,), dtype=float)
    dS = np.zeros((T, d), dtype=float)
    for t in range(T):
        accum += net_returns[t] * inv_one_plus[t]
        dS[t, :] = S[t] * accum
    return S, dS


def _drawdowns_and_derivatives(
    net_returns: np.ndarray, w: np.ndarray, compounded: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Compute D_t and ∂D_t/∂w for drawdowns.

    Parameters
    ----------
    net_returns : ndarray of shape (T, d)
        Asset net returns.
    w : ndarray of shape (d,)
        Portfolio weights.
    compounded : bool, default=False
        If True, use wealth-based drawdowns (multiplicative).
        If False, use cumsum-based drawdowns (additive) - default matches skfolio.measures.

    Returns
    -------
    D : ndarray of shape (T,)
        Drawdown series.
    dD : ndarray of shape (T, d)
        Derivatives of drawdowns w.r.t. weights.
    """
    T, d = net_returns.shape

    if compounded:
        # Wealth-based (original implementation)
        S, dS = _wealth_and_dwealth(net_returns, w)
        running_max = np.empty(T, dtype=float)
        argmax_idx = np.empty(T, dtype=int)
        current_max = -np.inf
        current_arg = -1

        for t in range(T):
            if S[t] > current_max:
                current_max = S[t]
                current_arg = t
            running_max[t] = current_max
            argmax_idx[t] = current_arg

        D = running_max - S
        dD = np.zeros_like(dS)
        for t in range(T):
            u = argmax_idx[t]
            dD[t, :] = dS[u, :] - dS[t, :]
        return D, dD
    else:
        # Cumsum-based (matches skfolio.measures default)
        # C_t(w) = sum_{s<=t} r_s^T w
        # dC_t/dw = sum_{s<=t} r_s
        portfolio_returns = net_returns @ w  # (T,)
        cum_returns = np.cumsum(portfolio_returns)  # (T,)

        # Find running maximum and its index
        running_max = np.empty(T, dtype=float)
        argmax_idx = np.empty(T, dtype=int)
        current_max = -np.inf
        current_arg = -1

        for t in range(T):
            if cum_returns[t] > current_max:
                current_max = cum_returns[t]
                current_arg = t
            running_max[t] = current_max
            argmax_idx[t] = current_arg

        # D_t = max_{s<=t} C_s - C_t
        D = running_max - cum_returns

        # dD_t/dw = dC_peak/dw - dC_t/dw
        # where dC_s/dw = sum_{i<=s} r_i
        dC = np.cumsum(net_returns, axis=0)  # (T, d): cumsum of returns
        dD = np.zeros((T, d))
        for t in range(T):
            peak_idx = argmax_idx[t]
            dD[t, :] = dC[peak_idx, :] - dC[t, :]

        return D, dD


def _analytical_evar_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical EVaR gradient using envelope theorem.

    EVaR = min_theta (theta * log(E[exp(-r/theta)] / (1-beta)))
    ∇EVaR = -Σ r_i * p_i where p_i ∝ exp(theta* * L_i) and L_i = -r_i^T w

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Additional parameters (beta).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    beta = kwargs.get("beta", 0.95)
    R = np.asarray(net_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    _N, _d = R.shape

    # Losses L_i = -(r_i^T w)
    L = -(R @ w)

    # Find optimal theta
    theta_star = _find_theta_star_empirical(L, beta=beta)

    # Compute Gibbs weights p_i ∝ exp(theta* * L_i)
    a = theta_star * L
    a_max = np.max(a)
    exp_shift = np.exp(a - a_max)
    denom = exp_shift.sum()

    if denom == 0:
        return np.zeros_like(w)

    p = exp_shift / denom
    # Gradient: -Σ r_i * p_i
    grad = -(R.T @ p)
    return grad


def _analytical_cdar_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical CDaR gradient (CVaR on drawdowns with fractional handling).

    Uses cumsum-based drawdowns (compounded=False) to match skfolio.measures default.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Additional parameters (beta, compounded).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    beta = kwargs.get("beta", 0.95)
    compounded = kwargs.get("compounded", False)
    R = np.asarray(net_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    T, d = R.shape

    if T == 0:
        return np.zeros_like(w)

    # Compute drawdowns and their derivatives (default: cumsum-based)
    D, dD = _drawdowns_and_derivatives(R, w, compounded=compounded)

    # Sort by drawdown (descending, largest first)
    order = np.argsort(D)[::-1]
    dD_sorted = dD[order]

    m = (1.0 - beta) * T
    if m <= 0:
        return np.zeros_like(w)

    k = int(np.floor(m))
    rem = m - k

    # Accumulate gradient: sum first k fully, then fraction of (k+1)-th
    grad = np.zeros(d, dtype=float)
    if k > 0:
        grad += dD_sorted[:k].sum(axis=0)
    if rem > 0 and k < T:
        grad += rem * dD_sorted[k]

    # Normalize
    grad = grad / ((1.0 - beta) * T)
    return grad


def _analytical_edar_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical EDaR gradient using envelope theorem on drawdowns.

    Uses cumsum-based drawdowns (compounded=False) to match skfolio.measures default.

    EDaR = min_theta (theta * log(E[exp(D_t/theta)] / (1-beta)))
    ∇EDaR = Σ ∂D_t/∂w * p_t where p_t ∝ exp(theta* * D_t)

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns.
    **kwargs
        Additional parameters (beta, compounded).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector.
    """
    beta = kwargs.get("beta", 0.95)
    compounded = kwargs.get("compounded", False)
    R = np.asarray(net_returns, dtype=float)
    w = np.asarray(weights, dtype=float)
    _T, _d = R.shape

    # Compute drawdowns and derivatives (default: cumsum-based)
    D, dD = _drawdowns_and_derivatives(R, w, compounded=compounded)

    # Find optimal theta
    theta_star = _find_theta_star_empirical(D, beta=beta)

    # Compute weights p_t ∝ exp(theta* * D_t)
    a = theta_star * D
    a_max = np.max(a)
    exp_shift = np.exp(a - a_max)
    denom = exp_shift.sum()

    if denom == 0:
        return np.zeros_like(w)

    weights_t = exp_shift / denom
    # Gradient: Σ dD_t * p_t
    grad = dD.T @ weights_t
    return grad


def _analytical_cvar_gradient(
    weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Analytical gradient of CVaR (Conditional Value at Risk).

    CVaR gradient with fractional tail handling matching skfolio.measures.cvar.

    Formula: CVaR = -sum(ret[:ik]) / k + ret[ik] * (ik / k - 1)
    where k = (1-β)*T and ik = ceil(k) - 1

    Gradient accounts for fractional weighting of the marginal observation.

    Parameters
    ----------
    weights : ndarray of shape (n_assets,)
        Portfolio weights.
    net_returns : ndarray of shape (n_observations, n_assets)
        Historical net returns. For single observations, automatically reshaped
        from (n_assets,) to (1, n_assets) by MeasureObjective.grad().
    **kwargs
        Additional parameters (beta).

    Returns
    -------
    ndarray of shape (n_assets,)
        Gradient vector (subgradient).

    Notes
    -----
    This implements the exact gradient of the empirical CVaR formula used in
    skfolio.measures.cvar, including proper handling of the fractional part.
    """
    beta = kwargs.get("beta", 0.95)
    T = net_returns.shape[0]
    k = (1 - beta) * T
    if k <= 0:
        return np.zeros(net_returns.shape[1])

    ik = max(0, int(np.ceil(k) - 1))

    portfolio_returns = net_returns @ weights
    # Sort portfolio returns (ascending, worst first)
    sort_idx = np.argsort(portfolio_returns)
    sorted_returns = net_returns[sort_idx]

    # Gradient components:
    # d/dw [-sum(ret[:ik]) / k] = -sum(r[:ik]) / k
    # d/dw [ret[ik] * (ik/k - 1)] = r[ik] * (ik/k - 1)

    grad = np.zeros(net_returns.shape[1])
    if ik > 0:
        grad -= sorted_returns[:ik].sum(axis=0) / k
    # Add contribution from marginal observation
    if ik < T:
        grad += sorted_returns[ik] * (ik / k - 1)

    return grad


# Measure properties including convexity info and measure functions from skfolio.measures
MEASURE_PROPERTIES = {
    # Special measure for log-wealth (default objective)
    PerfMeasure.LOG_WEALTH: {
        "measure_func": log_wealth,
        "convexity": "exp-concave",
        "sign": -1,  # maximize wealth = minimize negative log
        "notes": "1-exp-concave, O(log T) regret. Analytical gradient available.",
    },
    # Performance measures (maximize => sign = -1)
    PerfMeasure.MEAN: {
        "measure_func": mean,
        "convexity": "linear",
        "sign": -1,  # maximize mean = minimize negative mean
        "notes": "Linear, O(sqrt(T)) regret. Analytical gradient available.",
    },
    # Risk measures (minimize)
    RiskMeasure.VARIANCE: {
        "measure_func": variance,
        "convexity": "strongly_convex",
        "sign": 1,  # minimize
        "notes": "Strongly convex, quadratic. Analytical gradient available (requires Σ).",
    },
    RiskMeasure.STANDARD_DEVIATION: {
        "measure_func": standard_deviation,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, smooth. Uses autograd (square root complicates analytical form).",
    },
    RiskMeasure.SEMI_VARIANCE: {
        "measure_func": semi_variance,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Analytical gradient available.",
    },
    RiskMeasure.SEMI_DEVIATION: {
        "measure_func": semi_deviation,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Uses autograd.",
    },
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION: {
        "measure_func": mean_absolute_deviation,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, L1 norm, non-smooth. Analytical gradient available.",
    },
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT: {
        "measure_func": first_lower_partial_moment,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Analytical gradient available.",
    },
    RiskMeasure.CVAR: {
        "measure_func": cvar,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth (sorting). Analytical gradient available.",
    },
    RiskMeasure.EVAR: {
        "measure_func": evar,
        "convexity": "exp-concave",
        "sign": 1,
        "notes": "Convex, exp-concave, smooth. Analytical gradient available (envelope theorem with theta optimization).",
    },
    RiskMeasure.WORST_REALIZATION: {
        "measure_func": worst_realization,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth (max operator). Analytical gradient available (one-hot).",
    },
    RiskMeasure.GINI_MEAN_DIFFERENCE: {
        "measure_func": gini_mean_difference,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Analytical gradient available (OWA).",
    },
    RiskMeasure.CDAR: {
        "measure_func": cdar,
        "convexity": "convex",
        "sign": 1,
        "notes": "Convex, non-smooth. Analytical gradient available (CVaR on drawdowns with path derivatives).",
    },
    RiskMeasure.EDAR: {
        "measure_func": edar,
        "convexity": "exp-concave",
        "sign": 1,
        "notes": "Convex, exp-concave, smooth. Analytical gradient available (envelope theorem on drawdowns).",
    },
}


# Individual measure functions for clarity and testability
def _autograd_log_wealth(portfolio_returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Compute log-wealth."""
    return anp.sum(anp.log(1.0 + portfolio_returns))


def _autograd_mean(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible mean matching skfolio.measures.mean.

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D (reduce along axis=0)

    Reference
    - skfolio.measures.mean
    """
    return anp.mean(returns) if returns.ndim == 1 else anp.mean(returns, axis=0)


def _autograd_variance(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible variance matching skfolio.measures.variance.

    - Unbiased (ddof=1) for T>1; axis=0 for matrix inputs
    - T==1 proxy: returns[0]**2 (per-asset if 2D)

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.variance
    """
    T = returns.shape[0]
    if returns.ndim == 1:
        if T == 1:
            return returns[0] ** 2
        m = anp.mean(returns)
        return anp.sum((returns - m) ** 2) / (T - 1)
    else:
        if T == 1:
            return returns[0] ** 2
        m = anp.mean(returns, axis=0)
        return anp.sum((returns - m) ** 2, axis=0) / (T - 1)


def _autograd_standard_deviation(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible standard deviation matching skfolio.measures.standard_deviation.

    - Unbiased variance under sqrt; axis=0 for matrix inputs
    - T==1 proxy: |returns[0]| (per-asset if 2D)

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.standard_deviation
    """
    T = returns.shape[0]
    if returns.ndim == 1:
        if T == 1:
            return anp.abs(returns[0])
        m = anp.mean(returns)
        var = anp.sum((returns - m) ** 2) / (T - 1)
        return anp.sqrt(var + CLIP_EPSILON)
    else:
        if T == 1:
            return anp.abs(returns[0])
        m = anp.mean(returns, axis=0)
        var = anp.sum((returns - m) ** 2, axis=0) / (T - 1)
        return anp.sqrt(var + CLIP_EPSILON)


def _autograd_semi_variance(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible semi-variance matching skfolio.measures.semi_variance.

    - Unbiased factor T/(T-1); MAR defaults per-asset for 2D; axis=0

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.semi_variance
    """
    T = returns.shape[0]
    mar = kwargs.get("min_acceptable_return", None)
    if returns.ndim == 1:
        if mar is None:
            mar = anp.mean(returns)
        downside = anp.maximum(0, mar - returns)
        biased_sv = anp.mean(downside**2)
        return biased_sv * T / (T - 1) if T > 1 else biased_sv
    else:
        if mar is None:
            mar = anp.mean(returns, axis=0)
        downside = anp.maximum(0, mar - returns)
        biased_sv = anp.mean(downside**2, axis=0)
        return biased_sv * T / (T - 1) if T > 1 else biased_sv


def _autograd_semi_deviation(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible semi-deviation (sqrt of semi-variance).

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.semi_deviation
    """
    semi_var = _autograd_semi_variance(returns, **kwargs)
    return anp.sqrt(semi_var)


def _autograd_mean_absolute_deviation(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible MAD matching skfolio.measures.mean_absolute_deviation.

    - MAR defaults per-asset for 2D; axis=0 reduction

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.mean_absolute_deviation
    """
    T = returns.shape[0]
    if returns.ndim == 1:
        if T == 1:
            mar = kwargs.get("min_acceptable_return", 0.0)
            return anp.abs(returns[0] - mar)
        m = anp.mean(returns)
        return anp.mean(anp.abs(returns - m))
    else:
        if T == 1:
            mar = kwargs.get("min_acceptable_return", anp.zeros(returns.shape[1]))
            return anp.abs(returns[0] - mar)
        m = anp.mean(returns, axis=0)
        return anp.mean(anp.abs(returns - m), axis=0)


def _autograd_first_lower_partial_moment(
    returns: np.ndarray, **kwargs: Any
) -> np.ndarray:
    """Autograd-compatible FLPM matching skfolio.measures.first_lower_partial_moment.

    - MAR defaults per-asset for 2D; axis=0 reduction

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.first_lower_partial_moment
    """
    mar = kwargs.get("min_acceptable_return", None)
    if returns.ndim == 1:
        if mar is None:
            mar = anp.mean(returns)
        downside = anp.maximum(0, mar - returns)
        return anp.mean(downside)
    else:
        if mar is None:
            mar = anp.mean(returns, axis=0)
        downside = anp.maximum(0, mar - returns)
        return anp.mean(downside, axis=0)


def _autograd_cvar(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible CVaR matching skfolio.measures.cvar.

    - Empirical CVaR with fractional handling; axis=0 for matrix inputs

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.cvar
    """
    beta = kwargs.get("beta", 0.95)
    T = returns.shape[0]
    k = (1.0 - beta) * T
    if k <= 0:
        k = 1.0
    ik = anp.maximum(0, anp.ceil(k) - 1).astype(int)
    if returns.ndim == 1:
        sorted_ret = anp.sort(returns)
        return -anp.sum(sorted_ret[:ik]) / k + sorted_ret[ik] * (ik / k - 1.0)
    else:
        sorted_ret = anp.sort(returns, axis=0)
        head_sum = (
            anp.sum(sorted_ret[:ik, :], axis=0)
            if ik > 0
            else anp.zeros(sorted_ret.shape[1])
        )
        return -head_sum / k + sorted_ret[ik, :] * (ik / k - 1.0)


def _autograd_evar(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-friendly EVaR via smooth-min over a theta grid (axis-aware).

    - Uses a log-sum-exp softmin over a logarithmic grid of theta values to
      approximate min_theta phi(theta), where
      phi(theta) = theta * (log(mean(exp(-returns/theta))) - log(1-beta)).
    - Fully differentiable through autograd (anp operations only).

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.evar
    """
    returns = anp.asarray(returns)
    beta = kwargs.get("beta", 0.95)

    # Construct theta grid based on scale of losses
    max_loss = anp.max(-returns)
    max_loss = anp.where(max_loss <= 0, 1.0, max_loss)
    theta_min = max_loss / 200.0
    theta_max = max_loss / 2.0
    thetas = anp.exp(anp.linspace(anp.log(theta_min), anp.log(theta_max), 256))

    # Compute phi(theta) for each theta
    if returns.ndim == 1:
        phi_vals = []
        for theta in thetas:
            exp_term = anp.exp(-returns / theta)
            mean_exp = anp.mean(exp_term)
            phi_vals.append(theta * (anp.log(mean_exp) - anp.log(1.0 - beta)))
        phi = anp.stack(phi_vals, axis=0)
        return anp.min(phi)
    else:
        phi_cols = []
        for theta in thetas:
            exp_term = anp.exp(-returns / theta)  # (T, n)
            mean_exp = anp.mean(exp_term, axis=0)  # (n,)
            phi_cols.append(theta * (anp.log(mean_exp) - anp.log(1.0 - beta)))
        phi = anp.stack(phi_cols, axis=0)  # (N, n)
        return anp.min(phi, axis=0)


def _autograd_worst_realization(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible worst realization matching skfolio.measures.worst_realization.

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.worst_realization
    """
    return -anp.min(returns) if returns.ndim == 1 else -anp.min(returns, axis=0)


def _autograd_gini_mean_difference(returns: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Autograd-compatible GMD matching skfolio.measures.gini_mean_difference.

    - Uses OWA weights; sorts along time (axis=0); axis=0 reduction for 2D

    Shapes
    - Input: (T,) or (T, n)
    - Output: scalar if 1D, (n,) if 2D

    Reference
    - skfolio.measures.gini_mean_difference
    """
    T = returns.shape[0]
    if returns.ndim == 1:
        if T == 1:
            return anp.abs(returns[0])
        w = owa_gmd_weights(T)
        return anp.dot(w, anp.sort(returns))
    else:
        if T == 1:
            return anp.abs(returns[0])
        w = owa_gmd_weights(T)
        sorted_rets = anp.sort(returns, axis=0)
        return anp.dot(w, sorted_rets)


def smooth_max(
    a: np.ndarray | float, b: np.ndarray | float, beta: float = 100.0
) -> np.ndarray | float:
    """Smooth approximation of max(a,b). Higher beta = sharper approximation."""
    return (1.0 / beta) * anp.log(anp.exp(beta * a) + anp.exp(beta * b))


def smooth_cummax(x: np.ndarray, beta: float = 100.0) -> np.ndarray:
    """Differentiable smooth running maximum."""
    out = anp.copy(x)
    for i in range(1, len(x)):
        out[i] = smooth_max(out[i - 1], x[i], beta)
    return out


def _autograd_get_drawdowns(
    returns: np.ndarray, compounded: bool = False, beta: float | None = None
) -> np.ndarray:
    """Compute drawdowns with optional smooth approximation.

    Parameters
    ----------
    returns : array-like
        Portfolio returns.
    compounded : bool, default=False
        Whether returns are compounded.
    beta : float or None, default=None
        Smoothing parameter for differentiable cummax. If None, uses exact
        cummax via np.maximum.accumulate (not differentiable).

    Returns
    -------
    drawdowns : array-like
        Drawdown series.
    """
    cumulative_returns = get_cumulative_returns(returns, compounded=compounded)
    if beta is None:
        # Use regular numpy for exact cummax (not differentiable)
        running_max = np.maximum.accumulate(cumulative_returns)
    else:
        # Use smooth approximation (differentiable)
        running_max = smooth_cummax(cumulative_returns, beta=beta)
    if compounded:
        drawdowns = cumulative_returns / running_max - 1
    else:
        drawdowns = cumulative_returns - running_max
    return drawdowns


def _autograd_cdar(
    returns: np.ndarray,
    beta: float | None = 0.95,
    compounded: bool = False,
    smooth_beta: float | None = None,
) -> np.ndarray:
    """Compute CDaR (Conditional Drawdown at Risk).

    Parameters
    ----------
    returns : array-like
        Portfolio returns.
    beta : float, default=0.95
        Confidence level for CVaR on drawdowns.
    compounded : bool, default=False
        Whether returns are compounded.
    smooth_beta : float or None, default=None
        Smoothing parameter for differentiable cummax. If None, uses exact
        cummax (not differentiable but numerically exact).

    Notes
    -----
    For numerical accuracy in testing, use smooth_beta=None. For gradient
    computation in OCO, use smooth_beta=1000 or higher for good approximation.
    """
    drawdowns = _autograd_get_drawdowns(
        returns, compounded=compounded, beta=smooth_beta
    )
    return _autograd_cvar(drawdowns, beta=beta)


def _autograd_edar(
    returns: np.ndarray,
    beta: float | None = 0.95,
    compounded: bool = False,
    smooth_beta: float | None = None,
) -> np.ndarray:
    """Compute EDaR (Entropic Drawdown at Risk) - approximated via EVaR on drawdowns.

    Parameters
    ----------
    returns : array-like
        Portfolio returns.
    beta : float, default=0.95
        Confidence level for EVaR on drawdowns.
    compounded : bool, default=False
        Whether returns are compounded.
    smooth_beta : float or None, default=None
        Smoothing parameter for differentiable cummax. If None, uses exact
        cummax (not differentiable but numerically exact).

    Notes
    -----
    This approximates EDaR using CVaR on drawdowns (since true EVaR requires
    optimization). For numerical accuracy in testing, use smooth_beta=None.
    For gradient computation in OCO, use smooth_beta=1000 or higher.
    """
    drawdowns = _autograd_get_drawdowns(
        returns, compounded=compounded, beta=smooth_beta
    )
    return _autograd_evar(drawdowns, beta=beta)


def _get_autograd_measure_fn(measure: BaseMeasure | None, **kwargs: Any) -> Callable:
    """Select the differentiable implementation of the requested measure."""
    match measure:
        case PerfMeasure.LOG_WEALTH:
            return partial[Any](_autograd_log_wealth, **kwargs)
        case PerfMeasure.MEAN:
            return partial[Any](_autograd_mean, **kwargs)
        case RiskMeasure.VARIANCE:
            return partial[Any](_autograd_variance, **kwargs)
        case RiskMeasure.STANDARD_DEVIATION:
            return partial[Any](_autograd_standard_deviation, **kwargs)
        case RiskMeasure.SEMI_VARIANCE:
            return partial[Any](_autograd_semi_variance, **kwargs)
        case RiskMeasure.SEMI_DEVIATION:
            return partial[Any](_autograd_semi_deviation, **kwargs)
        case RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
            return partial[Any](_autograd_mean_absolute_deviation, **kwargs)
        case RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT:
            return partial[Any](_autograd_first_lower_partial_moment, **kwargs)
        case RiskMeasure.CVAR:
            return partial[Any](_autograd_cvar, **kwargs)
        case RiskMeasure.EVAR:
            return partial[Any](_autograd_evar, **kwargs)
        case RiskMeasure.WORST_REALIZATION:
            return partial[Any](_autograd_worst_realization, **kwargs)
        case RiskMeasure.GINI_MEAN_DIFFERENCE:
            return partial[Any](_autograd_gini_mean_difference, **kwargs)
        case RiskMeasure.CDAR:
            return partial[Any](_autograd_cdar, **kwargs)
        case RiskMeasure.EDAR:
            return partial[Any](_autograd_edar, **kwargs)
        case _:
            raise ValueError(f"Unsupported measure: {measure}")


@dataclass
class MeasureObjective(BaseObjective):
    """Unified objective for all measures (risk, performance, and log-wealth).

    Parameters
    ----------
    measure : RiskMeasure | PerfMeasure | None
        The measure to optimize. If None, uses log-wealth objective.
    use_autograd : bool, default=False
        Whether to use autograd for gradient computation. If False, uses analytical
        gradients when available (faster but limited to certain measures).
    risk_free_rate : float, default=0.0
        Risk-free rate (used for some measures).
    min_acceptable_return : float, default=0.0
        Minimum acceptable return threshold (used for semi-variance, FLPM, etc.).
    cvar_beta : float, default=0.95
        Confidence level for CVaR.
    evar_beta : float, default=0.95
        Confidence level for EVaR.
    cdar_beta : float, default=0.95
        Confidence level for CDaR.
    edar_beta : float, default=0.95
        Confidence level for EDaR.
    """

    measure: BaseMeasure | None = None
    use_autograd: bool = False
    risk_free_rate: float = 0.0
    min_acceptable_return: float = 0.0
    cvar_beta: float = 0.95
    evar_beta: float = 0.95
    cdar_beta: float = 0.95
    edar_beta: float = 0.95

    def __post_init__(self) -> None:
        # Default to log-wealth if measure is None
        if self.measure is None:
            self.measure = PerfMeasure.LOG_WEALTH

        # Get measure properties
        if self.measure not in MEASURE_PROPERTIES:
            raise ValueError(f"Unknown measure: {self.measure}")

        self.properties = MEASURE_PROPERTIES[self.measure]
        self._sign = cast(float, self.properties["sign"])

        # Create gradient function
        if self.use_autograd:
            self._grad_fn = grad(self._loss_fn, argnum=0)
        else:
            # Use analytical gradient
            self._grad_fn = self._get_analytical_grad_fn()
            if self._grad_fn is None:
                raise NotImplementedError(
                    f"Analytical gradient not implemented for {self.measure}. "
                    "Set `use_autograd=True` or choose a different measure."
                )

    def _ensure_2d(self, net_returns: np.ndarray) -> np.ndarray:
        """Convert 1D input to 2D (1, n_assets) for unified handling.

        Parameters
        ----------
        net_returns : ndarray
            Net returns array, either 1D (n_assets,) or 2D (n_observations, n_assets).

        Returns
        -------
        ndarray
            2D array of shape (n_observations, n_assets).
        """
        if net_returns.ndim == 1:
            return net_returns.reshape(1, -1)
        return net_returns

    def _loss_fn(
        self, weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Autograd-compatible loss computation with unified 2D handling.

        Parameters
        ----------
        weights : array-like
            Portfolio weights.
        net_returns : array-like
            Net returns of assets (not price relatives).
            - 1D array of shape (n_assets,) for single observation
            - 2D array of shape (n_observations, n_assets) for time series
        **kwargs
            Additional parameters (e.g., beta for CVaR, min_acceptable_return, etc.).
        """
        # For all other measures: ensure 2D, then compute unified
        net_returns_2d = self._ensure_2d(net_returns)
        returns = anp.dot(net_returns_2d, weights)  # Shape: (T,)
        return self._sign * _get_autograd_measure_fn(self.measure)(returns, **kwargs)

    def _autograd_measure(self, returns: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Dispatch to individual measure functions.

        Parameters
        ----------
        portfolio_returns : array-like of shape (T,)
            Portfolio returns over T observations.
        **kwargs
            Additional parameters (min_acceptable_return, beta, etc.).

        Returns
        -------
        float
            Measure value.

        Notes
        -----
        For T=1, some measures use simplified proxies since statistical measures
        like variance are undefined for a single sample. Each measure has its own
        dedicated function for clarity and testability.
        """
        return _get_autograd_measure_fn(self.measure)(returns, **kwargs)

    def _get_analytical_grad_fn(self) -> Callable:
        """Get analytical gradient function for the measure.

        Returns
        -------
        callable or None
            Analytical gradient function if available, None otherwise.
        """
        # Get the base gradient function and wrap it with sign multiplication
        match self.measure:
            case PerfMeasure.LOG_WEALTH:
                # Log-wealth gradient already has the sign baked in (negative)
                return _analytical_log_wealth_gradient

            case PerfMeasure.MEAN:
                base_grad_fn = _analytical_mean_gradient

            case RiskMeasure.VARIANCE:
                base_grad_fn = _analytical_variance_gradient

            case RiskMeasure.STANDARD_DEVIATION:
                base_grad_fn = _analytical_standard_deviation_gradient

            case RiskMeasure.SEMI_VARIANCE:
                base_grad_fn = _analytical_semi_variance_gradient

            case RiskMeasure.SEMI_DEVIATION:
                base_grad_fn = _analytical_semi_deviation_gradient

            case RiskMeasure.MEAN_ABSOLUTE_DEVIATION:
                base_grad_fn = _analytical_mean_absolute_deviation_gradient

            case RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT:
                base_grad_fn = _analytical_first_lower_partial_moment_gradient

            case RiskMeasure.WORST_REALIZATION:
                base_grad_fn = _analytical_worst_realization_gradient

            case RiskMeasure.GINI_MEAN_DIFFERENCE:
                base_grad_fn = _analytical_gini_mean_difference_gradient

            case RiskMeasure.CVAR:
                base_grad_fn = _analytical_cvar_gradient

            case RiskMeasure.EVAR:
                base_grad_fn = _analytical_evar_gradient

            case RiskMeasure.CDAR:
                base_grad_fn = _analytical_cdar_gradient

            case RiskMeasure.EDAR:
                base_grad_fn = _analytical_edar_gradient

            case _:
                # No analytical gradient available for this measure
                raise ValueError(f"No analytical gradient available for {self.measure}")

        # Wrap with sign multiplication to match loss function
        def signed_grad_fn(
            weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
        ) -> np.ndarray:
            """Apply the objective sign to its analytical portfolio gradient."""
            return float(self._sign) * base_grad_fn(weights, net_returns, **kwargs)

        return signed_grad_fn

    def loss(
        self, weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
    ) -> float:
        """Compute loss value with unified 2D handling.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        net_returns : ndarray of shape (n_assets,) or (n_observations, n_assets)
            Net returns of assets. Automatically converted to 2D for consistency.
        **kwargs
            Additional measure-specific parameters (e.g., beta=0.95 for CVaR).

        Returns
        -------
        float
            Loss value.
        """
        return float(self._loss_fn(weights, net_returns, **kwargs))

    def grad(
        self, weights: np.ndarray, net_returns: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Compute gradient with unified 2D handling.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        net_returns : ndarray of shape (n_assets,) or (n_observations, n_assets)
            Net returns of assets. For analytical gradients of non-LOG_WEALTH measures,
            automatically converted to 2D for consistent handling.
        **kwargs
            Additional measure-specific parameters (e.g., beta=0.95 for CVaR).

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        # For analytical gradients (except LOG_WEALTH): ensure 2D for consistency
        # For autograd: handles both 1D and 2D internally via _loss_fn
        net_returns_input = net_returns
        if not self.use_autograd and self.measure != PerfMeasure.LOG_WEALTH:
            net_returns_input = self._ensure_2d(net_returns)

        return self._grad_fn(weights, net_returns_input, **kwargs)


@dataclass
class MeanReversionHinge(BaseObjective):
    """Hinge loss for mean reversion: L(w) = max(0, ε - φ^T w).

    This loss encourages the portfolio to satisfy φ^T w ≥ ε, where φ is a
    reversion predictor (e.g., inverse moving average for OLMAR or current
    price relatives for PAMR).

    Parameters
    ----------
    epsilon : float
        Margin threshold. The loss is zero when φ^T w ≥ ε.
    use_autograd : bool, default=False
        If True, verify analytical gradient against autograd.

    Notes
    -----
    Gradient: ∇_w L(w) = -φ if φ^T w < ε, else 0 (subgradient at equality).
    The loss is convex and piecewise linear.
    """

    epsilon: float
    use_autograd: bool = False

    def loss(self, weights: np.ndarray, phi: np.ndarray, **kwargs: Any) -> float:
        """Compute hinge loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor (fixed during gradient computation).
        **kwargs
            Unused, included for signature consistency.

        Returns
        -------
        float
            Loss value.
        """
        margin = self.epsilon - np.dot(weights, phi)
        return float(max(0.0, margin))

    def grad(self, weights: np.ndarray, phi: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute hinge loss gradient w.r.t. weights.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor (fixed).
        **kwargs
            Unused, included for signature consistency.

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        margin = self.epsilon - np.dot(weights, phi)
        if margin > 0.0:
            return -phi
        else:
            return np.zeros_like(phi)


@dataclass
class MeanReversionSquaredHinge(BaseObjective):
    """Squared hinge loss: L(w) = [max(0, ε - φ^T w)]².

    Smooth variant of hinge loss with Lipschitz continuous gradient.

    Parameters
    ----------
    epsilon : float
        Margin threshold.
    use_autograd : bool, default=False
        If True, verify analytical gradient against autograd.

    Notes
    -----
    Gradient: ∇_w L(w) = -2(ε - φ^T w)φ if φ^T w < ε, else 0.
    The loss is convex and differentiable everywhere (smooth).
    """

    epsilon: float
    use_autograd: bool = False

    def loss(self, weights: np.ndarray, phi: np.ndarray, **kwargs: Any) -> float:
        """Compute squared hinge loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor.
        **kwargs
            Unused.

        Returns
        -------
        float
            Loss value.
        """
        margin = self.epsilon - np.dot(weights, phi)
        return float(max(0.0, margin) ** 2)

    def grad(self, weights: np.ndarray, phi: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute squared hinge gradient w.r.t. weights.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor.
        **kwargs
            Unused.

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        margin = self.epsilon - np.dot(weights, phi)
        if margin > 0.0:
            return -2.0 * margin * phi
        else:
            return np.zeros_like(phi)


@dataclass
class MeanReversionSoftplus(BaseObjective):
    """Softplus loss: L(w) = (1/β) log(1 + exp(β(ε - φ^T w))).

    Smooth approximation to hinge loss. As β → ∞, approaches hinge loss.

    Parameters
    ----------
    epsilon : float
        Margin threshold.
    beta : float, default=5.0
        Temperature parameter. Larger values → sharper hinge approximation.
    use_autograd : bool, default=False
        If True, verify analytical gradient against autograd.

    Notes
    -----
    Gradient: ∇_w L(w) = -sigmoid(β(ε - φ^T w))φ where sigmoid is the sigmoid function.
    The loss is convex, infinitely differentiable, and exp-concave for bounded β.
    Admits O(log T) regret in OCO with appropriate algorithms.
    """

    epsilon: float
    beta: float = 5.0
    use_autograd: bool = False

    def loss(self, weights: np.ndarray, phi: np.ndarray, **kwargs: Any) -> float:
        """Compute softplus loss value.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor.
        **kwargs
            Unused.

        Returns
        -------
        float
            Loss value.
        """
        margin = self.epsilon - np.dot(weights, phi)
        # Use log1p for numerical stability when beta*margin is small
        return float(np.log(1.0 + np.exp(self.beta * margin)) / self.beta)

    def grad(self, weights: np.ndarray, phi: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute softplus gradient w.r.t. weights.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
            Portfolio weights.
        phi : ndarray of shape (n_assets,)
            Reversion predictor.
        **kwargs
            Unused.

        Returns
        -------
        ndarray of shape (n_assets,)
            Gradient vector.
        """
        margin = self.epsilon - np.dot(weights, phi)
        # Sigmoid: sigmoid(z) = 1 / (1 + exp(-z))
        sigmoid = 1.0 / (1.0 + np.exp(-self.beta * margin))
        return -sigmoid * phi


def create_mean_reversion_objective(
    objective_type: str,
    epsilon: float,
    beta: float = 5.0,
    use_autograd: bool = False,
) -> BaseObjective:
    """Factory for mean-reversion objectives.

    Creates objective functions for Follow-the-Loser (mean-reversion) strategies
    like OLMAR, PAMR, and CWMR. These objectives measure how well portfolio weights
    satisfy a mean-reversion constraint.

    Parameters
    ----------
    objective_type : str
        One of "hinge", "squared_hinge", "softplus".

        - "hinge": Piecewise linear, non-smooth at kink
        - "squared_hinge": Smooth, Lipschitz gradient
        - "softplus": Infinitely smooth, exp-concave

    epsilon : float
        Margin threshold. Loss is zero when φ^T w ≥ ε.

    beta : float, default=5.0
        Temperature parameter for softplus. Only used when objective_type="softplus".
        Larger values approximate hinge more closely.

    use_autograd : bool, default=False
        If True, verify analytical gradient against autograd (for testing).

    Returns
    -------
    BaseObjective
        Configured mean-reversion objective with analytical gradients.

    Examples
    --------
    >>> # Hinge loss for PAMR
    >>> obj = create_mean_reversion_objective("hinge", epsilon=1.0)
    >>> weights = np.array([0.5, 0.5])
    >>> phi = np.array([1.1, 0.9])  # Price relatives
    >>> loss_val = obj.loss(weights, phi)
    >>> gradient = obj.grad(weights, phi)

    >>> # Smooth softplus for OLMAR
    >>> obj = create_mean_reversion_objective("softplus", epsilon=2.0, beta=10.0)
    """
    match objective_type.lower():
        case "hinge":
            return MeanReversionHinge(epsilon=epsilon, use_autograd=use_autograd)
        case "squared_hinge":
            return MeanReversionSquaredHinge(epsilon=epsilon, use_autograd=use_autograd)
        case "softplus":
            return MeanReversionSoftplus(
                epsilon=epsilon, beta=beta, use_autograd=use_autograd
            )
        case _:
            raise ValueError(
                f"Unknown mean-reversion objective: {objective_type}. "
                f"Must be one of: 'hinge', 'squared_hinge', 'softplus'"
            )


def create_objective(
    measure: BaseMeasure | None,
    use_autograd: bool = False,
    **kwargs: Any,
) -> BaseObjective:
    """Factory function to create objective from measure enum.

    Parameters
    ----------
    measure : BaseMeasure or None
        Measure enum (RiskMeasure, ExtraRiskMeasure, or PerfMeasure).
        If None, uses log-wealth (default).
    use_autograd : bool, default=False
        Whether to use autograd for gradients.
    **kwargs
        Additional parameters passed to MeasureObjective.

    Returns
    -------
    BaseObjective
        Configured objective function.

    Examples
    --------
    >>> # Default log-wealth
    >>> obj = create_objective(None)

    >>> # CVaR with analytical gradient (not available, will fall back to autograd)
    >>> obj = create_objective(RiskMeasure.CVAR, use_autograd=False, cvar_beta=0.95)

    >>> # Variance with autograd
    >>> obj = create_objective(RiskMeasure.VARIANCE)
    """
    return MeasureObjective(measure=measure, use_autograd=use_autograd, **kwargs)
