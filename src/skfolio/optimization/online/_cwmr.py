"""Confidence-Weighted Mean Reversion (CWMR) module."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
from scipy.optimize import fsolve


def stable_y(a: float, s: float) -> float:
    """
    Numerically stable computation of y in CWMR update.

    Computes y = 0.5(-a + sqrt(a^2 + 4s)) = 2s / (sqrt(a^2 + 4s) + a).

    Parameters
    ----------
    a : float
        tau * phi * s
    s : float
        x^T Sigma_0 x

    Returns
    -------
    float
        The stable y value.
    """
    return (2.0 * s) / (np.sqrt(a * a + 4.0 * s) + a)


def constraint_residual(
    tau: float, m0: float, s: float, eps: float, phi: float
) -> float:
    """
    Residual function for CWMR constraint equation.

    The constraint is: m_0 - tau*s + phi*y(tau) = eps
    We want to find tau such that this equals zero.

    Parameters
    ----------
    tau : float
        Lagrange multiplier
    m0 : float
        mu_0^T x
    s : float
        x^T Sigma_0 x
    eps : float
        Epsilon threshold
    phi : float
        Quantile (Phi^{-1}(eta))

    Returns
    -------
    float
        Residual value
    """
    if s <= 1e-18:
        return 0.0
    a = tau * phi * s
    y = stable_y(a, s)
    return (m0 - tau * s + phi * y) - eps


def solve_tau(m0: float, s: float, eps: float, phi: float) -> float:
    """
    Solve for the Lagrange multiplier tau in CWMR update via scipy.optimize.fsolve.

    Parameters
    ----------
    m0 : float
        mu_0^T x
    s : float
        x^T Sigma_0 x >= 0
    eps : float
        Epsilon threshold
    phi : float
        Quantile >= 0

    Returns
    -------
    float
        Optimal tau >= 0
    """
    if s <= 1e-18:
        return 0.0

    # Check if constraint already satisfied
    f0 = constraint_residual(0.0, m0, s, eps, phi)
    if f0 <= 0.0:
        return 0.0

    # Initial guess: linearize the constraint
    tau_init = max((m0 - eps) / max(s, 1e-18), 1.0)

    # Solve using fsolve
    tau_solution = fsolve(
        lambda t: constraint_residual(t, m0, s, eps, phi),
        x0=tau_init,
        full_output=False,
    )
    tau = float(tau_solution[0])
    return max(tau, 0.0)  # Ensure non-negativity


def pa_distribution_update(
    mu: np.ndarray,
    diag: np.ndarray,
    x: np.ndarray,
    epsilon: float,
    quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    CWMR passive-aggressive distribution update (closed-form KKT solution).

    Parameters
    ----------
    mu : np.ndarray
        Current mean vector
    diag : np.ndarray
        Current diagonal covariance
    x : np.ndarray
        Price relatives
    epsilon : float
        Margin threshold
    quantile : float
        Phi^{-1}(eta) where eta is confidence level

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Updated (mu, diag)
    """
    phi = float(quantile)
    s = float(np.dot(diag * x, x))
    m0 = float(np.dot(mu, x))

    # Check if constraint satisfied
    if m0 + phi * np.sqrt(max(s, 0.0)) <= epsilon + 1e-18:
        return mu.copy(), diag.copy()

    # Solve for tau
    tau = solve_tau(m0, s, epsilon, phi)

    # Update mean: mu <- mu_0 - tau * Sigma_0 * x
    sigma_x = diag * x
    mu_new = mu - tau * sigma_x

    # Update variance: Sigma <- Sigma_0 - coeff * (Sigma_0 x)(Sigma_0 x)^T
    a = tau * phi * s
    y = stable_y(a, s)
    denom = s * (y + a)
    if denom <= 0.0:
        coeff = 0.0
    else:
        coeff = a / denom

    diag_new = diag - coeff * (sigma_x * sigma_x)
    return mu_new, diag_new


def clip_variances(
    diag: np.ndarray,
    min_var: float | None,
    max_var: float | None,
) -> np.ndarray:
    """
    Clip CWMR diagonal variances to configured bounds.

    Parameters
    ----------
    diag : np.ndarray
        Diagonal variances
    min_var : float or None
        Lower bound
    max_var : float or None
        Upper bound

    Returns
    -------
    np.ndarray
        Clipped diagonal variances
    """
    out = np.array(diag, dtype=float, copy=True)
    if min_var is not None:
        out = np.maximum(out, float(min_var))
    if max_var is not None:
        out = np.minimum(out, float(max_var))
    return np.maximum(out, 1e-18)  # Hard floor to prevent numerical issues
