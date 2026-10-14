"""Automatic learning rate estimation for online portfolio optimization.

This module provides theory-driven learning rate selection based on OCO optimal rates
for different Follow-The-Winner strategies. Learning rates are estimated using:
- Domain geometry (simplex diameter)
- Objective convexity class (exp-concave, strongly convex, linear, convex)
- Strategy-specific formulas from OCO literature

References
----------
- Hazan, E. (2016). Introduction to Online Convex Optimization.
- Orabona, F. (2019). A Modern Introduction to Online Learning.
"""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal

import numpy as np

from skfolio.measures._enums import BaseMeasure, PerfMeasure
from skfolio.optimization.online._autograd_objectives import MEASURE_PROPERTIES
from skfolio.optimization.online._mixins import FTWStrategy


def simplex_diameter(
    n_assets: int,
    min_weights: np.ndarray | float = 0,
    max_weights: np.ndarray | float = 1,
    budget: float = 1,
) -> float:
    """Estimate the diameter of a constrained simplex.

    The simplex is defined by:
    - Sum constraint: sum(x_i) = budget
    - Bound constraints: min_weights[i] <= x_i <= max_weights[i]

    The diameter is the maximum Euclidean distance between any two feasible points.

    Parameters
    ----------
    n_assets: int
        Number of assets
    min_weights : ndarray or float, default=0
        Minimum values for each dimension. Can be scalar (broadcasts to all dims)
        or array of shape (n,).
    max_weights : ndarray or float, default=1
        Maximum values for each dimension. Can be scalar (broadcasts to all dims)
        or array of shape (n,).
    budget : float, default=1
        Total budget constraint (sum of all weights must equal this).

    Returns
    -------
    diameter : float
        Maximum Euclidean distance between any two feasible points.
    """
    # Determine dimensionality and convert to arrays
    if isinstance(min_weights, int | float):
        min_w = np.full(n_assets, min_weights, dtype=float)
        max_w = (
            np.full(n_assets, max_weights, dtype=float)
            if isinstance(max_weights, int | float)
            else np.asarray(max_weights, dtype=float)
        )
    else:
        min_w = np.asarray(min_weights, dtype=float)
        max_w = (
            np.full(n_assets, max_weights, dtype=float)
            if isinstance(max_weights, int | float)
            else np.asarray(max_weights, dtype=float)
        )

    sum_min = np.sum(min_w)
    sum_max = np.sum(max_w)

    # Compute the range each coordinate can span
    max_coords = np.minimum(max_w, budget - (sum_min - min_w))
    min_coords = np.maximum(min_w, budget - (sum_max - max_w))

    # The diameter is the Euclidean norm of the range vector
    ranges = max_coords - min_coords
    diameter = np.sqrt(np.sum(ranges**2))

    return diameter


def estimate_gradient_bound(
    objective: BaseMeasure | None = None,
    clip_relative_lower: float = 0.6,
    clip_relative_upper: float = 1.4,
    historical_returns: np.ndarray | None = None,
    n_assets: int | None = None,
    norm: Literal["linf", "l2"] = "linf",
) -> float:
    """Estimate gradient bound for given objective using MEASURE_PROPERTIES.

    Parameters
    ----------
    objective : BaseMeasure | None
        Objective measure (enum, string name, or None for logwealth).
        Uses MEASURE_PROPERTIES to determine convexity class.
    clip_relative_lower, clip_relative_upper : float
        Expected range of price relatives for exp-concave objectives.
        Default [0.6, 1.4] corresponds to ±40% daily returns.
    historical_returns : np.ndarray | None
        Historical returns for data-dependent estimation (variance, mean, etc.).
    n_assets : int | None
        Number of assets. Required when ``norm="l2"`` for exp-concave objectives
        to correctly scale the bound by sqrt(n).
    norm : {"linf", "l2"}, default="linf"
        Which norm to compute the gradient bound in:

        - ``"linf"``: L-infinity bound (suitable for entropy / EG geometry).
        - ``"l2"``: L2 bound (suitable for Euclidean / OGD geometry).
          For exp-concave objectives this scales the L-inf bound by sqrt(n).

    Returns
    -------
    float
        Estimated gradient bound G.

    Raises
    ------
    ValueError
        If objective is unknown and not in MEASURE_PROPERTIES.

    Notes
    -----
    Uses MEASURE_PROPERTIES['convexity'] to determine gradient estimation:

    **"exp-concave"** (log-wealth):
    - Gradient: g = -r / (1 + w^T r) where r are net returns
    - L-inf bound: max|r_i| / min(1 + w^T r)
    - L2 bound: sqrt(n) * L-inf bound (Cauchy-Schwarz on simplex)

    **"strongly_convex"** (variance):
    - Gradient: ∇Var(w) = 2Σw
    - Bound: G = 2||Σ||_op (requires historical data)
    - Typical daily: G ≈ 0.001-0.01 (much smaller!)

    **"linear"** (mean):
    - Gradient: ∇Mean(w) = E[X] (constant)
    - Bound: G = ||E[X]||_2, requires data

    **"convex"** (cvar, semi-deviation, etc.):
    - Data-dependent, use conservative default G ≈ 1.0
    """
    # Normalize objective to measure key
    objective = objective or PerfMeasure.LOG_WEALTH

    # Get measure properties
    if objective not in MEASURE_PROPERTIES:
        raise ValueError(
            f"Objective {objective} not found in MEASURE_PROPERTIES. "
            f"Cannot estimate gradient bound for automatic learning rate. "
            f"Please provide explicit gradient_bound parameter."
        )

    properties = MEASURE_PROPERTIES[objective]
    convexity = properties["convexity"]

    # Estimate gradient bound based on convexity class
    match convexity:
        case "exp-concave":
            # Log-wealth, EVaR, EDaR
            # L-inf bound: max|r_i| / min(1 + w^T r) <= max_relative / (1 - max_relative)
            # Conservative overestimate: max_relative * 2.0
            max_relative = max(
                abs(clip_relative_lower - 1.0),
                abs(clip_relative_upper - 1.0),
            )
            g_linf = float(max_relative * 2.0)  # ~0.8 for ±40% moves
            if norm == "l2" and n_assets is not None:
                # L2 bound: ||g||_2 <= sqrt(n) * ||g||_inf (Cauchy-Schwarz)
                return g_linf * np.sqrt(n_assets)
            return g_linf

        case "strongly_convex":
            # Variance: G = 2||Σ||_op
            if historical_returns is None:
                return 0.01  # conservative: ~1% daily vol
            cov = np.cov(historical_returns, rowvar=False)
            eigvals = np.linalg.eigvalsh(cov)
            return 2.0 * float(np.max(np.abs(eigvals)))

        case "linear":
            # Mean: G = ||E[X]||_2
            if historical_returns is None:
                return 0.001  # conservative: ~0.1% daily return
            mean_ret = np.mean(historical_returns, axis=0)
            return float(np.linalg.norm(mean_ret, ord=2))

        case _:  # "convex" or unknown
            # General convex measures (CVaR, Semi-Dev, etc.)
            return 1.0  # conservative default


def compute_ogd_learning_rate(
    t: int,
    diameter: float,
    gradient_bound: float,
    scale: Literal["theory", "moderate", "empirical"] = "theory",
) -> float:
    """Compute OGD learning rate.

    Parameters
    ----------
    t : int
        Current time step (0-indexed). Internally converted to 1-indexed.
    diameter : float
        Domain diameter (√2 for standard simplex).
    gradient_bound : float
        Gradient bound G (Lipschitz constant). For log-wealth, G≈0.8.
    scale : {"theory", "moderate", "empirical"}, default="theory"
        Scaling mode:

        - **"theory"**: D/(G√(t+1)) — Standard OCO bound (Zinkevich 2003)
        - **"moderate"**: 2√2 · D/(G√(t+1)) — Hazan's constant
        - **"empirical"**: √2 · D/(G√(t+1)) — Mild boost over theory

    Returns
    -------
    float
        Learning rate η_t.

    Notes
    -----
    For Online Gradient Descent (OGD) on a convex domain with diameter D
    and Lipschitz gradients bounded by G, the regret-optimal rate is
    η_t = D/(G√t).
    """
    base_rate = diameter / (gradient_bound * np.sqrt(t + 1))

    match scale:
        case "theory":
            return base_rate
        case "moderate":
            return base_rate * 2.0 * np.sqrt(2.0)
        case "empirical":
            return base_rate * np.sqrt(2.0)
        case _:
            raise ValueError(
                f"Unknown scale '{scale}'. Choose 'theory', 'moderate', or 'empirical'."
            )


def compute_eg_learning_rate(
    t: int,
    n_assets: int,
    scale: Literal["theory", "moderate", "empirical"] = "theory",
) -> float:
    """Compute EG (Exponentiated Gradient) learning rate.

    Parameters
    ----------
    t : int
        Current time step (0-indexed). Internally converted to 1-indexed.
    n_assets : int
        Number of assets (dimension).
    scale : {"theory", "moderate", "empirical"}, default="theory"
        Scaling mode:

        - **"theory"**: √(log(n)/(t+1)) — Hazan's worst-case OCO bound
        - **"moderate"**: √(8·log(n)/(t+1)) — 2√2 boost (Hazan's book constant)
        - **"empirical"**: √(2·log(n)/(t+1)) — Mild √2 boost over theory

    Returns
    -------
    float
        Learning rate η_t.

    Notes
    -----
    All three scales preserve the same O(√(T log n)) regret guarantee
    (up to constants). The "theory" rate is the most conservative and is
    recommended as default for general-purpose use.

    References
    ----------
    - Hazan (2016), "Introduction to Online Convex Optimization", Corollary 7.2
    """
    return _multiplicative_learning_rate(t, n_assets, scale)


def compute_prod_learning_rate(t: int, n_assets: int, scale: str = "theory") -> float:
    """Compute PROD (Soft-Bayes) learning rate.

    Parameters
    ----------
    t : int
        Current time step (0-indexed). Internally converted to 1-indexed.
    n_assets : int
        Number of assets (dimension).
    scale : {"theory", "moderate", "empirical"}, default="theory"
        Scaling mode (same as EG):

        - **"theory"**: √(log(n)/(t+1)) — Worst-case OCO bound
        - **"moderate"**: √(8·log(n)/(t+1)) — 2√2 boost
        - **"empirical"**: √(2·log(n)/(t+1)) — Mild √2 boost over theory

    Returns
    -------
    float
        Learning rate η_t.

    Notes
    -----
    PROD (Soft-Bayes Product algorithm) uses Burg entropy (log-barrier) mirror
    map. The optimal rate follows the same scaling as EG.
    """
    return _multiplicative_learning_rate(t, n_assets, scale)


def compute_adaptive_base_learning_rate(n_assets: int, diameter: float) -> float:
    """Compute adaptive base learning rate: η₀ = D/√n.

    Notes
    -----
    Scales the feasible-set diameter by the square root of the number of assets.
    This heuristic is not the per-coordinate diameter used in theoretical
    AdaGrad regret bounds. Supply an explicit learning rate to use another scale.
    """
    return diameter / np.sqrt(n_assets)


def get_auto_learning_rate(
    strategy: FTWStrategy,
    n_assets: int,
    min_weights: np.ndarray | float = 0,
    max_weights: np.ndarray | float = 1,
    budget: float = 1.0,
    objective: BaseMeasure | None = None,
    gradient_bound: float | None = None,
    historical_returns: np.ndarray | None = None,
    scale: str = "theory",
) -> Callable[[int], float]:
    """Get automatic learning rate for strategy and objective.

    Parameters
    ----------
    strategy : FTWStrategy
        Strategy: FTWStrategy enum.
    n_assets : int
        Number of assets
    budget : float
        Investment budget (for diameter)
    objective : BaseMeasure | None
        Objective function (enum, string, or None for logwealth).
        Determines gradient bound estimation. If None, assumes logwealth.
    gradient_bound : float | None
        If provided, use this G directly; else estimate from objective.
    historical_returns : np.ndarray | None
        Historical data for variance/mean gradient estimation.
    scale : {"theory", "moderate", "empirical"}, default="theory"
        Scaling mode for learning rates:

        - **"theory"**: √(log(n)/(t+1)) for EG/PROD, D/(G√(t+1)) for OGD
        - **"moderate"**: 2√2 boost over theory (Hazan's book constant)
        - **"empirical"**: √2 boost over theory (mild increase)

    Returns
    -------
    Callable[[int], float]
        Learning rate function η(t). Always returns a callable for consistency
        with FirstOrderOCO which accepts callables. For strategies with constant
        base rates (AdaGrad, SWORD), returns lambda t: constant.

    Raises
    ------
    ValueError
        If strategy is unknown or objective cannot be estimated.
    """
    try:
        strategy = FTWStrategy(strategy)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Unknown strategy {strategy}") from error

    objective = objective or PerfMeasure.LOG_WEALTH

    if scale not in ("theory", "moderate", "empirical"):
        raise ValueError(
            f"Unknown scale '{scale}'. Choose 'theory', 'moderate', or 'empirical'."
        )
    # Estimate diameter
    diameter = simplex_diameter(
        n_assets=n_assets,
        min_weights=min_weights,
        max_weights=max_weights,
        budget=budget,
    )

    # Estimate gradient bound (OGD needs L2 norm; others use L-inf or don't use G)
    if gradient_bound is None:
        ogd_norm = "l2" if strategy == FTWStrategy.OGD else "linf"
        gradient_bound = estimate_gradient_bound(
            objective=objective,
            historical_returns=historical_returns,
            n_assets=n_assets,
            norm=ogd_norm,
        )
        # Warn if using non-default objective without explicit bound
        # Check if objective is logwealth (None, PerfMeasure.LOG_WEALTH, or "logwealth" string)
        if objective != PerfMeasure.LOG_WEALTH:
            warnings.warn(
                f"Auto learning rate for objective '{objective}' uses estimated gradient bound G={gradient_bound:.4f}. For best results with custom objectives, provide explicit gradient_bound parameter.",
                UserWarning,
                stacklevel=2,
            )

    match strategy:
        case FTWStrategy.OGD:
            return lambda t: compute_ogd_learning_rate(
                t=t, diameter=diameter, gradient_bound=gradient_bound, scale=scale
            )

        case FTWStrategy.EG:
            return lambda t: compute_eg_learning_rate(t, n_assets, scale=scale)

        case FTWStrategy.PROD:
            return lambda t: compute_prod_learning_rate(t, n_assets, scale=scale)

        case _:
            base_rate = compute_adaptive_base_learning_rate(n_assets, diameter)
            return lambda t: base_rate


def _multiplicative_learning_rate(t: int, n_assets: int, scale: str) -> float:
    """Evaluate the schedule shared by EG and PROD."""
    match scale:
        case "theory":
            return np.sqrt(np.log(n_assets) / (t + 1))

        case "moderate":
            return np.sqrt(8.0 * np.log(n_assets) / (t + 1))

        case "empirical":
            return np.sqrt(2.0 * np.log(n_assets) / (t + 1))

        case _:
            raise ValueError(
                f"Unknown scale '{scale}'. Choose 'theory', 'moderate', or 'empirical'."
            )
