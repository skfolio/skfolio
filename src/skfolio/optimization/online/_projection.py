"""Projection utilities and projector classes for portfolio weights."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
from numpy.typing import ArrayLike
from scipy.special import softmax

import skfolio.typing as skt
from skfolio.optimization._constraints import (
    budget_constraints,
    call_custom_function,
    weight_constraints,
)
from skfolio.optimization._constraints import (
    linear_constraints as build_linear_constraints,
)
from skfolio.optimization.online._discrete_constraints import (
    discrete_constraints,
    has_discrete_constraints,
    validate_discrete_solver,
)
from skfolio.optimization.online._distribution_constraints import (
    DistributionConstraintConfig,
    distribution_constraints,
)
from skfolio.optimization.online._geometry import mirror_potential
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
    BaseMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
)
from skfolio.prior import ReturnDistribution
from skfolio.utils.tools import input_to_array


@dataclass
class ProjectionConfig:
    """Container for projection-related constraints.

    Parameters
    ----------
    lower, upper : array-like or float or None
        Box bounds. Defaults to [0, 1] if None.
    budget : float or None
        Equality budget. If None, no equality budget is enforced.
    min_budget, max_budget : float or None
        Budget bounds (mutually exclusive with equality `budget`).
    max_short, max_long : float or None
        Aggregate short/long caps.
    groups, linear_constraints, left_inequality, right_inequality : array-like or None
        Rich linear/group constraints.
    X_tracking, tracking_error_benchmark, max_tracking_error : array-like or None
        Time-series tracking error constraint components.
    covariance, variance_bound : array-like or None
        Variance-style quadratic bound.
    return_distribution : ReturnDistribution or None
        Causal factor information for named exposure constraints.
    distribution_config : DistributionConstraintConfig or None
        Minimum expected return and risk caps from the causal prior distribution.
    cardinality, group_cardinalities : int or dict or None
        Global and named-group support limits requiring a mixed-integer solver.
    add_constraints : callable or None
        Custom convex constraints, rebuilt for the current weight variable.
    constraint_estimator : object or None
        Estimator passed to a two-argument custom callback.
    strict_linear_constraints : bool
        Raise for unknown asset, group or factor names when True.
    previous_weights, max_turnover : array-like or float or None
        Turnover information for L1 cap.
    """

    lower: Any | None = 0.0
    upper: Any | None = 1.0
    budget: float | None = 1.0
    min_budget: float | None = None
    max_budget: float | None = None
    max_short: float | None = None
    max_long: float | None = None
    groups: Any | None = None
    linear_constraints: skt.LinearConstraints | None = None
    left_inequality: Any | None = None
    right_inequality: Any | None = None
    X_tracking: Any | None = None
    tracking_error_benchmark: Any | None = None
    max_tracking_error: float | None = None
    covariance: Any | None = None
    variance_bound: float | None = None
    previous_weights: Any | None = None
    max_turnover: float | None = None
    solver_params: dict[str, Any] | None = None
    return_distribution: ReturnDistribution | None = None
    distribution_config: DistributionConstraintConfig | None = None
    cardinality: int | None = None
    group_cardinalities: dict[str, int] | None = None
    add_constraints: skt.ConstraintFunction | None = None
    constraint_estimator: object | None = None
    strict_linear_constraints: bool = True


class BaseProjector:
    """Interface for projecting candidate portfolio weights."""

    def project(self, y: np.ndarray) -> np.ndarray:
        """Project candidate weights onto the configured feasible set."""
        raise NotImplementedError

    def solve_update(
        self,
        mirror: BaseMirrorMap,
        dual: np.ndarray,
        gradient: np.ndarray | None = None,
    ) -> np.ndarray:
        """Solve a native constrained update, or reject an unsupported projector."""
        if type(mirror) is EuclideanMirrorMap:
            return self.project(dual)
        raise NotImplementedError(
            "This projector does not implement native mirror updates"
        )


class IdentityProjector(BaseProjector):
    """Projector that returns the input unchanged. Useful to test pure mirror updates."""

    def project(self, y: np.ndarray) -> np.ndarray:
        """Project candidate weights onto the configured feasible set."""
        return np.asarray(y, dtype=float)


class FastProjector(BaseProjector):
    """Fast projector using box+sum and optional L1 turnover cap."""

    def __init__(self, config: ProjectionConfig) -> None:
        self.config = config

    def solve_update(
        self,
        mirror: BaseMirrorMap,
        dual: np.ndarray,
        gradient: np.ndarray | None = None,
    ) -> np.ndarray:
        """Dispatch the update through the common geometry-aware constraint path."""
        return AutoProjector(self.config).solve_update(mirror, dual, gradient)

    def project(self, y: np.ndarray) -> np.ndarray:
        """Project candidate weights onto the configured feasible set."""
        return project_with_turnover(
            w_raw=y,
            previous_weights=self.config.previous_weights,
            max_turnover=self.config.max_turnover,
            lower=0.0 if self.config.lower is None else self.config.lower,
            upper=1.0 if self.config.upper is None else self.config.upper,
            budget=self.config.budget,
        )


class ConvexProjector(BaseProjector):
    """Convex projector using cvxpy with rich constraints (see ProjectionConfig)."""

    def __init__(self, config: ProjectionConfig) -> None:
        self.config = config

    def solve_update(
        self,
        mirror: BaseMirrorMap,
        dual: np.ndarray,
        gradient: np.ndarray | None = None,
    ) -> np.ndarray:
        """Dispatch the update through the common geometry-aware constraint path."""
        return AutoProjector(self.config).solve_update(mirror, dual, gradient)

    def project(self, y: np.ndarray) -> np.ndarray:
        """Project candidate weights onto the configured feasible set."""
        return self._solve(y)

    def _solve(
        self,
        y: np.ndarray,
        *,
        objective: Callable[[cp.Variable], cp.Expression] | None = None,
    ) -> np.ndarray:
        """Solve with the same constraints for projection and mirror updates."""
        return project_convex(
            w_raw=y,
            objective=objective,
            budget=self.config.budget,
            lower=self.config.lower,
            upper=self.config.upper,
            min_budget=self.config.min_budget,
            max_budget=self.config.max_budget,
            max_short=self.config.max_short,
            max_long=self.config.max_long,
            groups=self.config.groups,
            linear_constraints=self.config.linear_constraints,
            return_distribution=self.config.return_distribution,
            distribution_config=self.config.distribution_config,
            cardinality=self.config.cardinality,
            group_cardinalities=self.config.group_cardinalities,
            add_constraints=self.config.add_constraints,
            constraint_estimator=self.config.constraint_estimator,
            strict_linear_constraints=self.config.strict_linear_constraints,
            left_inequality=self.config.left_inequality,
            right_inequality=self.config.right_inequality,
            X=self.config.X_tracking,
            tracking_error_benchmark=self.config.tracking_error_benchmark,
            max_tracking_error=self.config.max_tracking_error,
            covariance=self.config.covariance,
            variance_bound=self.config.variance_bound,
            previous_weights=self.config.previous_weights,
            max_turnover=self.config.max_turnover,
            solver_params=self.config.solver_params,
        )


class AutoProjector(BaseProjector):
    """
    Automatically choose fast or convex projector based on provided constraints.
    Fast projector is used when no rich constraints are provided.
    Convex projector is used when rich constraints are provided.
    """

    def __init__(self, config: ProjectionConfig) -> None:
        self.config = config
        self._fast = FastProjector(config)
        self._cvx = ConvexProjector(config)

    def _needs_cvx(self) -> bool:
        """Determine whether rich constraints require the convex solver."""
        return any(
            v is not None
            for v in [
                self.config.cardinality,
                self.config.group_cardinalities,
                self.config.add_constraints,
                self.config.distribution_config,
                self.config.groups,
                self.config.linear_constraints,
                self.config.left_inequality,
                self.config.right_inequality,
                self.config.X_tracking,
                self.config.tracking_error_benchmark,
                self.config.max_tracking_error,
                self.config.covariance,
                self.config.variance_bound,
                self.config.min_budget,
                self.config.max_budget,
                self.config.max_short,
                self.config.max_long,
            ]
        )

    def solve_update(
        self,
        mirror: BaseMirrorMap,
        dual: np.ndarray,
        gradient: np.ndarray | None = None,
    ) -> np.ndarray:
        """Minimize psi(w) - dual @ w over the configured constraint set.

        Work with the dual vector directly: projecting a normalized inverse can
        lose information for free budgets, barriers and adaptive/composite maps.
        """
        if has_discrete_constraints(
            n_assets=len(dual),
            cardinality=self.config.cardinality,
            group_cardinalities=self.config.group_cardinalities,
        ) and not isinstance(mirror, (EuclideanMirrorMap, AdaptiveMahalanobisMap)):
            raise ValueError(
                "Discrete constraints require Euclidean or adaptive quadratic geometry; entropy updates are unsupported"
            )
        if type(mirror) is EuclideanMirrorMap:
            return self.project(dual)
        config = self.config
        potential = mirror_potential(mirror, len(dual), gradient)
        if (
            isinstance(mirror, EntropyMirrorMap)
            and not self._needs_cvx()
            and config.budget == 1
            and np.all(np.asarray(config.lower) == 0)
            and np.all(np.asarray(config.upper) == 1)
            and config.max_turnover is None
        ):
            return softmax(dual)
        if (
            type(mirror) is EntropyMirrorMap
            and not self._needs_cvx()
            and config.budget is None
            and np.all(np.asarray(config.lower) == 0)
            and np.all(np.asarray(config.upper) == 1)
            and config.max_turnover is None
        ):
            # The separable objective has derivative log(w) + 1 - dual.
            # Clip in log space at the upper bound to avoid exponential overflow.
            return np.exp(np.minimum(np.asarray(dual) - 1.0, 0.0))
        return self._cvx._solve(dual, objective=lambda w: potential(w) - dual @ w)

    def project(self, y: np.ndarray) -> np.ndarray:
        """Project candidate weights onto the configured feasible set."""
        if self._needs_cvx():
            return self._cvx.project(y)
        return self._fast.project(y)


def _bound_array(
    value: ArrayLike | float | None,
    *,
    n_assets: int,
    default: float,
    name: str,
) -> np.ndarray:
    """Expand projection bounds, preserving scalar dtype and array validation."""
    if value is None:
        return np.full(n_assets, default, dtype=float)
    if np.isscalar(value):
        return np.full(n_assets, value)
    return input_to_array(
        items=value,
        n_assets=n_assets,
        fill_value=default,
        dim=1,
        assets_names=None,
        name=name,
    )


def project_box_and_sum(
    y: ArrayLike,
    lower: ArrayLike | float,
    upper: ArrayLike | float,
    budget: float | None = 1.0,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> np.ndarray:
    """
    Project onto box with budget equality using bisection.

    Projects a vector ``y`` onto the set
    ``{ w ∈ R^n | sum(w) = budget, lower ≤ w ≤ upper }``
    via a 1D bisection on a dual parameter ``theta`` such that
    ``w = clip(y - theta, lower, upper)`` satisfies the budget.

    Parameters
    ----------
    y : ArrayLike of shape (n,)
        Raw weights to be projected.
    lower : ArrayLike or float
        Lower bounds per coordinate (broadcasted if float).
    upper : ArrayLike or float
        Upper bounds per coordinate (broadcasted if float).
    budget : float, default=1.0
        Target sum of weights.
    tol : float, default=1e-12
        Tolerance on the budget equality.
    max_iter : int, default=100
        Maximum bisection iterations.

    Returns
    -------
    w : ndarray of shape (n,)
        Projected weights.
    """
    y_arr = np.asarray(y, dtype=float)
    n = int(y_arr.shape[0])
    lo_arr = _bound_array(lower, n_assets=n, default=0.0, name="lower_bounds")

    up_arr = _bound_array(upper, n_assets=n, default=1.0, name="upper_bounds")

    if (
        not np.isfinite(y_arr).all()
        or not np.isfinite(lo_arr).all()
        or not np.isfinite(up_arr).all()
    ):
        raise ValueError("Candidate weights and bounds must be finite")
    if np.any(lo_arr > up_arr):
        raise ValueError("Lower bounds exceed upper bounds")
    if budget is None:
        return np.clip(y_arr, lo_arr, up_arr)

    low_sum = float(np.sum(lo_arr))
    up_sum = float(np.sum(up_arr))
    b = float(budget)
    if b < low_sum - 1e-15 or b > up_sum + 1e-15:
        raise ValueError("Budget is infeasible under the supplied bounds")

    lo = float(np.min(y_arr - up_arr)) - 1.0
    hi = float(np.max(y_arr - lo_arr)) + 1.0

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        w = np.clip(y_arr - mid, lo_arr, up_arr)
        s = float(np.sum(w))
        if abs(s - b) <= tol:
            return w
        if s > b:
            lo = mid
        else:
            hi = mid
    return np.clip(y_arr - 0.5 * (lo + hi), lo_arr, up_arr)


def _validate_constraint_inputs(
    *,
    max_turnover: float | None = None,
    previous_weights: ArrayLike | None = None,
    variance_bound: float | None = None,
    covariance: ArrayLike | None = None,
    max_tracking_error: float | None = None,
    X: ArrayLike | None = None,
    tracking_error_benchmark: ArrayLike | None = None,
) -> None:
    """Reject active caps with invalid values or missing supporting inputs."""
    for name, value in (
        ("max_turnover", max_turnover),
        ("variance_bound", variance_bound),
        ("max_tracking_error", max_tracking_error),
    ):
        if value is not None and (
            np.ndim(value) != 0 or not np.isfinite(value) or value < 0
        ):
            raise ValueError(f"{name} must be a finite nonnegative scalar")
    if max_turnover is not None and previous_weights is None:
        raise ValueError("max_turnover requires previous_weights")
    if variance_bound is not None and covariance is None:
        raise ValueError("variance_bound requires covariance")
    if max_tracking_error is not None and (
        X is None or tracking_error_benchmark is None
    ):
        raise ValueError("max_tracking_error requires X and tracking_error_benchmark")


def project_with_turnover(
    w_raw: ArrayLike,
    previous_weights: ArrayLike | None,
    max_turnover: float | None,
    lower: ArrayLike | float | None = None,
    upper: ArrayLike | float | None = None,
    budget: float | None = 1.0,
) -> np.ndarray:
    """
    Project with box/budget and optional L1 turnover cap.

    Parameters
    ----------
    w_raw : ArrayLike of shape (n,)
        Proposed weights before projection.
    previous_weights : ArrayLike of shape (n,), optional
        Previous portfolio weights used to enforce turnover cap.
    max_turnover : float, optional
        L1 distance cap between projected weights and previous weights.
    lower, upper : ArrayLike or float, optional
        Box bounds; broadcasted if float. Defaults to [0, 1].
    budget : float, default=1.0
        Target sum of weights.

    Returns
    -------
    w : ndarray of shape (n,)
        Projected weights.
    """
    _validate_constraint_inputs(
        max_turnover=max_turnover, previous_weights=previous_weights
    )
    y = np.asarray(w_raw, dtype=float)
    n = int(y.shape[0])
    lower_bounds = _bound_array(lower, n_assets=n, default=0.0, name="lower_bounds")

    u = _bound_array(upper, n_assets=n, default=1.0, name="upper_bounds")

    # Iteratively enforce both box/budget and L1-turnover constraints
    w = project_box_and_sum(y=y, lower=lower_bounds, upper=u, budget=budget)

    if previous_weights is None or max_turnover is None:
        return w

    prev = np.asarray(previous_weights, dtype=float)
    if prev.shape != y.shape or not np.isfinite(prev).all():
        raise ValueError("previous_weights must be finite and match candidate weights")
    if max_turnover == 0:
        feasible = project_box_and_sum(
            y=prev, lower=lower_bounds, upper=u, budget=budget
        )
        if not np.allclose(prev, feasible, atol=1e-12, rtol=0):
            raise ValueError("Zero turnover holdings violate the bounds or budget")
        return prev.copy()
    if np.abs(w - prev).sum() <= max_turnover:
        return w
    # Solve the intersection, rather than alternate projections for a fixed count.
    return project_convex(
        y,
        budget=budget,
        lower=lower_bounds,
        upper=u,
        previous_weights=prev,
        max_turnover=max_turnover,
    )


def project_convex(
    w_raw: ArrayLike,
    budget: float | None = 1.0,
    lower: ArrayLike | float | None = 0.0,
    upper: ArrayLike | float | None = 1.0,
    min_budget: float | None = None,
    max_budget: float | None = None,
    max_short: float | None = None,
    max_long: float | None = None,
    # group/linear constraints
    groups: skt.Groups | ArrayLike | None = None,
    linear_constraints: skt.LinearConstraints | None = None,
    left_inequality: skt.Inequality | None = None,
    right_inequality: skt.Inequality | None = None,
    # tracking error (benchmark time-series)
    X: ArrayLike | None = None,
    tracking_error_benchmark: ArrayLike | None = None,
    max_tracking_error: float | None = None,
    # simple variance-style bound
    covariance: ArrayLike | None = None,
    variance_bound: float | None = None,
    # turnover
    previous_weights: ArrayLike | None = None,
    max_turnover: float | None = None,
    solver_params: dict[str, Any] | None = None,
    objective: Callable[[cp.Variable], cp.Expression] | None = None,
    distribution_config: DistributionConstraintConfig | None = None,
    return_distribution: ReturnDistribution | None = None,
    cardinality: int | None = None,
    group_cardinalities: dict[str, int] | None = None,
    add_constraints: skt.ConstraintFunction | None = None,
    constraint_estimator: object | None = None,
    strict_linear_constraints: bool = True,
) -> np.ndarray:
    r"""
    Convex projection with advanced constraints using cvxpy.

    Solves: minimize 0.5 * ||w - w_raw||_2^2 subject to
    sum(w) = budget
    lower <= w <= upper
    group_min <= G @ w <= group_max  (if provided)
    w' Sigma w <= variance_bound (if provided)

    Active turnover, variance and tracking-error caps require their supporting
    inputs. Supplying data without a cap does not activate a constraint.
    """
    _validate_constraint_inputs(
        max_turnover=max_turnover,
        previous_weights=previous_weights,
        variance_bound=variance_bound,
        covariance=covariance,
        max_tracking_error=max_tracking_error,
        X=X,
        tracking_error_benchmark=tracking_error_benchmark,
    )
    if (left_inequality is None) != (right_inequality is None):
        raise ValueError(
            "left_inequality and right_inequality must be provided together"
        )
    w0 = np.asarray(w_raw, dtype=float)
    n = int(w0.shape[0])
    lower_bounds = _bound_array(lower, n_assets=n, default=0.0, name="lower_bounds")

    u = _bound_array(upper, n_assets=n, default=1.0, name="upper_bounds")

    w = cp.Variable(n)
    minimization = cp.Minimize(
        0.5 * cp.sum_squares(w - w0) if objective is None else objective(w)
    )
    constraints = weight_constraints(
        w=w,
        factor=1.0,
        min_weights=lower_bounds,
        max_weights=u,
        allow_negative_weights=True,
        max_long=max_long,
        max_short=max_short,
    )
    constraints += budget_constraints(
        w=w, factor=1.0, budget=budget, min_budget=min_budget, max_budget=max_budget
    )
    constraints += build_linear_constraints(
        w=w,
        factor=1.0,
        linear_constraints=linear_constraints,
        groups=None if groups is None else np.asarray(groups),
        return_distribution=return_distribution,
        raise_if_group_missing=strict_linear_constraints,
    )

    if left_inequality is not None and right_inequality is not None:
        A = np.asarray(left_inequality)
        b = np.asarray(right_inequality)
        if A.ndim != 2 or b.ndim != 1 or A.shape[0] != b.shape[0] or A.shape[1] != n:
            raise ValueError("Invalid left/right inequality shapes.")
        constraints.append(A @ w - b <= 0)

    if (
        X is not None
        and tracking_error_benchmark is not None
        and max_tracking_error is not None
    ):
        # Tracking error as sample standard deviation of active returns:
        # std(R w - y) = ||(R w - y) - mean(R w - y)||_2 / sqrt(T-1)
        Rt = np.asarray(X, dtype=float)
        y = np.asarray(tracking_error_benchmark, dtype=float)
        if Rt.ndim != 2 or y.ndim != 1 or Rt.shape[0] != y.shape[0] or Rt.shape[1] != n:
            raise ValueError("Invalid shapes for TE: X (T,n), benchmark (T,)")
        T = Rt.shape[0]
        ones = np.ones((T, 1))
        resid = Rt @ w - y
        mean_resid = (1.0 / T) * (ones.T @ resid)
        centered = resid - ones @ mean_resid
        constraints.append(
            cp.norm(centered, 2) / np.sqrt(max(T - 1, 1)) <= float(max_tracking_error)
        )
    if covariance is not None and variance_bound is not None:
        # Variance-style bound alternative
        Sigma = np.asarray(covariance, dtype=float)
        constraints.append(cp.quad_form(w, Sigma) <= float(variance_bound))

    if distribution_config is not None:
        constraints.extend(
            distribution_constraints(
                w=w, distribution=return_distribution, config=distribution_config
            )
        )

    if add_constraints is not None:
        custom = call_custom_function(
            func=add_constraints,
            w=w,
            estimator=constraint_estimator,
            name="add_constraints",
        )
        custom = custom if isinstance(custom, list) else [custom]
        if any(not isinstance(item, cp.Constraint) for item in custom):
            raise TypeError(
                "add_constraints must return a CVXPY constraint or a list of constraints"
            )
        constraints.extend(custom)

    discrete = has_discrete_constraints(
        n_assets=n, cardinality=cardinality, group_cardinalities=group_cardinalities
    )
    if discrete:
        validate_discrete_solver(solver_params)
        constraints.extend(
            discrete_constraints(
                w=w,
                lower=lower_bounds,
                upper=u,
                cardinality=cardinality,
                group_cardinalities=group_cardinalities,
                groups=None if groups is None else np.asarray(groups),
            )
        )

    # Turnover L1 cap if requested
    if previous_weights is not None and max_turnover is not None:
        w_prev = np.asarray(previous_weights, dtype=float)
        if w_prev.shape != (n,) or not np.isfinite(w_prev).all():
            raise ValueError("previous_weights must be finite and have shape (n,)")
        constraints.append(cp.norm1(w - w_prev) <= float(max_turnover))

    # Zero turnover leaves a singleton feasible set. Preserve the supplied
    # holdings exactly instead of introducing solver-sized artificial trades.
    if (
        max_turnover == 0
        and previous_weights is not None
        and distribution_config is None
        and add_constraints is None
        and not discrete
    ):
        anchor = np.asarray(previous_weights, dtype=float)
        w.value = anchor
        if any(np.max(constraint.violation()) > 1e-10 for constraint in constraints):
            raise RuntimeError(
                "Convex projection failed: zero turnover holdings violate constraints."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            if not np.isfinite(minimization.value):
                raise RuntimeError(
                    "Convex projection failed: holdings lie outside the mirror domain."
                )
        return anchor.copy()

    prob = cp.Problem(minimization, constraints)
    if add_constraints is not None:
        if not prob.is_dcp():
            raise ValueError("add_constraints must define a convex DCP problem")
        if cp.Problem(cp.Minimize(0), custom).is_mixed_integer():
            raise ValueError(
                "Online custom constraints do not support integer variables"
            )
    if solver_params is None:
        solver_params = dict(solver=cp.CLARABEL, verbose=False)

    if discrete:
        try:
            prob.solve(**solver_params)
        except cp.SolverError as err:
            raise ValueError(
                "The selected mixed-integer solver cannot solve this projection; check its cone support and options"
            ) from err
        if prob.status != cp.OPTIMAL:
            raise RuntimeError(
                f"Discrete projection failed to reach an optimal solution: {prob.status}"
            )
    else:
        try:
            prob.solve(**solver_params)
        except Exception:
            prob.solve(verbose=False)

    if w.value is None:
        raise RuntimeError("Convex projection failed to converge.")
    if max_turnover == 0 and previous_weights is not None:
        return np.asarray(previous_weights, dtype=float).copy()
    return w.value
