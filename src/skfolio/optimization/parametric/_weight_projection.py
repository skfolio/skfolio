"""Project regression predictions onto static convex portfolio constraints."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from skfolio.utils.equations import equations_to_matrix
from skfolio.utils.tools import input_to_array


class _WeightProjection:
    """Reusable Euclidean projection in the fitted investment asset order."""

    def __init__(
        self,
        asset_names: list[str] | np.ndarray | pd.Index,
        long_only: bool,
        min_weights: float | ArrayLike | dict[str, float] | None,
        max_weights: float | ArrayLike | dict[str, float] | None,
        groups: ArrayLike | None,
        linear_constraints: list[str] | None,
        *,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
    ) -> None:
        self.solver = solver
        self.solver_params = {} if solver_params is None else dict(solver_params)
        if "solver" in self.solver_params:
            raise ValueError(
                "Set the solver with `projection_solver`, not solver parameters."
            )
        n = len(asset_names)
        self.target = cp.Parameter(n)
        self.weights = cp.Variable(n)
        self.budget = cp.Parameter(nonneg=True, value=1.0)
        self.constraints = [cp.sum(self.weights) == self.budget]
        if long_only:
            self.constraints.append(self.weights >= 0)
        for name, bound, fill in (
            ("min_weights", min_weights, 0),
            ("max_weights", max_weights, 1),
        ):
            if bound is None:
                continue
            values = (
                np.full(n, bound, dtype=float)
                if np.isscalar(bound)
                else input_to_array(
                    items=bound,
                    n_assets=n,
                    fill_value=fill,
                    dim=1,
                    assets_names=np.asarray(asset_names),
                    name=name,
                )
            )
            if not np.isfinite(values).all():
                raise ValueError(f"`{name}` must contain finite values.")
            self.constraints.append(
                self.weights >= values
                if name == "min_weights"
                else self.weights <= values
            )
        if linear_constraints is not None:
            labels = np.asarray([asset_names])
            if groups is not None:
                extra = np.asarray(groups)
                if extra.ndim != 2 or extra.shape[1] != n:
                    raise ValueError("`groups` must have shape (n_groups, n_assets).")
                labels = np.concatenate([labels, extra], axis=0)
            a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
                groups=labels,
                equations=linear_constraints,
                raise_if_group_missing=True,
            )
            if len(a_eq):
                self.constraints.append(a_eq @ self.weights == b_eq)
            if len(a_ineq):
                self.constraints.append(a_ineq @ self.weights <= b_ineq)
        self.problem = cp.Problem(
            cp.Minimize(cp.sum_squares(self.weights - self.target)), self.constraints
        )

    def project(
        self,
        proposed: np.ndarray,
        covariance: np.ndarray | None = None,
        max_variance: float | None = None,
        holdings: np.ndarray | None = None,
        max_turnover: float | None = None,
        eligible: np.ndarray | None = None,
        budget: float = 1.0,
    ) -> np.ndarray:
        """Return feasible weights or fail explicitly; never clip after solving."""
        self.budget.value = budget
        result = np.empty_like(proposed)
        for i, row in enumerate(proposed):
            self.target.value = row
            constraints = list(self.constraints)
            if eligible is not None and not eligible[i].all():
                constraints.append(self.weights[~eligible[i]] == 0)
            if max_variance is not None:
                if covariance is None:
                    raise ValueError("`covariance` is required for a risk limit.")
                constraints.append(
                    cp.quad_form(self.weights, cp.psd_wrap(covariance[i]))
                    <= max_variance
                )
            if max_turnover is not None:
                if holdings is None:
                    raise ValueError("`holdings` is required for a turnover limit.")
                constraints.append(cp.norm1(self.weights - holdings[i]) <= max_turnover)
            # Reuse canonicalization when only the target or budget changes.
            problem = (
                self.problem
                if len(constraints) == len(self.constraints)
                else cp.Problem(self.problem.objective, constraints)
            )
            problem.solve(solver=self.solver, **self.solver_params)
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise ValueError(
                    f"Portfolio weight projection failed: {problem.status}."
                )
            weights = self.weights.value
            if weights is None or not np.isfinite(weights).all():
                raise ValueError(
                    "Portfolio weight projection returned invalid weights."
                )
            if any(np.max(c.violation()) > 1e-6 for c in constraints):
                raise ValueError(
                    "Portfolio weight projection violated its constraints."
                )
            result[i] = weights
        return result
