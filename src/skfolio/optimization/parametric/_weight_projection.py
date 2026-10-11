"""Project regression predictions onto static convex portfolio constraints."""

import cvxpy as cp
import numpy as np

from skfolio.utils.equations import equations_to_matrix
from skfolio.utils.tools import input_to_array


class _WeightProjection:
    """Reusable Euclidean projection in the fitted investment asset order."""

    def __init__(
        self,
        asset_names,
        long_only,
        min_weights,
        max_weights,
        groups,
        linear_constraints,
    ):
        n = len(asset_names)
        self.target = cp.Parameter(n)
        self.weights = cp.Variable(n)
        self.constraints = [cp.sum(self.weights) == 1]
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
                else input_to_array(bound, n, fill, 1, np.asarray(asset_names), name)
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

    def project(self, proposed):
        """Return feasible weights or fail explicitly; never clip after solving."""
        result = np.empty_like(proposed)
        for i, row in enumerate(proposed):
            self.target.value = row
            self.problem.solve(solver="CLARABEL")
            if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                raise ValueError(
                    f"Portfolio weight projection failed: {self.problem.status}."
                )
            weights = self.weights.value
            if weights is None or not np.isfinite(weights).all():
                raise ValueError(
                    "Portfolio weight projection returned invalid weights."
                )
            if any(np.max(c.violation()) > 1e-6 for c in self.constraints):
                raise ValueError(
                    "Portfolio weight projection violated its constraints."
                )
            result[i] = weights
        return result
