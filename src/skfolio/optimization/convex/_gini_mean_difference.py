"""Constraint generation for the empirical Gini mean difference."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import cvxpy as cp
import numpy as np

from skfolio.measures import owa_gmd_weights
from skfolio.typing import FloatArray

_GMD_ABSOLUTE_TOLERANCE = 1e-8
_GMD_RELATIVE_TOLERANCE = 1e-8
_GMD_MAX_ITERATIONS = 500


class _GiniMeanDifference:
    r"""Generate permutation epigraph cuts for the empirical GMD.

    For ordered GMD weights ``a`` and portfolio returns ``r = Xw``, the
    rearrangement inequality gives

    .. math::

        \operatorname{GMD}(r) = \max_{P \in \mathcal{P}_T} a^T P r.

    At a master-problem solution, sorting ``r`` therefore identifies a maximizing
    permutation and an exact separation oracle. For each permutation, the symbolic
    observation-level expression is reduced before it enters CVXPY:

    .. math::

        a^T P X w = (a^T P X) w.

    Consequently, every generated constraint contains only ``n_assets`` numerical
    coefficients instead of a symbolic expression of length ``n_observations``.

    Parameters
    ----------
    returns : ndarray of shape (n_observations, n_assets)
        Asset returns matrix.

    weights : cvxpy Expression of shape (n_assets,)
        Weight expression in the optimization problem. For maximum-ratio problems,
        this is deliberately the homogeneous weight variable, not normalized weights.

    scale_constraints : cvxpy Constant
        Scale applied to generated constraints.
    """

    def __init__(
        self,
        returns: FloatArray,
        weights: cp.Expression,
        scale_constraints: cp.Constant,
        *,
        absolute_tolerance: float = _GMD_ABSOLUTE_TOLERANCE,
        relative_tolerance: float = _GMD_RELATIVE_TOLERANCE,
        max_iterations: int = _GMD_MAX_ITERATIONS,
    ) -> None:
        returns = np.asarray(returns, dtype=float)
        if returns.ndim != 2:
            raise ValueError("`returns` must be a 2D array")
        if returns.shape[0] < 2:
            raise ValueError("GMD requires at least two observations")
        if not np.isfinite(returns).all():
            raise ValueError("`returns` must contain only finite values")
        if weights.shape != (returns.shape[1],):
            raise ValueError(
                "`weights` must contain one expression per asset, got shape "
                f"{weights.shape} for {returns.shape[1]} assets"
            )
        if absolute_tolerance < 0 or relative_tolerance < 0:
            raise ValueError("GMD separation tolerances must be non-negative")
        if max_iterations < 1:
            raise ValueError("GMD maximum iterations must be strictly positive")

        self._returns = returns
        self._weights = weights
        self._scale_constraints = scale_constraints
        self._owa_weights = np.asarray(owa_gmd_weights(returns.shape[0]), dtype=float)
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance
        self._max_iterations = max_iterations

        self.expression = cp.Variable(nonneg=True, name="gmd_epigraph")
        self._permutations: set[tuple[int, ...]] = set()
        self._cut_normalization_factors: dict[tuple[int, ...], float] = {}
        self._iterations = 0
        self._exact_value: float | None = None
        self._violation: float | None = None
        self._converged = False

        # A cut induced by the equal-weight portfolio gives the first master a useful
        # GMD lower bound. Any permutation defines a globally valid epigraph cut.
        equal_weight_returns = returns @ np.full(returns.shape[1], 1 / returns.shape[1])
        self._initial_permutation = self._stable_permutation(equal_weight_returns)
        self.initial_constraint = self._create_cut(
            self._initial_permutation, normalization_factor=1.0
        )

    @property
    def exact_value(self) -> float:
        """Return the last exact GMD value in normalized portfolio scale."""
        if not self._converged or self._exact_value is None:
            raise cp.SolverError(
                "GMD constraint generation did not establish convergence"
            )
        return self._exact_value

    @property
    def converged(self) -> bool:
        """Whether the current master solution passed exact separation."""
        return self._converged

    @property
    def n_cuts(self) -> int:
        """Number of distinct permutation cuts in the current master problem."""
        return len(self._permutations)

    @property
    def n_iterations(self) -> int:
        """Number of cuts generated for the current parameter value."""
        return self._iterations

    @property
    def violation(self) -> float | None:
        """Last exact separation violation."""
        return self._violation

    def reset(self) -> None:
        """Reset separation state to the initial master problem."""
        self._permutations = {self._initial_permutation}
        self._cut_normalization_factors = {self._initial_permutation: 1.0}
        self._iterations = 0
        self._exact_value = None
        self._violation = None
        self._converged = False

    def separate(self, normalization_factor: float) -> cp.Constraint | None:
        """Return a violated cut, using normalized scale for convergence.

        Cuts remain in the master problem's homogeneous variable space. Because GMD
        is positively 1-homogeneous, dividing both the exact value and epigraph
        violation by the strictly positive homogenization factor makes the stopping
        tolerance invariant to the ratio transformation.
        """
        candidate_returns = self._candidate_returns()
        permutation = self._stable_permutation(candidate_returns)
        homogeneous_exact_value = float(
            self._owa_weights @ candidate_returns[np.asarray(permutation)]
        )
        epigraph_value = self._scalar_value(self.expression.value, "GMD epigraph")
        homogeneous_violation = homogeneous_exact_value - epigraph_value

        if not np.isfinite(homogeneous_exact_value) or not np.isfinite(
            homogeneous_violation
        ):
            raise cp.SolverError(
                "GMD constraint generation produced a non-finite separation value"
            )

        # Re-evaluate with normalized weights so the reported value exactly follows
        # the public empirical-risk convention, including target-relative weights.
        normalized_returns = self._candidate_returns(
            normalization_factor=normalization_factor
        )
        normalized_permutation = self._stable_permutation(normalized_returns)
        exact_value = float(
            self._owa_weights @ normalized_returns[np.asarray(normalized_permutation)]
        )
        violation = homogeneous_violation / normalization_factor
        if not np.isfinite(exact_value) or not np.isfinite(violation):
            raise cp.SolverError(
                "GMD constraint generation produced a non-finite normalized value"
            )

        self._exact_value = exact_value
        self._violation = violation
        normalized_tolerance = self._absolute_tolerance + (
            self._relative_tolerance * abs(exact_value)
        )
        if violation <= normalized_tolerance:
            self._converged = True
            return None

        if self._iterations >= self._max_iterations:
            raise cp.SolverError(
                "GMD constraint generation reached its maximum of "
                f"{self._max_iterations} iterations with violation {violation:.3e}"
            )

        if permutation in self._permutations:
            previous_factor = self._cut_normalization_factors[permutation]
            if normalization_factor >= previous_factor:
                raise cp.SolverError(
                    "GMD constraint generation found a duplicate violated permutation "
                    f"with violation {violation:.3e}"
                )

        self._iterations += 1
        return self._create_cut(permutation, normalization_factor=normalization_factor)

    def finalize_problem_values(
        self,
        expressions: dict[str, cp.Expression],
        problem_values: dict[str, object],
    ) -> None:
        """Replace a matching loose epigraph report with exact empirical GMD."""
        exact_value = self.exact_value
        for name, expression in expressions.items():
            if expression is self.expression:
                problem_values[name] = exact_value

    def _candidate_returns(self, normalization_factor: float = 1.0) -> FloatArray:
        """Evaluate returns in the master problem's homogeneous variable space."""
        weights = np.asarray(self._weights.value, dtype=float).reshape(-1)
        if weights.shape != (self._returns.shape[1],) or not np.isfinite(weights).all():
            raise cp.SolverError(
                "GMD constraint generation received invalid weight values"
            )
        candidate_returns = self._returns @ (weights / normalization_factor)
        if not np.isfinite(candidate_returns).all():
            raise cp.SolverError(
                "GMD constraint generation produced invalid portfolio returns"
            )
        return candidate_returns

    @staticmethod
    def _stable_permutation(values: FloatArray) -> tuple[int, ...]:
        """Return a deterministic nondecreasing ordering, including for ties."""
        return tuple(np.argsort(values, kind="stable").tolist())

    @staticmethod
    def _scalar_value(value: object, name: str) -> float:
        """Validate and return a scalar CVXPY value."""
        array = np.asarray(value, dtype=float)
        if array.size != 1 or not np.isfinite(array).all():
            raise cp.SolverError(f"{name} has no finite scalar value")
        return float(array.item())

    def _create_cut(
        self, permutation: tuple[int, ...], normalization_factor: float
    ) -> cp.Constraint:
        """Create and register one asset-coefficient permutation cut."""
        coefficients = self._owa_weights @ self._returns[np.asarray(permutation)]
        self._permutations.add(permutation)
        self._cut_normalization_factors[permutation] = min(
            normalization_factor,
            self._cut_normalization_factors.get(permutation, np.inf),
        )
        # Multiplication by a positive constant leaves the feasible set unchanged.
        # Scaling by the inverse homogenization factor makes solver feasibility errors
        # comparable to the normalized separation tolerance.
        constraint_scale = self._scale_constraints / normalization_factor
        return (
            self.expression * constraint_scale
            >= coefficients @ self._weights * constraint_scale
        )
