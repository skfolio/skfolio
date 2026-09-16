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
_GMD_MASTER_FEASIBILITY_TOLERANCE = 5e-8


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

        self._returns = returns
        self._weights = weights
        self._scale_constraints = scale_constraints
        self._owa_weights = np.asarray(owa_gmd_weights(returns.shape[0]), dtype=float)
        self._absolute_tolerance = absolute_tolerance
        self._relative_tolerance = relative_tolerance

        self.expression = cp.Variable(nonneg=True, name="gmd_epigraph")

        # Any permutation defines a globally valid epigraph cut. The equal-weight
        # portfolio gives the first master a useful lower bound.
        equal_weight_returns = returns @ np.full(returns.shape[1], 1 / returns.shape[1])
        self._initial_permutation = self._stable_permutation(equal_weight_returns)
        self.initial_constraint = self._create_cut(
            self._initial_permutation, normalization_factor=1.0
        )

    def solve(
        self,
        problem: cp.Problem,
        solver: str,
        solver_params: dict,
        factor: cp.Expression,
    ) -> cp.Problem:
        """Solve successive GMD masters and return the final solved problem.

        Cut history is local to this call. Each parameter target must pass the base
        problem containing only the initial cut. There is no arbitrary cut limit:
        accept only a separated solution, and reject a duplicate materially violated
        facet as solver feasibility error.

        A relaxation can be unbounded, or return a nonpositive ratio factor, even
        when the full GMD problem has a finite, normalizable optimum. In those cases,
        materialize the exact pairwise epigraph and solve it instead.
        This exceptional fallback has quadratic size in the number of observations;
        otherwise, masters retain the reduced asset-space cuts.
        """
        cut_normalization_factors = {self._initial_permutation: 1.0}
        exact_epigraph = False
        while True:
            problem.solve(solver=solver, **solver_params)
            if problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                normalization_factor = self._scalar_value(
                    factor.value, "GMD homogeneous normalization factor"
                )
                needs_exact_epigraph = (
                    not factor.is_constant() and normalization_factor <= 1e-12
                )
            else:
                needs_exact_epigraph = problem.status in {
                    cp.UNBOUNDED,
                    cp.UNBOUNDED_INACCURATE,
                }
            if needs_exact_epigraph and not exact_epigraph:
                problem = cp.Problem(
                    problem.objective,
                    [*problem.constraints, *self._pairwise_constraints()],
                )
                exact_epigraph = True
                continue
            if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                raise cp.SolverError(
                    "GMD constraint generation requires an acceptable solved master "
                    f"status, got status '{problem.status}'"
                )
            # A conic solver can represent zero by a tiny positive residual.
            if normalization_factor <= 1e-12:
                raise cp.SolverError(
                    "GMD constraint generation received an invalid homogeneous "
                    "normalization factor"
                )
            constraint = self._separate(normalization_factor, cut_normalization_factors)
            if constraint is None:
                return problem
            problem = cp.Problem(problem.objective, [*problem.constraints, constraint])

    def evaluate(self, normalization_factor: float) -> float:
        """Evaluate exact empirical GMD at the normalized, possibly active weights."""
        normalized_returns = self._candidate_returns(normalization_factor)
        value = float(self._owa_weights @ np.sort(normalized_returns, kind="stable"))
        if not np.isfinite(value):
            raise cp.SolverError(
                "GMD constraint generation produced a non-finite normalized value"
            )
        return value

    def _separate(
        self,
        normalization_factor: float,
        cut_normalization_factors: dict[tuple[int, ...], float],
    ) -> cp.Constraint | None:
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

        exact_value = self.evaluate(normalization_factor)
        violation = homogeneous_violation / normalization_factor
        if not np.isfinite(violation):
            raise cp.SolverError(
                "GMD constraint generation produced a non-finite normalized value"
            )
        normalized_tolerance = self._absolute_tolerance + (
            self._relative_tolerance * abs(exact_value)
        )
        if violation <= normalized_tolerance:
            return None

        previous_factor = cut_normalization_factors.get(permutation, np.inf)
        if normalization_factor >= previous_factor:
            # The maximally violated facet is already present and scaled at least
            # as strongly as the current homogeneous factor requires. Allow a small
            # solver feasibility residual, but reject material violations.
            master_tolerance = max(
                normalized_tolerance, _GMD_MASTER_FEASIBILITY_TOLERANCE
            )
            if violation <= master_tolerance:
                return None
            raise cp.SolverError(
                "GMD constraint generation found a duplicate violated permutation "
                f"with violation {violation:.3e}"
            )

        cut_normalization_factors[permutation] = normalization_factor
        return self._create_cut(permutation, normalization_factor=normalization_factor)

    def _pairwise_constraints(self) -> list[cp.Constraint]:
        """Build the exact fallback without a dense observation-pair/asset matrix."""
        n_observations = self._returns.shape[0]
        row, col = np.triu_indices(n_observations, k=1)
        # A separate return variable keeps the pairwise differences sparse during
        # canonicalization. Centering is valid because GMD is translation invariant.
        portfolio_returns = cp.Variable(n_observations, name="gmd_returns")
        centered_returns = self._returns - self._returns.mean(axis=0)
        risk = (
            2
            * cp.norm(portfolio_returns[row] - portfolio_returns[col], 1)
            / (n_observations * (n_observations - 1))
        )
        return [
            portfolio_returns * self._scale_constraints
            == centered_returns @ self._weights * self._scale_constraints,
            self.expression * self._scale_constraints >= risk * self._scale_constraints,
        ]

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
        """Create one asset-coefficient permutation cut."""
        coefficients = self._owa_weights @ self._returns[np.asarray(permutation)]
        # Multiplication by a positive constant leaves the feasible set unchanged.
        # Scaling by the inverse homogenization factor makes solver feasibility errors
        # comparable to the normalized separation tolerance.
        constraint_scale = self._scale_constraints / normalization_factor
        return (
            self.expression * constraint_scale
            >= coefficients @ self._weights * constraint_scale
        )
