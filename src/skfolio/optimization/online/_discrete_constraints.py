"""Mixed-integer online constraints using the batch convex formulations."""

# Copyright (c) 2023-2026 skfolio contributors
# SPDX-License-Identifier: BSD-3-Clause

from numbers import Integral

import cvxpy as cp
import numpy as np
from cvxpy.reductions.solvers.defines import MI_SOLVERS

from skfolio.optimization.convex import MeanRisk
from skfolio.utils.equations import group_cardinalities_to_matrix


def has_discrete_constraints(
    *,
    n_assets: int,
    cardinality: int | None = None,
    group_cardinalities: dict[str, int] | None = None,
) -> bool:
    """Validate counts and determine whether binary variables are needed."""
    counts = ([] if cardinality is None else [cardinality]) + list(
        (group_cardinalities or {}).values()
    )
    if any(isinstance(k, bool) or not isinstance(k, Integral) or k < 0 for k in counts):
        raise ValueError("Cardinality limits must be nonnegative integers")
    return (cardinality is not None and cardinality < n_assets) or bool(
        group_cardinalities
    )


def validate_discrete_solver(solver_params: dict | None) -> None:
    """Reject unavailable or continuous-only solvers without falling back."""
    solver = (solver_params or {}).get("solver", cp.CLARABEL)
    if solver not in MI_SOLVERS:
        raise ValueError(
            "Discrete constraints require a mixed-integer solver, such as SCIP"
        )
    if solver not in cp.installed_solvers():
        raise ValueError(f"Mixed-integer solver {solver} is not installed")


def discrete_constraints(
    *,
    w: cp.Variable,
    lower: np.ndarray,
    upper: np.ndarray,
    cardinality: int | None,
    group_cardinalities: dict[str, int] | None,
    groups: np.ndarray | None,
) -> list[cp.Constraint]:
    """Build aligned cardinality constraints with strict group-name validation."""
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ValueError("Discrete constraints require finite weight bounds")
    if group_cardinalities:
        if groups is None:
            raise ValueError("group_cardinalities requires groups")
        group_cardinalities_to_matrix(
            groups=np.asarray(groups),
            group_cardinalities=group_cardinalities,
            raise_if_group_missing=True,
        )
    context = MeanRisk(
        cardinality=cardinality, group_cardinalities=group_cardinalities or None
    )
    context._set_scale_constraints(default=1.0)
    return context._mixed_integer_constraints(
        n_assets=w.shape[0],
        w=w,
        factor=cp.Constant(1.0),
        min_weights=lower,
        max_weights=upper,
        threshold_long=None,
        threshold_short=None,
        groups=None if groups is None else np.asarray(groups),
    )
