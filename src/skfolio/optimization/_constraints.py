"""Shared convex expressions for portfolio constraints.

Inputs are already aligned to the investable asset universe by the caller.
"""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# SPDX-License-Identifier: BSD-3-Clause

import cvxpy as cp
import cvxpy.constraints as cpc
import numpy as np

import skfolio.typing as skt
from skfolio.prior import ReturnDistribution
from skfolio.typing import AnyArray, BoolArray, FloatArray, StrArray
from skfolio.utils.equations import equations_to_matrix


def weight_constraints(
    w: cp.Variable,
    factor: skt.Factor | float,
    min_weights: float | FloatArray | None,
    max_weights: float | FloatArray | None,
    allow_negative_weights: bool,
    max_long: float | None = None,
    max_short: float | None = None,
    scale: float | cp.Constant = 1.0,
    owner: str = "Optimizer",
) -> list[cpc.Constraint]:
    """Constrain aligned weights and aggregate long and short exposure."""
    constraints = []
    if min_weights is not None:
        if not allow_negative_weights and np.any(min_weights < 0):
            raise ValueError(
                f"{owner} must have non negative `min_weights` "
                f"constraint otherwise the problem becomes non-convex."
            )
        constraints.append(w * scale >= min_weights * factor * scale)

    if max_weights is not None:
        constraints.append(w * scale <= max_weights * factor * scale)

    if max_long is not None:
        max_long = float(max_long)
        if max_long <= 0:
            raise ValueError("`max_long` must be strictly positive")
        constraints.append(cp.sum(cp.pos(w)) * scale <= max_long * factor * scale)

    if max_short is not None:
        max_short = float(max_short)
        if max_short <= 0:
            raise ValueError("`max_short` must be strictly positive")
        constraints.append(cp.sum(cp.neg(w)) * scale <= max_short * factor * scale)

    return constraints


def budget_constraints(
    w: cp.Variable,
    factor: skt.Factor | float,
    budget: float | None,
    min_budget: float | None = None,
    max_budget: float | None = None,
    scale: float | cp.Constant = 1.0,
) -> list[cpc.Constraint]:
    """Constrain the equality budget or lower and upper budget limits."""
    constraints = []
    if min_budget is not None:
        constraints.append(cp.sum(w) * scale >= float(min_budget) * factor * scale)

    if max_budget is not None:
        constraints.append(cp.sum(w) * scale <= float(max_budget) * factor * scale)

    if budget is not None:
        if max_budget is not None:
            raise ValueError(
                "`max_budget`and `budget` cannot be provided at the same time"
            )
        if min_budget is not None:
            raise ValueError(
                "`min_budget`and `budget` cannot be provided at the same time"
            )
        constraints.append(cp.sum(w) * scale == float(budget) * factor * scale)

    return constraints


def linear_constraints(
    w: cp.Variable,
    factor: skt.Factor | float,
    linear_constraints: skt.LinearConstraints | None,
    groups: AnyArray | None,
    assets_names: StrArray | None = None,
    investable_mask: BoolArray | None = None,
    return_distribution: ReturnDistribution | None = None,
    scale: float | cp.Constant = 1.0,
) -> list[cpc.Constraint]:
    """Build named asset, group and factor constraints on the aligned universe."""
    constraints = []
    if linear_constraints is not None:
        if groups is None:
            if assets_names is None:
                raise ValueError(
                    "If `linear_constraints` is provided you must provide either"
                    " `groups` or `X` as a DataFrame with asset names in columns"
                )
            if investable_mask is None:
                groups = np.asarray([assets_names])
            else:
                groups = np.asarray([assets_names[investable_mask]])

        # Extract factor info from return_distribution for factor constraints
        loading_matrix = None
        factor_groups = None
        if return_distribution is not None:
            factor_model = return_distribution.factor_model
            if factor_model is not None:
                loading_matrix = factor_model.loading_matrix
                if factor_model.factor_families is not None:
                    factor_groups = np.array(
                        [
                            factor_model.factor_names,
                            factor_model.factor_families,
                        ]
                    )
                else:
                    factor_groups = np.array([factor_model.factor_names])

        a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
            groups=groups,
            equations=linear_constraints,
            raise_if_group_missing=False,
            loading_matrix=loading_matrix,
            factor_groups=factor_groups,
        )
        if len(a_eq) != 0:
            constraints.append(a_eq @ w * scale - b_eq * factor * scale == 0)
        if len(a_ineq) != 0:
            constraints.append(a_ineq @ w * scale - b_ineq * factor * scale <= 0)

    return constraints
