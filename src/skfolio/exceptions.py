"""
The :mod:`skfolio.exceptions` module includes all custom warnings and error
classes used across skfolio.
"""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import cvxpy as cp

__all__ = [
    "ConvexOptimizationError",
    "DuplicateGroupsError",
    "EquationToMatrixError",
    "FactorNotFoundError",
    "GroupNotFoundError",
    "NonPositiveVarianceError",
    "OptimizationError",
    "SkfolioError",
    "SolverError",
]


class SkfolioError(Exception):
    """Base class for all custom skfolio exceptions."""


class OptimizationError(SkfolioError):
    """An optimizer could not compute a valid portfolio allocation.

    See :ref:`optimization_failure_handling` for fallback behavior and
    :ref:`online_failure_handling` for online continuation and restart rules.
    """


class ConvexOptimizationError(OptimizationError, cp.SolverError):
    """A convex optimization step could not produce valid portfolio weights.

    This includes infeasible problems and unusable numerical results. It can be caught
    as `OptimizationError` or `cvxpy.SolverError`. See
    :ref:`optimization_failure_handling`.
    """


class SolverError(SkfolioError):
    """A solver failed during model estimation, such as entropy pooling.

    Portfolio optimization failures use `OptimizationError` instead.
    """


class EquationToMatrixError(SkfolioError):
    """Error while processing equations."""


class GroupNotFoundError(SkfolioError):
    """Group name not found in the groups."""


class FactorNotFoundError(SkfolioError):
    """Factor name not found in factor_groups or loading_matrix not provided."""


class DuplicateGroupsError(SkfolioError):
    """Group name appear in multiple group levels."""


class NonPositiveVarianceError(SkfolioError):
    """Variance negative or null."""
