"""
The :mod:`skfolio.exceptions` module includes all custom warnings and error
classes used across skfolio.
"""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

__all__ = [
    "DuplicateGroupsError",
    "EquationToMatrixError",
    "FactorNotFoundError",
    "GroupNotFoundError",
    "NonPositiveVarianceError",
    "OptimizationError",
    "SolverError",
]


class OptimizationError(Exception):
    """Optimization Did not converge."""


class SolverError(Exception):
    """Solver error."""


class EquationToMatrixError(Exception):
    """Error while processing equations."""


class GroupNotFoundError(Exception):
    """Group name not found in the groups."""


class FactorNotFoundError(Exception):
    """Factor name not found in factor_groups or loading_matrix not provided."""


class DuplicateGroupsError(Exception):
    """Group name appear in multiple group levels."""


class NonPositiveVarianceError(Exception):
    """Variance negative or null."""
