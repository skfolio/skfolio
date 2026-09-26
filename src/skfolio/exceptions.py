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
    "SkfolioError",
    "SolverError",
]


class SkfolioError(Exception):
    """Base class for all custom skfolio exceptions."""


class OptimizationError(SkfolioError):
    """Optimization Did not converge."""


class SolverError(SkfolioError):
    """Solver error."""


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
