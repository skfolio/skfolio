"""
The :mod:`skfolio.exceptions` module includes all custom warnings and error
classes used across skfolio.
"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "DuplicateGroupsError",
    "EquationToMatrixError",
    "GroupNotFoundError",
    "NonPositiveVarianceError",
    "OptimizationError",
]


class OptimizationError(Exception):
    """Optimization Did not converge."""


class EquationToMatrixError(Exception):
    """Error while processing equations."""


class GroupNotFoundError(Exception):
    """Group name not found in the groups."""


class DuplicateGroupsError(Exception):
    """Group name appear in multiple group levels."""


class NonPositiveVarianceError(Exception):
    """Variance negative or null."""
