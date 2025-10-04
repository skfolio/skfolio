"""Internal constants and enums used across skfolio."""

# Copyright (c) 2025
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from enum import Enum


class _ParamKey(str, Enum):
    """Shared parameter keys used by estimators and portfolio.

    These names are passed as keyword parameters between optimization
    estimators and `Portfolio` classes to ensure consistent behavior.
    """

    TRANSACTION_COSTS = "transaction_costs"
    MANAGEMENT_FEES = "management_fees"
    PREVIOUS_WEIGHTS = "previous_weights"
    RISK_FREE_RATE = "risk_free_rate"

    def __str__(self) -> str:
        return self.value
