"""Public exports for downside risk descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.factor_model.descriptor._downside_risk._ew_downside_beta import (
    EWDownsideBeta,
)

__all__ = [
    "EWDownsideBeta",
]
