"""Public exports for momentum descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.descriptor._momentum._ew_momentum import EWMomentum
from skfolio.descriptor._momentum._rolling_momentum import (
    RollingMomentum,
)

__all__ = [
    "EWMomentum",
    "RollingMomentum",
]
