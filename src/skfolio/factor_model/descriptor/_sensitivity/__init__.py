"""Public exports for sensitivity descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.factor_model.descriptor._sensitivity._ew_macro_sensitivity import (
    EWMacroSensitivity,
)
from skfolio.factor_model.descriptor._sensitivity._ew_market_beta import EWMarketBeta

__all__ = ["EWMacroSensitivity", "EWMarketBeta"]
