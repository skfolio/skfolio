"""Public exports for growth base descriptors."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.factor_model.descriptor._growth._base._change_in_intensity import (
    ChangeInIntensity,
)
from skfolio.factor_model.descriptor._growth._base._change_to_scale import ChangeToScale
from skfolio.factor_model.descriptor._growth._base._growth_rate import GrowthRate

__all__ = ["ChangeInIntensity", "ChangeToScale", "GrowthRate"]
