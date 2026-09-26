"""Factor exposure transformers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.factor_exposure._base import BaseFactorExposure
from skfolio.factor_exposure._derived_factor import DerivedFactor
from skfolio.factor_exposure._fixed_weighted_factor import (
    FixedWeightedFactor,
)
from skfolio.factor_exposure._global_factor import GlobalFactor
from skfolio.factor_exposure._one_hot_categorical_factors import (
    OneHotCategoricalFactors,
)

__all__ = [
    "BaseFactorExposure",
    "DerivedFactor",
    "FixedWeightedFactor",
    "GlobalFactor",
    "OneHotCategoricalFactors",
]
