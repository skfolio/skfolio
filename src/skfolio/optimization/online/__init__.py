"""Online portfolio optimization using shared algorithm families.

Adapted from Carlo Nicolini's online framework in CarloNicolini/skfolio,
source revision c1bd1f168cf7d660c81f74710aadd9941e64a2c7, under BSD-3-Clause.
"""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# SPDX-License-Identifier: BSD-3-Clause

from skfolio.optimization.online._base import BaseOnlineOptimization
from skfolio.optimization.online._estimators import (
    AdaGrad,
    ExponentiatedGradient,
    OnlineMovingAverageReversion,
    PassiveAggressiveMeanReversion,
)
from skfolio.optimization.online._regret import RegretType, plot_regret_curve, regret

__all__ = [
    "AdaGrad",
    "BaseOnlineOptimization",
    "ExponentiatedGradient",
    "OnlineMovingAverageReversion",
    "PassiveAggressiveMeanReversion",
    "RegretType",
    "plot_regret_curve",
    "regret",
]
