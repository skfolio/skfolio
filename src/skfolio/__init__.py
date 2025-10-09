"""skfolio package."""

# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
import importlib.metadata

from skfolio.measures import (
    BaseMeasure,
    ExtraRiskMeasure,
    PerfMeasure,
    RatioMeasure,
    RiskMeasure,
)
from skfolio.population import Population
from skfolio.portfolio import (
    BasePortfolio,
    FailedPortfolio,
    MultiPeriodPortfolio,
    Portfolio,
)

__version__ = importlib.metadata.version("skfolio")

__all__ = [
    "BaseMeasure",
    "BasePortfolio",
    "ExtraRiskMeasure",
    "FailedPortfolio",
    "MultiPeriodPortfolio",
    "PerfMeasure",
    "Population",
    "Portfolio",
    "RatioMeasure",
    "RiskMeasure",
]
