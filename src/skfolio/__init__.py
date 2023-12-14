"""skfolio package"""

# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
import importlib.metadata

from skfolio.measures import (
    BaseMeasure,
    ExtraRiskMeasure,
    PerfMeasure,
    RatioMeasure,
    RiskMeasure,
)
from skfolio.population import Population
from skfolio.portfolio import BasePortfolio, MultiPeriodPortfolio, Portfolio

__version__ = importlib.metadata.version("skfolio")

__all__ = [
    "BaseMeasure",
    "PerfMeasure",
    "RiskMeasure",
    "ExtraRiskMeasure",
    "RatioMeasure",
    "BasePortfolio",
    "Portfolio",
    "MultiPeriodPortfolio",
    "Population",
]
