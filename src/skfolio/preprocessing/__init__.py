"""Preprocessing module."""

from skfolio.preprocessing._returns import prices_to_returns
from skfolio.preprocessing._transformer._cross_sectional import (
    BaseCSTransformer,
    CSGaussianRankScaler,
    CSPercentileRankScaler,
    CSStandardScaler,
    CSTanhShrinker,
    CSWinsorizer,
)

__all__ = [
    "BaseCSTransformer",
    "CSGaussianRankScaler",
    "CSPercentileRankScaler",
    "CSStandardScaler",
    "CSTanhShrinker",
    "CSWinsorizer",
    "prices_to_returns",
]
