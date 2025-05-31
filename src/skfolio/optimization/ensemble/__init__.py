"""Ensemble Optimization module."""

from skfolio.optimization.ensemble._stacking import StackingOptimization
from skfolio.utils.composition import BaseComposition

__all__ = ["BaseComposition", "StackingOptimization"]
