"""Optimization module."""

from typing import TYPE_CHECKING

from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.convex import (
    BenchmarkTracker,
    ConvexOptimization,
    DistributionallyRobustCVaR,
    MaximumDiversification,
    MeanRisk,
    ObjectiveFunction,
    RiskBudgeting,
)
from skfolio.optimization.ensemble import StackingOptimization
from skfolio.optimization.hierarchical import (
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
    NestedClustersOptimization,
    SchurComplementary,
)
from skfolio.optimization.naive import EqualWeighted, InverseVolatility, Random
from skfolio.optimization.online import BaseOnlineOptimization, ExponentiatedGradient

if TYPE_CHECKING:
    from skfolio.optimization.cluster.hierarchical._base import (
        BaseHierarchicalOptimization,
    )

__all__ = [
    "BaseHierarchicalOptimization",
    "BaseOnlineOptimization",
    "BaseOptimization",
    "BenchmarkTracker",
    "ConvexOptimization",
    "DistributionallyRobustCVaR",
    "EqualWeighted",
    "ExponentiatedGradient",
    "HierarchicalEqualRiskContribution",
    "HierarchicalRiskParity",
    "InverseVolatility",
    "MaximumDiversification",
    "MeanRisk",
    "NestedClustersOptimization",
    "ObjectiveFunction",
    "Random",
    "RiskBudgeting",
    "SchurComplementary",
    "StackingOptimization",
]


# TODO remove the deprecated BaseHierarchicalOptimization export in v2.0
def __getattr__(name: str) -> type[BaseOptimization]:
    """Resolve the deprecated hierarchical base export."""
    if name == "BaseHierarchicalOptimization":
        from skfolio.optimization.cluster import _get_deprecated_estimator

        return _get_deprecated_estimator(name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
