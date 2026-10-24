"""Deprecated optimizer imports. Use :mod:`skfolio.optimization.hierarchical`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from skfolio.optimization.cluster import _get_deprecated_estimator

if TYPE_CHECKING:
    from skfolio.optimization._base import BaseOptimization
    from skfolio.optimization.cluster.hierarchical._base import (
        BaseHierarchicalOptimization,
    )
    from skfolio.optimization.hierarchical import (
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
        SchurComplementary,
    )

__all__ = [
    "BaseHierarchicalOptimization",
    "HierarchicalEqualRiskContribution",
    "HierarchicalRiskParity",
    "SchurComplementary",
]


# TODO remove this deprecated module in v2.0
def __getattr__(name: str) -> type[BaseOptimization]:
    """Resolve the supported legacy hierarchical optimizer exports."""
    if name in __all__:
        return _get_deprecated_estimator(name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
