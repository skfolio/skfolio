"""Deprecated optimizer imports. Use :mod:`skfolio.optimization.hierarchical`."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skfolio.optimization._base import BaseOptimization
    from skfolio.optimization.cluster.hierarchical._base import (
        BaseHierarchicalOptimization,
    )
    from skfolio.optimization.hierarchical import (
        HierarchicalEqualRiskContribution,
        HierarchicalRiskParity,
        NestedClustersOptimization,
        SchurComplementary,
    )

__all__ = [
    "BaseHierarchicalOptimization",
    "HierarchicalEqualRiskContribution",
    "HierarchicalRiskParity",
    "NestedClustersOptimization",
    "SchurComplementary",
]


# TODO remove this deprecated module in v2.0
def __getattr__(name: str) -> type[BaseOptimization]:
    """Resolve the supported legacy optimizer exports."""
    if name in __all__:
        return _get_deprecated_estimator(name, __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# TODO remove deprecated optimizer imports in v2.0
def _get_deprecated_estimator(name: str, module: str) -> type[BaseOptimization]:
    """Resolve a legacy export and warn with its replacement import.

    Parameters
    ----------
    name : str
        Exported estimator name, validated by the calling module.

    module : str
        Deprecated public import location.

    Returns
    -------
    estimator : type[BaseOptimization]
        The estimator class exposed by the deprecated import.
    """
    if name == "BaseHierarchicalOptimization":
        from skfolio.optimization.cluster.hierarchical._base import (
            BaseHierarchicalOptimization,
        )

        estimator = BaseHierarchicalOptimization
        replacement = "Inherit from `skfolio.optimization.BaseOptimization` for custom optimizers."
    else:
        from skfolio.optimization import hierarchical

        estimator = getattr(hierarchical, name)
        replacement = (
            f"Import `{name}` from `skfolio.optimization.hierarchical` instead."
        )
    warnings.warn(
        f"`{module}.{name}` is deprecated and will be removed in version 2.0. "
        f"{replacement}",
        FutureWarning,
        stacklevel=3,
    )
    return estimator
