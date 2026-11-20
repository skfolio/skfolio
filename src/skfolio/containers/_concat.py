"""Observation-axis concatenation of containers."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Iterable
from itertools import chain
from typing import cast, overload

import numpy as np

from skfolio.containers._asset_panel._base import _BaseAssetPanel
from skfolio.containers._asset_panel._concat import _concat_panels
from skfolio.containers._asset_panel._panel import AssetPanel
from skfolio.containers._graph._graph_series import GraphSeries
from skfolio.containers._utils import _validate_unique_observations

__all__ = ["concat"]


@overload
def concat(  # numpydoc ignore=GL08
    items: Iterable[_BaseAssetPanel], *, verify_observations: bool = False
) -> AssetPanel: ...


@overload
def concat(  # numpydoc ignore=GL08
    items: Iterable[GraphSeries], *, verify_observations: bool = False
) -> GraphSeries: ...


def concat(
    items: Iterable[_BaseAssetPanel] | Iterable[GraphSeries],
    *,
    verify_observations: bool = False,
) -> AssetPanel | GraphSeries:
    """Concatenate containers along the observation axis.

    Panels and panel views must have identical assets and field names in the same order,
    field types, field dtypes, `inactive_policy` values, categorical levels and 3D field
    metadata. Field arrays, observations, `active_mask` and `estimation_mask` are
    concatenated on axis 0.

    Graph series concatenate observation labels and graph references. Graphs may have
    different nodes, storage formats and dtypes. Repeated references share the original
    graph objects.

    Parameters
    ----------
    items : iterable of AssetPanel, AssetPanelView or GraphSeries
        Containers to concatenate, in the order they should appear in the result. The
        iterable must contain at least one object. Panels and panel views can be
        combined, but cannot be mixed with graph series.

    verify_observations : bool, default=False
        If True, raise an error when the concatenated observation labels contain
        duplicates. AssetPanel requires unique observations even when False; GraphSeries
        preserves repeated labels when False.

    Returns
    -------
    container : AssetPanel or GraphSeries
        Concatenated container. Panels and panel views produce an AssetPanel; graph
        series produce a GraphSeries.

    Raises
    ------
    TypeError
        If the inputs mix panels and graph series, contain unsupported objects, or have
        mismatched field types or dtypes.

    ValueError
        If the iterable is empty, panels have mismatched assets or field metadata, or
        duplicate observation labels are not allowed.

    Examples
    --------
    Concatenate two panels:

    >>> from skfolio.containers import AssetPanel, concat
    >>> panel_1 = AssetPanel(
    ...     fields={"returns": [[0.01, 0.02]]},
    ...     observations=["2024-01-01"],
    ...     asset_names=["A", "B"],
    ... )
    >>> panel_2 = AssetPanel(
    ...     fields={"returns": [[0.03, 0.04]]},
    ...     observations=["2024-01-02"],
    ...     asset_names=["A", "B"],
    ... )
    >>> concat([panel_1, panel_2])
    AssetPanel(n_observations=2, n_assets=2, n_fields=1)

    Concatenate graph series while preserving shared graphs:

    >>> from skfolio.containers import Graph, GraphSeries
    >>> graph = Graph([[1.0]], nodes=["A"])
    >>> first = GraphSeries([graph], observations=[0])
    >>> second = GraphSeries([graph, None], observations=[1, 2])
    >>> combined = concat([first, second])
    >>> combined
    GraphSeries(n_observations=3, n_graphs=1)
    >>> combined[0] is combined[1] is graph
    True
    """
    if isinstance(items, (_BaseAssetPanel, GraphSeries)):
        raise TypeError(
            f"items must be an iterable of containers, not a single {type(items).__name__}."
        )

    containers = list(items)
    if not containers:
        raise ValueError("items must contain at least one container.")

    first = containers[0]
    container_type = GraphSeries if isinstance(first, GraphSeries) else _BaseAssetPanel
    for position, item in enumerate(containers):
        if not isinstance(item, container_type):
            raise TypeError(
                "items must contain only AssetPanel/AssetPanelView instances or only "
                f"GraphSeries instances; item at position {position} has type "
                f"{type(item).__name__}."
            )

    observations = np.concatenate([item.observations for item in containers], axis=0)

    if verify_observations:
        _validate_unique_observations(observations)

    if isinstance(first, GraphSeries):
        return GraphSeries(
            chain.from_iterable(cast(list[GraphSeries], containers)),
            observations=observations,
        )

    return _concat_panels(
        panels=cast(list[_BaseAssetPanel], containers), observations=observations
    )
