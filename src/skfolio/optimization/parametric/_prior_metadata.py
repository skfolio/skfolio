"""Align prior metadata to the exact causal covariance training window."""

from __future__ import annotations

import pandas as pd

from skfolio.containers import AssetPanel, AssetPanelView


def slice_prior_metadata(metadata: dict | None, index: pd.Index) -> dict:
    """Slice dated metadata; reject positional arrays and implicit filling."""
    result = {}
    for name, value in (metadata or {}).items():
        if isinstance(value, pd.Series | pd.DataFrame):
            dates = value.index
            if not dates.is_unique or not index.isin(dates).all():
                raise ValueError(
                    f"Prior metadata {name!r} must cover each history date uniquely."
                )
            result[name] = value.loc[index].copy()
        elif isinstance(value, AssetPanel | AssetPanelView):
            dates = pd.Index(value.observations)
            if not dates.is_unique or not index.isin(dates).all():
                raise ValueError(
                    f"Prior metadata {name!r} must cover each history date uniquely."
                )
            result[name] = value[dates.get_indexer(index)]
        else:
            raise ValueError(
                f"Prior metadata {name!r} must be a dated Series, DataFrame or AssetPanel."
            )
    return result
