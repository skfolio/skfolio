"""Doctest configuration for the source-tree examples."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
import sklearn
from _pytest.doctest import DoctestItem

from skfolio.datasets import _base

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pandas as pd

NETWORK_DOCTESTS = {
    "skfolio.datasets._base.load_ftse100_dataset",
    "skfolio.datasets._base.load_nasdaq_dataset",
    "skfolio.datasets._base.load_sp500_implied_vol_dataset",
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark doctests that require remote datasets."""
    for item in items:
        if isinstance(item, DoctestItem) and item.name in NETWORK_DOCTESTS:
            item.add_marker(pytest.mark.network)


@pytest.fixture(autouse=True)
def _doctest_environment(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    remote_dataset: Callable[..., pd.DataFrame],
):
    """Keep doctest settings, file writes, and remote dataset handling local.

    Unit tests configure NumPy output globally, so doctests use their own output
    settings to render consistently regardless of test order. The temporary
    directory keeps examples that write files (`AssetPanel.save("asset_panel")`) out
    of the working tree. Optimizer and SyntheticData arrays use four decimal places;
    other arrays keep eight to preserve small values. NumPy scalars
    display as plain numbers, including inside dictionaries.
    """
    name = request.node.name
    if name in NETWORK_DOCTESTS:
        monkeypatch.setattr(
            _base,
            "download_dataset",
            partial(remote_dataset, _base.download_dataset),
        )
    monkeypatch.chdir(tmp_path)
    compact_arrays = name.startswith(
        ("skfolio.optimization.", "skfolio.prior._synthetic_data.")
    )
    suppress_scientific = compact_arrays or name.startswith(
        ("skfolio.alpha.", "skfolio.prior._entropy_pooling.")
    )
    with (
        np.printoptions(
            precision=4 if compact_arrays else 8,
            suppress=suppress_scientific,
            legacy="1.25",
        ),
        sklearn.config_context(transform_output="default"),
    ):
        yield
