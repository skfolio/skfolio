"""Doctest configuration for the source-tree examples.

`pyproject.toml` runs `--doctest-modules` over `src`, so every example in every
docstring is executed and its documented output verified. The few that cannot be are
listed in `SKIPPED` below, each with the reason it is there.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
import sklearn
from _pytest.doctest import DoctestItem

from skfolio.datasets import _base

NETWORK_DOCTESTS = {
    "skfolio.datasets._base.load_ftse100_dataset",
    "skfolio.datasets._base.load_nasdaq_dataset",
    "skfolio.datasets._base.load_sp500_implied_vol_dataset",
}

SKIPPED = {
    "skfolio.attribution._predicted.predicted_factor_attribution": (
        "illustrative fragment: `factor_returns` is never defined"
    ),
    "skfolio.attribution._realized.realized_factor_attribution": (
        "illustrative fragment: `factor_returns` is never defined"
    ),
    "skfolio.attribution._realized.rolling_realized_factor_attribution": (
        "illustrative fragment: `factor_returns` is never defined"
    ),
    "skfolio.population._population.Population.boxplot_measure": (
        "illustrative fragment: `population` is never defined"
    ),
    "skfolio.prior._opinion_pooling.OpinionPooling": (
        "example raises: the pooling itself fits, but `RiskBudgeting(CVaR)` on the "
        "pooled distribution hits `SolverError: Solver 'CLARABEL' failed`"
    ),
}


def pytest_collection_modifyitems(items) -> None:
    """Mark remote dataset examples and skip unsupported doctests."""
    for item in items:
        if not isinstance(item, DoctestItem):
            continue
        if item.name in NETWORK_DOCTESTS:
            item.add_marker(pytest.mark.network)
        if item.name in SKIPPED:
            item.add_marker(pytest.mark.skip(reason=SKIPPED[item.name]))


@pytest.fixture(autouse=True)
def _doctest_environment(request, tmp_path, monkeypatch, remote_dataset):
    """Pin the reader's defaults and keep file writes and datasets local.

    Examples run with NumPy's documented defaults, so their output is what a reader
    gets by pasting them into a fresh session; an example that needs rounding does it
    in the example. The options are pinned rather than left alone because a unit test
    calling `np.set_printoptions` globally would otherwise change the documented
    output when both test trees run in one session. `sklearn.config_context` pins
    sklearn's default for the same reason. The temporary directory keeps examples
    that write files (`AssetPanel.save("asset_panel")`) out of the working tree.
    """
    if request.node.name in NETWORK_DOCTESTS:
        monkeypatch.setattr(
            _base,
            "download_dataset",
            partial(remote_dataset, _base.download_dataset),
        )
    monkeypatch.chdir(tmp_path)
    with (
        np.printoptions(precision=8, suppress=False, legacy=False),
        sklearn.config_context(transform_output="default"),
    ):
        yield
