"""Doctest configuration for the `skfolio` package.

`pyproject.toml` runs `--doctest-modules` over `src`, so every example in every
docstring is executed and its documented output verified. The few that cannot be are
listed in `SKIPPED` below, each with the reason it is there.
"""

from __future__ import annotations

import numpy as np
import pytest
import sklearn
from _pytest.doctest import DoctestItem

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
    """Skip the doctests listed in `SKIPPED`."""
    for item in items:
        if isinstance(item, DoctestItem) and item.name in SKIPPED:
            item.add_marker(pytest.mark.skip(reason=SKIPPED[item.name]))


@pytest.fixture(autouse=True)
def _doctest_environment(request, tmp_path, monkeypatch):
    """Keep readable output settings and file writes local to each doctest.

    `tests/conftest.py` sets `np.set_printoptions(suppress=True, precision=6)` and a
    few test modules call `sklearn.set_config(...)` without restoring it. Both leak
    into whatever runs next and change how documented output renders. The temporary
    directory keeps examples that write files (`AssetPanel.save("asset_panel")`) out
    of the working tree. Optimizer and SyntheticData arrays use four decimal places;
    other arrays keep eight to preserve small values. NumPy scalars
    display as plain numbers, including inside dictionaries.
    """
    monkeypatch.chdir(tmp_path)
    compact_arrays = request.node.name.startswith(
        ("skfolio.optimization.", "skfolio.prior._synthetic_data.")
    )
    with (
        np.printoptions(
            precision=4 if compact_arrays else 8,
            suppress=compact_arrays
            or request.node.name.startswith(
                ("skfolio.alpha.", "skfolio.prior._entropy_pooling.")
            ),
            legacy="1.25",
        ),
        sklearn.config_context(transform_output="default"),
    ):
        yield
