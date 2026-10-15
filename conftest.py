"""Shared pytest configuration for unit tests and doctests."""

from __future__ import annotations

from collections.abc import Callable
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest


@pytest.fixture(scope="session")
def remote_dataset() -> Callable[..., pd.DataFrame]:
    """Use the loader's cache and skip when an uncached dataset is unreachable."""

    def _load(loader: Callable[..., pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
        try:
            return loader(*args, **kwargs)
        except HTTPError:
            # HTTPError subclasses URLError, but HTTP failures should fail the test.
            raise
        except (URLError, TimeoutError) as exc:
            pytest.skip(f"{loader.__name__} is not cached and unreachable: {exc}")

    return _load
