"""FXMacroData release-calendar dataset loader."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


def load_fxmacrodata_release_calendar(
    currency: str = "usd",
    limit: int = 100,
    min_tier: int | None = None,
    api_key: str | None = None,
    base_url: str = FXMACRODATA_BASE_URL,
) -> pd.DataFrame:
    """Load FXMacroData economic release-calendar events.

    Parameters
    ----------
    currency : str, default="usd"
        Three-letter currency code.
    limit : int, default=100
        Maximum number of events to return.
    min_tier : int, optional
        Optional maximum ``market_tier`` to keep.
    api_key : str, optional
        FXMacroData API key. Defaults to ``FXMACRODATA_API_KEY``.
    base_url : str, default="https://fxmacrodata.com/api/v1"
        FXMacroData API base URL.

    Returns
    -------
    pd.DataFrame
        Calendar events indexed by release date.
    """
    limit = max(1, int(limit))
    params = {"limit": limit}
    token = api_key or os.environ.get("FXMACRODATA_API_KEY")
    if token:
        params["api_key"] = token

    url = f"{base_url.rstrip('/')}/calendar/{currency.lower()}?{urlencode(params)}"
    with urlopen(url, timeout=30) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))

    events = payload.get("data", [])
    if min_tier is not None:
        events = [
            event
            for event in events
            if int(event.get("market_tier") or 99) <= int(min_tier)
        ]

    frame = pd.DataFrame(events[:limit])
    if frame.empty:
        return frame

    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.set_index("date").sort_index()
    if "announcement_datetime" in frame.columns:
        frame["announcement_datetime"] = pd.to_datetime(
            frame["announcement_datetime"], unit="s", utc=True, errors="coerce"
        )

    return frame
