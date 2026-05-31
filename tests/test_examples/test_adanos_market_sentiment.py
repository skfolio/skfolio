"""Tests for the Adanos market sentiment example."""

from __future__ import annotations

import runpy
from pathlib import Path

import pandas as pd


def test_adanos_market_sentiment_example_runs(monkeypatch):
    monkeypatch.setenv("ADANOS_USE_LIVE_API", "0")

    example = (
        Path(__file__).parents[2]
        / "examples"
        / "mean_risk"
        / "plot_18_adanos_market_sentiment.py"
    )
    namespace = runpy.run_path(str(example))

    assert not namespace["sentiment"].empty
    assert len(namespace["black_litterman_views"]) == 7
    assert namespace["entropy_pooling_views"] == ["AMD >= GE"]
    assert namespace["weights"].shape == (7, 3)

    custom_sentiment = namespace["sentiment_frame"](
        [
            {"symbol": "amd", "sentiment": 0.5},
            {"symbol": "GE", "sentiment": "not available"},
        ],
        pd.Index(["AMD", "GE"]),
    )
    assert custom_sentiment.loc["AMD", "sentiment_score"] == 0.5
    assert "GE" not in custom_sentiment.index
