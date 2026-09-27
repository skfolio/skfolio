"""Tests for AssetPanel utilities."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from skfolio.containers._asset_panel._utils import _format_observation_range


class TestFormatObservationRange:
    def test_empty_and_single(self):
        assert _format_observation_range(np.array([])) == ""
        assert _format_observation_range(np.array([3])) == "  (3)"

    def test_datetime_labels(self):
        observations = pd.date_range("2020-01-01", periods=5).to_numpy()
        assert _format_observation_range(observations) == "  (2020-01-01 -> 2020-01-05)"
        observations = np.array(
            [dt.date(2021, 3, 1), dt.date(2021, 3, 9)], dtype=object
        )
        assert _format_observation_range(observations) == "  (2021-03-01 -> 2021-03-09)"

    def test_numeric_labels_are_not_parsed_as_timestamps(self):
        assert _format_observation_range(np.arange(4)) == "  (0 -> 3)"
        assert _format_observation_range(np.array([0.5, 2.5])) == "  (0.5 -> 2.5)"

    def test_unparsable_labels_fall_back_to_str(self):
        # ValueError from pd.Timestamp
        assert _format_observation_range(np.array(["a", "b", "c"])) == "  (a -> c)"
        # TypeError from pd.Timestamp
        observations = np.array([object(), object()], dtype=object)
        expected = f"  ({observations[0]} -> {observations[-1]})"
        assert _format_observation_range(observations) == expected
        # Dates pandas cannot represent (a ValueError subclass)
        observations = np.array(["30000-01-01", "30001-01-01"])
        with pytest.raises(ValueError):
            pd.Timestamp(str(observations[0]))
        assert (
            _format_observation_range(observations) == "  (30000-01-01 -> 30001-01-01)"
        )

    def test_unrelated_errors_propagate(self):
        """Only parsing failures are swallowed; a bug in a label is raised."""

        class Broken:
            def __str__(self):
                raise RuntimeError("boom")

        observations = np.array([Broken(), Broken()], dtype=object)
        with pytest.raises(RuntimeError, match="boom"):
            _format_observation_range(observations)
