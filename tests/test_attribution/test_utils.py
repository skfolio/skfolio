"""Tests for skfolio.attribution._utils formatting helpers."""

import numpy as np

from skfolio.attribution._utils import (
    _format_ci,
    _format_decimal,
    _format_percent,
)


def test_format_percent():
    assert _format_percent(0.12345) == "12.35%"
    assert _format_percent(np.nan) == "NaN"


def test_format_decimal():
    assert _format_decimal(0.123456) == "0.1235"
    assert _format_decimal(0.5, decimals=1) == "0.5"
    assert _format_decimal(np.nan) == "NaN"


def test_format_ci():
    assert _format_ci(0.01, 0.02) == "[1.00%, 2.00%]"
    assert _format_ci(np.nan, 0.02) == ""
    assert _format_ci(0.01, np.nan) == ""
