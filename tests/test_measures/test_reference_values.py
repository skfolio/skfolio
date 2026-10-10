"""
Reference values for skfolio's risk measures.

Each measure is checked against the convention skfolio implements, on five
fixed return series: the 24-month portfolio from Bacon (2008), a seven-point
sample, a Gaussian series, a fat-tailed Student t(3) series and a left-skewed
one. Where R PerformanceAnalytics implements the convention, the expected
value is its output (the call is recorded in data/reference_values.json); the
rest were checked against numpy, scipy or a brute-force definition. The values
come from vetted (https://github.com/WatchTree-19/vetted).

A failure here means a measure changed. If the change was intended, the
convention recorded for that measure should change with it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import skfolio.measures as skm

DATA = json.loads(
    (Path(__file__).parent / "data" / "reference_values.json").read_text()
)

MEASURES = {
    "max_drawdown": lambda r: skm.max_drawdown(skm.get_drawdowns(r, compounded=True)),
    "value_at_risk_95": lambda r: skm.value_at_risk(r, beta=0.95),
    "cvar_95": lambda r: skm.cvar(r, beta=0.95),
    "ulcer_index": lambda r: skm.ulcer_index(skm.get_drawdowns(r, compounded=True)),
    "skew": lambda r: skm.skew(r),
    "kurtosis": lambda r: skm.kurtosis(r),
}

CASES = [
    pytest.param(measure, fixture, expected, id=f"{measure}-{fixture}")
    for measure, spec in DATA["measures"].items()
    for fixture, expected in spec["values"].items()
]


def test_every_measure_has_a_check():
    assert set(MEASURES) == set(DATA["measures"])


@pytest.mark.parametrize("measure, fixture, expected", CASES)
def test_reference_value(measure, fixture, expected):
    returns = np.asarray(DATA["fixtures"][fixture]["returns"], dtype=float)
    value = float(MEASURES[measure](returns))
    convention = DATA["measures"][measure]["convention"]
    assert value == pytest.approx(expected, rel=1e-9, abs=1e-12), (
        f"{measure} on {fixture} no longer matches the {convention} convention"
    )
