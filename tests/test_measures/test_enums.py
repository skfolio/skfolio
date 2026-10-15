"""Tests for measure-enum relationships."""

import pytest

from skfolio import ExtraRiskMeasure, PerfMeasure, RiskMeasure


@pytest.mark.parametrize(
    ("measure", "annualized"),
    [
        (PerfMeasure.MEAN, PerfMeasure.ANNUALIZED_MEAN),
        (RiskMeasure.VARIANCE, RiskMeasure.ANNUALIZED_VARIANCE),
    ],
)
def test_measure_annualization_round_trip(measure, annualized):
    """Measures map to and from their annualized counterparts."""
    assert measure.annualized_measure is annualized
    assert annualized.non_annualized_measure is measure


def test_measure_rejects_redundant_annualization_conversion():
    """Conversions reject measures already in the requested form."""
    with pytest.raises(ValueError, match="already an annualized measure"):
        _ = PerfMeasure.ANNUALIZED_MEAN.annualized_measure

    with pytest.raises(ValueError, match="already a non-annualized measure"):
        _ = PerfMeasure.MEAN.non_annualized_measure


def test_measure_without_annualized_counterpart_raises():
    """Measures without an annualized definition report that explicitly."""
    with pytest.raises(AttributeError, match="doesn't have an annualized version"):
        _ = ExtraRiskMeasure.SKEW.annualized_measure


def test_extra_risk_measure_classification():
    """Extra risk measures retain the common measure classification contract."""
    assert ExtraRiskMeasure.SKEW.is_risk
    assert not ExtraRiskMeasure.SKEW.is_perf
    assert not ExtraRiskMeasure.SKEW.is_ratio


def test_measure_enum_membership_by_value():
    """Measure enums expose value-based membership checks."""
    assert RiskMeasure.has("variance")
    assert not RiskMeasure.has("not_a_measure")
