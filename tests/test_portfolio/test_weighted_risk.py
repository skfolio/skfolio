"""Weight forwarding, path preservation and rolling evaluation of new measures."""

import numpy as np
import pandas as pd
import pytest

import skfolio.measures as mt
from skfolio import (
    BasePortfolio,
    ExtraRiskMeasure,
    MultiPeriodPortfolio,
    Portfolio,
    RiskMeasure,
)

NEW_MEASURES = [
    RiskMeasure.EVAR,
    ExtraRiskMeasure.DRAWDOWN_AT_RISK,
    RiskMeasure.AVERAGE_DRAWDOWN,
    RiskMeasure.ULCER_INDEX,
    RiskMeasure.CDAR,
    RiskMeasure.EDAR,
    ExtraRiskMeasure.FOURTH_LOWER_PARTIAL_MOMENT,
]


def test_empty_portfolio_accepts_empty_sample_weight():
    portfolio = Portfolio(np.empty((0, 2)), weights=[0.5, 0.5], sample_weight=[])
    assert portfolio.sample_weight.shape == (0,)
    assert np.isnan(portfolio.mean)
    assert np.isnan(portfolio.cvar)


@pytest.mark.parametrize("sample_weight", [None, [0, 0.3, 0.7]])
def test_portfolio_conditional_tail_at_confidence_one(sample_weight):
    portfolio = BasePortfolio(
        [-0.8, 0.4, -0.1],
        np.arange(3),
        sample_weight=sample_weight,
        cvar_beta=1,
        cdar_beta=1,
    )
    expected_cvar = 0.8 if sample_weight is None else 0.1
    expected_cdar = 0.8 if sample_weight is None else 0.5
    assert portfolio.cvar == pytest.approx(expected_cvar)
    assert portfolio.cdar == pytest.approx(expected_cdar)
    assert portfolio.max_drawdown == pytest.approx(0.8)


@pytest.mark.parametrize("measure", NEW_MEASURES)
@pytest.mark.parametrize("compounded", [False, True])
def test_weighted_rolling_uses_realized_returns(measure, compounded):
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.normal(0.005, 0.05, (12, 2)), index=np.repeat(np.arange(6), 2)
    )
    weights = np.arange(1.0, 13.0)
    weights /= weights.sum()
    portfolio = Portfolio(
        data,
        weights=[0.7, 0.3],
        weight_drift=True,
        sample_weight=weights,
        transaction_costs=0.001,
        management_fees=0.0001,
        compounded=compounded,
        min_acceptable_return=0.002,
        evar_beta=0.6,
        drawdown_at_risk_beta=0.6,
        cdar_beta=0.6,
        edar_beta=0.6,
    )
    rolling = portfolio.rolling_measure(measure, window=5, min_periods=3)
    drawdown_measure = measure not in [
        RiskMeasure.EVAR,
        ExtraRiskMeasure.FOURTH_LOWER_PARTIAL_MOMENT,
    ]
    for end in range(3, 13):
        start = max(0, end - 5)
        series = portfolio.returns[start:end]
        values = (
            mt.get_drawdowns(series, compounded=compounded)
            if drawdown_measure
            else series
        )
        kwargs = {}
        if measure in [
            RiskMeasure.EVAR,
            ExtraRiskMeasure.DRAWDOWN_AT_RISK,
            RiskMeasure.CDAR,
            RiskMeasure.EDAR,
        ]:
            kwargs["beta"] = 0.6
        if measure == ExtraRiskMeasure.FOURTH_LOWER_PARTIAL_MOMENT:
            kwargs["min_acceptable_return"] = 0.002
        expected = getattr(mt, measure.value)(
            values, sample_weight=weights[start:end], **kwargs
        )
        np.testing.assert_allclose(
            rolling.iloc[end - 1], expected, rtol=1e-12, atol=1e-14
        )
    assert rolling.iloc[:2].isna().all()
    np.testing.assert_array_equal(rolling.index, data.index)
    # Rebuilding a portfolio would reset the holdings at the window boundary.
    rebuilt = Portfolio(data.iloc[-5:], weights=[0.7, 0.3], weight_drift=True)
    assert not np.allclose(rebuilt.returns, portfolio.returns[-5:])


@pytest.mark.parametrize("measure", NEW_MEASURES)
def test_weighted_rolling_recovers_after_zero_mass(measure):
    portfolio = BasePortfolio(
        [-0.2, 0.1, 0.05, -0.03, 0.02],
        np.arange(5),
        sample_weight=[0, 0, 0, 0.4, 0.6],
    )
    result = portfolio.rolling_measure(measure, window=3)
    assert np.isnan(result.iloc[2])
    assert result.iloc[3:].notna().all()
    assert np.isfinite(
        portfolio.rolling_measure(RiskMeasure.MAX_DRAWDOWN, window=3).iloc[2]
    )


def test_weighted_drawdown_ratios_and_cache():
    portfolio = BasePortfolio(
        [-0.2, 0.1, 0.05, -0.03],
        np.arange(4),
        sample_weight=[0, 0.3, 0.3, 0.4],
        risk_free_rate=0.001,
    )
    for risk in ["evar", "average_drawdown", "ulcer_index", "cdar", "edar"]:
        ratio_name = f"{risk}_ratio"
        np.testing.assert_allclose(
            getattr(portfolio, ratio_name),
            (portfolio.mean - portfolio.risk_free_rate) / getattr(portfolio, risk),
        )
    np.testing.assert_allclose(
        portfolio.calmar_ratio, (portfolio.mean - portfolio.risk_free_rate) / 0.2
    )
    previous = portfolio.average_drawdown
    portfolio.sample_weight = [0.7, 0.1, 0.1, 0.1]
    assert portfolio.average_drawdown != previous
    assert portfolio.max_drawdown == 0.2


@pytest.mark.parametrize("measure", NEW_MEASURES)
def test_weighted_multi_period_inheritance_and_override(measure):
    data = pd.DataFrame(
        np.array(
            [[-0.2, 0.1], [0.1, -0.05], [0.05, 0.02], [-0.03, 0.04], [0.02, -0.01]]
        ),
        index=pd.date_range("2020-01-01", periods=5),
    )
    children = [
        Portfolio(data.iloc[:2], [0.5, 0.5], sample_weight=[0.1, 0.9]),
        Portfolio(data.iloc[2:], [0.4, 0.6]),
    ]
    portfolio = MultiPeriodPortfolio(children)
    np.testing.assert_allclose(portfolio.sample_weight, [0.04, 0.36, 0.2, 0.2, 0.2])
    for override in [None, np.array([0, 0, 0, 0.3, 0.7]), None]:
        portfolio.sample_weight = override
        reference = BasePortfolio(
            portfolio.returns,
            portfolio.observations,
            sample_weight=portfolio.sample_weight,
        )
        np.testing.assert_allclose(
            portfolio.get_measure(measure), reference.get_measure(measure), atol=1e-14
        )
        np.testing.assert_allclose(
            portfolio.rolling_measure(measure, 3),
            reference.rolling_measure(measure, 3),
            atol=1e-14,
        )
