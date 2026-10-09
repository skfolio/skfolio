"""Newly weighted risks flow through hierarchical risk evaluation and allocation."""

import numpy as np
import pytest

import skfolio.measures as mt
from skfolio import ExtraRiskMeasure, RiskMeasure
from skfolio.optimization import (
    HierarchicalEqualRiskContribution,
    HierarchicalRiskParity,
)


@pytest.mark.parametrize(
    "optimizer", [HierarchicalRiskParity, HierarchicalEqualRiskContribution]
)
@pytest.mark.parametrize(
    "risk_measure",
    [
        RiskMeasure.EVAR,
        RiskMeasure.AVERAGE_DRAWDOWN,
        RiskMeasure.ULCER_INDEX,
        RiskMeasure.CDAR,
        RiskMeasure.EDAR,
        ExtraRiskMeasure.FOURTH_LOWER_PARTIAL_MOMENT,
    ],
)
def test_hierarchical_weighted_risks_and_allocations(
    fixed_return_distribution_prior, optimizer, risk_measure
):
    rng = np.random.default_rng(12)
    returns = rng.normal(0.001, 0.02, (100, 6))
    returns[:30, :3] *= 2
    covariance = np.cov(returns.T)
    expected_returns = returns.mean(axis=0)
    probabilities = np.r_[np.zeros(30), np.arange(1.0, 71.0)]
    probabilities /= probabilities.sum()
    allocations = []
    for sample_weight in [None, np.full(100, 0.01), probabilities]:
        model = optimizer(
            risk_measure=risk_measure,
            prior_estimator=fixed_return_distribution_prior(
                expected_returns, covariance, sample_weight
            ),
        ).fit(returns)
        distribution = model.prior_estimator_.return_distribution_
        # Keep the distance input fixed, so allocation changes come from risk weights.
        allocations.append(model.weights_)
        for allocation in [np.eye(6)[0], np.array([0.5, 0.5, 0, 0, 0, 0])]:
            series = returns @ allocation
            if risk_measure == RiskMeasure.AVERAGE_DRAWDOWN:
                cumulative = np.cumsum(series)
                losses = np.maximum.accumulate(np.r_[0, cumulative])[1:] - cumulative
                reference = (
                    losses.mean() if sample_weight is None else sample_weight @ losses
                )
            else:
                values = (
                    mt.get_drawdowns(series)
                    if risk_measure
                    in [RiskMeasure.ULCER_INDEX, RiskMeasure.CDAR, RiskMeasure.EDAR]
                    else series
                )
                reference = getattr(mt, risk_measure.value)(
                    values, sample_weight=sample_weight
                )
            np.testing.assert_allclose(
                model._risk(allocation, distribution), reference, rtol=1e-10, atol=1e-14
            )
        unitary = model._unitary_risks(distribution)
        np.testing.assert_allclose(unitary[0], model._risk(np.eye(6)[0], distribution))
    np.testing.assert_allclose(allocations[0], allocations[1], rtol=1e-8, atol=1e-10)
    assert not np.allclose(allocations[0], allocations[2], rtol=1e-4, atol=1e-5)
