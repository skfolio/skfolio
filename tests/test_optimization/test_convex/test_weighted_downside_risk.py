import numpy as np
import pytest
from sklearn.base import clone

import skfolio.measures as mt
from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction, RiskBudgeting


@pytest.fixture(params=[False, True], ids=["uniform", "nonuniform"])
def weighted_returns(request):
    rng = np.random.default_rng(42)
    returns = rng.normal(size=(50, 3)) * [0.008, 0.018, 0.04]
    returns += np.array([0.0005, 0.0015, 0.003]) - returns.mean(axis=0)
    weights = np.arange(1, 51, dtype=float) if request.param else np.ones(50)
    weights /= weights.sum()
    return returns, weights


@pytest.mark.parametrize(
    "risk_measure", [RiskMeasure.SEMI_VARIANCE, RiskMeasure.SEMI_DEVIATION]
)
@pytest.mark.parametrize(
    "model",
    [
        MeanRisk(),
        MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO),
        MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_UTILITY),
        RiskBudgeting(),
    ],
    ids=["minimize_risk", "maximize_ratio", "maximize_utility", "risk_budgeting"],
)
@pytest.mark.parametrize("target", [None, 0.001])
def test_weighted_downside_risk_value(
    weighted_returns, fixed_return_distribution_prior, risk_measure, model, target
):
    returns, weights = weighted_returns
    # Deliberately differ from the empirical mean to check prior-based centering.
    mu = np.array([0.001, 0.002, 0.004])
    model = clone(model).set_params(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            mu, np.cov(returns.T), sample_weight=weights
        ),
        min_acceptable_return=target,
        scale_objective=10,
        scale_constraints=100,
        solver_params={"tol_gap_abs": 1e-9, "tol_feas": 1e-9},
    )
    model.fit(returns)
    target = mu if target is None else target
    portfolio_target = np.broadcast_to(target, returns.shape[1]) @ model.weights_
    expected = getattr(mt, risk_measure.value)(
        returns @ model.weights_,
        sample_weight=weights,
        min_acceptable_return=portfolio_target,
    )
    np.testing.assert_allclose(model.problem_values_["risk"], expected, rtol=1e-5)


@pytest.mark.parametrize(
    "risk_measure,limit",
    [(RiskMeasure.SEMI_VARIANCE, 0.008**2), (RiskMeasure.SEMI_DEVIATION, 0.008)],
)
def test_weighted_downside_risk_binding_limit(
    weighted_returns, fixed_return_distribution_prior, risk_measure, limit
):
    returns, weights = weighted_returns
    mu = weights @ returns
    model = MeanRisk(
        risk_measure=risk_measure,
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        prior_estimator=fixed_return_distribution_prior(
            mu, np.cov(returns.T), sample_weight=weights
        ),
        scale_objective=100,
        scale_constraints=100,
        solver_params={"tol_gap_abs": 1e-9, "tol_feas": 1e-9},
        **{f"max_{risk_measure.value}": limit},
    )
    model.fit(returns)
    realized = getattr(mt, risk_measure.value)(
        returns @ model.weights_, sample_weight=weights
    )
    np.testing.assert_allclose(realized, limit, rtol=1e-5)

    if np.all(weights == weights[0]):
        model.set_params(prior_estimator=None).fit(returns)
        unweighted = getattr(mt, risk_measure.value)(returns @ model.weights_)
        np.testing.assert_allclose(unweighted, realized, rtol=1e-5)


@pytest.mark.parametrize(
    "risk_measure", [RiskMeasure.SEMI_VARIANCE, RiskMeasure.SEMI_DEVIATION]
)
@pytest.mark.parametrize("optimizer", [MeanRisk, RiskBudgeting])
def test_weighted_downside_risk_undefined_correction(
    fixed_return_distribution_prior, risk_measure, optimizer
):
    returns = np.array([[0.01, 0.02], [-0.01, -0.02]])
    model = optimizer(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            mu=returns[0],
            covariance=np.eye(2),
            sample_weight=np.array([1.0, 0.0]),
        ),
    )
    with pytest.raises(
        ValueError,
        match=r"correction 1 - sum\(sample_weight\*\*2\) "
        r"must be a positive number, got 0\.0$",
    ):
        model.fit(returns)
