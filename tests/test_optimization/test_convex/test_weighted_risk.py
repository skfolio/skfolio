"""Compare weighted convex risk expressions with independent numerical references."""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from sklearn.base import clone

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction, RiskBudgeting

RISK_MEASURES = [
    RiskMeasure.EVAR,
    RiskMeasure.AVERAGE_DRAWDOWN,
    RiskMeasure.ULCER_INDEX,
    RiskMeasure.CDAR,
    RiskMeasure.EDAR,
]


def _risk_reference(returns, weights, risk_measure, beta=0.8):
    if weights is None:
        weights = np.full(len(returns), 1 / len(returns))
    if risk_measure in [RiskMeasure.CVAR, RiskMeasure.EVAR]:
        losses = -returns
    else:
        cumulative = np.cumsum(returns)
        losses = np.maximum.accumulate(np.r_[0, cumulative])[1:] - cumulative
    positive = weights > 0
    losses, weights = losses[positive], weights[positive]
    weights = weights / weights.sum()
    if risk_measure == RiskMeasure.AVERAGE_DRAWDOWN or beta == 0:
        return weights @ losses
    if risk_measure == RiskMeasure.ULCER_INDEX:
        return np.sqrt(weights @ losses**2)
    if beta == 1:
        return losses.max()
    if risk_measure in [RiskMeasure.CVAR, RiskMeasure.CDAR]:
        remaining = 1 - beta
        weighted_losses = 0.0
        for index in np.argsort(-losses):
            mass = min(remaining, weights[index])
            weighted_losses += mass * losses[index]
            remaining -= mass
        return weighted_losses / (1 - beta)
    if weights[losses == losses.max()].sum() >= 1 - beta:
        return losses.max()

    def objective(log_temperature):
        temperature = np.exp(log_temperature)
        return temperature * (
            logsumexp(np.log(weights) + losses / temperature) - np.log1p(-beta)
        )

    result = minimize_scalar(objective, bounds=(-30, 10), method="bounded")
    assert result.success
    return result.fun


@pytest.fixture(params=["uniform", "nonuniform", "zeros"])
def weighted_scenarios(request):
    rng = np.random.default_rng(42)
    returns = rng.normal(size=(50, 3)) * [0.01, 0.02, 0.04]
    returns += np.array([0.001, 0.003, 0.006]) - returns.mean(axis=0)
    weights = (
        np.ones(50) if request.param == "uniform" else np.arange(1, 51, dtype=float)
    )
    if request.param == "zeros":
        weights[::3] = 0
    weights /= weights.sum()
    return returns, weights


@pytest.mark.parametrize("risk_measure", RISK_MEASURES)
@pytest.mark.parametrize("fee", [0.0, 0.001])
def test_weighted_fixed_allocation_risk(
    weighted_scenarios, fixed_return_distribution_prior, risk_measure, fee
):
    returns, weights = weighted_scenarios
    allocation = np.array([0.2, 0.3, 0.5])
    model = MeanRisk(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            weights @ returns, np.cov(returns.T), weights
        ),
        min_weights=allocation,
        max_weights=allocation,
        management_fees=fee,
        portfolio_params={"evar_beta": 0.8, "cdar_beta": 0.8, "edar_beta": 0.8},
        evar_beta=0.8,
        cdar_beta=0.8,
        edar_beta=0.8,
        solver_params={"tol_gap_abs": 1e-9, "tol_feas": 1e-9},
    ).fit(returns)
    expected = _risk_reference(returns @ allocation - fee, weights, risk_measure)
    np.testing.assert_allclose(
        model.problem_values_["risk"], expected, rtol=2e-5, atol=2e-8
    )
    portfolio = model.predict(model.prior_estimator_.return_distribution_)
    np.testing.assert_allclose(
        getattr(portfolio, risk_measure.value), expected, rtol=2e-5, atol=2e-8
    )


def test_weighted_evar_preserves_transaction_cost_relaxation(
    weighted_scenarios, fixed_return_distribution_prior
):
    returns, weights = weighted_scenarios
    allocation = np.array([0.2, 0.3, 0.5])
    model = MeanRisk(
        risk_measure=RiskMeasure.EVAR,
        prior_estimator=fixed_return_distribution_prior(
            weights @ returns, np.cov(returns.T), weights
        ),
        min_weights=allocation,
        max_weights=allocation,
        management_fees=0.001,
        transaction_costs=0.002,
        previous_weights=np.zeros(3),
        evar_beta=0.8,
        solver_params={"tol_gap_abs": 1e-9, "tol_feas": 1e-9},
    )
    with pytest.warns(UserWarning, match="EVaR problem will be relaxed"):
        model.fit(returns)
    expected = _risk_reference(returns @ allocation - 0.001, weights, RiskMeasure.EVAR)
    np.testing.assert_allclose(
        model.problem_values_["risk"], expected, rtol=2e-5, atol=2e-8
    )


@pytest.mark.parametrize("risk_measure", RISK_MEASURES)
@pytest.mark.parametrize(
    "model",
    [
        MeanRisk(),
        MeanRisk(objective_function=ObjectiveFunction.MAXIMIZE_RATIO),
        RiskBudgeting(),
    ],
    ids=["minimize_risk", "maximize_ratio", "risk_budgeting"],
)
def test_weighted_risk_objectives(
    weighted_scenarios, fixed_return_distribution_prior, risk_measure, model
):
    returns, weights = weighted_scenarios
    model = (
        clone(model)
        .set_params(
            risk_measure=risk_measure,
            prior_estimator=fixed_return_distribution_prior(
                weights @ returns, np.cov(returns.T), weights
            ),
            evar_beta=0.8,
            cdar_beta=0.8,
            edar_beta=0.8,
            solver_params={"tol_gap_abs": 1e-9, "tol_feas": 1e-9},
        )
        .fit(returns)
    )
    expected = _risk_reference(returns @ model.weights_, weights, risk_measure)
    np.testing.assert_allclose(
        model.problem_values_["risk"], expected, rtol=2e-5, atol=2e-8
    )


@pytest.mark.parametrize("risk_measure", RISK_MEASURES)
def test_weighted_risk_binding_limit(
    weighted_scenarios, fixed_return_distribution_prior, risk_measure
):
    returns, weights = weighted_scenarios
    expected_returns = weights @ returns
    model = MeanRisk(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            expected_returns, np.cov(returns.T), weights
        ),
        evar_beta=0.8,
        cdar_beta=0.8,
        edar_beta=0.8,
        solver_params={"tol_gap_abs": 1e-9, "tol_feas": 1e-9},
    ).fit(returns)
    minimum = _risk_reference(returns @ model.weights_, weights, risk_measure)
    unconstrained = _risk_reference(
        returns[:, np.argmax(expected_returns)], weights, risk_measure
    )
    limit = (minimum + unconstrained) / 2
    model.set_params(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        **{f"max_{risk_measure.value}": limit},
    ).fit(returns)
    actual = _risk_reference(returns @ model.weights_, weights, risk_measure)
    np.testing.assert_allclose(actual, limit, rtol=1e-4, atol=2e-8)


@pytest.mark.parametrize("risk_measure", [RiskMeasure.EVAR, RiskMeasure.EDAR])
def test_entropic_cone_tiny_probability(fixed_return_distribution_prior, risk_measure):
    returns = np.array(
        [[-1.0, -1.0], [0.0, 0.0] if risk_measure == RiskMeasure.EVAR else [1.0, 1.0]]
    )
    weights = np.array([1e-150, 1.0])
    model = MeanRisk(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            returns.mean(axis=0), np.eye(2), weights
        ),
        min_weights=[0.5, 0.5],
        max_weights=[0.5, 0.5],
        solver_params={"tol_gap_abs": 1e-10, "tol_feas": 1e-10},
    ).fit(returns)
    expected = _risk_reference(returns[:, 0], weights, risk_measure, beta=0.95)
    np.testing.assert_allclose(expected, 0.00881976004, atol=1e-10)
    np.testing.assert_allclose(
        model.problem_values_["risk"], expected, rtol=1e-5, atol=1e-9
    )


@pytest.mark.parametrize(
    "risk_measure",
    [*RISK_MEASURES, RiskMeasure.MAX_DRAWDOWN, RiskMeasure.WORST_REALIZATION],
)
def test_zero_weight_loss_remains_in_drawdown_path(
    fixed_return_distribution_prior, risk_measure
):
    returns = np.tile([-0.2, 0.1, 0.1], (2, 1)).T
    weights = np.array([0.0, 0.5, 0.5])
    model = MeanRisk(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            returns.mean(axis=0), np.eye(2), weights
        ),
        min_weights=[0.5, 0.5],
        max_weights=[0.5, 0.5],
        evar_beta=0.8,
        cdar_beta=0.8,
        edar_beta=0.8,
    ).fit(returns)
    expected = (
        0.2
        if risk_measure in [RiskMeasure.MAX_DRAWDOWN, RiskMeasure.WORST_REALIZATION]
        else _risk_reference(returns[:, 0], weights, risk_measure)
    )
    np.testing.assert_allclose(model.problem_values_["risk"], expected, atol=1e-7)


@pytest.mark.parametrize(
    "risk_measure",
    [RiskMeasure.CVAR, RiskMeasure.EVAR, RiskMeasure.CDAR, RiskMeasure.EDAR],
)
@pytest.mark.parametrize("beta", [0.0, 1.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_tail_risk_endpoints(
    weighted_scenarios, fixed_return_distribution_prior, risk_measure, beta, weighted
):
    returns, weights = weighted_scenarios
    if not weighted:
        weights = None
    allocation = np.full(3, 1 / 3)
    model = MeanRisk(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            returns.mean(axis=0), np.cov(returns.T), weights
        ),
        min_weights=allocation,
        max_weights=allocation,
        evar_beta=beta,
        edar_beta=beta,
        cvar_beta=beta,
        cdar_beta=beta,
    ).fit(returns)
    expected = _risk_reference(returns @ allocation, weights, risk_measure, beta=beta)
    np.testing.assert_allclose(model.problem_values_["risk"], expected, atol=1e-7)


@pytest.mark.parametrize("risk_measure", [RiskMeasure.CVAR, RiskMeasure.CDAR])
@pytest.mark.parametrize("fee", [0, 0.001])
def test_conditional_endpoint_keeps_zero_weight_returns_in_drawdown_path(
    fixed_return_distribution_prior, risk_measure, fee
):
    returns = np.array([[-0.8, -0.7], [0.4, 0.3], [-0.1, -0.15]])
    sample_weight = np.array([0, 0.3, 0.7])
    allocation = np.array([0.5, 0.5])
    model = MeanRisk(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            returns.mean(axis=0), np.cov(returns.T), sample_weight
        ),
        min_weights=allocation,
        max_weights=allocation,
        cvar_beta=1,
        cdar_beta=1,
        transaction_costs=fee,
        management_fees=fee,
        previous_weights=[0, 0],
    ).fit(returns)
    # The first loss is excluded from the tail, but remains in the drawdown path.
    expected = 0.125 + 2 * fee if risk_measure == RiskMeasure.CVAR else 0.525 + 6 * fee
    np.testing.assert_allclose(model.problem_values_["risk"], expected, atol=1e-7)


@pytest.mark.parametrize("optimizer", [MeanRisk, RiskBudgeting])
@pytest.mark.parametrize("risk_measure", [RiskMeasure.CVAR, RiskMeasure.CDAR])
def test_optimized_conditional_endpoint(
    weighted_scenarios, fixed_return_distribution_prior, optimizer, risk_measure
):
    returns, sample_weight = weighted_scenarios
    model = optimizer(
        risk_measure=risk_measure,
        prior_estimator=fixed_return_distribution_prior(
            returns.mean(axis=0), np.cov(returns.T), sample_weight
        ),
        cvar_beta=1,
        cdar_beta=1,
    ).fit(returns)
    expected = _risk_reference(
        returns @ model.weights_, sample_weight, risk_measure, beta=1
    )
    np.testing.assert_allclose(model.problem_values_["risk"], expected, atol=1e-7)
