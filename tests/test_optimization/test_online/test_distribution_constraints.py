"""Distribution limits share convex definitions and consume only observed rows."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skfolio.measures import RiskMeasure
from skfolio.optimization.online import (
    AdaGrad,
    ExponentiatedGradient,
    OnlineMovingAverageReversion,
    PassiveAggressiveMeanReversion,
)
from skfolio.optimization.online._distribution_constraints import (
    RISK_LIMITS,
    DistributionConstraintConfig,
)
from skfolio.optimization.online._projection import AutoProjector, ProjectionConfig
from skfolio.prior import BasePrior, ReturnDistribution


@pytest.fixture
def distribution():
    returns = np.array([[-0.2, 0], [0.1, 0], [-0.15, 0], [0.05, 0]])
    return ReturnDistribution(
        mu=returns.mean(axis=0), covariance=np.cov(returns.T), returns=returns
    )


@pytest.mark.parametrize("measure", RISK_LIMITS)
def test_each_risk_limit_binds_and_rejects_infeasible_holdings(distribution, measure):
    config = DistributionConstraintConfig(risk_limits={measure: 0.001})
    projector = AutoProjector(
        ProjectionConfig(return_distribution=distribution, distribution_config=config)
    )
    weights = projector.project(np.array([1.0, 0.0]))
    assert weights[0] < 0.5
    assert_allclose(weights.sum(), 1, atol=1e-7)
    locked = AutoProjector(
        ProjectionConfig(
            return_distribution=distribution,
            distribution_config=config,
            previous_weights=np.array([1.0, 0.0]),
            max_turnover=0,
        )
    )
    with pytest.raises(RuntimeError, match="failed"):
        locked.project(np.array([1.0, 0.0]))
    locked = AutoProjector(
        ProjectionConfig(
            return_distribution=distribution,
            distribution_config=config,
            previous_weights=np.array([0.0, 1.0]),
            max_turnover=0,
        )
    )
    assert_allclose(locked.project(np.array([1.0, 0.0])), [0.0, 1.0], atol=0)


def test_expected_return_accounts_for_costs_and_fees():
    rd = ReturnDistribution(
        mu=np.array([0.1, 0.0]), covariance=np.eye(2), returns=np.array([[0.1, 0.0]])
    )
    config = DistributionConstraintConfig(
        min_return=0.04,
        management_fees=0.01,
        transaction_costs=0.01,
        previous_weights=np.array([0.0, 1.0]),
    )
    weights = AutoProjector(
        ProjectionConfig(return_distribution=rd, distribution_config=config)
    ).project(np.array([0.0, 1.0]))
    assert_allclose(weights, [0.625, 0.375], atol=1e-6)


def test_standard_deviation_matches_covariance(distribution):
    weights = AutoProjector(
        ProjectionConfig(
            return_distribution=distribution,
            distribution_config=DistributionConstraintConfig(
                risk_limits={RiskMeasure.STANDARD_DEVIATION: 0.05}
            ),
        )
    ).project(np.array([1.0, 0.0]))
    assert_allclose(
        np.sqrt(weights @ distribution.covariance @ weights), 0.05, atol=1e-7
    )


class HistoryPrior(BasePrior):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        raise AssertionError("Must update causally")

    def partial_fit(self, X, y=None):
        self.rows_ = np.vstack([getattr(self, "rows_", np.empty((0, 2))), X])
        self.return_distribution_ = ReturnDistribution(
            mu=self.rows_.mean(axis=0),
            covariance=np.eye(2) * 0.01,
            returns=self.rows_.copy(),
        )
        return self


@pytest.mark.parametrize(
    "estimator_class",
    [
        ExponentiatedGradient,
        AdaGrad,
        OnlineMovingAverageReversion,
        PassiveAggressiveMeanReversion,
    ],
)
def test_public_estimators_clone_and_enforce_causal_limits(estimator_class):
    X = np.array([[0.1, 0.0], [0.05, 0.0], [0.08, 0.0]])
    estimator = estimator_class(
        prior_estimator=HistoryPrior(), min_return=0.04, max_standard_deviation=0.1
    )
    fitted = clone(estimator).fit(X)
    streamed = clone(estimator)
    for row in X:
        streamed.partial_fit(row)
        rd = streamed.prior_estimator_.return_distribution_
        assert rd.mu @ streamed.weights_ >= 0.04 - 1e-7
    assert_allclose(fitted.weights_, streamed.weights_, atol=1e-6)
    assert_allclose(fitted.prior_estimator_.rows_, X)


def test_missing_statistics_fail_at_decision_but_allow_warmup():
    estimator = ExponentiatedGradient(prior_estimator=HistoryPrior(), max_cvar=0.1)
    with pytest.raises(ValueError, match="before an execution decision"):
        estimator.partial_fit([[0.1, 0.0]])
    estimator.fit([[0.1, 0.0], [0.05, 0.0]])
    with pytest.raises(ValueError, match="causal warmup"):
        estimator.fit_predict([[0.1, 0.0], [0.05, 0.0]])


def test_missing_prior_is_rejected():
    with pytest.raises(ValueError, match="causal prior_estimator"):
        ExponentiatedGradient(min_return=0.01).fit([[0.1, 0.0]])


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_invalid_limit(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        DistributionConstraintConfig(risk_limits={RiskMeasure.CVAR: value}).validate()


@pytest.mark.parametrize(
    "measure, limit, expected",
    [
        (RiskMeasure.CVAR, 0.028, 0.2),
        (RiskMeasure.MAX_DRAWDOWN, 0.05, 0.2),
    ],
)
def test_weighted_tail_and_initial_wealth_drawdown(
    distribution, measure, limit, expected
):
    from dataclasses import replace

    rd = replace(distribution, sample_weight=np.array([0.7, 0.1, 0.1, 0.1]))
    weights = AutoProjector(
        ProjectionConfig(
            return_distribution=rd,
            distribution_config=DistributionConstraintConfig(
                risk_limits={measure: limit}, cvar_beta=0
            ),
        )
    ).project(np.array([1.0, 0.0]))
    assert_allclose(weights[0], expected, atol=1e-6)


@pytest.mark.parametrize("geometry", ["entropy", "adaptive"])
@pytest.mark.parametrize("measure", RISK_LIMITS)
def test_risk_caps_apply_to_native_mirror_updates(distribution, geometry, measure):
    from skfolio.optimization.online._mirror_maps import (
        AdaptiveMahalanobisMap,
        EntropyMirrorMap,
    )

    mirror = EntropyMirrorMap() if geometry == "entropy" else AdaptiveMahalanobisMap()
    if geometry == "adaptive":
        mirror.update_state(np.array([1.0, 2.0]))
    projector = AutoProjector(
        ProjectionConfig(
            return_distribution=distribution,
            distribution_config=DistributionConstraintConfig(
                risk_limits={measure: 0.001}
            ),
        )
    )
    weights = projector.solve_update(mirror, mirror.grad_psi(np.array([0.9, 0.1])))
    assert weights[0] < 0.5
    assert_allclose(weights.sum(), 1, atol=1e-7)


def test_empirical_prior_evaluation_with_causal_warmup():
    from skfolio.model_selection import online_predict
    from skfolio.moments import EWCovariance, EWMu
    from skfolio.prior import EmpiricalPrior

    X = np.random.default_rng(261).normal(0, 0.01, (12, 2))
    estimator = ExponentiatedGradient(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(min_observations=3),
            covariance_estimator=EWCovariance(min_observations=3),
            max_history=10,
        ),
        min_return=-0.1,
        max_cvar=0.1,
        max_standard_deviation=0.1,
    )
    result = online_predict(estimator, X, warmup_size=4, test_size=1)
    assert len(result.returns) == 8
    with pytest.raises(ValueError, match="before an execution decision"):
        clone(estimator).fit(X[:1])


def test_distribution_constraints_cannot_be_silently_ignored():
    config = DistributionConstraintConfig(min_return=0.01)
    with pytest.raises(ValueError, match="ready causal prior"):
        AutoProjector(ProjectionConfig(distribution_config=config)).project(
            np.array([0.5, 0.5])
        )


def test_cost_constraints_require_execution_holdings(distribution):
    config = DistributionConstraintConfig(min_return=-0.1, transaction_costs=0.001)
    with pytest.raises(ValueError, match="previous_weights"):
        AutoProjector(
            ProjectionConfig(
                return_distribution=distribution, distribution_config=config
            )
        ).project(np.array([0.5, 0.5]))


def test_infeasible_minimum_return_fails(distribution):
    config = DistributionConstraintConfig(min_return=0.1)
    with pytest.raises(RuntimeError, match="failed"):
        AutoProjector(
            ProjectionConfig(
                return_distribution=distribution, distribution_config=config
            )
        ).project(np.array([0.5, 0.5]))
