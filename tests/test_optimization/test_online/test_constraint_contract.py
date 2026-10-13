"""Constraint enforcement through the public online API."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from skfolio.exceptions import GroupNotFoundError
from skfolio.optimization.online import (
    AdaGrad,
    ExponentiatedGradient,
    OnlineMovingAverageReversion,
    PassiveAggressiveMeanReversion,
)
from skfolio.prior import BasePrior, FactorModel, ReturnDistribution

ESTIMATORS = [
    ExponentiatedGradient,
    AdaGrad,
    OnlineMovingAverageReversion,
    PassiveAggressiveMeanReversion,
]


class ChangingFactorPrior(BasePrior):
    """Make the constrained asset alternate after each causal observation."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        raise AssertionError("Only causal partial_fit is allowed")

    def partial_fit(self, X, y=None):
        self.count_ = getattr(self, "count_", 0) + 1
        loading = np.array([[1.0], [0.0]])
        if self.count_ % 2 == 0:
            loading = loading[::-1].copy()
        self.return_distribution_ = ReturnDistribution(
            mu=np.zeros(2),
            covariance=np.eye(2),
            returns=np.asarray(X),
            factor_model=FactorModel(
                observations=np.array([str(self.count_)]),
                asset_names=np.array(["a", "b"]),
                factor_names=np.array(["Momentum"]),
                factor_families=np.array(["Style"]),
                loading_matrix=loading,
                exposures=None,
                factor_covariance=np.eye(1),
                factor_mu=np.zeros(1),
                factor_returns=None,
                idio_covariance=np.eye(2),
                idio_mu=None,
                idio_returns=None,
                idio_variances=None,
            ),
        )
        return self


@pytest.mark.parametrize("factory", ESTIMATORS)
@pytest.mark.parametrize("constraint", ["Momentum <= 0.2", "Style <= 0.2"])
def test_causal_factor_constraint_changes_target(factory, constraint):
    X = pd.DataFrame([[0.01, 0.02], [0.02, 0.01], [0.01, 0.02]], columns=["a", "b"])
    prior = ChangingFactorPrior()
    model = factory(prior_estimator=prior, linear_constraints=[constraint])
    targets = []
    for i in range(len(X)):
        model.partial_fit(X.iloc[i : i + 1])
        loading = (
            model.prior_estimator_.return_distribution_.factor_model.loading_matrix
        )
        assert float((loading.T @ model.weights_)[0]) <= 0.2 + 1e-6
        assert_allclose(model.weights_.sum(), 1.0, atol=1e-6)
        targets.append(model.weights_.copy())
    assert targets[0][0] < targets[1][0] - 0.5
    assert not hasattr(prior, "count_")
    block = factory(
        prior_estimator=ChangingFactorPrior(), linear_constraints=[constraint]
    )
    block.fit(X)
    assert_allclose(block.weights_, model.weights_, atol=1e-6)


@pytest.mark.parametrize("factory", ESTIMATORS)
def test_unknown_factor_constraint_raises(factory):
    X = pd.DataFrame([[0.01, 0.02]], columns=["a", "b"])
    with pytest.raises(GroupNotFoundError, match="missing"):
        factory(
            prior_estimator=ChangingFactorPrior(), linear_constraints=["Typo <= 0.2"]
        ).fit(X)


@pytest.mark.parametrize("factory", ESTIMATORS)
def test_factor_constraint_cannot_be_deferred_for_initial_execution(factory):
    X = pd.DataFrame([[0.01, 0.02], [0.02, 0.01]], columns=["a", "b"])
    model = factory(
        prior_estimator=ChangingFactorPrior(), linear_constraints=["Momentum <= 0.2"]
    )
    with pytest.raises(GroupNotFoundError, match="missing"):
        model.fit_predict(X)
    assert not hasattr(model, "prior_estimator_")
