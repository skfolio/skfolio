from __future__ import annotations

import numpy as np
import pytest

from skfolio.distribution import (
    BaseBivariateCopula,
    ClaytonCopula,
    GaussianCopula,
    IndependentCopula,
    SelectionCriterion,
    select_bivariate_copula,
)


def test_select_bivariate_copula_independence():
    # Create data with very weak dependence.
    X = np.random.RandomState(42).random((100, 2))
    copula_candidates = [GaussianCopula(), ClaytonCopula()]
    best = select_bivariate_copula(X, copula_candidates, independence_level=0.05)
    # When independence holds, we expect IndependentCopula to be returned.
    assert isinstance(best, IndependentCopula)


def test_select_bivariate_copula_aic_selection():
    # Create data with some dependence.
    X = np.random.RandomState(42).random((100, 2))
    # Disturb X slightly to ensure dependence.
    X[:, 1] = 0.5 * X[:, 0] + 0.5 * np.random.RandomState(42).random(100)
    copula_candidates = [GaussianCopula(), ClaytonCopula()]
    best = select_bivariate_copula(X, copula_candidates, independence_level=0.01)
    assert isinstance(best, GaussianCopula)


def test_select_bivariate_copula_invalid_X():
    # Test that an error is raised if X does not have exactly 2 columns.
    X = np.random.rand(100, 3)
    copula_candidates = [GaussianCopula()]
    with pytest.raises(ValueError):
        _ = select_bivariate_copula(X, copula_candidates)


def test_select_bivariate_copula_invalid_candidate():
    # Test that an error is raised if a candidate does not inherit from BaseBivariateCopula.
    X = np.random.rand(100, 2)
    # Disturb X slightly to ensure dependence.
    X[:, 1] = 0.5 * X[:, 0] + 0.5 * np.random.rand(100)
    copula_candidates = [GaussianCopula(), "not a copula"]
    with pytest.raises(ValueError):
        _ = select_bivariate_copula(X, copula_candidates)


@pytest.fixture
def dependent_data():
    rng = np.random.default_rng(0)
    X = rng.random((100, 2))
    X[:, 1] = 0.5 * X[:, 0] + 0.5 * rng.random(100)
    return X


def test_select_bivariate_copula_default_candidates(dependent_data):
    best = select_bivariate_copula(dependent_data, independence_level=0.01)
    assert isinstance(best, BaseBivariateCopula)
    assert not isinstance(best, IndependentCopula)


def test_select_bivariate_copula_bic_selection(dependent_data):
    best = select_bivariate_copula(
        dependent_data,
        [GaussianCopula(), ClaytonCopula()],
        selection_criterion=SelectionCriterion.BIC,
        independence_level=0.01,
    )
    assert isinstance(best, GaussianCopula)


def test_select_bivariate_copula_invalid_criterion(dependent_data):
    # The enum class exposes `.AIC`/`.BIC` but is neither member.
    with pytest.raises(ValueError, match="not implemented"):
        select_bivariate_copula(
            dependent_data,
            [GaussianCopula()],
            selection_criterion=SelectionCriterion,
            independence_level=0.01,
        )
