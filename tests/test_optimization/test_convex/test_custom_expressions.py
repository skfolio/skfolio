"""Tests for the custom expression hooks of convex optimization estimators:
`add_objective`, `add_constraints` and `overwrite_expected_return`.

The tests cover the docstring examples of these parameters and the one or two
positional arguments dispatch of `_call_custom_func`.
"""

from __future__ import annotations

import functools

import cvxpy as cp
import numpy as np
import pytest

from skfolio import RiskMeasure
from skfolio.optimization import MeanRisk, ObjectiveFunction


def test_add_constraints_single_argument(X_medium):
    # Effective number of assets of at least 20 (docstring example).
    model = MeanRisk(add_constraints=lambda w: cp.sum_squares(w) <= 1 / 20)
    model.fit(X_medium)
    hhi = np.sum(model.weights_**2)
    assert hhi <= 1 / 20 + 1e-8

    # The unconstrained minimum variance portfolio is more concentrated, so the
    # constraint is active and binds.
    ref = MeanRisk().fit(X_medium)
    assert np.sum(ref.weights_**2) > 1 / 20
    np.testing.assert_allclose(hhi, 1 / 20, rtol=1e-4)


def test_add_constraints_with_estimator_argument(X_medium):
    # Position risk cap using the fitted prior (docstring example).
    seen = {}

    def position_risk_cap(w, model):
        seen["estimator"] = model
        covariance = model.prior_estimator_.return_distribution_.covariance
        vols = np.sqrt(np.diag(covariance))
        return cp.multiply(vols, w) <= 0.002

    model = MeanRisk(add_constraints=position_risk_cap)
    model.fit(X_medium)

    assert seen["estimator"] is model
    vols = np.sqrt(np.diag(model.prior_estimator_.return_distribution_.covariance))
    assert np.max(model.weights_ * vols) <= 0.002 + 1e-8


def test_add_constraints_list_of_constraints(X_small):
    model = MeanRisk(
        add_constraints=lambda w: [w[0] >= 0.05, cp.sum_squares(w) <= 1 / 10]
    )
    model.fit(X_small)
    assert model.weights_[0] >= 0.05 - 1e-8
    assert np.sum(model.weights_**2) <= 1 / 10 + 1e-8


def test_add_objective_with_estimator_argument(X_medium):
    seen = {}

    def diversification_penalty(w, model):
        seen["estimator"] = model
        return 10.0 * cp.sum_squares(w)

    model = MeanRisk(add_objective=diversification_penalty).fit(X_medium)
    ref = MeanRisk().fit(X_medium)

    assert seen["estimator"] is model
    # The penalty dominates the variance objective and spreads the weights.
    assert np.sum(model.weights_**2) < np.sum(ref.weights_**2)


def test_overwrite_expected_return_matches_native_utility(X_medium):
    # Volatility drag adjustment (docstring example). Maximizing the adjusted
    # expected return mu @ w - 0.5 * w.T @ covariance @ w is equivalent to the
    # native mean-variance utility with a risk aversion of 0.5.
    def geometric_expected_return(w, model):
        dist = model.prior_estimator_.return_distribution_
        return dist.mu @ w - 0.5 * cp.quad_form(w, dist.covariance)

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        overwrite_expected_return=geometric_expected_return,
    ).fit(X_medium)

    ref = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=RiskMeasure.VARIANCE,
        risk_aversion=0.5,
    ).fit(X_medium)

    # The two paths use different default objective scalings, so the solutions
    # only agree up to solver tolerance.
    np.testing.assert_allclose(model.weights_, ref.weights_, atol=1e-3)


def test_custom_func_without_code_object_raises(X_small):
    model = MeanRisk(
        add_constraints=functools.partial(lambda w, scale: w >= 0, scale=1.0)
    )
    with pytest.raises(ValueError, match="Custom functions is invalid"):
        model.fit(X_small)


def test_custom_func_too_many_arguments_raises(X_small):
    model = MeanRisk(add_constraints=lambda w, estimator, extra: w >= 0)
    with pytest.raises(
        ValueError, match="Custom functions must have 1 or 2 positional arguments"
    ):
        model.fit(X_small)


def test_custom_func_internal_error_raises_type_error(X_small):
    def failing_constraint(w):
        raise RuntimeError("boom")

    model = MeanRisk(add_constraints=failing_constraint)
    with pytest.raises(TypeError, match="Error while calling add_constraint"):
        model.fit(X_small)
