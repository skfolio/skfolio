"""Independent finite-difference checks for retained objective kernels."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._autograd_objectives import (
    _drawdowns_and_derivatives,
    _wealth_and_dwealth,
    create_mean_reversion_objective,
)


def numerical_jacobian(function, weights):
    step = 1e-6
    directions = np.eye(len(weights)) * step
    return np.stack(
        [
            (function(weights + direction) - function(weights - direction)) / (2 * step)
            for direction in directions
        ],
        axis=-1,
    )


@pytest.mark.parametrize("first_return", [0.1, -2.0, -3.0])
def test_wealth_kernel_matches_product_and_derivative(first_return):
    weights = np.array([0.5, 0.5])
    returns = np.array([[first_return, 0.0], [-0.03, 0.04], [0.02, -0.01]])

    # Includes zero and negative gross factors to exercise the direct-product
    # derivative. These are algebraic kernel cases, not supported public inputs.
    def reference(w):
        return np.cumprod(1 + returns @ w)

    wealth, derivative = _wealth_and_dwealth(returns, weights)
    assert_allclose(wealth, reference(weights), atol=1e-12)
    assert_allclose(derivative, numerical_jacobian(reference, weights), atol=1e-9)


@pytest.mark.parametrize("compounded", [False, True])
def test_drawdown_kernel_matches_independent_path_derivative(compounded):
    weights = np.array([0.4, 0.6])
    returns = np.array([[0.03, 0.05], [-0.08, -0.04], [0.2, 0.1], [-0.03, -0.07]])

    def reference(w):
        path = np.cumprod(1 + returns @ w) if compounded else np.cumsum(returns @ w)
        return np.maximum.accumulate(path) - path

    drawdown, derivative = _drawdowns_and_derivatives(returns, weights, compounded)
    assert_allclose(drawdown, reference(weights), atol=1e-12)
    assert_allclose(derivative, numerical_jacobian(reference, weights), atol=1e-9)


@pytest.mark.parametrize("kind", ["hinge", "squared_hinge", "softplus"])
@pytest.mark.parametrize("epsilon", [0.5, 2.0])
def test_reversion_objective_matches_scalar_loss_and_derivative(kind, epsilon):
    weights = np.array([0.3, 0.7])
    phi = np.array([1.2, 0.8])
    beta = 3.0
    obj = create_mean_reversion_objective(kind, epsilon=epsilon, beta=beta)

    def reference(w):
        margin = epsilon - phi @ w
        if kind == "hinge":
            return max(0.0, margin)
        if kind == "squared_hinge":
            return max(0.0, margin) ** 2
        return np.logaddexp(0, beta * margin) / beta

    assert_allclose(obj.loss(weights, phi), reference(weights))
    assert_allclose(
        obj.grad(weights, phi), numerical_jacobian(reference, weights), atol=1e-9
    )


def test_unknown_reversion_objective_rejected():
    with pytest.raises(ValueError, match="Unknown mean-reversion objective"):
        create_mean_reversion_objective("unknown", epsilon=1.0)
