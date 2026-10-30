"""EVaR uses the shared value and independently checked envelope gradients."""

import numpy as np
import pytest
from autograd import grad

from skfolio.measures import edar, evar, get_drawdowns
from skfolio.optimization.online._autograd_objectives import (
    _analytical_edar_gradient,
    _analytical_evar_gradient,
    _autograd_edar,
    _autograd_evar,
    _evar_probabilities,
)


@pytest.mark.parametrize("beta", [0.0, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e3])
def test_evar_values_and_gradients(beta, scale):
    X = scale * np.random.default_rng(421).normal(0, 0.01, (40, 3))
    w = np.array([0.2, 0.5, 0.3])
    expected = evar(X @ w, beta=beta)
    np.testing.assert_allclose(_autograd_evar(X @ w, beta=beta), expected, atol=1e-14)
    gradient = _analytical_evar_gradient(w, X, beta=beta)
    automatic = grad(lambda weights: _autograd_evar(X @ weights, beta=beta))(w)
    eps = 1e-5
    finite = np.array(
        [
            (evar(X @ (w + eps * e), beta=beta) - evar(X @ (w - eps * e), beta=beta))
            / (2 * eps)
            for e in np.eye(3)
        ]
    )
    np.testing.assert_allclose(gradient, finite, rtol=2e-5, atol=scale * 1e-9)
    np.testing.assert_allclose(automatic, gradient, atol=scale * 1e-12)


@pytest.mark.parametrize("beta", [0, 0.5, 1])
def test_constant_and_tied_maximum_subgradients(beta):
    np.testing.assert_allclose(_evar_probabilities(np.ones(4), beta), 0.25)
    if beta:
        np.testing.assert_allclose(
            _evar_probabilities(np.array([2.0, 2.0, 0.0, 0.0]), beta), [0.5, 0.5, 0, 0]
        )


@pytest.mark.parametrize("compounded", [False, True])
def test_edar_envelope_matches_shared_measure(compounded):
    X = np.random.default_rng(19).normal(0, 0.03, (30, 2))
    X[0] = [-0.1, -0.05]
    w = np.array([0.7, 0.3])
    beta = 0.5

    def objective(weights):
        return edar(get_drawdowns(X @ weights, compounded=compounded), beta=beta)

    np.testing.assert_allclose(
        _autograd_edar(X @ w, beta=beta, compounded=compounded), objective(w)
    )
    finite = np.array(
        [(objective(w + 1e-6 * e) - objective(w - 1e-6 * e)) / 2e-6 for e in np.eye(2)]
    )
    analytical = _analytical_edar_gradient(w, X, beta=beta, compounded=compounded)
    automatic = grad(
        lambda weights: _autograd_edar(X @ weights, beta=beta, compounded=compounded)
    )(w)
    np.testing.assert_allclose(analytical, finite, atol=1e-8)
    np.testing.assert_allclose(automatic, finite, atol=1e-8)
