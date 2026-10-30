"""Online drawdowns follow shared measure values and path derivatives."""

import numpy as np
import pytest
from autograd import grad

from skfolio.measures import get_drawdowns
from skfolio.optimization.online._autograd_objectives import (
    _autograd_get_drawdowns,
    _drawdowns_and_derivatives,
)


@pytest.mark.parametrize("compounded", [False, True])
@pytest.mark.parametrize("returns", [[-0.1, 0.05], [0.1, -0.2, 0.03], [0.0, -0.1, 0.0]])
def test_drawdown_values_include_initial_peak(returns, compounded):
    returns = np.asarray(returns)
    expected = get_drawdowns(returns, compounded=compounded)
    values, _ = _drawdowns_and_derivatives(returns[:, None], np.ones(1), compounded)
    np.testing.assert_allclose(-values, expected)
    np.testing.assert_allclose(_autograd_get_drawdowns(returns, compounded), expected)


@pytest.mark.parametrize("compounded", [False, True])
def test_drawdown_gradient_against_shared_measure(compounded):
    X = np.array([[-0.12, -0.03], [0.3, 0.15], [-0.2, -0.1], [0.03, -0.02]])
    w = np.array([0.6, 0.4])
    _, derivatives = _drawdowns_and_derivatives(X, w, compounded)
    eps = 1e-6
    finite = np.column_stack(
        [
            -(
                get_drawdowns(X @ (w + eps * e), compounded)
                - get_drawdowns(X @ (w - eps * e), compounded)
            )
            / (2 * eps)
            for e in np.eye(2)
        ]
    )
    np.testing.assert_allclose(derivatives, finite, atol=1e-9)
    automatic = grad(
        lambda weights: -_autograd_get_drawdowns(X @ weights, compounded).sum()
    )(w)
    np.testing.assert_allclose(automatic, finite.sum(axis=0), atol=1e-9)
