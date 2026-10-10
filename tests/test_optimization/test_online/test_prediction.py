"""Test standalone prediction dependencies before exposing estimator families."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.optimization.online._prediction import (
    OLMAR1Predictor,
    OLMAR2Predictor,
    RMRPredictor,
    ZeroPredictor,
)


@pytest.mark.parametrize("factory", [OLMAR1Predictor, OLMAR2Predictor, RMRPredictor])
def test_reversion_history_resets_to_fresh_predictions(factory):
    predictor = factory()
    predictor.reset(2)
    rows = np.array([[1.02, 0.99], [0.97, 1.03], [1.01, 1.02]])
    expected = np.array([predictor.update_and_predict(row) for row in rows])
    predictor.reset(2)
    actual = np.array([predictor.update_and_predict(row) for row in rows])
    assert_allclose(actual, expected)
    assert np.isfinite(actual).all()


def test_olmar_preserves_source_window_and_warmup_convention():
    predictor = OLMAR1Predictor(window=2)
    predictor.reset(2)
    first = np.array([1.1, 0.9])
    second = np.array([1.2, 0.8])
    third = np.array([1.05, 0.95])
    assert_allclose(predictor.update_and_predict(first), first)
    assert_allclose(predictor.update_and_predict(second), second)
    # The pinned source averages inverse products over W relatives, with no
    # current-price unit term. Review this convention with the estimator port.
    assert_allclose(
        predictor.update_and_predict(third), (1 / third + 1 / (second * third)) / 2
    )


def test_zero_gradient_prediction_uses_current_dimension():
    assert_allclose(ZeroPredictor()(0, np.ones(3), None), np.zeros(3))
