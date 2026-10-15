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


@pytest.mark.parametrize("window", [1, 2, 3, 5])
def test_olmar_matches_price_level_reference(window):
    prices = np.array(
        [
            [10.0, 20.0],
            [11.0, 18.0],
            [13.2, 14.4],
            [12.0, 16.0],
            [10.0, 19.0],
            [11.0, 21.0],
        ]
    )
    predictor = OLMAR1Predictor(window=window)
    predictor.reset(2)
    for t in range(1, len(prices)):
        prediction = predictor.update_and_predict(prices[t] / prices[t - 1])
        expected = (
            prices[t - window + 1 : t + 1].mean(axis=0) / prices[t]
            if t + 1 >= window
            else np.ones(2)
        )
        assert_allclose(prediction, expected)
        assert len(predictor._history) <= window - 1


def test_olmar_copies_inputs_and_rejects_invalid_observation_before_update():
    predictor = OLMAR1Predictor(window=3)
    first = np.array([1.1, 0.9])
    predictor.update_and_predict(first)
    first[:] = 100
    with pytest.raises(ValueError, match="positive"):
        predictor.update_and_predict([0, 1])
    assert_allclose(
        predictor.update_and_predict([1.2, 0.8]),
        (1 + 1 / np.array([1.2, 0.8]) + 1 / (np.array([1.1, 0.9]) * [1.2, 0.8])) / 3,
    )


def test_zero_gradient_prediction_uses_current_dimension():
    assert_allclose(ZeroPredictor()(0, np.ones(3), None), np.zeros(3))
