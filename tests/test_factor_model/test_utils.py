"""Tests for factor-model utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.factor_model._utils import _market_returns


def test_market_returns_uses_estimation_mask():
    """Market returns are computed from estimable entries only."""
    returns = np.array(
        [
            [0.10, 0.01],
            [0.20, 0.02],
        ]
    )
    weights = np.array(
        [
            [1_000.0, 1.0],
            [1_000.0, 1.0],
        ]
    )
    estimation_mask = np.array(
        [
            [False, True],
            [False, True],
        ]
    )

    result = _market_returns(
        asset_returns=returns,
        weights=weights,
        estimation_mask=estimation_mask,
    )

    np.testing.assert_allclose(result, [0.01, 0.02])


def test_market_returns_raises_on_undefined_row():
    """Raises when no estimable asset can define a market return."""
    returns = np.array(
        [
            [np.nan, 0.01],
            [0.20, 0.02],
        ]
    )
    weights = np.ones_like(returns)
    estimation_mask = np.array(
        [
            [True, False],
            [True, True],
        ]
    )

    with pytest.raises(ValueError, match="observation index 0"):
        _market_returns(
            asset_returns=returns,
            weights=weights,
            estimation_mask=estimation_mask,
        )


def test_market_returns_raises_on_shape_mismatch():
    """Input arrays must share the same 2D shape."""
    returns = np.ones((3, 2))
    weights = np.ones((3, 3))

    with pytest.raises(ValueError, match="weights must have the same shape"):
        _market_returns(asset_returns=returns, weights=weights)
