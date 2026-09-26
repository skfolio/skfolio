"""Tests for factor-model utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from skfolio.utils._factor_tools import _expand_factor_names
from skfolio.utils.stats import _market_returns


def test_expand_factor_names_deduplicates_family_members():
    """Expanded target lists keep first occurrence order and remove duplicates."""
    factor_to_idx = {"size": 0, "value": 1, "market": 2}
    family_to_idx = {"style": [0, 1], "market": [2]}

    result = _expand_factor_names(
        ["style", "size", "market"],
        factor_to_idx=factor_to_idx,
        family_to_idx=family_to_idx,
    )

    assert result == [0, 1, 2]


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
