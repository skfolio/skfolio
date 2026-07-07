from __future__ import annotations

import numpy as np

from skfolio.factor_exposure import GlobalFactor


def test_fit_transform_returns_constant_exposure(simple_panel):
    """Test that GlobalFactor returns one exposure per observation and asset."""
    factor = GlobalFactor()

    result = factor.fit_transform(simple_panel)

    assert result.shape == (simple_panel.n_observations, simple_panel.n_assets)
    np.testing.assert_array_equal(result, np.ones_like(result))
    assert factor.family == "market"
    assert factor.n_assets_ == simple_panel.n_assets
    np.testing.assert_array_equal(factor.asset_names_, simple_panel.asset_names)


def test_partial_fit_transform_matches_fit_transform(simple_panel):
    """Test stateless partial_fit_transform delegation."""
    fit_result = GlobalFactor(family="market").fit_transform(simple_panel)
    partial_fit_result = GlobalFactor(family="market").partial_fit_transform(
        simple_panel
    )

    np.testing.assert_array_equal(partial_fit_result, fit_result)
