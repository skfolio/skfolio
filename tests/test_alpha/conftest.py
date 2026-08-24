"""Shared alpha test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio._constants import _BENCHMARK_WEIGHTS, _IDIO_RETURNS, _IDIO_VARIANCES
from skfolio.containers import AssetPanel


@pytest.fixture
def alpha_deterministic_panel():
    """Create deterministic panel data with required fields for alpha tests."""
    np.random.seed(123)
    n_obs = 20
    n_assets = 4

    observations = pd.bdate_range("2020-01-01", periods=n_obs).to_numpy()
    assets = np.array(["X", "Y", "Z", "W"])

    base_returns = np.array(
        [
            [0.01, -0.02, 0.015, -0.005],
            [-0.01, 0.025, -0.01, 0.02],
            [0.02, -0.015, 0.005, -0.01],
            [-0.005, 0.01, -0.02, 0.015],
            [0.015, -0.005, 0.02, -0.015],
        ]
    )
    returns = np.tile(base_returns, (4, 1))
    idio_returns = returns + np.random.randn(n_obs, n_assets) * 0.005
    idio_vol = np.tile([0.02, 0.025, 0.018, 0.022], (n_obs, 1))
    signal = np.cumsum(returns, axis=0)
    benchmark_weights = np.ones((n_obs, n_assets)) / n_assets

    panel = AssetPanel(
        fields={
            "returns": returns,
            _IDIO_RETURNS: idio_returns,
            _IDIO_VARIANCES: idio_vol**2,
            "signal": signal,
        },
        observations=observations,
        asset_names=assets,
    )
    panel[_BENCHMARK_WEIGHTS] = benchmark_weights
    return panel
