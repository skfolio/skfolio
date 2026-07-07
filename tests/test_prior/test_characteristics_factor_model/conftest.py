"""Shared helpers and fixtures for CharacteristicsFactorModel tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from skfolio.containers import AssetPanel
from skfolio.descriptor import Passthrough
from skfolio.factor_exposure import FixedWeightedFactor


def make_panel(
    returns: np.ndarray,
    *,
    extra_fields: dict[str, np.ndarray] | None = None,
    market_cap: np.ndarray | None = None,
    asset_names: np.ndarray | None = None,
    estimation_mask: np.ndarray | None = None,
    active_mask: np.ndarray | None = None,
) -> tuple[AssetPanel, pd.DataFrame]:
    """Build an AssetPanel and matching X DataFrame from simulated returns.

    Parameters
    ----------
    returns : ndarray of shape (n_obs, n_assets)
    extra_fields : additional 2-D fields to include in the panel.
    market_cap : if None, defaults to equal weights (ones).
    asset_names : if None, auto-generated as ``asset_0 .. asset_{n-1}``.
    estimation_mask : if None, defaults to all True.
    active_mask : if None, defaults to all True.

    Returns
    -------
    panel : AssetPanel
    X : DataFrame (same data as ``returns``, columns = asset names)
    """
    n_obs, n_assets = returns.shape

    if asset_names is None:
        asset_names = np.array([f"asset_{i}" for i in range(n_assets)])

    if market_cap is None:
        market_cap = np.ones((n_obs, n_assets))

    fields: dict[str, np.ndarray] = {
        "returns": returns,
        "market_cap": market_cap,
    }
    if extra_fields is not None:
        fields.update(extra_fields)

    panel = AssetPanel(
        fields=fields,
        observations=np.arange(n_obs),
        asset_names=asset_names,
        estimation_mask=estimation_mask,
        active_mask=active_mask,
    )

    X = pd.DataFrame(returns, columns=asset_names)
    return panel, X


def passthrough_factor(
    field_name: str,
    family: str = "style",
) -> FixedWeightedFactor:
    """Single-descriptor factor with no outlier/scoring transforms."""
    return FixedWeightedFactor(
        descriptors=[(field_name, Passthrough(field_name))],
        family=family,
        outlier_transformer="passthrough",
        scoring_transformer="passthrough",
    )
