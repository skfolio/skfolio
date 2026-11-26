"""Validate external, point-in-time inputs for portfolio projection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def projection_context(
    asset_names: list[str] | np.ndarray | pd.Index,
    observations: pd.Index | np.ndarray,
    covariance: ArrayLike | pd.DataFrame | None = None,
    holdings: ArrayLike | pd.DataFrame | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Align named inputs and validate positional covariance/holding batches.

    A single covariance matrix is explicitly shared by all decisions. Holdings
    require one vector per decision and are never broadcast over time.
    """
    names = list(asset_names)
    n, t = len(names), len(observations)
    cov = None
    held = None
    if covariance is not None:
        if isinstance(covariance, pd.DataFrame):
            if (
                not covariance.index.is_unique
                or not covariance.columns.is_unique
                or set(covariance.index) != set(names)
                or set(covariance.columns) != set(names)
            ):
                raise ValueError(
                    "Covariance asset names must match the fitted universe."
                )
            covariance = covariance.loc[names, names].to_numpy()
        cov = np.asarray(covariance, dtype=float)
        if cov.shape == (n, n):
            cov = np.broadcast_to(cov, (t, n, n)).copy()
        if cov.shape != (t, n, n) or not np.isfinite(cov).all():
            raise ValueError("Covariance must be finite with shape (n,n) or (t,n,n).")
        if not np.allclose(cov, cov.swapaxes(1, 2), rtol=0, atol=1e-12):
            raise ValueError("Covariance must be symmetric.")
        cov = (cov + cov.swapaxes(1, 2)) / 2
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Singular sample covariances can have negative eigenvalues from roundoff.
        scale = np.max(np.abs(eigenvalues), axis=-1, keepdims=True)
        tolerance = 10 * n * np.finfo(float).eps * scale
        if (eigenvalues < -tolerance).any():
            raise ValueError("Covariance must be positive semidefinite.")
        for i in np.flatnonzero((eigenvalues < 0).any(axis=-1)):
            cov[i] = (eigenvectors[i] * np.maximum(eigenvalues[i], 0)) @ eigenvectors[
                i
            ].T
            cov[i] = (cov[i] + cov[i].T) / 2
    if holdings is not None:
        if isinstance(holdings, pd.DataFrame):
            if (
                not holdings.columns.is_unique
                or set(holdings.columns) != set(names)
                or not holdings.index.equals(pd.Index(observations))
            ):
                raise ValueError(
                    "Holdings must match decision dates and fitted assets."
                )
            holdings = holdings.loc[:, names].to_numpy()
        held = np.asarray(holdings, dtype=float)
        if t == 1 and held.shape == (n,):
            held = held[None, :]
        if held.shape != (t, n) or not np.isfinite(held).all():
            raise ValueError("Holdings must contain one finite vector per decision.")
        if not np.allclose(held.sum(axis=1), 1, rtol=0, atol=1e-8):
            raise ValueError("Holdings must sum to one; cash is not supported yet.")
    return cov, held
