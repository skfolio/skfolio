"""Tests for skfolio.alpha._base."""

from __future__ import annotations

import numpy as np

from skfolio.alpha._base import _neutralize_scores


def test_neutralize_scores_excludes_missing_score_and_exposure():
    """Missing score or exposure entries should receive zero regression weight."""
    exposure = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, -1.0, 2.0, -2.0],
        ]
    )
    residual = np.array(
        [
            [1.0, -1.0, 0.5, -0.5],
            [0.3, 0.2, -0.1, -0.4],
        ]
    )
    scores = (2.0 * exposure + residual)[:, :, None]
    exposures = exposure[:, :, None].copy()
    cs_weights = np.ones_like(exposure)

    scores[0, 1, 0] = np.nan
    exposures[1, 2, 0] = np.nan

    result = _neutralize_scores(
        neutralize_against=["market"],
        scores=scores,
        exposures=exposures,
        cs_weights=cs_weights,
        factor_names=np.array(["market"]),
        factor_families=np.array(["market"]),
    )

    assert result is scores
    assert np.isnan(scores[0, 1, 0])
    assert np.isnan(scores[1, 2, 0])

    for t in range(exposure.shape[0]):
        valid = np.isfinite(scores[t, :, 0]) & np.isfinite(exposures[t, :, 0])
        weighted_dot = np.sum(scores[t, valid, 0] * exposures[t, valid, 0])
        assert abs(weighted_dot) < 1e-12
