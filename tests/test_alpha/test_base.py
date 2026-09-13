"""Tests for skfolio.alpha._base."""

from __future__ import annotations

import numpy as np

from skfolio.alpha._base import BaseAlpha, _neutralize_scores


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


class _MinimalAlpha(BaseAlpha):
    """Minimal concrete alpha that delegates to the abstract base `fit`."""

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        self.alpha_ = np.zeros(X.n_assets)
        return self


def test_base_alpha_abstract_fit_is_a_no_op(alpha_deterministic_panel):
    """A concrete subclass can call the abstract `fit`; it does nothing."""
    model = _MinimalAlpha()

    assert model.fit(alpha_deterministic_panel) is model
    np.testing.assert_array_equal(
        model.alpha_, np.zeros(alpha_deterministic_panel.n_assets)
    )


def test_neutralize_scores_without_targets_returns_input_unchanged():
    """An empty neutralization target list leaves the scores untouched."""
    scores = np.arange(8, dtype=float).reshape(2, 4, 1)
    expected = scores.copy()
    exposures = np.ones((2, 4, 1))

    result = _neutralize_scores(
        neutralize_against=[],
        scores=scores,
        exposures=exposures,
        cs_weights=np.ones((2, 4)),
        factor_names=np.array(["market"]),
        factor_families=None,
    )

    assert result is scores
    np.testing.assert_array_equal(result, expected)
