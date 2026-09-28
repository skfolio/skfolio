"""Utility functions for cross-sectional linear models."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

from skfolio.linear_model._cross_sectional._base import BaseCSLinearModel
from skfolio.linear_model._cross_sectional._cs_linear_regression import (
    CSLinearRegression,
)
from skfolio.typing import FloatArray


def _cs_neutralize(
    y: FloatArray,
    x: FloatArray,
    cs_weights: FloatArray,
    cs_regressor: BaseCSLinearModel | None = None,
) -> tuple[FloatArray, FloatArray]:
    r"""Neutralize a panel against cross-sectional explanatory variables.

    For each observation :math:`t`, regress :math:`y_t` on :math:`X_t` across
    assets with cross-sectional weights and return the residuals:

    .. math::

        \tilde{y}_t = y_t - X_t \hat{\beta}_t

    Missing `y` values or rows of `x` with missing features are excluded by
    setting their effective regression weights to zero.

    Parameters
    ----------
    y : ndarray of shape (n_observations, n_assets)
        Cross-sectional response values to neutralize.

    x : ndarray of shape (n_observations, n_assets, n_features)
        Cross-sectional explanatory variables.

    cs_weights : ndarray of shape (n_observations, n_assets)
        Base cross-sectional weights. Entries whose `y` or `x` values are
        missing receive zero effective weight.

    cs_regressor : BaseCSLinearModel, optional
        Cross-sectional linear regressor used for residualization. If `None`,
        `CSLinearRegression(fit_intercept=False)` is used.

    Returns
    -------
    neutralized : ndarray of shape (n_observations, n_assets)
        Cross-sectional residuals. Missing values in `y` or `x` propagate to
        the corresponding residual entries.

    weights : ndarray of shape (n_observations, n_assets)
        Effective regression weights after missing-value exclusion.
    """
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=2)
    cs_weights = np.where(valid, cs_weights, 0.0)
    regressor = (
        CSLinearRegression(fit_intercept=False)
        if cs_regressor is None
        else cs_regressor
    )
    neutralized = y - regressor.fit(x, y, cs_weights=cs_weights).predict(x)
    return neutralized, cs_weights
