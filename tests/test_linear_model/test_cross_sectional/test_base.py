"""Tests for the cross-sectional linear model base class."""

from __future__ import annotations

import numpy as np

from skfolio.linear_model._cross_sectional._base import BaseCSLinearModel


class _Delegating(BaseCSLinearModel):
    """Concrete model delegating `fit` to the abstract base implementation."""

    def fit(self, X, y, cs_weights=None):
        return super().fit(X, y, cs_weights=cs_weights)


def test_base_fit_is_abstract_and_has_no_behavior():
    X = np.zeros((2, 3, 1))
    y = np.zeros((2, 3))
    assert _Delegating().fit(X, y) is None
