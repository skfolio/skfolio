"""Tests for the abstract variance base class."""

from __future__ import annotations

import numpy as np

from skfolio.moments import BaseVariance


class _MinimalVariance(BaseVariance):
    """Concrete subclass delegating `fit` to the abstract base implementation."""

    def fit(self, X, y=None):
        super().fit(X, y)
        return self


def test_base_variance_abstract_fit_is_no_op():
    model = _MinimalVariance()
    assert model.fit(np.zeros((3, 2))) is model
