"""Tests for the abstract expected-returns base class."""

from __future__ import annotations

import numpy as np

from skfolio.moments import BaseMu


class _MinimalMu(BaseMu):
    """Concrete subclass delegating to the abstract base implementations."""

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        super().fit(X, y)
        return self


def test_base_mu_abstract_methods_are_no_ops():
    model = _MinimalMu()
    assert model.fit(np.zeros((3, 2))) is model
