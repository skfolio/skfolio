"""Tests for the abstract prior base class."""

from __future__ import annotations

import numpy as np

from skfolio.prior import BasePrior


class _MinimalPrior(BasePrior):
    """Concrete subclass delegating to the abstract base implementations."""

    def __init__(self):
        super().__init__()

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        return self


def test_base_prior_abstract_methods_are_no_ops():
    model = _MinimalPrior()
    assert model.fit(np.zeros((3, 2))) is model
