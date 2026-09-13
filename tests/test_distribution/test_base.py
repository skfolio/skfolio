from __future__ import annotations

import numpy as np
import pytest

from skfolio.distribution import BaseDistribution


class DummyDistribution(BaseDistribution):
    """Minimal concrete distribution forwarding to the abstract base bodies."""

    @property
    def n_params(self) -> int:
        return super().n_params

    @property
    def fitted_repr(self) -> str:
        return super().fitted_repr

    def fit(self, X, y=None):
        return super().fit(X, y)

    def score_samples(self, X):
        return super().score_samples(X)


def test_base_distribution_is_abstract():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseDistribution()


def test_base_distribution_abstract_bodies_return_none():
    dist = DummyDistribution()
    X = np.zeros((3, 1))
    assert dist.n_params is None
    assert dist.fitted_repr is None
    assert dist.fit(X) is None
    assert dist.score_samples(X) is None
    # `sample` is not abstract but has no default implementation.
    assert dist.sample(n_samples=2) is None
