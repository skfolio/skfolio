"""Shared deterministic priors for optimizer tests."""

import numpy as np
import pytest

from skfolio.prior import BasePrior, ReturnDistribution


class FixedReturnDistributionPrior(BasePrior):
    def __init__(self, mu, covariance, sample_weight=None):
        self.mu = mu
        self.covariance = covariance
        self.sample_weight = sample_weight

    def fit(self, X, y=None):
        self.return_distribution_ = ReturnDistribution(
            mu=np.asarray(self.mu, dtype=float),
            covariance=np.asarray(self.covariance, dtype=float),
            returns=np.asarray(X, dtype=float),
            sample_weight=self.sample_weight,
        )
        return self


@pytest.fixture
def fixed_return_distribution_prior():
    return FixedReturnDistributionPrior
