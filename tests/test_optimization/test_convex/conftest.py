from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio.prior import BasePrior, ReturnDistribution


class FixedReturnDistributionPrior(BasePrior):
    def __init__(self, mu, covariance):
        self.mu = mu
        self.covariance = covariance

    def fit(self, X, y=None):
        self.return_distribution_ = ReturnDistribution(
            mu=np.asarray(self.mu, dtype=float),
            covariance=np.asarray(self.covariance, dtype=float),
            returns=np.asarray(X, dtype=float),
        )
        return self


@pytest.fixture
def fixed_return_distribution_prior():
    return FixedReturnDistributionPrior


@pytest.fixture
def nan_investable_test_data():
    X = pd.DataFrame(
        [
            [0.01, 0.02, np.nan, 0.03],
            [0.02, 0.01, np.nan, 0.01],
            [0.00, 0.03, np.nan, 0.02],
            [0.01, 0.01, np.nan, 0.04],
            [0.03, 0.02, np.nan, 0.01],
            [0.02, 0.00, np.nan, 0.03],
        ],
        columns=["A", "B", "C", "D"],
    )
    mu = np.array([0.01, 0.02, np.nan, 0.03])
    covariance = np.array(
        [
            [0.10, 0.01, np.nan, 0.02],
            [0.01, 0.08, np.nan, 0.01],
            [np.nan, np.nan, np.nan, np.nan],
            [0.02, 0.01, np.nan, 0.12],
        ]
    )
    investable_mask = np.array([True, True, False, True])
    return X, mu, covariance, investable_mask
