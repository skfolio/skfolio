"""Tests for the abstract uncertainty-set base classes."""

from __future__ import annotations

import numpy as np

from skfolio.uncertainty_set import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
    EmpiricalCovarianceUncertaintySet,
)


class _MinimalMuUncertaintySet(BaseMuUncertaintySet):
    def __init__(self, prior_estimator=None):
        super().__init__(prior_estimator=prior_estimator)

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        return self


class _MinimalCovarianceUncertaintySet(BaseCovarianceUncertaintySet):
    def __init__(self, prior_estimator=None):
        super().__init__(prior_estimator=prior_estimator)

    def fit(self, X, y=None, **fit_params):
        super().fit(X, y, **fit_params)
        return self


def test_base_mu_uncertainty_set_abstract_fit_is_no_op():
    model = _MinimalMuUncertaintySet()
    assert model.fit(np.zeros((3, 2))) is model
    assert model.prior_estimator is None


def test_base_covariance_uncertainty_set_abstract_fit_is_no_op():
    model = _MinimalCovarianceUncertaintySet()
    assert model.fit(np.zeros((3, 2))) is model
    assert model.prior_estimator is None


def test_validate_X_y_without_y(X_small):
    model = EmpiricalCovarianceUncertaintySet()
    X_val, y_val = model._validate_X_y(X_small)
    assert X_val.shape == X_small.shape
    assert y_val is None
    assert model.n_features_in_ == X_small.shape[1]


def test_validate_X_y_with_y(X_small, factors):
    model = EmpiricalCovarianceUncertaintySet()
    y = factors.loc[X_small.index]
    X_val, y_val = model._validate_X_y(X_small, y)
    assert X_val.shape == X_small.shape
    assert y_val.shape == y.shape
