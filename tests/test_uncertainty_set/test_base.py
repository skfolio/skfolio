"""Tests for the abstract uncertainty-set base classes."""

from __future__ import annotations

from skfolio.uncertainty_set import EmpiricalCovarianceUncertaintySet


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
