"""Shared contracts for copula CDFs and joint survival probabilities."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from skfolio.distribution import (
    BaseBivariateCopula,
    ClaytonCopula,
    CopulaRotation,
    GaussianCopula,
    GumbelCopula,
    IndependentCopula,
    JoeCopula,
    StudentTCopula,
)

COPULA_CLASSES = (
    IndependentCopula,
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    GumbelCopula,
    JoeCopula,
)
PROBABILITY_METHODS = ("cdf", "survival_probability")
COPULA_CASES = [
    pytest.param((cls, rotation), id=f"{cls.__name__}-{rotation.name}")
    for cls in COPULA_CLASSES
    for rotation in (
        list(CopulaRotation)
        if cls in (ClaytonCopula, GumbelCopula, JoeCopula)
        else [CopulaRotation.R0]
    )
]


@pytest.fixture(params=COPULA_CASES)
def copula(request):
    cls, rotation = request.param
    model = cls()
    model.n_features_in_ = 2
    if cls in (GaussianCopula, StudentTCopula):
        model.rho_ = 0.5
    if cls is StudentTCopula:
        model.dof_ = 4.0
    if cls in (ClaytonCopula, GumbelCopula, JoeCopula):
        model.theta_ = 2.0
        model.rotation_ = rotation
    return model


@pytest.mark.parametrize("method", PROBABILITY_METHODS)
def test_probability_shape_and_input_preservation(copula, method):
    X = np.array([[0.1, 0.7], [0.9, 0.4], [0.5, 0.5]])
    original = X.copy()
    X.setflags(write=False)
    values = getattr(copula, method)(X)
    assert values.shape == (3,)
    assert values.dtype == np.float64
    assert getattr(copula, method)(X[:1].tolist()).shape == (1,)
    np.testing.assert_array_equal(X, original)


def test_probability_boundaries(copula):
    q = np.r_[0.0, 1e-12, np.linspace(0.01, 0.99, 11), 1 - 1e-12, 1.0]
    zero = np.zeros_like(q)
    one = np.ones_like(q)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for X, cdf, survival in (
            (np.column_stack((zero, q)), zero, 1 - q),
            (np.column_stack((q, zero)), zero, 1 - q),
            (np.column_stack((one, q)), q, zero),
            (np.column_stack((q, one)), q, zero),
        ):
            np.testing.assert_array_equal(copula.cdf(X), cdf)
            np.testing.assert_array_equal(copula.survival_probability(X), survival)


@pytest.mark.parametrize("method", PROBABILITY_METHODS)
@pytest.mark.parametrize(
    "X",
    [
        [[-0.1, 0.2]],
        [[0.2, 1.1]],
        [[np.nan, 0.2]],
        [[0.2, np.inf]],
        [[0.1, 0.2, 0.3]],
        [0.1, 0.2],
        np.empty((0, 2)),
    ],
)
def test_probability_invalid_input(copula, method, X):
    with pytest.raises(ValueError):
        getattr(copula, method)(X)


@pytest.mark.parametrize("cls", COPULA_CLASSES)
@pytest.mark.parametrize("method", PROBABILITY_METHODS)
def test_probability_requires_fitted_estimator(cls, method):
    with pytest.raises(NotFittedError):
        getattr(cls(), method)([[0.0, 1.0]])


def test_probability_identities_and_monotonicity(copula):
    q = np.array([0, 1e-12, 1e-9, 0.01, 0.3, 0.5, 0.9, 1 - 1e-12, 1])
    X = np.array(np.meshgrid(q, q)).reshape(2, -1).T
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cdf = copula.cdf(X)
        survival = copula.survival_probability(X)
    for values, margins, direction in ((cdf, X, 1), (survival, 1 - X, -1)):
        assert np.all(np.isfinite(values))
        assert np.all((values >= 0) & (values <= 1))
        lower = np.maximum(margins.sum(axis=1) - 1, 0)
        upper = margins.min(axis=1)
        assert np.all(values >= lower - 1e-10)
        assert np.all(values <= upper + 1e-10)
        grid = values.reshape(len(q), len(q))
        for axis in (0, 1):
            assert np.all(direction * np.diff(grid, axis=axis) >= -1e-10)
    np.testing.assert_allclose(survival, 1 - X.sum(axis=1) + cdf, atol=1e-10, rtol=0)


@pytest.mark.parametrize("quantiles", [[], [0], [1], [0, 1], [0.1, 0.5], [0.8, 0.95]])
def test_tail_concentration_subsets(copula, quantiles):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = copula.tail_concentration(quantiles)
    assert result.shape == (len(quantiles),)
    assert result.dtype == np.float64
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))
    np.testing.assert_array_equal(
        copula.tail_concentration([0, 1]),
        [copula.lower_tail_dependence, copula.upper_tail_dependence],
    )


def test_tail_concentration_preserves_quantile_order(copula):
    quantiles = np.array([0.95, 0.0, 0.5, 1.0, 0.1, 0.7])
    original = quantiles.copy()
    values = copula.tail_concentration(quantiles)
    order = np.argsort(quantiles)
    np.testing.assert_array_equal(
        copula.tail_concentration(quantiles[order]), values[order]
    )
    np.testing.assert_array_equal(quantiles, original)


def test_tail_concentration_bounds_near_endpoints(copula):
    q = np.array([3e-17, 1e-16, 1e-15, 5e-15, 1e-12, 1 - 1e-12, 1 - 1e-16])
    concentration = copula.tail_concentration(q)
    assert np.all(np.isfinite(concentration))
    assert np.all((concentration >= 0) & (concentration <= 1))


@pytest.mark.parametrize("cls", COPULA_CLASSES)
@pytest.mark.parametrize("quantiles", [[], [0], [1], [0, 1], [0.5]])
def test_tail_concentration_requires_fitted_estimator(cls, quantiles):
    with pytest.raises(NotFittedError):
        cls().tail_concentration(quantiles)


@pytest.mark.parametrize("quantiles", [0.5, [[0.2]], [-0.1], [1.1], [np.nan], [np.inf]])
def test_tail_concentration_invalid_input(copula, quantiles):
    with pytest.raises(ValueError):
        copula.tail_concentration(quantiles)


@pytest.mark.parametrize("cls", COPULA_CLASSES)
def test_probability_feature_names(cls):
    X = pd.DataFrame(
        np.random.default_rng(0).uniform(0.05, 0.95, size=(30, 2)),
        columns=["first", "second"],
    )
    model = cls().fit(X)
    thresholds = pd.DataFrame([[0.25, 0.75], [0.9, 0.95]], columns=X.columns)
    original = thresholds.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for method in PROBABILITY_METHODS:
            assert getattr(model, method)(thresholds).shape == (2,)
        model.tail_concentration([0, 0.1, 0.5, 0.9, 1])
        model.plot_tail_concentration()
    pd.testing.assert_frame_equal(thresholds, original)
    for method in PROBABILITY_METHODS:
        for invalid in (
            thresholds.iloc[:, ::-1],
            thresholds.rename(columns={"first": "other"}),
        ):
            with pytest.raises(ValueError, match="feature names"):
                getattr(model, method)(invalid)


class CustomCopula(BaseBivariateCopula):
    """Existing-style subclass implementing only the original abstract interface."""

    _n_params = 0
    fit = IndependentCopula.fit
    cdf = IndependentCopula.cdf
    partial_derivative = IndependentCopula.partial_derivative
    inverse_partial_derivative = IndependentCopula.inverse_partial_derivative
    score_samples = IndependentCopula.score_samples
    lower_tail_dependence = IndependentCopula.lower_tail_dependence
    upper_tail_dependence = IndependentCopula.upper_tail_dependence
    fitted_repr = IndependentCopula.fitted_repr


def test_base_survival_probability_default():
    X = pd.DataFrame([[0.25, 0.5], [0.75, 0.25]], columns=["first", "second"])
    model = CustomCopula().fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        np.testing.assert_array_equal(model.survival_probability(X), [0.375, 0.1875])


def test_independent_tail_probabilities():
    model = IndependentCopula().fit([[0.2, 0.3], [0.7, 0.8]])
    q = np.array([0.0, 2.0**-40, 0.25, 0.5, 0.75, 1 - 2.0**-40, 1.0])
    np.testing.assert_array_equal(model.tail_concentration(q), np.minimum(q, 1 - q))
    X = np.array([[2.0**-40, 0.25], [1 - 2.0**-40, 0.75]])
    np.testing.assert_array_equal(model.cdf(X), X[:, 0] * X[:, 1])
    np.testing.assert_array_equal(
        model.survival_probability(X), (1 - X[:, 0]) * (1 - X[:, 1])
    )


@pytest.mark.parametrize(
    "cls,theta",
    [
        (ClaytonCopula, 1e-4),
        (ClaytonCopula, 50.0),
        (GumbelCopula, 1.0001),
        (GumbelCopula, 80.0),
        (JoeCopula, 1.005),
        (JoeCopula, 20.0),
    ],
)
@pytest.mark.parametrize("rotation", list(CopulaRotation))
def test_archimedean_probability_extreme_parameters(cls, theta, rotation):
    model = cls()
    model.theta_, model.rotation_ = theta, rotation
    q = np.array(
        [
            np.nextafter(0.0, 1.0),
            1e-20,
            3e-17,
            1e-16,
            1e-15,
            5e-15,
            1e-12,
            1e-5,
            0.5,
            1 - 1e-5,
            1 - 1e-12,
            np.nextafter(1.0, 0.0),
        ]
    )
    X = np.array(np.meshgrid(q, q)).reshape(2, -1).T
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for method, margins in (("cdf", X), ("survival_probability", 1 - X)):
            values = getattr(model, method)(X)
            assert np.all(np.isfinite(values))
            assert np.all((values >= 0) & (values <= 1))
            assert np.all(values <= margins.min(axis=1))
            assert np.all(values >= np.maximum(margins.sum(axis=1) - 1, 0) - 1e-14)
        concentration = model.tail_concentration(q)
        assert np.all(np.isfinite(concentration))
        assert np.all((concentration >= 0) & (concentration <= 1))
