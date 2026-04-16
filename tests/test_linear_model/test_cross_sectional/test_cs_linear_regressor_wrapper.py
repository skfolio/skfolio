"""Tests for CSLinearRegressorWrapper estimator."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import HuberRegressor, Lasso, LinearRegression

from skfolio.linear_model import CSLinearRegressorWrapper


def make_data(
    n_observations: int = 6,
    n_assets: int = 20,
    n_factors: int = 4,
    seed: int = 123,
    noise: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic cross-sectional data."""
    rng = np.random.default_rng(seed)

    beta_true = rng.normal(size=(n_observations, n_factors))
    intercept_true = rng.normal(size=n_observations)
    X = rng.normal(size=(n_observations, n_assets, n_factors))
    y = (
        (X @ beta_true[..., None])[..., 0]
        + intercept_true[:, None]
        + noise * rng.normal(size=(n_observations, n_assets))
    )
    weights = 1.0 + rng.random(size=(n_observations, n_assets))
    return X, y, weights


class MissingFitInterceptRegressor(BaseEstimator, RegressorMixin):
    """Regressor missing the fit_intercept attribute."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: npt.ArrayLike | None = None,
    ) -> MissingFitInterceptRegressor:
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        self.intercept_ = 0.0
        return self


class MissingCoefRegressor(BaseEstimator, RegressorMixin):
    """Regressor missing the fitted coef_ attribute."""

    def __init__(self, fit_intercept: bool = False) -> None:
        self.fit_intercept = fit_intercept

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: npt.ArrayLike | None = None,
    ) -> MissingCoefRegressor:
        self.intercept_ = 0.0
        return self


class MissingInterceptRegressor(BaseEstimator, RegressorMixin):
    """Regressor missing the fitted intercept_ attribute."""

    def __init__(self, fit_intercept: bool = False) -> None:
        self.fit_intercept = fit_intercept

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: npt.ArrayLike | None = None,
    ) -> MissingInterceptRegressor:
        self.coef_ = np.zeros(X.shape[1], dtype=float)
        return self


def sklearn_reference(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    fit_intercept: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Fit one sklearn regression per observation."""
    n_observations, _, n_factors = X.shape
    coef = np.zeros((n_observations, n_factors))
    intercept = np.zeros(n_observations)
    n_valid_assets = np.zeros(n_observations, dtype=int)
    predictions = np.full(y.shape, np.nan)
    r2_per_obs = []

    for t in range(n_observations):
        keep = weights[t] > 0
        n_valid_assets[t] = keep.sum()
        model = LinearRegression(fit_intercept=fit_intercept)
        model.fit(X[t, keep], y[t, keep], sample_weight=weights[t, keep])
        coef[t] = model.coef_
        intercept[t] = model.intercept_
        predictions[t, keep] = model.predict(X[t, keep])
        r2_per_obs.append(
            model.score(X[t, keep], y[t, keep], sample_weight=weights[t, keep])
        )

    return coef, intercept, n_valid_assets, predictions, float(np.nanmean(r2_per_obs))


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_equivalence_to_sklearn_linear_regression(fit_intercept: bool) -> None:
    """Wrapper must match sklearn LinearRegression fitted per observation."""
    X, y, weights = make_data(seed=42)

    model = CSLinearRegressorWrapper(
        regressor=LinearRegression(fit_intercept=fit_intercept)
    )
    model.fit(X, y, cs_weights=weights)

    coef_ref, intercept_ref, n_valid_assets_ref, predictions_ref, score_ref = (
        sklearn_reference(X, y, weights, fit_intercept)
    )

    assert_allclose(model.coef_, coef_ref, atol=1e-8, rtol=1e-6)
    assert_allclose(model.intercept_, intercept_ref, atol=1e-8, rtol=1e-6)
    assert_allclose(model.n_valid_assets_, n_valid_assets_ref)
    assert_allclose(model.predict(X), predictions_ref, atol=1e-8, rtol=1e-6)
    assert_allclose(
        model.score(X, y, cs_weights=weights), score_ref, atol=1e-8, rtol=1e-6
    )


@pytest.mark.parametrize(
    "regressor",
    [
        Lasso(alpha=0.01, fit_intercept=True, max_iter=10000),
        HuberRegressor(fit_intercept=True, max_iter=200),
    ],
)
def test_sklearn_regressors_fit_successfully(regressor: BaseEstimator) -> None:
    """Wrapper must fit common sklearn regressors exposing the required contract."""
    X, y, weights = make_data(seed=21, noise=0.01)

    model = CSLinearRegressorWrapper(regressor=regressor)
    model.fit(X, y, cs_weights=weights)
    y_pred = model.predict(X)

    assert model.coef_.shape == (X.shape[0], X.shape[2])
    assert model.intercept_.shape == (X.shape[0],)
    assert model.n_valid_assets_.shape == (X.shape[0],)
    assert y_pred.shape == y.shape
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.intercept_))
    assert np.all(np.isfinite(y_pred))


def test_missing_fit_intercept_raises_error() -> None:
    """The wrapped regressor must define fit_intercept."""
    with pytest.raises(ValueError, match="define `fit_intercept`"):
        CSLinearRegressorWrapper(regressor=MissingFitInterceptRegressor())


def test_missing_fitted_coef_raises_error() -> None:
    """The wrapped regressor must expose coef_ after fit."""
    X, y, weights = make_data(seed=7)
    model = CSLinearRegressorWrapper(regressor=MissingCoefRegressor())

    with pytest.raises(ValueError, match="expose `coef_` after `fit`"):
        model.fit(X, y, cs_weights=weights)


def test_missing_fitted_intercept_raises_error() -> None:
    """The wrapped regressor must expose intercept_ after fit."""
    X, y, weights = make_data(seed=9)
    model = CSLinearRegressorWrapper(regressor=MissingInterceptRegressor())

    with pytest.raises(ValueError, match="expose `intercept_` after `fit`"):
        model.fit(X, y, cs_weights=weights)


def test_observation_without_valid_assets_raises_error() -> None:
    """Each observation must contain at least one valid positive-weight asset."""
    X, y, weights = make_data(seed=11)
    X[2] = np.nan
    y[2] = np.nan
    weights[2] = 0.0

    model = CSLinearRegressorWrapper(regressor=LinearRegression())

    with pytest.raises(ValueError, match="at least one asset with positive weight"):
        model.fit(X, y, cs_weights=weights)


def test_zero_weight_pairs_allow_missing_data_and_are_ignored() -> None:
    """Zero-weight pairs may contain missing data and are excluded from fit and score."""
    X = np.array(
        [
            [[1.0, 0.0], [2.0, 1.0], [np.nan, 3.0], [4.0, 2.0]],
            [[0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [3.0, np.nan]],
        ]
    )
    y = np.array(
        [
            [1.0, 3.0, np.nan, 6.0],
            [2.0, 5.0, 5.0, np.nan],
        ]
    )
    weights = np.array(
        [
            [1.0, 1.0, 0.0, 1.0],
            [1.0, 2.0, 1.0, 0.0],
        ]
    )

    model = CSLinearRegressorWrapper(regressor=LinearRegression(fit_intercept=True))
    model.fit(X, y, cs_weights=weights)

    coef_ref, intercept_ref, n_valid_assets_ref, predictions_ref, score_ref = (
        sklearn_reference(X, y, weights, fit_intercept=True)
    )

    assert_allclose(model.coef_, coef_ref, atol=1e-8, rtol=1e-6)
    assert_allclose(model.intercept_, intercept_ref, atol=1e-8, rtol=1e-6)
    assert_allclose(model.n_valid_assets_, n_valid_assets_ref)

    predictions = model.predict(X)
    valid_predictions = ~np.isnan(predictions_ref)
    assert_allclose(
        predictions[valid_predictions],
        predictions_ref[valid_predictions],
        atol=1e-8,
        rtol=1e-6,
    )
    assert np.all(np.isnan(predictions[~valid_predictions]))

    assert_allclose(
        model.score(X, y, cs_weights=weights), score_ref, atol=1e-8, rtol=1e-6
    )
