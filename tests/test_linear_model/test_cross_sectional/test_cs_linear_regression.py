"""Tests for CSLinearRegression estimator."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression

from skfolio.linear_model import CSLinearRegression


def make_data_with_nans(
    n_observations=8,
    n_assets=40,
    n_factors=5,
    seed=123,
    noise=0.05,
    nan_frac_X=0.03,
    nan_frac_y=0.04,
    zero_weight_frac=0.05,
):
    """Generate synthetic cross-sectional data with NaNs for testing."""
    rng = np.random.default_rng(seed)

    beta_true = rng.normal(size=(n_observations, n_factors))
    X = rng.normal(size=(n_observations, n_assets, n_factors))
    y = (X @ beta_true[..., None])[..., 0] + noise * rng.normal(
        size=(n_observations, n_assets)
    )
    weights = 1.0 + rng.random(size=(n_observations, n_assets))

    n_cells_X = max(1, int(nan_frac_X * n_observations * n_assets * n_factors))
    idx_t = rng.integers(0, n_observations, size=n_cells_X)
    idx_n = rng.integers(0, n_assets, size=n_cells_X)
    idx_k = rng.integers(0, n_factors, size=n_cells_X)
    X[idx_t, idx_n, idx_k] = np.nan

    n_cells_y = max(1, int(nan_frac_y * n_observations * n_assets))
    idx_t = rng.integers(0, n_observations, size=n_cells_y)
    idx_n = rng.integers(0, n_assets, size=n_cells_y)
    y[idx_t, idx_n] = np.nan

    missing_pairs = np.any(~np.isfinite(X), axis=2) | ~np.isfinite(y)
    weights[missing_pairs] = 0.0

    n_zero = max(1, int(zero_weight_frac * n_observations * n_assets))
    idx_t = rng.integers(0, n_observations, size=n_zero)
    idx_n = rng.integers(0, n_assets, size=n_zero)
    weights[idx_t, idx_n] = 0.0

    return X, y, weights, beta_true


def compute_valid_mask(X, y, weights):
    """Compute validity mask for (observation, asset) pairs.

    Under the weight-driven contract, positive-weight rows are guaranteed to
    have finite X and y, so this reduces to ``weights > 0``.
    """
    return np.isfinite(weights) & (weights > 0)


def sklearn_reference(X, y, weights, fit_intercept=True):
    """Reference implementation using sklearn's LinearRegression per observation."""
    n_observations, n_assets, n_factors = X.shape
    coef = np.zeros((n_observations, n_factors))
    intercept = np.zeros(n_observations)
    n_used = np.zeros(n_observations, dtype=int)
    predictions = np.full((n_observations, n_assets), np.nan)
    r2_per_obs = []

    valid = compute_valid_mask(X, y, weights)

    for t in range(n_observations):
        keep = valid[t]
        n_used[t] = keep.sum()
        X_t = X[t, keep, :]
        y_t = y[t, keep]
        w_t = weights[t, keep]
        lr = LinearRegression(fit_intercept=fit_intercept)
        lr.fit(X_t, y_t, sample_weight=w_t)
        coef[t] = lr.coef_
        intercept[t] = lr.intercept_ if fit_intercept else 0.0
        predictions[t, keep] = lr.predict(X_t)
        r2_per_obs.append(lr.score(X_t, y_t, sample_weight=w_t))

    score = np.nanmean(r2_per_obs) if r2_per_obs else np.nan
    return coef, intercept, n_used, predictions, score


@pytest.mark.parametrize(
    "n_observations,n_assets,n_factors",
    [(8, 40, 5), (5, 25, 3), (10, 50, 7), (20, 120, 30)],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_equivalence_to_sklearn(n_observations, n_assets, n_factors, fit_intercept):
    """CrossSectionalOLS must match sklearn for coef, predict, and score."""
    X, y, weights, _ = make_data_with_nans(
        n_observations=n_observations,
        n_assets=n_assets,
        n_factors=n_factors,
        seed=42,
    )
    model = CSLinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y, cs_weights=weights)

    coef_ref, intercept_ref, n_valid_assets_ref, predictions_ref, score_ref = (
        sklearn_reference(X, y, weights, fit_intercept=fit_intercept)
    )

    assert_allclose(model.n_valid_assets_, n_valid_assets_ref)
    assert_allclose(model.coef_, coef_ref, atol=1e-8, rtol=1e-6)
    assert_allclose(model.intercept_, intercept_ref, atol=1e-8, rtol=1e-6)

    predictions = model.predict(X)
    valid_preds = ~np.isnan(predictions_ref)
    assert_allclose(
        predictions[valid_preds], predictions_ref[valid_preds], atol=1e-8, rtol=1e-6
    )

    score = model.score(X, y, cs_weights=weights)
    if np.isnan(score_ref):
        assert np.isnan(score)
    else:
        assert_allclose(score, score_ref, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_underdetermined_system_returns_finite_outputs(fit_intercept):
    """Underdetermined systems must fit and predict without non-finite outputs."""
    n_observations = 4
    n_assets = 20
    n_factors = 30
    rng = np.random.default_rng(1234)
    beta_true = rng.normal(size=(n_observations, n_factors))
    X = rng.normal(size=(n_observations, n_assets, n_factors))
    y = (X @ beta_true[..., None])[..., 0] + 0.01 * rng.normal(
        size=(n_observations, n_assets)
    )
    weights = np.ones((n_observations, n_assets))
    model = CSLinearRegression(fit_intercept=fit_intercept)
    model.fit(X, y, cs_weights=weights)

    y_pred = model.predict(X)
    score = model.score(X, y, cs_weights=weights)

    assert model.coef_.shape == (n_observations, n_factors)
    assert model.intercept_.shape == (n_observations,)
    assert_allclose(model.n_valid_assets_, np.full(n_observations, n_assets))
    assert np.all(np.isfinite(model.coef_))
    assert np.all(np.isfinite(model.intercept_))
    assert np.all(np.isfinite(y_pred))
    assert np.isfinite(score)


def test_predict_matches_manual():
    """predict() must match manual einsum computation."""
    X, y, weights, _ = make_data_with_nans(
        n_observations=5, n_assets=20, n_factors=3, seed=123
    )
    model = CSLinearRegression(fit_intercept=True)
    model.fit(X, y, cs_weights=weights)
    y_pred = model.predict(X)
    y_pred_manual = np.einsum("tnk,tk->tn", X, model.coef_) + model.intercept_[:, None]
    assert_allclose(y_pred, y_pred_manual)


def test_score_computes_r2():
    """score() must compute weighted mean R2 correctly."""
    n_observations = 7
    X, y, weights, _ = make_data_with_nans(
        n_observations=n_observations, n_assets=30, n_factors=4, seed=7
    )
    model = CSLinearRegression()
    model.fit(X, y, cs_weights=weights)
    score_est = model.score(X, y, cs_weights=weights)

    y_pred = model.predict(X)
    valid = compute_valid_mask(X, y, weights)
    weights_masked = np.where(valid, weights, 0.0)
    y_masked = np.where(valid, y, 0.0)
    y_pred_masked = np.where(valid, y_pred, 0.0)

    weight_sums = weights_masked.sum(axis=1)
    y_mean = np.divide(
        (weights_masked * y_masked).sum(axis=1),
        weight_sums,
        out=np.full(n_observations, np.nan),
        where=weight_sums > 0,
    )

    ss_res = (weights_masked * (y_masked - y_pred_masked) ** 2).sum(axis=1)
    ss_tot = (weights_masked * (y_masked - y_mean[:, None]) ** 2).sum(axis=1)
    r2_per_obs = 1.0 - np.divide(
        ss_res,
        ss_tot,
        out=np.full(n_observations, np.nan),
        where=ss_tot > 0,
    )

    score_manual = np.mean(r2_per_obs)
    if np.isnan(score_manual):
        assert np.isnan(score_est)
    else:
        assert_allclose(score_est, score_manual, atol=1e-10, rtol=1e-8)


def test_score_ignores_observations_with_undefined_r2():
    """score() must ignore observations whose cross-sectional R2 is undefined."""
    X = np.array(
        [
            [[1.0], [2.0], [3.0]],
            [[5.0], [np.nan], [np.nan]],
        ]
    )
    y = np.array(
        [
            [2.0, 4.0, 6.0],
            [10.0, np.nan, np.nan],
        ]
    )
    cs_weights = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    model = CSLinearRegression()
    model.fit(X, y, cs_weights=cs_weights)

    score = model.score(X, y, cs_weights=cs_weights)

    assert_allclose(score, 1.0)


def test_handles_all_nans_gracefully():
    """The estimator must handle observations with all NaN/zero weights."""
    n_observations = 4
    X, y, weights, _ = make_data_with_nans(
        n_observations=n_observations, n_assets=20, n_factors=3, seed=99
    )
    X[2] = np.nan
    y[2] = np.nan
    weights[2] = 0.0

    model = CSLinearRegression()
    model.fit(X, y, cs_weights=weights)
    assert model.n_valid_assets_[2] == 0
    assert model.predict(X).shape == (n_observations, 20)


def test_zero_weight_pairs_allow_missing_data():
    """Zero-weight pairs may contain partially missing `X` or missing `y`."""
    X = np.array([[[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]])
    y = np.array([[np.nan, 1.0, 2.0]])
    weights = np.array([[0.0, 1.0, 1.0]])

    model = CSLinearRegression()
    model.fit(X, y, cs_weights=weights)

    assert model.n_valid_assets_[0] == 2


def test_positive_weight_nan_in_X_raises_error():
    """Positive-weight pairs with missing features must raise ValueError."""
    X = np.array([[[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]])
    y = np.array([[0.0, 1.0, 2.0]])
    weights = np.ones((1, 3))

    model = CSLinearRegression()
    with pytest.raises(ValueError, match="positive `cs_weights`"):
        model.fit(X, y, cs_weights=weights)


def test_positive_weight_nan_in_y_raises_error():
    """Positive-weight pairs with missing targets must raise ValueError."""
    X = np.array([[[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]]])
    y = np.array([[0.0, np.nan, 2.0]])
    weights = np.ones((1, 3))

    model = CSLinearRegression()
    with pytest.raises(ValueError, match="positive `cs_weights`"):
        model.fit(X, y, cs_weights=weights)


def test_no_intercept():
    """fit_intercept=False must produce zero intercepts."""
    rng = np.random.RandomState(42)
    X = rng.randn(5, 15, 3)
    y = rng.randn(5, 15)
    model = CSLinearRegression(fit_intercept=False)
    model.fit(X, y)
    assert_allclose(model.intercept_, np.zeros(5))


def test_negative_weights_raise_error():
    """Negative weights must raise ValueError."""
    rng = np.random.RandomState(42)
    X = rng.randn(3, 10, 2)
    y = rng.randn(3, 10)
    weights = rng.rand(3, 10)
    weights[0, 0] = -0.5
    model = CSLinearRegression()
    with pytest.raises(ValueError, match="Negative values"):
        model.fit(X, y, cs_weights=weights)


def test_nonfinite_weights_raise_error():
    """NaN/Inf weights must raise ValueError."""
    rng = np.random.RandomState(42)
    X = rng.randn(3, 10, 2)
    y = rng.randn(3, 10)
    weights = rng.rand(3, 10)
    weights[0, 0] = np.nan
    model = CSLinearRegression()
    with pytest.raises(ValueError, match="NaN"):
        model.fit(X, y, cs_weights=weights)


def test_wrong_weight_shape_raises_error():
    """Incorrect weight shapes must raise ValueError."""
    rng = np.random.RandomState(42)
    X = rng.randn(3, 10, 2)
    y = rng.randn(3, 10)

    model = CSLinearRegression()
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
        model.fit(X, y, cs_weights=rng.rand(3))

    with pytest.raises(ValueError, match="must have shape"):
        model.fit(X, y, cs_weights=rng.rand(3, 11))


def test_fit_requires_y():
    """fit must reject y=None explicitly."""
    rng = np.random.RandomState(42)
    X = rng.randn(3, 10, 2)
    model = CSLinearRegression()

    with pytest.raises(ValueError, match="requires y to be passed"):
        model.fit(X, None)


def test_score_requires_y():
    """score must reject y=None explicitly."""
    rng = np.random.RandomState(42)
    X = rng.randn(3, 10, 2)
    y = rng.randn(3, 10)
    model = CSLinearRegression()
    model.fit(X, y)

    with pytest.raises(ValueError, match="requires y to be passed"):
        model.score(X, None)


def test_score_positive_weight_nan_in_y_raises_error():
    """score must reject missing targets on positive-weight pairs."""
    X = np.array([[[1.0, 2.0], [2.0, 3.0], [4.0, 5.0]]])
    y_fit = np.array([[0.0, 1.0, 2.0]])
    y_score = np.array([[0.0, np.nan, 2.0]])
    weights = np.ones((1, 3))

    model = CSLinearRegression()
    model.fit(X, y_fit, cs_weights=weights)

    with pytest.raises(ValueError, match="positive `cs_weights`"):
        model.score(X, y_score, cs_weights=weights)


def test_predict_shape_validation():
    """predict must validate input shapes."""
    rng = np.random.RandomState(42)
    X = rng.randn(3, 10, 2)
    y = rng.randn(3, 10)
    model = CSLinearRegression()
    model.fit(X, y)

    with pytest.raises(ValueError, match="expecting 2 features as input"):
        model.predict(rng.randn(3, 10, 3))
    with pytest.raises(ValueError, match="observations but model was fitted"):
        model.predict(rng.randn(4, 10, 2))


def test_fitted_attributes():
    """All expected attributes must be set after fitting."""
    rng = np.random.RandomState(42)
    X = rng.randn(3, 10, 2)
    y = rng.randn(3, 10)
    model = CSLinearRegression()
    model.fit(X, y)

    assert model.coef_.shape == (3, 2)
    assert model.intercept_.shape == (3,)
    assert model.n_features_in_ == 2
    assert model.n_valid_assets_.shape == (3,)


def test_zero_weights_excluded():
    """Zero weights must exclude assets from regression."""
    rng = np.random.RandomState(42)
    X = rng.randn(2, 5, 2)
    y = rng.randn(2, 5)
    weights = np.ones((2, 5))
    weights[0, [0, 1]] = 0.0

    model = CSLinearRegression()
    model.fit(X, y, cs_weights=weights)
    assert model.n_valid_assets_[0] == 3
    assert model.n_valid_assets_[1] == 5
