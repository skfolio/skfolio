"""Tests for CrossSectionalOLS estimator."""

import time

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.linear_model import LinearRegression

from skfolio.linear_model import CrossSectionalOLS


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
    """Generate synthetic cross-sectional data with NaNs for testing.

    Parameters
    ----------
    n_observations : int
        Number of observations (e.g., time periods).
    n_assets : int
        Number of assets.
    n_factors : int
        Number of factors.
    seed : int
        Random seed.
    noise : float
        Noise level in targets.
    nan_frac_X : float
        Fraction of X cells to set to NaN.
    nan_frac_y : float
        Fraction of y cells to set to NaN.
    zero_weight_frac : float
        Fraction of weights to set to zero.

    Returns
    -------
    X : ndarray of shape (n_observations, n_assets, n_factors)
    y : ndarray of shape (n_observations, n_assets)
    weights : ndarray of shape (n_observations, n_assets)
    beta_true : ndarray of shape (n_observations, n_factors)
    """
    rng = np.random.default_rng(seed)

    # True per-observation betas
    beta_true = rng.normal(size=(n_observations, n_factors))

    # Features
    X = rng.normal(size=(n_observations, n_assets, n_factors))

    # Targets: y = X @ beta + noise
    y = (X @ beta_true[..., None])[..., 0] + noise * rng.normal(
        size=(n_observations, n_assets)
    )

    # Base weights in (0, 2]
    weights = 1.0 + rng.random(size=(n_observations, n_assets))

    # Inject NaNs in X (random cells)
    n_cells_X = max(1, int(nan_frac_X * n_observations * n_assets * n_factors))
    idx_t = rng.integers(0, n_observations, size=n_cells_X)
    idx_n = rng.integers(0, n_assets, size=n_cells_X)
    idx_k = rng.integers(0, n_factors, size=n_cells_X)
    X[idx_t, idx_n, idx_k] = np.nan

    # Inject NaNs in y
    n_cells_y = max(1, int(nan_frac_y * n_observations * n_assets))
    idx_t = rng.integers(0, n_observations, size=n_cells_y)
    idx_n = rng.integers(0, n_assets, size=n_cells_y)
    y[idx_t, idx_n] = np.nan

    # Some zero weights
    n_zero = max(1, int(zero_weight_frac * n_observations * n_assets))
    idx_t = rng.integers(0, n_observations, size=n_zero)
    idx_n = rng.integers(0, n_assets, size=n_zero)
    weights[idx_t, idx_n] = 0.0

    return X, y, weights, beta_true


def _benchmark():
    X, y, weights, _ = make_data_with_nans(
        n_observations=3_000,
        n_assets=5_000,
        n_factors=70,
        seed=42,
    )
    print(3_000 * 5_000 * 70 / 1e9)  # 1.05B entries
    3000 / 252

    model = CrossSectionalOLS(fit_intercept=False)

    s = time.time()
    model.fit(X, y, sample_weight=weights)
    e = time.time()
    print(e - s)  # 7s
    assert (e - s) < 7.5


def compute_valid_mask(X, y, weights):
    """Compute validity mask for (observation, asset) pairs.

    A cell is valid if:
    - All factors in X are finite
    - y is finite
    - weight is finite and strictly positive

    Returns
    -------
    valid : ndarray of shape (n_observations, n_assets)
        Boolean mask of valid cells.
    """
    finite_X = np.all(np.isfinite(X), axis=2)
    finite_y = np.isfinite(y)
    valid_weights = np.isfinite(weights) & (weights > 0)
    return finite_X & finite_y & valid_weights


def sklearn_reference(X, y, weights, fit_intercept=True):
    """Reference implementation using sklearn's LinearRegression per observation.

    Parameters
    ----------
    X : ndarray of shape (n_observations, n_assets, n_factors)
    y : ndarray of shape (n_observations, n_assets)
    weights : ndarray of shape (n_observations, n_assets)
    fit_intercept : bool

    Returns
    -------
    coef : ndarray of shape (n_observations, n_factors)
    intercept : ndarray of shape (n_observations,)
    n_used : ndarray of shape (n_observations,)
    predictions : ndarray of shape (n_observations, n_assets)
    score : float
    """
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

        # Predictions for valid assets
        predictions[t, keep] = lr.predict(X_t)

        # R² for this observation
        r2_t = lr.score(X_t, y_t, sample_weight=w_t)
        r2_per_obs.append(r2_t)

    # Average R² across observations
    # Use regular mean (not nanmean) to propagate NaN like sklearn does
    score = np.nanmean(r2_per_obs) if r2_per_obs else np.nan

    return coef, intercept, n_used, predictions, score


@pytest.mark.parametrize(
    "n_observations,n_assets,n_factors",
    [(8, 40, 5), (5, 25, 3), (10, 50, 7), (20, 10, 30)],
)
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_equivalence_to_sklearn(n_observations, n_assets, n_factors, fit_intercept):
    """Test that CrossSectionalOLS matches sklearn's LinearRegression for coef, predict, and score."""
    X, y, weights, _ = make_data_with_nans(
        n_observations=n_observations,
        n_assets=n_assets,
        n_factors=n_factors,
        seed=42,
    )

    # Fit our estimator
    model = CrossSectionalOLS(fit_intercept=fit_intercept)

    model.fit(X, y, sample_weight=weights)

    # Reference implementation (returns coef, intercept, n_used, predictions, score)
    coef_ref, intercept_ref, n_used_ref, predictions_ref, score_ref = sklearn_reference(
        X, y, weights, fit_intercept=fit_intercept
    )

    # Check diagnostics
    assert model.n_used_.shape == (n_observations,)
    assert_allclose(model.n_used_, n_used_ref)

    # Check coefficients
    assert_allclose(
        model.coef_,
        coef_ref,
        atol=1e-8,
        rtol=1e-6,
    )
    assert_allclose(
        model.intercept_,
        intercept_ref,
        atol=1e-8,
        rtol=1e-6,
    )

    # Check predictions match sklearn
    predictions = model.predict(X)
    # Compare predictions where reference has valid predictions (not NaN)
    valid_preds = ~np.isnan(predictions_ref)
    assert_allclose(
        predictions[valid_preds],
        predictions_ref[valid_preds],
        atol=1e-8,
        rtol=1e-6,
        err_msg="Predictions don't match sklearn",
    )

    # Check score matches sklearn's average R²
    score = model.score(X, y, sample_weight=weights)

    if np.isnan(score_ref):
        assert np.isnan(score)
    else:
        assert_allclose(
            score,
            score_ref,
            atol=1e-8,
            rtol=1e-6,
            err_msg="Score doesn't match sklearn average R²",
        )


def test_predict_matches_manual():
    """Test that predict() gives correct results."""
    n_observations, n_assets, n_factors = 5, 20, 3
    X, y, weights, _ = make_data_with_nans(
        n_observations=n_observations,
        n_assets=n_assets,
        n_factors=n_factors,
        seed=123,
    )

    model = CrossSectionalOLS(fit_intercept=True)
    model.fit(X, y, sample_weight=weights)

    y_pred = model.predict(X)

    # Manual prediction using einsum
    y_pred_manual = np.einsum("tnk,tk->tn", X, model.coef_) + model.intercept_[:, None]

    assert_allclose(y_pred, y_pred_manual)


def test_score_computes_r2():
    """Test that score() computes weighted R² correctly."""
    n_observations, n_assets, n_factors = 7, 30, 4
    X, y, weights, _ = make_data_with_nans(
        n_observations=n_observations,
        n_assets=n_assets,
        n_factors=n_factors,
        seed=7,
    )

    model = CrossSectionalOLS()
    model.fit(X, y, sample_weight=weights)
    score_est = model.score(X, y, sample_weight=weights)

    # Manual R² computation (must match implementation)
    y_pred = model.predict(X)
    valid = compute_valid_mask(X, y, weights)
    weights_masked = np.where(valid, weights, 0.0)
    y_masked = np.where(valid, y, 0.0)
    y_pred_masked = np.where(valid, y_pred, 0.0)

    # Per-observation weighted R²
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

    # Use np.mean (not nanmean) to match sklearn behavior
    score_manual = np.mean(r2_per_obs)

    if np.isnan(score_manual):
        assert np.isnan(score_est)
    else:
        assert_allclose(score_est, score_manual, atol=1e-10, rtol=1e-8)


def test_handles_all_nans_gracefully():
    """Test that the estimator handles observations with all NaN/zero weights."""
    n_observations, n_assets, n_factors = 4, 20, 3
    X, y, weights, _ = make_data_with_nans(
        n_observations=n_observations,
        n_assets=n_assets,
        n_factors=n_factors,
        seed=99,
    )

    # Make one observation completely invalid
    bad_idx = 2
    X[bad_idx] = np.nan
    y[bad_idx] = np.nan
    weights[bad_idx] = 0.0

    model = CrossSectionalOLS()
    model.fit(X, y, sample_weight=weights)

    # Should have zero used assets for that observation
    assert model.n_used_[bad_idx] == 0

    # Predictions should still work (will be based on zero-data fit)
    y_pred = model.predict(X)
    assert y_pred.shape == (n_observations, n_assets)


def test_no_intercept():
    """Test fit_intercept=False."""
    n_observations, n_assets, n_factors = 5, 15, 3
    rng = np.random.RandomState(42)
    X = rng.randn(n_observations, n_assets, n_factors)
    y = rng.randn(n_observations, n_assets)

    model = CrossSectionalOLS(fit_intercept=False)
    model.fit(X, y)

    # Intercept should be zero
    assert_allclose(model.intercept_, np.zeros(n_observations))


def test_dataframe_input():
    """Test that DataFrame inputs work correctly."""
    n_observations, n_assets, n_factors = 3, 5, 2
    rng = np.random.RandomState(42)

    # Create DataFrame with MultiIndex columns
    dates = pd.date_range("2020-01-01", periods=n_observations)
    assets = [f"Asset_{i}" for i in range(n_assets)]
    factors = [f"Factor_{j}" for j in range(n_factors)]

    X_array = rng.randn(n_observations, n_assets, n_factors)
    y_array = rng.randn(n_observations, n_assets)

    # Convert to DataFrame
    multi_index = pd.MultiIndex.from_product([assets, factors])
    X_df = pd.DataFrame(
        X_array.reshape(n_observations, -1), index=dates, columns=multi_index
    )
    y_df = pd.DataFrame(y_array, index=dates, columns=assets)

    # Fit with DataFrame
    model_df = CrossSectionalOLS()
    model_df.fit(X_df, y_df)

    # Fit with array
    model_array = CrossSectionalOLS()
    model_array.fit(X_array, y_array)

    # Results should match
    assert_allclose(model_df.coef_, model_array.coef_)
    assert_allclose(model_df.intercept_, model_array.intercept_)

    # Check that labels are stored
    assert hasattr(model_df, "assets_")
    assert hasattr(model_df, "factors_")
    assert hasattr(model_df, "index_")
    assert len(model_df.assets_) == n_assets
    assert len(model_df.factors_) == n_factors


def test_negative_weights_raise_error():
    """Test that negative weights raise an error."""
    n_observations, n_assets, n_factors = 3, 10, 2
    rng = np.random.RandomState(42)
    X = rng.randn(n_observations, n_assets, n_factors)
    y = rng.randn(n_observations, n_assets)
    weights = rng.rand(n_observations, n_assets)
    weights[0, 0] = -0.5  # Negative weight

    model = CrossSectionalOLS()
    with pytest.raises(ValueError, match="negative value"):
        model.fit(X, y, sample_weight=weights)


def test_nonfinite_weights_raise_error():
    """Test that NaN/Inf weights raise an error."""
    n_observations, n_assets, n_factors = 3, 10, 2
    rng = np.random.RandomState(42)
    X = rng.randn(n_observations, n_assets, n_factors)
    y = rng.randn(n_observations, n_assets)
    weights = rng.rand(n_observations, n_assets)
    weights[0, 0] = np.nan

    model = CrossSectionalOLS()
    with pytest.raises(ValueError, match="non-finite value"):
        model.fit(X, y, sample_weight=weights)


def test_wrong_weight_shape_raises_error():
    """Test that incorrect weight shapes raise an error."""
    n_observations, n_assets, n_factors = 3, 10, 2
    rng = np.random.RandomState(42)
    X = rng.randn(n_observations, n_assets, n_factors)
    y = rng.randn(n_observations, n_assets)

    # Test 1D weights (fails during check_array coercion)
    weights_1d = rng.rand(n_observations)
    model = CrossSectionalOLS()
    with pytest.raises(ValueError, match="Expected 2D array, got 1D array"):
        model.fit(X, y, sample_weight=weights_1d)

    # Test wrong 2D shape (fails during shape check)
    weights_wrong_shape = rng.rand(n_observations, n_assets + 1)
    model = CrossSectionalOLS()
    with pytest.raises(ValueError, match="must have shape"):
        model.fit(X, y, sample_weight=weights_wrong_shape)


def test_predict_shape_validation():
    """Test that predict validates input shapes."""
    n_observations, n_assets, n_factors = 3, 10, 2
    rng = np.random.RandomState(42)
    X = rng.randn(n_observations, n_assets, n_factors)
    y = rng.randn(n_observations, n_assets)

    model = CrossSectionalOLS()
    model.fit(X, y)

    # Wrong number of factors
    X_wrong = rng.randn(n_observations, n_assets, n_factors + 1)
    with pytest.raises(ValueError, match="Expected n_factors"):
        model.predict(X_wrong)

    # Wrong number of observations
    X_wrong = rng.randn(n_observations + 1, n_assets, n_factors)
    with pytest.raises(ValueError, match="observations but model was fitted"):
        model.predict(X_wrong)


def test_fitted_attributes():
    """Test that all expected attributes are set after fitting."""
    n_observations, n_assets, n_factors = 3, 10, 2
    rng = np.random.RandomState(42)
    X = rng.randn(n_observations, n_assets, n_factors)
    y = rng.randn(n_observations, n_assets)

    model = CrossSectionalOLS()
    model.fit(X, y)

    # Check required attributes
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert hasattr(model, "n_features_in_")
    assert hasattr(model, "n_used_")

    # Check shapes
    assert model.coef_.shape == (n_observations, n_factors)
    assert model.intercept_.shape == (n_observations,)
    assert model.n_features_in_ == n_factors
    assert model.n_used_.shape == (n_observations,)


def test_zero_weights_excluded():
    """Test that zero weights exclude assets from regression."""
    n_observations, n_assets, n_factors = 2, 5, 2
    rng = np.random.RandomState(42)
    X = rng.randn(n_observations, n_assets, n_factors)
    y = rng.randn(n_observations, n_assets)
    weights = np.ones((n_observations, n_assets))

    # Set some weights to zero
    weights[0, [0, 1]] = 0.0

    model = CrossSectionalOLS()
    model.fit(X, y, sample_weight=weights)

    # First observation should use only 3 assets
    assert model.n_used_[0] == 3
    assert model.n_used_[1] == 5
