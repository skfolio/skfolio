import numpy as np
import pytest

from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._prediction import l1median_vazhz_vec

from ..utils import assert_box_budget


def _make_rmr(
    *,
    epsilon: float = 1.0,
    update_mode: str = "pa",
    rmr_window: int = 5,
    rmr_max_iter: int = 200,
    rmr_tolerance: float = 1e-9,
    **kwargs,
) -> FollowTheLoser:
    return FollowTheLoser(
        strategy="rmr",
        epsilon=epsilon,
        update_mode=update_mode,
        rmr_window=rmr_window,
        rmr_max_iter=rmr_max_iter,
        rmr_tolerance=rmr_tolerance,
        **kwargs,
    )


def test_l1_median_analytical_cases():
    """Test L1-median on known geometric median cases."""
    # Case 1: 3 collinear points -> median is middle point
    X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    median = l1median_vazhz_vec(X)
    np.testing.assert_allclose(median, [1.0, 1.0], atol=1e-8)

    # Case 2: equilateral triangle vertices -> centroid
    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    median = l1median_vazhz_vec(X)
    expected = np.array([0.5, np.sqrt(3) / 6])
    np.testing.assert_allclose(median, expected, atol=1e-6)


def test_l1_median_vs_scipy():
    """Compare L1-median against scipy minimize with L1 objective."""
    from scipy.optimize import minimize

    rng = np.random.RandomState(42)
    X = rng.randn(10, 3)

    # Our implementation
    our_median = l1median_vazhz_vec(X)

    # Scipy with L1 objective
    def l1_objective(y):
        return np.sum(np.linalg.norm(X - y, axis=1))

    result = minimize(l1_objective, x0=np.median(X, axis=0), method="BFGS")
    scipy_median = result.x

    # Should be very close
    np.testing.assert_allclose(our_median, scipy_median, rtol=1e-4)


def test_rmr_pa_runs_and_is_feasible(X_small):
    """Test RMR runs and produces feasible weights."""
    model = _make_rmr(rmr_window=3)
    model.fit(X_small)
    W = model.all_weights_
    assert W.shape == X_small.shape
    for w in W:
        assert_box_budget(w)


def test_rmr_cold_start_behavior():
    """Test RMR cold-start matches OLMAR-1 pattern."""
    # Use more extreme data to ensure deviation after cold-start
    X = np.array([[0.0, 0.0], [0.3, -0.2], [0.5, -0.4], [0.8, -0.6], [1.0, -0.8]])
    model = _make_rmr(rmr_window=3, epsilon=2.0)
    model.fit(X)

    # First 3 periods (t=0,1,2) should stay uniform
    # (cold-start until we have window+1=4 prices)
    W = model.all_weights_
    for t in range(3):
        np.testing.assert_allclose(W[t], [0.5, 0.5], atol=1e-10)

    # Period 3 onwards should allow updates (t >= window)
    # Check that at some point weights deviate from uniform
    assert not np.allclose(W[4], [0.5, 0.5], atol=0.05)


def test_rmr_handles_outliers_without_failure():
    """Test RMR handles data with outliers without numerical issues.

    RMR uses L1-median which is theoretically more robust to outliers
    than arithmetic mean (OLMAR). This test verifies RMR runs successfully
    on data with extreme outliers and produces valid portfolio weights.
    """
    # Data with extreme outlier
    X_outlier = np.array(
        [
            [0.0, 0.0],
            [0.02, -0.02],
            [0.02, -0.02],
            [0.02, -0.02],
            [5.0, -0.02],  # Extreme outlier in asset 1
            [0.02, -0.02],
        ]
    )

    # Build and fit RMR model
    model = _make_rmr(rmr_window=4, epsilon=1.5)
    model.fit(X_outlier)

    # Verify it produces valid weights
    W = model.all_weights_
    assert W.shape == X_outlier.shape
    for w in W:
        assert_box_budget(w)
        # Check no NaN or Inf values
        assert np.all(np.isfinite(w))

    # Verify L1-median computation doesn't fail with outliers
    test_data = np.array([[1.0, 1.0], [1.1, 0.9], [1.2, 0.8], [10.0, 1.0]])  # outlier
    median = l1median_vazhz_vec(test_data)
    assert np.all(np.isfinite(median))
    # Median should be closer to non-outlier points
    assert median[0] < 5.0  # Not pulled all the way to outlier


def test_rmr_param_validation():
    """Test RMR parameter validation."""
    X = np.array([[0.0, 0.0]])

    # Invalid window
    with pytest.raises(ValueError):
        model = _make_rmr(rmr_window=0)
        model.fit(X)

    # Invalid max_iter
    with pytest.raises(ValueError):
        model = _make_rmr(rmr_max_iter=0)
        model.fit(X)

    # Invalid tolerance
    with pytest.raises(ValueError):
        model = _make_rmr(rmr_tolerance=-1.0)
        model.fit(X)


def test_rmr_vs_olmar_on_clean_data():
    """On outlier-free data, RMR and OLMAR should behave similarly."""
    rng = np.random.RandomState(123)
    X = 0.01 * rng.randn(20, 3)

    rmr_model = _make_rmr(rmr_window=5)
    olmar_model = FollowTheLoser(
        strategy="olmar",
        olmar_predictor="sma",
        olmar_window=5,
        epsilon=1.0,
    )

    rmr_model.fit(X)
    olmar_model.fit(X)

    # Final weights should be correlated
    correlation = np.corrcoef(rmr_model.weights_, olmar_model.weights_)[0, 1]
    assert correlation > 0.7  # Should be similar


def test_rmr_md_mode_runs(X_small):
    """Test RMR with mirror descent mode."""
    model = _make_rmr(update_mode="md", rmr_window=3, mirror="euclidean")
    model.fit(X_small)
    W = model.all_weights_
    assert W.shape == X_small.shape
    for w in W:
        assert_box_budget(w)
