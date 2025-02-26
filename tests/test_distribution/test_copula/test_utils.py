import numpy as np
import plotly.graph_objects as go
import pytest

from skfolio.distribution.copula._utils import (
    CopulaRotation,
    _apply_copula_rotation,
    _apply_margin_swap,
    _apply_rotation_cdf,
    _apply_rotation_partial_derivatives,
    _select_rotation_itau,
    _select_theta_and_rotation_mle,
    compute_pseudo_observations,
    empirical_tail_concentration,
    plot_tail_concentration,
)


def dummy_func(X, theta):
    # Return a simple function of theta and the mean of X.
    return abs(theta - np.mean(X))


def dummy_neg_log_likelihood(theta, X):
    # A simple quadratic function: minimum at theta=2.
    return (theta - 2) ** 2


def dummy_cdf(X, **kwargs):
    # Return sum of columns divided by 2.
    return np.sum(X, axis=1) / 2


def dummy_partial_derivative(X, first_margin, **kwargs):
    # Return constant 0.5 for simplicity.
    return np.full(X.shape[0], 0.5)


def test_compute_pseudo_observations():
    X = np.array([[3, 1], [2, 4], [5, 3]])
    pseudo_obs = compute_pseudo_observations(X)
    # Check shape
    assert pseudo_obs.shape == X.shape
    # Check values are strictly between 0 and 1.
    assert np.all((pseudo_obs > 0) & (pseudo_obs < 1))


def test_empirical_tail_concentration():
    # Create a simple 2-column pseudo-observation array.
    X = np.array([[0.1, 0.2], [0.2, 0.25], [0.3, 0.35], [0.4, 0.45]])
    quantiles = np.linspace(0.0, 1.0, 5)
    concentration = empirical_tail_concentration(X, quantiles)
    # Check output shape
    assert concentration.shape == quantiles.shape
    # Concentration values should be non-negative.
    assert np.all(concentration >= 0)


def test_empirical_tail_concentration_raise():
    with pytest.raises(
        ValueError, match="X must be a 2D array with exactly 2 columns."
    ):
        X = np.array(
            [[0.1, 0.2, 0.2], [0.2, 0.25, 0.2], [0.3, 0.35, 0.2], [0.4, 1.5, 0.2]]
        )
        quantiles = np.linspace(0.0, 1.0, 5)
        _ = empirical_tail_concentration(X, quantiles)

    with pytest.raises(
        ValueError, match="X must be pseudo-observation in the interval"
    ):
        X = np.array([[0.1, 0.2], [0.2, 0.25], [0.3, 0.35], [0.4, 1.5]])
        quantiles = np.linspace(0.0, 1.0, 5)
        _ = empirical_tail_concentration(X, quantiles)

    with pytest.raises(ValueError, match="quantiles must be between 0.0 and 1.0."):
        X = np.array([[0.1, 0.2], [0.2, 0.25], [0.3, 0.35], [0.4, 0.5]])
        quantiles = np.linspace(0.0, 1.5, 5)
        _ = empirical_tail_concentration(X, quantiles)


def test_plot_tail_concentration():
    # Create a dummy tail concentration dict.
    quantiles = np.linspace(0.0, 1.0, 50)
    dummy_concentration = np.linspace(0.2, 0.8, 50)
    tail_concentration_dict = {"Dummy": dummy_concentration}
    fig = plot_tail_concentration(
        tail_concentration_dict,
        quantiles,
        title="Test Tail Concentration",
        smoothing=0.5,
    )
    assert isinstance(fig, go.Figure)
    # Check that x-axis ticks are formatted as percentages
    assert fig.layout.xaxis.tickformat == ".0%"
    fig = plot_tail_concentration(
        tail_concentration_dict,
        quantiles,
        title="Test Tail Concentration",
        smoothing=None,
    )
    assert isinstance(fig, go.Figure)

    with pytest.raises(
        ValueError, match="The smoothing parameter must be between 0 and 1.3."
    ):
        _ = plot_tail_concentration(
            tail_concentration_dict,
            quantiles,
            title="Test Tail Concentration",
            smoothing=5,
        )


def test_select_rotation_itau():
    # Create dummy data and test that the function returns a valid rotation.
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    theta = 2.0
    rotation = _select_rotation_itau(dummy_func, X, theta)
    # Check that the returned rotation is a member of CopulaRotation.
    assert isinstance(rotation, CopulaRotation)


def test_select_theta_and_rotation_mle():
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    bounds = (0.5, 3.0)
    theta, rotation = _select_theta_and_rotation_mle(
        dummy_neg_log_likelihood, X, bounds, tolerance=1e-3
    )
    # Expect theta to be close to 2.0 given the dummy function.
    assert np.isclose(theta, 2.0, atol=1e-2)
    # Check that rotation is a valid CopulaRotation.
    assert isinstance(rotation, CopulaRotation)


def test_apply_copula_rotation():
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    # R0 should not change X.
    X_r0 = _apply_copula_rotation(X, CopulaRotation.R0)
    np.testing.assert_allclose(X, X_r0)
    # R180 should be 1 - X.
    X_r180 = _apply_copula_rotation(X, CopulaRotation.R180)
    np.testing.assert_allclose(X_r180, 1 - X)
    # Test R90 and R270 (they swap columns with adjustments)
    X_r90 = _apply_copula_rotation(X, CopulaRotation.R90)
    np.testing.assert_allclose(X_r90[:, 0], X[:, 1])
    np.testing.assert_allclose(X_r90[:, 1], 1 - X[:, 0])
    X_r270 = _apply_copula_rotation(X, CopulaRotation.R270)
    np.testing.assert_allclose(X_r270[:, 0], 1 - X[:, 1])
    np.testing.assert_allclose(X_r270[:, 1], X[:, 0])


def test_apply_margin_swap():
    X = np.array([[0.1, 0.2], [0.3, 0.4]])
    # When first_margin is True, columns should be swapped.
    swapped = _apply_margin_swap(X, True)
    np.testing.assert_allclose(swapped, np.hstack((X[:, [1]], X[:, [0]])))
    # When first_margin is False, no swap is performed.
    not_swapped = _apply_margin_swap(X, False)
    np.testing.assert_allclose(not_swapped, X)


def test_apply_rotation_cdf():
    X = np.array([[0.2, 0.3], [0.4, 0.5]])
    # Test for rotation R0; the dummy cdf returns mean of columns.
    cdf_r0 = _apply_rotation_cdf(dummy_cdf, X, CopulaRotation.R0)
    expected = dummy_cdf(X)
    np.testing.assert_allclose(cdf_r0, expected)
    # For other rotations, we don't have a closed-form expected value,
    # so we just check that output has correct shape.
    cdf_r90 = _apply_rotation_cdf(dummy_cdf, X, CopulaRotation.R90)
    assert cdf_r90.shape == (X.shape[0],)


def test_apply_rotation_partial_derivatives():
    X = np.array([[0.2, 0.3], [0.4, 0.5]])
    # Using dummy_partial_derivative which returns 0.5 for all.
    pd_r0 = _apply_rotation_partial_derivatives(
        dummy_partial_derivative, X, CopulaRotation.R0, first_margin=True
    )
    assert np.allclose(pd_r0, 0.5)
    pd_r90 = _apply_rotation_partial_derivatives(
        dummy_partial_derivative, X, CopulaRotation.R90, first_margin=True
    )
    assert np.allclose(pd_r90, 1 - 0.5)
