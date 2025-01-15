import numpy as np
import pytest

from skfolio.distribution import GaussianCopula
from skfolio.distribution.copula.bivariate._base import _RHO_BOUNDS


@pytest.fixture
def X():
    # Using same convention as other libraries for Benchmark
    X = np.array(
        [
            [0.05, 0.1],
            [0.4, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.2, 0.3],
        ]
    )
    return X


@pytest.fixture
def fitted_model():
    # Using same convention as other libraries for Benchmark
    fitted_model = GaussianCopula()
    fitted_model.rho_ = 0.8090169943749475
    return fitted_model


def test_gaussian_copula_init():
    """Test initialization of GaussianCopula with default parameters."""
    model = GaussianCopula()
    assert model.use_kendall_tau_inversion is True
    assert model.kendall_tau is None


def test_gaussian_copula_fit_with_kendall_tau(random_data):
    """Test fit() using Kendall's tau inversion (default)."""
    model = GaussianCopula().fit(random_data)
    # Check that rho_ has been fitted and lies within the valid range
    assert hasattr(model, "rho_")
    assert _RHO_BOUNDS[0] < model.rho_ < _RHO_BOUNDS[1]


def test_gaussian_copula_fit_with_provided_kendall_tau(random_data):
    """Test fit() using a manually provided Kendall's tau."""
    model = GaussianCopula(use_kendall_tau_inversion=True, kendall_tau=0.25)
    model.fit(random_data)
    # Check that the resulting rho_ is sin(pi * 0.25 / 2) = sin(pi/8)
    expected_rho = np.sin(np.pi * 0.25 / 2.0)
    assert np.isclose(model.rho_, expected_rho)


def test_gaussian_copula_fit_mle(random_data):
    """Test fit() using MLE optimization for rho_."""
    model = GaussianCopula(use_kendall_tau_inversion=False)
    model.fit(random_data)
    # Just check it's between -1 and 1
    assert -1 < model.rho_ < 1


def test_gaussian_partial_derivative_shape(random_data):
    """Test partial_derivative() returns correct shape."""
    model = GaussianCopula().fit(random_data)
    h = model.partial_derivative(random_data)
    assert h.shape == (100,)
    # All values should remain in (0,1) for a well-behaved CDF
    assert np.all(h >= 0) and np.all(h <= 1)


def test_gaussian_inverse_partial_derivative_shape(random_data):
    """Test inverse_partial_derivative() returns correct shape."""
    model = GaussianCopula().fit(random_data)
    h_inv = model.inverse_partial_derivative(random_data)
    assert h_inv.shape == (100,)
    # Should lie within [0,1]
    assert np.all(h_inv >= 0) and np.all(h_inv <= 1)


def test_gaussian_score_samples(random_data):
    """Test score_samples() for shape and type."""
    model = GaussianCopula().fit(random_data)
    log_pdf = model.score_samples(random_data)
    assert log_pdf.shape == (100,)
    # log-pdf can be negative or positive, so we won't do a bound check here.


def test_gaussian_score(random_data):
    """Test the total log-likelihood via score()."""
    model = GaussianCopula().fit(random_data)
    total_ll = model.score(random_data)
    # It's a scalar
    assert isinstance(total_ll, float)


def test_gaussian_aic_bic(random_data):
    """Test AIC and BIC computation."""
    model = GaussianCopula().fit(random_data)
    aic_val = model.aic(random_data)
    bic_val = model.bic(random_data)

    # Both are floats
    assert isinstance(aic_val, float)
    assert isinstance(bic_val, float)

    # Typically, BIC >= AIC for large n, but not guaranteed. Just check they're finite.
    assert np.isfinite(aic_val)
    assert np.isfinite(bic_val)


def test_gaussian_sample():
    """Test sample() method for shape and range."""
    model = GaussianCopula().fit(np.random.rand(100, 2))
    samples = model.sample(n_samples=50, random_state=123)
    assert samples.shape == (50, 2)
    # Should lie strictly in (0,1).
    assert np.all(samples >= 1e-8) and np.all(samples <= 1 - 1e-8)


def test_gaussian_rho_out_of_bounds():
    """Check that setting rho_ out of bounds triggers an error in score_samples()."""
    model = GaussianCopula(use_kendall_tau_inversion=False)
    # Manually set rho_ out-of-bounds after fitting
    model.rho_ = 1.2
    with pytest.raises(ValueError, match="rho must be between -1 and 1."):
        _ = model.score_samples(np.random.rand(5, 2))


@pytest.mark.parametrize(
    "use_kendall_tau_inversion,expected",
    [
        [True, 0.8090169943749475],
        [False, 0.9273756082619797],
    ],
)
def test_gaussian_fit(X, use_kendall_tau_inversion, expected):
    model = GaussianCopula(use_kendall_tau_inversion=use_kendall_tau_inversion)
    model.fit(X)
    assert np.isclose(model.rho_, expected)


def test_gaussian_score_exact(X, fitted_model):
    assert np.isclose(fitted_model.score(X), 3.2732998035445258)
    np.testing.assert_almost_equal(
        fitted_model.score_samples(X),
        np.array([1.34908291, 0.29895068, 0.5212166, 0.47059694, 0.63345267]),
    )


def test_gaussian_aic_bic_exact(X, fitted_model):
    np.isclose(fitted_model.aic(X), -4.5465996070890515)
    np.isclose(fitted_model.bic(X), -4.937161694654951)


def test_gaussian_partial_derivative_exact(X, fitted_model):
    np.testing.assert_almost_equal(
        fitted_model.partial_derivative(X),
        np.array([0.15045411, 0.76650108, 0.29340619, 0.36365638, 0.23882845]),
    )


def test_gaussian_inverse_partial_derivative_exact(X, fitted_model):
    np.testing.assert_almost_equal(
        fitted_model.inverse_partial_derivative(X),
        np.array([0.02255551, 0.20332606, 0.30390676, 0.58119914, 0.17906309]),
    )


def test_gaussian_sample_exact(X, fitted_model):
    samples = fitted_model.sample(n_samples=5, random_state=42)

    np.testing.assert_almost_equal(
        samples,
        np.array(
            [
                [0.874587, 0.95071431],
                [0.71427177, 0.59865848],
                [0.07894734, 0.15599452],
                [0.4893664, 0.86617615],
                [0.72366369, 0.70807258],
            ]
        ),
    )


def test_gaussian_sample_refitting(X, fitted_model):
    samples = fitted_model.sample(n_samples=int(1e5), random_state=42)

    m1 = GaussianCopula().fit(samples)
    assert np.isclose(fitted_model.rho_, m1.rho_, 1e-3)

    m2 = GaussianCopula(use_kendall_tau_inversion=False).fit(samples)
    assert np.isclose(fitted_model.rho_, m2.rho_, 1e-3)
