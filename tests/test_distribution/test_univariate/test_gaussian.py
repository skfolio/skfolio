import numpy as np
import pytest
from scipy.stats import norm

from skfolio.distribution import Gaussian


@pytest.fixture(scope="module")
def gaussian_model():
    """Fixture for creating and fitting a Gaussian estimator on synthetic data."""
    # Create synthetic data drawn from a standard normal distribution.
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=500).reshape(-1, 1)
    # Instantiate Gaussian estimator with both parameters free.
    model = Gaussian(random_state=123)
    model.fit(X)
    return model


def test_fit_estimates_parameters(gaussian_model):
    """Test that the fit method estimates parameters close to the true values."""
    # For data generated with loc=0, scale=1, estimates should be close.
    assert np.abs(gaussian_model.loc_) < 0.2
    assert np.abs(gaussian_model.scale_ - 1.0) < 0.2


def test_scipy_params(gaussian_model):
    """Test that scipy_params returns the correct fitted parameter dictionary."""
    params = gaussian_model._scipy_params
    np.testing.assert_allclose(params["loc"], gaussian_model.loc_)
    np.testing.assert_allclose(params["scale"], gaussian_model.scale_)


def test_fitted_repr(gaussian_model):
    """Test that fitted_repr returns a non-empty string representation."""
    repr_str = gaussian_model.fitted_repr
    assert isinstance(repr_str, str)
    assert "Gaussian(" in repr_str


def test_score_samples(gaussian_model):
    """Test that score_samples returns log-density values close to SciPy's norm.logpdf."""
    X_test = np.array([[-1.0], [0.0], [1.0]])
    log_densities = gaussian_model.score_samples(X_test)
    expected = norm.logpdf(
        X_test, loc=gaussian_model.loc_, scale=gaussian_model.scale_
    ).ravel()
    np.testing.assert_allclose(log_densities, expected, rtol=1e-5)
    assert log_densities.shape[0] == X_test.shape[0]


def test_score(gaussian_model):
    """Test that score returns the sum of log-likelihoods from score_samples."""
    X_test = np.linspace(-2, 2, 50).reshape(-1, 1)
    total_log_likelihood = gaussian_model.score(X_test)
    np.testing.assert_almost_equal(
        total_log_likelihood, np.sum(gaussian_model.score_samples(X_test)), decimal=5
    )


def test_sample(gaussian_model):
    """Test that sample generates an array with the correct shape and reasonable values."""
    samples = gaussian_model.sample(n_samples=20)
    assert samples.shape == (20, 1)
    # Check that the samples roughly lie within a reasonable range for a normal distribution.
    assert np.all(samples > -5) and np.all(samples < 5)


def test_cdf_ppf(gaussian_model):
    """Test that cdf and ppf are inverses for the fitted model."""
    probabilities = np.linspace(0.1, 0.9, 5)
    quantiles = gaussian_model.ppf(probabilities)
    computed_probs = gaussian_model.cdf(quantiles.reshape(-1, 1))
    np.testing.assert_allclose(computed_probs.flatten(), probabilities, atol=1e-5)


def test_plot_pdf(gaussian_model):
    """Test that plot_pdf_2d returns a Plotly Figure with expected layout and data."""
    fig = gaussian_model.plot_pdf(title="Gaussian PDF")
    # Verify that at least one trace is present and the title is set correctly.
    assert len(fig.data) >= 1
    assert "Gaussian PDF" in fig.layout.title.text


def test_qq_plot(gaussian_model):
    samples = gaussian_model.sample(n_samples=20)
    fig = gaussian_model.qq_plot(samples, title="Gaussian QQ")
    # Verify that at least one trace is present and the title is set correctly.
    assert len(fig.data) >= 1
    assert "Gaussian QQ" in fig.layout.title.text
