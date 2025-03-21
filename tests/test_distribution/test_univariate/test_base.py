import numpy as np
import pytest
from scipy.stats import norm

from skfolio.distribution import BaseUnivariateDist


class DummyUnivariate(BaseUnivariateDist):
    """Dummy univariate estimator using the standard normal distribution."""

    _scipy_model = norm

    def __init__(self, random_state):
        super().__init__(random_state=random_state)

    @property
    def _scipy_params(self) -> dict[str, float]:
        # Standard normal: mean 0, std 1.
        return {"loc": self.loc_, "scale": self.scale_}

    def fit(self):
        self.loc_ = 0
        self.scale_ = 1
        return self

    @property
    def fitted_repr(self) -> str:
        return "Standard Normal"


@pytest.fixture
def dummy_model():
    """Fixture for creating a fitted DummyUnivariate instance."""
    model = DummyUnivariate(random_state=42).fit()
    # "Fitting" in our context just means that scikit-learn's check_is_fitted will pass.
    # We simulate fitting by calling _validate_X once.
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    model._validate_X(X, reset=True)
    return model


def test_n_params(dummy_model):
    assert dummy_model.n_params == 2


def test_score_samples(dummy_model):
    """Test that score_samples returns log-density values."""
    X = np.array([[-1.0], [0.0], [1.0]])
    log_dens = dummy_model.score_samples(X)
    # For standard normal, logpdf at 0 should be approx -0.9189
    np.testing.assert_almost_equal(log_dens[1], norm.logpdf(0), decimal=5)
    assert log_dens.shape[0] == X.shape[0]


def test_score(dummy_model):
    """Test that score returns the sum of log-likelihoods."""
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    total_log_likelihood = dummy_model.score(X)
    np.testing.assert_almost_equal(
        total_log_likelihood, np.sum(dummy_model.score_samples(X)), decimal=5
    )


def test_sample(dummy_model):
    """Test that sample returns an array of the correct shape."""
    samples = dummy_model.sample(n_samples=10)
    assert samples.shape == (10, 1)
    # Check that samples are roughly in the range for a standard normal
    assert np.all(samples > -5) and np.all(samples < 5)


def test_cdf_ppf(dummy_model):
    """Test that cdf and ppf are inverses of each other."""
    probabilities = np.linspace(0.1, 0.9, 5)
    quantiles = dummy_model.ppf(probabilities)
    computed_probabilities = dummy_model.cdf(quantiles.reshape(-1, 1))
    np.testing.assert_allclose(
        computed_probabilities.flatten(), probabilities, atol=1e-5
    )


def test_plot_pdf(dummy_model):
    """Test that plot_pdf_2d returns a Plotly Figure with expected data."""
    fig = dummy_model.plot_pdf(title="Test PDF")
    # Check that figure has at least one trace
    assert len(fig.data) >= 1
    # Check that layout title matches
    assert "Test PDF" in fig.layout.title.text


def test_qq_plot(dummy_model):
    """Test that plot_pdf_2d returns a Plotly Figure with expected data."""
    fig = dummy_model.qq_plot(
        X=np.array([1, 2, 3, 4]).reshape(-1, 1), title="Test Q-Q Plot"
    )
    # Check that figure has at least one trace
    assert len(fig.data) >= 1
    # Check that layout title matches
    assert "Test Q-Q Plot" in fig.layout.title.text
