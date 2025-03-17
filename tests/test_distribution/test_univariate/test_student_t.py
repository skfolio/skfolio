import numpy as np
import pytest
from scipy.stats import t

from skfolio.distribution import StudentT


@pytest.fixture(scope="module")
def student_t_model():
    """
    Fixture for creating and fitting a StudentT estimator.

    Synthetic data is generated from a Student's t distribution with known parameters.
    For example, we set degrees of freedom (df)=5, loc=0.0, and scale=1.0.
    """
    # Known parameters for the synthetic Student's t distribution.
    df_true = 5
    loc_true = 0.0
    scale_true = 1.0

    np.random.seed(42)
    X = t.rvs(df_true, loc=loc_true, scale=scale_true, size=1000).reshape(-1, 1)

    # Create the estimator with both parameters free (they will be estimated).
    model = StudentT(random_state=123)
    model.fit(X)
    return model


def test_fit_estimates_parameters(student_t_model):
    """
    Test that the fit method estimates parameters reasonably close to the true values.

    For synthetic data generated with loc=0.0, scale=1.0 and df=5, the fitted
    location and scale should be close to these values.
    """
    # Verify that the location and scale estimates are near the true values.
    np.testing.assert_allclose(student_t_model.dof_, 5.0, atol=0.2)
    np.testing.assert_allclose(student_t_model.loc_, 0.0, atol=0.2)
    np.testing.assert_allclose(student_t_model.scale_, 1.0, atol=0.2)


def test_scipy_params(student_t_model):
    """
    Test that scipy_params returns the correct dictionary of fitted parameters.
    """
    params = student_t_model._scipy_params
    np.testing.assert_allclose(params["loc"], student_t_model.loc_)
    np.testing.assert_allclose(params["scale"], student_t_model.scale_)
    np.testing.assert_allclose(params["df"], student_t_model.dof_)


def test_fitted_repr(student_t_model):
    """
    Test that fitted_repr returns a valid string representation of the model.
    """
    repr_str = student_t_model.fitted_repr
    assert isinstance(repr_str, str)
    assert "StudentT(" in repr_str


def test_score_samples(student_t_model):
    """
    Test that score_samples returns log-density values that match SciPy's t.logpdf.
    """
    X_test = np.array([[-1.0], [0.0], [1.0]])
    log_densities = student_t_model.score_samples(X_test)
    expected = t.logpdf(X_test, **student_t_model._scipy_params).ravel()
    np.testing.assert_allclose(log_densities, expected, rtol=1e-5)
    assert log_densities.shape[0] == X_test.shape[0]


def test_score(student_t_model):
    """
    Test that score returns the sum of log-likelihoods from score_samples.
    """
    X_test = np.linspace(-3, 3, 50).reshape(-1, 1)
    total_log_likelihood = student_t_model.score(X_test)
    np.testing.assert_almost_equal(
        total_log_likelihood, np.sum(student_t_model.score_samples(X_test)), decimal=5
    )


def test_sample(student_t_model):
    """
    Test that sample generates an array with the correct shape and finite values.
    """
    samples = student_t_model.sample(n_samples=20)
    assert samples.shape == (20, 1)
    # Ensure samples are finite
    assert np.all(np.isfinite(samples))


def test_cdf_ppf(student_t_model):
    """
    Test that cdf and ppf are approximately inverse functions for the fitted model.
    """
    probabilities = np.linspace(0.1, 0.9, 5)
    quantiles = student_t_model.ppf(probabilities)
    computed_probs = student_t_model.cdf(quantiles.reshape(-1, 1))
    np.testing.assert_allclose(computed_probs.flatten(), probabilities, atol=1e-5)


def test_plot_pdf(student_t_model):
    """
    Test that plot_pdf_2d returns a Plotly Figure with expected layout and data.
    """
    fig = student_t_model.plot_pdf(title="Student's t PDF")
    # Check that the figure has at least one trace and that the title is correct.
    assert len(fig.data) >= 1
    assert "Student's t PDF" in fig.layout.title.text
