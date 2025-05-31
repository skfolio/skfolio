import numpy as np
import pytest

from skfolio.distribution import Gaussian, StudentT, select_univariate_dist


@pytest.fixture
def gaussian_data():
    """
    Generate synthetic data from a standard normal distribution.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 1) with data drawn from N(0, 1).
    """
    np.random.seed(42)
    X = np.random.normal(loc=0.0, scale=1.0, size=1000).reshape(-1, 1)
    return X


def test_select_univariate_aic(gaussian_data):
    """
    Test that select_univariate_dist returns the Gaussian estimator when using AIC.

    Since the synthetic data are drawn from a normal distribution, the Gaussian estimator,
    having fewer parameters, should have a lower AIC than the StudentT estimator.
    """
    # Create candidate estimators with no fixed parameters.
    candidate_gaussian = Gaussian()
    candidate_student_t = StudentT()
    candidates = [candidate_gaussian, candidate_student_t]

    selected = select_univariate_dist(gaussian_data, candidates)
    # Expect the selected candidate to be an instance of Gaussian.
    assert isinstance(selected, Gaussian), (
        "Expected Gaussian estimator to be selected using AIC on Gaussian data."
    )


def test_select_univariate_bic(gaussian_data):
    """
    Test that select_univariate_dist returns the Gaussian estimator when using BIC.

    With data generated from a Gaussian distribution, the simpler model (Gaussian)
    should yield a lower BIC than the more complex StudentT estimator.
    """
    candidate_gaussian = Gaussian(loc=None, scale=None)
    candidate_student_t = StudentT(loc=None, scale=None)
    candidates = [candidate_gaussian, candidate_student_t]

    selected = select_univariate_dist(gaussian_data, candidates)
    assert isinstance(selected, Gaussian), (
        "Expected Gaussian estimator to be selected using BIC on Gaussian data."
    )


def test_invalid_X_shape():
    """
    Test that providing an input X with an invalid shape raises a ValueError.

    X must be a two-dimensional array with a single column.
    """
    # Create a one-dimensional array.
    X_invalid = np.array([1, 2, 3])
    candidate = Gaussian(loc=None, scale=None)
    with pytest.raises(
        ValueError, match="X must contains one column for Univariate Distributio"
    ):
        select_univariate_dist(X_invalid, [candidate])


def test_invalid_candidate_type(gaussian_data):
    """
    Test that providing a candidate that does not inherit from BaseUnivariateDist raises a ValueError.
    """
    invalid_candidate = "not a valid estimator"
    with pytest.raises(
        ValueError, match="Each candidate must inherit from `BaseUnivariateDist`"
    ):
        select_univariate_dist(gaussian_data, [invalid_candidate])
