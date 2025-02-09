import numpy as np
import pytest

from skfolio.distribution import BaseBivariateCopula


@pytest.fixture
def random_data():
    """Fixture that returns a random numpy array in [0,1] of shape (100, 2)."""
    rng = np.random.default_rng(seed=42)
    return rng.random((100, 2))


def test_base_bivariate_copula_is_abstract():
    """Check that BaseBivariateCopula cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseBivariateCopula()


def test_validate_X_correct_shape(random_data):
    """Check _validate_X passes with correct input shape and range."""

    # We'll create a minimal subclass that just implements abstract methods.
    class MockCopula(BaseBivariateCopula):
        _n_params = 1

        def fit(self, X, y=None):
            return self

        def cdf(self, X):
            return np.zeros(X.shape[0])

        def partial_derivative(self, X, first_margin=False):
            return np.zeros(X.shape[0])

        def inverse_partial_derivative(self, X, first_margin=False):
            return np.zeros(X.shape[0])

        def score_samples(self, X):
            return np.zeros(X.shape[0])

        def fitted_repr(self):
            return "Copula"

    cop = MockCopula()
    X_validated = cop._validate_X(random_data, reset=True)
    assert X_validated.shape == (100, 2)
    # Check the data remain in [0,1] (they might be clipped slightly)
    assert np.all(X_validated >= 1e-8) and np.all(X_validated <= 1 - 1e-8)


def test_validate_X_wrong_shape():
    """Check _validate_X raises error if not exactly 2 columns."""

    class MockCopula(BaseBivariateCopula):
        _n_params = 1

        def fit(self, X, y=None):
            return self

        def cdf(self, X):
            return np.zeros(X.shape[0])

        def partial_derivative(self, X, first_margin=False):
            return np.zeros(X.shape[0])

        def inverse_partial_derivative(self, X, first_margin=False):
            return np.zeros(X.shape[0])

        def score_samples(self, X):
            return np.zeros(X.shape[0])

        def fitted_repr(self):
            return "Copula"

    cop = MockCopula()

    # 3 columns -> should fail
    with pytest.raises(ValueError, match="X must contains two columns"):
        data_3cols = np.random.rand(10, 3)
        cop._validate_X(data_3cols, reset=True)


def test_validate_X_out_of_bounds():
    """Check _validate_X raises error if values are out of [0,1]."""

    class MockCopula(BaseBivariateCopula):
        _n_params = 1

        def fit(self, X, y=None):
            return self

        def cdf(self, X):
            return np.zeros(X.shape[0])

        def partial_derivative(self, X, first_margin=False):
            return np.zeros(X.shape[0])

        def inverse_partial_derivative(self, X, first_margin=False):
            return np.zeros(X.shape[0])

        def score_samples(self, X):
            return np.zeros(X.shape[0])

        def fitted_repr(self):
            return "Copula"

    cop = MockCopula()
    data_negative = np.array([[0.2, -0.1], [0.3, 0.4]])
    with pytest.raises(ValueError, match="X must be in the interval"):
        cop._validate_X(data_negative, reset=True)
