from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from skfolio.distribution import BaseMultivariateDist


class DummyMultivariate(BaseMultivariateDist):
    """Minimal concrete multivariate distribution sampling independent uniforms."""

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)

    @property
    def n_params(self) -> int:
        return 0

    @property
    def fitted_repr(self) -> str:
        return "Independent uniforms"

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array([f"x{i}" for i in range(X.shape[1])])
        return self

    def score_samples(self, X):
        return np.zeros(len(X))

    def sample(self, n_samples=1, conditioning=None):
        rng = np.random.default_rng(self.random_state)
        samples = rng.random((n_samples, self.n_features_in_))
        if conditioning is not None:
            for asset, value in conditioning.items():
                samples[:, asset] = value
        return samples


@pytest.fixture
def dummy_model():
    return DummyMultivariate(random_state=0).fit(np.zeros((5, 2)))


def test_base_multivariate_is_abstract():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseMultivariateDist()


def test_plot_scatter_matrix_wrong_columns(dummy_model):
    with pytest.raises(ValueError, match="X should have 2 columns"):
        dummy_model.plot_scatter_matrix(X=np.zeros((4, 3)))


def test_plot_scatter_matrix_small_X_adjusts_n_samples(dummy_model):
    X = np.random.default_rng(0).random((7, 2))
    fig = dummy_model.plot_scatter_matrix(X=X, n_samples=1000)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2
    # Generated sample size is matched to the number of rows of X.
    assert len(fig.data[1].dimensions[0]["values"]) == 7


def test_plot_scatter_matrix_conditioning_reverses_traces(dummy_model):
    X = np.random.default_rng(0).random((7, 2))
    fig = dummy_model.plot_scatter_matrix(X=X, conditioning={0: 0.5})
    assert fig.data[0].name == "Generated"
    assert fig.data[1].name == "Historical"
    np.testing.assert_array_equal(fig.data[0].dimensions[0]["values"], 0.5)
