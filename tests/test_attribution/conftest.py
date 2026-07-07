import numpy as np
import pytest

from ._utils import _create_realized_model


@pytest.fixture
def simple_factor_model():
    """Create a simple 3-asset, 2-factor model for testing."""
    return {
        "weights": np.array([0.4, 0.35, 0.25]),
        "loading_matrix": np.array([[1.0, 0.5], [0.5, 1.0], [0.3, 0.2]]),
        "factor_covariance": np.array([[0.04, 0.01], [0.01, 0.02]]),
        "idio_covariance": np.array([0.01, 0.015, 0.02]),
        "asset_names": ["IBM", "JPM", "AAPL"],
        "factor_names": ["Momentum", "Value"],
    }


@pytest.fixture
def simple_factor_model_with_perf(simple_factor_model):
    """Add performance inputs to the simple factor model."""
    return {
        **simple_factor_model,
        "factor_mu": np.array([0.05, 0.03]),
        "idio_mu": np.array([0.01, 0.005, -0.002]),
    }


@pytest.fixture
def model_with_families(simple_factor_model_with_perf):
    """Add factor families to the model."""
    return {
        **simple_factor_model_with_perf,
        "factor_families": ["Style", "Value"],
    }


@pytest.fixture
def multi_factor_model():
    """Create a 4-asset, 3-factor model with families for testing aggregation."""
    return {
        "weights": np.ones(4) / 4,
        "loading_matrix": np.eye(4, 3),
        "factor_covariance": np.eye(3) * 0.01,
        "idio_covariance": np.ones(4) * 0.005,
        "factor_names": ["Mom", "Val", "Size"],
        "asset_names": ["A1", "A2", "A3", "A4"],
        "factor_families": ["Style", "Style", "Industry"],
    }


@pytest.fixture
def multi_factor_model_with_perf(multi_factor_model):
    """Add performance inputs to the multi-factor model."""
    return {
        **multi_factor_model,
        "factor_mu": np.array([0.05, 0.03, 0.02]),
        "idio_mu": np.ones(4) * 0.005,
    }


@pytest.fixture
def static_realized_model():
    """Create a model with static exposures and weights for realized attribution."""
    return _create_realized_model(
        n_obs=100,
        n_assets=5,
        n_factors=3,
        static_exposures=True,
        static_weights=True,
        include_observations=False,
    )


@pytest.fixture
def time_varying_realized_model():
    """Create a model with time-varying exposures and weights."""
    return _create_realized_model(
        n_obs=100,
        n_assets=5,
        n_factors=3,
        static_exposures=False,
        static_weights=False,
        include_observations=False,
    )


@pytest.fixture
def rolling_static_model():
    """Create a model with static exposures/weights for rolling attribution (longer)."""
    return _create_realized_model(
        n_obs=200,
        n_assets=5,
        n_factors=3,
        static_exposures=True,
        static_weights=True,
        include_observations=True,
    )


@pytest.fixture
def rolling_time_varying_model():
    """Create a model with time-varying exposures/weights for rolling attribution."""
    return _create_realized_model(
        n_obs=200,
        n_assets=5,
        n_factors=3,
        static_exposures=False,
        static_weights=False,
        include_observations=True,
    )
