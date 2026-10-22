import numpy as np
import pytest
from autograd import grad
from tests.test_optimization.test_online.objective_utils import (
    AUTOGRAD_MEASURE_MAPPING,
    finite_difference_gradient,
)

from skfolio.measures._enums import PerfMeasure, RiskMeasure
from skfolio.optimization.online._autograd_objectives import (
    _analytical_cdar_gradient,
    _analytical_cvar_gradient,
    _analytical_edar_gradient,
    _analytical_evar_gradient,
    _analytical_first_lower_partial_moment_gradient,
    _analytical_gini_mean_difference_gradient,
    _analytical_mean_absolute_deviation_gradient,
    _analytical_mean_gradient,
    _analytical_semi_deviation_gradient,
    _analytical_semi_variance_gradient,
    _analytical_standard_deviation_gradient,
    _analytical_variance_gradient,
    _analytical_worst_realization_gradient,
    _autograd_cdar,
    _autograd_cvar,
    _autograd_edar,
    _autograd_evar,
    _autograd_first_lower_partial_moment,
    _autograd_gini_mean_difference,
    _autograd_mean,
    _autograd_mean_absolute_deviation,
    _autograd_semi_deviation,
    _autograd_semi_variance,
    _autograd_standard_deviation,
    _autograd_variance,
    _autograd_worst_realization,
)


# Convert DataFrame to numpy for these tests
@pytest.fixture(scope="module")
def net_returns(X):
    """Convert DataFrame X to numpy array for testing."""
    return X.values


@pytest.fixture(scope="module")
def weights(net_returns):
    """Generate random portfolio weights."""
    np.random.seed(42)  # For reproducibility
    weights = np.random.randn(net_returns.shape[1])
    return weights / np.sum(np.abs(weights))


# Wrapper that makes autograd work with weights
def make_autograd_weight_function(measure_fn, net_returns, **kwargs):
    """Convert a portfolio_returns-based measure to a weights-based function."""

    def weight_objective(weights):
        portfolio_returns = net_returns @ weights
        return measure_fn(portfolio_returns, **kwargs)

    return weight_objective


@pytest.mark.parametrize("measure", AUTOGRAD_MEASURE_MAPPING, ids=str)
def test_autograd_objective_against_reference(net_returns, measure):
    """Test that autograd objective functions match skfolio.measures reference functions.

    Parameters
    ----------
    net_returns : ndarray
        Portfolio net returns (numpy array).
    measure : PerfMeasure | RiskMeasure
        Measure to test.
    """
    autograd_func, ref_func = AUTOGRAD_MEASURE_MAPPING[measure]
    expected = ref_func(net_returns)
    actual = autograd_func(net_returns)
    assert np.shape(expected) == np.shape(actual), (
        f"Shapes mismatch for metric {measure}"
    )
    np.testing.assert_allclose(
        actual, expected, err_msg=f"Autograd {measure} differs from reference"
    )


# Mapping from measure enums to their gradient implementations
ANALYTICAL_GRADIENT_MAPPING = {
    # Simple measures: can use autograd for verification
    PerfMeasure.MEAN: (
        _autograd_mean,
        _analytical_mean_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.VARIANCE: (
        _autograd_variance,
        _analytical_variance_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.STANDARD_DEVIATION: (
        _autograd_standard_deviation,
        _analytical_standard_deviation_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.SEMI_VARIANCE: (
        _autograd_semi_variance,
        _analytical_semi_variance_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.SEMI_DEVIATION: (
        _autograd_semi_deviation,
        _analytical_semi_deviation_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION: (
        _autograd_mean_absolute_deviation,
        _analytical_mean_absolute_deviation_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT: (
        _autograd_first_lower_partial_moment,
        _analytical_first_lower_partial_moment_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.CVAR: (
        _autograd_cvar,
        _analytical_cvar_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.WORST_REALIZATION: (
        _autograd_worst_realization,
        _analytical_worst_realization_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    RiskMeasure.GINI_MEAN_DIFFERENCE: (
        _autograd_gini_mean_difference,
        _analytical_gini_mean_difference_gradient,
        "autograd",
        1e-5,
        1e-7,
    ),
    # Advanced measures: use finite differences (autograd can't differentiate through grid search/cummax)
    RiskMeasure.EVAR: (
        _autograd_evar,
        _analytical_evar_gradient,
        "finite_diff",
        2e-3,
        2e-3,
    ),
    RiskMeasure.CDAR: (
        _autograd_cdar,
        _analytical_cdar_gradient,
        "finite_diff",
        1e-3,
        2e-3,
    ),
    RiskMeasure.EDAR: (
        _autograd_edar,
        _analytical_edar_gradient,
        "finite_diff",
        1e-3,
        3e-3,
    ),
}


@pytest.mark.parametrize("measure", ANALYTICAL_GRADIENT_MAPPING, ids=str)
def test_analytical_gradient(net_returns, weights, measure):
    """Test analytical gradients against autograd or finite differences.

    Parameters
    ----------
    net_returns : ndarray
        Portfolio net returns (numpy array).
    weights : ndarray
        Portfolio weights.
    measure : PerfMeasure | RiskMeasure
        Measure to test.
    """
    measure_fn, analytical_grad_fn, method, rtol, atol = ANALYTICAL_GRADIENT_MAPPING[
        measure
    ]

    if method == "autograd":
        # Create weight-based objective for autograd
        weight_objective = make_autograd_weight_function(measure_fn, net_returns)

        # Compute gradients
        autograd_grad = grad(weight_objective)(weights)
        analytical_grad = analytical_grad_fn(weights, net_returns)

        np.testing.assert_allclose(
            autograd_grad,
            analytical_grad,
            rtol=rtol,
            atol=atol,
            err_msg=f"Analytical gradient for {measure} differs from autograd",
        )
    elif method == "finite_diff":
        # Compute gradients via finite differences
        finite_diff_grad = finite_difference_gradient(
            lambda w: measure_fn(net_returns @ w), weights, eps=1e-6
        )
        analytical_grad = analytical_grad_fn(weights, net_returns)

        np.testing.assert_allclose(
            analytical_grad,
            finite_diff_grad,
            rtol=rtol,
            atol=atol,
            err_msg=f"Analytical gradient for {measure} differs from finite differences",
        )
