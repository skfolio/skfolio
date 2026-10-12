import numpy as np
import pytest
from autograd import grad

from skfolio.measures import (
    cdar,
    cvar,
    edar,
    evar,
    first_lower_partial_moment,
    get_drawdowns,
    gini_mean_difference,
    log_wealth,
    mean,
    mean_absolute_deviation,
    semi_deviation,
    semi_variance,
    standard_deviation,
    variance,
    worst_realization,
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
    _autograd_log_wealth,
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


def _finite_difference_gradient_from_measure(
    measure_fn, weights, net_returns, eps=1e-6, **kwargs
):
    """Compute gradient using finite differences for a measure function.

    Parameters
    ----------
    measure_fn : callable
        Measure function that takes portfolio_returns and returns a scalar.
    weights : ndarray
        Portfolio weights.
    net_returns : ndarray
        Net returns matrix (T, n_assets).
    eps : float
        Perturbation size for finite differences.
    **kwargs
        Additional parameters for measure_fn.

    Returns
    -------
    ndarray
        Gradient vector computed via finite differences.
    """
    grad_numerical = np.zeros_like(weights)
    for i in range(len(weights)):
        weights_plus = weights.copy()
        weights_plus[i] += eps
        weights_minus = weights.copy()
        weights_minus[i] -= eps

        portfolio_returns_plus = net_returns @ weights_plus
        portfolio_returns_minus = net_returns @ weights_minus

        loss_plus = measure_fn(portfolio_returns_plus, **kwargs)
        loss_minus = measure_fn(portfolio_returns_minus, **kwargs)
        grad_numerical[i] = (loss_plus - loss_minus) / (2 * eps)

    return grad_numerical


# For drawdown-based measures, autograd functions accept returns and compute drawdowns internally,
# while reference functions expect pre-computed drawdowns. Create wrappers for consistent testing.
def ref_cdar_from_returns(returns, beta=0.95):
    """Wrapper to compute CDaR from returns (computes drawdowns first)."""
    drawdowns = get_drawdowns(returns, compounded=False)
    return cdar(drawdowns, beta=beta)


def ref_edar_from_returns(returns, beta=0.95):
    """Wrapper to compute EDaR from returns (computes drawdowns first).

    The reference edar expects a 1D drawdown series; if a 2D matrix is provided,
    compute EDaR per asset (along axis=0).
    """
    drawdowns = get_drawdowns(returns, compounded=False)
    drawdowns = np.asarray(drawdowns)
    if drawdowns.ndim == 1:
        return edar(drawdowns, beta=beta)
    n = drawdowns.shape[1]
    out = np.empty(n, dtype=float)
    for j in range(n):
        out[j] = edar(drawdowns[:, j], beta=beta)
    return out


def ref_evar_from_returns(returns, beta=0.95):
    """Wrapper to compute EVaR from returns for 1D or 2D inputs.

    The reference evar expects a 1D series; if a 2D matrix is provided,
    compute EVaR per asset (along axis=0).
    """
    returns = np.asarray(returns)
    if returns.ndim == 1:
        return evar(returns, beta=beta)
    n = returns.shape[1]
    out = np.empty(n, dtype=float)
    for j in range(n):
        out[j] = evar(returns[:, j], beta=beta)
    return out


# Mapping from measure enums to their implementations
AUTOGRAD_MEASURE_MAPPING = {
    PerfMeasure.LOG_WEALTH: (_autograd_log_wealth, log_wealth, None),
    PerfMeasure.MEAN: (_autograd_mean, mean, None),
    RiskMeasure.VARIANCE: (_autograd_variance, variance, None),
    RiskMeasure.STANDARD_DEVIATION: (
        _autograd_standard_deviation,
        standard_deviation,
        None,
    ),
    RiskMeasure.SEMI_VARIANCE: (_autograd_semi_variance, semi_variance, None),
    RiskMeasure.SEMI_DEVIATION: (_autograd_semi_deviation, semi_deviation, None),
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION: (
        _autograd_mean_absolute_deviation,
        mean_absolute_deviation,
        None,
    ),
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT: (
        _autograd_first_lower_partial_moment,
        first_lower_partial_moment,
        None,
    ),
    RiskMeasure.CVAR: (_autograd_cvar, cvar, None),
    RiskMeasure.EVAR: (
        _autograd_evar,
        ref_evar_from_returns,
        1e-3,
    ),  # Grid approximation
    RiskMeasure.WORST_REALIZATION: (
        _autograd_worst_realization,
        worst_realization,
        None,
    ),
    RiskMeasure.GINI_MEAN_DIFFERENCE: (
        _autograd_gini_mean_difference,
        gini_mean_difference,
        None,
    ),
    RiskMeasure.CDAR: (_autograd_cdar, ref_cdar_from_returns, None),
    RiskMeasure.EDAR: (
        _autograd_edar,
        ref_edar_from_returns,
        1e-3,
    ),  # Grid approximation
}


@pytest.mark.parametrize(
    "measure",
    [
        pytest.param(m, id=str(m))
        for m in [
            PerfMeasure.LOG_WEALTH,
            PerfMeasure.MEAN,
            RiskMeasure.VARIANCE,
            RiskMeasure.STANDARD_DEVIATION,
            RiskMeasure.SEMI_VARIANCE,
            RiskMeasure.SEMI_DEVIATION,
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
            RiskMeasure.CVAR,
            RiskMeasure.EVAR,
            RiskMeasure.WORST_REALIZATION,
            RiskMeasure.GINI_MEAN_DIFFERENCE,
            RiskMeasure.CDAR,
            RiskMeasure.EDAR,
        ]
    ],
)
def test_autograd_objective_against_reference(net_returns, measure):
    """Test that autograd objective functions match skfolio.measures reference functions.

    Parameters
    ----------
    net_returns : ndarray
        Portfolio net returns (numpy array).
    measure : PerfMeasure | RiskMeasure
        Measure to test.
    """
    autograd_func, ref_func, special_rtol = AUTOGRAD_MEASURE_MAPPING[measure]
    assert np.shape(ref_func(net_returns)) == np.shape(autograd_func(net_returns)), (
        f"Shapes mismatch for metric {measure}"
    )
    # Use relaxed tolerance for measures with approximations (EVaR, EDaR)
    if special_rtol is not None:
        np.testing.assert_allclose(
            ref_func(net_returns),
            autograd_func(net_returns),
            rtol=special_rtol,
            err_msg=f"Autograd {measure} differs from reference (grid approximation)",
        )
    else:
        np.testing.assert_allclose(
            ref_func(net_returns),
            autograd_func(net_returns),
            err_msg=f"Autograd {measure} differs from reference",
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


@pytest.mark.parametrize(
    "measure",
    [
        pytest.param(m, id=str(m))
        for m in [
            PerfMeasure.MEAN,
            RiskMeasure.VARIANCE,
            RiskMeasure.STANDARD_DEVIATION,
            RiskMeasure.SEMI_VARIANCE,
            RiskMeasure.SEMI_DEVIATION,
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
            RiskMeasure.CVAR,
            RiskMeasure.WORST_REALIZATION,
            RiskMeasure.GINI_MEAN_DIFFERENCE,
            RiskMeasure.EVAR,
            RiskMeasure.CDAR,
            RiskMeasure.EDAR,
        ]
    ],
)
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
        finite_diff_grad = _finite_difference_gradient_from_measure(
            measure_fn=measure_fn, weights=weights, net_returns=net_returns, eps=1e-6
        )
        analytical_grad = analytical_grad_fn(weights, net_returns)

        np.testing.assert_allclose(
            analytical_grad,
            finite_diff_grad,
            rtol=rtol,
            atol=atol,
            err_msg=f"Analytical gradient for {measure} differs from finite differences",
        )
