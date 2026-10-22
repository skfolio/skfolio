"""Independent reference cases and numerical derivatives for objective tests."""

import numpy as np

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


AUTOGRAD_MEASURE_MAPPING = {
    PerfMeasure.LOG_WEALTH: (_autograd_log_wealth, log_wealth),
    PerfMeasure.MEAN: (_autograd_mean, mean),
    RiskMeasure.VARIANCE: (_autograd_variance, variance),
    RiskMeasure.STANDARD_DEVIATION: (_autograd_standard_deviation, standard_deviation),
    RiskMeasure.SEMI_VARIANCE: (_autograd_semi_variance, semi_variance),
    RiskMeasure.SEMI_DEVIATION: (_autograd_semi_deviation, semi_deviation),
    RiskMeasure.MEAN_ABSOLUTE_DEVIATION: (
        _autograd_mean_absolute_deviation,
        mean_absolute_deviation,
    ),
    RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT: (
        _autograd_first_lower_partial_moment,
        first_lower_partial_moment,
    ),
    RiskMeasure.CVAR: (_autograd_cvar, cvar),
    RiskMeasure.EVAR: (_autograd_evar, ref_evar_from_returns),
    RiskMeasure.WORST_REALIZATION: (_autograd_worst_realization, worst_realization),
    RiskMeasure.GINI_MEAN_DIFFERENCE: (
        _autograd_gini_mean_difference,
        gini_mean_difference,
    ),
    RiskMeasure.CDAR: (_autograd_cdar, ref_cdar_from_returns),
    RiskMeasure.EDAR: (_autograd_edar, ref_edar_from_returns),
}


def finite_difference_gradient(function, weights, eps=1e-6):
    """Differentiate a scalar reference function with symmetric perturbations."""
    result = np.zeros_like(weights)
    for i in range(len(weights)):
        plus = weights.copy()
        minus = weights.copy()
        plus[i] += eps
        minus[i] -= eps
        result[i] = (function(plus) - function(minus)) / (2 * eps)
    return result
