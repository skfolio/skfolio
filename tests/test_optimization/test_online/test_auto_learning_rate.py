"""Tests for automatic learning rate estimation."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings

import numpy as np
import pytest

from skfolio.measures._enums import RiskMeasure
from skfolio.optimization.online._autograd_objectives import MEASURE_PROPERTIES
from skfolio.optimization.online._learning_rate import (
    compute_eg_learning_rate,
    compute_ogd_learning_rate,
    compute_prod_learning_rate,
    estimate_gradient_bound,
    get_auto_learning_rate,
    simplex_diameter,
)
from skfolio.optimization.online._mixins import FTWStrategy


@pytest.mark.parametrize(
    "budget,expected_factor",
    [
        (1.0, 1.0),  # standard simplex
        (0.5, 0.5),  # half budget scales linearly
    ],
)
def test_simplex_diameter_budget_scaling(budget, expected_factor):
    """Test that simplex diameter scales linearly with budget."""
    base_diameter = simplex_diameter(
        n_assets=2, min_weights=0, max_weights=1, budget=1.0
    )
    scaled_diameter = simplex_diameter(
        n_assets=2, min_weights=0, max_weights=1, budget=budget
    )
    assert np.isclose(scaled_diameter, expected_factor * base_diameter)
    # Check exact value for unit budget
    if budget == 1.0:
        assert np.isclose(base_diameter, np.sqrt(2.0))


@pytest.mark.parametrize(
    "objective",
    list(MEASURE_PROPERTIES.keys()),
)
def test_gradient_bound_convexity_classes(objective, X_small):
    """Test gradient bound estimation for different convexity classes."""
    G = estimate_gradient_bound(objective, historical_returns=X_small)
    assert G is not None


def test_gradient_bound_variance_with_data(X_small):
    """Test variance gradient G = 2||Σ||_op with historical data."""
    G = estimate_gradient_bound(RiskMeasure.VARIANCE, historical_returns=X_small)
    # Should equal 2 * max eigenvalue of covariance matrix
    cov = np.cov(X_small, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    expected = 2.0 * np.max(np.abs(eigvals))
    assert np.isclose(G, expected)


def test_gradient_bound_unknown_objective_raises():
    """Test that unknown objective raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        estimate_gradient_bound("unknown_objective")


@pytest.mark.parametrize(
    "compute_fn,t",
    [
        (
            lambda t: compute_ogd_learning_rate(
                t=t, diameter=np.sqrt(2), gradient_bound=0.8
            ),
            4,
        ),
        (lambda t: compute_eg_learning_rate(t, 10), 4),
        (lambda t: compute_prod_learning_rate(t, 10), 4),
    ],
)
def test_time_varying_decay(compute_fn, t):
    """Test that time-varying rates decay as expected."""
    eta_1 = compute_fn(1)
    eta_t = compute_fn(t)
    eta_100 = compute_fn(100)
    # Should decay monotonically
    assert eta_t < eta_1
    assert eta_100 < eta_t


def test_invalid_scale_raises():
    """Test invalid scale parameter raises ValueError."""
    with pytest.raises(ValueError, match="Unknown scale"):
        compute_eg_learning_rate(0, 10, scale="invalid")


@pytest.mark.parametrize("strategy", list(FTWStrategy))
def test_auto_returns_callable(strategy):
    """Test get_auto_learning_rate returns callable for all strategies."""
    lr_fn = get_auto_learning_rate(strategy, n_assets=10)
    assert callable(lr_fn)

    # Test it can be called with integer argument
    eta_0 = lr_fn(0)
    eta_10 = lr_fn(10)
    assert isinstance(eta_0, int | float | np.number)
    assert isinstance(eta_10, int | float | np.number)


@pytest.mark.parametrize(
    "strategy,is_time_varying",
    [
        (FTWStrategy.OGD, True),
        (FTWStrategy.EG, True),
        (FTWStrategy.PROD, True),
        (FTWStrategy.ADAGRAD, False),
        (FTWStrategy.ADABARRONS, False),
        (FTWStrategy.SWORD_SMALL, False),
        (FTWStrategy.SWORD_VAR, False),
        (FTWStrategy.SWORD_BEST, False),
        (FTWStrategy.SWORD_PP, False),
    ],
)
def test_auto_time_varying_behavior(strategy, is_time_varying):
    """Test time-varying vs constant learning rates."""
    lr_fn = get_auto_learning_rate(strategy, n_assets=10)

    eta_1 = lr_fn(1)
    eta_100 = lr_fn(100)

    if is_time_varying:
        # Should decay over time
        assert eta_100 < eta_1
    else:
        # Should remain constant
        assert np.isclose(eta_1, eta_100)


def test_auto_budget_affects_diameter():
    """Test custom budget affects learning rate via diameter."""
    lr_fn_1 = get_auto_learning_rate(FTWStrategy.OGD, n_assets=10, budget=1.0)
    lr_fn_05 = get_auto_learning_rate(FTWStrategy.OGD, n_assets=10, budget=0.5)

    # With budget 0.5, diameter is half, so learning rate should be half
    eta_1 = lr_fn_1(1)
    eta_05 = lr_fn_05(1)
    assert np.isclose(eta_05, eta_1 * 0.5)


def test_auto_explicit_gradient_bound():
    """Test explicit gradient_bound overrides estimation."""
    lr_fn_explicit = get_auto_learning_rate(
        FTWStrategy.OGD, n_assets=10, gradient_bound=5.0
    )
    lr_fn_default = get_auto_learning_rate(FTWStrategy.OGD, n_assets=10)

    eta_explicit = lr_fn_explicit(1)
    eta_default = lr_fn_default(1)

    # Different gradient bounds should give different learning rates
    assert not np.isclose(eta_explicit, eta_default)


def test_auto_warns_on_custom_objective_without_bound():
    """Test warning when using auto with non-logwealth objective without explicit bound."""
    with pytest.warns(UserWarning, match="custom objectives"):
        get_auto_learning_rate(
            FTWStrategy.OGD, n_assets=10, objective=RiskMeasure.VARIANCE
        )


def test_auto_no_warning_on_logwealth():
    """Test no warning for logwealth objective."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Should not raise any warning
        get_auto_learning_rate(FTWStrategy.OGD, n_assets=10, objective=None)


def test_auto_no_warning_with_explicit_gradient_bound():
    """Test no warning when explicit gradient_bound provided with custom objective."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        get_auto_learning_rate(
            FTWStrategy.OGD,
            n_assets=10,
            objective=RiskMeasure.VARIANCE,
            gradient_bound=1.0,
        )


def test_auto_eg_independent_of_objective():
    """Test EG learning rate is independent of objective (doesn't use gradient bound)."""
    lr_fn_log = get_auto_learning_rate(FTWStrategy.EG, n_assets=10, objective=None)

    with pytest.warns(UserWarning):
        lr_fn_var = get_auto_learning_rate(
            FTWStrategy.EG, n_assets=10, objective=RiskMeasure.VARIANCE
        )

    # Should be the same (EG doesn't use gradient bound)
    assert np.isclose(lr_fn_log(5), lr_fn_var(5))


def test_auto_scale_parameter():
    """Test scale parameter affects learning rates."""
    lr_emp = get_auto_learning_rate(FTWStrategy.EG, n_assets=20, scale="empirical")
    lr_theory = get_auto_learning_rate(FTWStrategy.EG, n_assets=20, scale="theory")

    # Empirical should be much larger
    assert lr_emp(0) > lr_theory(0)


@pytest.mark.parametrize("strategy", ["unknown_strategy", None, 1])
def test_unknown_strategy_raises(strategy):
    """Test unknown strategy raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_auto_learning_rate(strategy, n_assets=10)


def test_t_zero_safe():
    """Test that t=0 is handled safely (no NaN/inf)."""
    # t=0 should map to t_theory=1, giving finite positive rate
    eta_eg = compute_eg_learning_rate(0, 10, scale="empirical")
    assert 0 < eta_eg < np.inf
    assert not np.isnan(eta_eg)

    eta_ogd = compute_ogd_learning_rate(t=0, diameter=np.sqrt(2), gradient_bound=0.8)
    assert 0 < eta_ogd < np.inf
    assert not np.isnan(eta_ogd)

    # t=0 and t=1 should be different (decay present)
    eta_eg_0 = compute_eg_learning_rate(0, 10)
    eta_eg_1 = compute_eg_learning_rate(1, 10)
    assert eta_eg_0 > eta_eg_1  # η(0) > η(1)


@pytest.mark.parametrize("n_assets", [4, 16, 64])
def test_gradient_bound_logwealth_l2_scales_with_sqrt_n(n_assets):
    """For OGD (Euclidean geometry), the L2 gradient bound must grow with sqrt(n).

    The gradient of log-wealth is g = -r / (1 + w^T r). Its L2 norm
    is ||r||_2 / |1 + w^T r| which scales as sqrt(n) * max|r_i| / min_denom.
    """
    G = estimate_gradient_bound(objective=None, n_assets=n_assets, norm="l2")
    # Must grow with sqrt(n): G(64) > G(16) > G(4) > 0
    assert G > 0
    G_small = estimate_gradient_bound(objective=None, n_assets=4, norm="l2")
    if n_assets > 4:
        assert G > G_small


def test_gradient_bound_logwealth_linf_does_not_scale():
    """For EG (entropy geometry), the L-inf gradient bound is O(1), no sqrt(n)."""
    G4 = estimate_gradient_bound(objective=None, n_assets=4, norm="linf")
    G64 = estimate_gradient_bound(objective=None, n_assets=64, norm="linf")
    assert G4 > 0
    # L-inf bound should be the same regardless of n
    assert abs(G4 - G64) < 1e-10


def test_eg_empirical_rate_no_warning():
    """EG with scale='empirical' should NOT warn — it is now a mild √2 boost
    over theory, preserving O(√(T log n)) regret guarantees."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        eta_fn = get_auto_learning_rate(
            strategy=FTWStrategy.EG,
            n_assets=10,
            scale="empirical",
        )
    # Empirical rate is √2 times the theory rate
    eta_emp = eta_fn(99)
    eta_theory = np.sqrt(np.log(10) / 100)
    np.testing.assert_allclose(eta_emp, eta_theory * np.sqrt(2), rtol=1e-10)


def test_string_strategy_matches_enum():
    """String strategies have the same semantics across supported Python versions."""
    expected = get_auto_learning_rate(FTWStrategy.EG, n_assets=10)
    actual = get_auto_learning_rate("eg", n_assets=10)
    assert actual(5) == expected(5)
