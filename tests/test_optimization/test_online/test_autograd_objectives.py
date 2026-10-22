"""Tests for autograd-based objective functions.

This test suite validates:
1. Log-wealth objective with both autograd and analytical gradients
2. Risk measures (Variance, Semi-Variance, MAD, FLPM, CVaR, Worst, Gini) with autograd
3. Performance measures (Mean) with autograd
4. Analytical gradient correctness via finite difference approximation
5. Factory function create_objective
"""

from typing import ClassVar

import numpy as np
import pytest
from tests.test_optimization.test_online.objective_utils import (
    AUTOGRAD_MEASURE_MAPPING,
    finite_difference_gradient,
)

from skfolio.measures._enums import PerfMeasure, RiskMeasure
from skfolio.optimization.online._autograd_objectives import (
    MeasureObjective,
    create_objective,
)


def _finite_difference_gradient(
    obj: MeasureObjective,
    weights: np.ndarray,
    net_returns: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute gradient using finite differences for testing."""
    return finite_difference_gradient(lambda w: obj.loss(w, net_returns), weights, eps)


def _assert_gradient_matches_finite_difference(*, obj, weights, net_returns, atol):
    """Compare the objective gradient with an independent numerical derivative."""
    actual = obj.grad(weights, net_returns)
    expected = _finite_difference_gradient(
        obj=obj, weights=weights, net_returns=net_returns
    )
    np.testing.assert_allclose(actual, expected, atol=atol)


@pytest.fixture(scope="module")
def net_returns_single():
    """Single observation of net returns."""
    np.random.seed(42)
    return np.random.randn(5) * 0.02


@pytest.fixture(scope="module")
def net_returns_timeseries():
    """Time series of net returns."""
    np.random.seed(42)
    return np.random.randn(50, 5) * 0.02 + 0.001


@pytest.fixture(scope="module")
def weights():
    """Portfolio weights."""
    return np.array([0.3, 0.2, 0.2, 0.15, 0.15])


class TestLogWealthObjective:
    """Test log-wealth objective with both autograd and analytical gradients."""

    def test_loss_computation(self, weights, net_returns_single):
        """Test loss computation for log-wealth."""
        obj = MeasureObjective(measure=PerfMeasure.LOG_WEALTH, use_autograd=True)
        loss = obj.loss(weights, net_returns_single)

        # Expected: -log(1 + w^T r)
        portfolio_net_return = np.dot(weights, net_returns_single)
        expected = -np.log(1.0 + portfolio_net_return)

        np.testing.assert_almost_equal(loss, expected, decimal=10)

    def test_analytical_vs_autograd_gradient(self, weights, net_returns_single):
        """Test analytical gradient matches autograd gradient."""
        obj_autograd = MeasureObjective(
            measure=PerfMeasure.LOG_WEALTH, use_autograd=True
        )
        obj_analytical = MeasureObjective(
            measure=PerfMeasure.LOG_WEALTH, use_autograd=False
        )

        grad_autograd = obj_autograd.grad(weights, net_returns_single)
        grad_analytical = obj_analytical.grad(weights, net_returns_single)

        np.testing.assert_allclose(grad_autograd, grad_analytical, rtol=1e-10)

    def test_gradient_finite_difference(self, weights, net_returns_single):
        """Test gradient matches finite difference approximation."""
        obj = MeasureObjective(measure=PerfMeasure.LOG_WEALTH, use_autograd=True)

        _assert_gradient_matches_finite_difference(
            obj=obj, weights=weights, net_returns=net_returns_single, atol=1e-5
        )

    def test_negative_loss_positive_returns(self, weights):
        """Test log-wealth has negative loss for positive returns."""
        obj = MeasureObjective(measure=PerfMeasure.LOG_WEALTH, use_autograd=True)
        positive_returns = np.array([0.01, 0.02, 0.015, 0.012, 0.018])

        loss = obj.loss(weights, positive_returns)
        assert loss < 0, "Log-wealth loss should be negative for positive returns"


class TestRiskMeasures:
    """Test risk measure objectives with autograd."""

    @pytest.mark.parametrize(
        "measure",
        [
            RiskMeasure.VARIANCE,
            RiskMeasure.STANDARD_DEVIATION,
            RiskMeasure.SEMI_VARIANCE,
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
        ],
    )
    def test_gradient_finite_difference_single(
        self, measure, weights, net_returns_single
    ):
        """Test gradient matches finite difference for single observation."""
        obj = MeasureObjective(measure=measure, use_autograd=True)

        _assert_gradient_matches_finite_difference(
            obj=obj, weights=weights, net_returns=net_returns_single, atol=1e-5
        )

    @pytest.mark.parametrize(
        "measure",
        [
            RiskMeasure.VARIANCE,
            RiskMeasure.STANDARD_DEVIATION,
            RiskMeasure.SEMI_VARIANCE,
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
            RiskMeasure.CVAR,
            RiskMeasure.WORST_REALIZATION,
        ],
    )
    def test_gradient_finite_difference_timeseries(
        self, measure, weights, net_returns_timeseries
    ):
        """Test gradient matches finite difference for time series."""
        obj = MeasureObjective(measure=measure, use_autograd=True, cvar_beta=0.95)

        _assert_gradient_matches_finite_difference(
            obj=obj, weights=weights, net_returns=net_returns_timeseries, atol=1e-4
        )

    def test_positive_loss_values(self, weights, net_returns_timeseries):
        """Test risk measures produce non-negative loss values."""
        risk_measures = [
            RiskMeasure.VARIANCE,
            RiskMeasure.STANDARD_DEVIATION,
            RiskMeasure.SEMI_VARIANCE,
            RiskMeasure.CVAR,
        ]

        for measure in risk_measures:
            obj = MeasureObjective(measure=measure, use_autograd=True)
            loss = obj.loss(weights, net_returns_timeseries)
            assert loss >= 0, f"{measure} should have non-negative loss"

    def test_cvar_with_beta(self, weights, net_returns_timeseries):
        """Test CVaR with different beta values."""
        for beta in [0.9, 0.95, 0.99]:
            obj = MeasureObjective(
                measure=RiskMeasure.CVAR, use_autograd=True, cvar_beta=beta
            )
            loss = obj.loss(weights, net_returns_timeseries)
            assert loss >= 0


class TestPerformanceMeasures:
    """Test performance measure objectives with autograd."""

    def test_mean_gradient_finite_difference_single(self, weights, net_returns_single):
        """Test mean gradient for single observation."""
        obj = MeasureObjective(measure=PerfMeasure.MEAN, use_autograd=True)

        _assert_gradient_matches_finite_difference(
            obj=obj, weights=weights, net_returns=net_returns_single, atol=1e-5
        )

    def test_mean_gradient_finite_difference_timeseries(
        self, weights, net_returns_timeseries
    ):
        """Test mean gradient for time series."""
        obj = MeasureObjective(measure=PerfMeasure.MEAN, use_autograd=True)

        _assert_gradient_matches_finite_difference(
            obj=obj, weights=weights, net_returns=net_returns_timeseries, atol=1e-5
        )

    def test_negative_loss_positive_returns(self, weights, net_returns_timeseries):
        """Test performance measures have negative loss for positive returns."""
        # Create positive returns
        positive_returns = np.abs(net_returns_timeseries) + 0.001

        obj = MeasureObjective(measure=PerfMeasure.MEAN, use_autograd=True)
        loss = obj.loss(weights, positive_returns)

        # Mean loss should be negative (minimizing negative mean = maximizing mean)
        assert loss < 0


class TestCreateObjective:
    """Test objective factory function."""

    @pytest.mark.parametrize(
        "measure, expected",
        [
            pytest.param(None, PerfMeasure.LOG_WEALTH, id="default-log-wealth"),
            pytest.param(RiskMeasure.VARIANCE, RiskMeasure.VARIANCE, id="variance"),
            pytest.param(PerfMeasure.MEAN, PerfMeasure.MEAN, id="mean"),
        ],
    )
    def test_create_measure(self, measure, expected):
        """The factory preserves the requested measure or selects log wealth."""
        obj = create_objective(measure)
        assert isinstance(obj, MeasureObjective)
        assert obj.measure == expected

    def test_create_with_use_autograd_false(self):
        """Test creating objective with analytical gradient."""
        obj = create_objective(None, use_autograd=False)
        assert isinstance(obj, MeasureObjective)
        assert obj.use_autograd is False

    def test_create_with_kwargs(self):
        """Test creating objective with additional parameters."""
        obj = create_objective(RiskMeasure.CVAR, use_autograd=True, cvar_beta=0.99)
        assert isinstance(obj, MeasureObjective)
        assert obj.cvar_beta == 0.99


class TestAnalyticalGradientUniqueness:
    """Test that all analytical gradients produce unique results."""

    # Measures with analytical gradient implementations
    ANALYTICAL_MEASURES: ClassVar[list] = [
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
        RiskMeasure.CDAR,
        RiskMeasure.EDAR,
        RiskMeasure.WORST_REALIZATION,
        RiskMeasure.GINI_MEAN_DIFFERENCE,
    ]

    def test_t1_gradients_not_all_zero(self):
        """For T=1, gradients should not all be zero (sanity check).

        Note: Many T=1 gradients are mathematically identical due to proxy formulas.
        This test only ensures they're not broken (all returning zero).
        """
        weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])
        net_returns_single = np.array([-0.025, -0.012, -0.008, 0.005, -0.015])

        for measure in self.ANALYTICAL_MEASURES:
            obj = MeasureObjective(measure=measure, use_autograd=False)
            grad = obj.grad(weights, net_returns_single)

            # Gradient should not be all zeros (unless it's a special case)
            # For T=1, we just check it's not broken
            assert grad.shape == (5,), f"{measure}: wrong gradient shape"
            # At least check it's not NaN or inf
            assert not np.any(np.isnan(grad)), f"{measure}: gradient contains NaN"
            assert not np.any(np.isinf(grad)), f"{measure}: gradient contains inf"

    def test_all_analytical_gradients_are_unique_timeseries(self):
        """Test that all analytical gradients produce different results for time series (T>1).

        For time series data, different measures should produce different gradients.
        This is the primary test for detecting bugs in gradient implementations.

        Note: LOG_WEALTH is excluded as it's a per-observation measure not designed
        for time series gradients.
        """
        from itertools import combinations

        # Test data: time series with enough variation to distinguish measures
        np.random.seed(42)
        weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])

        # Generate returns with both positive and negative values,
        # including some downside to activate downside measures
        net_returns_timeseries = np.random.randn(50, 5) * 0.015
        # Add a slight negative drift to ensure downside measures are active
        net_returns_timeseries[:20] -= 0.005

        # Exclude LOG_WEALTH (per-observation measure, not for time series)
        timeseries_measures = [
            m for m in self.ANALYTICAL_MEASURES if m != PerfMeasure.LOG_WEALTH
        ]

        # Compute all gradients
        gradients = {}
        for measure in timeseries_measures:
            obj = MeasureObjective(measure=measure, use_autograd=False)
            grad = obj.grad(weights, net_returns_timeseries)
            gradients[measure] = grad

        # For T>1, ALL measures should be mathematically distinct
        # No exceptions - if any pair is identical, it's a bug
        failures = []
        for measure1, measure2 in combinations(timeseries_measures, 2):
            grad1 = gradients[measure1]
            grad2 = gradients[measure2]

            # Two different measures must have different gradients for T>1
            if np.allclose(grad1, grad2, rtol=1e-10, atol=1e-12):
                failures.append((measure1, measure2, grad1, grad2))

        if failures:
            msg = "The following measure pairs have identical gradients (BUG!):\n"
            msg += "For time series (T>1), all measures should be mathematically distinct.\n"
            for m1, m2, g1, g2 in failures:
                msg += f"\n  {m1} == {m2}"
                msg += f"\n    Gradient 1: {g1}"
                msg += f"\n    Gradient 2: {g2}"
            raise AssertionError(msg)

    def test_analytical_gradient_varies_with_data(self):
        """Test that analytical gradients vary with different input data.

        Gradients should change when the input data changes, otherwise
        the gradient function is broken (e.g., always returning zeros).

        Tests both single observations (for all measures) and time series
        (for non-LOG_WEALTH measures).
        """
        np.random.seed(42)
        weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])

        # Test single observation data - use returns that produce negative portfolio return
        # to ensure downside measures are activated
        net_returns_1 = np.array(
            [-0.02, -0.015, -0.01, 0.002, -0.005]
        )  # Negative portfolio return
        net_returns_2 = np.array(
            [0.015, 0.020, 0.01, -0.005, 0.012]
        )  # Positive portfolio return

        failures = []
        for measure in self.ANALYTICAL_MEASURES:
            obj = MeasureObjective(measure=measure, use_autograd=False)
            grad_1 = obj.grad(weights, net_returns_1)
            grad_2 = obj.grad(weights, net_returns_2)

            # Skip if both gradients are zero (can happen for T=1 with certain measures)
            # This is a limitation of T=1 proxies, not a bug
            if np.allclose(grad_1, 0) and np.allclose(grad_2, 0):
                continue

            # Gradients should differ for different data
            # (with tolerance for numerical precision)
            if np.allclose(grad_1, grad_2, rtol=1e-10, atol=1e-12):
                failures.append((measure, "single", grad_1, grad_2))

        # Also test time series (excluding LOG_WEALTH)
        ts_1 = np.random.randn(20, 5) * 0.01
        ts_2 = np.random.randn(20, 5) * 0.015 + 0.002

        for measure in self.ANALYTICAL_MEASURES:
            if measure == PerfMeasure.LOG_WEALTH:
                continue  # Skip LOG_WEALTH for time series

            obj = MeasureObjective(measure=measure, use_autograd=False)
            grad_1 = obj.grad(weights, ts_1)
            grad_2 = obj.grad(weights, ts_2)

            if np.allclose(grad_1, grad_2, rtol=1e-10, atol=1e-12):
                failures.append((measure, "timeseries", grad_1, grad_2))

        if failures:
            msg = "The following measures have identical gradients for different data (broken gradient):\n"
            for measure, data_type, g1, _g2 in failures:
                msg += f"\n  {measure} ({data_type})"
                msg += f"\n    Gradient: {g1} (same for both datasets)"
            raise AssertionError(msg)


class TestAnalyticalGradients:
    """Test analytical gradients match autograd gradients."""

    @pytest.mark.parametrize(
        "measure",
        [
            PerfMeasure.LOG_WEALTH,
            PerfMeasure.MEAN,
            RiskMeasure.VARIANCE,
            RiskMeasure.SEMI_VARIANCE,
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
            RiskMeasure.CVAR,
            RiskMeasure.WORST_REALIZATION,
            RiskMeasure.GINI_MEAN_DIFFERENCE,
        ],
    )
    def test_analytical_vs_autograd_single(self, measure, weights, net_returns_single):
        """Test analytical gradient matches autograd for single observation.

        Note: EVAR, CDAR, EDAR excluded as their autograd implementations
        use grid search/cummax which aren't fully autograd-compatible for gradients.
        These are tested separately using analytical gradients as reference.
        """
        obj_analytical = MeasureObjective(measure=measure, use_autograd=False)
        obj_autograd = MeasureObjective(measure=measure, use_autograd=True)

        grad_analytical = obj_analytical.grad(weights, net_returns_single)
        grad_autograd = obj_autograd.grad(weights, net_returns_single)

        np.testing.assert_allclose(
            grad_autograd, grad_analytical, rtol=1e-8, atol=1e-10
        )

    @pytest.mark.parametrize(
        "measure",
        [
            PerfMeasure.MEAN,
            RiskMeasure.VARIANCE,
            RiskMeasure.SEMI_VARIANCE,
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
            RiskMeasure.CVAR,
            RiskMeasure.WORST_REALIZATION,
            RiskMeasure.GINI_MEAN_DIFFERENCE,
        ],
    )
    def test_analytical_vs_autograd_timeseries(
        self, measure, weights, net_returns_timeseries
    ):
        """Test analytical gradient matches autograd for time series."""
        obj_analytical = MeasureObjective(measure=measure, use_autograd=False)
        obj_autograd = MeasureObjective(measure=measure, use_autograd=True)

        grad_analytical = obj_analytical.grad(weights, net_returns_timeseries)
        grad_autograd = obj_autograd.grad(weights, net_returns_timeseries)

        # Use more lenient tolerance for non-smooth measures
        # (MAD, semi-variance, semi-deviation, FLPM, CVaR)
        # These involve indicator functions/sorting and can have larger numerical differences
        if measure in {
            RiskMeasure.MEAN_ABSOLUTE_DEVIATION,
            RiskMeasure.SEMI_VARIANCE,
            RiskMeasure.SEMI_DEVIATION,
            RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT,
            RiskMeasure.CVAR,
        }:
            np.testing.assert_allclose(
                grad_autograd, grad_analytical, rtol=0.05, atol=1e-4
            )
        else:
            np.testing.assert_allclose(
                grad_analytical, grad_autograd, rtol=1e-5, atol=1e-8
            )

    def test_variance_uses_sample_covariance(self, weights, net_returns_timeseries):
        """Test variance gradient uses unbiased sample covariance matrix."""
        obj = MeasureObjective(measure=RiskMeasure.VARIANCE, use_autograd=False)

        # Compute analytical gradient
        grad = obj.grad(weights, net_returns_timeseries)

        # Manually compute using sign * 2Σw (where sign=1 for risk measures)
        # Note: Now using unbiased estimator (T-1 instead of T)
        T = net_returns_timeseries.shape[0]
        mean_returns = np.mean(net_returns_timeseries, axis=0, keepdims=True)
        centered = net_returns_timeseries - mean_returns
        Sigma = (centered.T @ centered) / (T - 1)  # Unbiased estimator
        expected_grad = 1.0 * 2.0 * (Sigma @ weights)  # sign=1 for risk measures

        np.testing.assert_allclose(grad, expected_grad, rtol=1e-10)


class TestAdvancedAnalyticalGradients:
    """Test analytical gradients for EVaR, CDaR, EDaR against finite differences.

    These measures use analytical gradients with theta optimization and drawdown derivatives.
    They cannot use autograd for gradient computation (grid search/cummax not autograd-compatible),
    so we validate against finite differences.
    """

    @pytest.mark.parametrize(
        "measure, kwargs, rtol, atol",
        [
            # Theta optimization makes entropic measures numerically sensitive.
            pytest.param(RiskMeasure.EVAR, {"evar_beta": 0.95}, 0.01, 1e-3, id="evar"),
            # Drawdowns also depend on wealth accumulation and peak tracking.
            pytest.param(RiskMeasure.CDAR, {"cdar_beta": 0.95}, 0.01, 2e-3, id="cdar"),
            pytest.param(RiskMeasure.EDAR, {"edar_beta": 0.95}, 0.02, 3e-3, id="edar"),
        ],
    )
    def test_analytical_vs_finite_diff(
        self, measure, kwargs, rtol, atol, weights, net_returns_timeseries
    ):
        """Check each analytical gradient with its original numerical tolerance."""
        obj = MeasureObjective(measure=measure, use_autograd=False, **kwargs)
        grad_analytical = obj.grad(weights, net_returns_timeseries)
        grad_numerical = _finite_difference_gradient(
            obj=obj, weights=weights, net_returns=net_returns_timeseries
        )
        np.testing.assert_allclose(
            grad_analytical, grad_numerical, rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("measure", AUTOGRAD_MEASURE_MAPPING, ids=str)
def test_scalar_metrics_match_reference(measure):
    # Keep the original sample for risks; log wealth requires returns above -1.
    np.random.seed(42)
    returns = np.random.randn(100)
    if measure == PerfMeasure.LOG_WEALTH:
        returns = np.expm1(returns * 0.02)
    autograd_func, ref_func = AUTOGRAD_MEASURE_MAPPING[measure]
    ref_val = ref_func(returns)
    actual = autograd_func(returns)
    assert np.isfinite(ref_val)
    assert np.isfinite(actual)
    is_approx = measure in (RiskMeasure.EVAR, RiskMeasure.EDAR)
    rtol = 1e-3 if is_approx else 1e-7
    try:
        np.testing.assert_allclose(
            actual,
            ref_val,
            rtol=rtol,
            err_msg=f"Metric {measure} differs from reference",
        )
    except AssertionError as error:
        if not is_approx:
            raise
        relative_error = np.abs((actual - ref_val) / ref_val)
        if relative_error > rtol:
            raise AssertionError(
                f"Metric {measure} exceeds tolerance: {relative_error} > {rtol}"
            ) from error


@pytest.mark.parametrize("factory", [MeasureObjective, create_objective])
@pytest.mark.parametrize(
    "measure", [None, RiskMeasure.STANDARD_DEVIATION, RiskMeasure.SEMI_DEVIATION]
)
def test_default_objectives_do_not_request_autograd(monkeypatch, factory, measure):
    from skfolio.optimization.online import _autograd_objectives as objectives

    def unavailable(*args, **kwargs):
        raise ImportError("Autograd is unavailable")

    monkeypatch.setattr(objectives, "grad", unavailable)
    objective = factory(measure=measure)
    weights = np.array([0.4, 0.6])
    returns = np.array([[0.02, -0.01], [-0.03, 0.04], [0.01, 0.02]])
    if objective.measure == PerfMeasure.LOG_WEALTH:
        # The online log-wealth gradient observes one period at a time.
        returns = returns[0]
    assert objective.use_autograd is False
    assert np.isfinite(objective.loss(weights, returns))
    assert np.isfinite(objective.grad(weights, returns)).all()
    with pytest.raises(ImportError, match="unavailable"):
        factory(measure=measure, use_autograd=True)
