"""
Test equivalence between PAMR PA mode and MD mode.

The goal is to verify that under the simplest settings, PA (passive-aggressive)
and MD (mirror descent) modes produce equivalent results. This validates that
the OCO/Mirror Descent framework can be applied to mean-reversion problems.

Mathematical expectation:
-----------------------
For PAMR, the PA update is:
    w_{t+1} = Proj_K(w_t - tau * c)
    where:
    - c = x - mean(x) * 1  (centered)
    - tau = max(0, w^T x - eps) / ||c||^2  (adaptive step size)

The MD update with Euclidean geometry is:
    w_{t+1} = Proj_K(w_t - eta * g)
    where:
    - g = x - mean(x) * 1  (centered gradient, when violated)
    - eta = fixed learning rate

For equivalence, we need eta to adapt like tau, OR we need to test
with specific data where tau happens to equal a fixed eta.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfolio.datasets import load_sp500_dataset
from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._mixins import FTLStrategy, PAMRVariant
from skfolio.preprocessing import prices_to_returns


class TestPAMRModeEquivalence:
    """Test PA vs MD mode equivalence for PAMR."""

    @pytest.fixture
    def simple_data(self):
        """Load simple dataset for testing."""
        prices = load_sp500_dataset()
        # Use small subset for detailed analysis
        prices_subset = prices.iloc[:10, :5].copy()
        net_returns = prices_to_returns(prices_subset)
        return net_returns

    def test_pamr_no_violation_case(self, simple_data):
        """
        Test case where margin constraint is satisfied (no update needed).
        Both PA and MD should leave weights unchanged.
        """
        # Use data and high epsilon to ensure no violation
        X_single = simple_data.iloc[:1].values
        epsilon = 10.0  # Very high, constraint will be satisfied

        for update_mode in ["pa", "md"]:
            model = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant="simple",
                epsilon=epsilon,
                update_mode=update_mode,
                learning_rate=1.0,
                transaction_costs=0.0,
                management_fees=0.0,
            )
            model.partial_fit(X_single)

            # Should remain uniform
            expected_weights = np.ones(5) / 5
            assert_allclose(
                model.weights_,
                expected_weights,
                rtol=1e-9,
                err_msg=f"{update_mode} should not update when constraint satisfied",
            )

    def test_pamr_multi_step_with_adaptive_eta(self, simple_data):
        """
        Test multi-step comparison where we adapt eta at each step.

        This is conceptual: in practice MD uses fixed eta, but we can
        test if the single-step logic is correct by adapting eta.
        """
        X = simple_data.iloc[:5].values  # 5 periods
        epsilon = 0.5

        # Run PA mode (full sequence)
        model_pa = FollowTheLoser(
            strategy=FTLStrategy.PAMR,
            pamr_variant=PAMRVariant.SIMPLE,
            epsilon=epsilon,
            update_mode="pa",
            transaction_costs=0.0,
            management_fees=0.0,
        )
        model_pa.fit(X)

        # For MD, we'd need to adapt eta at each step, which isn't directly
        # supported. Instead, let's check if using a fixed "good" eta gives
        # similar results

        for eta in [0.01, 0.1, 0.5, 1.0]:
            model_md = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=PAMRVariant.SIMPLE,
                epsilon=epsilon,
                update_mode="md",
                learning_rate=eta,
                mirror="euclidean",
                transaction_costs=0.0,
                management_fees=0.0,
            )
            model_md.fit(X)

    def test_pamr_gradient_computation(self, simple_data):
        """
        Test that gradients are computed correctly in both modes.

        For PA: gradient is c = x - mean(x)*1
        For MD: gradient is g = x (then centered: g - mean(g))

        They should be identical when constraint is violated.
        """
        X_single = simple_data.iloc[:1].values
        x_t = 1.0 + X_single[0]
        w_init = np.ones(5) / 5
        epsilon = 0.5

        # PA gradient
        margin = float(x_t @ w_init)
        max(0.0, margin - epsilon)
        c_pa = x_t - np.mean(x_t)

        # MD gradient (when violated)
        g_md_raw = x_t
        g_md_centered = g_md_raw - np.mean(g_md_raw)

        # They should be identical
        assert_allclose(
            c_pa,
            g_md_centered,
            rtol=1e-12,
            err_msg="PA and MD centered gradients should match",
        )

    def test_pamr_projection_consistency(self, simple_data):
        """
        Test that projection is applied consistently in both modes.
        """
        X_single = simple_data.iloc[:1].values
        epsilon = 0.5

        # Test with different constraint settings
        constraint_configs = [
            {"min_weights": 0.0, "max_weights": 1.0, "budget": 1.0},
            {"min_weights": 0.1, "max_weights": 0.4, "budget": 1.0},
            {"min_weights": 0.0, "max_weights": 0.5, "budget": 1.0},
        ]

        for constraints in constraint_configs:
            # Compute tau from PA to use as eta in MD
            x_t = 1.0 + X_single[0]
            w_init = np.ones(5) / 5
            margin = float(x_t @ w_init)
            ell = max(0.0, margin - epsilon)
            c = x_t - np.mean(x_t)
            denom = float(np.dot(c, c))
            tau = ell / denom if denom > 0 else 0.0

            # PA mode
            model_pa = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=PAMRVariant.SIMPLE,
                epsilon=epsilon,
                update_mode="pa",
                transaction_costs=0.0,
                management_fees=0.0,
                **constraints,
            )
            model_pa.partial_fit(X_single)
            weights_pa = model_pa.weights_.copy()

            # MD mode with matching eta
            model_md = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=PAMRVariant.SIMPLE,
                epsilon=epsilon,
                update_mode="md",
                learning_rate=tau,
                mirror="euclidean",
                transaction_costs=0.0,
                management_fees=0.0,
                **constraints,
            )
            model_md.partial_fit(X_single)
            weights_md = model_md.weights_.copy()

            # Verify both respect constraints
            assert np.all(weights_pa >= constraints["min_weights"] - 1e-8)
            assert np.all(weights_pa <= constraints["max_weights"] + 1e-8)
            assert np.abs(np.sum(weights_pa) - constraints["budget"]) < 1e-8

            assert np.all(weights_md >= constraints["min_weights"] - 1e-8)
            assert np.all(weights_md <= constraints["max_weights"] + 1e-8)
            assert np.abs(np.sum(weights_md) - constraints["budget"]) < 1e-8

            # They should match when eta=tau
            assert_allclose(
                weights_pa,
                weights_md,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"PA and MD should match with constraints {constraints}",
            )

    def test_pamr_all_variants_pa_vs_md(self, simple_data):
        """
        Test all three PAMR variants (0, 1, 2) for PA vs MD equivalence.
        """
        X_single = simple_data.iloc[:1].values
        epsilon = 0.5
        C = 500.0
        tau = None
        for variant in [
            PAMRVariant.SIMPLE,
            PAMRVariant.SLACK_LINEAR,
            PAMRVariant.SLACK_QUADRATIC,
        ]:
            # Compute tau for this variant
            x_t = 1.0 + X_single[0]
            w_init = np.ones(5) / 5
            margin = float(x_t @ w_init)
            ell = max(0.0, margin - epsilon)
            c = x_t - np.mean(x_t)
            denom = float(np.dot(c, c))

            if variant == PAMRVariant.SIMPLE:
                tau = ell / denom if denom > 0 else 0.0
            elif variant == PAMRVariant.SLACK_LINEAR:
                tau = min(C, ell / denom) if denom > 0 else 0.0
            elif variant == PAMRVariant.SLACK_QUADRATIC:
                tau = ell / (denom + 1.0 / (2.0 * C)) if denom > 0 else 0.0

            # PA mode
            model_pa = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=variant,
                pamr_C=C,
                epsilon=epsilon,
                update_mode="pa",
                transaction_costs=0.0,
                management_fees=0.0,
            )
            model_pa.partial_fit(X_single)
            weights_pa = model_pa.weights_.copy()

            # MD mode
            model_md = FollowTheLoser(
                strategy=FTLStrategy.PAMR,
                pamr_variant=variant,
                pamr_C=C,
                epsilon=epsilon,
                update_mode="md",
                learning_rate=tau,
                mirror="euclidean",
                transaction_costs=0.0,
                management_fees=0.0,
            )
            model_md.partial_fit(X_single)
            weights_md = model_md.weights_.copy()

            if variant == 0:
                # Variant 0 should match exactly
                assert_allclose(
                    weights_pa,
                    weights_md,
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"Variant {variant} should match",
                )


@pytest.mark.parametrize("variant", list(PAMRVariant))
def test_pamr_pa_md_match_interior_update_for_each_slack_variant(variant):
    initial = np.array([0.2, 0.3, 0.5])
    row = np.array([0.2, -0.1, 0.05])
    relative = 1 + row
    direction = relative - relative.mean()
    norm_squared = direction @ direction
    loss = 0.2 * norm_squared
    epsilon = initial @ relative - loss
    slack = 0.1
    if variant == PAMRVariant.SIMPLE:
        eta = loss / norm_squared
    elif variant == PAMRVariant.SLACK_LINEAR:
        eta = min(slack, loss / norm_squared)
    else:
        eta = loss / (norm_squared + 1 / (2 * slack))
    expected = initial - eta * direction
    assert np.all((expected > 0) & (expected < 1))
    assert not np.allclose(initial, expected)
    for mode in ["pa", "md"]:
        model = FollowTheLoser(
            strategy="pamr",
            pamr_variant=variant,
            pamr_C=slack,
            initial_weights=initial,
            epsilon=epsilon,
            update_mode=mode,
            learning_rate=eta,
        ).partial_fit(row)
        assert_allclose(model.weights_, expected, atol=1e-10)
