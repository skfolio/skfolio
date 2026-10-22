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

    @pytest.mark.parametrize(
        "mode,rate", [("pa", 1.0), ("md", 0.01), ("md", 0.1), ("md", 0.5), ("md", 1.0)]
    )
    def test_pamr_multi_step_reference_updates(self, mode, rate):
        """Check PA and optimistic MD against their distinct two-asset recurrences."""
        returns = np.array(
            [[0.02, -0.01], [-0.03, 0.04], [0.01, 0.02], [-0.02, 0.01], [0.03, -0.02]]
        )
        model = FollowTheLoser(
            strategy="pamr", epsilon=1.0, update_mode=mode, learning_rate=rate
        )
        expected = np.array([0.5, 0.5])
        last_gradient = np.zeros(2)
        previous_prediction = np.zeros(2)
        active_updates = 0
        for row in returns:
            relatives = 1.0 + row
            direction = relatives - relatives.mean()
            loss = max(0.0, expected @ relatives - 1.0)
            if mode == "pa":
                step = loss / (direction @ direction)
                candidate = expected - step * direction
            else:
                gradient = direction if loss > 0 else np.zeros(2)
                # The default MD engine predicts with the preceding gradient;
                # consequently multi-step MD need not equal plain PA.
                candidate = expected - rate * (
                    gradient + last_gradient - previous_prediction
                )
                previous_prediction = last_gradient.copy()
                last_gradient = gradient.copy()
            # Two-asset simplex projection is clipping plus the complement.
            first = np.clip(candidate[0], 0.0, 1.0)
            expected = np.array([first, 1.0 - first])
            active_updates += loss > 0
            model.partial_fit(row)
            assert_allclose(model.weights_, expected, atol=1e-10)
        assert active_updates >= 2

    def test_pamr_gradient_computation(self, simple_data):
        """The actual interior MD update follows the centered PAMR gradient."""
        row = simple_data.iloc[0].values
        relatives = 1.0 + row
        initial = np.full(len(row), 1 / len(row))
        eta = 0.1
        expected = initial - eta * (relatives - relatives.mean())
        assert initial @ relatives > 0.5
        assert np.all((expected > 0) & (expected < 1))
        model = FollowTheLoser(
            strategy="pamr",
            epsilon=0.5,
            update_mode="md",
            initial_weights=initial,
            learning_rate=eta,
        )
        model.partial_fit(row)
        assert_allclose(model.weights_, expected, rtol=1e-12, atol=1e-12)

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

            assert_allclose(
                weights_pa,
                weights_md,
                rtol=1e-6,
                atol=1e-8,
                err_msg=f"PA and MD should match for {variant} with its matching step size",
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
