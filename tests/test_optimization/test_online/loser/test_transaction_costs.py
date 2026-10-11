"""Test transaction costs and management fees in Follow-The-Loser strategies."""

import numpy as np
import pytest

from skfolio.optimization.online._loser import FollowTheLoser

"""Test transaction costs work in both PA and MD modes."""


@pytest.mark.parametrize("strategy", ["olmar", "pamr"])
@pytest.mark.parametrize("update_mode", ["pa", "md"])
def test_transaction_costs_reduce_turnover(X_small, strategy, update_mode):
    anchor = np.array([0.8, 0.2])
    model = FollowTheLoser(
        strategy=strategy, update_mode=update_mode, previous_weights=anchor
    )
    model.partial_fit([0.1, -0.1])
    candidate = np.array([0.2, 0.8])
    model._transaction_costs_arr = 0.1
    adjusted = model._apply_transaction_cost_regularization(
        candidate, model.weights_, anchor
    )
    assert np.abs(adjusted - anchor).sum() < np.abs(candidate - anchor).sum()


@pytest.mark.parametrize("strategy", ["olmar", "pamr"])
@pytest.mark.parametrize("update_mode", ["pa"])
def test_higher_costs_reduce_turnover_more(X_small, strategy, update_mode):
    anchor = np.array([0.8, 0.2])
    model = FollowTheLoser(
        strategy=strategy, update_mode=update_mode, previous_weights=anchor
    )
    model.partial_fit([0.1, -0.1])
    candidate = np.array([0.2, 0.8])
    model._transaction_costs_arr = 0.01
    low = model._apply_transaction_cost_regularization(
        candidate, model.weights_, anchor
    )
    model._transaction_costs_arr = 0.1
    high = model._apply_transaction_cost_regularization(
        candidate, model.weights_, anchor
    )
    assert np.abs(high - anchor).sum() <= np.abs(low - anchor).sum()


"""Test management fees application."""


@pytest.mark.parametrize("strategy", ["olmar", "pamr", "rmr"])
def test_management_fees_affect_weights(X_small, strategy):
    """Management fees should affect portfolio weights."""
    np.random.seed(42)

    model_no_fee = FollowTheLoser(
        strategy=strategy,
        management_fees=0.0,
        warm_start=False,
    )
    model_no_fee.fit(X_small)

    model_with_fee = FollowTheLoser(
        strategy=strategy,
        management_fees=0.001,  # 0.1% per period
        warm_start=False,
    )
    model_with_fee.fit(X_small)

    # Weights should differ when management fees are applied
    final_no_fee = model_no_fee.weights_
    final_with_fee = model_with_fee.weights_

    # Should not be identical (fees affect the optimization)
    assert not np.allclose(final_no_fee, final_with_fee, atol=1e-6), (
        f"Management fees should affect weights for {strategy}"
    )


@pytest.mark.parametrize("strategy", ["olmar", "rmr"])
def test_apply_fees_to_phi(X_small, strategy):
    """Test apply_fees_to_phi parameter affects predictor."""
    np.random.seed(42)
    fees = 0.001

    model_fees_to_phi = FollowTheLoser(
        strategy=strategy,
        management_fees=fees,
        apply_fees_to_phi=True,
        warm_start=False,
    )
    model_fees_to_phi.fit(X_small)

    model_no_fees_to_phi = FollowTheLoser(
        strategy=strategy,
        management_fees=fees,
        apply_fees_to_phi=False,
        warm_start=False,
    )
    model_no_fees_to_phi.fit(X_small)

    # Weights should differ when fees are applied to phi
    final_with = model_fees_to_phi.weights_
    final_without = model_no_fees_to_phi.weights_

    # Should not be identical
    assert not np.allclose(final_with, final_without, rtol=1e-3), (
        f"apply_fees_to_phi should affect weights for {strategy}"
    )


"""Test interaction of transaction costs and management fees."""


@pytest.mark.parametrize("strategy", ["olmar"])
@pytest.mark.parametrize("update_mode", ["pa"])
def test_combined_effects(X_small, strategy, update_mode):
    from skfolio.model_selection import online_predict

    model = FollowTheLoser(
        strategy=strategy,
        update_mode=update_mode,
        transaction_costs=0.005,
        management_fees=0.001,
        portfolio_params={"weight_drift": True},
    )
    path = online_predict(model, X_small, warmup_size=3, test_size=1)
    for p in path:
        raw = np.asarray(p.X)[0]
        expected = (
            p.weights @ raw
            - 0.005 * np.abs(p.weights - p.previous_weights).sum()
            - 0.001 * p.weights.sum()
        )
        np.testing.assert_allclose(p.returns, [expected])


@pytest.mark.parametrize("strategy", ["olmar", "pamr"])
@pytest.mark.parametrize("update_mode", ["pa", "md"])
def test_costs_fees_consistency(X_small, strategy, update_mode):
    """Test that costs and fees work consistently across modes."""
    np.random.seed(42)

    model = FollowTheLoser(
        strategy=strategy,
        update_mode=update_mode,
        transaction_costs=0.003,
        management_fees=0.0005,
        warm_start=False,
    )
    model.fit(X_small)

    # Should complete without errors and have valid weights
    assert hasattr(model, "weights_")
    assert hasattr(model, "all_weights_")
    assert np.isclose(np.sum(model.weights_), 1.0, atol=1e-6)
    assert np.all(model.weights_ >= -1e-6)  # Allow small numerical errors
    assert np.all(model.weights_ <= 1.0 + 1e-6)
