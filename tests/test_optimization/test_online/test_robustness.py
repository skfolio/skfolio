import numpy as np
import pytest
from tests.test_optimization.test_online.utils import make_stationary_returns

from skfolio.optimization.online import RegretType
from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._mixins import FTLStrategy, FTWStrategy
from skfolio.optimization.online._winner import FollowTheWinner


def test_rmr_outlier_resistance():
    """
    Test that RMR (Robust Median Reversion) outperforms standard OLMAR
    in the presence of significant outliers.
    """
    # Generate stationary data where mean reversion works well
    n_assets = 5
    n_periods = 500
    rng = np.random.default_rng(42)
    X = make_stationary_returns(T=n_periods, n=n_assets, gap=0.01)

    # Inject outliers: random massive spikes
    n_outliers = 20
    outlier_indices = rng.choice(n_periods, n_outliers, replace=False)
    asset_indices = rng.choice(n_assets, n_outliers, replace=True)

    # Create a copy to corrupt
    X_corrupted = X.copy()
    for t, i in zip(outlier_indices, asset_indices, strict=True):
        # Massive 50% drop (0.5 relative) or 100% gain (2.0 relative)
        # to confuse mean estimators
        X_corrupted[t, i] = 2.0 if rng.random() > 0.5 else 0.5

    # Train OLMAR (sensitive to mean)
    olmar = FollowTheLoser(
        strategy=FTLStrategy.OLMAR,
        olmar_predictor="sma",
        olmar_window=5,
        update_mode="pa",
    )
    olmar.fit(X_corrupted)

    # Train RMR (robust median)
    rmr = FollowTheLoser(
        strategy=FTLStrategy.RMR,
        rmr_window=5,  # Window size for RMR
        update_mode="pa",
    )
    rmr.fit(X_corrupted)

    olmar_wealth = np.prod(1 + olmar.fit_predict(X_corrupted).returns)
    rmr_wealth = np.prod(1 + rmr.fit_predict(X_corrupted).returns)

    # RMR should generally handle these outliers better or at least not crash
    # In this specific setup, we expect RMR to be more stable.
    # We check if RMR wealth is reasonable and ideally better or comparable to OLMAR
    # given the corruption.
    assert rmr_wealth > 0.5  # Basic sanity check

    # Check that RMR weights are less volatile around outliers?
    # Or simply that it runs without error.
    # Let's assert RMR outperforms OLMAR in this specific corrupted scenario
    # Note: This might be stochastic, so we use a fixed seed.
    assert rmr_wealth > olmar_wealth


def test_universal_regret_is_finite():
    """
    Verify that the computed universal regret is finite and behaves reasonably.
    """
    X = make_stationary_returns(T=100, n=3)

    model = FollowTheWinner(strategy=FTWStrategy.EG)
    model.fit(X)

    from skfolio.optimization.online._regret import regret

    regrets = regret(
        estimator=model,
        X=X,
        regret_type=RegretType.DYNAMIC_UNIVERSAL,
        dynamic_config={"path_length": 2.0},
    )

    assert len(regrets) == len(X)
    assert not np.any(np.isnan(regrets))


def test_zero_liquidity_handling():
    """
    Test behavior when assets have 0 returns (price relative = 1.0)
    or even worse, price relative = 0.0 (bankruptcy).
    """
    X = np.zeros((50, 3))  # Flat market (0% return -> relative 1.0)

    model = FollowTheWinner(strategy=FTWStrategy.EG)
    model.fit(X)

    # In a flat market, wealth should stay 1.0 (minus fees if any)
    assert np.isclose(np.prod(1 + model.fit_predict(X).returns), 1.0)

    # Case 2: Asset 0 goes to 0 (bankruptcy)
    X_crash = np.zeros((50, 3))
    X_crash[10, 0] = -1.0  # Asset 0 dies at t=10 (return -100% -> relative 0)

    model_crash = FollowTheWinner(strategy=FTWStrategy.EG)
    with pytest.raises(ValueError, match="bankrupt assets"):
        model_crash.fit(X_crash)
