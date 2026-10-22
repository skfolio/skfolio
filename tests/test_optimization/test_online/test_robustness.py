import numpy as np
import pytest
from tests.test_optimization.test_online.utils import make_stationary_returns

from skfolio.optimization.online import RegretType
from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._mixins import FTLStrategy, FTWStrategy
from skfolio.optimization.online._winner import FollowTheWinner


@pytest.mark.parametrize("strategy", [FTLStrategy.OLMAR, FTLStrategy.RMR])
def test_mean_reversion_execution_with_outliers(strategy):
    """Real losses and gains preserve finite allocations and gross accounting."""
    rng = np.random.default_rng(42)
    returns = make_stationary_returns(T=500, n=5, gap=0.01)
    times = rng.choice(len(returns), 20, replace=False)
    assets = rng.choice(returns.shape[1], 20, replace=True)
    for t, asset in zip(times, assets, strict=True):
        returns[t, asset] = 1.0 if rng.random() > 0.5 else -0.5
    assert np.any(returns == -0.5)
    assert np.any(returns == 1.0)
    model = FollowTheLoser(
        strategy=strategy, olmar_window=5, rmr_window=5, update_mode="pa"
    )
    portfolio = model.fit_predict(returns)
    weights = model.all_weights_
    assert np.isfinite(weights).all()
    assert np.all(weights >= -1e-10)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    expected_returns = np.sum(weights * returns, axis=1)
    np.testing.assert_allclose(portfolio.returns, expected_returns, atol=1e-12)
    wealth = np.cumprod(1 + portfolio.returns)
    assert np.isfinite(wealth).all()
    assert np.all(wealth > 0)


@pytest.mark.parametrize("outlier_price", [0.5, 2.0])
def test_rmr_median_resists_transient_price_outlier(outlier_price):
    """Four identical prices anchor the median despite one large price shock."""
    from skfolio.optimization.online._prediction import RMRPredictor

    predictor = RMRPredictor(window=5)
    predictor.reset(2)
    prices = np.ones((6, 2))
    prices[3, 0] = outlier_price
    relatives = np.vstack([np.ones(2), prices[1:] / prices[:-1]])
    for relative in relatives:
        forecast = predictor.update_and_predict(relative)
    np.testing.assert_allclose(forecast, np.ones(2), atol=1e-9)
    # The corresponding arithmetic-mean forecast is moved by this same shock.
    mean_forecast = prices[-5:].mean(axis=0) / prices[-1]
    assert abs(mean_forecast[0] - 1) >= 0.09


def test_universal_regret_is_finite():
    """
    Verify that the computed universal regret is finite and behaves reasonably.
    """
    X = make_stationary_returns(T=100, n=3)

    model = FollowTheWinner(strategy=FTWStrategy.EG)

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

    # In a flat market, wealth should stay 1.0 (minus fees if any)
    assert np.isclose(np.prod(1 + model.fit_predict(X).returns), 1.0)

    # Case 2: Asset 0 goes to 0 (bankruptcy)
    X_crash = np.zeros((50, 3))
    X_crash[10, 0] = -1.0  # Asset 0 dies at t=10 (return -100% -> relative 0)

    model_crash = FollowTheWinner(strategy=FTWStrategy.EG)
    with pytest.raises(ValueError, match="bankrupt assets"):
        model_crash.fit(X_crash)
