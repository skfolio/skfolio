"""Exercise cash, eligibility changes and factor metadata on historical prices.

Eligibility exclusions below are deliberately synthetic test scenarios, not
historical claims about the companies. The example is a software validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from historical_walk_forward import ASSETS, make_panel
from sklearn.ensemble import RandomForestRegressor

from skfolio.datasets import load_sp500_dataset
from skfolio.descriptor import Passthrough
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.optimization import InverseVolatility, SupervisedPortfolioPolicy
from skfolio.prior import TimeSeriesFactorModel


def run(output_dir: Path) -> None:
    """Run a funded cash ledger through exclusions, an all-cash phase and reentry."""
    prices = load_sp500_dataset()[ASSETS]
    returns = prices.pct_change(fill_method=None).dropna()
    features = {"momentum": prices.pct_change(20, fill_method=None)}
    forward = returns.loc["2020"].iloc[:100]
    decisions = returns.index[returns.index.get_indexer(forward.index) - 1]
    train = returns.loc[returns.index < decisions[0]].tail(756)
    model = SupervisedPortfolioPolicy(
        characteristics_exposures=[
            (
                "momentum",
                FixedWeightedFactor(
                    descriptors=[("momentum", Passthrough(field="momentum"))],
                    outlier_transformer="passthrough",
                    scoring_transformer="passthrough",
                ),
            )
        ],
        optimizer=InverseVolatility(),
        regressor=RandomForestRegressor(
            n_estimators=30, min_samples_leaf=10, random_state=7
        ),
        window_size=21,
        benchmark_mcap_power=0,
        long_only=True,
        allow_universe_changes=True,
        max_weights=0.3,
        max_turnover=1.2,
        max_variance=0.6**2 / 252,
    ).fit(train, characteristics=make_panel(train.index, returns, features))
    eligibility = pd.DataFrame(True, index=decisions, columns=ASSETS)
    eligibility.iloc[20:40, 0] = False
    eligibility.iloc[60:65] = False
    # A point-in-time market proxy, supplied only on the causal prior window.
    factors = returns.mean(axis=1).to_frame("market_proxy")
    result = model.evaluate_execution(
        forward,
        make_panel(decisions, returns, features),
        initial_holdings=np.zeros(len(ASSETS)),
        initial_cash=1,
        cash_weight=0.05,
        cash_returns=0.00005,
        transaction_costs=0.001,
        rebalance_every=21,
        prior_estimator=TimeSeriesFactorModel(),
        prior_fit_params={"factors": factors},
        covariance_window=60,
        eligibility=eligibility,
    )
    np.testing.assert_allclose(
        result.targets.sum(axis=1) + result.ledger.target_cash, 1, atol=1e-7
    )
    np.testing.assert_allclose(
        result.ending_weights.sum(axis=1) + result.ledger.ending_cash, 1, atol=1e-7
    )
    np.testing.assert_allclose(
        result.ledger.cost, result.trades.abs().sum(axis=1) * 0.001
    )
    np.testing.assert_allclose(result.ledger.ending_cash.iloc[60:65], 1)
    assert (result.ledger.turnover <= 1.2 + 1e-6).all()
    assert (result.targets.loc[result.ledger.rebalanced].to_numpy() <= 0.3 + 1e-6).all()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("ledger", "targets", "trades", "ending_weights"):
        getattr(result, name).to_csv(output_dir / f"{name}.csv")
    print(
        f"Validated {len(result.ledger)} periods, {int(result.ledger.rebalanced.sum())} rebalances, and five all-cash periods."
    )
    print(f"Final wealth per initial unit: {result.ledger.wealth.iloc[-1]:.6f}")
    print(f"Total cost fractions: {result.ledger.cost.sum():.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
