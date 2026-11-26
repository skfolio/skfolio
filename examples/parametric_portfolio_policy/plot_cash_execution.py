"""Cash-funded supervised portfolio evaluation
===========================================

Start entirely in cash, reserve 5% cash when invested, and fund transaction
costs from the account. Use historical returns with a causal factor prior.
One asset is temporarily excluded, then the entire universe is excluded for
five periods before reentry. Eligibility changes trigger additional trades.

Eligibility exclusions below are deliberately synthetic test scenarios, not
historical claims about the companies. The example is a software validation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from skfolio.containers import AssetPanel
from skfolio.datasets import load_sp500_dataset
from skfolio.descriptor import Passthrough
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.optimization import InverseVolatility, SupervisedPortfolioPolicy
from skfolio.optimization.parametric import PortfolioExecution
from skfolio.prior import TimeSeriesFactorModel

ASSETS = ["AAPL", "CVX", "GE", "JNJ", "JPM", "KO", "MSFT", "WMT"]


def make_panel(
    index: pd.Index, returns: pd.DataFrame, features: dict[str, pd.DataFrame]
) -> AssetPanel:
    """Select only characteristics available on each decision date."""
    return AssetPanel(
        fields={
            "returns": returns.loc[index].to_numpy(),
            **{name: values.loc[index].to_numpy() for name, values in features.items()},
        },
        asset_names=np.asarray(ASSETS),
        observations=index,
    )


def run(output_dir: Path | None = None) -> PortfolioExecution:
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
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ("ledger", "targets", "trades", "ending_weights"):
            getattr(result, name).to_csv(output_dir / f"{name}.csv")
    print(
        f"Validated {len(result.ledger)} periods, {int(result.ledger.rebalanced.sum())} rebalances, and five all-cash periods."
    )
    print(f"Final wealth per initial unit: {result.ledger.wealth.iloc[-1]:.6f}")
    print(f"Total cost fractions: {result.ledger.cost.sum():.6f}")

    return result


# %%
# Inspect wealth, cash and trading costs
# ----------------------------------------------
# The shaded region is the synthetic all-cash interval. The cash ledger includes
# funded fees; it uses different accounting from the Portfolio-based sequential
# evaluator. No borrowing fees, margin or execution slippage are modeled.
if __name__ == "__main__":
    execution = run()
    ledger = execution.ledger
    print(ledger.iloc[58:67].round(6).to_string())
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    ledger.wealth.plot(ax=axes[0], title="Funded account wealth")
    axes[0].set_ylabel("Wealth per initial unit")
    ledger.ending_cash.plot(ax=axes[1], title="Cash fraction after each period")
    axes[1].set_ylabel("Cash weight")
    ledger.cost.cumsum().plot(ax=axes[2], title="Cumulative cost fractions")
    axes[2].set_ylabel("Sum of period cost fractions")
    for ax in axes:
        ax.axvspan(ledger.index[60], ledger.index[64], alpha=0.15, color="grey")
    fig.tight_layout()
    plt.show()
