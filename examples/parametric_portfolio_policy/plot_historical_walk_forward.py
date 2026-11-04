"""Historical supervised portfolio evaluation
==========================================

Refit a random forest policy each year and compare it with a constrained
equal-weight policy on bundled stock prices. Both trade every 21 observations
and pay the same proportional transaction costs.

This fixed-universe dataset is for examples, not a point-in-time constituent
universe. Results illustrate software behavior and are not investment evidence.
Characteristics precede each holding period. Training uses the preceding three
years; the evaluation covers 2019 through 2021. Holdings carry across yearly
refits. The model and baseline share weight, turnover and variance limits.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from skfolio.containers import AssetPanel
from skfolio.datasets import load_sp500_dataset
from skfolio.descriptor import Passthrough
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.optimization import InverseVolatility, SupervisedPortfolioPolicy

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


def run(output_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    """Refit each year, hold for 21 observations, and export execution diagnostics."""
    prices = load_sp500_dataset()[ASSETS]
    returns = prices.pct_change(fill_method=None).dropna()
    features = {
        "momentum_20": prices.pct_change(20, fill_method=None),
        "momentum_60": prices.pct_change(60, fill_method=None),
        "volatility_20": returns.rolling(20, min_periods=20).std(),
    }
    exposures = [
        (
            name,
            FixedWeightedFactor(
                descriptors=[(name, Passthrough(field=name))],
                outlier_transformer="passthrough",
                scoring_transformer="passthrough",
            ),
        )
        for name in features
    ]
    initial = np.full(len(ASSETS), 1 / len(ASSETS))
    holdings = {"random_forest": initial.copy(), "equal_weight_policy": initial.copy()}
    records = []
    folds = []
    for year in (2019, 2020, 2021):
        # One characteristic row immediately precedes each realized return row.
        forward = returns.loc[str(year)]
        positions = returns.index.get_indexer(forward.index)
        decisions = returns.index[positions - 1]
        train = returns.loc[returns.index < decisions[0]].tail(756)
        train_panel = make_panel(train.index, returns, features)
        test_panel = make_panel(decisions, returns, features)
        folds.append(
            {
                "test_year": year,
                "train_start": str(train.index[0].date()),
                "train_end": str(train.index[-1].date()),
                "first_decision": str(decisions[0].date()),
                "test_rows": len(forward),
            }
        )
        for name, regressor in (
            (
                "random_forest",
                RandomForestRegressor(
                    n_estimators=50, min_samples_leaf=10, random_state=7
                ),
            ),
            ("equal_weight_policy", DummyRegressor(strategy="constant", constant=0)),
        ):
            model = SupervisedPortfolioPolicy(
                characteristics_exposures=exposures,
                optimizer=InverseVolatility(),
                regressor=regressor,
                window_size=21,
                benchmark_mcap_power=0,
                long_only=True,
                max_weights=0.25,
                max_turnover=0.3,
                max_variance=0.6**2 / 252,
            ).fit(train, characteristics=train_panel)
            evaluation = model.evaluate_sequential(
                forward,
                test_panel,
                initial_holdings=holdings[name],
                rebalance_every=21,
                covariance_window=60,
                transaction_costs=0.001,
            )
            for t, period in enumerate(evaluation.portfolios):
                rebalance = t % 21 == 0
                assert np.isfinite(period.returns).all()
                np.testing.assert_allclose(period.weights.sum(), 1, atol=1e-6)
                if rebalance:
                    assert period.weights.max() <= 0.25 + 1e-6
                    assert period.turnover <= 0.3 + 1e-6
                else:
                    assert period.turnover == 0 and period.total_cost == 0
                records.append(
                    {
                        "strategy": name,
                        "date": str(forward.index[t].date()),
                        "decision": str(decisions[t].date()),
                        "rebalance": rebalance,
                        "net_return": float(period.returns[0]),
                        "gross_return": float(
                            period.weights @ forward.iloc[t].to_numpy()
                        ),
                        "turnover": float(period.turnover),
                        "cost": float(period.total_cost),
                        **{
                            f"weight_{asset}": float(weight)
                            for asset, weight in zip(
                                ASSETS, period.weights, strict=True
                            )
                        },
                    }
                )
            holdings[name] = evaluation.portfolios[-1].ending_weights.copy()
    results = pd.DataFrame(records)
    summary = {}
    for name, group in results.groupby("strategy", sort=False):
        net = group.net_return.to_numpy()
        wealth = np.cumprod(1 + net)
        peak = np.maximum.accumulate(np.r_[1, wealth])[1:]
        summary[name] = {
            "observations": len(net),
            "rebalances": int(group.rebalance.sum()),
            "net_total_return": float(wealth[-1] - 1),
            "annualized_volatility": float(np.std(net, ddof=1) * np.sqrt(252)),
            "max_drawdown": float(np.min(wealth / peak - 1)),
            "sum_turnover": float(group.turnover.sum()),
            "sum_cost_fractions": float(group.cost.sum()),
        }
    report = {
        "assets": ASSETS,
        "settings": {
            "training_observations": 756,
            "label_window": 21,
            "rebalance_every": 21,
            "covariance_window": 60,
            "max_weight": 0.25,
            "max_turnover": 0.3,
            "max_daily_variance": 0.6**2 / 252,
            "transaction_cost_rate": 0.001,
            "random_state": 7,
        },
        "folds": folds,
        "summary": summary,
        "limitations": "Historical fixed universe; no cash ledger, slippage or corporate-action simulation. Cost reporting follows Portfolio conventions. No hyperparameter search or performance claim.",
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / "daily_execution.csv", index=False)
        (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return results, report


# %%
# Compare net wealth and cumulative turnover
# --------------------------------------------------
# Wealth starts at one and compounds the daily returns after transaction costs.
# The turnover chart shows how much each policy trades, including entry trades.
# These are illustrative results on a fixed historical universe, not a strategy
# selection exercise or a point-in-time investment study.
if __name__ == "__main__":
    results, report = run()
    print(pd.DataFrame(report["folds"]).to_string(index=False))
    print(pd.DataFrame(report["summary"]).T.round(4).to_string())
    daily = results.pivot(index="date", columns="strategy", values="net_return")
    daily.index = pd.to_datetime(daily.index)
    turnover = results.pivot(index="date", columns="strategy", values="turnover")
    turnover.index = pd.to_datetime(turnover.index)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    (1 + daily).cumprod().plot(ax=axes[0], title="Net wealth after trading costs")
    axes[0].set_ylabel("Wealth per initial unit")
    turnover.cumsum().plot(ax=axes[1], title="Cumulative turnover")
    axes[1].set_ylabel("Sum of absolute traded weights")
    fig.tight_layout()
    plt.show()
