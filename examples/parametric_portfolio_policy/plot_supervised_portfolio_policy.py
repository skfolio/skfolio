"""
===========================
Supervised Portfolio Policy
===========================

Fit a nonlinear regressor to portfolio-weight targets, then constrain predictions
on later characteristics. This synthetic example illustrates the API, not
investment performance. The historical walk-forward scripts in this directory
use bundled equity prices for chronological evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from skfolio.containers import AssetPanel
from skfolio.descriptor import Passthrough
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.optimization import InverseVolatility, SupervisedPortfolioPolicy

rng = np.random.default_rng(7)
assets = np.array(["a", "b", "c", "d"])
dates = pd.date_range("2020-01-01", periods=100)
signal = rng.normal(size=(100, 4))
returns = rng.normal(scale=0.01, size=(100, 4))
returns[1:] += 0.002 * signal[:-1]


def panel(start: int, end: int) -> AssetPanel:
    """Create one chronological slice of the synthetic data."""
    return AssetPanel(
        fields={"signal": signal[start:end], "returns": returns[start:end]},
        asset_names=assets,
        observations=dates[start:end],
    )


train, test = panel(0, 80), panel(80, 100)
policy = SupervisedPortfolioPolicy(
    characteristics_exposures=[
        (
            "signal",
            FixedWeightedFactor(
                descriptors=[("signal", Passthrough(field="signal"))],
                outlier_transformer="passthrough",
                scoring_transformer="passthrough",
            ),
        )
    ],
    optimizer=InverseVolatility(),
    regressor=RandomForestRegressor(n_estimators=20, random_state=7),
    benchmark_mcap_power=0,
    window_size=10,
    long_only=True,
    max_weights=0.4,
    groups=[["equity", "equity", "bond", "bond"]],
    linear_constraints=["equity <= 0.55"],
)
policy.fit(train.to_dataframe(fields="returns"), characteristics=train)
weights = policy.predict_weights(test)
np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-6)
assert (weights.to_numpy() >= -1e-6).all()
assert (weights.to_numpy() <= 0.4 + 1e-6).all()
assert (weights[["a", "b"]].sum(axis=1) <= 0.55 + 1e-6).all()
print(
    f"Fitted {len(policy.target_weights_)} label windows; checked {len(weights)} predictions."
)
print(weights.head())
