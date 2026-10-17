"""Shared synthetic inputs and estimators for supervised policy tests."""

from typing import ClassVar

import numpy as np
import pandas as pd

from skfolio.containers import AssetPanel
from skfolio.descriptor import Passthrough
from skfolio.factor_exposure import FixedWeightedFactor
from skfolio.optimization import (
    BaseOptimization,
    SupervisedPortfolioPolicy,
)


class RecordingOptimizer(BaseOptimization):
    """Use encoded target weights to audit exactly which returns enter each label."""

    windows: ClassVar[list[pd.DataFrame]] = []

    def fit(self, X, y=None):
        type(self).windows.append(X.copy())
        self.weights_ = X.mean().to_numpy()
        return self


def data(n=11, offset=0):
    signal = np.tile([-1.0, 0.0, 1.0], (n, 1))
    signal[:, 0] += np.arange(n) * 0.03
    signal -= signal.mean(axis=1, keepdims=True)
    returns = np.full((n, 3), 1 / 3)
    for t in range(0, n - 2, 2):
        returns[t + 1 : t + 3] += 0.4 * signal[t] / 3
    dates = pd.date_range("2020-01-01", periods=n + offset)[offset:]
    panel = AssetPanel(
        fields={"signal": signal, "returns": returns},
        asset_names=np.array(["a", "b", "c"]),
        observations=dates,
    )
    return panel.to_dataframe(fields="returns"), panel


def model(**kwargs):
    return SupervisedPortfolioPolicy(
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
        optimizer=kwargs.pop("optimizer", RecordingOptimizer()),
        benchmark_mcap_power=0,
        window_size=kwargs.pop("window_size", 2),
        alpha=0,
        **kwargs,
    )


def expanded_data(n=11, offset=0):
    X, panel = data(n=n, offset=offset)
    X["new"] = np.linspace(0.01, 0.02, n)
    signal = np.column_stack([panel["signal"], np.full(n, 2.0)])
    return X, AssetPanel(
        fields={"signal": signal, "returns": X.to_numpy()},
        asset_names=X.columns.to_numpy(),
        observations=X.index,
    )
