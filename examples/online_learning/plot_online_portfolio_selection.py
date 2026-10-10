"""
Online portfolio selection with three shared-kernel estimators
==============================================================

Compare next-period allocations after a historical learning window. All three
estimators use Portfolio for execution and accounting. The synthetic example
requires no external data and does not imply performance superiority.
"""

import numpy as np

from skfolio.model_selection import online_predict
from skfolio.optimization.online import (
    AdaGrad,
    ExponentiatedGradient,
    OnlineMovingAverageReversion,
)

X = np.random.default_rng(261).normal(0.0005, 0.01, (120, 4))
models = [
    ExponentiatedGradient(update_mode="ftrl"),
    AdaGrad(learning_rate=0.05),
    OnlineMovingAverageReversion(window=5),
]

for model in models:
    model.set_params(portfolio_params={"weight_drift": True})
    portfolio = online_predict(model, X, warmup_size=20, test_size=1)
    print(type(model).__name__, portfolio.returns.shape)
