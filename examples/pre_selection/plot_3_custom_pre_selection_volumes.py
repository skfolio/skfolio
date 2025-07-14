"""
=====================================
Custom Pre-selection Using Volumes
=====================================

This tutorial demonstrates how to implement a custom :ref:`pre-selection transformer
<pre_selection>` with :ref:`metadata-routing <metadata_routing>`, integrate it into
a `Pipeline`, and run walk-forward cross-validation.
"""

# %%
# Data
# ====
# We will use the S&P 500 :ref:`dataset <datasets>`, which contains daily prices
# of 20 assets from the S&P 500 Index, spanning from 1990-01-02 to 2022-12-28:
import numpy as np
import sklearn.base as skb
import sklearn.feature_selection as skf
import sklearn.utils.validation as skv
from plotly.io import show
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import validate_data

from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import (
    WalkForward,
    cross_val_predict,
)
from skfolio.optimization import EqualWeighted
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)

# %%
# For simplicity, we will generate random volume data:
volumes_usd = np.random.rand(*X.shape) * 1e6

# %%
# Custom Pre-selection Transformer
# ================================
# Let's create a custom pre-selection transformer to retain the top x% of assets
# with the highest average volumes during the fitting period.

class VolumePreSelection(skf.SelectorMixin, skb.BaseEstimator):
    to_keep_: np.ndarray

    def __init__(self, pct_to_keep: float = 0.5):
        self.pct_to_keep = pct_to_keep

    def fit(self, X, y=None, volumes=None):
        # Validate and convert X to a NumPy array
        X = validate_data(self, X)

        # Check parameters
        if not 0 < self.pct_to_keep <= 1:
            raise ValueError(
                "`pct_to_keep` must be between 0 and 1"
            )

        # Validate and convert volumes to a NumPy array
        volumes = skv.check_array(
            volumes,
            accept_sparse=False,
            ensure_2d=False,
            dtype=[np.float64, np.float32],
            order="C",
            copy=False,
            input_name="volumes",
        )
        if volumes.shape != X.shape:
            raise ValueError(
                f"Volume data {volumes.shape} must have the same dimensions as X {X.shape}"
            )

        n_assets = X.shape[1]
        mean_volumes = volumes.mean(axis=0)

        # Select the top `pct_to_keep` assets with the highest average volumes
        n_to_keep = max(1, int(round(self.pct_to_keep * n_assets)))
        selected_idx = np.argsort(mean_volumes)[-n_to_keep:]

        # Performance tip: `argpartition` could be used here for better efficiency
        # (O(n log(n)) vs O(n)).
        self.to_keep_ = np.isin(np.arange(n_assets), selected_idx)
        return self

    def _get_support_mask(self):
        skv.check_is_fitted(self)
        return self.to_keep_

# %%
# Pipeline
# ========
# We create a `Pipeline` that uses our custom pre-selection transformer to retain the
# top 30% of assets based on average volume, followed by an equal-weighted allocation.
# Since we are using volume metadata, we enable metadata-routing and specify how
# to route it with `set_fit_request`:
set_config(enable_metadata_routing=True, transform_output="pandas")

model = Pipeline(
    [
        (
            "pre_selection",
            VolumePreSelection(pct_to_keep=0.3).set_fit_request(
                volumes=True
            ),
        ),
        ("optimization", EqualWeighted()),
    ]
)

# %%
# Cross-Validation
# ================
# We will cross-validate the model using a Walk Forward that rebalances
# the portfolio every 3 months on the 3rd Friday, training on the preceding 6 months:
cv = WalkForward(test_size=3, train_size=6, freq="WOM-3FRI")

pred = cross_val_predict(model, X, cv=cv, params={"volumes": volumes_usd})

# %%
# Display the weights for each rebalancing period:
pred.composition

# %%
# You can also view the weights for each day:
pred.weights_per_observation.tail()

# %%
# Plot the weights per rebalancing period:
fig = pred.plot_composition()
show(fig)

# %%
# |
#
# Plot the full out-of-sample walk-forward path:
pred.plot_cumulative_returns()
