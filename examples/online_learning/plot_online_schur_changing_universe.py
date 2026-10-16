"""
================================================
Online Schur Allocation with a Changing Universe
================================================

This tutorial demonstrates online learning with
:class:`~skfolio.optimization.SchurComplementary`. We follow a newly listed asset
through its warm-up period, then examine a holiday and a delisting.

`partial_fit` updates the estimates with new observations and computes a new
allocation. Assets can become investable or leave the investment universe without
restarting estimation.
"""

# %%
# Returns and Active Mask
# =======================
# We generate 180 observations for four assets with a common market component.
# The data includes three events:
#
# * Asset "C" is listed at observation 61.
# * Asset "A" has a holiday at observations 101 to 105.
# * Asset "B" is delisted at observation 141.
#
# The `active_mask` distinguishes missing observations from inactive periods.
# During a holiday, a NaN return with `active_mask=True` tells the EW estimators
# to freeze the asset's mean and variance estimates until observations resume.
# Before listing and after delisting, `active_mask=False` resets its estimates
# and marks it unavailable for allocation. See :ref:`fixed_asset_schema` for
# the convention used to represent changing universes in the data.
import numpy as np
import pandas as pd
from plotly.io import show
from sklearn import set_config

from skfolio.distance import CovarianceDistance
from skfolio.model_selection import online_predict
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import SchurComplementary
from skfolio.prior import EmpiricalPrior
from skfolio.seriation import SpectralSeriation

set_config(enable_metadata_routing=True)

rng = np.random.default_rng(42)
market = rng.normal(0, 0.006, size=(180, 1))
X = pd.DataFrame(
    market + rng.normal(0, 0.01, size=(180, 4)),
    index=pd.RangeIndex(1, 181, name="Observation"),
    columns=["A", "B", "C", "D"],
)
active_mask = pd.DataFrame(True, index=X.index, columns=X.columns)

X.loc[:60, "C"] = np.nan
active_mask.loc[:60, "C"] = False

X.loc[101:105, "A"] = np.nan

X.loc[141:, "B"] = np.nan
active_mask.loc[141:, "B"] = False

# %%
# We inspect six consecutive observations around the listing of asset "C".
# Its first return appears at observation 61.
X.loc[58:63]

# %%
# Choosing a Seriation Algorithm
# ==============================
# Schur orders the assets, then recursively splits them into groups for
# allocation. The `seriation_estimator` determines this order, so it can affect
# portfolio weights even when the covariance estimate is unchanged.
#
# :class:`~skfolio.seriation.HierarchicalSeriation` is the default for HRP and
# Schur. It derives the asset order from a hierarchical clustering tree, using
# Ward linkage and optimal leaf ordering by default. The linkage rule can be
# changed, and the clustering tree can be inspected in a dendrogram.
#
# :class:`~skfolio.seriation.SpectralSeriation` computes the order directly from
# the distance matrix, without building a clustering tree. It can keep allocation
# groups more stable under small changes in distances, which may reduce turnover.
# Its online updates also take the previous seriation into account. The benefit
# depends on the data, allocation method and rebalancing schedule. See
# :ref:`seriation_turnover` for more details.
#
# Both can be used in online Schur allocations as assets become investable or
# leave the universe. Schur refits `HierarchicalSeriation` at each update and
# calls `partial_fit` for `SpectralSeriation`.
#
# Here we use `SpectralSeriation`. Leaving `seriation_estimator` unset selects
# `HierarchicalSeriation` instead. See :ref:`asset_seriation` for the distance
# and seriation settings.

# %%
# Model Configuration
# ===================
# We use :class:`~skfolio.prior.EmpiricalPrior` with exponentially weighted
# estimates from :class:`~skfolio.moments.EWMu` and
# :class:`~skfolio.moments.EWCovariance`. Both support `partial_fit` and use
# `active_mask` to distinguish missing observations from inactive periods.
#
# `half_life=30` controls how quickly older returns lose their influence.
# With `min_observations=20`, an asset needs 20 valid returns before its mean
# and covariance estimates become available for allocation. Until then, these
# estimates are NaN. Inactive periods and missing returns do not count toward
# this threshold.
#
# We configure :class:`~skfolio.distance.CovarianceDistance` with
# `covariance_estimator="precomputed"` to convert the prior's covariance into
# distances. This avoids fitting and warming up a second covariance estimator.
# Schur uses the same prior covariance to compute portfolio weights.
model = SchurComplementary(
    gamma=0.5,
    prior_estimator=EmpiricalPrior(
        mu_estimator=EWMu(half_life=30, min_observations=20),
        covariance_estimator=EWCovariance(half_life=30, min_observations=20),
    ),
    distance_estimator=CovarianceDistance("precomputed"),
    seriation_estimator=SpectralSeriation(),
)

# %%
# Initial Warm-Up
# ===============
# We call `fit` on the first 40 observations. Assets "A", "B" and "D" have
# enough data to become investable. Asset "C" is still inactive and receives
# zero weight.
#
# With metadata routing enabled, the EW estimators request `active_mask` by
# default, so we can pass the mask directly to Schur.
model.fit(X.loc[:40], active_mask=active_mask.loc[:40])
print(model.weights_)

# %%
# Online Updates
# ==============
# We update the fitted model with the next observation using `partial_fit`.
# Only the new data is supplied. Larger batches are also supported, with an
# allocation computed at the end of each batch. Calling `fit` instead would
# restart estimation from the supplied data.
model.partial_fit(X.loc[[41]], active_mask=active_mask.loc[[41]])
print(model.weights_)

# %%
# Online Evaluation
# =================
# :func:`~skfolio.model_selection.online_predict` automates the updates and
# collects the predicted portfolios in a
# :class:`~skfolio.portfolio.MultiPeriodPortfolio`. It starts from an unfitted
# clone of `model` and initializes it on the first 40 observations with
# `partial_fit`. It then predicts each following observation before learning
# from it, so the weights only use previously observed returns.
#
# The active mask is passed through `params` and sliced with each update.
# `test_size=1` gives one portfolio per observation.
prediction = online_predict(
    model,
    X,
    warmup_size=40,
    test_size=1,
    params={"active_mask": active_mask},
)

# %%
# We use the portfolio's weight plot and display each asset as a separate line
# to make the changes easier to follow. The shaded areas mark the listing,
# warm-up, holiday and delisting periods.
fig = prediction.plot_weights_per_observation()

fig.update_traces(
    stackgroup=None,
    fill="none",
    line_width=2,
    hovertemplate="Observation %{x}<br>Weight: %{y:.2%}",
)
fig.update_layout(title="Schur Allocation as the Universe Changes")
fig.update_yaxes(title_text="Target weight")
fig.update_xaxes(title_text="Observation", range=[41, X.index[-1]])
for start, end, color, label in [
    (41, 61, "#D9A066", '"C" not yet listed'),
    (61, 81, "#FFD43B", '"C" warm-up'),
    (101, 106, "#70AD47", '"A" holiday'),
    (141, X.index[-1], "#D65F5F", '"B" delisted'),
]:
    fig.add_vrect(
        x0=start,
        x1=end,
        fillcolor=color,
        opacity=0.3,
        line_width=0,
        layer="below",
        annotation_text=label,
        annotation_position="top left",
    )
show(fig)

# %%
# Warm-Up, Holidays and Delisting
# ===============================
# Asset "C" completes warm-up with its twentieth valid return at observation 80
# and first receives a nonzero weight at observation 81.
#
# The EW estimators also correct the initialization bias caused by starting
# their estimates at zero. They rescale the estimates using each asset's valid
# observation count, so early estimates are not damped toward zero. This
# correction is separate from the warm-up threshold, which controls when the
# estimates become available for allocation.
#
# During the holiday for asset "A" (observations 101 to 105), its mean and
# variance estimates freeze. It remains investable, and its target weight can
# change as the estimates for other assets are updated.
#
# When asset "B" is delisted at observation 141, `active_mask=False` resets its
# EW estimates. The next portfolio, at observation 142, assigns it zero weight.
# If the asset re-enters the universe later, it needs another warm-up period.
#
# See :ref:`online_failure_handling` for handling failed updates.
