r"""
========================================
Online Covariance Hyperparameter Tuning
========================================

This tutorial shows how to tune covariance estimator hyperparameters in an online
setting using :class:`~skfolio.model_selection.OnlineGridSearch` and
:class:`~skfolio.model_selection.OnlineRandomizedSearch`.

The online approach is equivalent to combining scikit-learn's
:class:`~sklearn.model_selection.GridSearchCV`
(or :class:`~sklearn.model_selection.RandomizedSearchCV`) with
:class:`~skfolio.model_selection.WalkForward` using `expand_train=True`, but instead of
refitting every candidate from scratch at each split, it calls `partial_fit` to
incrementally update the estimator. This is significantly faster for estimators that
support this method.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2010-01-04 up to 2022-12-28.
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import show
from scipy.stats import uniform

from skfolio.datasets import load_sp500_dataset
from skfolio.metrics import (
    diagonal_calibration_loss,
    make_scorer,
    portfolio_variance_qlike_loss,
)
from skfolio.model_selection import (
    OnlineGridSearch,
    OnlineRandomizedSearch,
    online_score,
)
from skfolio.moments import RegimeAdjustedEWCovariance
from skfolio.preprocessing import prices_to_returns

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X = X["2010":]

# %%
# Build Scorers
# =============
# We build scorers with :func:`~skfolio.metrics.make_scorer`.
# We set `response_method=None` because a covariance estimator is a non-predictor
# estimator (it does not implement `predict`), and `greater_is_better=False` because
# both losses are minimized.

qlike_scorer = make_scorer(
    portfolio_variance_qlike_loss,
    greater_is_better=False,
    response_method=None,
)

calibration_scorer = make_scorer(
    diagonal_calibration_loss,
    greater_is_better=False,
    response_method=None,
)

# %%
# OnlineGridSearch
# ================
# We now tune :class:`~skfolio.moments.RegimeAdjustedEWCovariance` with
# :class:`~skfolio.model_selection.OnlineGridSearch`.
#
# We search over `half_life`, `corr_half_life`, and `regime_half_life`.
# `corr_half_life` controls the correlation smoothing separately from the
# variance half-life, while `regime_half_life` controls how quickly the
# regime adjustment adapts to market changes.
#
# Each candidate is evaluated with a full online walk-forward pass. Here,
# `warmup_size=252` uses the first year for initialization and `test_size=5`
# evaluates windows of 5 consecutive daily observations (one trading week).

grid_search = OnlineGridSearch(
    estimator=RegimeAdjustedEWCovariance(),
    param_grid={
        "half_life": [20, 40, 60],
        "corr_half_life": [40, 80],
        "regime_half_life": [10, 20],
    },
    scoring=qlike_scorer,
    warmup_size=252,
    test_size=5,
    n_jobs=-1,
)
grid_search.fit(X)

# %%
# Let's display the best grid-search hyperparameters and score:
print(f"Grid best params: {grid_search.best_params_}")
print(f"Grid best score: {grid_search.best_score_:.6f}")

# %%
# OnlineRandomizedSearch with Multi-Metric Scoring
# ==================================================
# We can also use
# :class:`~skfolio.model_selection.OnlineRandomizedSearch`, which samples from
# continuous distributions instead of evaluating a full grid. Here, we search
# over the same three hyperparameters with 100 random combinations.
#
# We track both QLIKE and calibration loss. Since multi-metric search requires
# an explicit selection rule, we set `refit="neg_qlike"` so that the best
# estimator is selected according to the QLIKE scorer.

random_search = OnlineRandomizedSearch(
    estimator=RegimeAdjustedEWCovariance(),
    param_distributions={
        "half_life": uniform(loc=20, scale=40),
        "corr_half_life": uniform(loc=40, scale=40),
        "regime_half_life": uniform(loc=10, scale=10),
    },
    n_iter=100,
    scoring={
        "neg_qlike": qlike_scorer,
        "neg_calibration_loss": calibration_scorer,
    },
    refit="neg_qlike",
    warmup_size=252,
    test_size=5,
    n_jobs=-1,
    random_state=1,
)
random_search.fit(X)

# %%
# Let's display the best randomized-search hyperparameters and score:
print(f"Random best params: {random_search.best_params_}")
print(f"Random best score (neg_qlike): {random_search.best_score_:.6f}")

# %%
# Online Score
# ============
# `OnlineRandomizedSearch` already stores the aggregate online scores of all
# sampled candidates in `cv_results_`. We use
# :func:`~skfolio.model_selection.online_score` below only to evaluate a baseline
# specification that was not part of the search. This also illustrates the
# standalone scoring API.

baseline_cov = RegimeAdjustedEWCovariance(
    half_life=40,
    corr_half_life=80,
    regime_half_life=20,
)
cv_results = random_search.cv_results_
best_idx = random_search.best_index_

baseline_scores = online_score(
    baseline_cov,
    X,
    warmup_size=252,
    test_size=5,
    scoring={
        "neg_qlike": qlike_scorer,
        "neg_calibration_loss": calibration_scorer,
    },
)
tuned_scores = {
    "neg_qlike": cv_results["mean_score_neg_qlike"][best_idx],
    "neg_calibration_loss": cv_results["mean_score_neg_calibration_loss"][best_idx],
}

# %%
# Let's compare the baseline and tuned scores. The tuned scores are retrieved
# directly from the search results, while the baseline is scored separately with
# :func:`~skfolio.model_selection.online_score`. Since the scorers negate the
# losses, higher values indicate better performance:

print("Baseline scores:")
print(baseline_scores)
print("Tuned scores:")
print(tuned_scores)

# %%
# Search Trade-Off Plot
# =====================
# The scatter plot below summarizes the aggregate online losses of the 100
# sampled parameter combinations, with the selected candidate highlighted and
# the baseline shown for reference.

results = pd.DataFrame(cv_results["params"])
results["qlike_loss"] = -cv_results["mean_score_neg_qlike"]
results["calibration_loss"] = -cv_results["mean_score_neg_calibration_loss"]

fig = px.scatter(
    results,
    x="qlike_loss",
    y="calibration_loss",
    color="regime_half_life",
    hover_data=["half_life", "corr_half_life", "regime_half_life"],
    color_continuous_scale="Viridis",
    labels={
        "qlike_loss": "QLIKE loss",
        "calibration_loss": "Diagonal calibration loss",
        "regime_half_life": "Regime half-life",
    },
    title="Online Random Search: QLIKE vs Calibration Loss",
)
fig.update_traces(marker=dict(size=9, opacity=0.75, line=dict(width=0)))

fig.add_trace(
    go.Scatter(
        x=[results.loc[best_idx, "qlike_loss"]],
        y=[results.loc[best_idx, "calibration_loss"]],
        mode="markers+text",
        name="Selected candidate",
        text=["Selected"],
        textposition="top right",
        marker=dict(symbol="star", size=16, color="green", line=dict(width=1)),
        showlegend=False,
    )
)
fig.add_trace(
    go.Scatter(
        x=[-baseline_scores["neg_qlike"]],
        y=[-baseline_scores["neg_calibration_loss"]],
        mode="markers+text",
        name="Baseline",
        text=["Baseline"],
        textposition="bottom right",
        marker=dict(symbol="x", size=13, color="black", line=dict(width=2)),
        showlegend=False,
    )
)
show(fig)

# %%
# Conclusion
# ==========
# This tutorial demonstrated how to tune online covariance estimator hyperparameters.
#
# 1. Build scorers with :func:`~skfolio.metrics.make_scorer` and
#    `response_method=None` for non-predictor estimators.
# 2. Use :class:`~skfolio.model_selection.OnlineGridSearch` for small,
#    structured search spaces.
# 3. Use :class:`~skfolio.model_selection.OnlineRandomizedSearch` for larger
#    continuous search spaces, specifying `refit` when scoring is multi-metric.
# 4. Evaluate and compare estimators numerically with
#    :func:`~skfolio.model_selection.online_score`.
#
# In the :ref:`next tutorial <sphx_glr_auto_examples_online_learning_plot_3_online_portfolio_optimization_evaluation.py>`,
# we move from covariance tuning to end-to-end online portfolio optimization
# evaluation with :class:`~skfolio.optimization.MeanRisk`.
