r"""
===========================================
Online Evaluation of Portfolio Optimization
===========================================

This tutorial shows how to tune a :class:`~skfolio.optimization.MeanRisk` estimator with
online search and evaluate it out-of-sample with an online walk-forward procedure.

Unlike the :ref:`previous tutorial
<sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py>`,
which tuned the covariance estimator in isolation, here we optimize the portfolio model
end-to-end using a portfolio-level metric.

The online approach is equivalent to combining scikit-learn's
:class:`~sklearn.model_selection.GridSearchCV` with
:class:`~skfolio.model_selection.WalkForward` using `expand_train=True`, but instead of
refitting every candidate from scratch at each split, it calls `partial_fit` to
incrementally update each estimator. This is significantly faster for estimators that
support this method.
"""

# %%
# Data
# ====
# We load the S&P 500 :ref:`dataset <datasets>` composed of the daily prices of 20
# assets from the S&P 500 Index composition starting from 2010-01-04 up to 2022-12-28.
from plotly.io import show
import numpy as np

from skfolio import Population
from skfolio.datasets import load_sp500_dataset
from skfolio.measures import RatioMeasure
from skfolio.model_selection import OnlineGridSearch, online_predict, online_score
from skfolio.moments import EWMu, RegimeAdjustedEWCovariance
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior

prices = load_sp500_dataset()
X = prices_to_returns(prices)
X = X["2010":]

# %%
# Baseline Portfolio Model
# ========================
# We start with a simple Minimum Variance optimization via
# :class:`~skfolio.optimization.MeanRisk`. All sub-estimators support `partial_fit`, so
# the entire pipeline can be updated incrementally during walk-forward evaluation.

baseline_model = MeanRisk(
    prior_estimator=EmpiricalPrior(
        mu_estimator=EWMu(half_life=40),
        covariance_estimator=RegimeAdjustedEWCovariance(
            half_life=40,
            corr_half_life=80,
            regime_half_life=20,
        ),
    ),
)

# %%
# Online Portfolio Search
# =======================
# We search directly over the portfolio estimator using a portfolio-level metric. For
# portfolio optimization estimators, :class:`~skfolio.model_selection.OnlineGridSearch`
# defaults to the Sharpe ratio when `scoring=None`.
#
# We tune both the optimization objective and a few covariance hyperparameters.
# The double-underscore syntax reaches into nested sub-estimators, just like in
# scikit-learn model selection.
#
# Here, `warmup_size=252` reserves the first year of observations for initialization and
# `test_size=5` evaluates windows of 5 consecutive daily observations (one trading week).

portfolio_search = OnlineGridSearch(
    estimator=baseline_model,
    param_grid={
        "objective_function": [
            ObjectiveFunction.MINIMIZE_RISK,
            ObjectiveFunction.MAXIMIZE_RATIO,
        ],
        "prior_estimator__covariance_estimator__half_life": [20, 40, 60],
        "prior_estimator__covariance_estimator__corr_half_life": [40, 80],
    },
    warmup_size=252,
    test_size=5,
    n_jobs=-1,
)
portfolio_search.fit(X)

# %%
print(f"Best params: {portfolio_search.best_params_}")
print(f"Best score (Annualized Sharpe): {np.sqrt(252) * portfolio_search.best_score_:.6f}")

# %%
# Online Evaluation
# =================
# :func:`~skfolio.model_selection.online_predict` walks forward through the data,
# updates the estimator via `partial_fit` at each step, and predicts on the next test
# window. The result is a :class:`~skfolio.portfolio.MultiPeriodPortfolio`.

baseline_prediction = online_predict(
    baseline_model,
    X,
    warmup_size=252,
    test_size=5,
    portfolio_params=dict(name="Baseline"),
)

tuned_prediction = online_predict(
    portfolio_search.best_estimator_,
    X,
    warmup_size=252,
    test_size=5,
    portfolio_params=dict(name="Tuned"),
)

# %%
# Portfolio Comparison
# ====================
# We collect both portfolio evaluations into a :class:`~skfolio.population.Population`
# for side-by-side comparison.

population = Population([baseline_prediction, tuned_prediction])
population.summary()

# %%
fig = population.plot_cumulative_returns()
show(fig)

# %%
# Online Score
# ============
# :func:`~skfolio.model_selection.online_score` provides the same walk-forward
# evaluation as a single scalar, which is useful for quick comparisons in code.

baseline_score = online_score(
    baseline_model,
    X,
    warmup_size=252,
    test_size=5,
    scoring=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
)
tuned_score = online_score(
    portfolio_search.best_estimator_,
    X,
    warmup_size=252,
    test_size=5,
    scoring=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
)

print(f"Baseline Sharpe: {baseline_score:.4f}")
print(f"Tuned Sharpe: {tuned_score:.4f}")

# %%
# Conclusion
# ==========
# This tutorial demonstrated the portfolio-level online workflow:
#
# 1. Define an incremental :class:`~skfolio.optimization.MeanRisk` estimator whose
#    sub-estimators all support `partial_fit`.
# 2. Tune it with :class:`~skfolio.model_selection.OnlineGridSearch` using a
#    portfolio-level metric.
# 3. Evaluate the tuned estimator out-of-sample with
#    :func:`~skfolio.model_selection.online_predict` and visualize the results
#    with :class:`~skfolio.population.Population`.
# 4. Summarize the walk-forward performance as a scalar with
#    :func:`~skfolio.model_selection.online_score`.
#
# This complements the
# :ref:`previous tutorial <sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py>`:
# covariance tuning improves the statistical forecast, while direct portfolio search
# optimizes the full allocation problem end-to-end.
