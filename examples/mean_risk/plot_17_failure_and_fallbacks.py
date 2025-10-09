"""
=====================
Failure and Fallbacks
=====================

This tutorial introduces the optimization parameters `fallback` and `raise_on_failure`.

Optimization can sometimes fail during a given rebalancing. For example, a convex
mean-variance problem with strict risk or sector constraints may become infeasible on
specific dates. Such failures must be handled explicitly depending on the use case
(production vs. research).

Fallback
========
The `fallback` parameter lets you define an estimator, or a list of estimators, to try
in order when the primary optimization raises an error during `fit`. Alternatively, you
can use `"previous_weights"` to reuse the last valid allocation.

Each attempt is recorded in `fallback_chain_`, and the successful estimator is available
through `fallback_`.

This mechanism is essential in automated pipelines, ensuring that optimization failures
never halt production runs while preserving full reproducibility and traceability.
Beyond safeguarding workflows, it can also be used to deliberately relax constraints in
a controlled manner when strict convergence cannot be achieved.

Raise on Failure
================
In research, cross-validation and hyperparameter tuning (e.g. walk-forward, multiple
randomized cross-validation), it's often useful to let all runs complete while keeping
a full record of failures instead of stopping on the first failed rebalancing.

- Set `raise_on_failure=True` (default) to fail fast. This is useful in production when
  the primary optimization or the fallback cascade is expected to succeed.

- Set `raise_on_failure=False` to continue uninterrupted. This is useful in research
  and cross-validation. When a failure occurs, `predict` returns a
  :class:`~skfolio.portfolio.FailedPortfolio` (think of it as an augmented NaN) that
  carries diagnostics such as `optimization_error` and `fallback_chain`, while remaining
  API-compatible with downstream analytics.

"""

# %%
# Data and Setup
# ==============
# Load the S&P 500 :ref:`dataset <datasets>` and split into train/test.
import pandas as pd
from plotly.io import show
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import validate_data

from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import BaseOptimization, EqualWeighted, MeanRisk
from skfolio.preprocessing import prices_to_returns
from skfolio.typing import Fallback, MultiInput
from skfolio.utils.stats import rand_weights

# Load S&P 500 dataset and split train/test
prices = load_sp500_dataset()
prices = prices["2010":]
X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)


# %%
# Fallback
# ========
# Let's start with a simple example.
# The primary model is a minimum-variance optimization made intentionally infeasible
# (the assets' minimum weights are set to 10%, which exceeds the feasible upper bound
# of 1/n_assets = 5%). As a fallback, we provide a feasible minimum-variance model
# with a 2% minimum weight constraint:
model = MeanRisk(
    min_weights=0.1,  # intentionally infeasible
    fallback=MeanRisk(min_weights=0.02),  # feasible fallback
)
model.fit(X_train)
print(model.weights_)

# %%
# Diagnostics
# -----------
# Let's retrieve the fitted fallback that produced the final result:
print(model.fallback_)

# %%
# Let's display the sequence of attempts and their outcomes:
print(model.fallback_chain_)

# %%
# The fallback audit trail is also propagated to the predicted portfolio:
portfolio = model.predict(X_test)
assert portfolio.fallback_chain == model.fallback_chain_


# %%
# Multiple fallbacks
# ------------------
# We can also provide a list of fallbacks to be tried in order, including
# "previous_weights" as a terminal safety net:
model = MeanRisk(
    min_weights=0.1,
    previous_weights={
        "AAPL": 0.4,
        "AMD": 0.2,
        "UNH": 0.4,
    },  # any missing assets default to 0
    fallback=[
        MeanRisk(min_weights=0.02),
        MeanRisk(min_weights=0.01),
        EqualWeighted(),
        "previous_weights",
    ],
)

# %%
# Chaining
# --------
# We can also nest fallbacks.
# The chain is evaluated depth-first from the primary estimator to the
# first successful fallback, recording each attempt in `fallback_chain_`.
# This is equivalent to providing an ordered list:
model = MeanRisk(
    min_weights=0.1,
    fallback=MeanRisk(
        min_weights=0.02,
        fallback=MeanRisk(
            min_weights=0.01,
            fallback=EqualWeighted(),
        ),
    ),
)

# %%
# Fallback in cross-validation
# ============================
# Fallback behavior is fully preserved in cross-validation.
#
# When using `cross_val_predict`, all diagnostics (e.g., fallback chains and errors)
# are propagated to the resulting portfolios in the `MultiPeriodPortfolio`:
#
# - Each individual :class:`~skfolio.portfolio.Portfolio`
#   (or :class:`~skfolio.portfolio.FailedPortfolio`) produced during rebalancing
#   carries its own `fallback_chain` and `optimization_error`.
# - Global counts and statistics (e.g., the number of portfolios that required a
#   fallback) are available through summary attributes such as
#   `n_fallback_portfolios` and `n_failed_portfolios`.
# - The `summary()` method consolidates performance and diagnostic information
#   across all rebalances.
model = MeanRisk(min_weights=0.1, fallback=MeanRisk(min_weights=0.02))

# Rebalance semiannually on the third Friday (WOM-3FRI), training on the prior 12 months
walk_forward = WalkForward(test_size=6, train_size=12, freq="WOM-3FRI")

pred = cross_val_predict(model, X, cv=walk_forward)

# %%
# Let's retrieve the fallback chain of the first portfolio:
print(pred[0].fallback_chain)

# %%
# Let's print the number of portfolios in
# :class:`~skfolio.portfolio.MultiPeriodPortfolio` where a fallback was used:
print(pred.n_fallback_portfolios)

# %%
# Finally, let's display the last four rows of the `MultiPeriodPortfolio` summary,
# which contain the fallback statistics:
print(pred.summary().iloc[-4:])


# %%
# Failure handling
# ================
# In this section, we show how to handle optimization failures using the
# `raise_on_failure` parameter.
# As an example, we create a custom optimization that intentionally fails during `fit`
# when the first date of the input window falls on an even day of the month, or when
# `always_fail=True`.
class CustomOptimization(BaseOptimization):
    """Dummy optimization that intentionally fails during `fit` when the first
    date of the input window is an even day-of-month, or when `always_fail=True`."""

    def __init__(
        self,
        always_fail: bool = False,
        portfolio_params: dict | None = None,
        fallback: Fallback = None,
        previous_weights: MultiInput | None = None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
            previous_weights=previous_weights,
        )
        self.always_fail = always_fail

    def fit(self, X: pd.DataFrame, y=None):
        validate_data(self, X)
        # Fail when first observation date has an even day-of-month, or always.
        if self.always_fail:
            raise RuntimeError("Forced failure")
        first_day = X.index[0].day
        if first_day % 2 == 0:
            raise RuntimeError("Forced failure (even-start window)")
        n_assets = X.shape[1]
        self.weights_ = rand_weights(n_assets)
        return self


# %%
# By default, as with all scikit-learn estimators, failures raise an error during `fit`:
model = CustomOptimization(always_fail=True)
try:
    model.fit(X_train)
except RuntimeError as err:
    print(err)


# %%
# By setting `raise_on_failure=False`, a warning is emitted instead of raising an error,
# and `weights_` are set to `None`, with the error message stored in `error_`:
model = CustomOptimization(always_fail=True, raise_on_failure=False)
model.fit(X_train)
print(model.weights_)
print(model.error_)

# %%
# In this case, calling `predict` will return a `FailedPortfolio` carrying the audit
# trail in `optimization_error` and `fallback_chain` (if any fallbacks occurred).
portfolio = model.predict(X_test)
print(portfolio)
print(portfolio.optimization_error)

# %%
# Setting `raise_on_failure=False` is useful for cross-validation and hyperparameter
# tuning as it allows all runs to complete without stopping at the first rebalancing
# failure. Let's instantiate our custom optimization and run a walk-forward analysis
# where failures occur deterministically on even-start windows:
model = CustomOptimization(raise_on_failure=False)
pred = cross_val_predict(model, X, cv=walk_forward)


# %%
# `cross_val_predict` completed without interruption.
# The resulting `MultiPeriodPortfolio` is composed of both `Portfolio` and
# `FailedPortfolio` objects:
print(pred.portfolios)

# %%
# Let's print the number of failed portfolios:
print(pred.n_failed_portfolios)

# %%
# Even though `MultiPeriodPortfolio` contains failed portfolios, all statistics and
# plots still work properly. This is because `FailedPortfolio` is designed to behave
# like non-propagating NaNs:
print(pred.summary())

# %%
# As shown below, `MultiPeriodPortfolio` plots gracefully handle `FailedPortfolio`
# instances; for cumulative returns, these appear as gaps corresponding to failed
# periods:
fig = pred.plot_cumulative_returns()
show(fig)

# %%
# |
# Finally, let's inspect the first failed portfolio:
failed_ptf = pred.failed_portfolios[0]
print(failed_ptf.optimization_error)

# %%
# To replay the optimization on the failed period, we can run:

# model.fit(failed_ptf.X)

