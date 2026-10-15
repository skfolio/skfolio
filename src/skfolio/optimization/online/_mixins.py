"""Mixins for online portfolio selection."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from enum import auto
from numbers import Real
from typing import ClassVar

from sklearn.utils._param_validation import Interval

from skfolio.prior import BasePrior
from skfolio.utils.tools import AutoEnum


class FTWStrategy(AutoEnum):
    """Available follow-the-winner numerical strategies."""

    OGD = auto()
    EG = auto()
    ADAGRAD = auto()
    ADABARRONS = auto()
    SWORD_SMALL = auto()
    SWORD_VAR = auto()
    SWORD_BEST = auto()
    SWORD_PP = auto()
    PROD = auto()  # Soft-Bayes Prod algorithm


class FTLStrategy(AutoEnum):
    """
    Follow-The-Loser families (mean-reversion).

    - OLMAR: Online Moving Average Reversion
    - PAMR: Passive-Aggressive Mean Reversion
    - CWMR: Confidence-Weighted Mean Reversion (distributional, second-order)
    - RMR: Robust Median Reversion (L1-median, outlier-robust)
    """

    OLMAR = auto()
    PAMR = auto()
    CWMR = auto()
    RMR = auto()


class PAMRVariant(AutoEnum):
    """Variants for Passive-Aggressive Mean Reversion (PAMR).

    - SIMPLE: the original PAMR formula with τ = loss / ||c||²
    - SLACK_LINEAR: introduces a linear slack variable ξ with weight C (PAMR-1)
                    τ = min(C, loss / ||c||²)
    - SLACK_QUADRATIC: quadratic slack regularisation ξ² with weight C (PAMR-2)
                       τ = loss / (||c||² + 1/(2C))
    See Li & Hoi book (2012), Equations 9.2, 9.3, 9.4.
    """

    SIMPLE = auto()  # PAMR (original)
    SLACK_LINEAR = auto()  # PAMR-1
    SLACK_QUADRATIC = auto()  # PAMR-2


class UpdateMode(AutoEnum):
    """Update mode for mean-reversion strategies."""

    PA = auto()  # Passive-Aggressive (closed-form)
    MD = auto()  # Mirror Descent (OCO-style)


class OLMARPredictor(AutoEnum):
    """Predictor for OLMAR strategy.

    Two variants are implemented from the Li and Hoi "Online Portfolio Selection" book:
    - MAR-1 (SMA) Simple Moving Average Eq. 11.1 ("1" version in OLMAR)
    - MAR-2 (EWMA) Exponentially Weighted Moving Average Eq. 11.2 ("2" version in OLMAR)
    """

    SMA = "sma"
    EWMA = "ewma"  # Exponentially Weighted Moving Average


class OnlineParameterConstraintsMixin:
    """Shared parameter validation for online portfolio estimators."""

    _parameter_constraints: ClassVar[dict] = {
        "store_weights": ["boolean"],
        "initial_weights": [Real, dict, "array-like", None],
        # initial_wealth accepts: scalar float > 0, array of floats summing > 0, or None
        "initial_wealth": [
            Interval(Real, 0, None, closed="neither"),
            "array-like",
            None,
        ],
        "previous_weights": [Real, dict, "array-like", None],
        "transaction_costs": [
            Interval(Real, 0, None, closed="left"),
            dict,
            "array-like",
        ],
        "management_fees": [
            Interval(Real, 0, 1, closed="left"),
            dict,
            "array-like",
        ],
        "min_weights": [Real, "array-like", dict, None],
        "max_weights": [Real, "array-like", dict, None],
        "budget": [Real, None],
        "min_budget": [Real, None],
        "max_budget": [Real, None],
        "max_turnover": [Interval(Real, 0, None, closed="left"), None],
        "groups": ["array-like", dict, None],
        "linear_constraints": [list, None],
        "left_inequality": ["array-like", None],
        "right_inequality": ["array-like", None],
        "X_tracking": ["array-like", None],
        "tracking_error_benchmark": ["array-like", None],
        "max_tracking_error": [Interval(Real, 0, None, closed="neither"), None],
        "min_return": [Real, None],
        "min_acceptable_return": [Real, None],
        "risk_free_rate": [Real],
        "max_standard_deviation": [Interval(Real, 0, None, closed="left"), None],
        "max_semi_variance": [Interval(Real, 0, None, closed="left"), None],
        "max_semi_deviation": [Interval(Real, 0, None, closed="left"), None],
        "max_mean_absolute_deviation": [Interval(Real, 0, None, closed="left"), None],
        "max_first_lower_partial_moment": [
            Interval(Real, 0, None, closed="left"),
            None,
        ],
        "max_worst_realization": [Interval(Real, 0, None, closed="left"), None],
        "max_cvar": [Interval(Real, 0, None, closed="left"), None],
        "max_evar": [Interval(Real, 0, None, closed="left"), None],
        "max_max_drawdown": [Interval(Real, 0, None, closed="left"), None],
        "max_average_drawdown": [Interval(Real, 0, None, closed="left"), None],
        "max_cdar": [Interval(Real, 0, None, closed="left"), None],
        "max_edar": [Interval(Real, 0, None, closed="left"), None],
        "max_ulcer_index": [Interval(Real, 0, None, closed="left"), None],
        "max_gini_mean_difference": [Interval(Real, 0, None, closed="left"), None],
        "cvar_beta": [Interval(Real, 0, 1, closed="both")],
        "evar_beta": [Interval(Real, 0, 1, closed="both")],
        "cdar_beta": [Interval(Real, 0, 1, closed="both")],
        "edar_beta": [Interval(Real, 0, 1, closed="both")],
        "add_constraints": [callable, None],
        "prior_estimator": [BasePrior, None],
        "covariance": ["array-like", None],
        "max_variance": [Interval(Real, 0, None, closed="neither"), None],
    }
