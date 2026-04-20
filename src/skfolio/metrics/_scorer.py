"""Scorer module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

from collections.abc import Callable

import skfolio.typing as skt
from skfolio.measures import BaseMeasure
from skfolio.portfolio import Portfolio
from skfolio.typing import ArrayLike

__all__ = ["make_scorer"]


class _BaseScorer:
    """Base scorer that stores the score function, sign, and bound kwargs."""

    def __init__(
        self,
        score_func: Callable,
        sign: int,
        kwargs: dict,
        response_method: str | None = "predict",
    ):
        self._score_func = score_func
        self._sign = sign
        self._kwargs = kwargs
        self._response_method = response_method

    def __repr__(self) -> str:
        name = getattr(self._score_func, "__name__", repr(self._score_func))
        kwargs_string = "".join([f", {k}={v}" for k, v in self._kwargs.items()])
        response_str = (
            ""
            if self._response_method == "predict"
            else f", response_method={self._response_method!r}"
        )
        return (
            f"make_scorer({name}"
            f"{'' if self._sign > 0 else ', greater_is_better=False'}"
            f"{response_str}"
            f"{kwargs_string})"
        )


class _PortfolioScorer(_BaseScorer):
    """Scorer for portfolio optimization estimators.

    Calls `estimator.predict(X)` and passes the resulting
    :class:`~skfolio.portfolio.Portfolio` to `score_func`.

    Created by :func:`make_scorer` with `response_method="predict"`.
    """

    def __call__(self, estimator, X_test: ArrayLike, y=None) -> float:
        """Compute the score of the estimator prediction on X.

        Parameters
        ----------
        estimator : BaseOptimization
            Trained Portfolio Optimization estimator to use for scoring (e.g.
            :class:`~skfolio.optimization.MeanRisk`).

        X_test : array-like of shape (n_observations, n_assets)
            Test data that will be fed to `estimator.predict`.

        y : ignored
            Present for scikit-learn scorer protocol compatibility.

        Returns
        -------
        score : float
            Score of the estimator prediction on X_test.
        """
        pred = estimator.predict(X_test)
        return self._sign * self._score_func(pred, **self._kwargs)


class _EstimatorScorer(_BaseScorer):
    """Scorer for non-predictor estimators (covariance, expected returns, prior).

    These estimators implement `fit` but not `predict`, so the scorer passes
    the fitted estimator and test data directly to
    `score_func(estimator, X_test, **kwargs)`.

    Created by :func:`make_scorer` with `response_method=None`.
    """

    def __call__(self, estimator, X_test: ArrayLike, y=None) -> float:
        """Score a fitted non-predictor estimator against test data.

        Parameters
        ----------
        estimator : BaseEstimator
            Fitted non-predictor estimator (e.g.
            :class:`~skfolio.moments.EWCovariance`,
            :class:`~skfolio.moments.EWMu`,
            :class:`~skfolio.prior.EmpiricalPrior`).

        X_test : array-like of shape (n_observations, n_assets)
            Test data passed directly to `score_func` alongside
            `estimator`.

        y : ignored
            Present for scikit-learn scorer protocol compatibility.

        Returns
        -------
        score : float
            Score of the estimator on `X_test`.
        """
        return self._sign * self._score_func(estimator, X_test, **self._kwargs)


def make_scorer(
    score_func: skt.Measure | Callable,
    greater_is_better: bool | None = None,
    response_method: str | None = "predict",
    **kwargs,
) -> _PortfolioScorer | _EstimatorScorer:
    """Make a scorer from a :ref:`measure <measures_ref>`, a portfolio score
    function, or a non-predictor estimator score function.

    This function wraps scoring functions for use in model selection:

    * `response_method="predict"` (default): for portfolio optimization
      estimators (e.g.
      :class:`~skfolio.optimization.MeanRisk`). Compatible with
      :class:`~sklearn.model_selection.GridSearchCV` and
      :func:`~sklearn.model_selection.cross_val_score`.
    * `response_method=None`: for non-predictor estimators
      (covariance, expected returns, prior) that implement `fit` but not
      `predict`. Compatible with both sklearn cross-validation utilities
      and skfolio online utilities
      (:class:`~skfolio.model_selection.OnlineGridSearch`,
      :func:`~skfolio.model_selection.online_score`).

    .. note ::

        For online evaluation of portfolio optimization estimators, pass a
        :ref:`measure <measures_ref>` directly to the `scoring` parameter instead of using
        `make_scorer`. Online evaluation scores the full
        aggregated :class:`~skfolio.portfolio.MultiPeriodPortfolio` rather than averaging
        per-fold scores.

    Parameters
    ----------
    score_func : Measure | callable
        If `score_func` is a :ref:`measure <measures_ref>`, we return the
        measure of the predicted :class:`~skfolio.portfolio.Portfolio` times
        `1` or `-1` depending on `greater_is_better`.
        `response_method` must be `"predict"` in this case.

        If `response_method="predict"`, `score_func` must be a score
        function (or loss function) with signature
        `score_func(pred, **kwargs)` where `pred` is the predicted
        :class:`~skfolio.portfolio.Portfolio`.

        If `response_method=None`, `score_func` must be a score function
        (or loss function) with signature
        `score_func(estimator, X_test, **kwargs)` where `estimator` is
        the fitted non-predictor estimator and `X_test` the realized
        returns.

    greater_is_better : bool, optional
        Whether `score_func` is a score function (high is good) or a loss
        function (low is good).  In the latter case the scorer sign-flips
        the outcome so that higher values always indicate a better model.
        The default (`None`) is:

        * If `score_func` is a :ref:`measure <measures_ref>`:

            * `True` for :class:`~skfolio.measures.PerfMeasure` and
              :class:`~skfolio.measures.RatioMeasure`.
            * `False` for :class:`~skfolio.measures.RiskMeasure` and
              :class:`~skfolio.measures.ExtraRiskMeasure`.

        * Otherwise, `True`.

    response_method : str or None, default="predict"
        Determines how the scorer obtains predictions. Only `"predict"` and
        `None` are supported:

        * `"predict"`: call `estimator.predict(X)` and pass the resulting
          :class:`~skfolio.portfolio.Portfolio` to `score_func`. Use for
          portfolio optimization estimators (e.g.
          :class:`~skfolio.optimization.MeanRisk`).
        * `None`: pass `(estimator, X_test)` directly to `score_func`
          without calling any response method. Use for non-predictor
          estimators (covariance, expected returns, prior).

    **kwargs : additional arguments
        Additional parameters to be passed to `score_func`.

    Returns
    -------
    scorer : callable
        Callable object with signature `scorer(estimator, X, y=None)`
        that returns a scalar score (higher is better).

    Examples
    --------
    Portfolio scorer from a measure:

    >>> from skfolio.measures import RatioMeasure
    >>> scorer = make_scorer(RatioMeasure.SHARPE_RATIO)

    Portfolio scorer from a custom function:

    >>> def custom(pred):
    ...     return pred.mean - 2 * pred.variance
    >>> scorer = make_scorer(custom)

    Non-predictor estimator scorer for covariance evaluation:

    >>> from skfolio.metrics import portfolio_variance_qlike_loss
    >>> import numpy as np
    >>> scorer = make_scorer(
    ...     portfolio_variance_qlike_loss,
    ...     greater_is_better=False,
    ...     response_method=None,
    ...     portfolio_weights=np.ones(20) / 20,
    ... )
    """
    if response_method not in ("predict", None):
        raise ValueError(
            f"response_method must be 'predict' (for portfolio optimization "
            f"estimators) or None (for non-predictor estimators), "
            f"got {response_method!r}."
        )

    if callable(score_func):
        if greater_is_better is None:
            greater_is_better = True
        measure = None
    else:
        measure = score_func
        if not isinstance(measure, BaseMeasure):
            raise TypeError("`score_func` must be a callable or a measure")
        if response_method is None:
            raise ValueError(
                "response_method=None is not supported when score_func is a "
                "measure. Measures require response_method='predict'."
            )
        if greater_is_better is None:
            if measure.is_perf or measure.is_ratio:
                greater_is_better = True
            else:
                greater_is_better = False

        def score_func(pred: Portfolio) -> float:
            return getattr(pred, measure.value)

        score_func.__name__ = repr(measure)

    sign = 1 if greater_is_better else -1

    if response_method == "predict":
        return _PortfolioScorer(score_func, sign, kwargs)
    return _EstimatorScorer(score_func, sign, kwargs, response_method=None)
