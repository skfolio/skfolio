"""Scorer module"""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from collections.abc import Callable

import numpy.typing as npt

import skfolio.typing as skt
from skfolio.optimization import BaseOptimization
from skfolio.portfolio import Portfolio


class _PortfolioScorer:
    """Portfolio Scorer wrapper"""

    def __init__(self, score_func: Callable, sign: int, kwargs: dict):
        self._score_func = score_func
        self._kwargs = kwargs
        self._sign = sign

    def __repr__(self) -> str:
        """String representation of the `PortfolioScorer`."""
        kwargs_string = "".join([f", {k}={v}" for k, v in self._kwargs.items()])
        return (
            f"make_scorer({self._score_func.__name__}"
            f"{'' if self._sign > 0 else ', greater_is_better=False'}"
            f"{kwargs_string})"
        )

    def __call__(self, estimator: BaseOptimization, X: npt.ArrayLike) -> float:
        """Compute the score of the estimator prediction on X.

        Parameters
        ----------
        estimator : BaseOptimization
            Trained estimator to use for scoring.

        X : array-like of shape (n_observations, n_assets)
            Test data that will be fed to estimator.predict.

        Returns
        -------
        score : float
            Score of the estimator prediction on X.
        """
        pred = estimator.predict(X)
        return self._sign * self._score_func(pred, **self._kwargs)


def make_scorer(
    score_func: skt.Measure | Callable,
    greater_is_better: bool | None = None,
    **kwargs,
) -> Callable:
    """Make a scorer from a :ref:`measure <measures_ref>` or from a custom score
    function.

    This is a modified version from `scikit-learn` `make_scorer` for enhanced
    functionalities with `Portfolio` objects.

    This factory function wraps scoring functions for use in
    `sklearn.model_selection.GridSearchCV` and
    `sklearn.model_selection.cross_val_score`.

    Parameters
    ----------
    score_func : Measure | callable
        If `score_func` is a :ref:`measure <measures_ref>`, we return the measure of
        the predicted :class:`~skfolio.portfolio.Portfolio` times `1` or `-1`
        depending on the `greater_is_better` parameter.

        Otherwise, `score_func` must be a score function (or loss function) with
        signature `score_func(pred, **kwargs)`. The argument `pred` is the predicted
        :class:`~skfolio.portfolio.Portfolio`.

        Note that you can convert this portfolio object into a numpy array of price
        returns with `np.asarray(pred)`.

    greater_is_better : bool, optional
        If this is set to True, `score_func` is a score function (default) meaning high
        is good, otherwise it is a loss function, meaning low is good.
        In the latter case, the scorer object will sign-flip the outcome of the `score_func`.
        The default (`None`) is to use:

        * If `score_func` is a :ref:`measure <measures_ref>`:

            * True for `PerfMeasure` and `RationMeasure`
            * False for `RiskMeasure` and `ExtraRiskMeasure`.

        * Otherwise, True.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    scorer : callable
        Callable object that returns a scalar score.
    """
    if callable(score_func):
        if greater_is_better is None:
            greater_is_better = True

    else:
        measure = score_func
        if not isinstance(measure, skt.Measure):
            raise TypeError("`score_func` must be a callable or a measure")
        if greater_is_better is None:
            if measure.is_perf or measure.is_ratio:
                greater_is_better = True
            else:
                greater_is_better = False

        def score_func(pred: Portfolio) -> float:
            """Score function"""
            return getattr(pred, measure.value)

    sign = 1 if greater_is_better else -1
    return _PortfolioScorer(score_func, sign, kwargs)
