"""Online model validation module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import datetime as dt
import numbers
from collections.abc import Generator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.base as skb
import sklearn.utils as sku
from sklearn.pipeline import Pipeline

import skfolio.typing as skt
from skfolio.measures import BaseMeasure, RatioMeasure
from skfolio.metrics._scorer import _BaseScorer, _EstimatorScorer
from skfolio.model_selection._validation import (
    _asset_names_enabled,
    _get_last_step,
    _is_portfolio_optimization_estimator,
    _route_params,
)
from skfolio.model_selection._walk_forward import WalkForward
from skfolio.portfolio import MultiPeriodPortfolio
from skfolio.typing import ArrayLike, FloatArray
from skfolio.utils.tools import fit_single_estimator

if TYPE_CHECKING:
    from skfolio.optimization._base import BaseOptimization

__all__ = ["online_predict", "online_score"]


def online_predict(
    estimator: BaseOptimization,
    X: ArrayLike,
    y: ArrayLike | None = None,
    warmup_size: int = 252,
    test_size: int = 1,
    freq: str | pd.offsets.BaseOffset | None = None,
    freq_offset: pd.offsets.BaseOffset | dt.timedelta | None = None,
    previous: bool = False,
    purged_size: int = 0,
    reduce_test: bool = False,
    params: dict | None = None,
    portfolio_params: dict | None = None,
) -> MultiPeriodPortfolio:
    r"""Generate out-of-sample portfolios using online learning.

    Walks forward through the data, updating the estimator incrementally via
    `partial_fit` and predicting on each subsequent test window. Unlike
    :func:`~skfolio.model_selection.cross_val_predict`, which clones the estimator for
    each fold, this function maintains a single stateful estimator that accumulates
    knowledge over time.

    The algorithm:

    1. Clone the estimator to ensure a clean, unfitted starting state.
    2. Initialize the estimator on the first `warmup_size` observations via `partial_fit`.
    3. At each step, predict on the test window, then update the model with the newly
       observed data via `partial_fit`.

    If the estimator declares `needs_previous_weights=True`, portfolio weights are
    automatically propagated from one step to the next.

    Parameters
    ----------
    estimator : BaseOptimization
        Portfolio optimization estimator. It must implement `partial_fit`.
        Pipelines are not supported.

    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets. Must be a DataFrame with a `DatetimeIndex` when
        `freq` is provided.

    y : array-like of shape (n_observations, n_targets), optional
        Target data to pass to `partial_fit`.

    warmup_size : int, default=252
        Number of initial observations (or periods when `freq` is set) used for the
        first `partial_fit` call. No predictions are made during warmup.

    test_size : int, default=1
        Length of each test set.
        If `freq` is `None` (default), it represents the number of observations.
        Otherwise, it represents the number of periods defined by `freq`. Controls the
        rebalancing frequency.

    freq : str | pandas.offsets.BaseOffset, optional
        If provided, it must be a frequency string or a pandas DateOffset, and `X` must
        be a DataFrame with an index of type `DatetimeIndex`. In that case,
        `warmup_size` and `test_size` represent the number of periods defined by `freq`
        instead of the number of observations.

    freq_offset : pandas.offsets.BaseOffset | datetime.timedelta, optional
        Only used if `freq` is provided. Offsets `freq` by a pandas DateOffset or a
        datetime timedelta offset.

    previous : bool, default=False
        Only used if `freq` is provided. If set to `True`, and if the period start or
        period end is not in the `DatetimeIndex`, the previous observation is used;
        otherwise, the next observation is used.

    purged_size : int, default=0
        The number of observations to exclude from the end of each training window
        before the test window. Use `purged_size >= 1` when execution is delayed
        relative to observation.

    reduce_test : bool, default=False
        If set to `True`, the last test window is returned even if it is partial,
        otherwise it is ignored.

    params : dict, optional
        Parameters to pass to the underlying estimator's `partial_fit` through metadata
        routing.

    portfolio_params : dict, optional
        Additional parameters forwarded to the resulting
        :class:`~skfolio.portfolio.MultiPeriodPortfolio`.

    Returns
    -------
    prediction : MultiPeriodPortfolio
        A :class:`~skfolio.portfolio.MultiPeriodPortfolio` containing one
        :class:`~skfolio.portfolio.Portfolio` per test window, ordered chronologically.

    Raises
    ------
    TypeError
        If the estimator is not a portfolio optimization estimator, does not
        implement `partial_fit`, or is a pipeline.

    ValueError
        If `warmup_size < 1`, `test_size < 1`, or the data is too short for at least one
        test window.

    See Also
    --------
    :ref:`sphx_glr_auto_examples_online_learning_plot_3_online_portfolio_optimization_evaluation.py`
        Online evaluation of portfolio optimization using `online_predict`.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.model_selection import online_predict
    >>> from skfolio.moments import EWCovariance, EWMu
    >>> from skfolio.optimization import MeanRisk
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.prior import EmpiricalPrior
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>>
    >>> model = MeanRisk(
    ...     prior_estimator=EmpiricalPrior(
    ...         mu_estimator=EWMu(half_life=40),
    ...         covariance_estimator=EWCovariance(half_life=40),
    ...     ),
    ... )
    >>> pred = online_predict(model, X, warmup_size=252, test_size=5)
    """
    _validate_online_estimator(
        estimator, caller="online_predict", require_portfolio=True
    )
    _validate_sizes(warmup_size, test_size)

    estimator = sk.clone(estimator)
    X, y = sku.indexable(X, y)
    routed_params = _route_params(
        estimator, params, owner="online_predict", callee="partial_fit"
    )

    return _online_predict(
        estimator,
        X,
        y,
        routed_params,
        warmup_size=warmup_size,
        test_size=test_size,
        freq=freq,
        freq_offset=freq_offset,
        previous=previous,
        purged_size=purged_size,
        reduce_test=reduce_test,
        portfolio_params=portfolio_params,
    )


def online_score(
    estimator: skb.BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None = None,
    warmup_size: int = 252,
    test_size: int = 1,
    freq: str | pd.offsets.BaseOffset | None = None,
    freq_offset: pd.offsets.BaseOffset | dt.timedelta | None = None,
    previous: bool = False,
    purged_size: int = 0,
    reduce_test: bool = False,
    scoring: skt.Scoring = None,
    params: dict | None = None,
    per_step: bool = False,
    portfolio_params: dict | None = None,
) -> float | dict[str, float] | FloatArray | dict[str, FloatArray]:
    r"""Score an online estimator using walk-forward evaluation.

    Walks forward through the data, updating the estimator incrementally via
    `partial_fit` and scoring on each subsequent test window. This is the scoring
    counterpart of :func:`online_predict`.

    The function handles both *non-predictor estimators* (e.g. covariance, expected
    returns, prior) and *portfolio optimization* estimators:

    * **non-predictor estimators** are scored on each test window independently.
      By default the average of per-step scores is returned.
    * **Portfolio optimization estimators** are evaluated by collecting out-of-sample
      predictions into a :class:`~skfolio.portfolio.MultiPeriodPortfolio` and computing
      the requested measure on the full multi-period portfolio.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator instance to use to fit the data. It must implement `partial_fit`.
        Pipelines are not supported.

    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets. Must be a DataFrame with a `DatetimeIndex` when
        `freq` is provided.

    y : array-like of shape (n_observations, n_targets), optional
        Target data to pass to `partial_fit`.

    warmup_size : int, default=252
        Number of initial observations (or periods when `freq` is set) used for the
        first `partial_fit` call. No scores are produced during warmup.

    test_size : int, default=1
        Length of each test set.
        If `freq` is `None` (default), it represents the number of observations.
        Otherwise, it represents the number of periods defined by `freq`.

    freq : str | pandas.offsets.BaseOffset, optional
        If provided, it must be a frequency string or a pandas DateOffset, and `X` must
        be a DataFrame with an index of type `DatetimeIndex`. In that case,
        `warmup_size` and `test_size` represent the number of periods defined by `freq`
        instead of the number of observations.

    freq_offset : pandas.offsets.BaseOffset | datetime.timedelta, optional
        Only used if `freq` is provided. Offsets `freq` by a pandas DateOffset or a
        datetime timedelta offset.

    previous : bool, default=False
        Only used if `freq` is provided. If set to `True`, and if the period start or
        period end is not in the `DatetimeIndex`, the previous observation is used;
        otherwise, the next observation is used.

    purged_size : int, default=0
        The number of observations to exclude from the end of each training
        window before the test window.

    reduce_test : bool, default=False
        If set to `True`, the last test window is returned even if it is partial,
        otherwise it is ignored.

    scoring : callable, dict, BaseMeasure, or None
        Scoring specification. Semantics depend on the estimator type:

        * **Non-predictor estimators** (e.g. covariance, expected returns, prior):
          `None` uses `estimator.score`; otherwise pass a callable
          scorer(estimator, X_test)` or a dict of such callables.
        * **Portfolio optimization estimators**:
          a :class:`~skfolio.measures.BaseMeasure` or a dict of measures. `None`
          defaults to :attr:`~skfolio.measures.RatioMeasure.SHARPE_RATIO`.

        .. note ::
            For portfolio optimization estimators, online evaluation scores the
            aggregated out-of-sample :class:`~skfolio.portfolio.MultiPeriodPortfolio`,
            rather than scoring each test window independently and averaging as in
            :class:`~sklearn.model_selection.GridSearchCV`. Pass the measure enum
            directly; `make_scorer` is not supported.

    params : dict, optional
        Parameters to pass to the underlying estimator's `partial_fit`
        through metadata routing.

    per_step : bool, default=False
        If `True`, return per-step score arrays instead of aggregated
        scalars. Only supported for non-predictor estimators; raises
        `ValueError` for portfolio optimization estimators.

    portfolio_params : dict, optional
        Additional parameters forwarded to the resulting
        :class:`~skfolio.portfolio.MultiPeriodPortfolio` when scoring a
        portfolio optimization estimator.

    Returns
    -------
    score : float | dict[str, float] | ndarray | dict[str, ndarray]
        By default, an aggregate `float` (or `dict` for multi-metric).
        When `per_step=True`, a `FloatArray` of per-step scores (or
        `dict` thereof).

    Raises
    ------
    TypeError
        If the estimator does not implement `partial_fit` or is a pipeline.

    ValueError
        If `per_step=True` is used with a portfolio optimization estimator,
        or if `warmup_size < 1`, `test_size < 1`, or the data is too
        short for at least one test window.

    See Also
    --------
    :ref:`sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py`
        Programmatic comparison of covariance estimators with `online_score`.
    :ref:`sphx_glr_auto_examples_online_learning_plot_3_online_portfolio_optimization_evaluation.py`
        Portfolio-level evaluation with `online_score`.

    Examples
    --------
    non-predictor estimator (default `estimator.score`):

    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.model_selection import online_score
    >>> from skfolio.moments import EWCovariance
    >>> from skfolio.preprocessing import prices_to_returns
    >>>
    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)
    >>> score = online_score(EWCovariance(), X, warmup_size=252)

    Portfolio optimization estimator:

    >>> from skfolio.measures import RatioMeasure
    >>> from skfolio.moments import EWMu
    >>> from skfolio.optimization import MeanRisk
    >>> from skfolio.prior import EmpiricalPrior
    >>>
    >>> model = MeanRisk(
    ...     prior_estimator=EmpiricalPrior(
    ...         mu_estimator=EWMu(half_life=40),
    ...         covariance_estimator=EWCovariance(half_life=40),
    ...     ),
    ... )
    >>> score = online_score(  # doctest: +SKIP
    ...     model,
    ...     X,
    ...     warmup_size=252,
    ...     test_size=5,
    ...     scoring=RatioMeasure.SHARPE_RATIO,
    ... )
    """
    _validate_online_estimator(estimator, caller="online_score")
    _validate_sizes(warmup_size, test_size)

    estimator = sk.clone(estimator)
    X, y = sku.indexable(X, y)
    routed_params = _route_params(
        estimator, params, owner="online_score", callee="partial_fit"
    )

    is_portfolio = _is_portfolio_optimization_estimator(estimator)
    _validate_scoring(scoring, is_portfolio)

    if per_step and is_portfolio:
        raise ValueError(
            "per_step=True is not supported for portfolio optimization "
            "estimators. Use online_predict to obtain the "
            "MultiPeriodPortfolio and compute measures on it directly."
        )

    if per_step:
        return _online_score(
            estimator,
            X,
            y,
            scoring,
            routed_params,
            warmup_size=warmup_size,
            test_size=test_size,
            freq=freq,
            freq_offset=freq_offset,
            previous=previous,
            purged_size=purged_size,
            reduce_test=reduce_test,
        )

    agg, _ = _evaluate_online(
        estimator,
        X,
        y,
        scoring=scoring,
        routed_params=routed_params,
        warmup_size=warmup_size,
        test_size=test_size,
        freq=freq,
        freq_offset=freq_offset,
        previous=previous,
        purged_size=purged_size,
        reduce_test=reduce_test,
        portfolio_params=portfolio_params,
    )
    return agg


def _online_walk_forward(
    estimator: skb.BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None,
    warmup_size: int,
    test_size: int,
    routed_params: sku.Bunch,
    freq: str | pd.offsets.BaseOffset | None = None,
    freq_offset: pd.offsets.BaseOffset | dt.timedelta | None = None,
    previous: bool = False,
    purged_size: int = 0,
    reduce_test: bool = False,
    refit_last: bool = False,
) -> Generator[slice, None, None]:
    """Walk-forward generator shared by :func:`online_predict` and
    :func:`online_score`.

    Yields `test_slice` at each step after warming up and incrementally
    updating the estimator. The caller owns the estimator reference and may
    call `predict`, `score`, or `set_params` between yields.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator instance used to fit the data. Must support `partial_fit`.

    X : array-like of shape (n_observations, n_assets)
        The data to fit, already validated via `sku.indexable`.

    y : array-like or None
        Target data to pass to `partial_fit`.

    warmup_size : int
        Size of the first training window. The estimator is initialized via
        `partial_fit` on this window (not `fit`), so the caller is responsible
        for any cold-start configuration the estimator may require.

    test_size : int
        Size of each test window.

    routed_params : Bunch
        Parameters passed to the underlying estimator's `partial_fit`.

    freq : str or pandas DateOffset, optional
        Calendar frequency forwarded to
        :class:`~skfolio.model_selection.WalkForward`.

    freq_offset : pandas DateOffset or datetime timedelta, optional
        Optional offset applied to the walk-forward schedule.

    previous : bool, default=False
        Alignment rule used for calendar-based schedules.

    purged_size : int, default=0
        Number of observations purged between train and test windows.

    reduce_test : bool, default=False
        Whether to keep the final partial test window.

    refit_last : bool, default=False
        If `True`, perform a final `partial_fit` on the last test window
        after yielding it. Used by :class:`OnlineGridSearch` so that the
        best estimator is fully trained on all available data.

    Yields
    ------
    test_slice : slice
        Slice into `X` for the current test window.
    """
    cv = WalkForward(
        test_size=test_size,
        train_size=warmup_size,
        freq=freq,
        freq_offset=freq_offset,
        previous=previous,
        purged_size=purged_size,
        expand_train=True,
        reduce_test=reduce_test,
    )

    splits = list(cv.split(X))

    if len(splits) == 0:
        raise ValueError(
            f"Not enough observations for at least one test window with "
            f"warmup_size={warmup_size}, test_size={test_size}, "
            f"purged_size={purged_size}."
        )

    # WalkForward with expand_train=True produces contiguous index arrays,
    # so converting first/last to a slice is safe and avoids a copy.
    initial_train = splits[0][0]
    warmup_slice = slice(int(initial_train[0]), int(initial_train[-1]) + 1)
    fit_single_estimator(
        estimator,
        X,
        y,
        fit_params=routed_params.estimator_params,
        indices=warmup_slice,
        method="partial_fit",
    )

    last_train_end = warmup_slice.stop
    last_test_slice = None

    for train_idx, test_idx in splits:
        train_end = int(train_idx[-1]) + 1

        if train_end > last_train_end:
            fit_single_estimator(
                estimator,
                X,
                y,
                fit_params=routed_params.estimator_params,
                indices=slice(last_train_end, train_end),
                method="partial_fit",
            )
            last_train_end = train_end

        last_test_slice = slice(int(test_idx[0]), int(test_idx[-1]) + 1)
        yield last_test_slice

    if refit_last and last_test_slice is not None:
        if last_test_slice.stop > last_train_end:
            fit_single_estimator(
                estimator,
                X,
                y,
                fit_params=routed_params.estimator_params,
                indices=slice(last_train_end, last_test_slice.stop),
                method="partial_fit",
            )


def _online_predict(
    estimator: skb.BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None,
    routed_params: sku.Bunch,
    *,
    warmup_size: int,
    test_size: int,
    freq=None,
    freq_offset=None,
    previous: bool = False,
    purged_size: int = 0,
    reduce_test: bool = False,
    refit_last: bool = False,
    portfolio_params: dict | None = None,
) -> MultiPeriodPortfolio:
    """Online prediction.

    Operates on an already-cloned, validated estimator. Public callers
    should use :func:`online_predict` instead. Expects routed `partial_fit` parameters
    and supports the internal `refit_last` option used by the online search utilities.

    Returns
    -------
    multi_period_portfolio : MultiPeriodPortfolio
        Predicted portfolios aggregated across test windows.
    """
    portfolio_params = {} if portfolio_params is None else portfolio_params.copy()
    last_step = _get_last_step(estimator)
    needs_prev_weights = getattr(last_step, "needs_previous_weights", False)
    use_dict = _asset_names_enabled(X)
    prev_weights = last_step.previous_weights if needs_prev_weights else None

    portfolios = []
    for test_slice in _online_walk_forward(
        estimator,
        X,
        y,
        warmup_size,
        test_size,
        routed_params,
        freq=freq,
        freq_offset=freq_offset,
        previous=previous,
        purged_size=purged_size,
        reduce_test=reduce_test,
        refit_last=refit_last,
    ):
        if needs_prev_weights:
            last_step.set_params(previous_weights=prev_weights)

        portfolio = estimator.predict(X[test_slice])
        portfolios.append(portfolio)

        if needs_prev_weights:
            prev_weights = portfolio.weights_dict if use_dict else portfolio.weights

    return MultiPeriodPortfolio(portfolios=portfolios, **portfolio_params)


def _online_score(
    estimator: skb.BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None,
    scoring,
    routed_params: sku.Bunch,
    *,
    warmup_size: int,
    test_size: int,
    freq=None,
    freq_offset=None,
    previous: bool = False,
    purged_size: int = 0,
    reduce_test: bool = False,
    refit_last: bool = False,
) -> FloatArray | dict[str, FloatArray]:
    """Per-step scoring for non-predictor estimators.

    Returns per-step score arrays. Operates on an already-cloned, validated estimator.
    Expects routed `partial_fit` parameters and applies the provided scorer to each test
    window yielded by :func:`_online_walk_forward`.

    Returns
    -------
    scores : ndarray or dict[str, ndarray]
        Per-step score arrays.
    """
    multi_scoring = isinstance(scoring, dict)

    if multi_scoring:
        scores = {name: [] for name in scoring}
    else:
        scores = []

    for test_slice in _online_walk_forward(
        estimator,
        X,
        y,
        warmup_size,
        test_size,
        routed_params,
        freq=freq,
        freq_offset=freq_offset,
        previous=previous,
        purged_size=purged_size,
        reduce_test=reduce_test,
        refit_last=refit_last,
    ):
        X_test = X[test_slice]
        if multi_scoring:
            for name, score_func in scoring.items():
                scores[name].append(score_func(estimator, X_test))
        elif scoring is not None:
            scores.append(scoring(estimator, X_test))
        else:
            scores.append(estimator.score(X_test))

    if multi_scoring:
        return {name: np.array(vals) for name, vals in scores.items()}
    return np.array(scores)


def _evaluate_online(
    estimator: skb.BaseEstimator,
    X: ArrayLike,
    y: ArrayLike | None,
    *,
    scoring,
    routed_params: sku.Bunch,
    warmup_size: int,
    test_size: int,
    freq=None,
    freq_offset=None,
    previous: bool = False,
    purged_size: int = 0,
    reduce_test: bool = False,
    refit_last: bool = False,
    portfolio_params: dict | None = None,
) -> tuple[float | dict[str, float], MultiPeriodPortfolio | None]:
    """Unified online evaluation dispatcher.

    For portfolio estimators the score is derived from the aggregated
    :class:`~skfolio.portfolio.MultiPeriodPortfolio`. For non-predictor estimators the
    score is the average of per-step values. Expects routed `partial_fit` parameters and
    forwards `refit_last` and `portfolio_params` to the internal evaluation path when
    needed.

    Returns
    -------
    aggregate_score : float or dict[str, float]
        Aggregated score over all test windows.

    multi_period_portfolio : MultiPeriodPortfolio or None
        Multi-Period Portfolio for portfolio estimators, otherwise `None`.
    """
    is_portfolio = _is_portfolio_optimization_estimator(estimator)
    multi_scoring = isinstance(scoring, dict)

    if is_portfolio:
        multi_period_portfolio = _online_predict(
            estimator,
            X,
            y,
            routed_params,
            warmup_size=warmup_size,
            test_size=test_size,
            freq=freq,
            freq_offset=freq_offset,
            previous=previous,
            purged_size=purged_size,
            reduce_test=reduce_test,
            refit_last=refit_last,
            portfolio_params=portfolio_params,
        )
        if multi_scoring:
            agg = {
                name: _score_multi_period_portfolio(
                    multi_period_portfolio, single_scoring
                )
                for name, single_scoring in scoring.items()
            }
        else:
            agg = _score_multi_period_portfolio(multi_period_portfolio, scoring)
        return agg, multi_period_portfolio

    per_step = _online_score(
        estimator,
        X,
        y,
        scoring,
        routed_params,
        warmup_size=warmup_size,
        test_size=test_size,
        freq=freq,
        freq_offset=freq_offset,
        previous=previous,
        purged_size=purged_size,
        reduce_test=reduce_test,
        refit_last=refit_last,
    )
    if multi_scoring:
        agg = {name: float(np.mean(vals)) for name, vals in per_step.items()}
    else:
        agg = float(np.mean(per_step))
    return agg, None


def _score_multi_period_portfolio(
    multi_period_portfolio: MultiPeriodPortfolio,
    scoring: BaseMeasure | None,
) -> float:
    """Score a :class:`MultiPeriodPortfolio` using a measure.

    Risk measures are negated so that higher is always better.

    Parameters
    ----------
    multi_period_portfolio : MultiPeriodPortfolio
        The multi-period portfolio to score.

    scoring : BaseMeasure or None
        The measure to evaluate. `None` defaults to
        :attr:`~skfolio.measures.RatioMeasure.SHARPE_RATIO`.

    Returns
    -------
    score : float
    """
    if scoring is None:
        scoring = RatioMeasure.SHARPE_RATIO

    value = multi_period_portfolio.get_measure(scoring)
    if scoring.is_risk:
        value = -value
    return float(value)


def _validate_online_estimator(
    estimator: skb.BaseEstimator,
    *,
    caller: str,
    require_portfolio: bool = False,
) -> skb.BaseEstimator:
    """Validate that an estimator is compatible with online evaluation.

    Pipelines are rejected explicitly because online evaluation updates the
    estimator in place with repeated `partial_fit` calls, while
    :class:`~sklearn.pipeline.Pipeline` does not yet expose a stable incremental
    interface.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator instance to validate.

    caller : str
        Public API function performing the validation. Used in error messages.

    require_portfolio : bool, default=False
        If `True`, require `estimator` to be a portfolio optimization estimator.
        If `False`, any non-pipeline estimator implementing `partial_fit` is accepted.

    Returns
    -------
    estimator : BaseEstimator
        The validated estimator.
    """
    if isinstance(estimator, Pipeline):
        raise TypeError(f"Pipeline is not supported for `{caller}`.")

    if require_portfolio and not _is_portfolio_optimization_estimator(estimator):
        raise TypeError(
            f"skfolio's `{caller}` only supports portfolio optimization estimators."
        )

    if not hasattr(estimator, "partial_fit"):
        raise TypeError(
            f"The estimator ({type(estimator).__name__}) does not "
            "implement partial_fit. Use an estimator with incremental learning "
            "support (e.g., `EmpiricalPrior` using exponentially weighted moments)."
        )
    return estimator


def _validate_sizes(warmup_size: int, test_size: int) -> None:
    """Validate `warmup_size` and `test_size`.

    Parameters
    ----------
    warmup_size : int
        Number of observations in the initial training window.

    test_size : int
        Number of observations in each test window.
    """
    if isinstance(warmup_size, bool) or not isinstance(warmup_size, numbers.Integral):
        raise TypeError(
            f"warmup_size must be an integer, got {type(warmup_size).__name__}."
        )
    if isinstance(test_size, bool) or not isinstance(test_size, numbers.Integral):
        raise TypeError(
            f"test_size must be an integer, got {type(test_size).__name__}."
        )
    if warmup_size < 1:
        raise ValueError(f"warmup_size must be >= 1, got {warmup_size}.")
    if test_size < 1:
        raise ValueError(f"test_size must be >= 1, got {test_size}.")


def _validate_scoring(scoring: skt.Scoring, is_portfolio: bool) -> None:
    """Validate that the scoring specification matches the estimator type.

    Parameters
    ----------
    scoring : callable, dict, BaseMeasure, or None
        Scoring specification.

    is_portfolio : bool
        Whether the estimator is a portfolio optimization estimator.
    """
    values = scoring.values() if isinstance(scoring, dict) else [scoring]
    for s in values:
        if s is None:
            continue
        if is_portfolio:
            if isinstance(s, _BaseScorer):
                raise TypeError(
                    f"Got {s!r} as scoring, but make_scorer is not supported "
                    "for online portfolio evaluation. Pass the measure "
                    "directly (e.g. scoring=RatioMeasure.SHARPE_RATIO). "
                    "make_scorer is designed for sklearn's GridSearchCV / "
                    "cross_val_score which use a per-fold predict/score "
                    "cycle. Online evaluation scores the full aggregated "
                    "MultiPeriodPortfolio instead."
                )
            if not isinstance(s, BaseMeasure):
                raise TypeError(
                    "For portfolio optimization estimators, `scoring` must be "
                    "`None`, a `BaseMeasure`, or a dict[str, BaseMeasure]."
                )
        else:
            if isinstance(s, BaseMeasure):
                raise TypeError(
                    f"Got {s!r} as scoring, but BaseMeasure scoring is only "
                    "supported for portfolio optimization estimators. For "
                    "non-predictor estimators, pass a callable scorer (e.g. "
                    "make_scorer(my_loss, response_method=None)) or use "
                    "estimator.score (scoring=None)."
                )
            if isinstance(s, _BaseScorer) and not isinstance(s, _EstimatorScorer):
                raise TypeError(
                    f"Got {s!r} as scoring, but portfolio scorers created with "
                    "`make_scorer(..., response_method='predict')` are not "
                    "supported for non-predictor estimators. Use "
                    "`make_scorer(..., response_method=None)` instead."
                )
            if not callable(s):
                raise TypeError(
                    "For non-predictor estimators, `scoring` must be `None`, a "
                    "callable, or a dict[str, callable]."
                )
