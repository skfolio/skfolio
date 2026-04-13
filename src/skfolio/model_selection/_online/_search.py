"""Online hyperparameter search module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import datetime as dt
import numbers
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn as sk
import sklearn.base as skb
import sklearn.utils as sku
import sklearn.utils.validation as skv
from scipy.stats import rankdata
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.utils.parallel import Parallel, delayed

import skfolio.typing as skt
from skfolio.model_selection._online._validation import (
    _evaluate_online,
    _route_params,
    _validate_online_estimator,
    _validate_scoring,
    _validate_sizes,
)
from skfolio.model_selection._validation import _is_portfolio_optimization_estimator

__all__ = ["OnlineGridSearch", "OnlineRandomizedSearch"]


class BaseOnlineSearch(skb.MetaEstimatorMixin, skb.BaseEstimator, ABC):
    """Abstract base class for online hyperparameter search.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator that supports `partial_fit`.

    scoring : callable, dict, BaseMeasure, or None
        Scoring specification. Semantics depend on the estimator type:

        * **Component estimators** (e.g. covariance, expected returns):
          `None` uses `estimator.score`; otherwise pass a callable
          `scorer(estimator, X_test)` or a dict of such callables.
        * **Portfolio optimization estimators**:
          a :class:`~skfolio.measures.BaseMeasure` or a dict of measures.
          `None` defaults to
          :attr:`~skfolio.measures.RatioMeasure.SHARPE_RATIO`.

        For portfolio optimization estimators, online evaluation scores the
        aggregated out-of-sample
        :class:`~skfolio.portfolio.MultiPeriodPortfolio`, rather than scoring
        each test window independently and averaging as in
        :class:`~sklearn.model_selection.GridSearchCV`. Pass the measure enum
        directly; `make_scorer` is not supported.

    warmup_size : int, default=252
        Number of initial observations (or periods when `freq` is set) used
        for the first `partial_fit` call.

    test_size : int, default=1
        Number of observations (or periods when `freq` is set) per test
        window.

    freq : str | pandas.offsets.BaseOffset, optional
        Rebalancing frequency. When provided, `warmup_size` and `test_size`
        are interpreted as period counts rather than observation counts, and
        `X` must be a DataFrame with a `DatetimeIndex`. See
        :class:`~skfolio.model_selection.WalkForward` for details and
        examples.

    freq_offset : pandas.offsets.BaseOffset | datetime.timedelta, optional
        Offset applied to the `freq` boundaries. Only used when `freq` is
        provided.

    previous : bool, default=False
        Only used when `freq` is provided. If `True`, period boundaries
        that fall between observations snap to the previous observation;
        otherwise they snap to the next.

    purged_size : int, default=0
        Number of observations (or periods) to skip between the last data the
        model sees and the start of the test window.

    reduce_test : bool, default=False
        If `True`, the last test window is included even when it contains
        fewer observations than `test_size`.

    refit : bool, str, or callable, default=True
        Controls how the best candidate is selected and whether the
        selected fitted candidate is exposed as `best_estimator_`.

        This parameter is named for API alignment with scikit-learn.
        Unlike scikit-learn search estimators, enabling `refit` does
        not trigger an additional fit after model selection because
        each candidate is already evaluated through a full online
        walk-forward pass and updated through the full sample.

        * Single-metric scoring: `True` or `False` are both supported.
          If `False`, `best_estimator_` is not stored, but
          `best_index_`, `best_params_`, and `best_score_` remain
          available.
        * Multi-metric scoring: set to a scorer name to select the best
          candidate for that metric, or to `False` to disable
          best-candidate selection and storage of `best_estimator_`.
        * A callable receives `cv_results_` and must return the best
          candidate index.

    n_jobs : int or None, default=None
        Number of parallel jobs. `None` means 1.

    verbose : int, default=0
        Verbosity level for `joblib.Parallel`.

    error_score : "raise" or float, default=np.nan
        Value to assign to the score if an error occurs during fitting.
        If set to `"raise"`, the error is raised.

    portfolio_params : dict, optional
        Parameters forwarded to
        :class:`~skfolio.portfolio.MultiPeriodPortfolio` when scoring
        portfolio estimators.

    return_predictions : bool, default=False
        If `True`, store
        :class:`~skfolio.portfolio.MultiPeriodPortfolio` objects per
        candidate in `cv_results_["predictions"]`. Only applies to
        portfolio optimization estimators.

    Attributes
    ----------
    cv_results_ : dict[str, ndarray]
        A dict with keys:

        * `params`: list of candidate parameter dicts.
        * `mean_score`: array of aggregate scores (or
          `mean_score_<name>` for multi-metric).
        * `rank`: array of ranks where 1 is best (or `rank_<name>`
          for multi-metric).
        * `fit_time`: array of wall-clock times.
        * `predictions`: object array of `MultiPeriodPortfolio` or `None`
          aligned with candidates (only when `return_predictions=True` and
          the estimator is portfolio-based).

    best_estimator_ : BaseEstimator
        Estimator fitted on the full data with the best parameters.
        Only available when `refit` is not `False`.

    best_index_ : int
        Index into `cv_results_` of the best candidate. Available for
        single-metric scoring and for multi-metric scoring when `refit` is
        not `False`.

    best_score_ : float
        Aggregate score of the selected best candidate. Available when
        `best_index_` is defined and `refit` is not callable.

    best_params_ : dict
        Parameter setting that gave the selected best score. Available when
        `best_index_` is defined.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    is_portfolio_estimator_ : bool
        Whether or not the estimator is a portfolio estimator (from BaseOptimization).
    """

    cv_results_: dict[str, np.ndarray]
    best_estimator_: skb.BaseEstimator
    best_score_: float
    best_params_: dict
    best_index_: int
    multimetric_: bool
    is_portfolio_estimator_: bool

    def __init__(
        self,
        estimator: skb.BaseEstimator,
        *,
        scoring=None,
        warmup_size: int = 252,
        test_size: int = 1,
        freq: str | pd.offsets.BaseOffset | None = None,
        freq_offset: pd.offsets.BaseOffset | dt.timedelta | None = None,
        previous: bool = False,
        purged_size: int = 0,
        reduce_test: bool = False,
        refit: bool | str | Callable[[dict[str, Any]], int] = True,
        error_score=np.nan,
        return_predictions: bool = False,
        portfolio_params: dict | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.scoring = scoring
        self.warmup_size = warmup_size
        self.test_size = test_size
        self.freq = freq
        self.freq_offset = freq_offset
        self.previous = previous
        self.purged_size = purged_size
        self.reduce_test = reduce_test
        self.refit = refit
        self.error_score = error_score
        self.return_predictions = return_predictions
        self.portfolio_params = portfolio_params
        self.n_jobs = n_jobs
        self.verbose = verbose

    @abstractmethod
    def _get_candidate_params(self) -> Iterable[dict]:
        """Return the parameter dicts to evaluate."""

    def _get_refit_metric_name(self) -> str | None:
        """Return the metric name used to select the best candidate."""
        if not self.multimetric_:
            return "score"
        _check_refit_for_multimetric(self.refit, self.scoring)
        if isinstance(self.refit, str):
            return self.refit
        return None

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params,
    ):
        """Run the online search over all candidate parameter combinations.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns.

        y : array-like, optional
            Optional Target.

        **fit_params
            Additional parameters routed via metadata routing.

        Returns
        -------
        self
        """
        _validate_online_estimator(
            self.estimator,
            caller=f"{type(self).__name__}.fit",
        )
        _validate_sizes(self.warmup_size, self.test_size)
        _validate_error_score(self.error_score)

        X, y = sku.indexable(X, y)
        self.is_portfolio_estimator_ = _is_portfolio_optimization_estimator(
            self.estimator
        )
        self.multimetric_ = isinstance(self.scoring, dict)
        _validate_scoring(self.scoring, self.is_portfolio_estimator_)

        routed_params = _route_params(
            self.estimator,
            fit_params,
            owner=f"{type(self).__name__}.fit",
            callee="partial_fit",
        )

        candidate_params_list = list(self._get_candidate_params())

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_evaluate_candidate)(
                self.estimator,
                candidate_params,
                X,
                y,
                scoring=self.scoring,
                routed_params=routed_params,
                warmup_size=self.warmup_size,
                test_size=self.test_size,
                freq=self.freq,
                freq_offset=self.freq_offset,
                previous=self.previous,
                purged_size=self.purged_size,
                reduce_test=self.reduce_test,
                return_predictions=self.return_predictions,
                error_score=self.error_score,
                portfolio_params=self.portfolio_params,
            )
            for candidate_params in candidate_params_list
        )

        self._store_results(results)
        return self

    def _store_results(self, results: list[dict]) -> None:
        """Populate `cv_results_` and best-candidate attributes."""
        all_params = [res["params"] for res in results]
        fit_times = np.asarray([res["fit_time"] for res in results], dtype=np.float64)

        cv_results: dict[str, Any] = {
            "params": all_params,
            "fit_time": fit_times,
        }

        if self.multimetric_:
            for name in self.scoring:
                scores = np.asarray(
                    [res["score"][name] for res in results],
                    dtype=np.float64,
                )
                cv_results[f"mean_score_{name}"] = scores
                cv_results[f"rank_{name}"] = _rank_scores(scores)
        else:
            scores = np.asarray([res["score"] for res in results], dtype=np.float64)
            cv_results["mean_score"] = scores
            cv_results["rank"] = _rank_scores(scores)

        _raise_if_all_candidates_failed(results)

        if self.return_predictions and self.is_portfolio_estimator_:
            predictions = np.empty(len(results), dtype=object)
            for i, res in enumerate(results):
                predictions[i] = res["prediction"]
            cv_results["predictions"] = predictions

        self.cv_results_ = cv_results

        if self.refit or not self.multimetric_:
            refit_metric = self._get_refit_metric_name()
            selection_metric = refit_metric or "score"
            self.best_index_ = _select_best_index(
                self.refit,
                selection_metric,
                cv_results,
            )
            self.best_params_ = all_params[self.best_index_]

            if not callable(self.refit):
                score_key = (
                    "mean_score"
                    if selection_metric == "score"
                    else f"mean_score_{selection_metric}"
                )
                self.best_score_ = float(cv_results[score_key][self.best_index_])

        if self.refit:
            best_candidate = results[self.best_index_]["estimator"]
            if best_candidate is None:
                warnings.warn(
                    "All parameter candidates failed during evaluation. "
                    "`best_estimator_` is not available.",
                    category=UserWarning,
                    stacklevel=2,
                )
            else:
                self.best_estimator_ = best_candidate

    def predict(self, X: npt.ArrayLike):
        """Predict using the best estimator found during search.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns.

        Returns
        -------
        prediction : Portfolio | Population
        """
        skv.check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.predict(X)

    def score(self, X: npt.ArrayLike, y=None):
        """Score using the best estimator found during search.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns.

        y : ignored

        Returns
        -------
        score : float
        """
        skv.check_is_fitted(self, "best_estimator_")
        return self.best_estimator_.score(X)


class OnlineGridSearch(BaseOnlineSearch):
    """Online exhaustive hyperparameter search over a parameter grid.

    Each parameter combination is evaluated by running a full online
    walk-forward pass. The best estimator is selected based on the
    aggregate out-of-sample score.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator that supports `partial_fit`.

    param_grid : dict or list[dict]
        Dictionary with parameters names (`str`) as keys and lists of parameter
        settings to try as values, or a list of such dictionaries, in which case the
        grids spanned by each dictionary in the list are explored. This enables
        searching over any sequence of parameter settings.

    scoring : callable, dict, BaseMeasure, or None
        Scoring specification. Semantics depend on the estimator type:

        * **Component estimators** (e.g. covariance, expected returns):
          `None` uses `estimator.score`; otherwise pass a callable
          `scorer(estimator, X_test)` or a dict of such callables.
        * **Portfolio optimization estimators**:
          a :class:`~skfolio.measures.BaseMeasure` or a dict of measures.
          `None` defaults to
          :attr:`~skfolio.measures.RatioMeasure.SHARPE_RATIO`.

        For portfolio optimization estimators, online evaluation scores the
        aggregated out-of-sample
        :class:`~skfolio.portfolio.MultiPeriodPortfolio`, rather than scoring
        each test window independently and averaging as in
        :class:`~sklearn.model_selection.GridSearchCV`. Pass the measure enum
        directly; `make_scorer` is not supported.

    warmup_size : int, default=252
        Number of initial observations (or periods when `freq` is set) used
        for the first `partial_fit` call.

    test_size : int, default=1
        Number of observations (or periods when `freq` is set) per test
        window.

    freq : str | pandas.offsets.BaseOffset, optional
        Rebalancing frequency. When provided, `warmup_size` and `test_size`
        are interpreted as period counts rather than observation counts, and
        `X` must be a DataFrame with a `DatetimeIndex`. See
        :class:`~skfolio.model_selection.WalkForward` for details and
        examples.

    freq_offset : pandas.offsets.BaseOffset | datetime.timedelta, optional
        Offset applied to the `freq` boundaries. Only used when `freq` is
        provided.

    previous : bool, default=False
        Only used when `freq` is provided. If `True`, period boundaries
        that fall between observations snap to the previous observation;
        otherwise they snap to the next.

    purged_size : int, default=0
        Number of observations (or periods) to skip between the last data the
        model sees and the start of the test window.

    reduce_test : bool, default=False
        If `True`, the last test window is included even when it contains
        fewer observations than `test_size`.

    refit : bool, str, or callable, default=True
        Controls how the best candidate is selected and whether the
        selected fitted candidate is exposed as `best_estimator_`.

        This parameter is named for API alignment with scikit-learn.
        Unlike scikit-learn search estimators, enabling `refit` does
        not trigger an additional fit after model selection because
        each candidate is already evaluated through a full online
        walk-forward pass and updated through the full sample.

        * Single-metric scoring: `True` or `False` are both supported.
          If `False`, `best_estimator_` is not stored, but
          `best_index_`, `best_params_`, and `best_score_` remain
          available.
        * Multi-metric scoring: set to a scorer name to select the best
          candidate for that metric, or to `False` to disable
          best-candidate selection and storage of `best_estimator_`.
        * A callable receives `cv_results_` and must return the best
          candidate index.

    error_score : "raise" or float, default=np.nan
        Value to assign to the score if an error occurs during fitting.
        If set to `"raise"`, the error is raised.

    return_predictions : bool, default=False
        If `True`, store
        :class:`~skfolio.portfolio.MultiPeriodPortfolio` objects per
        candidate in `cv_results_["predictions"]`. Only applies to
        portfolio optimization estimators.

    portfolio_params : dict, optional
        Parameters forwarded to
        :class:`~skfolio.portfolio.MultiPeriodPortfolio` when scoring
        portfolio estimators.

    n_jobs : int or None, default=None
        Number of parallel jobs. `None` means 1.

    verbose : int, default=0
        Verbosity level for `joblib.Parallel`.

    Attributes
    ----------
    cv_results_ : dict[str, ndarray]
        A dict with keys:

        * `params`: list of candidate parameter dicts.
        * `mean_score`: array of aggregate scores (or
          `mean_score_<name>` for multi-metric).
        * `rank`: array of ranks where 1 is best (or `rank_<name>`
          for multi-metric).
        * `fit_time`: array of wall-clock times.
        * `predictions`: object array of `MultiPeriodPortfolio` or `None`
          aligned with candidates (only when `return_predictions=True` and
          the estimator is portfolio-based).

    best_estimator_ : BaseEstimator
        Estimator fitted on the full data with the best parameters.
        Only available when `refit` is not `False`.

    best_score_ : float
        Aggregate score of the selected best candidate. Available when
        `best_index_` is defined and `refit` is not callable.

    best_params_ : dict
        Parameter setting that gave the selected best score. Available when
        `best_index_` is defined.

    best_index_ : int
        Index into `cv_results_` of the best candidate. Available for
        single-metric scoring and for multi-metric scoring when `refit` is
        not `False`.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    is_portfolio_estimator_ : bool
        Whether or not the estimator is a portfolio optimization estimator.

    See Also
    --------
    :ref:`sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py`
        Exhaustive online tuning of covariance estimator hyperparameters.
    :ref:`sphx_glr_auto_examples_online_learning_plot_3_online_portfolio_optimization_evaluation.py`
        Exhaustive online tuning of a `MeanRisk` estimator.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.model_selection import OnlineGridSearch
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
    ...         mu_estimator=EWMu(),
    ...         covariance_estimator=EWCovariance(),
    ...     ),
    ... )
    >>> search = OnlineGridSearch(  # doctest: +SKIP
    ...     model,
    ...     param_grid={
    ...         "prior_estimator__mu_estimator__half_life": [20, 40, 60],
    ...         "prior_estimator__covariance_estimator__half_life": [20, 40, 60],
    ...     },
    ...     warmup_size=252,
    ...     test_size=5,
    ...     n_jobs=-1,
    ... )
    >>> search.fit(X)  # doctest: +SKIP
    >>> search.best_params_  # doctest: +SKIP
    >>> search.best_estimator_  # doctest: +SKIP
    """

    def __init__(
        self,
        estimator: skb.BaseEstimator,
        param_grid: dict | list[dict],
        *,
        scoring=None,
        warmup_size: int = 252,
        test_size: int = 1,
        freq: str | pd.offsets.BaseOffset | None = None,
        freq_offset: pd.offsets.BaseOffset | dt.timedelta | None = None,
        previous: bool = False,
        purged_size: int = 0,
        reduce_test: bool = False,
        refit: bool | str | Callable[[dict[str, Any]], int] = True,
        error_score=np.nan,
        return_predictions: bool = False,
        portfolio_params: dict | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            warmup_size=warmup_size,
            test_size=test_size,
            freq=freq,
            freq_offset=freq_offset,
            previous=previous,
            purged_size=purged_size,
            reduce_test=reduce_test,
            refit=refit,
            error_score=error_score,
            return_predictions=return_predictions,
            portfolio_params=portfolio_params,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.param_grid = param_grid

    def _get_candidate_params(self) -> Iterable[dict]:
        return ParameterGrid(self.param_grid)


class OnlineRandomizedSearch(BaseOnlineSearch):
    """Online randomized search on hyper parameters.

    Each sampled parameter combination is evaluated by running a full online
    walk-forward pass. Unlike :class:`OnlineGridSearch`, not all parameter values are
    tried out but a fixed number of parameter settings is sampled from the specified
    distributions. The number of parameter settings that are tried is given by `n_iter`.

    If all parameters are presented as a list, sampling without replacement is
    performed. If at least one parameter is given as a distribution, sampling with
    replacement is used. It is highly recommended to use continuous distributions for
    continuous parameters.

    Parameters
    ----------
    estimator : BaseEstimator
        Estimator that supports `partial_fit`.

    param_distributions : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a `rvs`
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.

    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.

    scoring : callable, dict, BaseMeasure, or None
        Scoring specification. Semantics depend on the estimator type:

        * **Component estimators** (e.g. covariance, expected returns):
          `None` uses `estimator.score`; otherwise pass a callable
          `scorer(estimator, X_test)` or a dict of such callables.
        * **Portfolio optimization estimators**:
          a :class:`~skfolio.measures.BaseMeasure` or a dict of measures.
          `None` defaults to
          :attr:`~skfolio.measures.RatioMeasure.SHARPE_RATIO`.

        For portfolio optimization estimators, online evaluation scores the
        aggregated out-of-sample
        :class:`~skfolio.portfolio.MultiPeriodPortfolio`, rather than scoring
        each test window independently and averaging as in
        :class:`~sklearn.model_selection.GridSearchCV`. Pass the measure enum
        directly; `make_scorer` is not supported.

    warmup_size : int, default=252
        Number of initial observations (or periods when `freq` is set) used
        for the first `partial_fit` call.

    test_size : int, default=1
        Number of observations (or periods when `freq` is set) per test
        window.

    freq : str | pandas.offsets.BaseOffset, optional
        Rebalancing frequency. When provided, `warmup_size` and `test_size`
        are interpreted as period counts rather than observation counts, and
        `X` must be a DataFrame with a `DatetimeIndex`. See
        :class:`~skfolio.model_selection.WalkForward` for details and
        examples.

    freq_offset : pandas.offsets.BaseOffset | datetime.timedelta, optional
        Offset applied to the `freq` boundaries. Only used when `freq` is
        provided.

    previous : bool, default=False
        Only used when `freq` is provided. If `True`, period boundaries
        that fall between observations snap to the previous observation;
        otherwise they snap to the next.

    purged_size : int, default=0
        Number of observations (or periods) to skip between the last data the
        model sees and the start of the test window.

    reduce_test : bool, default=False
        If `True`, the last test window is included even when it contains
        fewer observations than `test_size`.

    refit : bool, str, or callable, default=True
        Controls how the best candidate is selected and whether the
        selected fitted candidate is exposed as `best_estimator_`.

        This parameter is named for API alignment with scikit-learn.
        Unlike scikit-learn search estimators, enabling `refit` does
        not trigger an additional fit after model selection because
        each candidate is already evaluated through a full online
        walk-forward pass and updated through the full sample.

        * Single-metric scoring: `True` or `False` are both supported.
          If `False`, `best_estimator_` is not stored, but
          `best_index_`, `best_params_`, and `best_score_` remain
          available.
        * Multi-metric scoring: set to a scorer name to select the best
          candidate for that metric, or to `False` to disable
          best-candidate selection and storage of `best_estimator_`.
        * A callable receives `cv_results_` and must return the best
          candidate index.

    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple function calls.

    error_score : "raise" or float, default=np.nan
        Value to assign to the score if an error occurs during fitting.
        If set to `"raise"`, the error is raised.

    return_predictions : bool, default=False
        If `True`, store
        :class:`~skfolio.portfolio.MultiPeriodPortfolio` objects per
        candidate in `cv_results_["predictions"]`. Only applies to
        portfolio optimization estimators.

    portfolio_params : dict, optional
        Parameters forwarded to
        :class:`~skfolio.portfolio.MultiPeriodPortfolio` when scoring
        portfolio estimators.

    n_jobs : int or None, default=None
        Number of parallel jobs. `None` means 1.

    verbose : int, default=0
        Verbosity level for `joblib.Parallel`.

    Attributes
    ----------
    cv_results_ : dict[str, ndarray]
        A dict with keys:

        * `params`: list of candidate parameter dicts.
        * `mean_score`: array of aggregate scores (or
          `mean_score_<name>` for multi-metric).
        * `rank`: array of ranks where 1 is best (or `rank_<name>`
          for multi-metric).
        * `fit_time`: array of wall-clock times.
        * `predictions`: object array of `MultiPeriodPortfolio` or `None`
          aligned with candidates (only when `return_predictions=True` and
          the estimator is portfolio-based).

    best_estimator_ : BaseEstimator
        Estimator fitted on the full data with the best parameters.
        Only available when `refit` is not `False`.

    best_score_ : float
        Aggregate score of the selected best candidate. Available when
        `best_index_` is defined and `refit` is not callable.

    best_params_ : dict
        Parameter setting that gave the selected best score. Available when
        `best_index_` is defined.

    best_index_ : int
        Index into `cv_results_` of the best candidate. Available for
        single-metric scoring and for multi-metric scoring when `refit` is
        not `False`.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    is_portfolio_estimator_ : bool
        Whether or not the estimator is a portfolio optimization estimator.

    See Also
    --------
    :ref:`sphx_glr_auto_examples_online_learning_plot_2_online_hyperparameter_tuning.py`
        Randomized online tuning of covariance estimator hyperparameters.

    Examples
    --------
    >>> from scipy.stats import uniform
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.model_selection import OnlineRandomizedSearch
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
    ...         mu_estimator=EWMu(),
    ...         covariance_estimator=EWCovariance(),
    ...     ),
    ... )
    >>> search = OnlineRandomizedSearch(  # doctest: +SKIP
    ...     model,
    ...     param_distributions={
    ...         "prior_estimator__mu_estimator__half_life": uniform(10, 90),
    ...         "prior_estimator__covariance_estimator__half_life": uniform(10, 90),
    ...     },
    ...     n_iter=20,
    ...     warmup_size=252,
    ...     test_size=5,
    ...     n_jobs=-1,
    ...     random_state=42,
    ... )
    >>> search.fit(X)  # doctest: +SKIP
    >>> search.best_params_  # doctest: +SKIP
    >>> search.best_estimator_  # doctest: +SKIP
    """

    def __init__(
        self,
        estimator: skb.BaseEstimator,
        param_distributions: dict,
        *,
        n_iter: int = 10,
        scoring=None,
        warmup_size: int = 252,
        test_size: int = 1,
        freq: str | pd.offsets.BaseOffset | None = None,
        freq_offset: pd.offsets.BaseOffset | dt.timedelta | None = None,
        previous: bool = False,
        purged_size: int = 0,
        reduce_test: bool = False,
        refit: bool | str | Callable[[dict[str, Any]], int] = True,
        random_state: int | None = None,
        error_score=np.nan,
        return_predictions: bool = False,
        portfolio_params: dict | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            estimator=estimator,
            scoring=scoring,
            warmup_size=warmup_size,
            test_size=test_size,
            freq=freq,
            freq_offset=freq_offset,
            previous=previous,
            purged_size=purged_size,
            reduce_test=reduce_test,
            refit=refit,
            error_score=error_score,
            return_predictions=return_predictions,
            portfolio_params=portfolio_params,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_candidate_params(self) -> Iterable[dict]:
        return ParameterSampler(
            self.param_distributions,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )


def _evaluate_candidate(
    estimator: skb.BaseEstimator,
    candidate_params: dict[str, Any],
    X: npt.ArrayLike,
    y: npt.ArrayLike | None,
    *,
    scoring: skt.Scoring,
    routed_params: sku.Bunch,
    warmup_size: int,
    test_size: int,
    freq: str | pd.offsets.BaseOffset | None,
    freq_offset: pd.offsets.BaseOffset | dt.timedelta | None,
    previous: bool,
    purged_size: int,
    reduce_test: bool,
    return_predictions: bool,
    portfolio_params: dict | None,
    error_score: float | Literal["raise"],
) -> dict[str, Any]:
    """Evaluate a single parameter combination.

    Clones the estimator, sets the candidate parameters, runs the full
    online walk-forward, and returns a result dict.
    """
    candidate = sk.clone(estimator)
    candidate.set_params(**candidate_params)

    result: dict[str, Any] = {"params": candidate_params}
    start = time.perf_counter()
    try:
        agg_score, multi_period_portfolio = _evaluate_online(
            candidate,
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
            refit_last=True,
            portfolio_params=portfolio_params,
        )
    except Exception as e:
        if error_score != "raise":
            warnings.warn(
                f"Estimator fit failed. The score will be set to {error_score}. "
                f"Details: \n{e!r}",
                category=UserWarning,
                stacklevel=2,
            )
            agg_score = _make_error_score(scoring, error_score)
            multi_period_portfolio = None
            candidate = None
        else:
            raise
    elapsed = time.perf_counter() - start

    result["score"] = agg_score
    result["fit_time"] = elapsed
    result["estimator"] = candidate
    result["prediction"] = multi_period_portfolio if return_predictions else None
    return result


def _rank_scores(scores: np.ndarray) -> np.ndarray:
    """Rank scores in descending order (1 = best / highest).

    NaN values are assigned the worst rank (tied last).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return np.array([], dtype=np.int32)
    if np.isnan(scores).all():
        return np.ones_like(scores, dtype=np.int32)
    min_score = np.nanmin(scores) - 1
    scores = np.nan_to_num(scores, nan=min_score)
    return rankdata(-scores, method="min").astype(np.int32, copy=False)


def _check_refit_for_multimetric(
    refit: bool | str | Callable[[dict[str, Any]], int],
    scores: dict[str, Any],
) -> None:
    """Validate `refit` for multi-metric scoring."""
    if refit is False:
        return
    if isinstance(refit, str) and refit in scores:
        return
    if callable(refit):
        return
    raise ValueError(
        "For multi-metric scoring, the parameter refit must be set to a "
        "scorer key or a callable to refit an estimator with the best "
        "parameter setting on the whole data and make the best_* "
        f"attributes available for that metric. If this is not needed, refit "
        f"should be set to False explicitly. {refit!r} was passed."
    )


def _select_best_index(
    refit: bool | str | Callable[[dict[str, Any]], int],
    refit_metric: str,
    cv_results: dict[str, Any],
) -> int:
    """Select the best candidate index from `cv_results_`."""
    if callable(refit):
        best_index = refit(cv_results)
        if not isinstance(best_index, numbers.Integral):
            raise TypeError("best_index_ returned is not an integer")
        if best_index < 0 or best_index >= len(cv_results["params"]):
            raise IndexError("best_index_ index out of range")
        return int(best_index)
    rank_key = "rank" if refit_metric == "score" else f"rank_{refit_metric}"
    return int(cv_results[rank_key].argmin())


def _make_error_score(
    scoring: skt.Scoring, error_score: float
) -> float | dict[str, float]:
    """Build an error score matching the scoring output shape."""
    error_score = float(error_score)
    if isinstance(scoring, dict):
        return {name: error_score for name in scoring}
    return error_score


def _raise_if_all_candidates_failed(results: list[dict[str, Any]]) -> None:
    """Raise when every candidate failed during evaluation."""
    if not results or any(result["estimator"] is not None for result in results):
        return
    raise ValueError(
        "All parameter candidates failed during evaluation. It is very likely "
        "that your model is misconfigured. You can try to debug the error by "
        "setting error_score='raise'."
    )


def _validate_error_score(error_score: float | Literal["raise"]) -> None:
    """Validate the `error_score` parameter."""
    if error_score == "raise":
        return
    if isinstance(error_score, bool) or not isinstance(error_score, numbers.Real):
        raise ValueError(
            "error_score must be the string 'raise' or a real number, "
            f"got {error_score!r}."
        )
