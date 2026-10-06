"""Model validation module."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import sklearn as sk
import sklearn.base as skb
import sklearn.exceptions as ske
import sklearn.model_selection as sks
import sklearn.utils as sku
import sklearn.utils.metadata_routing as skm
import sklearn.utils.parallel as skp
from sklearn.pipeline import Pipeline

from skfolio._constants import _RISK_FREE_RATE
from skfolio.model_selection._combinatorial import BaseCombinatorialCV
from skfolio.model_selection._multiple_randomized_cv import MultipleRandomizedCV
from skfolio.model_selection._walk_forward import WalkForward
from skfolio.population import Population
from skfolio.portfolio import FailedPortfolio, MultiPeriodPortfolio, Portfolio
from skfolio.portfolio._base import (
    _PORTFOLIO_MEASURE_PARAMS,
    _normalize_annualization_factor_alias,
)
from skfolio.typing import ArrayLike, IntArray
from skfolio.utils.tools import fit_and_predict, safe_split

if TYPE_CHECKING:
    from skfolio.optimization._base import BaseOptimization

_EstimatorT = TypeVar("_EstimatorT", bound=skb.BaseEstimator)


def cross_val_predict(
    estimator: BaseOptimization
    | Pipeline
    | list[tuple[str, BaseOptimization | Pipeline]]
    | tuple[tuple[str, BaseOptimization | Pipeline], ...],
    X: ArrayLike,
    y: ArrayLike = None,
    cv: sks.BaseCrossValidator
    | BaseCombinatorialCV
    | MultipleRandomizedCV
    | int
    | None = None,
    n_jobs: int | None = None,
    method: str = "predict",
    verbose: int = 0,
    params: dict | None = None,
    pre_dispatch: str = "2*n_jobs",
    column_indices: IntArray | None = None,
    portfolio_params: dict | None = None,
    entry_rebalancing_params: dict | None = None,
) -> MultiPeriodPortfolio | Population:
    """Generate cross-validated `Portfolios` estimates.

    The data is split according to the `cv` parameter.
    The optimization estimator is fitted on the training set and portfolios are
    predicted on the corresponding test set.

    For single-path cross-validation such as `KFold` or
    :class:`~skfolio.model_selection.WalkForward`, the output is a
    :class:`~skfolio.portfolio.MultiPeriodPortfolio` where each
    :class:`~skfolio.portfolio.Portfolio` corresponds to a train/test split (`k`
    portfolios for `KFold`).

    For multi-path cross-validation such as
    :class:`~skfolio.model_selection.CombinatorialPurgedCV` or
    :class:`~skfolio.model_selection.MultipleRandomizedCV`, the output is a
    :class:`~skfolio.population.Population` of multiple
    :class:`~skfolio.portfolio.MultiPeriodPortfolio` objects (each test produces a
    collection of paths rather than a single path).

    `estimator` can also be a collection of named estimators, provided as a list or
    tuple of `(name, estimator)` pairs. All estimators are then evaluated on the same
    splits and the output is always a :class:`~skfolio.population.Population`, even
    with a single estimator. It contains one
    :class:`~skfolio.portfolio.MultiPeriodPortfolio` per estimator for single-path
    cross-validation, or one per estimator and path for multi-path cross-validation,
    ordered by estimator and then by path. Each estimator's name is used as the `tag`
    of its `MultiPeriodPortfolio` objects and as their `name` (suffixed with the path
    index for multi-path cross-validation), so the results of an estimator can be
    selected with `population.filter(tags=name)`.

    If the final estimator in the pipeline (or the estimator itself) declares
    `needs_previous_weights=True`, this function automatically propagates
    `previous_weights` from one fold to the next for sequential CV strategies
    (e.g., `WalkForward` or `MultipleRandomizedCV`).

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline | list[tuple[str, BaseEstimator | Pipeline]]
        Portfolio optimization estimator or pipeline whose last step is an optimization
        estimator. To evaluate several estimators on the same splits, provide a
        non-empty list or tuple of `(name, estimator)` pairs with unique, non-empty
        string names.

    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets.

    y : array-like of shape (n_observations, n_targets), optional
        Target data (optional).
        For example, the price returns of the factors.

    cv : int | cross-validation generator, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        * None, to use the default 5-fold cross validation,
        * int, to specify the number of folds in a `(Stratified)KFold`,
        * `CV splitter`,
        * An iterable that generates (train, test) splits as arrays of indices.

    n_jobs : int, optional
        The number of jobs to run in parallel for `fit` of all `estimators`.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. With a collection of estimators, a single pool of
        workers is shared by all estimators: independent folds run as separate
        `(estimator, fold)` jobs and each path of folds that depend on the previous
        holdings runs as one `(estimator, path)` job.

    method : str
        Invokes the passed method name of the passed estimator.

    verbose : int, default=0
        The verbosity level.

    params : dict, optional
        Parameters to pass to the underlying estimator's `fit` and the CV splitter.

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            * None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            * An int, giving the exact number of total jobs that are
              spawned

            * A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    column_indices : ndarray, optional
        Indices of the `X` columns to cross-validate on.

    portfolio_params : dict, optional
        Portfolio parameters for the evaluation.

        Parameters shared by `Portfolio` and `MultiPeriodPortfolio` (`compounded`,
        `risk_free_rate`, `annualization_factor`, `fitness_measures` and the risk
        measure parameters) are applied to the returned `MultiPeriodPortfolio` and to
        each `Portfolio` it contains. A value passed here takes precedence over the
        optimizer's `portfolio_params`. When omitted here, it is inherited from the
        optimizer's `portfolio_params`. When omitted from both, `risk_free_rate` falls
        back to the optimizer's `risk_free_rate` parameter when it has one. These
        parameters only affect how the portfolios are measured, not the optimization.

        `weight_drift` applies to each `Portfolio` of the path. With
        `weight_drift=True`, the weights held within each test window drift with the
        asset returns, and the path runs sequentially: the `ending_weights` of each
        portfolio are passed as `previous_weights` to the next fit. A value passed here
        overrides the optimizer's `portfolio_params`.

        Optimizer parameters such as `transaction_costs`, `management_fees` and
        `previous_weights` are not accepted here. Set them on the optimizer.

        `name`, `tag`, `sample_weight` and `check_observations_order` apply to the
        returned `MultiPeriodPortfolio` only.

        With a collection of estimators, these parameters apply to every estimator,
        following the same precedence rules, except `name` and `tag` which are not
        accepted because they are set from each estimator's name.

    entry_rebalancing_params : dict, optional
        Portfolio optimizer parameters applied only while constructing the first
        portfolio of each sequential path. This is useful when the strategy starts with
        no existing position, while later portfolios represent regular rebalancing from
        the previously predicted weights. For example, the entry rebalancing can relax
        `max_turnover` or use lower `transaction_costs` to avoid a slow ramp from cash
        caused by recurring rebalancing constraints. The first portfolio is included in
        the result. The regular optimizer parameters are used for all subsequent
        optimizations. When provided, `cross_val_predict` evaluates a sequential
        strategy path and propagates `previous_weights` between portfolios. This is only
        supported for sequential CV strategies such as
        :class:`~skfolio.model_selection.WalkForward`,
        :class:`~sklearn.model_selection.TimeSeriesSplit` and
        :class:`~skfolio.model_selection.MultipleRandomizedCV`. With a collection of
        estimators, they apply to every estimator.

    Returns
    -------
    predictions : MultiPeriodPortfolio | Population
        This is the result of calling `predict`. A `Population` is always returned
        for a collection of named estimators.

    Notes
    -----
    With a sequential CV, each portfolio's `ending_weights` are passed as
    `previous_weights` to the next fit when the estimator needs them. Otherwise,
    fits remain independent and previous weights are assigned to the predicted
    portfolios afterward for turnover and cost calculations. Ending weights equal
    the target `weights` when `weight_drift=False` and the weights after the last
    observation when `weight_drift=True`. Failed and empty portfolios are skipped
    when propagating holdings. With a non-sequential CV, drift is applied inside
    each test fold and nothing is propagated.

    Examples
    --------
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.model_selection import WalkForward, cross_val_predict
    >>> from skfolio.optimization import EqualWeighted, InverseVolatility
    >>> from skfolio.preprocessing import prices_to_returns
    >>> X = prices_to_returns(load_sp500_dataset())
    >>> cv = WalkForward(train_size=252, test_size=63)
    >>> pred = cross_val_predict(InverseVolatility(), X, cv=cv)
    >>> type(pred).__name__
    'MultiPeriodPortfolio'

    Evaluate several named estimators on the same splits:

    >>> population = cross_val_predict(
    ...     [("equal", EqualWeighted()), ("inv_vol", InverseVolatility())],
    ...     X,
    ...     cv=cv,
    ... )
    >>> [mpp.name for mpp in population]
    ['equal', 'inv_vol']
    >>> population.filter(tags="inv_vol")[0].tag
    'inv_vol'
    """
    if isinstance(estimator, list | tuple):
        return _cross_val_predict_collection(
            estimators=estimator,
            X=X,
            y=y,
            cv=cv,
            n_jobs=n_jobs,
            method=method,
            verbose=verbose,
            params=params,
            pre_dispatch=pre_dispatch,
            column_indices=column_indices,
            portfolio_params=portfolio_params,
            entry_rebalancing_params=entry_rebalancing_params,
        )

    if not _is_portfolio_optimization_estimator(estimator):
        raise TypeError(
            "skfolio's `cross_val_predict` only supports portfolio optimization "
            "estimators. For non-portfolio optimization estimators, use "
            "`sklearn.model_selection.cross_val_predict`."
        )

    estimator, portfolio_params, explicit_measure_param_names = (
        _resolve_evaluation_portfolio_params(estimator, portfolio_params)
    )

    X, y = safe_split(X, y, indices=column_indices, axis=1)
    X, y = sku.indexable(X, y)

    routed_params = _route_params(
        estimator,
        params,
        cv=cv,
        owner="cross_val_predict",
        callee="fit",
    )

    cv, splits, path_ids = _get_cv_splits(
        cv, X, y, split_params=routed_params.splitter.split
    )
    is_sequential_cv = _is_sequential_cv(cv)
    _check_entry_rebalancing_cv(entry_rebalancing_params, is_sequential_cv)
    run_sequential_path = is_sequential_cv and _uses_sequential_path(
        estimator, entry_rebalancing_params
    )

    if run_sequential_path and not isinstance(cv, MultipleRandomizedCV):
        # A single path of dependent folds cannot be parallelized.
        if n_jobs not in (None, 1):
            warnings.warn(
                "Parallel processing has been disabled because the optimization "
                "method requires sequential processing of previous weights or "
                "`entry_rebalancing_params`. To suppress this warning, set "
                "`n_jobs=None`, remove `entry_rebalancing_params`, or disable the "
                "options that require previous weights, such as `weight_drift`, "
                "transaction costs, `max_turnover`, or a previous-weights fallback.",
                stacklevel=2,
            )
        predictions = _run_path(
            estimator=estimator,
            X=X,
            y=y,
            routed_params=routed_params,
            method=method,
            path_splits=splits,
            entry_rebalancing_params=entry_rebalancing_params,
        )
    else:
        parallel = skp.Parallel(
            n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch
        )
        results = parallel(
            _cv_tasks(
                estimator,
                X=X,
                y=y,
                splits=splits,
                path_ids=path_ids,
                fit_params=routed_params.estimator_params,
                method=method,
                entry_rebalancing_params=entry_rebalancing_params,
                run_sequential_path=run_sequential_path,
            )
        )
        predictions = _flatten_cv_results(results, run_sequential_path)

    return _assemble_cv_prediction(
        cv=cv,
        splits=splits,
        path_ids=path_ids,
        predictions=predictions,
        portfolio_params=portfolio_params,
        explicit_measure_param_names=explicit_measure_param_names,
        propagate_previous_weights=is_sequential_cv and not run_sequential_path,
    )


def _cross_val_predict_collection(
    estimators: list[tuple[str, BaseOptimization | Pipeline]]
    | tuple[tuple[str, BaseOptimization | Pipeline], ...],
    X: ArrayLike,
    y: ArrayLike | None,
    cv: sks.BaseCrossValidator
    | BaseCombinatorialCV
    | MultipleRandomizedCV
    | int
    | None,
    n_jobs: int | None,
    method: str,
    verbose: int,
    params: dict | None,
    pre_dispatch: str,
    column_indices: IntArray | None,
    portfolio_params: dict | None,
    entry_rebalancing_params: dict | None,
) -> Population:
    """Cross-validate a collection of named estimators on shared splits.

    The splits are computed once and shared by all estimators. The fit and predict
    tasks of all estimators are dispatched to a single pool of `n_jobs` workers:
    independent folds run as `(estimator, fold)` tasks and each path of dependent
    folds runs as one `(estimator, path)` task. See :func:`cross_val_predict` for the
    description of the parameters.

    Returns
    -------
    population : Population
        `MultiPeriodPortfolio` objects ordered by estimator and then by path, each
        tagged with the name of its estimator.
    """
    named_estimators = _validate_named_estimators(estimators, owner="cross_val_predict")
    _validate_collection_portfolio_params(portfolio_params, owner="cross_val_predict")

    strategies = []
    for name, estimator in named_estimators:
        if not _is_portfolio_optimization_estimator(estimator):
            raise TypeError(
                "skfolio's `cross_val_predict` only supports portfolio optimization "
                f"estimators, but the estimator named {name!r} is of type "
                f"{type(estimator).__name__}. For non-portfolio optimization "
                "estimators, use `sklearn.model_selection.cross_val_predict`."
            )
        strategies.append(
            (name, *_resolve_evaluation_portfolio_params(estimator, portfolio_params))
        )

    X, y = safe_split(X, y, indices=column_indices, axis=1)
    X, y = sku.indexable(X, y)

    routed_params = _route_collection_params(
        [estimator for _, estimator, _, _ in strategies],
        params,
        cv=cv,
        owner="cross_val_predict",
        callee="fit",
    )

    cv, splits, path_ids = _get_cv_splits(
        cv, X, y, split_params=routed_params.splitter.split
    )
    is_sequential_cv = _is_sequential_cv(cv)
    _check_entry_rebalancing_cv(entry_rebalancing_params, is_sequential_cv)

    # All tasks are built upfront and dispatched to a single pool so that the pool is
    # shared by all estimators instead of nesting an estimator-level pool around a
    # fold-level pool.
    tasks = []
    task_ranges = []
    run_sequential_paths = []
    for (_, estimator, _, _), fit_params in zip(
        strategies, routed_params.estimator_params, strict=True
    ):
        run_sequential_path = is_sequential_cv and _uses_sequential_path(
            estimator, entry_rebalancing_params
        )
        estimator_tasks = _cv_tasks(
            estimator,
            X=X,
            y=y,
            splits=splits,
            path_ids=path_ids,
            fit_params=fit_params,
            method=method,
            entry_rebalancing_params=entry_rebalancing_params,
            run_sequential_path=run_sequential_path,
        )
        task_ranges.append((len(tasks), len(tasks) + len(estimator_tasks)))
        tasks.extend(estimator_tasks)
        run_sequential_paths.append(run_sequential_path)

    parallel = skp.Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(tasks)

    portfolios = []
    for strategy, (start, stop), run_sequential_path in zip(
        strategies, task_ranges, run_sequential_paths, strict=True
    ):
        name, _, estimator_portfolio_params, explicit_measure_param_names = strategy
        prediction = _assemble_cv_prediction(
            cv=cv,
            splits=splits,
            path_ids=path_ids,
            predictions=_flatten_cv_results(results[start:stop], run_sequential_path),
            portfolio_params={**estimator_portfolio_params, "name": name, "tag": name},
            explicit_measure_param_names=explicit_measure_param_names,
            propagate_previous_weights=is_sequential_cv and not run_sequential_path,
        )
        if isinstance(prediction, Population):
            portfolios.extend(prediction)
        else:
            portfolios.append(prediction)
    return Population(portfolios)


def _routing_enabled() -> bool:
    """Return whether metadata routing is enabled.

    Returns
    -------
    enabled: bool
        Whether metadata routing is enabled. If the config is not set, it
        defaults to False.
    """
    return sk.get_config().get("enable_metadata_routing", False)


def _route_params(
    estimator: skb.BaseEstimator | Pipeline,
    params: dict | None = None,
    *,
    owner: str,
    callee: str,
    cv: object | None = None,
) -> sku.Bunch:
    """Build routed parameter bunches for an estimator method.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        The estimator (or pipeline) to route parameters for.

    params : dict, optional
        Raw parameters from the caller.

    owner : str
        Name of the calling function, used in error messages.

    callee : str
        Estimator method that will receive the routed parameters. Use `"fit"` for batch
        evaluation and `"partial_fit"` for online evaluation.

    cv : cross-validator or None, default=None
        Cross-validation splitter. When provided, parameters are also routed to the
        splitter's `split` method and the result includes `routed_params.splitter.split`.

    Returns
    -------
    routed_params : Bunch
        Routed parameters with `.estimator_params` attribute and, when `cv` is provided,
        `.splitter.split`. Values are plain dictionaries, so the result can be passed to
        worker processes.

    Raises
    ------
    UnsetMetadataPassedError
        If metadata routing is enabled and `params` contains metadata that the estimator
        has not explicitly requested.
    """
    params = params or {}

    if not params or not _routing_enabled():
        # With routing disabled the parameters are passed through unchanged, and with no
        # metadata there is nothing to route. `process_routing` is bypassed because it
        # returns a placeholder object that cannot be pickled when metadata is empty.
        routed_params = sku.Bunch(estimator_params=params)
        if cv is not None:
            routed_params.splitter = sku.Bunch(split={})
        return routed_params

    # For estimators, a MetadataRouter is created in get_metadata_routing
    # methods. For these router methods, we create the router to use
    # `process_routing` on it.
    router = skm.MetadataRouter(owner=owner)
    if cv is not None:
        router.add(
            splitter=cv,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="split"),
        )
    router.add(
        estimator=estimator,
        method_mapping=skm.MethodMapping().add(caller="fit", callee=callee),
    )
    try:
        router_params = skm.process_routing(router, "fit", **params)
    except ske.UnsetMetadataPassedError as e:
        raise _unset_metadata_error(
            e,
            owner=owner,
            callee=callee,
            estimator_description=f"estimator: {estimator.__class__.__name__}",
        ) from None

    # Keep only the payload as plain dictionaries: the result is passed to worker
    # processes.
    routed_params = sku.Bunch(estimator_params=dict(router_params.estimator[callee]))
    if cv is not None:
        routed_params.splitter = sku.Bunch(split=dict(router_params.splitter.split))

    return routed_params


def _route_collection_params(
    estimators: list[skb.BaseEstimator | Pipeline],
    params: dict | None = None,
    *,
    owner: str,
    callee: str,
    cv: object | None = None,
) -> sku.Bunch:
    """Build routed parameter bunches for a collection of estimators.

    All estimators are added to a single router, so a metadata only needs to be
    requested by one of them, and each estimator only receives the metadata it
    requested.

    Parameters
    ----------
    estimators : list[BaseEstimator | Pipeline]
        The estimators (or pipelines) to route parameters for.

    params : dict, optional
        Raw parameters from the caller.

    owner : str
        Name of the calling function, used in error messages.

    callee : str
        Estimator method that will receive the routed parameters. Use `"fit"` for batch
        evaluation and `"partial_fit"` for online evaluation.

    cv : cross-validator or None, default=None
        Cross-validation splitter. When provided, parameters are also routed to the
        splitter's `split` method and the result includes `routed_params.splitter.split`.

    Returns
    -------
    routed_params : Bunch
        Routed parameters with `.estimator_params`, a list containing one dictionary
        per estimator in the order of `estimators`, and, when `cv` is provided,
        `.splitter.split`. Values are plain dictionaries, so the result can be passed
        to worker processes.

    Raises
    ------
    UnsetMetadataPassedError
        If metadata routing is enabled and `params` contains metadata that an estimator
        has not explicitly requested.
    """
    params = params or {}

    if not params or not _routing_enabled():
        # Same as `_route_params`: with routing disabled the parameters are passed
        # through unchanged to every estimator.
        routed_params = sku.Bunch(estimator_params=[params.copy() for _ in estimators])
        if cv is not None:
            routed_params.splitter = sku.Bunch(split={})
        return routed_params

    router = skm.MetadataRouter(owner=owner)
    if cv is not None:
        router.add(
            splitter=cv,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="split"),
        )
    # Positional keys are used because estimator names are arbitrary strings that could
    # clash with the `splitter` key.
    keys = [f"estimator_{i}" for i in range(len(estimators))]
    for key, estimator in zip(keys, estimators, strict=True):
        router.add(
            method_mapping=skm.MethodMapping().add(caller="fit", callee=callee),
            **{key: estimator},
        )
    try:
        router_params = skm.process_routing(router, "fit", **params)
    except ske.UnsetMetadataPassedError as e:
        estimator_names = ", ".join(
            estimator.__class__.__name__ for estimator in estimators
        )
        raise _unset_metadata_error(
            e,
            owner=owner,
            callee=callee,
            estimator_description=f"estimators: {estimator_names}",
        ) from None

    routed_params = sku.Bunch(
        estimator_params=[dict(router_params[key][callee]) for key in keys]
    )
    if cv is not None:
        routed_params.splitter = sku.Bunch(split=dict(router_params.splitter.split))

    return routed_params


def _unset_metadata_error(
    error: ske.UnsetMetadataPassedError,
    *,
    owner: str,
    callee: str,
    estimator_description: str,
) -> ske.UnsetMetadataPassedError:
    """Rephrase an `UnsetMetadataPassedError` raised while routing parameters.

    The default exception would mention `fit` since `process_routing` is called with
    `fit` as the caller. However, the user is not calling `fit` directly, so the
    message is changed to make it more suitable for this case.

    Parameters
    ----------
    error : UnsetMetadataPassedError
        The error raised by `process_routing`.

    owner : str
        Name of the calling function.

    callee : str
        Estimator method that receives the routed parameters.

    estimator_description : str
        Description of the routed estimator(s), e.g. `"estimator: MeanRisk"`.

    Returns
    -------
    error : UnsetMetadataPassedError
        The rephrased error.
    """
    unrequested_params = sorted(error.unrequested_params)
    request_method = f"set_{callee}_request"
    return ske.UnsetMetadataPassedError(
        message=(
            f"{unrequested_params} are passed to `{owner}` but are"
            " not explicitly set as requested or not requested for"
            f" {owner}'s {estimator_description}. Call"
            f" `.{request_method}({{metadata}}=True)` on the estimator"
            f" for each metadata in {unrequested_params} that you want"
            " to use and `metadata=False` if you are not using it. See the"
            " Metadata Routing User guide"
            " <https://scikit-learn.org/stable/metadata_routing.html>"
            " for more information."
        ),
        unrequested_params=error.unrequested_params,
        routed_params=error.routed_params,
    )


def _has_asset_names(X: ArrayLike) -> bool:
    """Return whether the optimizer's actual input carries string asset names."""
    return hasattr(X, "columns") and all(isinstance(name, str) for name in X.columns)  # ty: ignore[not-iterable]


def _propagate_previous_weights(portfolios: list[Portfolio]) -> list[Portfolio]:
    """Set previous weights along a path after independent fits.

    The first portfolio retains the supplied initial holdings. Later portfolios
    are reconstructed only when their previous holdings differ from the last
    successful period's ending weights. Failed and empty periods do not advance
    the holdings.
    """
    result = []
    previous_weights = None
    for portfolio in portfolios:
        if not isinstance(portfolio, FailedPortfolio) and portfolio.n_observations:
            if previous_weights is not None:
                params = portfolio._get_init_params()
                current = params["previous_weights"]
                if isinstance(current, dict) or isinstance(previous_weights, dict):
                    same_weights = (
                        isinstance(current, dict)
                        and isinstance(previous_weights, dict)
                        and current == previous_weights
                    )
                else:
                    same_weights = np.array_equal(current, previous_weights)
                if not same_weights:
                    params["previous_weights"] = previous_weights
                    portfolio = type(portfolio)(**params)
            previous_weights = (
                portfolio.ending_weights_dict
                if _has_asset_names(X=portfolio.X)
                else portfolio.ending_weights
            )
        result.append(portfolio)
    return result


def _get_last_step(estimator: skb.BaseEstimator | Pipeline) -> skb.BaseEstimator:
    """Return the final estimator to be fitted/predicted.

    If `estimator` is a `Pipeline`, returns its last step; otherwise returns
    `estimator` itself.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        Estimator or pipeline passed to cross-validation.

    Returns
    -------
    BaseEstimator
        The final estimator (last step when a pipeline).
    """
    if isinstance(estimator, Pipeline):
        return estimator[-1]
    return estimator


def _resolve_evaluation_portfolio_params(
    estimator: _EstimatorT,
    portfolio_params: dict | None,
    *,
    clone_estimator: bool = True,
) -> tuple[_EstimatorT, dict, set[str]]:
    """Resolve parameters for individual and multi-period portfolio evaluation.

    Settings listed in `_PORTFOLIO_MEASURE_PARAMS` configure every resulting
    `MultiPeriodPortfolio` and each `Portfolio` it contains. When absent from the
    evaluation call, the `MultiPeriodPortfolio` inherits the corresponding value from
    the portfolio optimizer's `portfolio_params`. `risk_free_rate` also falls back to
    the optimizer attribute when available. An evaluation-level `weight_drift` is
    copied into the final estimator step so `predict` can construct each `Portfolio`
    return series and its `ending_weights`.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        Estimator or pipeline whose last step produces `Portfolio` objects.

    portfolio_params : dict, optional
        Parameters supplied to the evaluation call, possibly including measure
        parameters and `weight_drift`.

    clone_estimator : bool, default=True
        If True, the estimator is cloned before being modified. Online helpers pass
        False because they operate on a clone whose fitted state must be kept.

    Returns
    -------
    estimator : BaseEstimator | Pipeline
        A clone carrying the evaluation-level `weight_drift`, or the input estimator
        when `weight_drift` is not supplied.

    multi_period_portfolio_params : dict
        Resolved parameters for the resulting `MultiPeriodPortfolio` objects, without
        `weight_drift`.

    explicit_measure_param_names : set[str]
        Canonical names of measure parameters explicitly supplied by the evaluation
        call. After constructing each `MultiPeriodPortfolio`, its resolved values for
        these parameters must be copied to every `Portfolio` it contains.
    """
    multi_period_portfolio_params = _normalize_annualization_factor_alias(
        {} if portfolio_params is None else portfolio_params,
        stacklevel=5,
    )
    explicit_measure_param_names = (
        set(multi_period_portfolio_params) & _PORTFOLIO_MEASURE_PARAMS
    )

    last_step = _get_last_step(estimator)
    estimator_portfolio_params = _normalize_annualization_factor_alias(
        {} if last_step.portfolio_params is None else last_step.portfolio_params,  # ty: ignore[unresolved-attribute]
        stacklevel=5,
    )
    for param in _PORTFOLIO_MEASURE_PARAMS:
        if (
            param not in multi_period_portfolio_params
            and param in estimator_portfolio_params
        ):
            multi_period_portfolio_params[param] = estimator_portfolio_params[param]

    if _RISK_FREE_RATE not in multi_period_portfolio_params and hasattr(
        last_step, _RISK_FREE_RATE
    ):
        multi_period_portfolio_params[_RISK_FREE_RATE] = getattr(
            last_step, _RISK_FREE_RATE
        )

    if "weight_drift" not in multi_period_portfolio_params:
        return estimator, multi_period_portfolio_params, explicit_measure_param_names

    weight_drift = multi_period_portfolio_params.pop("weight_drift")
    if clone_estimator:
        estimator = sk.clone(estimator)
    last_step = _get_last_step(estimator)
    individual_portfolio_params = (
        {} if last_step.portfolio_params is None else last_step.portfolio_params.copy()  # ty: ignore[unresolved-attribute]
    )
    individual_portfolio_params["weight_drift"] = weight_drift
    last_step.set_params(portfolio_params=individual_portfolio_params)
    return estimator, multi_period_portfolio_params, explicit_measure_param_names


def _sync_measure_params_to_portfolios(
    prediction: MultiPeriodPortfolio | Population,
    explicit_measure_param_names: set[str],
) -> None:
    """Copy the given measure parameters from each `MultiPeriodPortfolio` to the
    `Portfolio` objects it contains.

    The values are read from the constructed `MultiPeriodPortfolio` rather than from
    the evaluation call, so that constructor defaults are applied once: an explicit
    `annualization_factor=None` reaches the children as 252 and an explicit
    `fitness_measures=None` as the default measures.
    """
    if not explicit_measure_param_names:
        return

    multi_period_portfolios = (
        prediction if isinstance(prediction, Population) else [prediction]
    )
    for multi_period_portfolio in multi_period_portfolios:
        measure_params = {
            param: getattr(multi_period_portfolio, param)
            for param in explicit_measure_param_names
        }
        for portfolio in multi_period_portfolio:
            for param, value in measure_params.items():
                setattr(portfolio, param, value)


def _is_portfolio_optimization_estimator(
    estimator: skb.BaseEstimator | Pipeline,
) -> bool:
    """Return whether the estimator or the last pipeline step is a portfolio
    optimization estimator.

    Parameters
    ----------
    estimator : BaseEstimator or Pipeline
        Estimator to inspect. If a `Pipeline` is provided, its last step is
        inspected.

    Returns
    -------
    is_portfolio_optimization_estimator : bool
        `True` when `estimator` itself is a portfolio optimization estimator,
        or, for a `Pipeline`, when its last step is one.
    """
    # Imported here rather than at module scope: `skfolio.optimization` imports
    # `skfolio.model_selection` (through `optimization.cluster._nco` and
    # `optimization.ensemble._stacking`), which imports this module, so a
    # module-level import would close a package-level cycle between
    # `skfolio.model_selection` and `skfolio.optimization`. Annotations are
    # postponed, so the `TYPE_CHECKING` import above covers the signature.
    from skfolio.optimization._base import BaseOptimization

    return isinstance(_get_last_step(estimator), BaseOptimization)


def _apply_entry_rebalancing_params(
    estimator: skb.BaseEstimator | Pipeline,
    entry_rebalancing_params: dict | None,
) -> dict | None:
    """Apply temporary parameters to the final optimization estimator.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        Portfolio optimization estimator or pipeline whose last step is an
        optimization estimator.

    entry_rebalancing_params : dict, optional
        Parameters to apply while constructing the first portfolio.

    Returns
    -------
    previous_params : dict | None
        Original parameter values to restore after the first optimization, or `None`
        when no temporary parameters were provided.
    """
    if entry_rebalancing_params is None:
        return None

    _validate_entry_rebalancing_params(estimator, entry_rebalancing_params)
    last_step = _get_last_step(estimator)
    valid_params = last_step.get_params(deep=True)
    previous_params = {name: valid_params[name] for name in entry_rebalancing_params}
    last_step.set_params(**entry_rebalancing_params)
    return previous_params


def _validate_entry_rebalancing_params(
    estimator: skb.BaseEstimator | Pipeline,
    entry_rebalancing_params: dict | None,
) -> None:
    """Validate parameters applied only to the first portfolio."""
    if entry_rebalancing_params is None:
        return

    last_step = _get_last_step(estimator)
    valid_params = last_step.get_params(deep=True)
    unknown_params = sorted(set(entry_rebalancing_params) - set(valid_params))
    if unknown_params:
        raise ValueError(
            "`entry_rebalancing_params` contains invalid parameter names for "
            f"{last_step.__class__.__name__}: {unknown_params}."
        )


def _run_path(
    estimator: skb.BaseEstimator | Pipeline,
    X: ArrayLike,
    y: ArrayLike | None,
    routed_params: sku.Bunch,
    method: str,
    path_splits: list[tuple[IntArray, IntArray, IntArray | None]],
    entry_rebalancing_params: dict | None = None,
) -> list[Portfolio]:
    """Run sequential fit/predict along a single path of ordered splits.

    Used when the final estimator requires previous portfolio weights between
    consecutive folds (e.g. walk-forward validation). The function passes each
    portfolio's `ending_weights` as `previous_weights` to the next fit. A failed
    prediction leaves the propagated weights unchanged.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        Estimator or pipeline to clone and fit on each split.

    X : array-like of shape (n_observations, n_assets)
        Asset returns.

    y : array-like of shape (n_observations, n_targets), optional
        Optional target data (e.g., factor returns).

    routed_params : Bunch
        Fit parameters after metadata routing (`routed_params.estimator_params`).

    method : str
        Estimator method to call on the test fold (e.g. `"predict"`).

    path_splits : list of tuple
        Sequence of `(train_idx, test_idx[, column_indices])` describing one
        path of folds. `column_indices` can be `None`.

    entry_rebalancing_params : dict, optional
        Parameters applied only while constructing the first portfolio in the path.

    Returns
    -------
    list[Portfolio]
        Portfolios predicted for each test fold in the path, in order.
    """
    predictions = []
    prev_weights = _get_last_step(estimator).previous_weights  # ty: ignore[unresolved-attribute]
    for i, (train, test, *column_indices) in enumerate(path_splits):
        est = sk.clone(estimator)
        last_step = _get_last_step(est)
        last_step.set_params(previous_weights=prev_weights)
        if i == 0:
            _apply_entry_rebalancing_params(est, entry_rebalancing_params)
        ptf = fit_and_predict(
            est,
            X,
            y,
            train=train,
            test=test,
            fit_params=routed_params.estimator_params,
            method=method,
            column_indices=column_indices[0] if column_indices else None,
        )
        if isinstance(ptf, Population):
            raise ValueError(
                "Sequential propagation of `previous_weights` requires one "
                "Portfolio per fold. The estimator returned a Population."
            )
        predictions.append(ptf)
        if not isinstance(ptf, FailedPortfolio) and ptf.n_observations:
            prev_weights = (
                ptf.ending_weights_dict
                if _has_asset_names(X=ptf.X)
                else ptf.ending_weights
            )
    return predictions


def _validate_named_estimators(
    estimators: list | tuple, *, owner: str
) -> list[tuple[str, skb.BaseEstimator | Pipeline]]:
    """Validate a collection of `(name, estimator)` pairs.

    Parameters
    ----------
    estimators : list | tuple
        Collection of `(name, estimator)` pairs.

    owner : str
        Name of the calling function, used in error messages.

    Returns
    -------
    named_estimators : list[tuple[str, BaseEstimator | Pipeline]]
        The validated `(name, estimator)` pairs.

    Raises
    ------
    TypeError
        If an element is not a `(name, estimator)` pair or if a name is not a string.

    ValueError
        If the collection is empty or if names are empty or duplicated.
    """
    if len(estimators) == 0:
        raise ValueError(
            f"`{owner}` received an empty collection of estimators. Provide at least "
            "one `(name, estimator)` pair."
        )

    named_estimators = []
    for item in estimators:
        if not isinstance(item, tuple | list) or len(item) != 2:
            raise TypeError(
                f"When `{owner}` receives a collection of estimators, each element "
                f"must be a `(name, estimator)` pair, got {item!r}."
            )
        name, estimator = item
        if not isinstance(name, str):
            raise TypeError(
                f"Estimator names must be strings, got {name!r} of type "
                f"{type(name).__name__}."
            )
        if not name:
            raise ValueError("Estimator names must be non-empty strings.")
        named_estimators.append((name, estimator))

    names = [name for name, _ in named_estimators]
    duplicated_names = sorted({name for name in names if names.count(name) > 1})
    if duplicated_names:
        raise ValueError(
            f"Estimator names must be unique, got duplicated names: {duplicated_names}."
        )
    return named_estimators


def _validate_collection_portfolio_params(
    portfolio_params: dict | None, *, owner: str
) -> None:
    """Reject portfolio labels that conflict with the estimator names.

    When a collection of named estimators is evaluated, each estimator's name is used
    as the `name` and `tag` of its portfolios, so they cannot be shared.

    Parameters
    ----------
    portfolio_params : dict, optional
        Portfolio parameters shared by all estimators.

    owner : str
        Name of the calling function, used in error messages.

    Raises
    ------
    ValueError
        If `portfolio_params` contains `name` or `tag`.
    """
    conflicting_params = sorted({"name", "tag"} & set(portfolio_params or {}))
    if conflicting_params:
        raise ValueError(
            f"{conflicting_params} cannot be set in `portfolio_params` when `{owner}` "
            "receives a collection of named estimators: each estimator's name is used "
            "as the `name` and `tag` of its portfolios."
        )


def _get_cv_splits(
    cv: sks.BaseCrossValidator
    | BaseCombinatorialCV
    | MultipleRandomizedCV
    | int
    | None,
    X: ArrayLike,
    y: ArrayLike | None,
    split_params: dict,
) -> tuple[
    sks.BaseCrossValidator | BaseCombinatorialCV | MultipleRandomizedCV,
    list[tuple],
    IntArray | None,
]:
    """Check the cross-validator and compute its splits once.

    Parameters
    ----------
    cv : int | cross-validation generator, optional
        Cross-validation splitting strategy, see :func:`cross_val_predict`.

    X : array-like of shape (n_observations, n_assets)
        Asset returns.

    y : array-like of shape (n_observations, n_targets), optional
        Optional target data.

    split_params : dict
        Routed parameters for the splitter's `split` method.

    Returns
    -------
    cv : cross-validator
        The checked cross-validator.

    splits : list[tuple]
        The `(train, test)` or `(train, test, column_indices)` tuples of each split.

    path_ids : ndarray or None
        Path id of each test set for multi-path cross-validation
        (`BaseCombinatorialCV` and `MultipleRandomizedCV`), otherwise `None`. They are
        read right after splitting, so they always describe `splits`.

    Raises
    ------
    ValueError
        If the cross-validation strategy produces no splits or shuffled folds.
    """
    cv = sks.check_cv(cv, y)
    splits = list(cv.split(X, y, **split_params))
    if len(splits) == 0:
        raise ValueError(
            "The cross-validation strategy produced no splits. Check the number of "
            "observations and cross-validation parameters."
        )

    if isinstance(cv, BaseCombinatorialCV | MultipleRandomizedCV):
        return cv, splits, cv.get_path_ids()

    # We ensure that the folds are not shuffled
    try:
        if cv.shuffle:
            raise ValueError(
                "`cross_val_predict` only works with cross-validation setting"
                " `shuffle=False`"
            )
    except AttributeError:
        # If we cannot find the attribute shuffle, we check if the first folds
        # are shuffled
        for fold in splits[0]:
            if not np.all(np.diff(fold) > 0):
                raise ValueError(
                    "`cross_val_predict` only works with un-shuffled folds"
                ) from None
    return cv, splits, None


def _is_sequential_cv(cv: object) -> bool:
    """Return whether the cross-validator produces chronologically ordered folds.

    Parameters
    ----------
    cv : cross-validator
        The checked cross-validator.

    Returns
    -------
    is_sequential_cv : bool
        `True` for `WalkForward`, `MultipleRandomizedCV` and `TimeSeriesSplit`.
    """
    return isinstance(cv, WalkForward | MultipleRandomizedCV | sks.TimeSeriesSplit)


def _check_entry_rebalancing_cv(
    entry_rebalancing_params: dict | None, is_sequential_cv: bool
) -> None:
    """Raise if `entry_rebalancing_params` is used with a non-sequential CV.

    Parameters
    ----------
    entry_rebalancing_params : dict, optional
        Parameters applied only while constructing the first portfolio of each path.

    is_sequential_cv : bool
        Whether the cross-validator is sequential.

    Raises
    ------
    ValueError
        If `entry_rebalancing_params` is provided with a non-sequential CV.
    """
    if entry_rebalancing_params is not None and not is_sequential_cv:
        raise ValueError(
            "`entry_rebalancing_params` is only supported with sequential CV "
            "strategies: `WalkForward`, `TimeSeriesSplit` and `MultipleRandomizedCV`."
        )


def _uses_sequential_path(
    estimator: skb.BaseEstimator | Pipeline, entry_rebalancing_params: dict | None
) -> bool:
    """Return whether the folds of an estimator depend on the previous holdings.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        Estimator or pipeline whose last step is an optimization estimator.

    entry_rebalancing_params : dict, optional
        Parameters applied only while constructing the first portfolio of each path.

    Returns
    -------
    uses_sequential_path : bool
        `True` when the final estimator declares `needs_previous_weights=True` or
        when `entry_rebalancing_params` is provided.
    """
    return (
        getattr(_get_last_step(estimator), "needs_previous_weights", False)
        or entry_rebalancing_params is not None
    )


def _cv_tasks(
    estimator: skb.BaseEstimator | Pipeline,
    *,
    X: ArrayLike,
    y: ArrayLike | None,
    splits: list[tuple],
    path_ids: IntArray | None,
    fit_params: dict,
    method: str,
    entry_rebalancing_params: dict | None,
    run_sequential_path: bool,
) -> list:
    """Build the delayed fit and predict tasks of one estimator.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        Estimator or pipeline whose last step is an optimization estimator.

    X : array-like of shape (n_observations, n_assets)
        Asset returns.

    y : array-like of shape (n_observations, n_targets), optional
        Optional target data.

    splits : list[tuple]
        The `(train, test)` or `(train, test, column_indices)` tuples of each split.

    path_ids : ndarray or None
        Path id of each test set for multi-path cross-validation, otherwise `None`.

    fit_params : dict
        Routed parameters passed to the estimator's `fit`.

    method : str
        Estimator method to call on each test set.

    entry_rebalancing_params : dict, optional
        Parameters applied only while constructing the first portfolio of each path.

    run_sequential_path : bool
        Whether the folds depend on the previous holdings and must be run in order
        along each path.

    Returns
    -------
    tasks : list
        Delayed calls to dispatch with `joblib`. When `run_sequential_path` is `False`,
        one task per split in split order, each returning the prediction of a clone of
        the estimator. Otherwise, one task per path in path id order, each returning the
        ordered predictions of that path.
    """
    if run_sequential_path:
        if path_ids is None:
            path_splits = [splits]
        else:
            paths = defaultdict(list)
            for split, path_id in zip(splits, path_ids, strict=True):
                paths[path_id].append(split)
            path_splits = [paths[path_id] for path_id in sorted(paths)]
        routed_params = sku.Bunch(estimator_params=fit_params)
        return [
            skp.delayed(_run_path)(
                estimator=estimator,
                X=X,
                y=y,
                routed_params=routed_params,
                method=method,
                path_splits=path,
                entry_rebalancing_params=entry_rebalancing_params,
            )
            for path in path_splits
        ]

    # We clone the estimator to make sure that all the folds are independent
    # and that it is pickle-able.
    # TODO remove when https://github.com/joblib/joblib/issues/1071 is fixed
    return [
        skp.delayed(fit_and_predict)(
            sk.clone(estimator),
            X,
            y,
            train=train,
            test=test,
            fit_params=fit_params,
            method=method,
            column_indices=column_indices[0] if column_indices else None,
        )
        for train, test, *column_indices in splits
    ]


def _flatten_cv_results(results: list, run_sequential_path: bool) -> list:
    """Convert the task results of one estimator into one prediction per split.

    Parameters
    ----------
    results : list
        Results of the tasks built by `_cv_tasks`, in task order.

    run_sequential_path : bool
        Whether the tasks were sequential paths returning a list of predictions.

    Returns
    -------
    predictions : list
        Predictions in split order.
    """
    if run_sequential_path:
        return [prediction for path in results for prediction in path]
    return results


def _assemble_cv_prediction(
    *,
    cv: object,
    splits: list[tuple],
    path_ids: IntArray | None,
    predictions: list,
    portfolio_params: dict,
    explicit_measure_param_names: set[str],
    propagate_previous_weights: bool,
) -> MultiPeriodPortfolio | Population:
    """Assemble the predictions of one estimator into the cross-validation output.

    Parameters
    ----------
    cv : cross-validator
        The checked cross-validator.

    splits : list[tuple]
        The `(train, test)` or `(train, test, column_indices)` tuples of each split.

    path_ids : ndarray or None
        Path id of each test set for multi-path cross-validation, otherwise `None`.

    predictions : list
        Predictions in split order.

    portfolio_params : dict
        Resolved parameters of the resulting `MultiPeriodPortfolio` objects. For
        multi-path cross-validation, `name` is used as the prefix of the path names.

    explicit_measure_param_names : set[str]
        Measure parameters to copy from each `MultiPeriodPortfolio` to its portfolios.

    propagate_previous_weights : bool
        Whether to set the previous weights along each path after independent fits.

    Returns
    -------
    prediction : MultiPeriodPortfolio | Population
        A `MultiPeriodPortfolio` for single-path cross-validation, otherwise a
        `Population` with one `MultiPeriodPortfolio` per path.

    Raises
    ------
    ValueError
        If the test sets of a single-path cross-validation overlap.
    """
    portfolio_params = portfolio_params.copy()
    pred: MultiPeriodPortfolio | Population
    if path_ids is not None:
        path_nb = np.max(path_ids) + 1
        portfolios = [[] for _ in range(path_nb)]
        if isinstance(cv, BaseCombinatorialCV):
            # Combinatorial CV never runs the sequential path: each prediction is a
            # list of portfolios.
            for i, prediction in enumerate(predictions):
                for j, p in enumerate(prediction):
                    portfolios[path_ids[i, j]].append(p)
        else:
            for i, prediction in enumerate(predictions):
                portfolios[path_ids[i]].append(prediction)

        name = portfolio_params.pop("name", "path")
        pred = Population(
            [
                MultiPeriodPortfolio(
                    name=f"{name}_{i}", portfolios=portfolios[i], **portfolio_params
                )
                for i in range(path_nb)
            ]
        )
    else:
        # We need to re-order the test folds in case they were un-ordered by the
        # CV generator.
        # Because the tests folds are not shuffled, we use the first index of each
        # fold to order them.
        test_indices = [split[1] for split in splits]
        concat = np.concatenate(test_indices)
        if np.unique(concat, axis=0).shape[0] != concat.shape[0]:
            raise ValueError(
                "`cross_val_predict` only works with non-duplicated test indices"
            )
        sorted_fold_id = np.argsort([x[0] for x in test_indices])
        pred = MultiPeriodPortfolio(
            portfolios=[predictions[fold_id] for fold_id in sorted_fold_id],
            **portfolio_params,
        )

    if propagate_previous_weights:
        for path in pred if isinstance(pred, Population) else [pred]:
            path.portfolios = _propagate_previous_weights(portfolios=path.portfolios)

    _sync_measure_params_to_portfolios(pred, explicit_measure_param_names)
    return pred
