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
from collections.abc import Iterator, Sequence
from itertools import chain, islice
from typing import TYPE_CHECKING, Any, TypeVar

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
    | Sequence[tuple[str, BaseOptimization | Pipeline]],
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
    """Generate cross-validated portfolios.

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

    When a list of `(name, estimator)` pairs is provided, all estimators are
    evaluated on the same splits and their portfolios are returned together in a
    :class:`~skfolio.population.Population`.

    If the final estimator in the pipeline (or the estimator itself) declares
    `needs_previous_weights=True`, this function automatically propagates
    `previous_weights` from one fold to the next for sequential CV strategies
    (e.g., `WalkForward` or `MultipleRandomizedCV`).

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline | sequence of (str, estimator)
        Portfolio optimization estimator or pipeline whose last step is an optimization
        estimator. To evaluate several estimators, pass a sequence of `(name, estimator)`
        pairs with unique string names.

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
        The number of jobs to run in parallel.
        `None` means 1 unless in a `joblib.parallel_backend` context. -1 means
        using all processors. Named estimators share one worker pool. Folds that
        depend on previous holdings run sequentially within each path.

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

        Optimization parameters such as `transaction_costs`, `management_fees` and
        `previous_weights` are not accepted here. Set them on the optimization
        estimator.

        `name`, `tag`, `sample_weight` and `check_observations_order` apply to the
        returned `MultiPeriodPortfolio` only.

        For named estimators, these settings apply to every estimator. `name` and
        `tag` are set from the estimator names and cannot be overridden here.

    entry_rebalancing_params : dict, optional
        Optimization estimator parameters applied only while constructing the first
        portfolio of each sequential path. This is useful when the strategy starts with
        no existing position, while later portfolios represent regular rebalancing from
        the previously predicted weights. For example, the entry rebalancing can relax
        `max_turnover` or use lower `transaction_costs` to avoid a slow ramp from cash
        caused by recurring rebalancing constraints. The first portfolio is included in
        the result. The regular estimator parameters are used for all subsequent
        optimizations. When provided, `cross_val_predict` evaluates a sequential
        strategy path and propagates `previous_weights` between portfolios. This is only
        supported for sequential CV strategies such as
        :class:`~skfolio.model_selection.WalkForward`,
        :class:`~sklearn.model_selection.TimeSeriesSplit` and
        :class:`~skfolio.model_selection.MultipleRandomizedCV`.
        For named estimators, these settings apply to every estimator.

    Returns
    -------
    predictions : MultiPeriodPortfolio | Population
        A `MultiPeriodPortfolio` for single-path CV, or a `Population` for multi-path CV.
        Named estimators always return a `Population`, with one `MultiPeriodPortfolio`
        per estimator and path, ordered by estimator then path. Estimator names set
        `name` and `tag`.

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
    Evaluate an inverse-volatility portfolio with walk-forward cross-validation:

    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.model_selection import WalkForward, cross_val_predict
    >>> from skfolio.optimization import EqualWeighted, InverseVolatility
    >>> from skfolio.preprocessing import prices_to_returns

    >>> prices = load_sp500_dataset()
    >>> X = prices_to_returns(prices)

    >>> cv = WalkForward(train_size=252, test_size=63)
    >>> mpp = cross_val_predict(InverseVolatility(), X, cv=cv)  # MultiPeriodPortfolio

    To compare estimators on the same splits, pass a list of named estimators:

    >>> estimators = [
    ...     ("equal_weighted", EqualWeighted()),
    ...     ("inverse_volatility", InverseVolatility()),
    ... ]
    >>> population = cross_val_predict(estimators, X, cv=cv)
    >>> [mpp.name for mpp in population]
    ['equal_weighted', 'inverse_volatility']
    """
    is_collection = isinstance(estimator, Sequence) and not isinstance(
        estimator, str | bytes | bytearray
    )
    named_estimators = (
        _validate_named_estimators(estimator) if is_collection else [(None, estimator)]
    )
    if is_collection:
        conflicting_params = sorted({"name", "tag"} & set(portfolio_params or {}))
        if conflicting_params:
            raise ValueError(
                f"{conflicting_params} cannot be set in `portfolio_params` when "
                "`cross_val_predict` receives named estimators: their names determine "
                "the portfolio names and tags."
            )

    evaluations = []
    for name, est in named_estimators:
        if not _is_portfolio_optimization_estimator(est):
            description = (
                f", but the estimator named {name!r} is of type {type(est).__name__}"
                if is_collection
                else ""
            )
            raise TypeError(
                "skfolio's `cross_val_predict` only supports portfolio optimization "
                f"estimators{description}. For non-portfolio optimization estimators, "
                "use `sklearn.model_selection.cross_val_predict`."
            )
        est, evaluation_params, explicit_measure_params = (
            _resolve_evaluation_portfolio_params(est, portfolio_params)  # ty: ignore[invalid-argument-type]
        )
        if is_collection:
            evaluation_params.update(name=name, tag=name)
        evaluations.append((est, evaluation_params, explicit_measure_params))

    X, y = safe_split(X, y, indices=column_indices, axis=1)
    X, y = sku.indexable(X, y)
    cv = sks.check_cv(cv, y)
    routed_params = _route_params(
        [est for est, _, _ in evaluations],
        params,
        cv=cv,
        owner="cross_val_predict",
        callee="fit",
    )
    fit_params = routed_params.estimator_params

    splits = list(cv.split(X, y, **routed_params.splitter.split))
    if len(splits) == 0:
        raise ValueError(
            "The cross-validation strategy produced no splits. Check the number of "
            "observations and cross-validation parameters."
        )

    # We ensure that the folds are not shuffled
    path_ids = None
    if isinstance(cv, BaseCombinatorialCV | MultipleRandomizedCV):
        path_ids = cv.get_path_ids()
        path_nb = np.max(path_ids) + 1
    else:
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

        test_indices = [split[1] for split in splits]
        concat = np.concatenate(test_indices)
        if np.unique(concat, axis=0).shape[0] != concat.shape[0]:
            raise ValueError(
                "`cross_val_predict` only works with non-duplicated test indices"
            )
        # Unshuffled test folds can still be yielded in a different order.
        sorted_fold_id = np.argsort([indices[0] for indices in test_indices])

    is_sequential_cv = isinstance(
        cv, WalkForward | MultipleRandomizedCV | sks.TimeSeriesSplit
    )
    if entry_rebalancing_params is not None and not is_sequential_cv:
        raise ValueError(
            "`entry_rebalancing_params` is only supported with sequential CV "
            "strategies: `WalkForward`, `TimeSeriesSplit` and `MultipleRandomizedCV`."
        )
    use_sequential_paths = [
        is_sequential_cv
        and (
            getattr(_get_last_step(est), "needs_previous_weights", False)
            or entry_rebalancing_params is not None
        )
        for est, _, _ in evaluations
    ]
    paths = [splits]
    if isinstance(cv, MultipleRandomizedCV) and any(use_sequential_paths):
        grouped_splits = defaultdict(list)
        for split, path_id in zip(splits, path_ids, strict=True):  # ty: ignore[invalid-argument-type]
            grouped_splits[path_id].append(split)
        paths = [grouped_splits[path_id] for path_id in sorted(grouped_splits)]

    def tasks() -> Iterator[tuple]:
        """Yield jobs for `Parallel`, with sequential paths before independent folds."""
        for (est, _, _), est_params, use_sequential_path in sorted(
            zip(evaluations, fit_params, use_sequential_paths, strict=True),
            key=lambda item: not item[2],
        ):
            if use_sequential_path:
                for path in paths:
                    yield skp.delayed(_run_path)(
                        estimator=est,
                        X=X,
                        y=y,
                        routed_params=sku.Bunch(estimator_params=est_params),
                        method=method,
                        path_splits=path,
                        entry_rebalancing_params=entry_rebalancing_params,
                    )
            else:
                for train, test, *column_indices in splits:
                    yield skp.delayed(fit_and_predict)(
                        sk.clone(est),
                        X,
                        y,
                        train=train,
                        test=test,
                        fit_params=est_params,
                        method=method,
                        column_indices=column_indices[0] if column_indices else None,
                    )

    if len(evaluations) == 1 and use_sequential_paths[0] and len(paths) == 1:
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
        results = [
            _run_path(
                estimator=evaluations[0][0],
                X=X,
                y=y,
                routed_params=sku.Bunch(estimator_params=fit_params[0]),
                method=method,
                path_splits=paths[0],
                entry_rebalancing_params=entry_rebalancing_params,
            )
        ]
    else:
        parallel = skp.Parallel(
            n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch
        )
        results = parallel(tasks())

    n_path_tasks = len(paths) * sum(use_sequential_paths)
    path_results = islice(results, n_path_tasks)
    fold_results = islice(results, n_path_tasks, None)
    population = []
    predictions: list[Any]
    for (_, evaluation_params, explicit_measure_params), use_sequential_path in zip(
        evaluations, use_sequential_paths, strict=True
    ):
        predictions = list(
            islice(path_results, len(paths))
            if use_sequential_path
            else islice(fold_results, len(splits))
        )
        if use_sequential_path:
            predictions = list(chain.from_iterable(predictions))
        if path_ids is not None:
            portfolios = [[] for _ in range(path_nb)]
            if isinstance(cv, BaseCombinatorialCV):
                # Combinatorial CV never runs the sequential path: each prediction is a
                # list of portfolios.
                for i, prediction in enumerate(predictions):
                    for j, p in enumerate(prediction):
                        path_id = path_ids[i, j]
                        portfolios[path_id].append(p)
            else:
                for i, prediction in enumerate(predictions):
                    portfolios[path_ids[i]].append(prediction)

            name = evaluation_params.pop("name", "path")
            pred = Population(
                [
                    MultiPeriodPortfolio(
                        name=f"{name}_{i}",
                        portfolios=portfolios[i],
                        **evaluation_params,
                    )
                    for i in range(path_nb)
                ]
            )
        else:
            pred = MultiPeriodPortfolio(
                portfolios=[predictions[fold_id] for fold_id in sorted_fold_id],
                **evaluation_params,
            )

        if is_sequential_cv and not use_sequential_path:
            for path in pred if isinstance(pred, Population) else [pred]:
                path.portfolios = _propagate_previous_weights(
                    portfolios=path.portfolios
                )

        _sync_measure_params_to_portfolios(pred, explicit_measure_params)
        if not is_collection:
            return pred
        population.extend(pred if isinstance(pred, Population) else [pred])
    return Population(population)


def _validate_named_estimators(
    estimators: Sequence,
) -> list[tuple[str, skb.BaseEstimator | Pipeline]]:
    """Validate a collection of estimators with unique string names."""
    if not estimators:
        raise ValueError("Received an empty collection of estimators.")
    named_estimators = []
    names = set()
    for item in estimators:
        if not isinstance(item, tuple | list) or len(item) != 2:
            raise TypeError(
                f"Each element must be a `(name, estimator)` pair, got {item!r}."
            )
        name, estimator = item
        if not isinstance(name, str):
            raise TypeError(f"Estimator names must be strings, got {name!r}.")
        if not name:
            raise ValueError("Estimator names must be non-empty strings.")
        if name in names:
            raise ValueError(
                f"Estimator names must be unique, got duplicated names: {[name]}."
            )
        names.add(name)
        named_estimators.append((name, estimator))
    return named_estimators


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
    estimator: skb.BaseEstimator | Pipeline | list[skb.BaseEstimator | Pipeline],
    params: dict | None = None,
    *,
    owner: str,
    callee: str,
    cv: object | None = None,
) -> sku.Bunch:
    """Build routed parameter bunches for estimator methods.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline | list[BaseEstimator | Pipeline]
        The estimator (or pipeline) to route parameters for, or a list of them.

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
        Routed parameters with `.estimator_params` and, when `cv` is provided,
        `.splitter.split`. `.estimator_params` is a dictionary for a single estimator
        and a list with one dictionary per estimator, in input order, for a list input.
        The parameter payloads are plain dictionaries for use in worker processes.

    Raises
    ------
    UnsetMetadataPassedError
        If metadata routing is enabled and `params` contains metadata whose request
        status is unset for an estimator or the splitter.
    """
    is_collection = isinstance(estimator, list)
    estimators = (
        {f"estimator_{i}": est for i, est in enumerate(estimator)}
        if is_collection
        else {"estimator": estimator}
    )
    params = params or {}

    if not params or not _routing_enabled():
        # With routing disabled the parameters are passed through unchanged, and with no
        # metadata there is nothing to route. `process_routing` is bypassed because it
        # returns a placeholder object that cannot be pickled when metadata is empty.
        routed_params = sku.Bunch(
            estimator_params=[params.copy() for _ in estimators]
            if is_collection
            else params
        )
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
    for key, est in estimators.items():
        router.add(
            method_mapping=skm.MethodMapping().add(caller="fit", callee=callee),
            **{key: est},
        )
    try:
        router_params = skm.process_routing(router, "fit", **params)
    except ske.UnsetMetadataPassedError as e:
        # The default exception would mention `fit` since in the above
        # `process_routing` code, we pass `fit` as the caller. However,
        # the user is not calling `fit` directly, so we change the message
        # to make it more suitable for this case.
        unrequested_params = sorted(e.unrequested_params)
        request_method = f"set_{callee}_request"
        description = (
            "estimator: " if len(estimators) == 1 else "estimators: "
        ) + ", ".join(type(est).__name__ for est in estimators.values())
        raise ske.UnsetMetadataPassedError(
            message=(
                f"{unrequested_params} are passed to `{owner}` but are"
                " not explicitly set as requested or not requested for"
                f" {owner}'s {description}. Call"
                f" `.{request_method}({{metadata}}=True)` on the estimator"
                f" for each metadata in {unrequested_params} that you want"
                " to use and `metadata=False` if you are not using it. See the"
                " Metadata Routing User guide"
                " <https://scikit-learn.org/stable/metadata_routing.html>"
                " for more information."
            ),
            unrequested_params=e.unrequested_params,
            routed_params=e.routed_params,
        ) from None

    # Keep only the payload as plain dictionaries: the result is passed to worker
    # processes.
    payloads = [dict(router_params[key][callee]) for key in estimators]
    routed_params = sku.Bunch(
        estimator_params=payloads if is_collection else payloads[0]
    )
    if cv is not None:
        routed_params.splitter = sku.Bunch(split=dict(router_params.splitter.split))

    return routed_params


def _has_asset_names(X: ArrayLike) -> bool:
    """Return whether the estimator's actual input carries string asset names."""
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
    the estimator's `portfolio_params`. `risk_free_rate` also falls back to
    the estimator attribute when available. An evaluation-level `weight_drift` is
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
    estimator: object,
) -> bool:
    """Return whether the estimator or the last pipeline step is a portfolio
    optimization estimator.

    Parameters
    ----------
    estimator : object
        Estimator to inspect. If a `Pipeline` is provided, its last step is
        inspected.

    Returns
    -------
    is_portfolio_optimization_estimator : bool
        `True` when `estimator` itself is a portfolio optimization estimator,
        or, for a `Pipeline`, when its last step is one.
    """
    # Imported here rather than at module scope: `skfolio.optimization` imports
    # `skfolio.model_selection` (through `optimization.hierarchical._clustering._nco`
    # and `optimization.ensemble._stacking`), which imports this module, so a
    # module-level import would close a package-level cycle between
    # `skfolio.model_selection` and `skfolio.optimization`. Annotations are
    # postponed, so the `TYPE_CHECKING` import above covers the signature.
    from skfolio.optimization._base import BaseOptimization

    return isinstance(estimator, skb.BaseEstimator) and isinstance(
        _get_last_step(estimator), BaseOptimization
    )


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
