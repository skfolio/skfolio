"""Model validation module."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np
import numpy.typing as npt
import sklearn as sk
import sklearn.base as skb
import sklearn.exceptions as ske
import sklearn.model_selection as sks
import sklearn.utils as sku
import sklearn.utils.metadata_routing as skm
import sklearn.utils.parallel as skp
from sklearn.pipeline import Pipeline

from skfolio.model_selection._combinatorial import BaseCombinatorialCV
from skfolio.model_selection._multiple_randomized_cv import MultipleRandomizedCV
from skfolio.model_selection._walk_forward import WalkForward
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio
from skfolio.utils.tools import fit_and_predict, safe_split


def cross_val_predict(
    estimator: skb.BaseEstimator | Pipeline,
    X: npt.ArrayLike,
    y: npt.ArrayLike = None,
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
    column_indices: np.ndarray | None = None,
    portfolio_params: dict | None = None,
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

    If the final estimator in the pipeline (or the estimator itself) declares
    `needs_previous_weights=True`, this function automatically propagates
    `previous_weights` from one fold to the next for sequential CV strategies
    (e.g., `WalkForward` or `MultipleRandomizedCV`).

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        Estimator or pipeline whose last step is an optimization estimator.

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
        using all processors.

    method : str
        Invokes the passed method name of the passed estimator.

    verbose : int, default=0
        The verbosity level.

    params : dict, optional
        Parameters to pass to the underlying estimator's ``fit`` and the CV splitter.

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

    portfolio_params :  dict, optional
        Additional portfolio parameters passed to `MultiPeriodPortfolio`.

    Returns
    -------
    predictions : MultiPeriodPortfolio | Population
        This is the result of calling `predict`
    """
    params = {} if params is None else params

    X, y = safe_split(X, y, indices=column_indices, axis=1)
    X, y = sku.indexable(X, y)

    if _routing_enabled():
        # For estimators, a MetadataRouter is created in get_metadata_routing
        # methods. For these router methods, we create the router to use
        # `process_routing` on it.
        # noinspection PyTypeChecker
        router = (
            skm.MetadataRouter(owner="cross_validate")
            .add(
                splitter=cv,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="split"),
            )
            .add(
                estimator=estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        try:
            routed_params = skm.process_routing(router, "fit", **params)
        except ske.UnsetMetadataPassedError as e:
            # The default exception would mention `fit` since in the above
            # `process_routing` code, we pass `fit` as the caller. However,
            # the user is not calling `fit` directly, so we change the message
            # to make it more suitable for this case.
            unrequested_params = sorted(e.unrequested_params)
            raise ske.UnsetMetadataPassedError(
                message=(
                    f"{unrequested_params} are passed to `cross_val_predict` but are"
                    " not explicitly set as requested or not requested for"
                    f" cross_validate's estimator: {estimator.__class__.__name__} Call"
                    " `.set_fit_request({{metadata}}=True)` on the estimator for"
                    f" each metadata in {unrequested_params} that you want to use and"
                    " `metadata=False` for not using it. See the Metadata Routing User"
                    " guide <https://scikit-learn.org/stable/metadata_routing.html>"
                    " for more information."
                ),
                unrequested_params=e.unrequested_params,
                routed_params=e.routed_params,
            ) from None
    else:
        routed_params = sku.Bunch()
        routed_params.splitter = sku.Bunch(split={})
        routed_params.estimator = sku.Bunch(fit=params)

    cv = sks.check_cv(cv, y)
    splits = list(cv.split(X, y, **routed_params.splitter.split))

    portfolio_params = {} if portfolio_params is None else portfolio_params.copy()

    # We ensure that the folds are not shuffled
    if not isinstance(cv, BaseCombinatorialCV | MultipleRandomizedCV):
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

    # estimator can be a Pipeline
    last_step = _get_last_step(estimator)

    if getattr(last_step, "needs_previous_weights", False) and isinstance(
        cv, WalkForward | MultipleRandomizedCV | sks.TimeSeriesSplit
    ):
        if isinstance(cv, MultipleRandomizedCV):
            splits = list(cv.split(X, y, **routed_params.splitter.split))
            path_ids = cv.get_path_ids()
            paths = defaultdict(list)
            for (train, test, col_idx), pid in zip(splits, path_ids, strict=True):
                paths[pid].append((train, test, col_idx))

            parallel = skp.Parallel(
                n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch
            )
            predictions = parallel(
                skp.delayed(_run_path)(
                    estimator=estimator,
                    X=X,
                    y=y,
                    routed_params=routed_params,
                    method=method,
                    path_splits=paths[pid],
                )
                for pid in sorted(paths.keys())
            )
            predictions = [ptf for path in predictions for ptf in path]

        else:
            if n_jobs not in (None, 1):
                warnings.warn(
                    "Parallel processing has been disabled because the optimization "
                    "method requires sequential processing of previous weights. To "
                    "suppress this warning, set `n_jobs=None`, or disable sequential "
                    "processing of previous weights by setting your Optimization's "
                    "`needs_previous_weights` attribute to False.",
                    stacklevel=2,
                )
            predictions = _run_path(
                estimator=estimator,
                X=X,
                y=y,
                routed_params=routed_params,
                method=method,
                path_splits=splits,
            )

    else:
        # We clone the estimator to make sure that all the folds are independent
        # and that it is pickle-able.
        parallel = skp.Parallel(
            n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch
        )
        # TODO remove when https://github.com/joblib/joblib/issues/1071 is fixed
        predictions = parallel(
            skp.delayed(fit_and_predict)(
                sk.clone(estimator),
                X,
                y,
                train=train,
                test=test,
                fit_params=routed_params.estimator.fit,
                method=method,
                column_indices=column_indices[0] if column_indices else None,
            )
            for train, test, *column_indices in splits
        )

    if isinstance(cv, BaseCombinatorialCV | MultipleRandomizedCV):
        path_ids = cv.get_path_ids()
        path_nb = np.max(path_ids) + 1
        portfolios = [[] for _ in range(path_nb)]
        if isinstance(cv, BaseCombinatorialCV):
            for i, prediction in enumerate(predictions):
                for j, p in enumerate(prediction):
                    path_id = path_ids[i, j]
                    portfolios[path_id].append(p)
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
        test_indices = [test for _, test in splits]
        concat = np.concatenate(test_indices)
        if np.unique(concat, axis=0).shape[0] != concat.shape[0]:
            raise ValueError(
                "`cross_val_predict` only works with non-duplicated test indices"
            )
        sorted_fold_id = np.argsort([x[0] for x in test_indices])
        pred = MultiPeriodPortfolio(
            portfolios=[predictions[fold_id] for fold_id in sorted_fold_id],
            check_observations_order=False,
            **portfolio_params,
        )

    return pred


def _routing_enabled() -> bool:
    """Return whether metadata routing is enabled.

    Returns
    -------
    enabled: bool
        Whether metadata routing is enabled. If the config is not set, it
        defaults to False.
    """
    return sk.get_config().get("enable_metadata_routing", False)


def _asset_names_enabled(X: npt.ArrayLike) -> bool:
    """Return whether X is a DataFrame and its column names are transferred inside
    a Pipeline.
    """
    return hasattr(X, "columns") and sk.get_config().get("transform_output") in [
        "pandas",
        "polars",
    ]


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


def _run_path(
    estimator: skb.BaseEstimator | Pipeline,
    X: npt.ArrayLike,
    y: npt.ArrayLike | None,
    routed_params: sku.Bunch,
    method: str,
    path_splits: list[tuple[np.ndarray, np.ndarray, np.ndarray | None]],
) -> list[Portfolio]:
    """Run sequential fit/predict along a single path of ordered splits.

    Used when the final estimator requires previous portfolio weights between
    consecutive folds (e.g. walk-forward validation). The function propagates
    the `previous_weights` from the prediction of the previous fold to the
    next one.

    Parameters
    ----------
    estimator : BaseEstimator | Pipeline
        Estimator or pipeline to clone and fit on each split.

    X : array-like of shape (n_observations, n_assets)
        Asset returns.

    y : array-like of shape (n_observations, n_targets), optional
        Optional target data (e.g., factor returns).

    routed_params : Bunch
        Fit parameters after metadata routing (``routed_params.estimator.fit``).

    method : str
        Estimator method to call on the test fold (e.g. ``"predict"``).

    path_splits : list of tuple
        Sequence of ``(train_idx, test_idx[, column_indices])`` describing one
        path of folds. ``column_indices`` can be ``None``.

    Returns
    -------
    list[Portfolio]
        Portfolios predicted for each test fold in the path, in order.
    """
    use_dict = _asset_names_enabled(X)
    predictions = []
    prev_weights = _get_last_step(estimator).previous_weights
    for train, test, *column_indices in path_splits:
        est = sk.clone(estimator)
        last_step = _get_last_step(est)
        last_step.set_params(previous_weights=prev_weights)
        ptf = fit_and_predict(
            est,
            X,
            y,
            train=train,
            test=test,
            fit_params=routed_params.estimator.fit,
            method=method,
            column_indices=column_indices[0] if column_indices else None,
        )
        predictions.append(ptf)
        prev_weights = ptf.weights_dict if use_dict else ptf.weights
    return predictions
