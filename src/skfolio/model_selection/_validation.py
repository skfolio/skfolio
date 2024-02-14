"""Model validation module."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# scikit-portfolio, Copyright (c) 2022, Carlo Nicolini, Licensed under MIT Licence.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import sklearn as sk
import sklearn.base as skb
import sklearn.model_selection as skm
import sklearn.utils as sku
import sklearn.utils.parallel as skp

from skfolio.model_selection._combinatorial import BaseCombinatorialCV
from skfolio.population import Population
from skfolio.portfolio import MultiPeriodPortfolio
from skfolio.utils.tools import fit_and_predict, safe_split


def cross_val_predict(
    estimator: skb.BaseEstimator,
    X: npt.ArrayLike,
    y: npt.ArrayLike = None,
    groups: np.ndarray | None = None,
    cv: skm.BaseCrossValidator | BaseCombinatorialCV | int | None = None,
    n_jobs: int | None = None,
    method: str = "predict",
    verbose: int = 0,
    fit_params: dict | None = None,
    pre_dispatch: str = "2*n_jobs",
    column_indices: np.ndarray | None = None,
    portfolio_params: dict | None = None,
) -> MultiPeriodPortfolio | Population:
    """Generate cross-validated `Portfolios` estimates.

    The data is split according to the `cv` parameter.
    The optimization estimator is fitted on the training set and portfolios are
    predicted on the corresponding test set.

    For non-combinatorial cross-validation like `Kfold`, the output is the predicted
    :class:`~skfolio.portfolio.MultiPeriodPortfolio` where
    each :class:`~skfolio.portfolio.Portfolio` corresponds to the prediction on each
    train/test pair (`k` portfolios for `Kfold`).

    For combinatorial cross-validation
    like :class:`~skfolio.model_selection.CombinatorialPurgedCV`, the output is the
    predicted :class:`~skfolio.population.Population` of multiple
    :class:`~skfolio.portfolio.MultiPeriodPortfolio` (each test outputs are a
    collection of multiple paths instead of one single path).

    Parameters
    ----------
    estimator : BaseOptimization
        :ref:`Optimization estimators <optimization>` use to fit the data.

    X : array-like of shape (n_observations, n_assets)
        Price returns of the assets.

    y : array-like of shape (n_observations, n_targets), optional
        Target data (optional).
        For example, the price returns of the factors.

    groups : array-like of shape (n_observations,), optional
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" `cv`
        instance (e.g., `GroupKFold`).

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

    fit_params : dict, optional
        Parameters to pass to the fit method of the estimator.

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
    X, y = safe_split(X, y, indices=column_indices, axis=1)
    X, y, groups = sku.indexable(X, y, groups)
    cv = skm.check_cv(cv, y)
    splits = list(cv.split(X, y, groups))
    portfolio_params = {} if portfolio_params is None else portfolio_params.copy()

    # We ensure that the folds are not shuffled
    if not isinstance(cv, BaseCombinatorialCV):
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

    # We clone the estimator to make sure that all the folds are independent
    # and that it is pickle-able.
    parallel = skp.Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    # TODO remove when https://github.com/joblib/joblib/issues/1071 is fixed
    predictions = parallel(
        skp.delayed(fit_and_predict)(
            sk.clone(estimator),
            X,
            y,
            train=train,
            test=test,
            fit_params=fit_params,
            method=method,
        )
        for train, test in splits
    )

    if isinstance(cv, BaseCombinatorialCV):
        path_ids = cv.get_path_ids()
        path_nb = np.max(path_ids) + 1
        portfolios = [[] for _ in range(path_nb)]
        for i, prediction in enumerate(predictions):
            for j, p in enumerate(prediction):
                path_id = path_ids[i, j]
                portfolios[path_id].append(p)
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
        test_indices = np.concatenate([test for _, test in splits])
        if np.unique(test_indices, axis=0).shape[0] != test_indices.shape[0]:
            raise ValueError(
                "`cross_val_predict` only works with non-duplicated test indices"
            )
        test_indices = [test for _, test in splits]
        sorted_fold_id = np.argsort([x[0] for x in test_indices])
        pred = MultiPeriodPortfolio(
            portfolios=[predictions[fold_id] for fold_id in sorted_fold_id],
            check_observations_order=False,
            **portfolio_params,
        )

    return pred
