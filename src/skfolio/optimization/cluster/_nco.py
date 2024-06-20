"""Nested Clusters Optimization estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier
# Grisel Licensed under BSD 3 clause.

from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn as sk
import sklearn.base as skb
import sklearn.model_selection as skm
import sklearn.utils.parallel as skp

import skfolio.typing as skt
from skfolio.cluster import HierarchicalClustering
from skfolio.distance import BaseDistance, PearsonDistance
from skfolio.measures import RatioMeasure
from skfolio.model_selection import BaseCombinatorialCV, cross_val_predict
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.convex import MeanRisk
from skfolio.utils.tools import check_estimator, fit_single_estimator


class NestedClustersOptimization(BaseOptimization):
    """Nested Clusters Optimization estimator.

    Nested Clusters Optimization (NCO) is a portfolio optimization method developed by
    Marcos Lopez de Prado.

    It uses a distance matrix to compute clusters using a clustering algorithm (
    Hierarchical Tree Clustering, KMeans, etc..). For each cluster, the inner-cluster
    weights are computed by fitting the inner-estimator on each cluster using the whole
    training data. Then the outer-cluster weights are computed by training the
    outer-estimator using out-of-sample estimates of the inner-estimators with
    cross-validation. Finally, the final assets weights are the dot-product of the
    inner-weights and outer-weights.

    .. note ::

        The original paper uses KMeans as the clustering algorithm, minimum Variance for
        the inner-estimator and equal-weighted for the outer-estimator. Here we
        generalize it to all `sklearn` and `skfolio` clustering algorithms
        (HierarchicalClustering, KMeans, etc.), all portfolio optimizations
        (Mean-Variance, HRP, etc.) and risk measures (Variance, CVaR, etc.).
        To avoid data leakage at the outer-estimator, we use out-of-sample estimates to
        fit the outer estimator.

    Parameters
    ----------
    inner_estimator : BaseOptimization, optional
        :ref:`Optimization estimator <optimization>` used to estimate the inner-weights
        (also called intra-weights) which are the assets weights inside each cluster.
        The default `None` is to use :class:`~skfolio.optimization.MeanRisk`.

    outer_estimator : BaseOptimization, optional
        :ref:`Optimization estimator <optimization>` used to estimate the outer-weights
        (also called inter-weights) which are the weights applied to each cluster.
        The default `None` is to use :class:`~skfolio.optimization.MeanRisk`.

    distance_estimator : BaseDistance, optional
        :ref:`Distance estimator <distance>`.
        The distance estimator is used to estimate the codependence and the distance
        matrix needed for the computation of the linkage matrix.
        The default (`None`) is to use :class:`~skfolio.distance.PearsonDistance`.

    clustering_estimator : BaseEstimator, optional
        Clustering estimator. Must expose a `labels_` attribute after fitting.
        The clustering estimator is used to compute the clusters of the assets based on
        the distance matrix. The default (`None`) is to use
        :class:`~skfolio.cluster.HierarchicalClustering`.

        .. note ::

            Clustering estimators from `sklearn` are also supported. For example:
            `sklearn.cluster.KMeans`.

    cv : BaseCrossValidator | BaseCombinatorialCV | int | "ignore", optional
        Determines the cross-validation splitting strategy.
        The default (`None`) is to use the 5-fold cross validation `KFold()`.
        It is applied to the inner-estimators. Its out-of-sample outputs are used to
        train the outer-estimator.
        Possible inputs for `cv` are:

            * "ignore": no cross-validation is used (note that it will likely lead to data leakage with a high risk of overfitting)
            * Integer, to specify the number of folds in a :class:`sklearn.model_selection.KFold`
            * An object to be used as a cross-validation generator
            * An iterable yielding train, test splits
            * A :class:`~skfolio.model_selection.CombinatorialPurgedCV`

        If a `CombinatorialCV` cross-validator is used, each cluster out-of-sample
        outputs becomes a collection of multiple paths instead of one single path. The
        selected out-of-sample path among this collection of paths is chosen according
        to the `quantile` and `quantile_measure` parameters.

    n_jobs : int, optional
        The number of jobs to run in parallel for `fit` of all `estimators`.
        The value `-1` means using all processors.
        The default (`None`) means 1 unless in a `joblib.parallel_backend` context.

    quantile : float, default=0.5
        Quantile for a given measure (`quantile_measure`) of the out-of-sample
        inner-estimator paths when the `cv` parameter is a
        :class:`~skfolio.model_selection.CombinatorialPurgedCV` cross-validator.
        The default value is `0.5` corresponding to the path with the median measure.
        (see `cv`)

    quantile_measure : PerfMeasure or RatioMeasure or RiskMeasure or ExtraRiskMeasure, default=RatioMeasure.SHARPE_RATIO
        Measure used for the quantile path selection (see `quantile` and `cv`).
        The default is `RatioMeasure.SHARPE_RATIO`.

    verbose : int, default=0
        The verbosity level. The default value is `0`.

    portfolio_params :  dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name` is copied from the optimization
        model and systematically passed to the portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Weights of the assets.

    distance_estimator_ : BaseDistance
        Fitted `distance_estimator`.

    inner_estimators_ :  list[BaseOptimization]
        List of fitted `inner_estimator`. One per cluster for clusters containing more
        than one asset.

    outer_estimator_ : BaseOptimization
        Fitted `outer_estimator`.

    clustering_estimator_ : BaseEstimator
        Fitted `clustering_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1]  "Building diversified portfolios that outperform out of sample",
        The Journal of Portfolio Management,
        Marcos López de Prado (2016)

    .. [2]  "A robust estimator of the efficient frontier",
        SSRN Electronic Journal,
        Marcos López de Prado (2019)

    .. [3] "Machine Learning for Asset Managers",
        Elements in Quantitative Finance. Cambridge University Press,
        Marcos López de Prado (2020)
    """

    inner_estimators_: list[BaseOptimization]
    outer_estimator_: BaseOptimization
    distance_estimator_: BaseDistance
    clustering_estimator_: skb.BaseEstimator

    def __init__(
        self,
        inner_estimator: BaseOptimization | None = None,
        outer_estimator: BaseOptimization | None = None,
        distance_estimator: BaseDistance | None = None,
        clustering_estimator: skb.BaseEstimator | None = None,
        cv: skm.BaseCrossValidator | BaseCombinatorialCV | str | int | None = None,
        quantile: float = 0.5,
        quantile_measure: skt.Measure = RatioMeasure.SHARPE_RATIO,
        n_jobs: int | None = None,
        verbose: int = 0,
        portfolio_params: dict | None = None,
    ):
        super().__init__(portfolio_params=portfolio_params)
        self.distance_estimator = distance_estimator
        self.clustering_estimator = clustering_estimator
        self.inner_estimator = inner_estimator
        self.outer_estimator = outer_estimator
        self.cv = cv
        self.quantile = quantile
        self.quantile_measure = quantile_measure
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> "NestedClustersOptimization":
        """Fit the Nested Clusters Optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : array-like of shape (n_observations, n_targets), optional
            Price returns of factors or a target benchmark.
            The default is `None`.

        Returns
        -------
        self : NestedClustersOptimization
            Fitted estimator.
        """
        self.distance_estimator_ = check_estimator(
            self.distance_estimator,
            default=PearsonDistance(),
            check_type=BaseDistance,
        )
        self.clustering_estimator_ = check_estimator(
            self.clustering_estimator,
            default=HierarchicalClustering(),
            check_type=skb.BaseEstimator,
        )
        self.outer_estimator_ = check_estimator(
            self.outer_estimator,
            default=MeanRisk(),
            check_type=BaseOptimization,
        )
        _inner_estimator = check_estimator(
            self.inner_estimator,
            default=MeanRisk(),
            check_type=BaseOptimization,
        )

        self.distance_estimator_.fit(X)
        distance = self.distance_estimator_.distance_
        n_assets = distance.shape[0]

        # To keep the asset_names --> used for visualisation
        if isinstance(X, pd.DataFrame):
            distance = pd.DataFrame(distance, columns=X.columns)

        # noinspection PyUnresolvedReferences
        self.clustering_estimator_.fit(distance)
        # noinspection PyUnresolvedReferences
        labels = self.clustering_estimator_.labels_
        n_clusters = max(labels) + 1
        clusters = [np.argwhere(labels == i).flatten() for i in range(n_clusters)]

        # Intra cluster weights
        # Fit the inner estimator on the whole training data. Those
        # base estimators will be used to retrieve the inner weights.
        # They are exposed publicly.
        # noinspection PyCallingNonCallable
        fitted_inner_estimators = skp.Parallel(n_jobs=self.n_jobs)(
            skp.delayed(fit_single_estimator)(
                sk.clone(_inner_estimator), X, y, indices=cluster_ids, axis=1
            )
            for cluster_ids in clusters
            if len(cluster_ids) != 1
        )
        fitted_inner_estimators = iter(fitted_inner_estimators)

        self.inner_estimators_ = []
        inner_weights = []
        for cluster_ids in clusters:
            w = np.zeros(n_assets)
            # For single assets, we don't run the inner optimization estimator.
            if len(cluster_ids) == 1:
                w[cluster_ids] = 1
            else:
                fitted_inner_estimator = next(fitted_inner_estimators)
                self.inner_estimators_.append(fitted_inner_estimator)
                w[cluster_ids] = fitted_inner_estimator.weights_
            inner_weights.append(w)
        inner_weights = np.array(inner_weights)
        assert not any(
            fitted_inner_estimators
        ), "fitted_inner_estimator iterator must be empty"

        # Outer cluster weights
        # To train the outer-estimator using the most data as possible, we use
        # a cross-validation to obtain the output of the cluster estimators.
        # To ensure that the data provided to each estimator are the same,
        # we need to set the random state of the cv if there is one and we
        # need to take a copy.
        if self.cv == "ignore":
            cv_predictions = None
            test_indices = slice(None)
        else:
            cv = skm.check_cv(self.cv)
            if hasattr(cv, "random_state") and cv.random_state is None:
                cv.random_state = np.random.RandomState()
            # noinspection PyCallingNonCallable
            cv_predictions = skp.Parallel(n_jobs=self.n_jobs)(
                skp.delayed(cross_val_predict)(
                    sk.clone(_inner_estimator),
                    X,
                    y,
                    cv=deepcopy(cv),
                    n_jobs=self.n_jobs,
                    verbose=self.verbose,
                    column_indices=cluster_ids,
                    method="predict",
                )
                for cluster_ids in clusters
                if len(cluster_ids) != 1
            )
            cv_predictions = iter(cv_predictions)
            if isinstance(self.cv, BaseCombinatorialCV):
                test_indices = slice(None)
            else:
                test_indices = np.sort(
                    np.concatenate([test for _, test in cv.split(X, y)])
                )

        # We validate and convert to numpy array only after inner-estimator fitting to
        # keep the assets names in case they are used in the estimator.
        if y is not None:
            X, y = self._validate_data(X, y)
            y_pred = y[test_indices]
        else:
            X = self._validate_data(X)
            y_pred = None

        X_pred = []
        fitted_inner_estimators = iter(self.inner_estimators_)
        for cluster_ids in clusters:
            if len(cluster_ids) == 1:
                pred = X[test_indices, cluster_ids[0]]
            else:
                if cv_predictions is None:
                    fitted_inner_estimator = next(fitted_inner_estimators)
                    pred = fitted_inner_estimator.predict(X[test_indices, cluster_ids])
                else:
                    pred = next(cv_predictions)
                    if isinstance(self.cv, BaseCombinatorialCV):
                        pred = pred.quantile(
                            measure=self.quantile_measure, q=self.quantile
                        )
            X_pred.append(np.asarray(pred))
        X_pred = np.array(X_pred).T
        if cv_predictions is None:
            assert not any(
                fitted_inner_estimators
            ), "fitted_inner_estimator iterator must be empty"
        else:
            assert not any(cv_predictions), "cv_predictions iterator must be empty"

        fit_single_estimator(self.outer_estimator_, X=X_pred, y=y_pred)
        outer_weights = self.outer_estimator_.weights_
        self.weights_ = outer_weights @ inner_weights
        return self
