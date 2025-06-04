"""Schur Complementary Allocation estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Precise, Copyright (c) 2021, Peter Cotton.

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.cluster.hierarchy as sch
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio.cluster import HierarchicalClustering
from skfolio.distance import BaseDistance, PearsonDistance
from skfolio.optimization.cluster.hierarchical._base import (
    BaseHierarchicalOptimization,
)
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.utils.stats import (
    cov_nearest,
    inverse_multiply,
    is_cholesky_dec,
    multiply_by_inverse,
    symmetric_step_up_matrix,
)
from skfolio.utils.tools import bisection, check_estimator


class SchurComplementary(BaseHierarchicalOptimization):
    r"""Schur Complementary Allocation estimator.

    Schur Complementary Allocation is a portfolio allocation method developed by Peter
    Cotton [1]_.

    This algorithm uses a distance matrix to compute hierarchical clusters using the
    Hierarchical Tree Clustering algorithm. It then employs seriation to rearrange the
    assets in the dendrogram, minimizing the distance between leafs.

    The final step is the recursive bisection where each cluster is split between two
    sub-clusters by starting with the topmost cluster and traversing in a top-down
    manner.

    For each sub-cluster, we compute an augmented covariance matrix inspired by the
    Schur complement where additional information is used from off-diagonal
    matrix blocks. Based on this augmented covariance matrix, we calculate the total
    cluster variance of an inverse-variance allocation. A weighting factor is then
    computed from these two sub-cluster variance, which is used to update the cluster
    weight.

    The amount of off-diagonal matrix blocks information used is controlled by the
    regularization factor `gamma`.

    Parameters
    ----------
    gamma : float
        Regularization factor between 0 and 1.
        A value of 0 means that no additional information is used from off-diagonal
        matrix blocks and is equivalent to a Hierarchical Risk Parity.
        As the value increases to 1, the allocation tends to the Minimum Variance
        Optimization allocation.

    prior_estimator : BasePrior, optional
        :ref:`Prior estimator <prior>`.
        The prior estimator is used to estimate the :class:`~skfolio.prior.PriorModel`
        containing the estimation of the covariance matrix and returns.
        The moments and returns estimations are used for the risk computation
        and the returns estimation are used by the distance matrix estimator.
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    distance_estimator : BaseDistance, optional
        :ref:`Distance estimator <distance>`.
        The distance estimator is used to estimate the codependence and the distance
        matrix needed for the computation of the linkage matrix.
        The default (`None`) is to use :class:`~skfolio.distance.PearsonDistance`.

    hierarchical_clustering_estimator : HierarchicalClustering, optional
        :ref:`Hierarchical Clustering estimator <hierarchical_clustering>`.
        The hierarchical clustering estimator is used to compute the linkage matrix
        and the hierarchical clustering of the assets based on the distance matrix.
        The default (`None`) is to use
        :class:`~skfolio.cluster.HierarchicalClustering`.

    min_weights : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Minimum assets weights (weights lower bounds). Negative weights are not allowed.
        If a float is provided, it is applied to each asset. `None` is equivalent to
        `-np.Inf` (no lower bound). If a dictionary is provided, its (key/value) pair
        must be the (asset name/asset minium weight) and the input `X` of the `fit`
        methods must be a DataFrame with the assets names in columns. When using a
        dictionary, assets values that are not provided are assigned a minimum weight
        of `0.0`. The default is 0.0 (no short selling).

        Example:

           * min_weights = 0 --> long only portfolio (no short selling).
           * min_weights = None --> no lower bound (same as `-np.Inf`).
           * min_weights = {"SX5E": 0, "SPX": 0.1}
           * min_weights = [0, 0.1]

    max_weights : float | dict[str, float] | array-like of shape (n_assets, ), default=1.0
        Maximum assets weights (weights upper bounds). Weights above 1.0 are not
        allowed. If a float is provided, it is applied to each asset. `None` is
        equivalent to `+np.Inf` (no upper bound). If a dictionary is provided, its
        (key/value) pair must be the (asset name/asset maximum weight) and the input `X`
        of the `fit` method must be a DataFrame with the assets names in columns. When
        using a dictionary, assets values that are not provided are assigned a minimum
        weight of `1.0`. The default is 1.0 (each asset is below 100%).

        Example:

           * max_weights = 0 --> no long position (short only portfolio).
           * max_weights = 0.5 --> each weight must be below 50%.
           * max_weights = {"SX5E": 1, "SPX": 0.25}
           * max_weights = [1, 0.25]

    transaction_costs : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Transaction costs of the assets.
        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset cost) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.
        The default value is `0.0`.

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Management fees of the assets.
        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset fee) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.
        The default value is `0.0`.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Previous weights of the assets. Previous weights are used to compute the
        portfolio total cost. If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset previous weight) and the input `X` of the `fit` method must
        be a DataFrame with the assets names in columns.
        The default (`None`) means no previous weights.

    portfolio_params :  dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name`, `transaction_costs`,
        `management_fees` and `previous_weights` are copied from the optimization
        model and systematically passed to the portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Weights of the assets.

    distance_estimator_ : BaseDistance
        Fitted `distance_estimator`.

    hierarchical_clustering_estimator_ : HierarchicalClustering
        Fitted `hierarchical_clustering_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

    References
    ----------
    .. [1] "Schur Complementary Portfolios - A Unification of Machine Learning and
        Optimization-Based Allocation".
        Peter Cotton (2022).

    .. [2] "Building diversified portfolios that outperform out of sample",
        The Journal of Portfolio Management,
        Marcos López de Prado (2016).

    .. [3] "A robust estimator of the efficient frontier",
        SSRN Electronic Journal,
        Marcos López de Prado (2019).

    .. [4] "Machine Learning for Asset Managers",
        Elements in Quantitative Finance. Cambridge University Press,
        Marcos López de Prado (2020).

    .. [5] "A review of two decades of correlations, hierarchies, networks and
        clustering in financial markets",
        Gautier Marti, Frank Nielsen, Mikołaj Bińkowski, Philippe Donnat (2020).
    """

    def __init__(
        self,
        gamma: float = 0.5,
        min_cluster_size: int = 2,
        prior_estimator: BasePrior | None = None,
        distance_estimator: BaseDistance | None = None,
        hierarchical_clustering_estimator: HierarchicalClustering | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
        portfolio_params: dict | None = None,
    ):
        super().__init__(
            prior_estimator=prior_estimator,
            distance_estimator=distance_estimator,
            hierarchical_clustering_estimator=hierarchical_clustering_estimator,
            min_weights=min_weights,
            max_weights=max_weights,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            previous_weights=previous_weights,
            portfolio_params=portfolio_params,
        )
        self.gamma = gamma
        self.min_cluster_size = min_cluster_size

    def fit(
        self, X: npt.ArrayLike, y: None = None, **fit_params
    ) -> "SchurComplementary":
        """Fit the Schur Complementary Allocation estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : SchurComplementary
            Fitted estimator.
        """
        # Algorithm considerations:
        # We apply TCO (Tail Call Optimisation): the recursion is replaced by an
        # iteration and inplace covariance update to reduce the call stack and
        # space complexity.
        # Binary search on gamma is applied to both matrix A and D at the same time and
        # symmetrization is applied at the end of the schur augmentation. This seems
        # to improve the stability of the solution as gamma tends to 1.
        routed_params = skm.process_routing(self, "fit", **fit_params)

        # Validate
        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        self.distance_estimator_ = check_estimator(
            self.distance_estimator,
            default=PearsonDistance(),
            check_type=BaseDistance,
        )
        self.hierarchical_clustering_estimator_ = check_estimator(
            self.hierarchical_clustering_estimator,
            default=HierarchicalClustering(),
            check_type=HierarchicalClustering,
        )

        # Fit the estimators
        self.prior_estimator_.fit(X, y, **routed_params.prior_estimator.fit)
        return_distribution = self.prior_estimator_.return_distribution_
        returns = return_distribution.returns
        covariance = cov_nearest(return_distribution.covariance)

        # To keep the asset_names
        if isinstance(X, pd.DataFrame):
            returns = pd.DataFrame(returns, columns=X.columns)

        # noinspection PyArgumentList
        self.distance_estimator_.fit(returns, y, **routed_params.distance_estimator.fit)
        distance = self.distance_estimator_.distance_

        # To keep the asset_names
        if isinstance(X, pd.DataFrame):
            distance = pd.DataFrame(distance, columns=X.columns)

        # noinspection PyArgumentList
        self.hierarchical_clustering_estimator_.fit(
            X=distance, y=None, **routed_params.hierarchical_clustering_estimator.fit
        )

        X = skv.validate_data(self, X)
        n_assets = X.shape[1]

        min_weights, max_weights = self._convert_weights_bounds(n_assets=n_assets)

        ordered_linkage_matrix = sch.optimal_leaf_ordering(
            self.hierarchical_clustering_estimator_.linkage_matrix_,
            self.hierarchical_clustering_estimator_.condensed_distance_,
        )
        sorted_assets = sch.leaves_list(ordered_linkage_matrix)

        self.weights_ = _compute_monotonic_weights(
            max_gamma=self.gamma,
            sorted_assets=sorted_assets,
            covariance=covariance,
            tol=1e-5,
            maxiter=100,
        )

        return self


def _compute_monotonic_weights(
    max_gamma: float,
    sorted_assets: np.ndarray,
    covariance: np.ndarray,
    tol=1e-4,
    maxiter=30,
) -> np.ndarray:
    if max_gamma == 0:
        return _compute_weights(
            gamma=0,
            sorted_assets=sorted_assets,
            covariance=covariance,
        )

    def _func(x) -> tuple[np.ndarray, float]:
        weights = _compute_weights(
            gamma=x,
            sorted_assets=sorted_assets,
            covariance=covariance,
        )
        if weights is None:
            return weights, np.inf
        return weights, weights @ covariance @ weights.T

    step = 0.1

    weights, prev_variance = _func(0)
    prev_gamma = 0
    for gamma in np.linspace(0, max_gamma, int(np.ceil((max_gamma) / step)) + 1)[1:]:
        weights, variance = _func(gamma)
        if variance >= prev_variance:
            break
        valid_weights = weights
        prev_variance = variance
        prev_gamma = gamma
    else:
        return weights

    low_variance = prev_variance
    low = prev_gamma
    high = gamma

    for _ in range(maxiter):
        mid = 0.5 * (low + high)
        weights, variance = _func(mid)

        if variance <= low_variance:
            low = mid
            low_variance = variance
            valid_weights = weights
        else:
            high = mid
        if (high - low) <= tol:
            break

    return valid_weights


def _compute_weights(
    gamma: float,
    sorted_assets: np.ndarray,
    covariance: np.ndarray,
) -> np.ndarray | None:
    covariance = covariance.copy()

    n_assets = len(covariance)
    weights = np.ones(n_assets)
    items = [sorted_assets]
    while len(items) > 0:
        new_items = []

        for left_cluster, right_cluster in bisection(items):
            new_items += [left_cluster, right_cluster]

            a = covariance[np.ix_(left_cluster, left_cluster)]
            d = covariance[np.ix_(right_cluster, right_cluster)]

            if len(left_cluster) <= 1:
                a_aug, d_aug = a, d
            else:
                b = covariance[np.ix_(left_cluster, right_cluster)]
                a_aug = _schur_augmentation(a, b, d, gamma=gamma)
                d_aug = _schur_augmentation(d, b.T, a, gamma=gamma)

                covariance[np.ix_(left_cluster, left_cluster)] = a_aug
                covariance[np.ix_(right_cluster, right_cluster)] = d_aug

            if not is_cholesky_dec(a_aug) or not is_cholesky_dec(d_aug):
                return None

            left_variance = _naive_portfolio_variance(a_aug)
            right_variance = _naive_portfolio_variance(d_aug)

            alpha = 1 - left_variance / (left_variance + right_variance)

            weights[left_cluster] *= alpha
            weights[right_cluster] *= 1 - alpha

        items = new_items

    return weights


def _naive_portfolio_variance(covariance: np.ndarray) -> float:
    """Portfolio variance of an inverse variance allocation.

    Parameters
    ----------
    covariance : ndarray of shape (n, n)
        Covariance matrix.

    Returns
    -------
    variance : float
        Portfolio variance of an inverse variance allocation.
    """
    weights = 1 / np.diag(covariance)
    weights /= weights.sum()
    variance = weights @ covariance @ weights.T
    return variance


def _schur_augmentation(
    a: np.ndarray, b: np.ndarray, d: np.ndarray, gamma: float
) -> np.ndarray:
    """Compute an augmented covariance matrix `A` inspired by the
    Schur complement [1]_.

    Parameters
    ----------
    a : ndarray of shape (n1, n1)
        Upper left block matrix `A`

    b : ndarray of shape (n1, n2)
        Upper right block matrix `B`

    d : ndarray of shape (n2, n2)
        Lower right block matrix `D`

    gamma : float
        Regularization factor between 0 and 1.
        A value of 0 means that no additional information is used from off-diagonal
        matrix blocks and is equivalent to a Hierarchical Risk Parity.
        As the value increases to 1, the allocation tends to the Minimum Variance
        Optimization allocation.

    Returns
    -------
    a_aug : ndarray of shape (n1, n1)
        Augmented covariance matrix `A`.

    References
    ----------
    .. [1] "Schur Complementary Portfolios - A Unification of Machine Learning and
        Optimization-Based Allocation".
        Peter Cotton (2022).
    """
    n_a = a.shape[0]
    n_d = d.shape[0]

    if gamma == 0 or n_a == 1 or n_d == 1:
        return a

    a_aug = a - gamma * b @ inverse_multiply(d, b.T)
    m = symmetric_step_up_matrix(n1=n_a, n2=n_d)
    r = np.eye(n_a) - gamma * multiply_by_inverse(b, d) @ m.T
    a_aug = inverse_multiply(r, a_aug)
    # make it symmetric
    a_aug = (a_aug + a_aug.T) / 2.0
    return a_aug
