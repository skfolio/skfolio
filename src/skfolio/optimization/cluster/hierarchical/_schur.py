"""Schur Complementary Allocation estimator."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# Precise, Copyright (c) 2021, Peter Cotton.

# Algorithm considerations:
# We apply Tail Call Optimization (TCO), replacing recursion with iteration
# and updating the covariance in place, to reduce call-stack depth and memory usage.
# To ensure portfolio variance decreases monotonically with the regularization
# factor gamma, we identify the variance turning point and cap gamma at its maximum
# permissible value. See https://github.com/skfolio/skfolio/discussions/3

import math

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
from skfolio.optimization.cluster.hierarchical._hrp import (
    _apply_weight_constraints_to_split_factor,
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

    It uses Schur-complement-inspired augmentation of sub-covariance matrices,
    revealing a link between Hierarchical Risk Parity (HRP) and minimum-variance (MVO)
    portfolios.

    By tuning the regularization factor `gamma`, which governs how much off-diagonal
    information is incorporated into the augmented covariance blocks, the method
    smoothly interpolates from the heuristic divide-and-conquer allocation of HRP
    (`gamma = 0`) to the exact MVO solution (`gamma -> 1`).

    The algorithm begins by computing a distance matrix and performing hierarchical
    clustering, then applies seriation to reorder assets in the dendrogram so that
    adjacent leaves have minimal distance.

    Next, it uses recursive bisection: starting with the top-level cluster, each cluster
    is split into two sub-clusters in a top-down traversal.

    For each sub-cluster, an augmented covariance matrix is built based on the Schur
    complement to incorporate off-diagonal block information. From this matrix, the
    total cluster variance under an inverse-variance allocation is computed, and a
    weighting factor derived from the variances of the two sub-clusters is used to
    update their cluster weights.

    Note
    ----
    A poorly conditioned covariance matrix can produce unstable or extreme portfolio
    weights and prevent convergence to the true MVO solution as gamma approaches one.
    To improve numerical stability and estimation accuracy, apply shrinkage or other
    conditioning techniques via the `prior_estimator`.

    Parameters
    ----------
    gamma : float
        Regularization factor in [0, 1].
        When gamma is zero, no off-diagonal information is used (equivalent to HRP).
        As gamma approaches one, the allocation moves toward the minimum variance (MVO)
        solution. The better the conditioning of the initial covariance matrix, the
        closer the allocation will get to the exact MVO solution when gamma is near one.

    keep_monotonic : bool, default=True
        If True, portfolio variance is enforced to decrease monotonically
        with gamma by capping gamma at its maximum permissible value
        (`effective_gamma_`). If False, no monotonicity check or capping is applied.
        See https://github.com/skfolio/skfolio/discussions/3 for details.

    prior_estimator : BasePrior, optional
        :ref:`Prior estimator <prior>`.
        The prior estimator is used to estimate the :class:`~skfolio.prior.ReturnDistribution`
        containing the estimation of assets expected returns, covariance matrix and
        returns. The moments and returns estimations are used for the risk computation
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
        Minimum assets weights (weights lower bounds). The default is 0.0 (no short
        selling). Negative weights are not allowed. If a float is provided, it is
        applied to each asset. `None` is equivalent to the default `0.0`. If a
        dictionary is provided, its (key/value) pair must be the (asset name/asset
        minimum weight) and the input `X` of the `fit` methods must be a DataFrame with
        the asset names in columns. When using a dictionary, assets values that are not
        provided are assigned the default  minimum weight of `0.0`.

        Example:

           * `min_weights = 0.0` --> long only portfolio (default).
           * `min_weights = {"SX5E": 0.1, "SPX": 0.2}`
           * `min_weights = [0.1, 0.2]`

    max_weights : float | dict[str, float] | array-like of shape (n_assets, ), default=1.0
        Maximum assets weights (weights upper bounds). The default is 1.0 (each asset
        is below 100%). Weights above 1.0 are not allowed. If a float is provided, it is
        applied to each asset. `None` is equivalent to the default `1.0`. If a
        dictionary is provided, its (key/value) pair must be the (asset name/asset
        maximum weight) and the input `X` of the `fit` method must be a DataFrame with
        the asset names in columns. When using a dictionary, assets values that are not
        provided are assigned the default maximum weight of `1.0`.

        Example:

           * `max_weights = 1.0` --> each weight  must be below 100% (default).
           * `max_weights = 0.5` --> each weight must be below 50%.
           * `max_weights = {"SX5E": 0.8, "SPX": 0.9}`
           * `max_weights = [0.8, 0.9]`

    transaction_costs : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Transaction costs of the assets.
        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset cost) and the input `X` of the `fit` method must be a
        DataFrame with the asset names in columns.
        The default value is `0.0`.

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Management fees of the assets.
        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset fee) and the input `X` of the `fit` method must be a
        DataFrame with the asset names in columns.
        The default value is `0.0`.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Previous weights of the assets. Previous weights are used to compute the
        portfolio total cost. If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset previous weight) and the input `X` of the `fit` method must
        be a DataFrame with the asset names in columns.
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

    effective_gamma_ : float
        If `keep_monotonic` is True, the highest permissible `gamma` that preserves
        monotonic variance decrease; otherwise, equal to the input `gamma`.

    distance_estimator_ : BaseDistance
        Fitted `distance_estimator`.

    hierarchical_clustering_estimator_ : HierarchicalClustering
        Fitted `hierarchical_clustering_estimator`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has asset names that are all strings.

    References
    ----------
    .. [1] "Schur Complementary Allocation: A Unification of Hierarchical Risk Parity
       and Minimum Variance Portfolios". Peter Cotton (2024).

    .. [2] "Portfolio Optimization. Theory and Application".
        Chapter 12.3.4 "From Portfolio Risk Minimization to Hierarchical Portfolios"
        Daniel P. Palomar (2025).

    .. [3] "Building diversified portfolios that outperform out of sample",
        The Journal of Portfolio Management,
        Marcos López de Prado (2016).

    .. [4] "A robust estimator of the efficient frontier",
        SSRN Electronic Journal,
        Marcos López de Prado (2019).

    .. [5] "Machine Learning for Asset Managers",
        Elements in Quantitative Finance. Cambridge University Press,
        Marcos López de Prado (2020).

    .. [6] "A review of two decades of correlations, hierarchies, networks and
        clustering in financial markets",
        Gautier Marti, Frank Nielsen, Mikołaj Bińkowski, Philippe Donnat (2020).
    """

    effective_gamma_: float

    def __init__(
        self,
        gamma: float = 0.5,
        keep_monotonic: bool = True,
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
        self.keep_monotonic = keep_monotonic

    def fit(
        self, X: npt.ArrayLike, y: None = None, **fit_params
    ) -> "SchurComplementary":
        """Fit the Schur Complementary estimator.

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
        routed_params = skm.process_routing(self, "fit", **fit_params)

        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be between 0 and 1. Got {self.gamma}")

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

        self.distance_estimator_.fit(returns, y, **routed_params.distance_estimator.fit)
        distance = self.distance_estimator_.distance_

        # To keep the asset_names
        if isinstance(X, pd.DataFrame):
            distance = pd.DataFrame(distance, columns=X.columns)

        self.hierarchical_clustering_estimator_.fit(
            X=distance, y=None, **routed_params.hierarchical_clustering_estimator.fit
        )

        X = skv.validate_data(self, X)

        ordered_linkage_matrix = sch.optimal_leaf_ordering(
            self.hierarchical_clustering_estimator_.linkage_matrix_,
            self.hierarchical_clustering_estimator_.condensed_distance_,
        )
        sorted_assets = sch.leaves_list(ordered_linkage_matrix)

        # Prepare weight bounds
        n_assets = X.shape[1]
        min_weights, max_weights = self._convert_weights_bounds(n_assets=n_assets)

        # Compute allocations
        if self.keep_monotonic:
            self.weights_, self.effective_gamma_ = _compute_monotonic_weights(
                max_gamma=self.gamma,
                sorted_assets=sorted_assets,
                covariance=covariance,
                min_weights=min_weights,
                max_weights=max_weights,
            )
        else:
            self.weights_ = _compute_weights(
                gamma=self.gamma,
                sorted_assets=sorted_assets,
                covariance=covariance,
                min_weights=min_weights,
                max_weights=max_weights,
                check_spd=False,
            )
            self.effective_gamma_ = self.gamma

        return self


def _compute_monotonic_weights(
    max_gamma: float,
    sorted_assets: np.ndarray,
    covariance: np.ndarray,
    max_weights: np.ndarray,
    min_weights: np.ndarray,
    step: float = 0.1,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """
    Sweep gamma to find the monotonic "turning point" where risk stops decreasing.
    Returns weights and effective gamma.
    """
    if max_gamma == 0:
        weights = _compute_weights(
            gamma=0,
            sorted_assets=sorted_assets,
            covariance=covariance,
            max_weights=max_weights,
            min_weights=min_weights,
            check_spd=False,
        )
        return weights, 0.0

    def objective(x: float) -> tuple[float, np.ndarray | None]:
        w = _compute_weights(
            gamma=x,
            sorted_assets=sorted_assets,
            covariance=covariance,
            max_weights=max_weights,
            min_weights=min_weights,
            check_spd=True,
        )
        risk = np.inf if w is None else w @ covariance @ w.T
        return risk, w

    # Evenly spaced gamma vector in [0, max_gamma]
    n = int(np.ceil(max_gamma / step)) + 1
    gammas = np.linspace(0, max_gamma, n)
    variances = np.full_like(gammas, np.nan)

    # Initial sweep of the discrete gamma vector in [0, max_gamma] to find the range
    # of the variance turning point if any.
    variance, weights = objective(gammas[0])
    variances[0] = variance
    for i in range(1, n):
        variance, weights = objective(gammas[i])
        variances[i] = variance
        if variance >= variances[i - 1]:
            # Turning point lies in [gammas[i-2], gammas[i]], we find the exact
            # turning point by binary search
            low_idx = max(i - 2, 0)
            return _binary_search(
                objective,
                low_gamma=gammas[low_idx],
                high_gamma=gammas[i],
                low_variance=variances[low_idx],
                tol=tol,
            )

    # No turning point found in sweep: check local derivative at the terminal gamma
    variance_h = objective(max_gamma - tol)[0]
    if variance <= variance_h:
        # monotonically decreasing up to max_gamma
        return weights, max_gamma

    # Turning point lies between last two gammas, we find the exact turning point by
    # binary search
    return _binary_search(
        objective,
        low_gamma=gammas[-2],
        high_gamma=max_gamma,
        low_variance=variances[-2],
        tol=tol,
    )


def _binary_search(
    objective,
    low_gamma: float,
    high_gamma: float,
    low_variance: float,
    tol: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Locate the turning point inside the interval [low_gamma, high_gamma]."""
    max_iter = math.ceil(math.log2((high_gamma - low_gamma) / tol) * 2 + 1)

    for i in range(max_iter):
        mid_gamma = 0.5 * (low_gamma + high_gamma)
        variance, weights = objective(mid_gamma)
        variance_h = objective(mid_gamma - tol)[0]

        if variance <= low_variance and variance <= variance_h:
            low_gamma = mid_gamma
            low_variance = variance
            if (high_gamma - low_gamma) <= tol:
                print(i)
                return weights, low_gamma
        else:
            high_gamma = mid_gamma

    raise RuntimeError(
        "Unable to find a permissible regularization factor `gamma` for which "
        "the portfolio variance decreases monotonically as a function of gamma. "
        "This can occur when the specified weight bounds are overly restrictive."
    )


def _compute_weights(
    gamma: float,
    sorted_assets: np.ndarray,
    covariance: np.ndarray,
    max_weights: np.ndarray,
    min_weights: np.ndarray,
    check_spd: bool = True,
) -> np.ndarray | None:
    """
    Core Schur-complement allocation recursion.

    Parameters
    ----------
    gamma : float
        Regularization factor.

    sorted_assets : ndarray of shape (n_assets,)
        Asset indices in dendrogram order.

    covariance : ndarray of shape (n_assets, n_assets)
        Asset covariance matrix.

    min_weights : ndarray of shape (n_assets,)
        Minimum weights array.

    max_weights : ndarray of shape (n_assets,)
        Maximum weights array.

    check_spd : bool
        Return None if any augmented block is not SPD.

    Returns
    -------
    weights : ndarray | None
        Final portfolio weights, or None if SPD check fails.
    """
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

            if check_spd:
                if not is_cholesky_dec(a_aug) or not is_cholesky_dec(d_aug):
                    return None

            left_variance = _naive_portfolio_variance(a_aug)
            right_variance = _naive_portfolio_variance(d_aug)

            alpha = 1 - left_variance / (left_variance + right_variance)

            # Weights constraints
            alpha = _apply_weight_constraints_to_split_factor(
                alpha=alpha,
                weights=weights,
                max_weights=max_weights,
                min_weights=min_weights,
                left_cluster=left_cluster,
                right_cluster=right_cluster,
            )

            weights[left_cluster] *= alpha
            weights[right_cluster] *= 1 - alpha

        items = new_items

    return weights


def _naive_portfolio_variance(covariance: np.ndarray) -> float:
    """Portfolio variance of an inverse variance allocation.

    Parameters
    ----------
    covariance : ndarray of shape (n_assets, n_assets)
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
