"""Hierarchical Equal Risk Contribution estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# The risk measure generalization and constraint features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.cluster.hierarchy as sch

import skfolio.typing as skt
from skfolio.cluster import HierarchicalClustering
from skfolio.distance import BaseDistance, PearsonDistance
from skfolio.measures import ExtraRiskMeasure, RiskMeasure
from skfolio.optimization.cluster.hierarchical._base import (
    BaseHierarchicalOptimization,
)
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.utils.tools import check_estimator


class HierarchicalEqualRiskContribution(BaseHierarchicalOptimization):
    r"""Hierarchical Equal Risk Contribution estimator.

    The Hierarchical Equal Risk Contribution is a portfolio optimization method
    developed by Thomas Raffinot [2]_.

    This algorithm uses a distance matrix to compute hierarchical clusters using the
    Hierarchical Tree Clustering algorithm. It then computes, for each cluster, the
    total cluster risk of an inverse-risk allocation.

    The final step is the top-down recursive division of the dendrogram, where the
    assets weights are updated using a naive risk parity within clusters.

    It differs from the Hierarchical Risk Parity by exploiting the dendrogram shape
    during the top-down recursive division instead of bisecting it.

    .. note ::

        The default linkage method is set to the Ward variance minimization algorithm,
        which is more stable and has better properties than the single-linkage
        method [4]_.

    Parameters
    ----------
    risk_measure : RiskMeasure or ExtraRiskMeasure, default=RiskMeasure.VARIANCE
        :class:`~skfolio.meta.RiskMeasure` or :class:`~skfolio.meta.ExtraRiskMeasure`
        of the optimization.
        Can be any of:

            * MEAN_ABSOLUTE_DEVIATION
            * FIRST_LOWER_PARTIAL_MOMENT
            * VARIANCE
            * SEMI_VARIANCE
            * CVAR
            * EVAR
            * WORST_REALIZATION
            * CDAR
            * MAX_DRAWDOWN
            * AVERAGE_DRAWDOWN
            * EDAR
            * ULCER_INDEX
            * GINI_MEAN_DIFFERENCE_RATIO
            * VALUE_AT_RISK
            * DRAWDOWN_AT_RISK
            * ENTROPIC_RISK_MEASURE
            * FOURTH_CENTRAL_MOMENT
            * FOURTH_LOWER_PARTIAL_MOMENT
            * SKEW
            * KURTOSIS

        The default is `RiskMeasure.VARIANCE`.

    prior_estimator : BasePrior, optional
        :ref:`Prior estimator <prior>`.
        The prior estimator is used to estimate the :class:`~skfolio.prior.PriorModel`
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
        Transaction costs of the assets. It is used to add linear transaction costs to
        the optimization problem:

        .. math:: total\_cost = \sum_{i=1}^{N} c_{i} \times |w_{i} - w\_prev_{i}|

        with :math:`c_{i}` the transaction cost of asset i, :math:`w_{i}` its weight
        and :math:`w\_prev_{i}` its previous weight (defined in `previous_weights`).
        The float :math:`total\_cost` is impacting the portfolio expected return in the
        optimization:

        .. math:: expected\_return = \mu^{T} \cdot w - total\_cost

        with :math:`\mu` the vector af assets' expected returns and :math:`w` the
        vector of assets weights.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset cost) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.
        The default value is `0.0`.

        .. warning::

            Based on the above formula, the periodicity of the transaction costs
            needs to be homogenous to the periodicity of :math:`\mu`. For example, if
            the input `X` is composed of **daily** returns, the `transaction_costs` need
            to be expressed as **daily** costs.
            (See :ref:`sphx_glr_auto_examples_1_mean_risk_plot_6_transaction_costs.py`)

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Management fees of the assets. It is used to add linear management fees to the
        optimization problem:

        .. math:: total\_fee = \sum_{i=1}^{N} f_{i} \times w_{i}

        with :math:`f_{i}` the management fee of asset i and :math:`w_{i}` its weight.
        The float :math:`total\_fee` is impacting the portfolio expected return in the
        optimization:

        .. math:: expected\_return = \mu^{T} \cdot w - total\_fee

        with :math:`\mu` the vector af assets expected returns and :math:`w` the vector
        of assets weights.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset fee) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.
        The default value is `0.0`.

        .. warning::

            Based on the above formula, the periodicity of the management fees needs to
            be homogenous to the periodicity of :math:`\mu`. For example, if the input
            `X` is composed of **daily** returns, the `management_fees` need to be
            expressed in **daily** fees.

        .. note::

            Another approach is to directly impact the management fees to the input `X`
            in order to express the returns net of fees. However, when estimating the
            :math:`\mu` parameter using for example Shrinkage estimators, this approach
            would mix a deterministic value with an uncertain one leading to unwanted
            bias in the management fees.

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
        `management_fees`, `previous_weights` and `risk_free_rate` are copied from the
        optimization model and passed to the portfolio.

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
    .. [1]  "Hierarchical clustering-based asset allocation",
        The Journal of Portfolio Management,
        Thomas Raffinot  (2017).

    .. [2] "The hierarchical equal risk contribution portfolio",
        Thomas Raffinot (2018).

    .. [3] "Application of two-order difference to gap statistic".
        Yue, Wang & Wei (2009).

    .. [4] "A review of two decades of correlations, hierarchies, networks and
        clustering in financial markets",
        Gautier Marti, Frank Nielsen, Mikołaj Bińkowski, Philippe Donnat (2020).
    """

    def __init__(
        self,
        risk_measure: RiskMeasure | ExtraRiskMeasure = RiskMeasure.VARIANCE,
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
            risk_measure=risk_measure,
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

    def fit(
        self, X: npt.ArrayLike, y: None = None
    ) -> "HierarchicalEqualRiskContribution":
        """Fit the Hierarchical Equal Risk Contribution estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : HierarchicalEqualRiskContribution
            Fitted estimator.
        """
        # Validate
        if not isinstance(self.risk_measure, RiskMeasure | ExtraRiskMeasure):
            raise TypeError(
                "`risk_measure` must be of type `RiskMeasure` or `ExtraRiskMeasure`"
            )
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
        self.prior_estimator_.fit(X, y)
        prior_model = self.prior_estimator_.prior_model_
        returns = prior_model.returns

        # To keep the asset_names
        if isinstance(X, pd.DataFrame):
            returns = pd.DataFrame(returns, columns=X.columns)

        self.distance_estimator_.fit(returns)
        distance = self.distance_estimator_.distance_

        # To keep the asset_names
        if isinstance(X, pd.DataFrame):
            distance = pd.DataFrame(distance, columns=X.columns)

        self.hierarchical_clustering_estimator_.fit(distance)

        n_clusters = self.hierarchical_clustering_estimator_.n_clusters_
        labels = self.hierarchical_clustering_estimator_.labels_
        linkage_matrix = self.hierarchical_clustering_estimator_.linkage_matrix_

        X = self._validate_data(X)
        n_assets = X.shape[1]

        min_weights, max_weights = self._convert_weights_bounds(n_assets=n_assets)

        assets_risks = self._unitary_risks(prior_model=prior_model)
        weights = np.ones(n_assets)
        clusters_weights = np.ones(n_clusters)

        clusters = [np.argwhere(labels == i).flatten() for i in range(n_clusters)]
        clusters_sets = [set(cluster_ids) for cluster_ids in clusters]

        # Compute cluster total risk based on inverse-risk allocation
        cluster_risks = []
        for cluster_ids in clusters:
            inv_risk_w = np.zeros(n_assets)
            inv_risk_w[cluster_ids] = 1 / assets_risks[cluster_ids]
            inv_risk_w /= inv_risk_w.sum()
            cluster_risks.append(
                self._risk(weights=inv_risk_w, prior_model=prior_model)
            )
            weights[cluster_ids] = inv_risk_w[cluster_ids]
        cluster_risks = np.array(cluster_risks)

        # Compute the cluster weights using the dendrogram structure.
        # Recurse from the root until each of the defined cluster is reached and
        # update the weights using the naive risk parity.
        def _recurse(node):
            # Stop when the cluster is reached
            if set(node.pre_order()) in clusters_sets:
                return

            left_node = node.get_left()
            right_node = node.get_right()
            left_cluster_tree = set(left_node.pre_order())
            right_cluster_tree = set(right_node.pre_order())

            left_cluster = []
            right_cluster = []
            for i, cluster_ids in enumerate(clusters_sets):
                if cluster_ids.issubset(left_cluster_tree):
                    left_cluster.append(i)
                elif cluster_ids.issubset(right_cluster_tree):
                    right_cluster.append(i)

            if not left_cluster or not right_cluster:
                raise ValueError("Corrupted")

            left_cluster = np.array(left_cluster)
            right_cluster = np.array(right_cluster)
            left_risk = np.sum(cluster_risks[left_cluster])
            right_risk = np.sum(cluster_risks[right_cluster])

            alpha = 1 - left_risk / (left_risk + right_risk)

            # Weights constraints
            alpha = self._apply_weight_constraints_to_alpha(
                alpha=alpha,
                weights=weights,
                max_weights=max_weights,
                min_weights=min_weights,
                left_cluster=left_cluster,
                right_cluster=right_cluster,
            )

            clusters_weights[left_cluster] *= alpha
            clusters_weights[right_cluster] *= 1 - alpha

            _recurse(left_node)
            _recurse(right_node)

        root = sch.to_tree(linkage_matrix)
        _recurse(root)

        # Combine intra-cluster weights with inter-cluster weights
        for i, cluster_ids in enumerate(clusters):
            weights[cluster_ids] *= clusters_weights[i]

        self.weights_ = weights
        return self
