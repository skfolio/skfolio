"""Hierarchical Risk Parity Optimization estimator."""

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
from skfolio.utils.tools import bisection, check_estimator


class HierarchicalRiskParity(BaseHierarchicalOptimization):
    r"""Hierarchical Risk Parity estimator.

    Hierarchical Risk Parity is a portfolio optimization method developed by Marcos
    Lopez de Prado [2]_.

    This algorithm uses a distance matrix to compute hierarchical clusters using the
    Hierarchical Tree Clustering algorithm. It then employs seriation to rearrange the
    assets in the dendrogram, minimizing the distance between leafs.

    The final step is the recursive bisection where each cluster is split between two
    sub-clusters by starting with the topmost cluster and traversing in a top-down
    manner. For each sub-cluster, we compute the total cluster risk of an inverse-risk
    allocation. A weighting factor is then computed from these two sub-cluster risks,
    which is used to update the cluster weight.

    .. note ::
        The original paper uses the variance as the risk measure and the single-linkage
        method for the Hierarchical Tree Clustering algorithm. Here we generalize it to
        multiple risk measures and linkage methods.
        The default linkage method is set to the Ward
        variance minimization algorithm, which is more stable and has better properties
        than the single-linkage method [4]_.

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
        The float :math:`total\_cost` is impacting the portfolio expected return in the optimization:

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
        The float :math:`total\_fee` is impacting the portfolio expected return in the optimization:

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
    .. [1] "Building diversified portfolios that outperform out of sample",
        The Journal of Portfolio Management,
        Marcos López de Prado (2016).

    .. [2] "A robust estimator of the efficient frontier",
        SSRN Electronic Journal,
        Marcos López de Prado (2019).

    .. [3] "Machine Learning for Asset Managers",
        Elements in Quantitative Finance. Cambridge University Press,
        Marcos López de Prado (2020).

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

    def fit(self, X: npt.ArrayLike, y: None = None) -> "HierarchicalRiskParity":
        """Fit the Hierarchical Risk Parity Optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : HierarchicalRiskParity
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

        X = self._validate_data(X)
        n_assets = X.shape[1]

        min_weights, max_weights = self._convert_weights_bounds(n_assets=n_assets)
        assets_risks = self._unitary_risks(prior_model=prior_model)

        ordered_linkage_matrix = sch.optimal_leaf_ordering(
            self.hierarchical_clustering_estimator_.linkage_matrix_,
            self.hierarchical_clustering_estimator_.condensed_distance_,
        )
        sorted_assets = sch.leaves_list(ordered_linkage_matrix)

        weights = np.ones(n_assets)
        items = [sorted_assets]

        while len(items) > 0:
            new_items = []
            for clusters_ids in bisection(items):
                new_items += clusters_ids
                risks = []
                for ids in clusters_ids:
                    inv_risk_w = np.zeros(n_assets)
                    inv_risk_w[ids] = 1 / assets_risks[ids]
                    inv_risk_w /= inv_risk_w.sum()
                    risks.append(
                        self._risk(weights=inv_risk_w, prior_model=prior_model)
                    )
                left_risk, right_risk = risks
                left_cluster, right_cluster = clusters_ids
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
                weights[left_cluster] *= alpha
                weights[right_cluster] *= 1 - alpha
            items = new_items

        self.weights_ = weights
        return self
