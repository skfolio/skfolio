"""Base Hierarchical Clustering Optimization estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt

import skfolio.typing as skt
from skfolio.cluster import HierarchicalClustering
from skfolio.distance import BaseDistance
from skfolio.measures import ExtraRiskMeasure, RiskMeasure
from skfolio.optimization._base import BaseOptimization
from skfolio.portfolio import Portfolio
from skfolio.prior import BasePrior, PriorModel
from skfolio.utils.tools import input_to_array


class BaseHierarchicalOptimization(BaseOptimization, ABC):
    r"""Base Hierarchical Clustering Optimization estimator.

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

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    distance_estimator_ : BaseDistance
        Fitted `distance_estimator`.

    hierarchical_clustering_estimator_ : HierarchicalClustering
        Fitted `hierarchical_clustering_estimator`.
    """

    prior_estimator_: BasePrior
    distance_estimator_: BaseDistance
    hierarchical_clustering_estimator_: HierarchicalClustering

    @abstractmethod
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
        super().__init__(portfolio_params=portfolio_params)
        self.risk_measure = risk_measure
        self.prior_estimator = prior_estimator
        self.distance_estimator = distance_estimator
        self.hierarchical_clustering_estimator = hierarchical_clustering_estimator
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees
        self.previous_weights = previous_weights
        self._seriated = False

    def _clean_input(
        self,
        value: float | dict | np.ndarray | list,
        n_assets: int,
        fill_value: Any,
        name: str,
    ) -> np.ndarray:
        """Convert input to cleaned 1D array
         value : float, dict, array-like or None.
            Input value to clean and convert.

        Parameters
        ----------
        value : float, dict or array-like.
            Input value to clean.

        n_assets : int
            Number of assets. Used to verify the shape of the converted array.

        fill_value : Any
            When `items` is a dictionary, elements that are not in `asset_names` are
            filled with `fill_value` in the converted array.

        name : str
            Name used for error messages.

        Returns
        -------
        value :  ndarray of shape (n_assets,)
            The cleaned float or 1D array.
        """
        if value is None:
            raise ValueError("Cannot convert None to array")
        if np.isscalar(value):
            return value * np.ones(n_assets)
        return input_to_array(
            items=value,
            n_assets=n_assets,
            fill_value=fill_value,
            dim=1,
            assets_names=(
                self.feature_names_in_ if hasattr(self, "feature_names_in_") else None
            ),
            name=name,
        )

    def _risk(
        self,
        weights: np.ndarray,
        prior_model: PriorModel,
    ) -> float:
        """Compute the risk measure of a theoretical portfolio defined by the weights
        vector.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
           The vector of weights.

        prior_model : PriorModel
            The prior model of the assets distribution.

        Returns
        -------
        risk: float
            The risk measure of a theoretical portfolio defined by the weights
            vector.
        """
        ptf = Portfolio(
            X=prior_model.returns,
            weights=weights,
            transaction_costs=self.transaction_costs,
            management_fees=self.management_fees,
            previous_weights=self.previous_weights,
        )
        if self.risk_measure in [RiskMeasure.VARIANCE, RiskMeasure.STANDARD_DEVIATION]:
            risk = ptf.variance_from_assets(assets_covariance=prior_model.covariance)
            if self.risk_measure == RiskMeasure.STANDARD_DEVIATION:
                risk = np.sqrt(risk)
        else:
            risk = getattr(ptf, str(self.risk_measure.value))
        return risk

    def _unitary_risks(self, prior_model: PriorModel) -> np.ndarray:
        """Compute the vector of risk measure for each single assets.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distribution.

        Returns
        -------
        values: ndarray of shape (n_assets,)
            The risk measure of each asset.
        """
        n_assets = prior_model.returns.shape[1]
        risks = [
            self._risk(weights=weights, prior_model=prior_model)
            for weights in np.identity(n_assets)
        ]
        return np.array(risks)

    def _convert_weights_bounds(self, n_assets: int) -> tuple[np.ndarray, np.ndarray]:
        """Convert the input weights lower and upper bounds to two 1D arrays.

        Parameters
        ----------
        n_assets : int
            Number of assets.

        Returns
        -------
        min_weights : ndarray of shape (n_assets,)
            The weight lower bound 1D array.
        max_weights : ndarray of shape (n_assets,)
            The weight upper bound 1D array.
        """

        if self.min_weights is None:
            min_weights = np.zeros(n_assets)
        else:
            min_weights = self._clean_input(
                self.min_weights,
                n_assets=n_assets,
                fill_value=0,
                name="min_weights",
            )
            if np.any(min_weights < 0):
                raise ValueError("`min_weights` must be strictly positive")

        if self.max_weights is None:
            max_weights = np.ones(n_assets)
        else:
            max_weights = self._clean_input(
                self.max_weights,
                n_assets=n_assets,
                fill_value=1,
                name="max_weights",
            )
            if np.any(max_weights > 1):
                raise ValueError("`max_weights` must be less than or equal to 1.0")
            if np.sum(max_weights) < 1:
                raise ValueError(
                    "The sum of `max_weights` must be greater than or equal to 1.0"
                )

        if np.any(min_weights > max_weights):
            raise NameError(
                "Items of `min_weights` must be less than or equal to items of"
                " `max_weights`"
            )

        return min_weights, max_weights

    @staticmethod
    def _apply_weight_constraints_to_alpha(
        alpha: float,
        max_weights: np.ndarray,
        min_weights: np.ndarray,
        weights: np.ndarray,
        left_cluster: np.ndarray,
        right_cluster: np.ndarray,
    ) -> float:
        """Apply weight constraints to the alpha multiplication factor of the
        Hierarchical Tree Clustering algorithm.

        Parameters
        ----------
        alpha : float
            The alpha multiplication factor of the Hierarchical Tree Clustering
            algorithm.

         min_weights : ndarray of shape (n_assets,)
            The weight lower bound 1D array.

        max_weights : ndarray of shape (n_assets,)
            The weight upper bound 1D array.

        weights : np.ndarray of shape (n_assets,)
            The assets weights.

        left_cluster : ndarray of shape (n_left_cluster,)
            Indices of the left cluster weights.

        right_cluster : ndarray of shape (n_right_cluster,)
            Indices of the right cluster weights.

        Returns
        -------
        value : float
            The transformed alpha incorporating the weight constraints.
        """
        alpha = min(
            np.sum(max_weights[left_cluster]) / weights[left_cluster[0]],
            max(np.sum(min_weights[left_cluster]) / weights[left_cluster[0]], alpha),
        )
        alpha = 1 - min(
            np.sum(max_weights[right_cluster]) / weights[right_cluster[0]],
            max(
                np.sum(min_weights[right_cluster]) / weights[right_cluster[0]],
                1 - alpha,
            ),
        )
        return alpha

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: None = None):
        pass
