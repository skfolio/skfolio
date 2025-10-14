"""Base Hierarchical Clustering Optimization estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# Implementation derived from:
# Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
# scikit-learn, Copyright (c) 2007-2010 David Cournapeau, Fabian Pedregosa, Olivier

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm

import skfolio.typing as skt
from skfolio.cluster import HierarchicalClustering
from skfolio.distance import BaseDistance
from skfolio.measures import ExtraRiskMeasure, RiskMeasure
from skfolio.optimization._base import BaseOptimization
from skfolio.portfolio import Portfolio
from skfolio.prior import BasePrior, ReturnDistribution
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

        The default is `RiskMeasure.VARIANCE`.

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
        minium weight) and the input `X` of the `fit` methods must be a DataFrame with
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
        Transaction costs of the assets. It is used to add linear transaction costs to
        the optimization problem:

        .. math:: total\_cost = \sum_{i=1}^{N} c_{i} \times |w_{i} - w\_prev_{i}|

        with :math:`c_{i}` the transaction cost of asset i, :math:`w_{i}` its weight
        and :math:`w\_prev_{i}` its previous weight (defined in `previous_weights`).
        The float :math:`total\_cost` is impacting the portfolio expected return in the optimization:

        .. math:: expected\_return = \mu^{T} \cdot w - total\_cost

        with :math:`\mu` the vector of assets' expected returns and :math:`w` the
        vector of assets weights.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset cost) and the input `X` of the `fit` method must be a
        DataFrame with the asset names in columns.
        The default value is `0.0`.

        .. warning::

            Based on the above formula, the periodicity of the transaction costs
            needs to be homogeneous to the periodicity of :math:`\mu`. For example, if
            the input `X` is composed of **daily** returns, the `transaction_costs` need
            to be expressed as **daily** costs.
            (See :ref:`sphx_glr_auto_examples_mean_risk_plot_6_transaction_costs.py`)

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Management fees of the assets. It is used to add linear management fees to the
        optimization problem:

        .. math:: total\_fee = \sum_{i=1}^{N} f_{i} \times w_{i}

        with :math:`f_{i}` the management fee of asset i and :math:`w_{i}` its weight.
        The float :math:`total\_fee` is impacting the portfolio expected return in the optimization:

        .. math:: expected\_return = \mu^{T} \cdot w - total\_fee

        with :math:`\mu` the vector of assets' expected returns and :math:`w` the vector
        of assets weights.

        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset fee) and the input `X` of the `fit` method must be a
        DataFrame with the asset names in columns.
        The default value is `0.0`.

        .. warning::

            Based on the above formula, the periodicity of the management fees needs to
            be homogeneous to the periodicity of :math:`\mu`. For example, if the input
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
        be a DataFrame with the asset names in columns.
        The default (`None`) means no previous weights.
        Additionally, when `fallback="previous_weights"`, failures will fall back to
        these weights if provided.

    portfolio_params : dict, optional
        Portfolio parameters forwarded to the resulting `Portfolio` in `predict`.
        If not provided and if available on the estimator, the following
        attributes are propagated to the portfolio by default: `name`,
        `transaction_costs`, `management_fees`, `previous_weights` and `risk_free_rate`.

    fallback : BaseOptimization | "previous_weights" | list[BaseOptimization | "previous_weights"], optional
        Fallback estimator or a list of estimators to try, in order, when the primary
        optimization raises during `fit`. Alternatively, use `"previous_weights"`
        (alone or in a list) to fall back to the estimator's `previous_weights`.
        When a fallback succeeds, its fitted `weights_` are copied back to the primary
        estimator so that `fit` still returns the original instance. For traceability,
        `fallback_` stores the successful estimator (or the string `"previous_weights"`)
         and `fallback_chain_` stores each attempt with the associated outcome.

    raise_on_failure : bool, default=True
        Controls error handling when fitting fails.
        If True, any failure during `fit` is raised immediately, no `weights_` are
        set and subsequent calls to `predict` will raise a `NotFittedError`.
        If False, errors are not raised; instead, a warning is emitted, `weights_`
        is set to `None` and subsequent calls to `predict` will return a
        `FailedPortfolio`. When fallbacks are specified, this behavior applies only
        after all fallbacks have been exhausted.

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

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    fallback_ : BaseOptimization | "previous_weights" | None
        The fallback estimator instance, or the string `"previous_weights"`, that
        produced the final result. `None` if no fallback was used.

    fallback_chain_ : list[tuple[str, str]] | None
        Sequence describing the optimization fallback attempts. Each element is a
        pair `(estimator_repr, outcome)` where `estimator_repr` is the string
        representation of the primary estimator or a fallback (e.g. `"EqualWeighted()"`,
        `"previous_weights"`), and `outcome` is `"success"` if that step produced
        a valid solution, otherwise the stringified error message. For successful
        fits without any fallback, this is `None`.

    error_ : str | list[str] | None
        Captured error message(s) when `fit` fails. For multi-portfolio outputs
        (`weights_` is 2D), this is a list aligned with portfolios.

    Notes
    -----
    All estimators should specify all parameters as explicit keyword arguments in
    `__init__` (no `*args` or `**kwargs`), following scikit-learn conventions.
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
        fallback: skt.Fallback = None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            previous_weights=previous_weights,
            raise_on_failure=raise_on_failure,
        )
        self.risk_measure = risk_measure
        self.prior_estimator = prior_estimator
        self.distance_estimator = distance_estimator
        self.hierarchical_clustering_estimator = hierarchical_clustering_estimator
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees
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
        return_distribution: ReturnDistribution,
    ) -> float:
        """Compute the risk measure of a theoretical portfolio defined by the weights
        vector.

        Parameters
        ----------
        weights : ndarray of shape (n_assets,)
           The vector of weights.

        return_distribution : ReturnDistribution
            The assets return distribution.

        Returns
        -------
        risk: float
            The risk measure of a theoretical portfolio defined by the weights
            vector.
        """
        ptf = Portfolio(
            X=return_distribution.returns,
            sample_weight=return_distribution.sample_weight,
            weights=weights,
            transaction_costs=self.transaction_costs,
            management_fees=self.management_fees,
            previous_weights=self.previous_weights,
        )
        if self.risk_measure in [RiskMeasure.VARIANCE, RiskMeasure.STANDARD_DEVIATION]:
            risk = ptf.variance_from_assets(
                assets_covariance=return_distribution.covariance
            )
            if self.risk_measure == RiskMeasure.STANDARD_DEVIATION:
                risk = np.sqrt(risk)
        else:
            risk = getattr(ptf, str(self.risk_measure.value))
        return risk

    def _unitary_risks(self, return_distribution: ReturnDistribution) -> np.ndarray:
        """Compute the vector of risk measure for each single assets.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            The asset returns distribution.

        Returns
        -------
        values: ndarray of shape (n_assets,)
            The risk measure of each asset.
        """
        n_assets = return_distribution.returns.shape[1]
        risks = [
            self._risk(weights=weights, return_distribution=return_distribution)
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
            if min_weights.sum() >= 1.00001:
                raise ValueError(
                    f"Invalid `min_weights`: sum is {min_weights.sum():.4f}, "
                    f"but it must be less than 1.0."
                )

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
            if max_weights.sum() < 1:
                raise ValueError(
                    f"Invalid `max_weights`: sum is {max_weights.sum():.4f}, "
                    f"but it must be at least 1.0."
                )

        if np.any(min_weights > max_weights):
            raise NameError(
                "Items of `min_weights` must be less than or equal to items of"
                " `max_weights`"
            )

        return min_weights, max_weights

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = (
            skm.MetadataRouter(owner=self.__class__.__name__)
            .add(
                prior_estimator=self.prior_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                distance_estimator=self.distance_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                hierarchical_clustering_estimator=self.hierarchical_clustering_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: None = None, **fit_params):
        pass
