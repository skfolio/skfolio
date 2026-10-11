"""Hierarchical Risk Parity Optimization estimator."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause
# The risk measure generalization and constraint features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

from __future__ import annotations

from typing import Any

import numpy as np

import skfolio.typing as skt
from skfolio.cluster import HierarchicalClustering
from skfolio.distance import BaseDistance
from skfolio.exceptions import OptimizationError
from skfolio.measures import ExtraRiskMeasure, RiskMeasure
from skfolio.optimization.hierarchical._seriation._base import (
    _BaseSeriatedOptimization,
)
from skfolio.optimization.hierarchical._utils import (
    _PortfolioRiskMixin,
    _apply_weight_constraints_to_split_factor,
)
from skfolio.prior import BasePrior, ReturnDistribution
from skfolio.seriation import BaseSeriation
from skfolio.typing import ArrayLike, FloatArray, IntArray
from skfolio.utils.tools import bisection


class HierarchicalRiskParity(_PortfolioRiskMixin, _BaseSeriatedOptimization):
    r"""Hierarchical Risk Parity estimator.

    Hierarchical Risk Parity is a portfolio optimization method developed by Marcos
    Lopez de Prado [1]_.

    This algorithm uses a distance matrix to compute hierarchical clusters using the
    Hierarchical Tree Clustering algorithm. It then employs seriation to rearrange the
    assets in the dendrogram, minimizing the distance between leaves.

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
        than the single-linkage method [2]_.

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
        containing estimates of expected asset returns, covariance matrix and
        returns. The moments and returns estimations are used for the risk computation
        and the returns estimation are used by the distance matrix estimator.
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    distance_estimator : BaseDistance, optional
        :ref:`Distance estimator <distance>`.
        The distance estimator is used to estimate the codependence and the distance
        matrix used by the seriation estimator.
        The default (`None`) is to use :class:`~skfolio.distance.PearsonDistance`.

    hierarchical_clustering_estimator : HierarchicalClustering, optional
        Deprecated. Use `seriation_estimator=HierarchicalSeriation(
        hierarchical_clustering_estimator=...)`. Supplying both parameters is an error.
        Will be removed in version 2.0.

    seriation_estimator : BaseSeriation, optional
        Asset ordering estimator. The default is
        :class:`~skfolio.seriation.HierarchicalSeriation` with Ward linkage
        and optimal leaf ordering. Use :class:`~skfolio.seriation.SpectralSeriation`
        for spectral ordering with orientation carried across online updates.

    distance_from_prior : bool, default=True
        If True, fit the distance estimator on the return scenarios produced by
        the prior. With `CovarianceDistance(covariance_estimator="precomputed")`,
        use the prior's covariance instead.

        If False, use the `X` argument supplied to this optimizer's `fit(X)` or
        `partial_fit(X)`, before the prior processes it. In a pipeline, `X` is
        the output of the preceding steps. Precomputed covariance distances
        require True.

        During online learning, True refits the distance estimator on the prior's
        current scenarios or covariance at each update. False updates it from
        the new observations in `X` and requires support for `partial_fit`.
        Portfolio allocation always uses the prior's moments and return scenarios.
        See :ref:`asset_seriation`.

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
            must match the periodicity of :math:`\mu`. For example, if the input
            `X` is composed of **daily** returns, the `transaction_costs` need to be
            expressed as **daily** costs. A transaction cost is paid once per
            rebalancing while a position earns its expected return on every period it
            is held, so the one-off cost is converted by dividing it by the expected
            investment duration (e.g. `0.001 / 21` for a 10 bps cost with daily
            returns and a one-month expected holding period).
            (See :ref:`Periodicity Convention <periodicity_convention>`)

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

            Based on the above formula, the periodicity of the management fees
            must match the periodicity of :math:`\mu`. For example, if the input
            `X` is composed of **daily** returns, the `management_fees` need to be
            expressed in **daily** fees. Unlike transaction costs, management fees
            accrue with holding time, so a stated annual fee converts directly to the
            return periodicity (e.g. `0.02 / 252` for a 2% annual fee on daily
            returns).

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
        optimization raises during `fit`. Alternatively, use `"previous_weights"` (alone
        or in a list) to fall back to the estimator's `previous_weights`. When a
        fallback succeeds, its fitted `weights_` are copied back to the primary
        estimator so that `fit` still returns the original instance. For traceability,
        `fallback_` stores the successful estimator (or the string `"previous_weights"`)
        and `fallback_chain_` stores each attempt with the associated outcome. With
        `partial_fit`, only None or `"previous_weights"` is supported because fallback
        estimators have not accumulated the primary model's online history. See
        :ref:`optimization_fallbacks`.

    raise_on_failure : bool, default=True
        Controls error handling when fitting fails and no fallback succeeds. If True,
        the estimator raises the final error. If False, the estimator emits a warning
        and sets `weights_` to None, so subsequent calls to `predict` return a
        :class:`~skfolio.portfolio.FailedPortfolio`. During `fit`, `raise_on_failure`
        applies to any fitting error, including errors raised by the prior estimator.
        During `partial_fit`, `raise_on_failure` applies only to optimization failures
        after learning completes. Input validation failures and errors from the prior or
        other learning estimators are always raised. See
        :ref:`optimization_failure_handling` for batch recovery and
        :ref:`online_failure_handling` for online continuation and restart rules.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Weights of the assets.

    distance_estimator_ : BaseDistance or None
        Fitted `distance_estimator`. None when `distance_from_prior=True` and
        fewer than two assets are investable.

    seriation_estimator_ : BaseSeriation
        Fitted ordering estimator. Its `ordering_` contains positions in the
        full input schema. Hierarchical linkage diagnostics are available through
        `seriation_estimator_.hierarchical_clustering_estimator_`.

    hierarchical_clustering_estimator_ : HierarchicalClustering or None
        Deprecated alias of `seriation_estimator_.hierarchical_clustering_estimator_`.
        Access emits a FutureWarning. Will be removed in version 2.0.
        Unavailable with spectral seriation.

    investable_mask_ : ndarray of shape (n_assets,) or None
        Mask of investable assets from the fitted prior. None when all assets are
        investable. May be absent after batch fallback if fitting failed before
        the prior's investable mask was determined.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has asset names that are all strings.

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

    error_ : str | list[str | None] | None
        For a single portfolio, this is the recorded error message, or None after a
        successful allocation or fallback. For multiple portfolios, it is a list with
        one entry per row of `weights_`, containing an error message for each failed
        portfolio and None for each successful portfolio.

    Notes
    -----
    All estimators should specify all parameters as explicit keyword arguments in
    `__init__` (no `*args` or `**kwargs`), following scikit-learn conventions.

    References
    ----------
    .. [1] "A robust estimator of the efficient frontier",
        SSRN Electronic Journal,
        Marcos López de Prado (2019).

    .. [2] "A review of two decades of correlations, hierarchies, networks and
        clustering in financial markets",
        Gautier Marti, Frank Nielsen, Mikołaj Bińkowski, Philippe Donnat (2020).

    .. [3] "Portfolio Optimization: Theory and Application", Chapter 12,
        Daniel P. Palomar (2025)

    .. [4] "Building diversified portfolios that outperform out of sample",
        The Journal of Portfolio Management,
        Marcos López de Prado (2016).

    .. [5] "Machine Learning for Asset Managers",
        Elements in Quantitative Finance. Cambridge University Press,
        Marcos López de Prado (2020).

    Examples
    --------
    For online updates with late listings, delistings, holidays and asset warm-up,
    see
    :ref:`sphx_glr_auto_examples_online_learning_plot_online_schur_changing_universe.py`.
    That example uses Schur, and the same online setup applies to HRP.
    """

    def __init__(
        self,
        risk_measure: RiskMeasure | ExtraRiskMeasure = RiskMeasure.VARIANCE,
        prior_estimator: BasePrior | None = None,
        distance_estimator: BaseDistance | None = None,
        # TODO remove deprecated hierarchical_clustering_estimator in v2.0
        hierarchical_clustering_estimator: HierarchicalClustering | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
        portfolio_params: dict | None = None,
        fallback: skt.Fallback = None,
        raise_on_failure: bool = True,
        *,
        seriation_estimator: BaseSeriation | None = None,
        distance_from_prior: bool = True,
    ) -> None:
        super().__init__(
            prior_estimator=prior_estimator,
            distance_estimator=distance_estimator,
            # TODO remove deprecated hierarchical_clustering_estimator in v2.0
            hierarchical_clustering_estimator=hierarchical_clustering_estimator,
            min_weights=min_weights,
            max_weights=max_weights,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            previous_weights=previous_weights,
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
            seriation_estimator=seriation_estimator,
            distance_from_prior=distance_from_prior,
        )
        self.risk_measure = risk_measure

    def fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any
    ) -> HierarchicalRiskParity:
        """Fit the Hierarchical Risk Parity Optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Price returns of the assets.

        y : array-like, optional
            Targets passed to the prior and, when distance_from_prior=False,
            to the distance estimator. Prior scenarios do not receive this target.

        **fit_params : dict
            Metadata routed to the underlying estimators. Metadata supplied to
            a distance fitted on prior scenarios must align with those scenarios.

        Returns
        -------
        self : HierarchicalRiskParity
            Fitted estimator.
        """
        self._reset()
        self._warn_deprecated_clustering_estimator(stacklevel=4)
        self._fit(X, y, method="fit", **fit_params)
        return self

    def partial_fit(
        self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any
    ) -> HierarchicalRiskParity:
        """Update the prior, distances and seriation, then compute portfolio weights.

        The prior must support incremental learning. With
        `distance_from_prior=False`, the distance must also support it.
        Supply only new observations on each call. See
        :ref:`online_failure_handling` for failure handling and restarts.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            New returns with the same full asset schema as previous calls.

        y : array-like, optional
            Targets passed to the prior and, when distance_from_prior=False,
            to the distance estimator.

        **fit_params : dict
            Metadata routed to the underlying estimators. Requires metadata
            routing to be enabled. Observation metadata is not forwarded to
            distances fitted on the prior's return scenarios or covariance.

        Returns
        -------
        self : HierarchicalRiskParity
            Updated estimator.

        Raises
        ------
        OptimizationError
            If optimization fails with `raise_on_failure=True` and `fallback=None`.
        """
        self._warn_deprecated_clustering_estimator(stacklevel=3)
        self._fit(X, y, method="partial_fit", **fit_params)
        return self

    def _validate_params(self) -> None:
        """Validate parameters."""
        if not isinstance(self.risk_measure, RiskMeasure | ExtraRiskMeasure):
            raise TypeError(
                "`risk_measure` must be of type `RiskMeasure` or `ExtraRiskMeasure`"
            )
        if self.risk_measure in [ExtraRiskMeasure.SKEW, ExtraRiskMeasure.KURTOSIS]:
            raise ValueError(
                f"risk_measure {self.risk_measure} currently not supported in HRP"
            )

    def _compute_weights(
        self,
        return_distribution: ReturnDistribution,
        ordering: IntArray,
        min_weights: FloatArray,
        max_weights: FloatArray,
    ) -> FloatArray:
        """Apply recursive risk bisection in the supplied asset order."""
        n_assets = len(ordering)
        assets_risks = self._unitary_risks(return_distribution=return_distribution)
        if not np.isfinite(assets_risks).all() or np.any(assets_risks == 0):
            raise OptimizationError(
                "HRP cannot split assets with zero or nonfinite risk."
            )
        weights = np.ones(n_assets)
        items = [ordering]

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
                        self._risk(
                            weights=inv_risk_w, return_distribution=return_distribution
                        )
                    )
                left_risk, right_risk = risks
                left_cluster, right_cluster = clusters_ids
                if not np.isfinite(risks).all() or left_risk + right_risk == 0:
                    raise OptimizationError(
                        "HRP cannot split clusters with zero total or nonfinite risk."
                    )
                alpha = 1 - left_risk / (left_risk + right_risk)
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
