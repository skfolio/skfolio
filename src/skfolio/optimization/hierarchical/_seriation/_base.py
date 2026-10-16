"""Shared learning and ordering lifecycle for HRP and Schur allocation."""

# Copyright (c) 2023-2026
# Author: Hugo Delatte <hugo.delatte@skfoliolabs.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import Any, cast

import numpy as np
import pandas as pd
import sklearn.utils as sku
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv
from sklearn import get_config

import skfolio.typing as skt
from skfolio.cluster import HierarchicalClustering
from skfolio.distance import BaseDistance, PearsonDistance
from skfolio.exceptions import OptimizationError
from skfolio.optimization._base import BaseOptimization, _check_finite_weights
from skfolio.optimization.hierarchical._utils import (
    _WEIGHT_BOUNDS_TOL,
    _convert_weight_bounds,
)
from skfolio.prior import BasePrior, EmpiricalPrior, ReturnDistribution
from skfolio.seriation import BaseSeriation, HierarchicalSeriation
from skfolio.typing import ArrayLike, BoolArray, FloatArray, IntArray
from skfolio.utils.stats import assert_is_symmetric
from skfolio.utils.tools import (
    _call_estimator,
    _filter_supported_params,
    _validate_bool,
    check_estimator,
)

_FITTED_ATTR = "weights_"


class _BaseSeriatedOptimization(BaseOptimization):
    """Base class for portfolio optimizers that allocate using seriation.

    Shared by Hierarchical Risk Parity and Schur Complementary. The base fits the
    prior, computes distances, orders the investable assets and delegates weight
    computation to the subclass. Portfolio weights are then expanded to the full
    input asset universe, with zero weights for non-investable assets.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        :ref:`Prior estimator <prior>` used to estimate expected returns, covariance
        and return scenarios in a :class:`~skfolio.prior.ReturnDistribution`.
        These estimates are used for portfolio allocation. The default (`None`)
        is :class:`~skfolio.prior.EmpiricalPrior`.

    distance_estimator : BaseDistance, optional
        :ref:`Distance estimator <distance>` used to compute the distance matrix
        for seriation. Its input is controlled by `distance_from_prior`.
        The default (`None`) is :class:`~skfolio.distance.PearsonDistance`.

    seriation_estimator : BaseSeriation, optional
        :ref:`Seriation estimator <seriation>` used to order the investable assets.
        The default (`None`) is :class:`~skfolio.seriation.HierarchicalSeriation`
        with Ward linkage and optimal leaf ordering. During online learning,
        the estimator is updated with `partial_fit` if supported, otherwise `fit`.

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

    hierarchical_clustering_estimator : HierarchicalClustering, optional
        Deprecated. Use `seriation_estimator=HierarchicalSeriation(
        hierarchical_clustering_estimator=...)`. Supplying both parameters is an error.
        Will be removed in version 2.0.

    min_weights : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
        Minimum weight of each investable asset. Negative weights are not allowed.
        A scalar applies to every asset. An array follows the columns of `X`.
        A dictionary maps asset names to weights and requires named columns in `X`.
        Missing dictionary entries and `None` use the default of `0.0`.

    max_weights : float | dict[str, float] | array-like of shape (n_assets,), default=1.0
        Maximum weight of each investable asset. Weights above 1.0 are not allowed.
        A scalar applies to every asset. An array follows the columns of `X`.
        A dictionary maps asset names to weights and requires named columns in `X`.
        Missing dictionary entries and `None` use the default of `1.0`.

    transaction_costs : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
        Transaction costs of the assets, forwarded to the predicted portfolios.
        A scalar applies to every asset. An array follows the columns of `X`.
        A dictionary maps asset names to costs and requires named columns in `X`.
        Missing dictionary entries use zero. Their use during allocation depends
        on the subclass. See :ref:`periodicity_convention` for cost units.

    management_fees : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
        Management fees of the assets, forwarded to the predicted portfolios.
        A scalar applies to every asset. An array follows the columns of `X`.
        A dictionary maps asset names to fees and requires named columns in `X`.
        Missing dictionary entries use zero. Their use during allocation depends
        on the subclass. See :ref:`periodicity_convention` for fee units.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets,), optional
        Previous asset weights, used to compute transaction costs and to recover
        from failures when `fallback="previous_weights"`.
        A scalar applies to every asset. An array follows the columns of `X`.
        A dictionary maps asset names to weights and requires named columns in `X`.
        Missing dictionary entries use zero. The default (`None`) means no previous
        weights are supplied.

    portfolio_params : dict, optional
        Portfolio parameters forwarded to the resulting `Portfolio` in `predict`.
        Unless set in this dictionary, `transaction_costs`, `management_fees`,
        `previous_weights` and `risk_free_rate` are forwarded from the optimizer when
        available, and `name` defaults to the optimizer class name.

    fallback : BaseOptimization | "previous_weights" | list[BaseOptimization | "previous_weights"], optional
        Fallback estimator or a list of estimators to try, in order, when the primary
        optimization raises during `fit`. Alternatively, use `"previous_weights"`
        to reuse the supplied previous weights. When a fallback succeeds, its
        weights are copied to the primary estimator and `fallback_` records it.
        During `partial_fit`, only None or `"previous_weights"` is supported.
        Separate fallback estimators do not share the primary model's online
        history. See :ref:`optimization_fallbacks`.

    raise_on_failure : bool, default=True
        Controls error handling when fitting fails and no fallback succeeds. If True,
        the estimator raises the final error. If False, the estimator emits a warning
        and sets `weights_` to None, so subsequent calls to `predict` return a
        :class:`~skfolio.portfolio.FailedPortfolio`. During `fit`, this applies to any
        fitting error, including errors raised by the prior. During `partial_fit`,
        it applies only to optimization failures after learning completes. Input
        validation failures and errors from the prior, distance or seriation estimator
        are always raised. See :ref:`optimization_failure_handling` and
        :ref:`online_failure_handling`.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,) or (n_optimizations, n_assets), or None
        Portfolio weights aligned with the columns of `X`. A fallback may return
        multiple portfolios. None when fitting fails and no fallback succeeds with
        `raise_on_failure=False`.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    distance_estimator_ : BaseDistance or None
        Fitted `distance_estimator`. None when `distance_from_prior=True` and
        fewer than two assets are investable.

    seriation_estimator_ : BaseSeriation
        Fitted `seriation_estimator`. Its `ordering_` contains positions in the
        full set of input asset columns.

    hierarchical_clustering_estimator_ : HierarchicalClustering or None
        Deprecated alias of `seriation_estimator_.hierarchical_clustering_estimator_`.
        Access emits a FutureWarning. Will be removed in version 2.0.
        Unavailable with spectral seriation.

    investable_mask_ : ndarray of shape (n_assets,) or None
        Mask of investable assets from the fitted prior. None when all assets are
        investable. May be absent after batch fallback if fitting failed before
        the prior's investable mask was determined.

    n_features_in_ : int
        Number of assets supplied to `fit` or `partial_fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Asset names, defined when `X` provides names that are all strings.

    fallback_ : BaseOptimization | "previous_weights" | None
        The fallback estimator instance, or the string `"previous_weights"`, that
        produced the final result. None if no fallback was used.

    fallback_chain_ : list[tuple[str, str]] | None
        Sequence describing the primary failure and subsequent fallback attempts.
        Each entry contains the estimator's string representation and either
        `"success"` or its error message. None when no fallback chain was run.

    error_ : str | list[str | None] | None
        For a single portfolio, this is the recorded error message, or None after a
        successful allocation or fallback. For multiple portfolios, it is a list with
        one entry per row of `weights_`, containing an error message for each failed
        portfolio and None for each successful portfolio.

    Notes
    -----
    Subclasses call `_reset` before delegating `fit` to `_fit`. Their `partial_fit`
    delegates to `_fit` without resetting the fitted estimators.

    Subclasses implement `_compute_weights` for at least two investable assets.
    It receives the return distribution, asset ordering and weight bounds restricted
    to those assets, and returns weights in the distribution's asset order. The base
    validates the weights and expands them to the full universe. With one investable
    asset, the base assigns it weight one after checking the bounds.

    Subclasses raise :class:`~skfolio.exceptions.OptimizationError` when they cannot
    compute valid allocation weights. During online learning, the prior, distance and
    seriation updates complete before optimization failure handling begins.
    """

    prior_estimator_: BasePrior
    seriation_estimator_: BaseSeriation
    distance_estimator_: BaseDistance | None

    def __init__(
        self,
        *,
        prior_estimator: BasePrior | None = None,
        distance_estimator: BaseDistance | None = None,
        seriation_estimator: BaseSeriation | None = None,
        distance_from_prior: bool = True,
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
    ) -> None:
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            previous_weights=previous_weights,
            raise_on_failure=raise_on_failure,
        )
        self.prior_estimator = prior_estimator
        self.distance_estimator = distance_estimator
        self.seriation_estimator = seriation_estimator
        self.distance_from_prior = distance_from_prior
        self.hierarchical_clustering_estimator = hierarchical_clustering_estimator
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees

    # TODO remove deprecated hierarchical_clustering_estimator_ in v2.0
    @property
    def hierarchical_clustering_estimator_(self) -> HierarchicalClustering | None:
        """Deprecated alias for `seriation_estimator_.hierarchical_clustering_estimator_`."""
        clustering = cast(
            HierarchicalSeriation, self.seriation_estimator_
        ).hierarchical_clustering_estimator_
        warnings.warn(
            "`hierarchical_clustering_estimator_` is deprecated and will be removed "
            "in version 2.0. Use `seriation_estimator_.hierarchical_clustering_estimator_` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return clustering

    def get_metadata_routing(self) -> skm.MetadataRouter:
        """Get metadata routing for the prior, distance and seriation estimators."""
        distance_method = "fit" if self.distance_from_prior else "partial_fit"
        router = (
            skm.MetadataRouter(owner=type(self).__name__)
            .add(
                prior_estimator=self.prior_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee="partial_fit"),
            )
            .add(
                distance_estimator=self.distance_estimator,
                method_mapping=skm.MethodMapping()
                .add(caller="fit", callee="fit")
                .add(caller="partial_fit", callee=distance_method),
            )
        )
        seriation = self._resolve_seriation()
        seriation_method = (
            "partial_fit"
            if callable(getattr(seriation, "partial_fit", None))
            else "fit"
        )
        router.add(
            seriation_estimator=seriation,
            method_mapping=skm.MethodMapping()
            .add(caller="fit", callee="fit")
            .add(caller="partial_fit", callee=seriation_method),
        )
        return router

    # TODO remove deprecated hierarchical_clustering_estimator handling in v2.0
    def _resolve_seriation(self) -> BaseSeriation:
        """Select the seriation estimator from the constructor parameters.

        Returns the supplied `seriation_estimator`, or creates a
        `HierarchicalSeriation` using `hierarchical_clustering_estimator`.
        Constructor parameters are left unchanged.

        Returns
        -------
        seriation : BaseSeriation
            Configured or default seriation estimator. Cloning and fitting are
            performed by the caller.

        Raises
        ------
        ValueError
            If both `seriation_estimator` and `hierarchical_clustering_estimator`
            are provided.
        """
        if self.seriation_estimator is not None:
            if self.hierarchical_clustering_estimator is not None:
                raise ValueError(
                    "Use either seriation_estimator or hierarchical_clustering_estimator, not both."
                )
            return self.seriation_estimator
        return HierarchicalSeriation(
            hierarchical_clustering_estimator=self.hierarchical_clustering_estimator
        )

    def _fit(
        self, X: ArrayLike, y: ArrayLike | None, method: str, **fit_params: Any
    ) -> _BaseSeriatedOptimization:
        """Fit or update the estimators and compute portfolio weights.

        Updates the prior, computes distances and fits or updates seriation before
        computing weights for the investable assets. The weights are expanded to
        match the full set of columns in `X`.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
            Asset returns. For `method="partial_fit"`, supply only new observations
            with the same asset columns in the same order as previous updates.

        y : array-like or None
            Targets passed to the prior and, when `distance_from_prior=False`,
            to the distance estimator. Distances fitted on the prior's return
            scenarios or covariance do not receive these targets.

        method : {"fit", "partial_fit"}
            Fitting method requested by the public method. The caller invokes
            `_reset` before starting a new batch fit.

        **fit_params : dict
            Metadata routed to the underlying estimators. Requires metadata
            routing to be enabled. During batch fitting, metadata for a distance
            fitted on prior return scenarios must align with those scenarios.
            Sample weights for scenario distances always come from the prior,
            replacing routed weights. Other observation metadata cannot be routed
            to distances fitted from prior output during online updates.
            Distances using the prior's covariance receive no observation metadata.

        Returns
        -------
        self : _BaseSeriatedOptimization
            Fitted or updated estimator. After a handled optimization failure,
            `weights_` contains the fallback weights or None.

        Raises
        ------
        TypeError
            If configured estimators or parameter types are invalid.
        ValueError
            If inputs, estimator settings or fitted outputs fail validation.
        OptimizationError
            If allocation is infeasible or produces invalid weights. For
            `method="partial_fit"`, this error is handled according to `fallback`
            and `raise_on_failure`.

        Notes
        -----
        During `fit`, errors propagate to the public method's fallback wrapper.
        During `partial_fit`, errors from updating the prior, distance or seriation
        estimator propagate directly. Optimization failure handling starts only
        once those updates have completed. See :ref:`optimization_failure_handling`
        and :ref:`online_failure_handling`.
        """
        first_call = not hasattr(self, _FITTED_ATTR)

        if first_call:
            self._validate_params()
            _validate_bool(self.distance_from_prior, "distance_from_prior")

        # Distances fitted on prior output start afresh at each update.
        # Distances learned from X keep their state across online updates.
        if first_call or self.distance_from_prior:
            distance = check_estimator(
                self.distance_estimator,
                default=PearsonDistance(),
                check_type=BaseDistance,
            )
        else:
            distance = cast(BaseDistance, self.distance_estimator_)

        covariance_input = distance.requires_covariance_input
        if first_call:
            if covariance_input and not self.distance_from_prior:
                raise ValueError(
                    "Precomputed covariance requires distance_from_prior=True."
                )
            self._initialize(distance)

        if method == "partial_fit":
            self._validate_partial_fit_fallback()

        routed = skm.process_routing(self, method, **fit_params)
        distance_method = "fit" if self.distance_from_prior else method
        distance_params = getattr(routed.distance_estimator, distance_method)

        if self.distance_from_prior and not covariance_input and distance_params:
            # Scenario probabilities replace weights describing the original X.
            weight_names = _sample_weight_names(distance.get_metadata_routing())
            distance_params = {
                name: value
                for name, value in distance_params.items()
                if name not in weight_names
            }
            routed.distance_estimator.fit = distance_params

        # Covariance rows represent assets. Online prior scenarios may cover
        # a different history from the observations in the current batch.
        if distance_params and (
            covariance_input or (self.distance_from_prior and method == "partial_fit")
        ):
            raise ValueError(
                "Observation metadata for X cannot be passed to a distance "
                "estimator fitted on the prior's return scenarios or covariance, "
                "which may have different rows. Disable the corresponding fit "
                "requests on the distance estimator or its underlying estimators."
            )

        # Keep the asset schema consistent. The prior and distance validate X.
        skv.validate_data(self, X, reset=first_call, skip_check_array=True)

        # Bounds cover the full asset schema before selecting investable assets.
        min_weights, max_weights = _convert_weight_bounds(
            min_weights=self._clean_input(
                self.min_weights,
                n_assets=self.n_features_in_,
                fill_value=0,
                name="min_weights",
                apply_investable_mask=False,
            ),
            max_weights=self._clean_input(
                self.max_weights,
                n_assets=self.n_features_in_,
                fill_value=1,
                name="max_weights",
                apply_investable_mask=False,
            ),
            n_assets=self.n_features_in_,
        )

        _call_estimator(
            self.prior_estimator_, method, X, y, routed_params=routed.prior_estimator
        )
        if not self.distance_from_prior:
            _call_estimator(
                distance,
                method,
                X,
                y,
                routed_params=routed.distance_estimator,
            )

        return_distribution = self.prior_estimator_.return_distribution_
        self.investable_mask_ = return_distribution.investable_mask
        investable_mask = self.investable_mask_
        if investable_mask is None:
            investable_mask = np.ones(self.n_features_in_, dtype=bool)
        investable_indices = np.flatnonzero(investable_mask)

        distance_matrix = self._compute_distance_matrix(
            return_distribution, investable_mask, distance, routed.distance_estimator
        )

        seriation_method = "fit"
        if method == "partial_fit" and callable(
            getattr(self.seriation_estimator_, "partial_fit", None)
        ):
            seriation_method = "partial_fit"

        _call_estimator(
            self.seriation_estimator_,
            seriation_method,
            distance_matrix,
            routed_params=routed.seriation_estimator,
        )

        ordering = self.seriation_estimator_.ordering_
        if ordering.ndim != 1 or not np.array_equal(
            np.sort(ordering), investable_indices
        ):
            raise ValueError(
                "Seriation ordering must contain every investable asset exactly once."
            )

        # Online recovery starts after learning. Batch errors reach the outer
        # fit wrapper, which runs fallback once with the original inputs.
        with self._handle_optimization_errors(
            X, y, method=method, enabled=method == "partial_fit"
        ):
            lower = min_weights[investable_mask]
            upper = max_weights[investable_mask]
            if (
                lower.sum() > 1 + _WEIGHT_BOUNDS_TOL
                or upper.sum() < 1 - _WEIGHT_BOUNDS_TOL
            ):
                raise OptimizationError(
                    "Weight bounds are infeasible for the investable assets."
                )

            return_distribution = return_distribution.investable_subset(slim=True)
            covariance = return_distribution.covariance
            if not np.isfinite(covariance).all() or np.any(np.diag(covariance) <= 0):
                raise ValueError(
                    "The investable covariance must be finite with positive variances."
                )
            assert_is_symmetric(covariance)

            if len(investable_indices) == 1:
                weights = np.ones(1)
            else:
                # Seriation positions refer to the full universe. Allocation
                # uses positions in the investable subset.
                subset_ordering = np.searchsorted(investable_indices, ordering)
                weights = self._compute_weights(
                    return_distribution,
                    subset_ordering,
                    lower,
                    upper,
                )

            _check_finite_weights(weights)
            if (
                not np.isclose(weights.sum(), 1)
                or np.any(weights < lower - _WEIGHT_BOUNDS_TOL)
                or np.any(weights > upper + _WEIGHT_BOUNDS_TOL)
            ):
                raise OptimizationError(
                    "The allocation could not satisfy the weight bounds."
                )

            self.weights_ = self._expand_weights_to_full_universe(weights)

        return self

    def _initialize(self, distance_estimator: BaseDistance) -> None:
        """Initialize the prior, distance and seriation estimators.

        Creates unfitted prior and seriation estimators from the configured
        parameters, using defaults where needed. No observations are consumed.

        Parameters
        ----------
        distance_estimator : BaseDistance
            Validated clone of the configured distance estimator. Stored in
            `distance_estimator_` when `distance_from_prior=False`. With True,
            that attribute is set to None until a distance estimator is fitted
            on the prior's output.
        """
        self.prior_estimator_ = check_estimator(
            self.prior_estimator, default=EmpiricalPrior(), check_type=BasePrior
        )
        self.seriation_estimator_ = check_estimator(
            self._resolve_seriation(), default=None, check_type=BaseSeriation
        )
        self.distance_estimator_ = (
            None if self.distance_from_prior else distance_estimator
        )

    # TODO remove deprecated hierarchical_clustering_estimator warning in v2.0
    def _warn_deprecated_clustering_estimator(self, stacklevel: int) -> None:
        """Warn when a new learning run uses the deprecated clustering parameter.

        Parameters
        ----------
        stacklevel : int
            Warning stack level for the calling public method. The fallback
            wrapper adds one frame for `fit` compared with `partial_fit`.

        Warns
        -----
        FutureWarning
            If `hierarchical_clustering_estimator` is supplied without a
            `seriation_estimator` before the run has produced `weights_`.
        """
        if (
            self.hierarchical_clustering_estimator is not None
            and self.seriation_estimator is None
            and not hasattr(self, _FITTED_ATTR)
        ):
            warnings.warn(
                "`hierarchical_clustering_estimator` is deprecated and will be "
                "removed in version 2.0. Use "
                "`seriation_estimator=HierarchicalSeriation(hierarchical_clustering_estimator=...)` "
                "instead.",
                FutureWarning,
                stacklevel=stacklevel,
            )

    def _compute_distance_matrix(
        self,
        return_distribution: ReturnDistribution,
        investable_mask: BoolArray,
        distance_estimator: BaseDistance,
        routed_params: sku.Bunch,
    ) -> FloatArray | pd.DataFrame:
        """Compute the full distance matrix used for seriation.

        With `distance_from_prior=True`, fit the distance estimator on the prior's
        current return scenarios or covariance. With False, use the distance
        estimator already fitted or updated from `X` by `_fit`.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            Current prior distribution over the full asset universe.

        investable_mask : ndarray of bool of shape (n_assets,)
            Assets eligible for allocation according to the prior.

        distance_estimator : BaseDistance
            Unfitted estimator when `distance_from_prior=True`. Otherwise, the
            fitted estimator already updated from the optimizer's input `X`.

        routed_params : Bunch
            Metadata routing for the distance estimator. Its `fit` parameters
            are used when `distance_from_prior=True`.

        Returns
        -------
        distance_matrix : ndarray or DataFrame of shape (n_assets, n_assets)
            Distances aligned with the full input asset order, with NaN rows and
            columns for non-investable assets. Returns a DataFrame with matching
            row and column labels when asset names are available.

        Raises
        ------
        ValueError
            If the fitted distance matrix has an unexpected shape or contains
            non-finite distances between investable assets.

        Notes
        -----
        For a single investable asset, its distance to itself is zero. With
        `distance_from_prior=True`, distance fitting is then skipped and
        `distance_estimator_` is set to None.
        """
        distance_matrix = np.full((self.n_features_in_, self.n_features_in_), np.nan)
        n_investable = np.count_nonzero(investable_mask)
        ix = np.ix_(investable_mask, investable_mask)
        if n_investable < 2:
            distance_matrix[ix] = 0
            if self.distance_from_prior:
                self.distance_estimator_ = None
        else:
            scenario_input = (
                self.distance_from_prior
                and not distance_estimator.requires_covariance_input
            )
            if self.distance_from_prior:
                self.distance_estimator_ = distance_estimator
                extra_params = {}
                if scenario_input:
                    distance_input = return_distribution.returns[:, investable_mask]
                    if get_config()["enable_metadata_routing"]:
                        extra_params = dict.fromkeys(
                            _sample_weight_names(
                                distance_estimator.get_metadata_routing()
                            ),
                            return_distribution.sample_weight,
                        )
                    else:
                        extra_params = _filter_supported_params(
                            distance_estimator,
                            "fit",
                            sample_weight=return_distribution.sample_weight,
                        )
                else:
                    distance_input = return_distribution.covariance.copy()
                    distance_input[~investable_mask, :] = np.nan
                    distance_input[:, ~investable_mask] = np.nan
                if hasattr(self, "feature_names_in_"):
                    names = self.feature_names_in_
                    if scenario_input:
                        names = names[investable_mask]
                    distance_input = pd.DataFrame(
                        distance_input,
                        columns=names,
                        index=None if scenario_input else names,
                    )
                _call_estimator(
                    distance_estimator,
                    "fit",
                    distance_input,
                    routed_params=routed_params,
                    extra_params=extra_params,
                )
            expected_size = n_investable if scenario_input else self.n_features_in_
            distance = distance_estimator.distance_
            if distance.shape != (expected_size, expected_size):
                raise ValueError(
                    "The distance output does not match its input asset schema."
                )
            investable_distance = distance if scenario_input else distance[ix]
            if not np.isfinite(investable_distance).all():
                raise ValueError(
                    "Distances are unavailable for prior-investable assets. "
                    "Align the learners' readiness or use the prior covariance."
                )
            distance_matrix[ix] = investable_distance

        if hasattr(self, "feature_names_in_"):
            return pd.DataFrame(
                distance_matrix,
                index=self.feature_names_in_,
                columns=self.feature_names_in_,
            )
        return distance_matrix

    def _reset(self) -> None:
        """Mark the estimator for initialization on the next `_fit`.

        Removes `weights_`. The next `_fit` replaces the fitted prior, distance
        and seriation estimators through `_initialize`.
        """
        if hasattr(self, _FITTED_ATTR):
            delattr(self, _FITTED_ATTR)

    @abstractmethod
    def _compute_weights(
        self,
        return_distribution: ReturnDistribution,
        ordering: IntArray,
        min_weights: FloatArray,
        max_weights: FloatArray,
    ) -> FloatArray:
        """Compute weights for the investable assets in distribution order.

        Called for at least two investable assets after common input validation
        and updates to the prior, distance and seriation estimators. The base
        validates the returned weights, expands them to the full asset universe
        and assigns `weights_`. Implementations may record algorithm-specific
        diagnostics.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            Distribution restricted to the investable assets.

        ordering : ndarray of shape (n_investable_assets,)
            Permutation of positions in `return_distribution`, specifying the
            traversal order for the allocation algorithm.

        min_weights : ndarray of shape (n_investable_assets,)
            Lower weight bounds in distribution order.

        max_weights : ndarray of shape (n_investable_assets,)
            Upper weight bounds in distribution order.

        Returns
        -------
        weights : ndarray of shape (n_investable_assets,)
            Finite portfolio weights in distribution order, summing to one and
            satisfying the supplied lower and upper bounds.

        Raises
        ------
        OptimizationError
            If the algorithm cannot produce a valid allocation.
        """
        ...


def _sample_weight_names(
    routing: skm.MetadataRequest | skm.MetadataRouter,
    *,
    method: str = "fit",
    use_alias: bool = False,
) -> set[str]:
    """Find the keyword names that deliver sample weights to an estimator.

    Parameters
    ----------
    routing : MetadataRequest or MetadataRouter
        Routing description of the distance estimator or one of its children.

    method : str, default="fit"
        Method receiving the weights. Recursive calls follow the router's method
        mappings.

    use_alias : bool, default=False
        Use the name requested from the parent. A direct call to a consumer uses
        `sample_weight`, while a router receives the aliases requested by its
        children.

    Returns
    -------
    names : set of str
        Keywords requested for sample weights. Unrequested weights are excluded.
    """
    if isinstance(routing, skm.MetadataRequest):
        request = getattr(routing, method).requests.get("sample_weight")
        if request in (None, False, skm.WARN):
            return set()
        if request is True or not use_alias:
            return {"sample_weight"}
        return {request}

    names = set()
    for name, route in routing:
        for caller, callee in route.mapping:
            if caller == method:
                # A router's own consumer keeps the current naming rule.
                # Descendants request their weights from a parent using aliases.
                names.update(
                    _sample_weight_names(
                        route.router,
                        method=callee,
                        use_alias=use_alias or name != "$self_request",
                    )
                )
    return names
