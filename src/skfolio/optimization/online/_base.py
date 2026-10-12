"""Base Online Portfolio Selection estimator."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
import sklearn.utils.metadata_routing as skm
from sklearn.base import check_is_fitted, clone
from sklearn.utils.validation import _check_sample_weight, validate_data

import skfolio.typing as skt
from skfolio.optimization._base import BaseOptimization, _has_transaction_cost
from skfolio.optimization.online._mixins import (
    OnlineParameterConstraintsMixin,
)
from skfolio.optimization.online._projection import (
    AutoProjector,
    ProjectionConfig,
)
from skfolio.portfolio import MultiPeriodPortfolio, Portfolio
from skfolio.prior import BasePrior
from skfolio.utils.tools import (
    _call_estimator,
    _check_method_params,
    input_to_array,
    safe_indexing,
)


class BaseOnlineOptimization(BaseOptimization, OnlineParameterConstraintsMixin):
    """Online Portfolio Selection (OPS) base class.

    This class serves as a foundation for implementing various online portfolio
    selection algorithms. It handles the common logic for fitting data sequentially,
    managing weights, and applying projections. Subclasses should implement the
    core update logic.
    """

    weights_: np.ndarray

    def __init__(
        self,
        *,
        warm_start: bool = True,
        store_weights: bool = True,
        initial_weights: npt.ArrayLike | None = None,
        initial_wealth: float | npt.ArrayLike | None = None,
        previous_weights: skt.MultiInput | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        X_tracking: npt.ArrayLike | None = None,
        tracking_error_benchmark: npt.ArrayLike | None = None,
        max_tracking_error: float | None = None,
        covariance: npt.ArrayLike | None = None,
        max_variance: float | None = None,
        prior_estimator: BasePrior | None = None,
        portfolio_params: dict | None = None,
        fallback: skt.Fallback = None,
        raise_on_failure: bool = True,
    ) -> None:
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
        )
        self.warm_start = warm_start
        self.store_weights = store_weights
        self.initial_weights = initial_weights
        self.initial_wealth = initial_wealth
        self.previous_weights = previous_weights
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.max_turnover = max_turnover
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.X_tracking = X_tracking
        self.tracking_error_benchmark = tracking_error_benchmark
        self.max_tracking_error = max_tracking_error
        self.prior_estimator = prior_estimator
        self.covariance = covariance
        self.max_variance = max_variance

        self._is_initialized: bool = False
        self._weights_initialized: bool = False
        self._projector: AutoProjector | None = None
        self._t: int = 0
        self._previous_target_: np.ndarray | None = None

    def _initialize_projector(self) -> AutoProjector:
        """Build the projector from aligned constraints and available causal estimates."""
        n_assets = self.n_features_in_
        names = getattr(self, "feature_names_in_", None)
        groups = self.groups
        if groups is not None:
            groups = input_to_array(
                items=groups,
                n_assets=n_assets,
                fill_value="",
                dim=2,
                assets_names=names,
                name="groups",
            )
        elif names is not None and self.linear_constraints is not None:
            groups = np.asarray([names])
        covariance = self.covariance
        if self.max_variance is not None and hasattr(self, "prior_estimator_"):
            distribution = self.prior_estimator_.return_distribution_
            covariance = np.asarray(distribution.covariance)
            if covariance.shape != (n_assets, n_assets):
                raise ValueError(
                    "The causal prior covariance must match the fixed universe"
                )
            if not np.isfinite(covariance).all():
                if getattr(self, "_prior_warmup", False):
                    covariance = None
                else:
                    raise ValueError(
                        "The causal prior must supply a finite covariance before an execution decision"
                    )
        projection_config = ProjectionConfig(
            lower=self._clean_input(
                value=self.min_weights,
                n_assets=n_assets,
                fill_value=0.0,
                name="min_weights",
            ),
            upper=self._clean_input(
                value=self.max_weights,
                n_assets=n_assets,
                fill_value=1.0,
                name="max_weights",
            ),
            budget=self.budget,
            groups=groups,
            linear_constraints=self.linear_constraints,
            left_inequality=self.left_inequality,
            right_inequality=self.right_inequality,
            X_tracking=self.X_tracking,
            tracking_error_benchmark=self.tracking_error_benchmark,
            max_tracking_error=self.max_tracking_error,
            covariance=covariance,
            variance_bound=self.max_variance if covariance is not None else None,
            previous_weights=self._execution_anchor,
            max_turnover=self._decision_turnover,
        )
        return AutoProjector(projection_config)

    def _initialize_weights(self, num_assets: int) -> None:
        """Initialize an algorithm reference, without constructing an account."""
        initial = self.initial_weights
        if self.initial_wealth is not None:
            wealth = np.asarray(self.initial_wealth, dtype=float)
            if not np.isfinite(wealth).all() or np.any(wealth < 0) or wealth.sum() <= 0:
                raise ValueError("initial_wealth must be finite and positive")
            if wealth.ndim:
                if initial is not None:
                    raise ValueError(
                        "initial_weights and initial_wealth cannot both be arrays"
                    )
                if wealth.shape != (num_assets,):
                    raise ValueError(f"initial_wealth must have shape ({num_assets},)")
                initial = wealth / wealth.sum()
        if initial is None:
            initial = np.full(num_assets, 1 / num_assets)
        initial = np.asarray(
            self._clean_input(
                value=initial,
                n_assets=num_assets,
                fill_value=0.0,
                name="initial_weights",
            ),
            dtype=float,
        )
        if initial.ndim == 0:
            initial = np.full(num_assets, float(initial))
        if initial.shape != (num_assets,) or not np.isfinite(initial).all():
            raise ValueError(
                "initial_weights has incompatible shape or nonfinite values"
            )
        self.weights_ = initial.copy()
        self._weights_initialized = True

    def _apply_initial_projection(self) -> None:
        """Project the initial algorithm reference under static constraints."""
        if self._projector is None or not self._weights_initialized:
            return
        if self._t != 0 or self._previous_target_ is not None:
            return
        # Initial algorithm reference is not an executed allocation.
        static = AutoProjector(
            replace(self._projector.config, previous_weights=None, max_turnover=None)
        )
        self.weights_ = static.project(self.weights_)

    def _compute_effective_relatives(self, gross_relatives: np.ndarray) -> np.ndarray:
        """Use additive fees in fee-aware learning, matching Portfolio returns.

        This adjusts the decision objective only. Actual fees are booked exclusively
        by Portfolio. No holdings or wealth are computed here.
        """
        return gross_relatives - getattr(self, "_management_fees_arr", 0.0)

    _requires_single_period_evaluation = True

    def partial_fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        sample_weight: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> BaseOnlineOptimization:
        """Continue learning from one observation or a causal block of net returns.

        The target after observing row t is intended for period t+1. Validation
        covers the complete block before any numerical update. Targets recorded in
        ``all_weights_`` are diagnostic learning history, not trades.
        """
        if getattr(self, "_needs_reset", False):
            raise RuntimeError(
                "A previous update failed; call fit to reset before reuse"
            )
        self._validate_params()
        if self.prior_estimator is not None and self.covariance is not None:
            raise ValueError("Use prior_estimator or covariance, not both")
        if self.prior_estimator is not None and not callable(
            getattr(self.prior_estimator, "partial_fit", None)
        ):
            raise TypeError(
                "prior_estimator must implement partial_fit for causal updates"
            )
        if (
            self.max_variance is not None
            and self.prior_estimator is None
            and self.covariance is None
        ):
            raise ValueError(
                "max_variance requires covariance or a causal prior_estimator"
            )
        metadata_params = dict(fit_params)
        if self.prior_estimator is not None and sample_weight is not None:
            metadata_params["sample_weight"] = sample_weight
        routed = skm.process_routing(self, "partial_fit", **metadata_params)

        if not hasattr(X, "columns"):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        returns = validate_data(
            self,
            X=X,
            reset=not self._is_initialized,
            dtype=float,
            ensure_2d=True,
            accept_sparse=False,
        )
        if np.any(returns <= -1):
            raise ValueError(
                "Online learning requires returns greater than -1; bankrupt assets "
                "require an explicit universe reset with fit"
            )
        if y is not None and np.shape(y)[0] != len(returns):
            raise ValueError("y must have one row per return observation")
        if sample_weight is not None:
            _check_sample_weight(sample_weight, returns)
            if self.prior_estimator is None:
                warnings.warn(
                    "sample_weight is ignored for online estimators without a prior.",
                    UserWarning,
                    stacklevel=2,
                )
        warmup = getattr(self, "_in_warmup", False)
        if (
            len(returns) > 1
            and not warmup
            and (
                self.max_turnover is not None
                or _has_transaction_cost(self.transaction_costs)
            )
        ):
            raise ValueError(
                "Execution-constrained blocks require row-aligned holdings; "
                "use fit for warmup or one-row partial_fit for each rebalance"
            )
        if self.prior_estimator is not None:
            _check_method_params(X=returns, params=routed.prior_estimator.partial_fit)
        target_list = []
        try:
            for i, row in enumerate(1.0 + returns):
                decision = not warmup or i == len(returns) - 1
                self._prior_warmup = not decision
                self._decision_turnover = self.max_turnover if decision else None
                self._execution_anchor = (
                    None
                    if not decision or self.previous_weights is None
                    else self._clean_previous_weights(returns.shape[1]).copy()
                )
                if self._execution_anchor is not None:
                    if not np.isfinite(self._execution_anchor).all():
                        raise ValueError(
                            "previous_weights must contain finite holdings"
                        )
                    self._execution_anchor.setflags(write=False)
                if (
                    self._decision_turnover is not None
                    and self._execution_anchor is None
                ):
                    raise ValueError(
                        "max_turnover requires explicit previous_weights holdings"
                    )
                if self.prior_estimator is not None:
                    if not self._is_initialized:
                        self._initialize_algorithm(returns.shape[1])
                    if not hasattr(self, "prior_estimator_"):
                        self.prior_estimator_ = clone(self.prior_estimator)
                    row_slice = slice(i, i + 1)
                    metadata = _check_method_params(
                        X=returns,
                        params=routed.prior_estimator.partial_fit,
                        indices=row_slice,
                    )
                    _call_estimator(
                        estimator=self.prior_estimator_,
                        method="partial_fit",
                        X=safe_indexing(X, row_slice),
                        y=None if y is None else safe_indexing(y, row_slice),
                        extra_params=metadata,
                    )
                self._update_one(row)
                if (
                    np.shape(self.weights_) != (returns.shape[1],)
                    or not np.isfinite(self.weights_).all()
                ):
                    raise RuntimeError("Online update produced invalid target weights")
                assert self._previous_target_ is not None
                if not hasattr(self, "initial_weights_"):
                    self.initial_weights_ = self._previous_target_.copy()
                self.n_observations_ = self._t
                if self.store_weights:
                    target_list.append(self._previous_target_.copy())
        except Exception:
            # Engines/predictors may already have consumed the failing observation.
            # Do not pretend that keeping the previous public target rolls them back.
            self._needs_reset = True
            raise
        finally:
            if target_list:
                history = np.asarray(target_list)
                self.all_weights_ = (
                    np.vstack((self.all_weights_, history))
                    if hasattr(self, "all_weights_")
                    else history
                )
        return self

    @contextmanager
    def _warmup_context(self) -> Generator[None, None, None]:
        """Mark historical rows as learning-only until the final entry decision."""
        old = getattr(self, "_in_warmup", False)
        self._in_warmup = True
        try:
            yield
        finally:
            self._in_warmup = old

    def get_metadata_routing(self) -> skm.MetadataRouter:
        """Route both entry points to the prior's incremental update method."""
        return skm.MetadataRouter(owner=type(self).__name__).add(
            prior_estimator=self.prior_estimator,
            method_mapping=skm.MethodMapping()
            .add(caller="fit", callee="partial_fit")
            .add(caller="partial_fit", callee="partial_fit"),
        )

    def _update_one(self, gross_relatives: np.ndarray) -> None:
        """Observe one return vector using the existing strategy machinery."""
        raise NotImplementedError

    def _reset_state_for_fit(self) -> None:
        """Clear learned state; subclasses also reset their numerical components."""
        for name in list(vars(self)):
            if name.endswith("_") and not name.startswith("__"):
                delattr(self, name)
        self._needs_reset = False
        self._prior_warmup = False
        self._weights_initialized = False
        self._is_initialized = False
        self._projector = None
        self._t = 0
        self._previous_target_ = None

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike | None = None,
        **fit_params: Any,
    ) -> BaseOnlineOptimization:
        """Reset learned state and process observations causally.

        ``warm_start`` is temporarily accepted for source compatibility but no
        longer controls this method. Use ``partial_fit`` to continue a stream.
        """
        self._reset_state_for_fit()
        with self._warmup_context():
            return self.partial_fit(X, y, **fit_params)

    def _clean_input(
        self,
        value: float | dict | npt.ArrayLike | None,
        n_assets: int,
        fill_value: float,
        name: str,
        *,
        apply_investable_mask: bool = True,
    ) -> float | np.ndarray:
        """Use common asset alignment and validate per-period management fees."""
        cleaned = super()._clean_input(
            value=value,
            n_assets=n_assets,
            fill_value=fill_value,
            name=name,
            apply_investable_mask=apply_investable_mask,
        )
        if name == "management_fees" and np.any(np.asarray(cleaned) >= 1.0):
            raise ValueError("All management_fees values must be < 1.0 per period")
        return cleaned

    def _initialize_algorithm(self, n_assets: int) -> None:
        """Initialize numerical components without observing returns."""
        raise NotImplementedError

    def _synchronize_target(self) -> None:
        """Use the final decision as the next local update center."""
        for name in ("_foco_engine", "_engine"):
            engine = getattr(self, name, None)
            if engine is not None and hasattr(engine, "_x_t"):
                engine._x_t = self.weights_.copy()
        strategy = getattr(self, "_strategy_impl", None)
        if strategy is not None and hasattr(strategy, "_mu"):
            strategy._mu = self.weights_.copy()

    def predict(self, X: npt.ArrayLike) -> Portfolio:
        """Evaluate a target using skfolio's common accounting implementation."""
        if getattr(self, "_needs_reset", False):
            raise RuntimeError(
                "A previous update failed; call fit to reset before reuse"
            )
        portfolio = super().predict(X)
        assert isinstance(portfolio, Portfolio)
        return portfolio

    def _initialize_for_evaluation(self, X: npt.ArrayLike) -> None:
        """Initialize before the first return without consuming that observation."""
        self._validate_params()
        data = validate_data(self, X, reset=True, dtype=float)
        self._decision_turnover = self.max_turnover
        self._execution_anchor = (
            None
            if self.previous_weights is None
            else self._clean_previous_weights(data.shape[1]).copy()
        )
        if self.max_turnover is not None and self._execution_anchor is None:
            raise ValueError("max_turnover requires explicit previous_weights holdings")
        self._initialize_algorithm(data.shape[1])
        assert self._projector is not None
        if self.max_turnover is not None:
            self.weights_ = self._projector.project(self.weights_)
        self._synchronize_target()
        self.initial_weights_ = self.weights_.copy()
        self.n_observations_ = 0

    def fit_predict(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params: Any
    ) -> MultiPeriodPortfolio:
        """Evaluate from the initial allocation, with no warmup, using Portfolio.

        For a historical learning window followed by evaluation, use online_predict.
        Initial wealth, if supplied, configures the algorithm reference only; the
        returned portfolio uses skfolio's normalised return/wealth convention.
        """
        if fit_params:
            raise TypeError(
                "Use model_selection.online_predict for routed evaluation metadata"
            )

        from skfolio.model_selection._online._validation import _continue_online_predict

        self._reset_state_for_fit()
        previous = self.previous_weights
        try:
            self._initialize_for_evaluation(X)
            return _continue_online_predict(self, X, y, refit_last=True)
        except Exception:
            self._needs_reset = True
            raise
        finally:
            self.previous_weights = previous

    def predict_online(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> MultiPeriodPortfolio:
        """Continue evaluation on a fitted copy through common model selection."""
        from skfolio.model_selection._online._validation import _continue_online_predict

        check_is_fitted(self, "weights_")
        if getattr(self, "_needs_reset", False):
            raise RuntimeError(
                "A previous update failed; call fit to reset before reuse"
            )
        return _continue_online_predict(deepcopy(self), X, y, refit_last=False)


# Temporary compatibility alias for Carlo's imports and downstream experiments.
OnlinePortfolioSelection = BaseOnlineOptimization
