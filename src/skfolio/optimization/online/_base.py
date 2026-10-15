"""Base Online Portfolio Selection estimator."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
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
from skfolio.optimization.online._distribution_constraints import (
    RISK_LIMITS,
    DistributionConstraintConfig,
)
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
    """Base lifecycle for causal online portfolio optimizers.

    Subclasses implement ``_update_one``. ``fit`` resets learned state and
    ``partial_fit`` processes observations sequentially.

    Parameters
    ----------
    store_weights : bool, default=True
        Store diagnostic pre-observation targets. Public estimators default to False.
    initial_weights : array-like, optional
        Initial learning allocation. None uses equal weights, then applies constraints.
    initial_wealth : float or array-like, optional
        Legacy allocation input. An array supplies relative initial asset capital;
        its sum does not create an execution account. Cannot combine array capital
        with array initial_weights. Scalar capital must be positive.
    previous_weights : float, dict or array-like, optional
        Actual pre-trade holdings used as the turnover anchor.
    transaction_costs : float, dict or array-like, default=0.0
        Proportional trading costs used during portfolio evaluation.
    management_fees : float, dict or array-like, default=0.0
        Per-period asset fees used by evaluation and the learning objective.
    groups : array-like or dict, optional
        Asset group labels aligned with fitted asset names.
    linear_constraints : list of str, optional
        Named asset and group equations using the common constraint syntax.
    left_inequality : array-like, optional
        Left-hand matrix for linear weight inequalities.
    right_inequality : array-like, optional
        Right-hand vector paired with left_inequality.
    max_turnover : float, optional
        Maximum L1 trade size relative to explicit pre-trade holdings.
    min_weights : float, dict or array-like, default=0.0
        Lower asset weight bounds.
    max_weights : float, dict or array-like, default=1.0
        Upper asset weight bounds.
    budget : float, default=1.0
        Required sum of asset weights; None disables the equality constraint.
    min_budget : float, optional
        Lower bound on the sum of weights. Requires ``budget=None``.
    max_budget : float, optional
        Upper bound on the sum of weights. Requires ``budget=None``.
    X_tracking : array-like, optional
        Historical returns supplied for an explicit tracking-error constraint.
    tracking_error_benchmark : array-like, optional
        Benchmark returns aligned with X_tracking.
    max_tracking_error : float, optional
        Tracking-error limit computed from the explicit tracking inputs.
    covariance : array-like, optional
        Fixed covariance matrix for a risk constraint; exclusive with prior_estimator.
    max_variance : float, optional
        Maximum portfolio variance using fixed or causal prior covariance.
    min_return : float, optional
        Minimum expected return after costs and fees, using a causal prior.
    max_standard_deviation : float, optional
        Upper standard deviation limit using the causal prior and MeanRisk definition.
    max_semi_variance : float, optional
        Upper semi variance limit using the causal prior and MeanRisk definition.
    max_semi_deviation : float, optional
        Upper semi deviation limit using the causal prior and MeanRisk definition.
    max_mean_absolute_deviation : float, optional
        Upper mean absolute deviation limit using the causal prior and MeanRisk definition.
    max_first_lower_partial_moment : float, optional
        Upper first lower partial moment limit using the causal prior and MeanRisk definition.
    max_worst_realization : float, optional
        Upper worst realization limit using the causal prior and MeanRisk definition.
    max_cvar : float, optional
        Upper cvar limit using the causal prior and MeanRisk definition.
    max_evar : float, optional
        Upper evar limit using the causal prior and MeanRisk definition.
    max_max_drawdown : float, optional
        Upper max drawdown limit using the causal prior and MeanRisk definition.
    max_average_drawdown : float, optional
        Upper average drawdown limit using the causal prior and MeanRisk definition.
    max_cdar : float, optional
        Upper cdar limit using the causal prior and MeanRisk definition.
    max_edar : float, optional
        Upper edar limit using the causal prior and MeanRisk definition.
    max_ulcer_index : float, optional
        Upper ulcer index limit using the causal prior and MeanRisk definition.
    max_gini_mean_difference : float, optional
        Upper gini mean difference limit using the causal prior and MeanRisk definition.
    min_acceptable_return : float, optional
        Downside threshold per asset; None uses the causal prior mean.
    risk_free_rate : float, default=0.0
        Risk-free return used by the shared lower-partial-moment expression.
    cvar_beta : float, default=0.95
        Confidence level for the CVAR limit, between zero and one.
    evar_beta : float, default=0.95
        Confidence level for the EVAR limit, between zero and one.
    cdar_beta : float, default=0.95
        Confidence level for the CDAR limit, between zero and one.
    edar_beta : float, default=0.95
        Confidence level for the EDAR limit, between zero and one.
    prior_estimator : BasePrior, optional
        Incremental prior cloned and updated once per observed row.
    portfolio_params : dict, optional
        Parameters forwarded to Portfolio for prediction and evaluation.
    fallback : estimator, str or list, optional
        Inherited base configuration. Incremental update failures are not recovered
        through fallback; a failed update requires a fresh fit.
    raise_on_failure : bool, default=True
        Inherited base configuration. Incremental update failures always propagate.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Target for the next observation, after processing the latest return.
    initial_weights_ : ndarray of shape (n_assets,)
        Feasible learning allocation before the first observed return.
    n_observations_ : int
        Number of observations successfully processed since the last reset.
    n_features_in_ : int
        Number of assets seen during fitting.
    feature_names_in_ : ndarray of shape (n_assets,)
        Asset names, when fitted with named DataFrame columns.
    all_weights_ : ndarray of shape (n_stored_observations, n_assets)
        Diagnostic targets before each stored observation, available only when
        ``store_weights=True``. This is not a record of executed trades. Use
        ``all_weights_.copy()`` to retain an independent snapshot.
    prior_estimator_ : BasePrior
        Fitted incremental prior, present when ``prior_estimator`` is supplied.

    Notes
    -----
    Statistical estimates, algorithm targets and actual account holdings are
    separate. Portfolio evaluation owns execution and wealth. A failed numerical
    or prior update can partially advance state and requires a fresh ``fit``.
    """

    # Online updates initialize an array and propagate failures instead of using
    # the base fallback path that can assign None to weights_.
    weights_: np.ndarray  # ty: ignore[invalid-mutable-override]

    def __init__(
        self,
        *,
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
        min_budget: float | None = None,
        max_budget: float | None = None,
        X_tracking: npt.ArrayLike | None = None,
        tracking_error_benchmark: npt.ArrayLike | None = None,
        max_tracking_error: float | None = None,
        covariance: npt.ArrayLike | None = None,
        max_variance: float | None = None,
        min_return: float | None = None,
        max_standard_deviation: float | None = None,
        max_semi_variance: float | None = None,
        max_semi_deviation: float | None = None,
        max_mean_absolute_deviation: float | None = None,
        max_first_lower_partial_moment: float | None = None,
        max_worst_realization: float | None = None,
        max_cvar: float | None = None,
        max_evar: float | None = None,
        max_max_drawdown: float | None = None,
        max_average_drawdown: float | None = None,
        max_cdar: float | None = None,
        max_edar: float | None = None,
        max_ulcer_index: float | None = None,
        max_gini_mean_difference: float | None = None,
        min_acceptable_return: float | None = None,
        risk_free_rate: float = 0.0,
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
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
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.X_tracking = X_tracking
        self.tracking_error_benchmark = tracking_error_benchmark
        self.max_tracking_error = max_tracking_error
        self.min_return = min_return
        self.max_standard_deviation = max_standard_deviation
        self.max_semi_variance = max_semi_variance
        self.max_semi_deviation = max_semi_deviation
        self.max_mean_absolute_deviation = max_mean_absolute_deviation
        self.max_first_lower_partial_moment = max_first_lower_partial_moment
        self.max_worst_realization = max_worst_realization
        self.max_cvar = max_cvar
        self.max_evar = max_evar
        self.max_max_drawdown = max_max_drawdown
        self.max_average_drawdown = max_average_drawdown
        self.max_cdar = max_cdar
        self.max_edar = max_edar
        self.max_ulcer_index = max_ulcer_index
        self.max_gini_mean_difference = max_gini_mean_difference
        self.min_acceptable_return = min_acceptable_return
        self.risk_free_rate = risk_free_rate
        self.cvar_beta = cvar_beta
        self.evar_beta = evar_beta
        self.cdar_beta = cdar_beta
        self.edar_beta = edar_beta
        self.prior_estimator = prior_estimator
        self.covariance = covariance
        self.max_variance = max_variance

        self._is_initialized: bool = False
        self._weights_initialized: bool = False
        self._projector: AutoProjector | None = None
        self._t: int = 0
        self._previous_target_: np.ndarray | None = None

    def _validate_constraint_domain(self, n_assets: int) -> None:
        """Apply the same long-only domain to scalar, array and named bounds."""
        bounds = []
        for name, default in (("min_weights", 0.0), ("max_weights", 1.0)):
            raw = getattr(self, name)
            if raw is not None:
                raw_array = np.asarray(
                    list(raw.values()) if isinstance(raw, dict) else raw, dtype=float
                )
                if not np.isfinite(raw_array).all() or np.any(raw_array < 0):
                    raise ValueError(
                        f"{name} must contain finite nonnegative bounds; shorting is not supported"
                    )
            value = self._clean_input(
                value=getattr(self, name),
                n_assets=n_assets,
                fill_value=default,
                name=name,
            )
            array = np.asarray(value, dtype=float)
            if not np.isfinite(array).all() or np.any(array < 0):
                raise ValueError(
                    f"{name} must contain finite nonnegative bounds; shorting is not supported"
                )
            bounds.append(array)
        if np.any(bounds[0] > bounds[1]):
            raise ValueError("min_weights must not exceed max_weights")
        for name in ("budget", "min_budget", "max_budget"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and nonnegative, or None")

    def _distribution_config(self) -> DistributionConstraintConfig | None:
        """Collect distribution limits without consuming observations."""
        limits = {
            m: getattr(self, f"max_{m.value}")
            for m in RISK_LIMITS
            if getattr(self, f"max_{m.value}") is not None
        }
        if self.min_return is None and not limits:
            return None
        config = DistributionConstraintConfig(
            min_return=self.min_return,
            risk_limits=limits,
            min_acceptable_return=self.min_acceptable_return,
            risk_free_rate=self.risk_free_rate,
            cvar_beta=self.cvar_beta,
            evar_beta=self.evar_beta,
            cdar_beta=self.cdar_beta,
            edar_beta=self.edar_beta,
        )
        config.validate()
        if self.prior_estimator is None:
            raise ValueError(
                "Distribution constraints require a causal prior_estimator"
            )
        return config

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
        distribution = (
            self.prior_estimator_.return_distribution_
            if hasattr(self, "prior_estimator_")
            else None
        )
        distribution_config = self._distribution_config()
        if distribution_config is not None:
            ready = distribution is not None and distribution_config.ready(
                distribution, n_assets
            )
            if not ready:
                if (distribution is None and not self._weights_initialized) or getattr(
                    self, "_prior_warmup", False
                ):
                    distribution_config = None
                else:
                    raise ValueError(
                        "The causal prior must be ready before an execution decision"
                    )
            else:
                distribution_config.management_fees = self._clean_input(
                    value=self.management_fees,
                    n_assets=n_assets,
                    fill_value=0.0,
                    name="management_fees",
                )
                distribution_config.transaction_costs = (
                    0.0
                    if getattr(self, "_prior_warmup", False)
                    else self._clean_input(
                        value=self.transaction_costs,
                        n_assets=n_assets,
                        fill_value=0.0,
                        name="transaction_costs",
                    )
                )
                distribution_config.previous_weights = self._execution_anchor
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
            min_budget=self.min_budget,
            max_budget=self.max_budget,
            groups=groups,
            linear_constraints=self.linear_constraints,
            return_distribution=distribution,
            distribution_config=distribution_config,
            # Initial learning references precede the first causal prior update.
            strict_linear_constraints=(
                self.prior_estimator is None or distribution is not None
            ),
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
        self._distribution_config()
        if (self.left_inequality is None) != (self.right_inequality is None):
            raise ValueError(
                "left_inequality and right_inequality must be provided together"
            )
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
        self._validate_constraint_domain(returns.shape[1])
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
                self._append_weight_history(np.asarray(target_list))
        return self

    def _append_weight_history(self, history: np.ndarray) -> None:
        """Append targets with amortized linear storage cost across a stream."""
        size = len(self.all_weights_) if hasattr(self, "all_weights_") else 0
        required = size + len(history)
        buffer = getattr(self, "_weight_history_buffer_", None)
        if buffer is None or required > len(buffer):
            capacity = max(required, 16 if buffer is None else 2 * len(buffer))
            buffer = np.zeros((capacity, history.shape[1]), dtype=history.dtype)
            if size:
                buffer[:size] = self.all_weights_
            self._weight_history_buffer_ = buffer
        buffer[size:required] = history
        self.all_weights_ = buffer[:required]

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

        Use ``partial_fit`` to continue a stream without resetting state.
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
        if self._distribution_config() is not None:
            raise ValueError(
                "Distribution constraints require causal warmup; use online_predict with warmup_size"
            )
        self._validate_params()
        if (self.left_inequality is None) != (self.right_inequality is None):
            raise ValueError(
                "left_inequality and right_inequality must be provided together"
            )
        data = validate_data(self, X, reset=True, dtype=float)
        self._validate_constraint_domain(data.shape[1])
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
        if not self._projector.config.strict_linear_constraints:
            # This allocation will be executed before any prior observation.
            # Unknown factors must fail here, not be deferred as during fit.
            self._projector = AutoProjector(
                replace(self._projector.config, strict_linear_constraints=True)
            )
            self.weights_ = self._projector.project(self.weights_)
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
