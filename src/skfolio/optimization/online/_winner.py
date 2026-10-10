"""Follow-The-Winner online portfolio selection."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import ClassVar, Literal

import numpy as np
import numpy.typing as npt
from sklearn.utils._param_validation import Interval, StrOptions

import skfolio.typing as skt
from skfolio.measures._enums import ExtraRiskMeasure, PerfMeasure, RiskMeasure
from skfolio.optimization.online._autograd_objectives import create_objective
from skfolio.optimization.online._base import BaseOnlineOptimization
from skfolio.optimization.online._foco import (
    FirstOrderOCO,
    Predictor,
    SwordMeta,
)
from skfolio.optimization.online._learning_rate import get_auto_learning_rate
from skfolio.optimization.online._mirror_maps import (
    AdaptiveMahalanobisMap,
    AdaptiveVariationMap,
    BaseMirrorMap,
    BurgMirrorMap,
    CompositeMirrorMap,
    DynamicMirrorMap,
    EntropyMirrorMap,
    EuclideanMirrorMap,
    make_ada_barrons_mirror_map,
)
from skfolio.optimization.online._mixins import (
    FTWStrategy,
    OnlineParameterConstraintsMixin,
)
from skfolio.optimization.online._prediction import LastGradPredictor, SmoothPredictor
from skfolio.optimization.online._projection import IdentityProjector
from skfolio.optimization.online._utils import CLIP_EPSILON
from skfolio.prior import BasePrior


class FollowTheWinner(BaseOnlineOptimization):
    """FollowTheWinner: Online Portfolio Selection via Online Convex Optimization and momentum-based strategies.

    It unifies FTRL and OMD through the lens of FTRL-Proximal, which is a unified framework for FTRL and OMD.

    This family of algorithms increases weights on recently successful assets,
    implementing first-order OCO methods including:
    - Mirror Descent family: OGD, EG, PROD (Soft-Bayes)
    - Adaptive methods: AdaGrad, AdaBARRONS
    - Optimistic methods: Smooth Prediction, SWORD family

    Projection:
    - Fast path: box + budget (+ turnover) via project_box_and_sum/project_with_turnover
    - Rich constraints (groups/linear/tracking error/variance): fallback to project_convex

    Data conventions
    ----------------
    - Inputs ``X`` to :meth:`fit` and :meth:`partial_fit` must be NET returns
        (i.e., arithmetic returns r_t in [-1, +inf)). Internally, the estimator
        converts each row to gross returns via ``1.0 + r_t`` before computing
        losses/gradients. This keeps interfaces consistent with most skfolio
        preprocessing pipelines that output net returns.

    Fitting
    -------
    - :meth:`fit` iterates over ``X`` row by row, processing each observation causally through the shared learning loop.

    References
    ----------
    .. [1] Orseau, L., Lattimore, T., & Legg, S. (2017). Soft-Bayes: Prod for
           Mixtures of Experts with Log-Loss. In Algorithmic Learning Theory,
           PMLR 76:73-90.
    """

    cumulative_loss_: float

    _parameter_constraints: ClassVar[dict] = {
        **OnlineParameterConstraintsMixin._parameter_constraints,
        "strategy": [StrOptions({m.value.lower() for m in FTWStrategy})],
        "learning_rate": [
            Interval(Real, 0, None, closed="neither"),
            callable,
            StrOptions({"auto"}),
        ],
        "learning_rate_scale": [StrOptions({"theory", "moderate", "empirical"})],
        "objective": [
            StrOptions({m.value.lower() for m in RiskMeasure}),
            StrOptions({m.value.lower() for m in PerfMeasure}),
            None,
        ],
        "update_mode": [StrOptions({"omd", "ftrl"})],
        "grad_predictor": [StrOptions({"last", "smooth"}), None],
        "smooth_epsilon": [Interval(Real, 0, None, closed="neither")],
        "adagrad_D": [Interval(Real, 0, None, closed="neither"), "array-like", None],
        "adagrad_eps": [Interval(Real, 1e-12, 1e-3, closed="both"), None],
        "eg_tilde": ["boolean"],
        "eg_tilde_alpha": [Interval(Real, 0, 1, closed="both"), callable, None],
        "adabarrons_barrier_coef": [Interval(Real, 0, None, closed="neither")],
        "adabarrons_alpha": [Interval(Real, 0, None, closed="neither")],
        "adabarrons_euclidean_coef": [Interval(Real, 0, None, closed="neither")],
        "adabarrons_beta": [Interval(Real, 0, None, closed="neither")],
        "discount": [Interval(Real, 0, 1, closed="right"), None],
        "demean_gradient": ["boolean"],
        "gradient_lookback": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        strategy: FTWStrategy = FTWStrategy.EG,
        *,
        objective: RiskMeasure | ExtraRiskMeasure | PerfMeasure | None = None,
        learning_rate: float | Literal["auto"] | Callable[[int], float] = "auto",
        learning_rate_scale: Literal["theory", "moderate", "empirical"] = "theory",
        update_mode: Literal["ftrl", "omd"] = "ftrl",
        warm_start: bool = True,
        store_weights: bool = True,
        initial_weights: npt.ArrayLike | None = None,
        initial_wealth: float | npt.ArrayLike | None = None,
        previous_weights: skt.MultiInput | None = None,
        grad_predictor: Literal["last", "smooth"] | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        ## Smooth Prediction
        smooth_epsilon: float = 1.0,
        ## AdaGrad
        adagrad_D: float | npt.ArrayLike | None = None,
        adagrad_eps: float = 1e-8,
        ## Ada-BARRONS
        adabarrons_barrier_coef: float = 1.0,
        adabarrons_alpha: float = 1.0,
        adabarrons_euclidean_coef: float = 1.0,
        adabarrons_beta: float = 0.1,
        eg_tilde: bool = False,
        eg_tilde_alpha: float | Callable[[int], float] = 0.1,
        ## Gradient enhancements
        discount: float | None = None,
        demean_gradient: bool = False,
        gradient_lookback: int = 1,
        # Projection constraints (fast path)
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        # Rich constraints (fallback)
        X_tracking: npt.ArrayLike | None = None,
        tracking_error_benchmark: npt.ArrayLike | None = None,
        max_tracking_error: float | None = None,
        covariance: npt.ArrayLike | None = None,
        max_variance: float | None = None,
        prior_estimator: BasePrior | None = None,
        use_autograd: bool = False,
        portfolio_params: dict | None = None,
    ) -> None:
        r"""Follow-The-Winner online portfolio selection estimator.

        Parameters
        ----------
        strategy : FTWStrategy,  default=FTWStrategy.EG
            Follow-the-Winner strategy to use. Available strategies:

            - 'ogd': Online Gradient Descent (Euclidean geometry)
            - 'eg': Exponential Gradient (entropy geometry)
            - 'prod': Soft-Bayes Product algorithm (Burg entropy)
            - 'adagrad': Adaptive Gradient with diagonal preconditioning
            - 'adabarrons': Ada-BARRONS with adaptive barrier and second-order updates
            - 'sword_var': SWORD-Var (variation-adaptive)
            - 'sword_small': SWORD-Small (AdaGrad geometry with optimistic gradients)
            - 'sword_best': SWORD-Best (meta-aggregation of Var + Small)
            - 'sword_pp': SWORD++ (meta-aggregation of Var + Small + EG)

        objective : RiskMeasure | ExtraRiskMeasure | PerfMeasure | None, default=None
            Objective function to minimize. If None (default), uses the standard
            log-wealth objective (maximizing cumulative logarithmic returns).

            - For RiskMeasure or ExtraRiskMeasure: Minimizes the risk measure on portfolio returns.
            - For PerfMeasure: Maximizes the performance measure (minimizes its negative).

            The gradient is computed using automatic differentiation (autograd).
            Log-wealth (default) uses analytical gradients for better performance.

            .. note::
                Only convex objectives guarantee convergence to global optimum.
                Non-convex measures may converge to local optima.

        learning_rate : float | Callable[[int], float] | "auto", default="auto"
            Learning rate :math:`\eta_t` for the optimization step.

            - float: Constant :math:`\eta`
            - Callable: Time-varying :math:`\eta(t)`
            - **"auto"** (default): Empirically-validated rate per strategy (scale set by learning_rate_scale)

            For EG, the learning rate acts as an inverse temperature parameter.
            As :math:`\eta \to 0`, converges to uniform weights; as :math:`\eta \to \infty`,
            converges to best-performing asset (one-hot).

        learning_rate_scale : {"theory", "moderate", "empirical"}, default="theory"
            Scaling mode for automatic learning rates (only used when
            ``learning_rate="auto"``):

            - **"theory"**: √(log(n)/(t+1)) — Standard OCO bound (Hazan 2016)
            - **"moderate"**: √(8·log(n)/(t+1)) — 2√2 boost (Hazan's book constant)
            - **"empirical"**: √(2·log(n)/(t+1)) — Mild √2 boost over theory

            All three scales preserve the same O(√(T log n)) regret guarantee
            (up to constants).  The "theory" rate is the most conservative and
            recommended default.


        update_mode : {"ftrl", "omd"}, default="ftrl"
            Update rule to use:

            - "omd": Online Mirror Descent (local Bregman/proximal step)
            - "ftrl": Follow-the-Regularized-Leader (global cumulative minimization)

        warm_start : bool, default=True
            Legacy compatibility parameter. ``fit`` always resets; use ``partial_fit`` to continue.

        initial_weights : array-like of shape (n_assets,) | None, default=None
            Initial portfolio weights. If None, uses uniform weights (1/n_assets).

        store_weights : bool, default=True
            Record diagnostic algorithm targets in ``all_weights_``. Set False for
            bounded-memory streaming. These targets do not imply executed trades.

        initial_wealth : float | array-like of shape (n_assets,) | None, default=None
            Legacy allocation initializer. An array defines proportional initial
            weights and cannot be combined with ``initial_weights``. A positive
            scalar does not affect normalized returns. No account or wealth path
            is maintained by this estimator; use portfolio evaluation for that.

        previous_weights : float | dict[str, float] | array-like of shape (n_assets,) | None, default=None
            Previous weights for computing transaction costs and turnover.
            If float, applied to all assets. If dict, maps asset names to weights.

        grad_predictor : {"last", "smooth"} | None, default=None
            Gradient predictor for optimistic updates:

            - "last": Use last gradient as predictor (optimistic OMD)
            - "smooth": Smooth prediction with log-barrier regularization
            - None: No prediction (standard update)

        transaction_costs : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
            Proportional transaction costs per asset. Added as L1 penalty on turnover.
            If float, applied to all assets. If dict, maps asset names to costs.

        management_fees : float | dict[str, float] | array-like of shape (n_assets,), default=0.0
            Management fees per asset (per-period). Modeled as multiplicative drag
            on gross relatives: gradients use net-of-fee relatives.
            If float, applied to all assets. If dict, maps asset names to fees.

        groups : dict[str, list[str]] | array-like of shape (n_groups, n_assets) | None, default=None
            Asset groups for linear constraints. If dict, maps asset names to group labels.

        linear_constraints : list[str] | None, default=None
            Linear constraints on weights (e.g., ``"Equity >= 0.5"``).
            References asset names or group names from `groups`.

        left_inequality : array-like of shape (n_constraints, n_assets) | None, default=None
            Left-hand side matrix :math:`A` for inequality :math:`A w \leq b`.

        right_inequality : array-like of shape (n_constraints,) | None, default=None
            Right-hand side vector :math:`b` for inequality :math:`A w \leq b`.

        max_turnover : float | None, default=None
            Maximum allowed turnover per period (L1 norm of weight changes).

        smooth_epsilon : float, default=1.0
            Log-barrier regularization for Smooth Prediction method.
            Only used with ``grad_predictor="smooth"``.

        adagrad_D : float | array-like of shape (n_assets,) | None, default=None
            Diameter bound(s) for AdaGrad. If float, same for all assets.
            If None, defaults to :math:`\sqrt{2}` (simplex diameter).
            Only used with ``strategy="adagrad"``.

        adagrad_eps : float, default=1e-8
            Numerical stability constant for AdaGrad denominator.
            Only used with ``strategy="adagrad"``.

        adabarrons_barrier_coef : float, default=1.0
            Barrier coefficient for Ada-BARRONS weight-proximity adaptation.
            Only used with ``strategy="adabarrons"``.

        adabarrons_alpha : float, default=1.0
            Barrier scaling parameter for Ada-BARRONS.
            Only used with ``strategy="adabarrons"``.

        adabarrons_euclidean_coef : float, default=1.0
            Euclidean regularization coefficient for Ada-BARRONS.
            Only used with ``strategy="adabarrons"``.

        adabarrons_beta : float, default=0.1
            Second-order curvature coefficient for Ada-BARRONS full quadratic component.
            Only used with ``strategy="adabarrons"``.

        eg_tilde : bool, default=False
            Whether to apply EG-tilde mixing (uniform weight mixing) for EG strategy.
            Only used with ``strategy="eg"``.

        eg_tilde_alpha : float | Callable[[int], float], default=0.1
            Mixing coefficient for EG-tilde. Final weights are
            :math:`(1 - \\alpha) w_{EG} + \\alpha / n`.
            Can be constant or callable. Only used if ``eg_tilde=True``.

        discount : float | None, default=None
            Gradient discount factor :math:`\\gamma \\in (0, 1]` for discounted
            FTRL. When set, the cumulative gradient sum decays exponentially:

            .. math:: G_t = \\gamma \\, G_{t-1} + g_t

            giving recent observations more influence (effective memory window
            :math:`\\approx 1/(1-\\gamma)` rounds). Useful for non-stationary
            markets with regime changes.

            - ``None`` (default): standard FTRL with full cumulative sum
            - ``0.99``: ~100-day effective memory
            - ``0.95``: ~20-day effective memory

            When taken on its own, discounting slightly hurts EG because the smaller
            gradient magnitude pushes weights toward 1/n.
            Combine with ``gradient_lookback`` and a moderate constant
            ``learning_rate`` for best results.

            References: Cesa-Bianchi & Lugosi (2006), *Prediction, Learning, and Games*.

        demean_gradient : bool, default=False
            Whether to subtract the cross-sectional mean from the gradient
            before passing it to the optimization engine.

            .. note::
                This is **provably a no-op** for entropy-based mirror maps (EG,
                PROD) because softmax is shift-invariant:
                :math:`\\text{softmax}(z - c \\mathbf{1}) = \\text{softmax}(z)`.
                It is also nearly a no-op for simplex-projected OGD since the
                projection absorbs additive constants. Kept as parameter for
                non-simplex or experimental use cases.

        gradient_lookback : int, default=1
            Number of past gradients to average before passing to the OCO engine.
            When ``gradient_lookback=1`` (default), uses only the current gradient
            (standard behavior). When ``gradient_lookback=W``, averages the last
            :math:`W` gradients, capturing medium-term momentum signals.

            **Empirically validated**: ``gradient_lookback`` in ``[60, 120]``
            (3-6 month momentum) consistently turns EG from a UCRP-equivalent
            into a modest but reliable momentum strategy, achieving decent
            log-wealth above UCRP on average across 7 real datasets.

            References: Jegadeesh & Titman (1993), "Returns to Buying Winners
            and Selling Losers".

        min_weights : float | dict[str, float] | array-like of shape (n_assets,) | None, default=0.0
            Minimum weight per asset (lower bound). If float, applied to all assets.

        max_weights : float | dict[str, float] | array-like of shape (n_assets,) | None, default=1.0
            Maximum weight per asset (upper bound). If float, applied to all assets.

        budget : float | None, default=1.0
            Investment budget (sum of weights). If None, no budget constraint.

        X_tracking : array-like of shape (n_observations, n_assets) | None, default=None
            Tracking portfolio returns for tracking error constraint.

        tracking_error_benchmark : array-like of shape (n_observations,) | None, default=None
            Benchmark returns for tracking error calculation.

        max_tracking_error : float | None, default=None
            Maximum allowed tracking error (standard deviation of tracking difference).

        covariance : array-like of shape (n_assets, n_assets) | None, default=None
            Covariance matrix for variance constraint.

        max_variance : float | None, default=None
            Maximum allowed portfolio variance.

        prior_estimator : BasePrior or None, default=None
            Optional incremental statistical estimator. Each observed row is passed
            to ``partial_fit`` before computing the next decision. A variance bound
            requires finite covariance at execution time. No prior is created when None.

        use_autograd : bool, default=False
            Whether to use autograd for computation of objective measures with respect to weights. Only for debugging purposes.

        portfolio_params : dict | None, default=None
            Additional portfolio parameters passed to base class.

        """
        super().__init__(
            warm_start=warm_start,
            store_weights=store_weights,
            initial_weights=initial_weights,
            initial_wealth=initial_wealth,
            previous_weights=previous_weights,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            X_tracking=X_tracking,
            tracking_error_benchmark=tracking_error_benchmark,
            max_tracking_error=max_tracking_error,
            covariance=covariance,
            max_variance=max_variance,
            prior_estimator=prior_estimator,
            portfolio_params=portfolio_params,
        )

        # Public configuration
        self.strategy = strategy
        self.objective = objective
        self.learning_rate = learning_rate
        self.learning_rate_scale = learning_rate_scale
        self.update_mode = update_mode
        self.smooth_epsilon = smooth_epsilon
        self.adagrad_D = adagrad_D
        self.adagrad_eps = adagrad_eps
        self.adabarrons_barrier_coef = adabarrons_barrier_coef
        self.adabarrons_alpha = adabarrons_alpha
        self.adabarrons_euclidean_coef = adabarrons_euclidean_coef
        self.adabarrons_beta = adabarrons_beta
        self.eg_tilde = eg_tilde
        self.eg_tilde_alpha = eg_tilde_alpha
        self.discount = discount
        self.demean_gradient = demean_gradient
        self.gradient_lookback = gradient_lookback
        self.warm_start = warm_start
        self.initial_weights = initial_weights
        self.grad_predictor = grad_predictor

        # Costs and fees (public attributes preserved for predict())
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees

        # Projection parameters
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.previous_weights = previous_weights
        self.max_turnover = max_turnover

        # Rich constraints
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.X_tracking = X_tracking
        self.tracking_error_benchmark = tracking_error_benchmark
        self.max_tracking_error = max_tracking_error
        self.covariance = covariance
        self.max_variance = max_variance
        self.use_autograd = use_autograd

        # Internal state (initialized deterministically)
        self._foco_engine: FirstOrderOCO | SwordMeta | None = None
        self.cumulative_loss_: float = 0.0

    def _ensure_initialized(self, gross_relatives: np.ndarray) -> None:
        """Initialize the numerical kernel and refresh decision constraints."""
        num_assets = int(gross_relatives.shape[0])

        if not self._is_initialized:
            self.n_features_in_ = num_assets

        self._transaction_costs_arr = self._clean_input(
            self.transaction_costs,
            n_assets=num_assets,
            fill_value=0,
            name="transaction_costs",
        )

        self._management_fees_arr = self._clean_input(
            self.management_fees,
            n_assets=num_assets,
            fill_value=0,
            name="management_fees",
        )

        # Resolve transaction costs and management fees once number of assets is known

        self._projector = self._initialize_projector()
        if self._foco_engine is not None:
            self._foco_engine.projector = self._projector

        # Initialize objective function
        if not hasattr(self, "_objective_fn"):
            self._objective_fn = create_objective(
                self.objective, use_autograd=self.use_autograd
            )

        if self._foco_engine is None:
            mirror_map: BaseMirrorMap | None = None
            predictor: Predictor | None = None

            # Resolve learning rate if "auto"
            effective_learning_rate = self.learning_rate
            if self.learning_rate == "auto":
                effective_learning_rate = get_auto_learning_rate(
                    strategy=self.strategy,
                    n_assets=num_assets,
                    min_weights=self._clean_input(
                        self.min_weights, num_assets, 0.0, "min_weights"
                    ),
                    max_weights=self._clean_input(
                        self.max_weights, num_assets, 1.0, "max_weights"
                    ),
                    budget=1.0 if self.budget is None else self.budget,
                    objective=self.objective,  # pass enum directly
                    gradient_bound=None,  # use automatic estimation
                    historical_returns=None,  # no historical data at init
                    scale=self.learning_rate_scale,
                )

            assert not isinstance(effective_learning_rate, str)
            match self.grad_predictor:
                case "last":
                    predictor = LastGradPredictor()
                case "smooth":
                    predictor = SmoothPredictor()
                case None:
                    predictor = None

            match self.strategy:
                case FTWStrategy.EG | "eg":
                    mirror_map = EntropyMirrorMap()
                case FTWStrategy.OGD | "ogd":
                    mirror_map = EuclideanMirrorMap()
                case FTWStrategy.PROD | "prod":
                    mirror_map = BurgMirrorMap()
                case FTWStrategy.ADAGRAD | "adagrad":
                    mirror_map = AdaptiveMahalanobisMap(eps=self.adagrad_eps)
                case FTWStrategy.ADABARRONS | "adabarrons":
                    # Ada-BARRONS uses a compositional mirror map:
                    # - AdaBarronsBarrierMap: weight-proximity adaptive barrier
                    # - EuclideanMap: Euclidean regularization
                    # - FullQuadraticMap: second-order curvature
                    mirror_map = make_ada_barrons_mirror_map(
                        d=num_assets,
                        barrier_coef=self.adabarrons_barrier_coef,
                        alpha=self.adabarrons_alpha,
                        euclidean_coef=self.adabarrons_euclidean_coef,
                        beta=self.adabarrons_beta,
                    )
                case FTWStrategy.SWORD_VAR | "sword_var":
                    # SWORD-Var: variation-adaptive OMD with optimistic gradients
                    mirror_map = AdaptiveVariationMap(eps=self.adagrad_eps)
                case FTWStrategy.SWORD_SMALL | "sword_small":
                    # SWORD-Small: AdaGrad geometry with optimistic gradients
                    mirror_map = AdaptiveMahalanobisMap(eps=self.adagrad_eps)
                case (
                    FTWStrategy.SWORD_BEST
                    | FTWStrategy.SWORD_PP
                    | "sword_best"
                    | "sword_pp"
                ):
                    # Meta-aggregator combining SWORD-Var and SWORD-Small, plus EG for Sword++
                    var_engine = FirstOrderOCO(
                        mirror_map=AdaptiveVariationMap(eps=self.adagrad_eps),
                        projector=IdentityProjector(),  # inner unconstrained
                        eta=effective_learning_rate,
                        predictor=LastGradPredictor(),
                        mode="omd",
                    )
                    small_engine = FirstOrderOCO(
                        mirror_map=AdaptiveMahalanobisMap(eps=self.adagrad_eps),
                        projector=IdentityProjector(),
                        eta=effective_learning_rate,
                        predictor=LastGradPredictor(),
                        mode="omd",
                    )
                    experts: list[FirstOrderOCO] = [var_engine, small_engine]
                    if self.strategy in (FTWStrategy.SWORD_PP, "sword_pp"):
                        # add an entropy-geometry expert to stabilize/explore (only for Sword++)
                        eg_engine = FirstOrderOCO(
                            mirror_map=EntropyMirrorMap(),
                            projector=IdentityProjector(),
                            eta=effective_learning_rate,
                            predictor=LastGradPredictor(),
                            mode="omd",
                        )
                        experts.append(eg_engine)
                    # specific case treating SwordMeta differently
                    # we initialize it here because it needs to be initialized with the experts (only for SwordMeta)
                    self._foco_engine = SwordMeta(
                        experts=experts,
                        projector=self._projector,
                        eta_meta=effective_learning_rate,  # tie meta-eta to learning_rate
                    )
                case _:
                    raise ValueError(f"Unknown strategy provided {self.strategy}")

            # Finally initialize the FOCO engine to be used in the fit method
            if self._foco_engine is None:
                skip_auto_update = self.strategy in (
                    FTWStrategy.ADABARRONS,
                    "adabarrons",
                )
                assert mirror_map is not None
                self._foco_engine = FirstOrderOCO(
                    mirror_map=mirror_map,
                    projector=self._projector,
                    eta=effective_learning_rate,
                    predictor=predictor,
                    mode=self.update_mode,
                    skip_auto_update=skip_auto_update,
                    discount=self.discount,
                )

        # Initialize weights only once per fit/streaming session.
        # Resetting is handled at the beginning of fit()
        # via _reset_state_for_fit; we must not reinitialize at every partial_fit call.
        if not self._weights_initialized:
            self._initialize_weights(num_assets)
            self._apply_initial_projection()
            if isinstance(self._foco_engine, FirstOrderOCO):
                if self._foco_engine._x_t is None:
                    self._foco_engine._x_t = self.weights_.copy()

        # Mark overall init complete
        self._is_initialized = True

    def _compute_portfolio_gradient(
        self, effective_relatives: np.ndarray
    ) -> np.ndarray:
        """Compute the portfolio gradient with optional enhancements.

        Enhancements (applied in order):

        1. **Cross-sectional demeaning** (``demean_gradient=True``):
           Subtracts the mean gradient across assets, removing common-factor
           noise and amplifying idiosyncratic cross-sectional signal.

        2. **Momentum lookback** (``gradient_lookback > 1``):
           Averages the current gradient with the last ``gradient_lookback - 1``
           historical gradients, smoothing daily noise and capturing medium-term
           momentum signals (Jegadeesh and Titman, 1993).

        Parameters
        ----------
        effective_relatives : np.ndarray
            Effective price relatives after fees.

        Returns
        -------
        np.ndarray
            Gradient vector for the portfolio optimization.
        """
        # Convert effective relatives to net returns for objective function
        # Objective functions now work with net returns (r = x - 1)
        effective_net_returns = effective_relatives - 1.0

        # Compute gradient using objective function (log-wealth by default, or custom risk/perf measure)
        gradient = self._objective_fn.grad(self.weights_, effective_net_returns)

        # Enhancement 1: Cross-sectional demeaning
        if self.demean_gradient:
            gradient = gradient - np.mean(gradient)

        # Enhancement 2: Momentum lookback (gradient averaging)
        if self.gradient_lookback > 1:
            if not hasattr(self, "_grad_buffer"):
                self._grad_buffer: list[np.ndarray] = []
            self._grad_buffer.append(gradient.copy())
            if len(self._grad_buffer) > self.gradient_lookback:
                self._grad_buffer.pop(0)
            gradient = np.mean(self._grad_buffer, axis=0)

        if (
            getattr(self, "penalize_turnover", False)
            and self._execution_anchor is not None
        ):
            gradient += self._transaction_costs_arr * np.sign(
                self.weights_ - self._execution_anchor
            )

        return gradient

    def _execute_foco_step(self, gradient: np.ndarray) -> np.ndarray:
        """Execute the FTRL optimization step.

        Parameters
        ----------
        gradient : np.ndarray
            Portfolio gradient vector.

        Returns
        -------
        np.ndarray
            New portfolio weights from FTRL step.
        """
        if self._foco_engine:
            w_foco = self._foco_engine.step(gradient)
        else:
            raise RuntimeError("FTRL Engine not initialized")
        return w_foco

    def _apply_weights_mixing(self, w_foco: np.ndarray) -> np.ndarray:
        """Apply weights mixing (only for EG-tilde mixing).

        Parameters
        ----------
        w_foco : np.ndarray
            Weights from FOCO step.

        Returns
        -------
        np.ndarray
            Final weights.
        """
        if not (self.eg_tilde and self.strategy == FTWStrategy.EG):
            return w_foco

        assert self._foco_engine is not None
        assert self._projector is not None
        alpha_t = (
            self.eg_tilde_alpha(self._foco_engine._t)
            if callable(self.eg_tilde_alpha)
            else self.eg_tilde_alpha
        )

        alpha_t = float(np.asarray(alpha_t))
        if alpha_t <= 0:
            return w_foco

        n = w_foco.shape[0]
        mixed = (1.0 - alpha_t) * w_foco + alpha_t / n
        return self._projector.solve_update(EntropyMirrorMap(), np.log(mixed))

    def _update_adabarrons_components(
        self, weights: np.ndarray, gradient: np.ndarray
    ) -> None:
        """Special update logic for Ada-BARRONS compositional mirror map.

        Ada-BARRONS requires:
        - Barrier component updated with weights (weight-proximity adaptation)
        - Full quadratic component updated with gradients (second-order curvature)

        Parameters
        ----------
        weights : np.ndarray
            Current portfolio weights.
        gradient : np.ndarray
            Portfolio gradient vector.
        """
        if (
            self.strategy == FTWStrategy.ADABARRONS
            and isinstance(self._foco_engine, FirstOrderOCO)
            and isinstance(self._foco_engine.map, CompositeMirrorMap)
        ):
            # Access the components of the Ada-BARRONS mirror map:
            # [0] = AdaBarronsBarrierMap (weight-proximity adaptive)
            # [1] = EuclideanMap (static, no update needed)
            # [2] = FullQuadraticMap (gradient-based second-order)
            components = self._foco_engine.map.components_

            # Update barrier with weights (NOT gradients)
            assert isinstance(components[0], DynamicMirrorMap)
            components[0].update_state(weights)

            # Update full quadratic with gradients
            assert isinstance(components[2], DynamicMirrorMap)
            components[2].update_state(gradient)

    def _finalize_partial_fit_state(self, effective_relatives: np.ndarray) -> None:
        """Update internal state and compute loss after weight update.

        Parameters
        ----------
        effective_relatives : np.ndarray
            Effective price relatives used for loss computation.
        """
        # Compute loss for inspection (optional, can be simplified)
        final_return = np.dot(self._previous_target_, effective_relatives)
        self.loss_ = -np.log(np.maximum(final_return, CLIP_EPSILON))
        self.cumulative_loss_ += self.loss_
        self._t += 1

    def _update_one(self, gross_relatives: np.ndarray) -> None:
        """Run Carlo's winner update for one validated observation."""
        # Step 2: Initialize engine and weights if needed
        self._ensure_initialized(gross_relatives)

        # Step 3: Apply management fees to get effective relatives
        effective_relatives = self._compute_effective_relatives(gross_relatives)

        # Step 4: Compute portfolio gradient (including transaction costs)
        gradient = self._compute_portfolio_gradient(effective_relatives)

        # Step 5: Execute FTRL optimization step
        w_ftrl = self._execute_foco_step(gradient)

        # Store the algorithm target used before observing this return.
        self._previous_target_ = self.weights_.copy()

        # Step 6: Apply post-processing (e.g., EG-tilde mixing)
        self.weights_ = self._apply_weights_mixing(w_ftrl)
        # Mixing is part of the deployed decision; the next OMD step uses it.
        if self.eg_tilde and isinstance(self._foco_engine, FirstOrderOCO):
            self._foco_engine._x_t = self.weights_.copy()

        # Step 6.5: Manual component updates for Ada-BARRONS
        # (barrier with weights, full quadratic with gradients)
        self._update_adabarrons_components(self.weights_, gradient)

        # Step 7: Update internal state and compute loss
        self._finalize_partial_fit_state(effective_relatives)

    def _initialize_algorithm(self, n_assets: int) -> None:
        """Initialize the existing winner engine without learning an observation."""
        self._ensure_initialized(np.ones(n_assets))

    def _reset_state_for_fit(self) -> None:
        """Reset the shared engine and objective for a fresh fit."""
        super()._reset_state_for_fit()
        self._foco_engine = None
        if hasattr(self, "_objective_fn"):
            del self._objective_fn
        if hasattr(self, "_turnover_penalty_disabled_warned"):
            del self._turnover_penalty_disabled_warned
        self.cumulative_loss_ = 0.0
        if hasattr(self, "_grad_buffer"):
            del self._grad_buffer
