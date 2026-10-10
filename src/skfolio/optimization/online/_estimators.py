"""Concrete online estimators over Carlo Nicolini's shared algorithm families."""

from __future__ import annotations

from typing import ClassVar, Literal

import skfolio.typing as skt
from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._mixins import FTLStrategy, FTWStrategy, OLMARPredictor
from skfolio.optimization.online._winner import FollowTheWinner
from skfolio.prior import BasePrior


class ExponentiatedGradient(FollowTheWinner):
    """Entropy-based log-wealth learning through Carlo's shared OCO engine.

    Parameters
    ----------
    learning_rate : float, default=0.05
        Positive gradient step size.
    update_mode : str, default="ftrl"
        Either ftrl (Carlo's default) or omd (local mirror descent).
    initial_weights : skt.MultiInput | None, default=None
        Initial learning allocation. None uses uniform weights; named inputs follow X columns.
    previous_weights : skt.MultiInput | None, default=None
        Actual holdings at rebalance time. Required for a turnover limit.
    min_weights : skt.MultiInput, default=0.0
        Lower bounds, aligned by asset name when supplied as a dictionary.
    max_weights : skt.MultiInput, default=1.0
        Upper bounds, aligned by asset name when supplied as a dictionary.
    budget : float | None, default=1.0
        Sum of weights. None removes the equality budget.
    groups : skt.Groups | None, default=None
        Asset groups for named linear constraints.
    linear_constraints : skt.LinearConstraints | None, default=None
        Named asset or group constraints.
    left_inequality : skt.Inequality | None, default=None
        Left matrix A in A @ weights <= b.
    right_inequality : skt.Inequality | None, default=None
        Right vector b in A @ weights <= b.
    max_turnover : float | None, default=None
        Maximum L1 change from actual previous_weights. Warmup uses this only at entry.
    max_variance : float | None, default=None
        Maximum variance, using covariance or the optional causal prior.
    covariance : skt.ArrayLike | None, default=None
        Fixed covariance matrix. Mutually exclusive with prior_estimator.
    prior_estimator : BasePrior | None, default=None
        Optional incremental prior, cloned and updated once per observed row.
    transaction_costs : skt.MultiInput, default=0.0
        Proportional trading costs booked by Portfolio during evaluation.
    management_fees : skt.MultiInput, default=0.0
        Per-period asset fees, accounted for by Portfolio and the learning objective.
    store_weights : bool, default=False
        Whether to retain diagnostic target history. False avoids accumulating target history.
    portfolio_params : dict | None, default=None
        Portfolio evaluation settings, for example weight_drift=True.

    Notes
    -----
    fit resets learned state; partial_fit consumes rows causally. After row t,
    weights_ is the target for t+1. Portfolio handles execution, drift and wealth.
    Use model_selection.online_predict with test_size=1 for evaluation.
    Missing returns, bankrupt assets, changing universes and automatic recovery
    are not supported. A failed numerical update requires a fresh fit.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 0.05,
        update_mode: Literal["ftrl", "omd"] = "ftrl",
        initial_weights: skt.MultiInput | None = None,
        previous_weights: skt.MultiInput | None = None,
        min_weights: skt.MultiInput = 0.0,
        max_weights: skt.MultiInput = 1.0,
        budget: float | None = 1.0,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        max_variance: float | None = None,
        covariance: skt.ArrayLike | None = None,
        prior_estimator: BasePrior | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        store_weights: bool = False,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(
            strategy=FTWStrategy.EG,
            learning_rate=learning_rate,
            update_mode=update_mode,
            initial_weights=initial_weights,
            previous_weights=previous_weights,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            max_variance=max_variance,
            covariance=covariance,
            prior_estimator=prior_estimator,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            store_weights=store_weights,
            portfolio_params=portfolio_params,
        )


class AdaGrad(FollowTheWinner):
    """Adaptive diagonal log-wealth learning through the same OCO engine.

    Parameters
    ----------
    learning_rate : float, default=1.0
        Positive gradient step size.
    update_mode : str, default="omd"
        Either omd or ftrl. OMD uses pending adaptive geometry.
    adagrad_eps : float, default=1e-8
        Positive initial diagonal regularization.
    initial_weights : skt.MultiInput | None, default=None
        Initial learning allocation. None uses uniform weights; named inputs follow X columns.
    previous_weights : skt.MultiInput | None, default=None
        Actual holdings at rebalance time. Required for a turnover limit.
    min_weights : skt.MultiInput, default=0.0
        Lower bounds, aligned by asset name when supplied as a dictionary.
    max_weights : skt.MultiInput, default=1.0
        Upper bounds, aligned by asset name when supplied as a dictionary.
    budget : float | None, default=1.0
        Sum of weights. None removes the equality budget.
    groups : skt.Groups | None, default=None
        Asset groups for named linear constraints.
    linear_constraints : skt.LinearConstraints | None, default=None
        Named asset or group constraints.
    left_inequality : skt.Inequality | None, default=None
        Left matrix A in A @ weights <= b.
    right_inequality : skt.Inequality | None, default=None
        Right vector b in A @ weights <= b.
    max_turnover : float | None, default=None
        Maximum L1 change from actual previous_weights. Warmup uses this only at entry.
    max_variance : float | None, default=None
        Maximum variance, using covariance or the optional causal prior.
    covariance : skt.ArrayLike | None, default=None
        Fixed covariance matrix. Mutually exclusive with prior_estimator.
    prior_estimator : BasePrior | None, default=None
        Optional incremental prior, cloned and updated once per observed row.
    transaction_costs : skt.MultiInput, default=0.0
        Proportional trading costs booked by Portfolio during evaluation.
    management_fees : skt.MultiInput, default=0.0
        Per-period asset fees, accounted for by Portfolio and the learning objective.
    store_weights : bool, default=False
        Whether to retain diagnostic target history. False avoids accumulating target history.
    portfolio_params : dict | None, default=None
        Portfolio evaluation settings, for example weight_drift=True.

    Notes
    -----
    fit resets learned state; partial_fit consumes rows causally. After row t,
    weights_ is the target for t+1. Portfolio handles execution, drift and wealth.
    Use model_selection.online_predict with test_size=1 for evaluation.
    Missing returns, bankrupt assets, changing universes and automatic recovery
    are not supported. A failed numerical update requires a fresh fit.
    """

    def __init__(
        self,
        *,
        learning_rate: float = 1.0,
        update_mode: Literal["ftrl", "omd"] = "omd",
        adagrad_eps: float = 1e-8,
        initial_weights: skt.MultiInput | None = None,
        previous_weights: skt.MultiInput | None = None,
        min_weights: skt.MultiInput = 0.0,
        max_weights: skt.MultiInput = 1.0,
        budget: float | None = 1.0,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        max_variance: float | None = None,
        covariance: skt.ArrayLike | None = None,
        prior_estimator: BasePrior | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        store_weights: bool = False,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(
            strategy=FTWStrategy.ADAGRAD,
            learning_rate=learning_rate,
            update_mode=update_mode,
            adagrad_eps=adagrad_eps,
            initial_weights=initial_weights,
            previous_weights=previous_weights,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            max_variance=max_variance,
            covariance=covariance,
            prior_estimator=prior_estimator,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            store_weights=store_weights,
            portfolio_params=portfolio_params,
        )


class OnlineMovingAverageReversion(FollowTheLoser):
    """Moving-average reversion through Carlo's shared passive-aggressive strategy.

    Parameters
    ----------
    window : int, default=5
        Number of prices in the moving average, including the current price.
    epsilon : float, default=2.0
        Positive reversion margin threshold.
    initial_weights : skt.MultiInput | None, default=None
        Initial learning allocation. None uses uniform weights; named inputs follow X columns.
    previous_weights : skt.MultiInput | None, default=None
        Actual holdings at rebalance time. Required for a turnover limit.
    min_weights : skt.MultiInput, default=0.0
        Lower bounds, aligned by asset name when supplied as a dictionary.
    max_weights : skt.MultiInput, default=1.0
        Upper bounds, aligned by asset name when supplied as a dictionary.
    budget : float | None, default=1.0
        Sum of weights. None removes the equality budget.
    groups : skt.Groups | None, default=None
        Asset groups for named linear constraints.
    linear_constraints : skt.LinearConstraints | None, default=None
        Named asset or group constraints.
    left_inequality : skt.Inequality | None, default=None
        Left matrix A in A @ weights <= b.
    right_inequality : skt.Inequality | None, default=None
        Right vector b in A @ weights <= b.
    max_turnover : float | None, default=None
        Maximum L1 change from actual previous_weights. Warmup uses this only at entry.
    max_variance : float | None, default=None
        Maximum variance, using covariance or the optional causal prior.
    covariance : skt.ArrayLike | None, default=None
        Fixed covariance matrix. Mutually exclusive with prior_estimator.
    prior_estimator : BasePrior | None, default=None
        Optional incremental prior, cloned and updated once per observed row.
    transaction_costs : skt.MultiInput, default=0.0
        Proportional trading costs booked by Portfolio during evaluation.
    management_fees : skt.MultiInput, default=0.0
        Per-period asset fees, accounted for by Portfolio and the learning objective.
    store_weights : bool, default=False
        Whether to retain diagnostic target history. False avoids accumulating target history.
    portfolio_params : dict | None, default=None
        Portfolio evaluation settings, for example weight_drift=True.

    Notes
    -----
    fit resets learned state; partial_fit consumes rows causally. After row t,
    weights_ is the target for t+1. Portfolio handles execution, drift and wealth.
    Use model_selection.online_predict with test_size=1 for evaluation.
    Missing returns, bankrupt assets, changing universes and automatic recovery
    are not supported. A failed numerical update requires a fresh fit.
    """

    _parameter_constraints: ClassVar[dict] = {
        **FollowTheLoser._parameter_constraints,
        "window": FollowTheLoser._parameter_constraints["olmar_window"],
    }

    def __init__(
        self,
        *,
        window: int = 5,
        epsilon: float = 2.0,
        initial_weights: skt.MultiInput | None = None,
        previous_weights: skt.MultiInput | None = None,
        min_weights: skt.MultiInput = 0.0,
        max_weights: skt.MultiInput = 1.0,
        budget: float | None = 1.0,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        max_variance: float | None = None,
        covariance: skt.ArrayLike | None = None,
        prior_estimator: BasePrior | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        store_weights: bool = False,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(
            strategy=FTLStrategy.OLMAR,
            olmar_predictor=OLMARPredictor.SMA,
            olmar_window=window,
            epsilon=epsilon,
            initial_weights=initial_weights,
            previous_weights=previous_weights,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            max_variance=max_variance,
            covariance=covariance,
            prior_estimator=prior_estimator,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            store_weights=store_weights,
            portfolio_params=portfolio_params,
        )

    @property
    def window(self) -> int:
        """Expose the source predictor window for sklearn parameter tuning."""
        return self.olmar_window

    @window.setter
    def window(self, value: int) -> None:
        """Update the shared predictor parameter for subsequent fitting."""
        self.olmar_window = value
