"""Online portfolio estimators."""

# Copyright (c) 2023-2026 skfolio contributors (integration and modifications)
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import ClassVar, Literal

import skfolio.typing as skt
from skfolio.optimization.online._loser import FollowTheLoser
from skfolio.optimization.online._mixins import FTLStrategy, FTWStrategy, OLMARPredictor
from skfolio.optimization.online._winner import FollowTheWinner
from skfolio.prior import BasePrior


class ExponentiatedGradient(FollowTheWinner):
    """Online log-wealth optimization with entropy geometry.

    Parameters
    ----------
    learning_rate : float, default=0.05
        Positive gradient step size.
    update_mode : str, default="ftrl"
        Either ftrl (follow-the-regularized-leader) or omd (online mirror descent).
    initial_weights : skt.MultiInput | None, default=None
        Initial learning allocation. None uses uniform weights; named inputs follow X columns.
    previous_weights : skt.MultiInput | None, default=None
        Actual holdings at rebalance time. Required for a turnover limit.
    min_weights : skt.MultiInput, default=0.0
        Finite nonnegative lower bounds, aligned by asset name for dictionaries.
    max_weights : skt.MultiInput, default=1.0
        Finite nonnegative upper bounds; values above one allow leveraged positions.
    budget : float | None, default=1.0
        Finite nonnegative sum of weights. None removes the equality budget.
    min_budget : float, optional
        Lower bound on the sum of weights. Requires ``budget=None``.
    max_budget : float, optional
        Upper bound on the sum of weights. Requires ``budget=None``.
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
    cardinality : int, optional
        Maximum number of invested assets; requires a mixed-integer solver.
    group_cardinalities : dict[str, int], optional
        Maximum invested assets per named group. Unknown groups raise.
    solver : str, default="CLARABEL"
        CVXPY solver. Use a compatible mixed-integer solver for discrete constraints.
    solver_params : dict, optional
        Additional solver options, excluding the solver name.
    add_constraints : callable, optional
        Function accepting the CVXPY weight variable and optionally this estimator.
        Return one convex CVXPY constraint or a list, rebuilt at each projection.
        Prior-dependent callbacks require causal warmup before execution.
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
    ``fit`` resets learned state; ``partial_fit`` consumes rows causally. After row t,
    ``weights_`` is the target for t+1. Portfolio handles execution, drift and wealth.
    Use :func:`~skfolio.model_selection.online_predict` with ``test_size=1``
    for evaluation.
    Missing returns, bankrupt assets, changing universes and automatic recovery
    are not supported. A failed numerical update requires a fresh fit.

    References
    ----------
    .. [1] Helmbold, D. P., Schapire, R. E., Singer, Y., and Warmuth, M. K.
           (1998). "On-Line Portfolio Selection Using Multiplicative Updates."
           Mathematical Finance, 8(4), 325-347.
           https://doi.org/10.1111/1467-9965.00058

    Examples
    --------
    >>> from skfolio.optimization.online import ExponentiatedGradient
    >>> model = ExponentiatedGradient()
    >>> _ = model.partial_fit([[0.01, -0.02], [0.02, 0.01]])
    >>> model.n_observations_
    2
    >>> model.weights_.shape
    (2,)
    >>> portfolio = model.predict([[0.01, 0.0]])
    >>> portfolio.returns.shape
    (1,)
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
        min_budget: float | None = None,
        max_budget: float | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        max_variance: float | None = None,
        covariance: skt.ArrayLike | None = None,
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
        cardinality: int | None = None,
        group_cardinalities: dict[str, int] | None = None,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
        add_constraints: skt.ConstraintFunction | None = None,
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
            min_budget=min_budget,
            max_budget=max_budget,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            max_variance=max_variance,
            covariance=covariance,
            min_return=min_return,
            max_standard_deviation=max_standard_deviation,
            max_semi_variance=max_semi_variance,
            max_semi_deviation=max_semi_deviation,
            max_mean_absolute_deviation=max_mean_absolute_deviation,
            max_first_lower_partial_moment=max_first_lower_partial_moment,
            max_worst_realization=max_worst_realization,
            max_cvar=max_cvar,
            max_evar=max_evar,
            max_max_drawdown=max_max_drawdown,
            max_average_drawdown=max_average_drawdown,
            max_cdar=max_cdar,
            max_edar=max_edar,
            max_ulcer_index=max_ulcer_index,
            max_gini_mean_difference=max_gini_mean_difference,
            min_acceptable_return=min_acceptable_return,
            risk_free_rate=risk_free_rate,
            cvar_beta=cvar_beta,
            evar_beta=evar_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
            cardinality=cardinality,
            group_cardinalities=group_cardinalities,
            solver=solver,
            solver_params=solver_params,
            add_constraints=add_constraints,
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
        Finite nonnegative lower bounds, aligned by asset name for dictionaries.
    max_weights : skt.MultiInput, default=1.0
        Finite nonnegative upper bounds; values above one allow leveraged positions.
    budget : float | None, default=1.0
        Finite nonnegative sum of weights. None removes the equality budget.
    min_budget : float, optional
        Lower bound on the sum of weights. Requires ``budget=None``.
    max_budget : float, optional
        Upper bound on the sum of weights. Requires ``budget=None``.
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
    cardinality : int, optional
        Maximum number of invested assets; requires a mixed-integer solver.
    group_cardinalities : dict[str, int], optional
        Maximum invested assets per named group. Unknown groups raise.
    solver : str, default="CLARABEL"
        CVXPY solver. Use a compatible mixed-integer solver for discrete constraints.
    solver_params : dict, optional
        Additional solver options, excluding the solver name.
    add_constraints : callable, optional
        Function accepting the CVXPY weight variable and optionally this estimator.
        Return one convex CVXPY constraint or a list, rebuilt at each projection.
        Prior-dependent callbacks require causal warmup before execution.
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
    ``fit`` resets learned state; ``partial_fit`` consumes rows causally. After row t,
    ``weights_`` is the target for t+1. Portfolio handles execution, drift and wealth.
    Use :func:`~skfolio.model_selection.online_predict` with ``test_size=1``
    for evaluation.
    Missing returns, bankrupt assets, changing universes and automatic recovery
    are not supported. A failed numerical update requires a fresh fit.

    References
    ----------
    .. [1] Duchi, J., Hazan, E., and Singer, Y. (2011). "Adaptive Subgradient
           Methods for Online Learning and Stochastic Optimization."
           Journal of Machine Learning Research, 12, 2121-2159.
           https://jmlr.org/papers/v12/duchi11a.html

    Examples
    --------
    >>> from skfolio.optimization.online import AdaGrad
    >>> model = AdaGrad()
    >>> _ = model.partial_fit([[0.01, -0.02], [0.02, 0.01]])
    >>> model.n_observations_
    2
    >>> model.weights_.shape
    (2,)
    >>> portfolio = model.predict([[0.01, 0.0]])
    >>> portfolio.returns.shape
    (1,)
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
        min_budget: float | None = None,
        max_budget: float | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        max_variance: float | None = None,
        covariance: skt.ArrayLike | None = None,
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
        cardinality: int | None = None,
        group_cardinalities: dict[str, int] | None = None,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
        add_constraints: skt.ConstraintFunction | None = None,
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
            min_budget=min_budget,
            max_budget=max_budget,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            max_variance=max_variance,
            covariance=covariance,
            min_return=min_return,
            max_standard_deviation=max_standard_deviation,
            max_semi_variance=max_semi_variance,
            max_semi_deviation=max_semi_deviation,
            max_mean_absolute_deviation=max_mean_absolute_deviation,
            max_first_lower_partial_moment=max_first_lower_partial_moment,
            max_worst_realization=max_worst_realization,
            max_cvar=max_cvar,
            max_evar=max_evar,
            max_max_drawdown=max_max_drawdown,
            max_average_drawdown=max_average_drawdown,
            max_cdar=max_cdar,
            max_edar=max_edar,
            max_ulcer_index=max_ulcer_index,
            max_gini_mean_difference=max_gini_mean_difference,
            min_acceptable_return=min_acceptable_return,
            risk_free_rate=risk_free_rate,
            cvar_beta=cvar_beta,
            evar_beta=evar_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
            cardinality=cardinality,
            group_cardinalities=group_cardinalities,
            solver=solver,
            solver_params=solver_params,
            add_constraints=add_constraints,
            prior_estimator=prior_estimator,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            store_weights=store_weights,
            portfolio_params=portfolio_params,
        )


class OnlineMovingAverageReversion(FollowTheLoser):
    """Online moving-average reversion with passive-aggressive updates.

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
        Finite nonnegative lower bounds, aligned by asset name for dictionaries.
    max_weights : skt.MultiInput, default=1.0
        Finite nonnegative upper bounds; values above one allow leveraged positions.
    budget : float | None, default=1.0
        Finite nonnegative sum of weights. None removes the equality budget.
    min_budget : float, optional
        Lower bound on the sum of weights. Requires ``budget=None``.
    max_budget : float, optional
        Upper bound on the sum of weights. Requires ``budget=None``.
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
    cardinality : int, optional
        Maximum number of invested assets; requires a mixed-integer solver.
    group_cardinalities : dict[str, int], optional
        Maximum invested assets per named group. Unknown groups raise.
    solver : str, default="CLARABEL"
        CVXPY solver. Use a compatible mixed-integer solver for discrete constraints.
    solver_params : dict, optional
        Additional solver options, excluding the solver name.
    add_constraints : callable, optional
        Function accepting the CVXPY weight variable and optionally this estimator.
        Return one convex CVXPY constraint or a list, rebuilt at each projection.
        Prior-dependent callbacks require causal warmup before execution.
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
    ``fit`` resets learned state; ``partial_fit`` consumes rows causally. After row t,
    ``weights_`` is the target for t+1. Portfolio handles execution, drift and wealth.
    Use :func:`~skfolio.model_selection.online_predict` with ``test_size=1``
    for evaluation.
    Missing returns, bankrupt assets, changing universes and automatic recovery
    are not supported. A failed numerical update requires a fresh fit.

    References
    ----------
    .. [1] Li, B., and Hoi, S. C. H. (2012). "On-Line Portfolio Selection with
           Moving Average Reversion." ICML.
           https://icml.cc/2012/papers/168.pdf

    Examples
    --------
    >>> from skfolio.optimization.online import OnlineMovingAverageReversion
    >>> model = OnlineMovingAverageReversion()
    >>> _ = model.partial_fit([[0.01, -0.02], [0.02, 0.01]])
    >>> model.n_observations_
    2
    >>> model.weights_.shape
    (2,)
    >>> portfolio = model.predict([[0.01, 0.0]])
    >>> portfolio.returns.shape
    (1,)
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
        min_budget: float | None = None,
        max_budget: float | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        max_variance: float | None = None,
        covariance: skt.ArrayLike | None = None,
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
        cardinality: int | None = None,
        group_cardinalities: dict[str, int] | None = None,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
        add_constraints: skt.ConstraintFunction | None = None,
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
            min_budget=min_budget,
            max_budget=max_budget,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            max_variance=max_variance,
            covariance=covariance,
            min_return=min_return,
            max_standard_deviation=max_standard_deviation,
            max_semi_variance=max_semi_variance,
            max_semi_deviation=max_semi_deviation,
            max_mean_absolute_deviation=max_mean_absolute_deviation,
            max_first_lower_partial_moment=max_first_lower_partial_moment,
            max_worst_realization=max_worst_realization,
            max_cvar=max_cvar,
            max_evar=max_evar,
            max_max_drawdown=max_max_drawdown,
            max_average_drawdown=max_average_drawdown,
            max_cdar=max_cdar,
            max_edar=max_edar,
            max_ulcer_index=max_ulcer_index,
            max_gini_mean_difference=max_gini_mean_difference,
            min_acceptable_return=min_acceptable_return,
            risk_free_rate=risk_free_rate,
            cvar_beta=cvar_beta,
            evar_beta=evar_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
            cardinality=cardinality,
            group_cardinalities=group_cardinalities,
            solver=solver,
            solver_params=solver_params,
            add_constraints=add_constraints,
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


class PassiveAggressiveMeanReversion(FollowTheLoser):
    """Passive-aggressive mean reversion using the observed price relatives.

    Parameters
    ----------
    variant : {"simple", "slack_linear", "slack_quadratic"}, default="simple"
        PAMR, PAMR-1 or PAMR-2 respectively. The slack variants soften the margin.
    aggressiveness : float, default=500.0
        Positive slack penalty C. Used only by the two slack variants.
    epsilon : float, default=1.0
        Upper margin on the observed portfolio price relative.
    initial_weights : skt.MultiInput | None, default=None
        Initial learning allocation. None uses uniform weights; named inputs follow X columns.
    previous_weights : skt.MultiInput | None, default=None
        Actual holdings at rebalance time. Required for a turnover limit.
    min_weights : skt.MultiInput, default=0.0
        Finite nonnegative lower bounds, aligned by asset name for dictionaries.
    max_weights : skt.MultiInput, default=1.0
        Finite nonnegative upper bounds; values above one allow leveraged positions.
    budget : float | None, default=1.0
        Finite nonnegative sum of weights. None removes the equality budget.
    min_budget : float, optional
        Lower bound on the sum of weights. Requires ``budget=None``.
    max_budget : float, optional
        Upper bound on the sum of weights. Requires ``budget=None``.
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
    cardinality : int, optional
        Maximum number of invested assets; requires a mixed-integer solver.
    group_cardinalities : dict[str, int], optional
        Maximum invested assets per named group. Unknown groups raise.
    solver : str, default="CLARABEL"
        CVXPY solver. Use a compatible mixed-integer solver for discrete constraints.
    solver_params : dict, optional
        Additional solver options, excluding the solver name.
    add_constraints : callable, optional
        Function accepting the CVXPY weight variable and optionally this estimator.
        Return one convex CVXPY constraint or a list, rebuilt at each projection.
        Prior-dependent callbacks require causal warmup before execution.
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
    ``fit`` resets learned state; ``partial_fit`` consumes rows causally. After row t,
    ``weights_`` is the target for t+1. Portfolio handles execution, drift and wealth.
    Use :func:`~skfolio.model_selection.online_predict` with ``test_size=1``
    for evaluation.
    Missing returns, bankrupt assets, changing universes and automatic recovery
    are not supported. A failed numerical update requires a fresh fit.


    The update reduces exposure to assets with high recent price relatives.
    ``simple`` uses the unbounded passive-aggressive step; ``slack_linear``
    caps the step at C, and ``slack_quadratic`` adds 1/(2C) to its denominator.
    Only the passive-aggressive update is exposed.

    References
    ----------
    .. [1] Li, B., Zhao, P., Hoi, S. C. H., and Gopalkrishnan, V. (2012).
           "PAMR: Passive Aggressive Mean Reversion Strategy for Portfolio
           Selection." Machine Learning, 87, 221-258.
    """

    _parameter_constraints: ClassVar[dict] = {
        **FollowTheLoser._parameter_constraints,
        "variant": FollowTheLoser._parameter_constraints["pamr_variant"],
        "aggressiveness": FollowTheLoser._parameter_constraints["pamr_C"],
    }

    def __init__(
        self,
        *,
        variant: str = "simple",
        aggressiveness: float = 500.0,
        epsilon: float = 1.0,
        initial_weights: skt.MultiInput | None = None,
        previous_weights: skt.MultiInput | None = None,
        min_weights: skt.MultiInput = 0.0,
        max_weights: skt.MultiInput = 1.0,
        budget: float | None = 1.0,
        min_budget: float | None = None,
        max_budget: float | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        max_variance: float | None = None,
        covariance: skt.ArrayLike | None = None,
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
        cardinality: int | None = None,
        group_cardinalities: dict[str, int] | None = None,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
        add_constraints: skt.ConstraintFunction | None = None,
        prior_estimator: BasePrior | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        store_weights: bool = False,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(
            strategy=FTLStrategy.PAMR,
            update_mode="pa",
            pamr_variant=variant,
            pamr_C=aggressiveness,
            epsilon=epsilon,
            initial_weights=initial_weights,
            previous_weights=previous_weights,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            min_budget=min_budget,
            max_budget=max_budget,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            max_turnover=max_turnover,
            max_variance=max_variance,
            covariance=covariance,
            min_return=min_return,
            max_standard_deviation=max_standard_deviation,
            max_semi_variance=max_semi_variance,
            max_semi_deviation=max_semi_deviation,
            max_mean_absolute_deviation=max_mean_absolute_deviation,
            max_first_lower_partial_moment=max_first_lower_partial_moment,
            max_worst_realization=max_worst_realization,
            max_cvar=max_cvar,
            max_evar=max_evar,
            max_max_drawdown=max_max_drawdown,
            max_average_drawdown=max_average_drawdown,
            max_cdar=max_cdar,
            max_edar=max_edar,
            max_ulcer_index=max_ulcer_index,
            max_gini_mean_difference=max_gini_mean_difference,
            min_acceptable_return=min_acceptable_return,
            risk_free_rate=risk_free_rate,
            cvar_beta=cvar_beta,
            evar_beta=evar_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
            cardinality=cardinality,
            group_cardinalities=group_cardinalities,
            solver=solver,
            solver_params=solver_params,
            add_constraints=add_constraints,
            prior_estimator=prior_estimator,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            store_weights=store_weights,
            portfolio_params=portfolio_params,
        )

    @property
    def variant(self) -> str:
        """Expose the shared kernel parameter for sklearn tuning."""
        return self.pamr_variant

    @variant.setter
    def variant(self, value: str) -> None:
        """Set the kernel parameter for subsequent fitting."""
        self.pamr_variant = value

    @property
    def aggressiveness(self) -> float:
        """Expose the shared kernel parameter for sklearn tuning."""
        return self.pamr_C

    @aggressiveness.setter
    def aggressiveness(self, value: float) -> None:
        """Set the kernel parameter for subsequent fitting."""
        self.pamr_C = value
