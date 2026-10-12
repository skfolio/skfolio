"""Benchmark estimators for online portfolio selection."""

# Copyright (c) 2025
# Author: Carlo Nicolini <nicolini.carlo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd

import skfolio.typing as skt
from skfolio.measures import PerfMeasure, RiskMeasure
from skfolio.optimization._base import BaseOptimization
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.optimization.convex._mean_risk import MeanRisk


def net_to_relatives(X: npt.ArrayLike) -> np.ndarray:
    """Validate comparator returns without clipping bankrupt assets."""
    values = np.asarray(X, dtype=float)
    if values.ndim != 2 or 0 in values.shape or not np.all(np.isfinite(values)):
        raise ValueError("X must be a nonempty finite 2D return matrix")
    if np.any(values <= -1):
        raise ValueError("Returns must be greater than -1")
    return 1.0 + values


class CRP(BaseOptimization):
    """Constant Rebalanced Portfolio (CRP).

    Keeps a fixed portfolio w and rebalances to it every period.

    Parameters
    ----------
    weights : array-like of shape (n_assets,), optional
        Fixed weights to rebalance to. If None, uses uniform weights leading to the
        Uniform Constant Rebalanced Portfolio (UCRP).
    portfolio_params : dict, optional
        Portfolio parameters propagated to the produced portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Portfolio weights.

    wealth_ : float
        Gross hindsight wealth after all periods, excluding costs and fees.

    all_wealth_ : ndarray of shape (T+1,)
        Wealth history: wealth_[0] is initial wealth (1.0), wealth_[t] is wealth after period t.

    all_weights_ : ndarray of shape (T, n_assets)
        Weight history: weights used at the start of each period (before observing returns).
    """

    n_features_in_: int

    def __init__(
        self,
        weights: npt.ArrayLike | None,
        portfolio_params: dict | None = None,
    ) -> None:
        super().__init__(portfolio_params=portfolio_params)
        self.weights = weights
        self.portfolio_params = portfolio_params

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> CRP:
        """Update the comparator target using the supplied observations."""
        net_to_relatives(X)
        X = np.asarray(X)
        if self.weights is None:
            n = X.shape[1]
            self.weights_ = np.ones(n) / n
        else:
            self.weights_ = np.asarray(self.weights, dtype=float)
        if self.weights_.shape != (X.shape[1],) or not np.all(
            np.isfinite(self.weights_)
        ):
            raise ValueError("weights must contain one finite value per asset")
        return self

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> CRP:
        """Fit the CRP/UCRP and compute wealth trajectory.

        Parameters
        ----------
        X : array-like of shape (T, n_assets)
            Net returns per period.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        self
            The estimator instance.
        """
        X_arr = np.asarray(X, dtype=float)

        # Set weights
        self.partial_fit(X_arr, y)

        # Compute wealth trajectory
        relatives = net_to_relatives(X_arr)  # (T, n_assets)
        T = relatives.shape[0]

        # Initialize wealth tracking
        self.wealth_ = 1.0
        wealth_history = [self.wealth_]
        weight_history = []

        # Compute wealth evolution (constant rebalanced portfolio)
        for t in range(T):
            weight_history.append(np.asarray(self.weights_, dtype=float).copy())
            portfolio_return = 1.0 + np.dot(self.weights_, relatives[t] - 1.0)
            self.wealth_ *= portfolio_return
            wealth_history.append(self.wealth_)

        self.all_wealth_ = np.array(wealth_history, dtype=float)
        self.all_weights_ = np.vstack(weight_history)

        return self


# utility class for UCRP
class UCRP(CRP):
    """Uniform constant-rebalanced portfolio comparator."""

    def __init__(
        self, weights: npt.ArrayLike | None = None, portfolio_params: dict | None = None
    ) -> None:
        super().__init__(weights, portfolio_params)


class BestStock(BaseOptimization):
    """Best Stock (ex-post).

    Invests all capital in the single asset with highest cumulative log-return.
    This is a hindsight benchmark showing the best possible single-asset performance.

    Parameters
    ----------
    portfolio_params : dict, optional
        Portfolio parameters propagated to the produced portfolio.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Portfolio weights (one-hot vector with 1.0 on best asset).

    wealth_ : float
        Gross hindsight wealth after all periods, excluding costs and fees.

    all_wealth_ : ndarray of shape (T+1,)
        Wealth history: wealth_[0] is initial wealth (1.0), wealth_[t] is wealth after period t.

    all_weights_ : ndarray of shape (T, n_assets)
        Weight history: weights used at the start of each period (before observing returns).
    """

    def __init__(self, portfolio_params: dict | None = None) -> None:
        super().__init__(portfolio_params=portfolio_params)

    def partial_fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> BestStock:
        """Update the comparator target using the supplied observations."""
        relatives = net_to_relatives(X)
        best_stock = np.argmax(np.sum(np.log(relatives), axis=0, keepdims=True), axis=1)
        self.weights_ = np.zeros(relatives.shape[1])
        self.weights_[best_stock] = 1.0
        return self

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> BestStock:
        """Fit BestStock and compute wealth trajectory.

        Parameters
        ----------
        X : array-like of shape (T, n_assets)
            Net returns per period.
        y : Ignored
            Present for API consistency.

        Returns
        -------
        self
            The estimator instance.
        """
        X_arr = np.asarray(X, dtype=float)

        # Set weights (identifies best stock in hindsight)
        self.partial_fit(X_arr, y)

        # Compute wealth trajectory
        relatives = net_to_relatives(X_arr)  # (T, n_assets)
        T = relatives.shape[0]

        # Initialize wealth tracking
        self.wealth_ = 1.0
        wealth_history = [self.wealth_]
        weight_history = []

        # Compute wealth evolution (all capital in best stock)
        for t in range(T):
            weight_history.append(np.asarray(self.weights_, dtype=float).copy())
            portfolio_return = 1.0 + np.dot(self.weights_, relatives[t] - 1.0)
            self.wealth_ *= portfolio_return
            wealth_history.append(self.wealth_)

        self.all_wealth_ = np.array(wealth_history, dtype=float)
        self.all_weights_ = np.vstack(weight_history)

        return self


class BCRP(MeanRisk):
    """Best Constant Rebalanced Portfolio (BCRP) in hindsight.

    Optimizes portfolio weights on historical data using any convex objective:
    - Risk minimization (variance, CVaR, CDaR, etc.)
    - Log-wealth maximization (Kelly criterion)
    - Mean-risk utility maximization

    Parameters
    ----------
    objective_measure : RiskMeasure | PerfMeasure, default=PerfMeasure.LOG_WEALTH
        The measure to optimize:

        - If ``PerfMeasure.LOG_WEALTH``: maximizes cumulative log-wealth
          ``sum_t log((1 + r_t)^T w)``
        - If ``RiskMeasure``: uses specified ``objective_function``
          (minimize risk, maximize utility, etc.)

    objective_function : ObjectiveFunction, default=ObjectiveFunction.MAXIMIZE_RETURN
        Objective function when using a RiskMeasure:

        - ``MINIMIZE_RISK``: minimize the risk measure
        - ``MAXIMIZE_UTILITY``: maximize mean - risk_aversion * risk
        - ``MAXIMIZE_RATIO``: maximize Sharpe-like ratio

        Ignored when ``objective_measure`` is ``LOG_WEALTH``.

    risk_aversion : float, default=1.0
        Risk aversion parameter for ``MAXIMIZE_UTILITY`` objective.
        Higher values lead to more conservative portfolios.

    l2_coef : float, default=0.0
        L2 regularization coefficient for weight smoothness.

    transaction_costs : float | dict | array-like, default=0.0
        Transaction costs per asset (proportional).

    management_fees : float | dict | array-like, default=0.0
        Management fees per asset (per-period).

    previous_weights : float | dict | array-like | None, default=None
        Previous weights for computing transaction costs and turnover.

    groups : dict | array-like | None, default=None
        Asset groups for linear constraints.

    linear_constraints : list[str] | None, default=None
        Linear constraints on weights (e.g., ``"Equity >= 0.5"``).

    left_inequality : array-like | None, default=None
        Left-hand side matrix A for inequality Aw <= b.

    right_inequality : array-like | None, default=None
        Right-hand side vector b for inequality Aw <= b.

    max_turnover : float | None, default=None
        Maximum allowed turnover per period (L1 norm of weight changes).

    min_weights : float | dict | array-like | None, default=0.0
        Minimum weight per asset (lower bound).

    max_weights : float | dict | array-like | None, default=1.0
        Maximum weight per asset (upper bound).

    budget : float | None, default=1.0
        Investment budget (sum of weights). If None, no budget constraint.

    solver : str, default="CLARABEL"
        CVXPY solver to use.

    solver_params : dict | None, default=None
        Additional parameters passed to the solver.

    scale_objective : float | None, default=None
        Scale factor for the objective function.

    scale_constraints : float | None, default=None
        Scale factor for constraints.

    save_problem : bool, default=False
        If True, save the CVXPY problem in ``problem_`` attribute.

    raise_on_failure : bool, default=True
        If True, raise an error when optimization fails.

    add_objective : callable | None, default=None
        Custom objective term to add to the optimization.

    add_constraints : callable | None, default=None
        Custom constraints to add to the optimization.

    portfolio_params : dict | None, default=None
        Additional portfolio parameters.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Optimized portfolio weights.

    wealth_ : float
        Gross hindsight wealth after all periods, excluding costs and fees.

    all_wealth_ : ndarray of shape (T+1,)
        Wealth history: wealth_[0] is initial wealth (1.0), wealth_[t] is wealth after period t.

    all_weights_ : ndarray of shape (T, n_assets)
        Weight history from ``fit`` method. For ``fit_dynamic``, contains weights
        from each expanding window (dynamic regret).

    Examples
    --------
    >>> from skfolio.optimization.online._benchmark import BCRP
    >>> from skfolio.measures import RiskMeasure, PerfMeasure
    >>> from skfolio.optimization.convex._base import ObjectiveFunction
    >>> import numpy as np
    >>>
    >>> # Log-wealth maximization (default)
    >>> bcrp_log = BCRP()
    >>>
    >>> # Variance minimization
    >>> bcrp_var = BCRP(
    ...     objective_measure=RiskMeasure.VARIANCE,
    ...     objective_function=ObjectiveFunction.MINIMIZE_RISK
    ... )
    >>>
    >>> # CVaR minimization
    >>> bcrp_cvar = BCRP(
    ...     objective_measure=RiskMeasure.CVAR,
    ...     objective_function=ObjectiveFunction.MINIMIZE_RISK,
    ...     cvar_beta=0.95
    ... )
    >>>
    >>> # Mean-variance utility
    >>> bcrp_utility = BCRP(
    ...     objective_measure=RiskMeasure.VARIANCE,
    ...     objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
    ...     risk_aversion=2.0
    ... )

    References
    ----------
    .. [1] Li, B., & Hoi, S. C. H. (2013). Online Portfolio Selection: A Survey.
           arXiv:1212.2129.
    """

    name: str = "BCRP"
    n_features_in_: int

    @staticmethod
    def _log_wealth_expr(w: cp.Variable, estimator: MeanRisk) -> cp.Expression:
        """Log-wealth objective: sum_t log(r_t^T w)."""
        rd = estimator.prior_estimator_.return_distribution_
        relatives = 1.0 + rd.returns
        return cp.sum(cp.log(1.0 + (relatives - 1.0) @ w))

    def __init__(
        self,
        objective_measure: RiskMeasure | PerfMeasure = PerfMeasure.LOG_WEALTH,
        *,
        objective_function: ObjectiveFunction = ObjectiveFunction.MAXIMIZE_RETURN,
        risk_aversion: float = 1.0,
        initial_wealth: float | npt.ArrayLike | None = None,
        l2_coef: float = 0.0,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        max_turnover: float | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
        scale_objective: float | None = None,
        scale_constraints: float | None = None,
        save_problem: bool = False,
        raise_on_failure: bool = True,
        add_objective: skt.ExpressionFunction | None = None,
        add_constraints: skt.ConstraintFunction | None = None,
        portfolio_params: dict | None = None,
        # Additional MeanRisk parameters for risk measures
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
        min_acceptable_return: skt.Target | None = None,
        risk_free_rate: float = 0.0,
    ) -> None:
        # Determine configuration based on objective_measure
        if objective_measure == PerfMeasure.LOG_WEALTH:
            # Log-wealth maximization: override expected return with log-sum
            risk_measure = RiskMeasure.VARIANCE
            obj_func = ObjectiveFunction.MAXIMIZE_RETURN
            overwrite_expected_return = BCRP._log_wealth_expr
        elif objective_measure == PerfMeasure.MEAN:
            # Mean maximization uses the standard MeanRisk expected-return objective.
            # We keep a valid risk measure only to satisfy MeanRisk validation; the
            # MAXIMIZE_RETURN objective ignores it in the final program.
            risk_measure = RiskMeasure.VARIANCE
            obj_func = ObjectiveFunction.MAXIMIZE_RETURN
            overwrite_expected_return = None
        else:
            # Standard risk measure optimization
            if not isinstance(objective_measure, RiskMeasure):
                raise ValueError(
                    "objective_measure must be RiskMeasure, PerfMeasure.LOG_WEALTH, "
                    f"or PerfMeasure.MEAN, "
                    f"got {type(objective_measure).__name__}"
                )
            risk_measure = objective_measure
            obj_func = objective_function
            overwrite_expected_return = None

        super().__init__(
            objective_function=obj_func,
            risk_measure=risk_measure,
            risk_aversion=risk_aversion,
            min_weights=min_weights,
            max_weights=max_weights,
            max_turnover=max_turnover,
            budget=budget,
            l2_coef=l2_coef,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            previous_weights=previous_weights,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            solver=solver,
            solver_params=solver_params,
            scale_objective=scale_objective,
            scale_constraints=scale_constraints,
            save_problem=save_problem,
            raise_on_failure=raise_on_failure,
            add_objective=add_objective,
            add_constraints=add_constraints,
            overwrite_expected_return=overwrite_expected_return,
            portfolio_params=portfolio_params,
            cvar_beta=cvar_beta,
            evar_beta=evar_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
            min_acceptable_return=min_acceptable_return,
            risk_free_rate=risk_free_rate,
        )
        self.initial_wealth = initial_wealth
        self.max_turnover = max_turnover
        self.objective_measure = objective_measure

    def partial_fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> BCRP:
        """Update the comparator target using the supplied observations."""
        return self.fit(X, y)

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params: Any
    ) -> BCRP:
        """Fit BCRP and compute wealth trajectory.

        Parameters
        ----------
        X : array-like of shape (T, n_assets)
            Net returns per period.
        y : Ignored
            Present for API consistency.
        **fit_params
            Additional fit parameters.

        Returns
        -------
        self
            The estimator instance.
        """
        net_to_relatives(X)
        # Call parent fit (MeanRisk)
        super().fit(X, y, **fit_params)

        # Add wealth tracking
        X_arr = np.asarray(X, dtype=float)
        relatives = net_to_relatives(X_arr)  # (T, n_assets)
        T = relatives.shape[0]

        # Initialize wealth tracking
        self.wealth_ = self.initial_wealth or 1.0
        wealth_history = [self.wealth_]
        weight_history = []

        # Compute wealth evolution with constant optimal weights
        for t in range(T):
            weight_history.append(np.asarray(self.weights_, dtype=float).copy())
            portfolio_return = 1.0 + np.dot(self.weights_, relatives[t] - 1.0)
            self.wealth_ *= portfolio_return
            wealth_history.append(self.wealth_)

        self.all_wealth_ = np.array(wealth_history, dtype=float)
        self.all_weights_ = np.vstack(weight_history)

        return self

    def _fit_first_prefix(self, X: npt.ArrayLike) -> None:
        """Solve a one-period return comparator without estimating covariance."""
        if self.objective_measure not in (PerfMeasure.LOG_WEALTH, PerfMeasure.MEAN):
            raise ValueError(
                "fit_dynamic requires a return objective; risk cannot be estimated from one row"
            )
        if self.add_objective is not None or self.add_constraints is not None:
            raise ValueError(
                "fit_dynamic does not support custom callbacks on its one-row prefix"
            )
        values = np.asarray(X, dtype=float)
        n = values.shape[1]
        self.n_features_in_ = n
        if hasattr(self, "feature_names_in_"):
            del self.feature_names_in_
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns)
        if hasattr(self, "investable_mask_"):
            del self.investable_mask_
        self._scale_constraints = cp.Constant(1.0)
        w = cp.Variable(n)
        constraints = self._get_weight_constraints(
            n_assets=n, w=w, factor=cp.Constant(1.0)
        )
        previous = self._clean_input(
            value=self.previous_weights,
            n_assets=n,
            fill_value=0,
            name="previous_weights",
        )
        if self.max_turnover is not None:
            constraints.append(cp.norm1(w - previous) <= self.max_turnover)
        costs = self._clean_input(
            value=self.transaction_costs,
            n_assets=n,
            fill_value=0,
            name="transaction_costs",
        )
        fees = self._clean_input(
            value=self.management_fees, n_assets=n, fill_value=0, name="management_fees"
        )
        gain = values[0] @ w
        reward = (
            cp.log(1.0 + gain)
            if self.objective_measure == PerfMeasure.LOG_WEALTH
            else gain
        )
        objective = (
            reward
            - cp.sum(cp.multiply(costs, cp.abs(w - previous)))
            - cp.sum(cp.multiply(fees, w))
            - self.l2_coef * cp.sum_squares(w)
        )
        problem = cp.Problem(cp.Maximize(objective), constraints)
        problem.solve(solver=self.solver, **(self.solver_params or {}))
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or w.value is None:
            raise RuntimeError("First-prefix comparator optimization failed")
        self.weights_ = np.asarray(w.value, dtype=float)

    def fit_dynamic(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params: Any
    ) -> MeanRisk:
        """Fit over increasingly large folds and compute wealth trajectory.

        Helper method to fit over increasingly large folds of data from t=1 to T.
        Useful for legacy prefix regret. Requires a log-wealth or mean-return
        objective; risk objectives and custom callbacks are unsupported because
        the first prefix contains only one observation.

        Parameters
        ----------
        X : array-like of shape (T, n_assets)
            Net returns per period.
        y : Ignored
            Present for API consistency.
        **fit_params
            Additional fit parameters.

        Returns
        -------
        self
            The estimator instance with all_weights_ and all_wealth_ attributes.
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be 2D array of shape (T, n_assets)")
        T = X_arr.shape[0]

        relatives = net_to_relatives(X_arr)

        weights_list: list[np.ndarray] = []
        wealth_history = [1.0]  # Initial wealth
        current_wealth = 1.0

        for t in range(1, T + 1):
            prefix = X.iloc[:t] if isinstance(X, pd.DataFrame) else X_arr[:t]
            if t == 1:
                self._fit_first_prefix(prefix)
            else:
                super().fit(prefix, y, **fit_params)
            weights_list.append(np.asarray(self.weights_, dtype=float).copy())

            # Update wealth for this period
            portfolio_return = 1.0 + np.dot(weights_list[-1], relatives[t - 1] - 1.0)
            current_wealth *= portfolio_return
            wealth_history.append(current_wealth)

        self.all_weights_ = np.vstack(weights_list)
        self.all_wealth_ = np.array(wealth_history, dtype=float)
        self.wealth_ = current_wealth

        return self
