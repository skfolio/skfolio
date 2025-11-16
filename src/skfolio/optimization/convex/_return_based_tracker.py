"""Return-Based Tracker estimator."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio.measures import RiskMeasure
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.optimization.convex._mean_risk import MeanRisk
from skfolio.prior import BasePrior


class ReturnBasedTracker(MeanRisk):
    r"""Return-Based Tracker Optimization estimator.

    Minimizes the tracking error or tracking risk between the portfolio returns
    and a benchmark's returns by optimizing on excess returns (portfolio returns
    minus benchmark returns).

    This is a special case of the :class:`~skfolio.optimization.MeanRisk` estimator
    where the input returns are transformed to excess returns (X - y) before
    optimization.

    The tracking risk can be measured using different risk measures such as
    standard deviation (for traditional tracking error), variance (MSE),
    mean absolute deviation, etc.

    Parameters
    ----------
    risk_measure : RiskMeasure, default=RiskMeasure.STANDARD_DEVIATION
        :class:`~skfolio.measures.RiskMeasure` to minimize on excess returns.
        The default is `RiskMeasure.STANDARD_DEVIATION`.

    prior_estimator : BasePrior, optional
        :ref:`Prior estimator <prior>`.
        The prior estimator is used to estimate the :class:`~skfolio.prior.ReturnDistribution`
        of excess returns (portfolio returns - benchmark returns).
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    min_weights : float | dict[str, float] | array-like of shape (n_assets, ) | None, default=0.0
        Minimum assets weights (weights lower bounds).
        See :class:`~skfolio.optimization.MeanRisk` for details.

    max_weights : float | dict[str, float] | array-like of shape (n_assets, ) | None, default=1.0
        Maximum assets weights (weights upper bounds).
        See :class:`~skfolio.optimization.MeanRisk` for details.

    budget : float | None, default=1.0
        Investment budget.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    min_budget : float, optional
        Minimum budget.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    max_budget :  float, optional
        Maximum budget.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    max_short : float, optional
        Maximum short position.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    max_long : float, optional
        Maximum long position.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    transaction_costs : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Transaction costs of the assets.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Management fees of the assets.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    previous_weights : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Previous weights of the assets.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    l1_coef : float, default=0.0
        L1 regularization coefficient.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    l2_coef : float, default=0.0
        L2 regularization coefficient.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    groups : dict[str, list[str]] or array-like of shape (n_groups, n_assets), optional
        The assets groups.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    linear_constraints : array-like of shape (n_constraints,), optional
        Linear constraints.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    left_inequality : array-like of shape (n_constraints, n_assets), optional
        Left inequality matrix.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    right_inequality : array-like of shape (n_constraints, ), optional
        Right inequality vector.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    risk_free_rate : float, default=0.0
        Risk-free interest rate.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    solver : str, default="CLARABEL"
        The solver to use.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    solver_params : dict, optional
        Solver parameters.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    scale_objective : float, optional
        Scale each objective element by this value.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    scale_constraints : float, optional
        Scale each constraint element by this value.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    save_problem : bool, default=False
        If this is set to True, the CVXPY Problem is saved in `problem_`.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    add_objective : Callable[[cp.Variable], cp.Expression], optional
        Add a custom objective to the existing objective expression.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    add_constraints : Callable[[cp.Variable], cp.Expression|list[cp.Expression]], optional
        Add a custom constraint or a list of constraints.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    portfolio_params : dict, optional
        Portfolio parameters.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    fallback : BaseOptimization | "previous_weights" | list[BaseOptimization | "previous_weights"], optional
        Fallback estimator or list of estimators.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    raise_on_failure : bool, default=True
        Controls error handling when fitting fails.
        See :class:`~skfolio.optimization.MeanRisk` for details.

    Attributes
    ----------
    weights_ : ndarray of shape (n_assets,)
        Weights of the assets.

    problem_values_ :  dict[str, float]
        Expression values retrieved from the CVXPY problem.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator` on excess returns.

    problem_: cvxpy.Problem
        CVXPY problem used for the optimization.
        Only when `save_problem` is set to `True`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`.

    fallback_ : BaseOptimization | "previous_weights" | None
        The fallback estimator instance that produced the final result.

    fallback_chain_ : list[tuple[str, str]] | None
        Sequence describing the optimization fallback attempts.

    error_ : str | None
        Captured error message when `fit` fails.

    Notes
    -----
    The `y` parameter in the `fit` method must contain the benchmark returns
    with the same shape as `X` (n_observations,) or (n_observations, 1).
    """

    def __init__(
        self,
        risk_measure: RiskMeasure = RiskMeasure.STANDARD_DEVIATION,
        prior_estimator: BasePrior | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        min_budget: float | None = None,
        max_budget: float | None = None,
        max_short: float | None = None,
        max_long: float | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        l1_coef: float = 0.0,
        l2_coef: float = 0.0,
        risk_free_rate: float = 0.0,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
        scale_objective: float | None = None,
        scale_constraints: float | None = None,
        save_problem: bool = False,
        add_objective: skt.ExpressionFunction | None = None,
        add_constraints: skt.ExpressionFunction | None = None,
        portfolio_params: dict | None = None,
        fallback: skt.Fallback = None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=risk_measure,
            prior_estimator=prior_estimator,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            min_budget=min_budget,
            max_budget=max_budget,
            max_short=max_short,
            max_long=max_long,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            previous_weights=previous_weights,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            l1_coef=l1_coef,
            l2_coef=l2_coef,
            risk_free_rate=risk_free_rate,
            solver=solver,
            solver_params=solver_params,
            scale_objective=scale_objective,
            scale_constraints=scale_constraints,
            save_problem=save_problem,
            add_objective=add_objective,
            add_constraints=add_constraints,
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
        )

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike, **fit_params
    ) -> ReturnBasedTracker:
        """Fit the Return-Based Tracker estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : array-like of shape (n_observations,) or (n_observations, 1)
            Price returns of the benchmark.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : ReturnBasedTracker
           Fitted estimator.
        """

        if y is None:
            raise ValueError(
                "y (benchmark returns) must be provided for ReturnBasedTracker"
            )

        X, y = skv.validate_data(self, X, y)

        excess_returns = X - y[:, np.newaxis]

        super().fit(excess_returns, y=None, **fit_params)

        return self
