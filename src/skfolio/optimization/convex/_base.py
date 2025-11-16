"""Base Convex Optimization estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# The optimization features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from enum import auto

import cvxpy as cp
import cvxpy.constraints.constraint as cpc
import numpy as np
import numpy.typing as npt
import scipy as sc
import scipy.sparse.linalg as scl
import sklearn.utils.metadata_routing as skm
from cvxpy.reductions.solvers.defines import MI_SOLVERS

import skfolio.typing as skt
from skfolio._constants import _ParamKey
from skfolio.measures import RiskMeasure, owa_gmd_weights
from skfolio.optimization._base import BaseOptimization
from skfolio.prior import BasePrior, ReturnDistribution
from skfolio.uncertainty_set import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
    UncertaintySet,
)
from skfolio.utils.equations import equations_to_matrix, group_cardinalities_to_matrix
from skfolio.utils.tools import AutoEnum, cache_method, input_to_array

INSTALLED_SOLVERS = cp.installed_solvers()


class ObjectiveFunction(AutoEnum):
    r"""Enumeration of objective functions.

    Attributes
    ----------
    MINIMIZE_RISK : str
        Minimize the risk measure.

    MAXIMIZE_RETURN : str
        Maximize the expected return.

    MAXIMIZE_UTILITY : str
        Maximize the utility  :math:`w^T\mu - \lambda \times risk(w)`.

    MAXIMIZE_RATIO : str
        Maximize the ratio  :math:`\frac{w^T\mu - R_{f}}{risk(w)}`.
    """

    MINIMIZE_RISK = auto()
    MAXIMIZE_RETURN = auto()
    MAXIMIZE_UTILITY = auto()
    MAXIMIZE_RATIO = auto()


class ConvexOptimization(BaseOptimization, ABC):
    r"""Base class for all convex optimization estimators in skfolio.

    All risk measures that have a convex formulation are defined in class methods with
    naming convention: `_{risk_measure}_risk`. That naming convention is used for
    dynamic lookup.

    CVX expressions that are shared among multiple risk measures are cached in a
    dictionary named `_cvx_cache`.
    This is to avoid cvx expression duplication and improve performance and convergence.

    Parameters
    ----------
    risk_measure : RiskMeasure, default=RiskMeasure.VARIANCE
        :class:`~skfolio.meta.RiskMeasure` of the optimization.
        Can be any of:

            * VARIANCE
            * SEMI_VARIANCE
            * STANDARD_DEVIATION
            * SEMI_DEVIATION
            * MEAN_ABSOLUTE_DEVIATION
            * FIRST_LOWER_PARTIAL_MOMENT
            * CVAR
            * EVAR
            * WORST_REALIZATION
            * CDAR
            * MAX_DRAWDOWN
            * AVERAGE_DRAWDOWN
            * EDAR
            * ULCER_INDEX
            * GINI_MEAN_DIFFERENCE_RATIO

        The default is `RiskMeasure.VARIANCE`.

    prior_estimator : BasePrior, optional
        :ref:`Prior estimator <prior>`.
        The prior estimator is used to estimate the :class:`~skfolio.prior.ReturnDistribution`
        containing the estimation of assets expected returns, covariance matrix,
        returns and Cholesky decomposition of the covariance.
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    min_weights : float | dict[str, float] | array-like of shape (n_assets, ) | None, default=0.0
        Minimum assets weights (weights lower bounds).
        If a float is provided, it is applied to each asset.
        `None` is equivalent to `-np.Inf` (no lower bound).
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset minimum weight) and the input `X` of the `fit` method must
        be a DataFrame with the assets names in columns.
        When using a dictionary, assets values that are not provided are assigned
        a minimum weight of `0.0`.
        The default value is `0.0` (no short selling).

        Example:

           * `min_weights = 0` --> long only portfolio (no short selling).
           * `min_weights = None` --> no lower bound (same as `-np.Inf`).
           * `min_weights = -2` --> each weight must be above -200%.
           * `min_weights = {"SX5E": 0, "SPX": -2}`
           * `min_weights = [0, -2]`

    max_weights : float | dict[str, float] | array-like of shape (n_assets, ) | None, default=1.0
        Maximum assets weights (weights upper bounds).
        If a float is provided, it is applied to each asset.
        `None` is equivalent to `+np.Inf` (no upper bound).
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset maximum weight) and the input `X` of the `fit` method must
        be a DataFrame with the assets names in columns.
        When using a dictionary, assets values that are not provided are assigned
        a minimum weight of `1.0`.
        The default value is `1.0` (each asset is below 100%).

        Example:

           * `max_weights = 0` --> no long position (short only portfolio).
           * `max_weights = None` --> no upper bound.
           * `max_weights = 2` --> each weight must be below 200%.
           * `max_weights = {"SX5E": 1, "SPX": 2}`
           * `max_weights = [1, 2]`

    budget : float | None, default=1.0
        Investment budget. It is the sum of long positions and short positions (sum of
        all weights). `None` means no budget constraints.
        The default value is `1.0` (fully invested portfolio).

        For example:

             * `budget = 1` --> fully invested portfolio.
             * `budget = 0` --> market neutral portfolio.
             * `budget = None` --> no constraints on the sum of weights.

    min_budget : float, optional
        Minimum budget. It is the lower bound of the sum of long and short positions
        (sum of all weights). If provided, you must set `budget=None`.
        The default (`None`) means no minimum budget constraint.

    max_budget :  float, optional
        Maximum budget. It is the upper bound of the sum of long and short positions
        (sum of all weights). If provided, you must set `budget=None`.
        The default (`None`) means no maximum budget constraint.

    max_short : float, optional
        Maximum short position. The short position is defined as the sum of negative
        weights (in absolute term).
        The default (`None`) means no maximum short position.

    max_long : float, optional
        Maximum long position. The long position is defined as the sum of positive
        weights.
        The default (`None`) means no maximum long position.

    cardinality : int, optional
        Specifies the cardinality constraint to limit the number of invested assets
        (non-zero weights). This feature requires a mixed-integer solver. For an
        open-source option, we recommend using SCIP by setting `solver="SCIP"`.
        To install it, use: `pip install cvxpy[SCIP]`. For commercial solvers,
        supported options include MOSEK, GUROBI, or CPLEX.

    group_cardinalities : dict[str, int], optional
        A dictionary specifying cardinality constraints for specific groups of assets.
        The keys represent group names (strings), and the values specify the maximum
        number of assets allowed in each group. You must provide the groups using the
        `groups` parameter. This requires a mixed-integer solver (see `cardinality`
        for more details).

    threshold_long : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Specifies the minimum weight threshold for assets in the portfolio to be
        considered as a long position. Assets with weights below this threshold
        will not be included as part of the portfolio's long positions. This
        constraint can help eliminate insignificant allocations.
        This requires a mixed-integer solver (see `cardinality` for more details).
        It follows the same format as `min_weights` and `max_weights`.

    threshold_short : float | dict[str, float] | array-like of shape (n_assets, ), optional
        Specifies the maximum weight threshold for assets in the portfolio to be
        considered as a short position. Assets with weights above this threshold
        will not be included as part of the portfolio's short positions. This
        constraint can help control the magnitude of short positions.
        This requires a mixed-integer solver (see `cardinality` for more details).
        It follows the same format as `min_weights` and `max_weights`.

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
        DataFrame with the assets names in columns.
        The default value is `0.0`.

        .. warning::

            Based on the above formula, the periodicity of the transaction costs
            needs to be homogenous to the periodicity of :math:`\mu`. For example, if
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
        DataFrame with the assets names in columns.
        The default value is `0.0`.

        .. warning::

            Based on the above formula, the periodicity of the management fees needs to
            be homogenous to the periodicity of :math:`\mu`. For example, if the input
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
        portfolio cost and the portfolio turnover.
        If a float is provided, it is applied to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset previous weight) and the input `X` of the `fit` method must
        be a DataFrame with the assets names in columns.
        The default (`None`) means no previous weights.
        Additionally, when `fallback="previous_weights"`, failures will fall back to
        these weights if provided.

    l1_coef : float, default=0.0
        L1 regularization coefficient.
        It is used to penalize the objective function by the L1 norm:

        .. math:: l1\_coef \times \Vert w \Vert_{1} = l1\_coef \times \sum_{i=1}^{N} |w_{i}|

        Increasing this coefficient will reduce the number of non-zero weights
        (cardinality). It tends to increase robustness (out-of-sample stability) but
        reduces diversification.
        The default value is `0.0`.

    l2_coef : float, default=0.0
        L2 regularization coefficient.
        It is used to penalize the objective function by the L2 norm:

        .. math:: l2\_coef \times \Vert w \Vert_{2}^{2} = l2\_coef \times \sum_{i=1}^{N} w_{i}^2

        It tends to increase robustness (out-of-sample stability).
        The default value is `0.0`.

    mu_uncertainty_set_estimator : BaseMuUncertaintySet, optional
        :ref:`Mu Uncertainty set estimator <uncertainty_set_estimator>`.
        If provided, the assets expected returns are modelled with an ellipsoidal
        uncertainty set. It is called worst-case optimization and is a class of robust
        optimization. It reduces the instability that arises from the estimation errors
        of the expected returns.
        The worst case portfolio expect return is:

        .. math:: w^T\hat{\mu} - \kappa_{\mu}\lVert S_{\mu}^\frac{1}{2}w\rVert_{2}

        with :math:`\kappa` the size of the ellipsoid (confidence region) and
        :math:`S` its shape.
        The default (`None`) means that no uncertainty set is used.

    covariance_uncertainty_set_estimator : BaseCovarianceUncertaintySet, optional
        :ref:`Covariance Uncertainty set estimator <uncertainty_set_estimator>`.
        If provided, the assets covariance matrix is modelled with an ellipsoidal
        uncertainty set. It is called worst-case optimization and is a class of robust
        optimization. It reduces the instability that arises from the estimation errors
        of the covariance matrix.
        The default (`None`) means that no uncertainty set is used.

    linear_constraints : array-like of shape (n_constraints,), optional
        Linear constraints.
        The linear constraints must match any of following patterns:

            * `"ref1 >= a"`
            * `"ref1 == b"`
            * `"ref1 >= ref1"`
            * `"a * ref1 + b * ref2 + c <= d * ref3"`

        With `"ref1"`, `"ref2"` ... the assets names or the groups names provided
        in the parameter `groups`. Assets names can be referenced without the need of
        `groups` if the input `X` of the `fit` method is a DataFrame with these
        assets names in columns.

        For example:

            * `"SPX >= 0.10"` --> SPX weight must be greater than 10% (note that you can also use `min_weights`)
            * `"SX5E + TLT >= 0.2"` --> the sum of SX5E and TLT weights must be greater than 20%
            * `"US == 0.7"` --> the sum of all US weights must be equal to 70%
            * `"Equity == 3 * Bond"` --> the sum of all Equity weights must be equal to 3 times the sum of all Bond weights.
            * `"2*SPX + 3*Europe <= Bond + 0.05"` --> mixing assets and group constraints

    groups : dict[str, list[str]] or array-like of shape (n_groups, n_assets), optional
        The assets groups referenced in `linear_constraints`.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset groups) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.

        For example:

            * `groups = {"SX5E": ["Equity", "Europe"], "SPX": ["Equity", "US"], "TLT": ["Bond", "US"]}`
            * `groups = [["Equity", "Equity", "Bond"], ["Europe", "US", "US"]]`

    left_inequality : array-like of shape (n_constraints, n_assets), optional
        Left inequality matrix :math:`A` of the linear
        constraint :math:`A \cdot w \leq b`.

    right_inequality : array-like of shape (n_constraints, ), optional
        Right inequality vector :math:`b` of the linear
        constraint :math:`A \cdot w \leq b`.

    risk_free_rate : float, default=0.0
        Risk-free interest rate.
        The default value is `0.0`.

    min_acceptable_return : float, optional
        The minimum acceptable return used to distinguish "downside" and "upside"
        returns for the computation of lower partial moments:

            * First Lower Partial Moment
            * Semi-Variance
            * Semi-Deviation

        The default (`None`) is to use the mean.

    cvar_beta : float, default=0.95
        CVaR (Conditional Value at Risk) confidence level.
        The default value is `0.95`.

    evar_beta : float, default=0
        EVaR (Entropic Value at Risk) confidence level.
        The default value is `0.95`.

    cdar_beta : float, default=0.95
        CDaR (Conditional Drawdown at Risk) confidence level.
        The default value is `0.95`.

    edar_beta : float, default=0.95
        EDaR (Entropic Drawdown at Risk) confidence level.
        The default value is `0.95`.

    add_objective : Callable[[cp.Variable], cp.Expression], optional
        Add a custom objective to the existing objective expression.
        It is a function that must take as argument the weights `w` and returns a
        CVXPY expression.

    add_constraints : Callable[[cp.Variable], cp.Expression|list[cp.Expression]], optional
        Add a custom constraint or a list of constraints to the existing constraints.
        It is a function that must take as argument the weights `w` and returns a
        CVXPY expression or a list of CVXPY expressions.

    overwrite_expected_return : Callable[[cp.Variable], cp.Expression], optional
        Overwrite the expected return :math:`\mu \cdot w` with a custom expression.
        It is a function that must take as argument the weights `w` and returns a
        CVXPY expression.

    solver : str, default="CLARABEL"
        The solver to use. The default is "CLARABEL" which is written in Rust and has
        better numerical stability and performance than ECOS and SCS. Cvxpy will replace
        its default solver "ECOS" by "CLARABEL" in future releases.
        For more details about available solvers, check the CVXPY documentation:
        https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver

    solver_params : dict, optional
        Solver parameters. For example, `solver_params=dict(verbose=True)`.
        The default (`None`) is to use `{"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9}`
        for the solver "CLARABEL" and the CVXPY default otherwise.
        For more details about solver arguments, check the CVXPY documentation:
        https://www.cvxpy.org/tutorial/solvers

    scale_objective : float, optional
        Scale each objective element by this value.
        It can be used to increase the optimization accuracies in specific cases.
        The default (`None`) is set depending on the problem.

    scale_constraints : float, optional
        Scale each constraint element by this value.
        It can be used to increase the optimization accuracies in specific cases.
        The default (`None`) is set depending on the problem.

    save_problem : bool, default=False
        If this is set to True, the CVXPY Problem is saved in `problem_`.
        The default is `False`.

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
    weights_ : ndarray of shape (n_assets,) or (n_optimizations, n_assets)
        Weights of the assets.

    problem_values_ :  dict[str, float] | list[dict[str, float]] of size n_optimizations
        Expression values retrieved from the CVXPY problem.

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    mu_uncertainty_set_estimator_ : BaseMuUncertaintySet
        Fitted `mu_uncertainty_set_estimator` if provided.

    covariance_uncertainty_set_estimator_ : BaseCovarianceUncertaintySet
        Fitted `covariance_uncertainty_set_estimator` if provided.

    problem_: cvxpy.Problem
        CVXPY problem used for the optimization. Only when `save_problem` is set to
        `True`.

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

    _solver_params: dict
    _scale_objective: cp.Constant
    _scale_constraints: cp.Constant
    _cvx_cache: dict

    problem_: cp.Problem
    problem_values_: dict[str, float] | list[dict[str, float]]
    prior_estimator_: BasePrior
    mu_uncertainty_set_estimator_: BaseMuUncertaintySet
    covariance_uncertainty_set_estimator_: BaseCovarianceUncertaintySet

    @abstractmethod
    def __init__(
        self,
        risk_measure: RiskMeasure = RiskMeasure.VARIANCE,
        prior_estimator: BasePrior | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1.0,
        min_budget: float | None = None,
        max_budget: float | None = None,
        max_short: float | None = None,
        max_long: float | None = None,
        cardinality: int | None = None,
        group_cardinalities: dict[str, int] | None = None,
        threshold_long: skt.MultiInput | None = None,
        threshold_short: skt.MultiInput | None = None,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
        target_weights: skt.MultiInput | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        l1_coef: float = 0.0,
        l2_coef: float = 0.0,
        mu_uncertainty_set_estimator: BaseMuUncertaintySet | None = None,
        covariance_uncertainty_set_estimator: (
            BaseCovarianceUncertaintySet | None
        ) = None,
        risk_free_rate: float = 0.0,
        min_acceptable_return: skt.Target | None = None,
        cvar_beta: float = 0.95,
        evar_beta: float = 0.95,
        cdar_beta: float = 0.95,
        edar_beta: float = 0.95,
        solver: str = "CLARABEL",
        solver_params: dict | None = None,
        scale_objective: float | None = None,
        scale_constraints: float | None = None,
        save_problem: bool = False,
        add_objective: skt.ExpressionFunction | None = None,
        add_constraints: skt.ExpressionFunction | None = None,
        overwrite_expected_return: skt.ExpressionFunction | None = None,
        portfolio_params: dict | None = None,
        fallback: skt.Fallback = None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            previous_weights=previous_weights,
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
        )
        if risk_measure.is_annualized:
            warnings.warn(
                f"The annualized risk measure {risk_measure} will be converted"
                f"to its non-annualized version {risk_measure.non_annualized_measure}",
                stacklevel=2,
            )
            risk_measure = risk_measure.non_annualized_measure
        self.risk_measure = risk_measure
        self.prior_estimator = prior_estimator
        self.mu_uncertainty_set_estimator = mu_uncertainty_set_estimator
        self.covariance_uncertainty_set_estimator = covariance_uncertainty_set_estimator
        self.min_weights = min_weights
        self.max_weights = max_weights
        self.budget = budget
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.max_short = max_short
        self.max_long = max_long
        self.cardinality = cardinality
        self.group_cardinalities = group_cardinalities
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short
        self.min_acceptable_return = min_acceptable_return
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees
        self.target_weights = target_weights
        self.groups = groups
        self.linear_constraints = linear_constraints
        self.left_inequality = left_inequality
        self.right_inequality = right_inequality
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.risk_free_rate = risk_free_rate
        self.add_objective = add_objective
        self.add_constraints = add_constraints
        self.overwrite_expected_return = overwrite_expected_return
        self.solver = solver
        self.solver_params = solver_params
        self.save_problem = save_problem
        self.scale_objective = scale_objective
        self.scale_constraints = scale_constraints
        self.cvar_beta = cvar_beta
        self.evar_beta = evar_beta
        self.cdar_beta = cdar_beta
        self.edar_beta = edar_beta

        self._clear_models_cache()

    def _call_custom_func(
        self, func: skt.ExpressionFunction, w: cp.Variable, name: str = "custom_func"
    ) -> cp.Expression | list[cp.Expression]:
        """Call a user specific function, infer arguments and perform validation.

        Parameters
        ----------
        func : Callable[[cvxpy Variable, any], cvxpy Expression]
            The custom function. Must have one or two positional arguments.
            The first argument is the CVXPY weight variable `w` and the second is
            the reference to the class itself.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        Returns
        -------
        result : cvxpy Expression | list[cvxpy Expression]
            Result of calling the custom function.
        """
        try:
            # noinspection PyUnresolvedReferences
            func_code = func.__code__
        except AttributeError as err:
            raise ValueError("Custom functions is invalid") from err

        if func_code.co_argcount == 1:
            args = (w,)
        elif func_code.co_argcount == 2:
            args = (w, self)
        else:
            raise ValueError(
                "Custom functions must have 1 or 2 positional arguments, got"
                f" {func_code.co_argcount}"
            )
        try:
            return func(*args)
        except Exception as err:
            raise TypeError(
                f"Error while calling {name}. "
                f"{name} must be a function taking as argument "
                "the weight variable OR the weight variable and the estimator object."
            ) from err

    def _clear_models_cache(self):
        """Clear the cache of CVX models."""
        self._cvx_cache = {}

    def _get_weight_constraints(
        self,
        n_assets: int,
        w: cp.Variable,
        factor: skt.Factor,
        allow_negative_weights: bool = True,
    ) -> list[cpc.Constraint]:
        """Compute weight constraints from input parameters.

        Parameters
        ----------
        n_assets : int
            Number of assets.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
            Cvxpy variable or constant.

        Returns
        -------
        constraints : list[cvxpy Constraint]
            The list of weight constraints.
        """
        constraints = []

        # Clean and convert to array
        min_weights = self.min_weights
        max_weights = self.max_weights
        threshold_long = self.threshold_long
        threshold_short = self.threshold_short
        groups = self.groups

        if min_weights is not None:
            min_weights = self._clean_input(
                min_weights,
                n_assets=n_assets,
                fill_value=0,
                name="min_weights",
            )

        if max_weights is not None:
            max_weights = self._clean_input(
                max_weights,
                n_assets=n_assets,
                fill_value=1,
                name="max_weights",
            )

        if threshold_long is not None:
            threshold_long = self._clean_input(
                threshold_long,
                n_assets=n_assets,
                fill_value=0,
                name="threshold_long",
            )
            if np.all(threshold_long == 0):
                threshold_long = None

        if threshold_short is not None:
            threshold_short = self._clean_input(
                threshold_short,
                n_assets=n_assets,
                fill_value=0,
                name="threshold_short",
            )
            if np.all(threshold_short == 0):
                threshold_short = None

        if groups is not None:
            groups = input_to_array(
                items=groups,
                n_assets=n_assets,
                fill_value="",
                dim=2,
                assets_names=(
                    self.feature_names_in_
                    if hasattr(self, "feature_names_in_")
                    else None
                ),
                name="groups",
            )

        is_mip = (
            (self.cardinality is not None and self.cardinality < n_assets)
            or (self.group_cardinalities is not None)
            or self.threshold_long is not None
            or self.threshold_short is not None
        )

        if is_mip and self.solver not in MI_SOLVERS:
            raise ValueError(
                "You are using constraints that require a mixed-integer solver and "
                f"{self.solver} doesn't support MIP problems. For an open-source "
                "option, we recommend using SCIP by setting `solver='SCIP'`. "
                "To install it, use: `pip install cvxpy[SCIP]`. For commercial "
                "solvers, supported options include MOSEK, GUROBI, or CPLEX."
            )

        # Constraints
        if min_weights is not None:
            if not allow_negative_weights and np.any(min_weights < 0):
                raise ValueError(
                    f"{self.__class__.__name__} must have non negative `min_weights` "
                    f"constraint otherwise the problem becomes non-convex."
                )
            constraints.append(
                w * self._scale_constraints
                >= min_weights * factor * self._scale_constraints
            )

        if max_weights is not None:
            constraints.append(
                w * self._scale_constraints
                <= max_weights * factor * self._scale_constraints
            )

        if self.max_long is not None:
            max_long = float(self.max_long)
            if max_long <= 0:
                raise ValueError("`max_long` must be strictly positive")
            constraints.append(
                cp.sum(cp.pos(w)) * self._scale_constraints
                <= max_long * factor * self._scale_constraints
            )

        if self.max_short is not None:
            max_short = float(self.max_short)
            if max_short <= 0:
                raise ValueError("`max_short` must be strictly positive")
            constraints.append(
                cp.sum(cp.neg(w)) * self._scale_constraints
                <= max_short * factor * self._scale_constraints
            )

        if self.min_budget is not None:
            constraints.append(
                cp.sum(w) * self._scale_constraints
                >= float(self.min_budget) * factor * self._scale_constraints
            )

        if self.max_budget is not None:
            constraints.append(
                cp.sum(w) * self._scale_constraints
                <= float(self.max_budget) * factor * self._scale_constraints
            )

        if self.budget is not None:
            if self.max_budget is not None:
                raise ValueError(
                    "`max_budget`and `budget` cannot be provided at the same time"
                )
            if self.min_budget is not None:
                raise ValueError(
                    "`min_budget`and `budget` cannot be provided at the same time"
                )
            constraints.append(
                cp.sum(w) * self._scale_constraints
                == float(self.budget) * factor * self._scale_constraints
            )

        if is_mip:
            is_short = np.any(min_weights < 0)

            if max_weights is None or min_weights is None:
                raise ValueError(
                    "'max_weights' and 'min_weights' must be provided with cardinality "
                    "constraint"
                )
            if np.all(min_weights > 0):
                raise ValueError(
                    "Cardinality and Threshold constraint can only be applied "
                    "if 'min_weights' are not all strictly positive (you allow some "
                    "weights to be 0)"
                )

            if self.group_cardinalities is not None and groups is None:
                raise ValueError(
                    "When 'group_cardinalities' is provided, you must also "
                    "also provide 'groups'"
                )

            if (
                self.threshold_long is not None
                and self.threshold_short is None
                and is_short
            ):
                raise ValueError(
                    "When 'threshold_long' is provided and 'min_weights' can be negative "
                    "(short position are allowed), then 'threshold_short' must also be "
                    "provided"
                )

            if threshold_short is not None and threshold_long is None:
                raise ValueError(
                    "When 'threshold_short' is provided, 'threshold_long' must also be "
                    "provided"
                )

            if self.threshold_short is not None and is_short:
                constraints += _mip_weight_constraints_threshold_short(
                    n_assets=n_assets,
                    w=w,
                    factor=factor,
                    scale_constraints=self._scale_constraints,
                    cardinality=self.cardinality,
                    group_cardinalities=self.group_cardinalities,
                    max_weights=max_weights,
                    groups=groups,
                    min_weights=min_weights,
                    threshold_long=threshold_long,
                    threshold_short=threshold_short,
                )
            else:
                constraints += _mip_weight_constraints_no_short_threshold(
                    n_assets=n_assets,
                    w=w,
                    factor=factor,
                    scale_constraints=self._scale_constraints,
                    cardinality=self.cardinality,
                    group_cardinalities=self.group_cardinalities,
                    max_weights=max_weights,
                    groups=groups,
                    min_weights=min_weights,
                    threshold_long=threshold_long,
                )

        if self.linear_constraints is not None:
            if groups is None:
                if not hasattr(self, "feature_names_in_"):
                    raise ValueError(
                        "If `linear_constraints` is provided you must provide either"
                        " `groups` or `X` as a DataFrame with asset names in columns"
                    )
                groups = np.asarray([self.feature_names_in_])
            a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
                groups=groups,
                equations=self.linear_constraints,
                raise_if_group_missing=False,
            )
            if len(a_eq) != 0:
                constraints.append(
                    a_eq @ w * self._scale_constraints
                    - b_eq * factor * self._scale_constraints
                    == 0
                )
            if len(a_ineq) != 0:
                constraints.append(
                    a_ineq @ w * self._scale_constraints
                    - b_ineq * factor * self._scale_constraints
                    <= 0
                )

        if self.left_inequality is not None and self.right_inequality is not None:
            left_inequality = np.asarray(self.left_inequality)
            right_inequality = np.asarray(self.right_inequality)
            if left_inequality.ndim != 2:
                raise ValueError(
                    f"`left_inequality` must be a 2D array, got {left_inequality.ndim}D"
                    " array"
                )
            if right_inequality.ndim != 1:
                raise ValueError(
                    "`right_inequality` must be a 1D array, got"
                    f" {right_inequality.ndim}D array"
                )
            if left_inequality.shape[1] != n_assets:
                raise ValueError(
                    "`left_inequality` must be of shape (n_inequalities, n_assets) "
                    f"with n_assets={n_assets}, got {left_inequality.shape[1]}"
                )
            if left_inequality.shape[0] != right_inequality.shape[0]:
                raise ValueError(
                    "`left_inequality` and `right_inequality` must have same number of"
                    f" rows (i.e. n_inequalities) , got {left_inequality.shape[0]} and"
                    f" {right_inequality.shape[0]}"
                )
            constraints.append(
                left_inequality @ w * self._scale_constraints
                - right_inequality * factor * self._scale_constraints
                <= 0
            )

        return constraints

    def _set_solver_params(self, default: dict | None) -> None:
        """Set the solver params by saving its value in `_solver_params`.
        It uses `solver` if provided otherwise it uses the `default` solver.

        Parameters
        ----------
        default : str
            The default solver params to use when `solver_params` is `None`.
        """
        if self.solver_params is None:
            self._solver_params = default if default is not None else {}
        else:
            self._solver_params = self.solver_params

    def _set_scale_objective(self, default: float) -> None:
        """Set the objective scale by saving its value in `_scale_objective`.
        It uses `scale_objective` if provided otherwise it uses the `default` scale.

        Parameters
        ----------
        default : float
            The default objective scale to use when `scale_objective` is `None`.
        """
        if self.scale_objective is None:
            self._scale_objective = cp.Constant(default)
        else:
            self._scale_objective = cp.Constant(self.scale_objective)

    def _set_scale_constraints(self, default: float) -> None:
        """Set the constraints scale by saving its value in `_scale_constraints`.
        It uses `scale_constraints` if provided otherwise it uses the `default` scale.

        Parameters
        ----------
        default : float
            The default constraints scale to use when `scale_constraints` is `None`.
        """
        if self.scale_constraints is None:
            self._scale_constraints = cp.Constant(default)
        else:
            self._scale_constraints = cp.Constant(self.scale_constraints)

    def _get_custom_objective(self, w: cp.Variable) -> cp.Expression:
        """Return the CVXPY expression evaluated by calling the `add_objective`
        function if provided, otherwise returns the CVXPY constant `0`.

        Parameters
        ----------
        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY expression evaluated by calling the `add_objective`
            function if provided, otherwise returns the CVXPY constant `0`.
        """
        if self.add_objective is None:
            return cp.Constant(0)
        return self._call_custom_func(
            func=self.add_objective, w=w, name="add_objective"
        )

    def _get_custom_constraints(self, w: cp.Variable) -> list[cp.Expression]:
        """Return the list of CVXPY expressions evaluated by calling the
        `add_constraint`s function if provided, otherwise returns an empty list.

        Parameters
        ----------
        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        Returns
        -------
        expressions : list of cvxpy Expression
            The list of CVXPY expressions evaluated by calling the
            `add_constraints` function if provided, otherwise returns an empty list.
        """
        if self.add_constraints is None:
            return []
        constraints = self._call_custom_func(
            func=self.add_constraints, w=w, name="add_constraint"
        )
        if isinstance(constraints, list):
            return constraints
        return [constraints]

    @cache_method("_cvx_cache")
    def _cvx_expected_return(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> cp.Expression:
        """Expected Return expression."""
        if self.overwrite_expected_return is None:
            expected_return = return_distribution.mu @ w
        else:
            expected_return = self._call_custom_func(
                func=self.overwrite_expected_return,
                w=w,
                name="overwrite_expected_return",
            )
        return expected_return

    # Model reused among multiple risk measure
    def _solve_problem(
        self,
        problem: cp.Problem,
        w: cp.Variable,
        factor: skt.Factor,
        parameters_values: skt.ParametersValues = None,
        expressions: dict[str, cp.Expression] | None = None,
    ) -> None:
        """Solve the CVXPY Problem and save the results in `weights_`, `problem_values_`
        and `problem_`.

        Parameters
        ----------
        problem : cvxpy Problem
            The CVXPY Problem.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        expressions : dict[str, cvxpy Expression] | None, optional
            Dictionary of CVXPY Expressions from which values are retrieved and saved
            in `expression_values_`. It is used to save additional information about
            the problem.

        parameters_values: list[tuple[cvxpy Parameter, float | ndarray]], optional
            A list of tuple of CVXPY Parameter and their values.
            If The values are ndarray instead of float, the optimization is solved for
            each element in the array.

        factor: cvxpy Variable | cvxpy Constant
           CVXPY Variable or Constant used for RatioMeasure optimization problems.
        """
        if self.solver not in INSTALLED_SOLVERS:
            raise ValueError(f"The solver {self.solver} is not installed.")

        if parameters_values is None:
            parameters_values = []

        if expressions is None:
            expressions = {}

        n_optimizations = 1
        if len(parameters_values) != 0:
            # If the parameter value is a list, each element is the parameter value of
            # a distinct optimization. Therefore, each list must have same length.
            sizes = [len(v) for p, v in parameters_values if not np.isscalar(v)]
            if not np.all(sizes):
                raise ValueError(
                    "All list elements from `parameters_values` should have same length"
                )
            if len(sizes) != 0:
                n_optimizations = sizes[0]
            # Scalar parameter values will be used in each optimization, therefore we
            # transform them to a list.
            parameters_values = [
                (p, [v] * n_optimizations) if np.isscalar(v) else (p, v)
                for p, v in parameters_values
            ]

        if n_optimizations == 1:
            for parameter, values in parameters_values:
                parameter.value = values[0]

            self.weights_, self.problem_values_ = _solve(
                w=w,
                factor=factor,
                expressions=expressions,
                problem=problem,
                solver=self.solver,
                solver_params=self._solver_params,
                risk_measure=self.risk_measure,
                scale_objective=self._scale_objective,
            )
        else:
            all_weights = []
            all_problem_values = []
            all_errors = []
            with warnings.catch_warnings():
                warnings.simplefilter("once", UserWarning)
                for i in range(n_optimizations):
                    for parameter, values in parameters_values:
                        parameter.value = values[i]

                    try:
                        weights, problem_values = _solve(
                            w=w,
                            factor=factor,
                            expressions=expressions,
                            problem=problem,
                            solver=self.solver,
                            solver_params=self._solver_params,
                            risk_measure=self.risk_measure,
                            scale_objective=self._scale_objective,
                        )
                        error = None
                    except cp.SolverError as solver_error:
                        if self.raise_on_failure:
                            raise
                        error = str(solver_error)
                        warnings.warn(error, stacklevel=2)
                        problem_values = None
                        weights = np.full(w.shape, np.nan, dtype=float)

                    all_problem_values.append(problem_values)
                    all_weights.append(weights)
                    all_errors.append(error)

            all_weights = np.array(all_weights, dtype=float)
            if np.isnan(all_weights).all():
                raise cp.SolverError(
                    f"All {n_optimizations} optimizations failed, with last optimization error {all_errors[-1]}"
                )
            self.weights_ = all_weights
            self.problem_values_ = all_problem_values
            self.error_ = all_errors

        if self.save_problem:
            self.problem_ = problem

        self._clear_models_cache()

    @cache_method("_cvx_cache")
    def _cvx_mu_uncertainty_set(
        self, mu_uncertainty_set: UncertaintySet, w: cp.Variable
    ) -> cp.Expression:
        """Uncertainty Set expression of expected returns.

        Parameters
        ----------
        mu_uncertainty_set : UncertaintySet
            The uncertainty set model of expected returns.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY Expression of the uncertainty set of expected returns.
        """
        return mu_uncertainty_set.k * cp.pnorm(
            sc.linalg.sqrtm(mu_uncertainty_set.sigma) @ w, 2
        )

    @cache_method("_cvx_cache")
    def _cvx_regularization(self, w: cp.Variable) -> cp.Expression:
        """L1 and L2 regularization expression.

        Parameters
        ----------
        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : cvxpy Expression
           The CVXPY Expression of L1 and L2 regularization.
        """
        # Norm L1
        if self.l1_coef is None or self.l1_coef == 0:
            l1_reg = cp.Constant(0)
        else:
            l1_reg = cp.Constant(self.l1_coef) * cp.norm(w, 1)

        # Norm L2
        if self.l2_coef is None or self.l2_coef == 0:
            l2_reg = cp.Constant(0)
        else:
            l2_reg = self.l2_coef * cp.sum_squares(w)
        regularization = l1_reg + l2_reg
        return regularization

    @cache_method("_cvx_cache")
    def _cvx_transaction_cost(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> cp.Expression:
        """Transaction cost expression.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
            Additional variable used for the optimization of some objective function
            like the ratio maximization.

        Returns
        -------
        expression : cvxpy Expression
           The CVXPY Expression of transaction cost.
        """
        n_assets = return_distribution.returns.shape[1]

        transaction_costs = self._clean_input(
            self.transaction_costs,
            n_assets=n_assets,
            fill_value=0,
            name=_ParamKey.TRANSACTION_COSTS.value,
        )
        if np.all(transaction_costs == 0):
            return cp.Constant(0)

        previous_weights = self._clean_previous_weights(n_assets=n_assets)

        if np.isscalar(transaction_costs):
            return transaction_costs * cp.norm(previous_weights * factor - w, 1)
        return cp.norm(
            cp.multiply(transaction_costs, (previous_weights * factor - w)),
            1,
        )

    @cache_method("_cvx_cache")
    def _cvx_management_fee(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> cp.Expression:
        """Management fee expression.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : cvxpy Expression
           The CVXPY Expression of management fee .
        """
        n_assets = return_distribution.returns.shape[1]

        management_fees = self._clean_input(
            self.management_fees,
            n_assets=n_assets,
            fill_value=0,
            name=_ParamKey.MANAGEMENT_FEES.value,
        )
        if np.all(management_fees == 0):
            return cp.Constant(0)

        if np.isscalar(management_fees):
            management_fees *= np.ones(n_assets)
        return management_fees @ w

    @cache_method("_cvx_cache")
    def _cvx_returns(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> cp.Expression:
        """Expression of the portfolio returns series.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            asset returns distribution DataModel.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY Expression the portfolio returns series.
        """
        returns = return_distribution.returns @ w
        return returns

    @cache_method("_cvx_cache")
    def _turnover(
        self, n_assets: int, w: cp.Variable, factor: skt.Factor
    ) -> cp.Expression:
        """Expression of the portfolio turnover.

        Parameters
        ----------
        n_assets : int
            The number of assets.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
            Additional variable used for the optimization of some objective function
            like the ratio maximization.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY Expression the portfolio turnover.
        """
        if self.previous_weights is None:
            raise ValueError(
                "If you provide `max_turnover`, you must also provide "
                " `previous_weights`"
            )
        previous_weights = self._clean_input(
            self.previous_weights,
            n_assets=n_assets,
            fill_value=0,
            name=_ParamKey.PREVIOUS_WEIGHTS.value,
        )
        if np.isscalar(previous_weights):
            previous_weights *= np.ones(n_assets)
        turnover = cp.abs(w - previous_weights * factor)
        return turnover

    @cache_method("_cvx_cache")
    def _cvx_min_acceptable_return(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        min_acceptable_return: skt.Target = None,
    ) -> cp.Expression:
        """Expression of the portfolio Minimum Acceptable Returns.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            asset returns distribution DataModel..

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        min_acceptable_return : float | ndarray of shape (n_assets,)
            The minimum acceptable return used to distinguish "downside" and "upside"
            returns for the computation of lower partial moments.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY Expression the portfolio Minimum Acceptable Returns.
        """
        if min_acceptable_return is None:
            min_acceptable_return = return_distribution.mu
        if not np.isscalar(min_acceptable_return) and min_acceptable_return.shape != (
            len(min_acceptable_return),
            1,
        ):
            min_acceptable_return = min_acceptable_return[np.newaxis, :]
        mar = (return_distribution.returns - min_acceptable_return) @ w
        return mar

    @cache_method("_cvx_cache")
    def __cvx_drawdown(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> tuple[cp.Variable, list[cp.Expression]]:
        """Expression of the portfolio drawdown.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            asset returns distribution DataModel.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
            Additional variable used for the optimization of some objective function
            like the ratio maximization.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY Expression the portfolio drawdown.
        """
        n_observations = return_distribution.returns.shape[0]
        ptf_returns = self._cvx_returns(return_distribution=return_distribution, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            return_distribution=return_distribution, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(
            return_distribution=return_distribution, w=w
        )
        v = cp.Variable(n_observations + 1)
        constraints = [
            v[1:] * self._scale_constraints
            >= v[:-1] * self._scale_constraints
            - ptf_returns * self._scale_constraints
            + ptf_transaction_cost * self._scale_constraints
            + ptf_management_fee * self._scale_constraints,
            v[1:] * self._scale_constraints >= 0,
            v[0] * self._scale_constraints == 0,
        ]
        return v, constraints

    def _cvx_drawdown(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> tuple[cp.Variable, list[cp.Expression]]:
        """Expression of the portfolio drawdown.
        Wrapper around __cvx_drawdown to avoid re-adding the constraints when they
        have already been included in the problem.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            asset returns distribution DataModel.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
            Additional variable used for the optimization of some objective function
            like the ratio maximization.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY Expression the portfolio drawdown.
        """
        if "__cvx_drawdown" in self._cvx_cache:
            v, _ = self.__cvx_drawdown(
                return_distribution=return_distribution, w=w, factor=factor
            )
            return v, []
        return self.__cvx_drawdown(
            return_distribution=return_distribution, w=w, factor=factor
        )

    def _tracking_error(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        y: np.ndarray,
        factor: skt.Factor,
    ) -> cp.Expression:
        """Expression of the portfolio tracking error.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            asset returns distribution DataModel.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        y : ndarray of shape (n_observations,)
            Benchmark for the tracking error computation.

        factor : cvxpy Variable | cvxpy Constant
            Additional variable used for the optimization of some objective function
            like the ratio maximization.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY Expression the portfolio tracking error.
        """
        n_observations = return_distribution.returns.shape[0]
        ptf_returns = self._cvx_returns(return_distribution=return_distribution, w=w)
        tracking_error = cp.norm(ptf_returns - y * factor, "fro") / cp.sqrt(
            n_observations - 1
        )
        return tracking_error

    # Risk Measures risk models
    # They need to be named f'_{risk_measure}_risk' as they are loaded dynamically in
    # mean_risk_optimization()
    def _mean_absolute_deviation_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        min_acceptable_return: skt.Target,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Mean Absolute Deviation risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            asset returns distribution DataModel.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        min_acceptable_return : float | ndarray of shape (n_assets,)
            The minimum acceptable return used to distinguish "downside" and "upside"
            returns for the computation of lower partial moments.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints of the Mean Absolute Deviation risk
            measure.
        """
        n_observations = return_distribution.returns.shape[0]
        ptf_min_acceptable_return = self._cvx_min_acceptable_return(
            return_distribution=return_distribution,
            w=w,
            min_acceptable_return=min_acceptable_return,
        )
        v = cp.Variable(n_observations, nonneg=True)

        if return_distribution.sample_weight is None:
            risk = 2 * cp.sum(v) / n_observations
        else:
            risk = 2 * cp.sum(cp.multiply(return_distribution.sample_weight, v))

        constraints = [
            ptf_min_acceptable_return * self._scale_constraints
            >= -v * self._scale_constraints
        ]
        return risk, constraints

    def _first_lower_partial_moment_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        min_acceptable_return: skt.Target,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the First Lower Partial Moment risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            asset returns distribution DataModel.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        min_acceptable_return : float | ndarray of shape (n_assets,)
            The minimum acceptable return used to distinguish "downside" and "upside"
            returns for the computation of lower partial moments.

        factor : cvxpy Variable | cvxpy Constant
            Additional variable used for the optimization of some objective function
            like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints of the First Lower Partial Moment risk
            measure.
        """
        n_observations = return_distribution.returns.shape[0]
        ptf_min_acceptable_return = self._cvx_min_acceptable_return(
            return_distribution=return_distribution,
            w=w,
            min_acceptable_return=min_acceptable_return,
        )
        v = cp.Variable(n_observations, nonneg=True)

        if return_distribution.sample_weight is None:
            risk = cp.sum(v) / n_observations
        else:
            risk = cp.sum(cp.multiply(return_distribution.sample_weight, v))

        constraints = [
            self.risk_free_rate * factor * self._scale_constraints
            - ptf_min_acceptable_return * self._scale_constraints
            <= v * self._scale_constraints
        ]
        return risk, constraints

    def _standard_deviation_risk(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> skt.RiskResult:
        """Expression and Constraints of the Standard Deviation risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
            asset returns distribution DataModel.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints of the Standard Deviation risk measure.
        """
        v = cp.Variable(
            nonneg=True
        )  # nonneg=True instead of constraint v>=0 is preferred for better DCP analysis
        if return_distribution.cholesky is not None:
            z = return_distribution.cholesky
        else:
            z = np.linalg.cholesky(return_distribution.covariance)
        risk = v
        constraints = [
            cp.SOC(v * self._scale_constraints, z.T @ w * self._scale_constraints)
        ]
        return risk, constraints

    def _variance_risk(
        self, return_distribution: ReturnDistribution, w: cp.Variable
    ) -> skt.RiskResult:
        """Expression and Constraints of the Variance risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
           CVXPY Expression and Constraints the Variance risk measure.
        """
        risk, constraints = self._standard_deviation_risk(
            return_distribution=return_distribution, w=w
        )
        risk = cp.square(risk)
        return risk, constraints

    def _worst_case_variance_risk(
        self,
        return_distribution: ReturnDistribution,
        covariance_uncertainty_set: UncertaintySet,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Worst Case Variance.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        covariance_uncertainty_set : UncertaintySet
             :ref:`Covariance Uncertainty set estimator <uncertainty_set_estimator>`.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
           CVXPY Expression and Constraints the Worst Case Variance.
        """
        n_assets = return_distribution.returns.shape[1]
        x = cp.Variable((n_assets, n_assets), symmetric=True)
        y = cp.Variable((n_assets, n_assets), symmetric=True)
        w_reshaped = cp.reshape(w, (n_assets, 1), order="F")
        factor_reshaped = cp.reshape(factor, (1, 1), order="F")
        z1 = cp.vstack([x, w_reshaped.T])
        z2 = cp.vstack([w_reshaped, factor_reshaped])

        risk = covariance_uncertainty_set.k * cp.pnorm(
            sc.linalg.sqrtm(covariance_uncertainty_set.sigma)
            @ (cp.vec(x, order="F") + cp.vec(y, order="F")),
            2,
        ) + cp.trace(return_distribution.covariance @ (x + y))
        # semi-definite positive constraints
        constraints = [
            cp.hstack([z1, z2]) * self._scale_constraints >> 0,
            y * self._scale_constraints >> 0,
        ]
        return risk, constraints

    def _semi_variance_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        min_acceptable_return: skt.Target = None,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Semi Variance risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        min_acceptable_return : float | ndarray of shape (n_assets,)
            The minimum acceptable return used to distinguish "downside" and "upside"
            returns for the computation of lower partial moments.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the Semi Variance risk measure.
        """
        n_observations = return_distribution.returns.shape[0]
        ptf_min_acceptable_return = self._cvx_min_acceptable_return(
            return_distribution=return_distribution,
            w=w,
            min_acceptable_return=min_acceptable_return,
        )
        v = cp.Variable(n_observations, nonneg=True)

        if return_distribution.sample_weight is None:
            risk = cp.sum_squares(v) / (n_observations - 1)
        else:
            risk = cp.sum_squares(
                cp.multiply(np.sqrt(return_distribution.sample_weight), v)
            )

        constraints = [
            ptf_min_acceptable_return * self._scale_constraints
            >= -v * self._scale_constraints
        ]
        return risk, constraints

    def _semi_deviation_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        min_acceptable_return: skt.Target = None,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Semi Standard Deviation risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        min_acceptable_return : float | ndarray of shape (n_assets,)
            The minimum acceptable return used to distinguish "downside" and "upside"
            returns for the computation of lower partial moments.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the Semi Standard Deviation risk measure.
        """
        n_observations = return_distribution.returns.shape[0]
        ptf_min_acceptable_return = self._cvx_min_acceptable_return(
            return_distribution=return_distribution,
            w=w,
            min_acceptable_return=min_acceptable_return,
        )
        v = cp.Variable(n_observations, nonneg=True)

        if return_distribution.sample_weight is None:
            risk = cp.norm(v, 2) / np.sqrt(n_observations - 1)
        else:
            risk = cp.norm(
                cp.multiply(np.sqrt(return_distribution.sample_weight), v), 2
            )

        constraints = [
            ptf_min_acceptable_return * self._scale_constraints
            >= -v * self._scale_constraints
        ]
        return risk, constraints

    def _fourth_central_moment_risk(self, w: cp.Variable, factor: skt.Factor):
        raise NotImplementedError

    def _fourth_lower_partial_moment_risk(self, w: cp.Variable, factor: skt.Factor):
        raise NotImplementedError

    def _worst_realization_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Worst Realization risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the Worst Realization risk measure.
        """
        ptf_returns = self._cvx_returns(return_distribution=return_distribution, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            return_distribution=return_distribution, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(
            return_distribution=return_distribution, w=w
        )
        v = cp.Variable()
        risk = v
        constraints = [
            -ptf_returns * self._scale_constraints
            + ptf_transaction_cost * self._scale_constraints
            + ptf_management_fee * self._scale_constraints
            <= v * self._scale_constraints
        ]
        return risk, constraints

    def _cvar_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the CVaR risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the CVaR risk measure.
        """
        n_observations = return_distribution.returns.shape[0]
        ptf_returns = self._cvx_returns(return_distribution=return_distribution, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            return_distribution=return_distribution, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(
            return_distribution=return_distribution, w=w
        )
        alpha = cp.Variable()
        v = cp.Variable(n_observations, nonneg=True)
        if return_distribution.sample_weight is None:
            risk = alpha + cp.sum(v) / (n_observations * (1 - self.cvar_beta))
        else:
            risk = alpha + cp.sum(cp.multiply(return_distribution.sample_weight, v)) / (
                1 - self.cvar_beta
            )

        constraints = [
            ptf_returns * self._scale_constraints
            - ptf_transaction_cost * self._scale_constraints
            - ptf_management_fee * self._scale_constraints
            + alpha * self._scale_constraints
            + v * self._scale_constraints
            >= 0
        ]
        return risk, constraints

    def _evar_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the EVaR risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the EVaR risk measure.
        """
        n_observations = return_distribution.returns.shape[0]
        ptf_returns = self._cvx_returns(return_distribution=return_distribution, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            return_distribution=return_distribution, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(
            return_distribution=return_distribution, w=w
        )
        # We don't include the transaction_cost in the constraint otherwise the problem
        # is not DCP
        if not isinstance(ptf_transaction_cost, cp.Constant):
            warnings.warn(
                "The EVaR problem will be relaxed by removing the transaction costs"
                " from the Cone constraint to keep the problem DCP. The solution may"
                " not be accurate.",
                stacklevel=2,
            )

        x = cp.Variable()
        y = cp.Variable(nonneg=True)
        z = cp.Variable(n_observations)
        risk = x + y * np.log(1 / (n_observations * (1 - self.evar_beta)))
        constraints = [
            cp.sum(z) * self._scale_constraints <= y * self._scale_constraints,
            cp.constraints.ExpCone(
                -ptf_returns * self._scale_constraints
                + ptf_management_fee * self._scale_constraints
                - x * self._scale_constraints,
                np.ones(n_observations) * y * self._scale_constraints,
                z * self._scale_constraints,
            ),
        ]
        return risk, constraints

    def _max_drawdown_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the EVaR risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the EVaR risk measure.
        """
        v, constraints = self._cvx_drawdown(
            return_distribution=return_distribution, w=w, factor=factor
        )
        u = cp.Variable()
        risk = u
        constraints += [u * self._scale_constraints >= v[1:] * self._scale_constraints]
        return risk, constraints

    def _average_drawdown_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Average Drawdown risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the Average Drawdown risk measure.
        """
        n_observations = return_distribution.returns.shape[0]
        v, constraints = self._cvx_drawdown(
            return_distribution=return_distribution, w=w, factor=factor
        )
        risk = cp.sum(v[1:]) / n_observations
        return risk, constraints

    def _cdar_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the CDaR risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the CDaR risk measure.
        """
        n_observations = return_distribution.returns.shape[0]
        v, constraints = self._cvx_drawdown(
            return_distribution=return_distribution, w=w, factor=factor
        )
        alpha = cp.Variable()
        z = cp.Variable(n_observations, nonneg=True)
        risk = alpha + 1.0 / (n_observations * (1 - self.cdar_beta)) * cp.sum(z)
        constraints += [
            z * self._scale_constraints
            >= v[1:] * self._scale_constraints - alpha * self._scale_constraints
        ]
        return risk, constraints

    def _edar_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the EDaR risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the EDaR risk measure.
        """
        n_observations = return_distribution.returns.shape[0]
        v, constraints = self._cvx_drawdown(
            return_distribution=return_distribution, w=w, factor=factor
        )
        x = cp.Variable()
        y = cp.Variable(nonneg=True)
        z = cp.Variable(n_observations)
        risk = x + y * np.log(1 / (n_observations * (1 - self.edar_beta)))
        constraints += [
            cp.sum(z) * self._scale_constraints <= y * self._scale_constraints,
            cp.constraints.ExpCone(
                v[1:] * self._scale_constraints - x * self._scale_constraints,
                np.ones(n_observations) * y * self._scale_constraints,
                z * self._scale_constraints,
            ),
        ]
        return risk, constraints

    def _ulcer_index_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Ulcer Index risk measure.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the Ulcer Index risk measure.
        """
        v, constraints = self._cvx_drawdown(
            return_distribution=return_distribution, w=w, factor=factor
        )
        n_observations = return_distribution.returns.shape[0]
        risk = cp.norm(v[1:], 2) / (np.sqrt(n_observations))
        return risk, constraints

    def _gini_mean_difference_risk(
        self,
        return_distribution: ReturnDistribution,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Gini Mean Difference risk measure.

        The Gini mean difference (GMD) is a measure of dispersion introduced in the
        context of portfolio optimization by Yitzhaki (1982).
        The initial formulation was not used by practitioners due to the high number of
        variables that increases proportional to T(T-1)/2 ,

        Cajas (2021) proposed an alternative reformulation based on the ordered weighted
        averaging (OWA) operator for monotonic weights proposed by Chassein and
        Goerigk (2015). We implement this formulation which is more efficient for large
        scale problems.

        Parameters
        ----------
        return_distribution : ReturnDistribution
           asset returns distribution DataModel.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        factor : cvxpy Variable | cvxpy Constant
           Additional variable used for the optimization of some objective function
           like the ratio maximization.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
            CVXPY Expression and Constraints the Ulcer Index risk measure.
        """
        ptf_returns = self._cvx_returns(return_distribution=return_distribution, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            return_distribution=return_distribution, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(
            return_distribution=return_distribution, w=w
        )
        observation_nb = return_distribution.returns.shape[0]
        x = cp.Variable((observation_nb, 1))
        y = cp.Variable((observation_nb, 1))
        z = cp.Variable((observation_nb, 1))
        ones = np.ones((observation_nb, 1))
        risk = 2 * cp.sum(x + y)
        gmd_w = np.array(owa_gmd_weights(observation_nb) / 2).reshape(-1, 1)
        # noinspection PyTypeChecker
        constraints = [
            ptf_returns * self._scale_constraints
            - ptf_transaction_cost * self._scale_constraints
            - ptf_management_fee * self._scale_constraints
            == cp.reshape(z, (observation_nb,), order="F") * self._scale_constraints,
            z @ gmd_w.T <= ones @ x.T + y @ ones.T,
        ]
        return risk, constraints

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            prior_estimator=self.prior_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params):
        pass


def _mip_weight_constraints_no_short_threshold(
    n_assets: int,
    w: cp.Variable,
    factor: skt.Factor,
    scale_constraints: cp.Constant,
    cardinality: int | None,
    group_cardinalities: dict[str, int] | None,
    max_weights: np.ndarray | None,
    groups: np.ndarray | None,
    min_weights: np.ndarray | None,
    threshold_long: np.ndarray | None,
) -> list[cp.Expression]:
    """
    Create a list of MIP constraints for cardinality and threshold conditions
    when no short threshold is present. This only requires the creation of a single
    boolean variable array.
    """
    constraints = []

    is_short = np.any(min_weights < 0)

    is_invested_bool = cp.Variable(n_assets, boolean=True)

    if cardinality is not None and cardinality < n_assets:
        constraints.append(cp.sum(is_invested_bool) <= cardinality)

    if group_cardinalities is not None:
        a_card, b_card = group_cardinalities_to_matrix(
            groups=groups,
            group_cardinalities=group_cardinalities,
            raise_if_group_missing=False,
        )
        constraints.append(a_card @ is_invested_bool - b_card <= 0)

    if isinstance(factor, cp.Variable):
        is_invested_factor = cp.Variable(n_assets, nonneg=True)
        # We want (w <= cp.multiply(is_invested_short_bool, max_weights) * factor
        # but this is not DCP. So we introduce another variable and set
        # constraint to ensure its value is equal to is_invested_short_bool * factor

        M = 1e3
        # Big M method to activate or deactivate constraints
        # In the ratio homogenization procedure, the factor has been calibrated
        # to be around 0.1-10. By using M=1e3, we ensure that M is large enough while
        # not too large for improved MIP convergence.

        constraints += [
            is_invested_factor <= factor,
            is_invested_factor <= M * is_invested_bool,
            is_invested_factor >= factor - M * (1 - is_invested_bool),
        ]
        is_invested = is_invested_factor
    else:
        is_invested = is_invested_bool

    if threshold_long is not None:
        constraints.append(
            w * scale_constraints
            >= cp.multiply(is_invested, threshold_long) * scale_constraints
        )

    constraints.append(
        w * scale_constraints
        <= cp.multiply(is_invested, max_weights) * scale_constraints
    )

    if is_short:
        constraints.append(
            w * scale_constraints
            >= cp.multiply(is_invested, min_weights) * scale_constraints
        )

    return constraints


def _mip_weight_constraints_threshold_short(
    n_assets: int,
    w: cp.Variable,
    factor: skt.Factor,
    scale_constraints: cp.Constant,
    max_weights: np.ndarray,
    min_weights: np.ndarray,
    threshold_long: np.ndarray,
    threshold_short: np.ndarray,
    cardinality: int | None,
    group_cardinalities: dict[str, int] | None,
    groups: np.ndarray | None,
) -> list[cp.Expression]:
    """
    Create a list of MIP constraints for cardinality and threshold constraints
    when a short threshold is allowed. This requires the creation of two boolean
    variable arrays, one for long positions and one for short positions.
    """
    constraints = []

    is_invested_short_bool = cp.Variable(n_assets, boolean=True)
    is_invested_long_bool = cp.Variable(n_assets, boolean=True)
    is_invested_bool = is_invested_short_bool + is_invested_long_bool

    if cardinality is not None and cardinality < n_assets:
        constraints.append(cp.sum(is_invested_bool) <= cardinality)

    if group_cardinalities is not None:
        a_card, b_card = group_cardinalities_to_matrix(
            groups=groups,
            group_cardinalities=group_cardinalities,
            raise_if_group_missing=False,
        )
        constraints.append(a_card @ is_invested_bool - b_card <= 0)

    M = 1e3
    # Big M method to activate or deactivate constraints
    # In the ratio homogenization procedure, the factor has been calibrated
    # to be around 0.1-10. By using M=1e3, we ensure that M is large enough while
    # not too large for improved MIP convergence.

    if isinstance(factor, cp.Variable):
        is_invested_short_factor = cp.Variable(n_assets, nonneg=True)
        is_invested_long_factor = cp.Variable(n_assets, nonneg=True)
        # We want (w <= cp.multiply(is_invested_short_bool, max_weights) * factor
        # but this is not DCP. So we introduce another variable and set
        # constraint to ensure its value is equal to is_invested_short_bool * factor

        constraints += [
            is_invested_short_factor <= factor,
            is_invested_long_factor <= factor,
            is_invested_short_factor <= M * is_invested_short_bool,
            is_invested_long_factor <= M * is_invested_long_bool,
            is_invested_short_factor >= factor - M * (1 - is_invested_short_bool),
            is_invested_long_factor >= factor - M * (1 - is_invested_long_bool),
        ]
        is_invested_short = is_invested_short_factor
        is_invested_long = is_invested_long_factor
    else:
        is_invested_short = is_invested_short_bool
        is_invested_long = is_invested_long_bool

    constraints += [
        is_invested_bool <= 1.0,
        w * scale_constraints
        <= cp.multiply(is_invested_long, max_weights) * scale_constraints,
        w * scale_constraints
        >= cp.multiply(is_invested_short, min_weights) * scale_constraints,
        # Apply threshold_long if is_invested_long == 1,
        # unrestricted if is_invested_long == 0
        w * scale_constraints
        >= cp.multiply(is_invested_long, threshold_long) * scale_constraints
        - M * (1 - is_invested_long_bool) * scale_constraints,
        # # Apply threshold_short if is_invested_short == 1,
        # # unrestricted if is_invested_short == 0
        w * scale_constraints
        <= cp.multiply(is_invested_short, threshold_short) * scale_constraints
        + M * (1 - is_invested_short_bool) * scale_constraints,
    ]

    return constraints


def _solve(
    w,
    factor,
    expressions,
    problem,
    solver,
    solver_params,
    risk_measure,
    scale_objective,
):
    try:
        # We suppress cvxpy warning as it is redundant with our warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            problem.solve(solver=solver, **solver_params)

        if w.value is None:
            raise cp.SolverError("No solution found")

        weights = w.value / factor.value
        problem_values = {
            name: expression.value / factor.value
            if name != "factor"
            else expression.value
            for name, expression in expressions.items()
        }
        problem_values["objective"] = problem.value / scale_objective.value

        if (
            risk_measure in [RiskMeasure.VARIANCE, RiskMeasure.SEMI_VARIANCE]
            and "risk" in problem_values
        ):
            problem_values["risk"] /= factor.value

        weights = np.array(weights, dtype=float)
        if not problem.status == cp.OPTIMAL:
            warnings.warn(
                "Solution may be inaccurate. Try changing the solver params or the"
                " scale. For more details, set `solver_params=dict(verbose=True)`",
                stacklevel=2,
            )
        return weights, problem_values
    except (cp.SolverError, scl.ArpackNoConvergence):
        params_string = " ".join([f"{p.value:0g}" for p in problem.parameters()])
        if len(params_string) != 0:
            params_string = f" with parameters {params_string}"
        error = (
            f"Solver '{solver}' failed{params_string}. Try another"
            " solver, or solve with solver_params=dict(verbose=True) for more"
            " information"
        )
        raise cp.SolverError(error) from None
        # elif n_optimizations > 1:
        #     warnings.warn(error, stacklevel=2)
        #
        # problem_values = None
        # weights = np.full(w.shape, np.nan, dtype=float)
