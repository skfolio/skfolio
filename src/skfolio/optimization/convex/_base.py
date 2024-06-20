"""Base Convex Optimization estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# The optimization features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

import warnings
from abc import ABC, abstractmethod
from enum import auto
from typing import Any

import cvxpy as cp
import cvxpy.constraints.constraint as cpc
import numpy as np
import numpy.typing as npt
import scipy as sc
import scipy.sparse.linalg as scl

import skfolio.typing as skt
from skfolio.measures import RiskMeasure, owa_gmd_weights
from skfolio.optimization._base import BaseOptimization
from skfolio.prior import BasePrior, PriorModel
from skfolio.uncertainty_set import (
    BaseCovarianceUncertaintySet,
    BaseMuUncertaintySet,
    UncertaintySet,
)
from skfolio.utils.equations import equations_to_matrix
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
        The prior estimator is used to estimate the :class:`~skfolio.prior.PriorModel`
        containing the estimation of assets expected returns, covariance matrix,
        returns and Cholesky decomposition of the covariance.
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    min_weights : float | dict[str, float] | array-like of shape (n_assets, ) | None, default=0.0
        Minimum assets weights (weights lower bounds).
        If a float is provided, it is applied to each asset.
        `None` is equivalent to `-np.Inf` (no lower bound).
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset minium weight) and the input `X` of the `fit` method must
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

        Examples:

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

    transaction_costs : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Transaction costs of the assets. It is used to add linear transaction costs to
        the optimization problem:

        .. math:: total\_cost = \sum_{i=1}^{N} c_{i} \times |w_{i} - w\_prev_{i}|

        with :math:`c_{i}` the transaction cost of asset i, :math:`w_{i}` its weight
        and :math:`w\_prev_{i}` its previous weight (defined in `previous_weights`).
        The float :math:`total\_cost` is impacting the portfolio expected return in the optimization:

        .. math:: expected\_return = \mu^{T} \cdot w - total\_cost

        with :math:`\mu` the vector af assets' expected returns and :math:`w` the
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
            (See :ref:`sphx_glr_auto_examples_1_mean_risk_plot_6_transaction_costs.py`)

    management_fees : float | dict[str, float] | array-like of shape (n_assets, ), default=0.0
        Management fees of the assets. It is used to add linear management fees to the
        optimization problem:

        .. math:: total\_fee = \sum_{i=1}^{N} f_{i} \times w_{i}

        with :math:`f_{i}` the management fee of asset i and :math:`w_{i}` its weight.
        The float :math:`total\_fee` is impacting the portfolio expected return in the optimization:

        .. math:: expected\_return = \mu^{T} \cdot w - total\_fee

        with :math:`\mu` the vector af assets expected returns and :math:`w` the vector
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

           * "2.5 * ref1 + 0.10 * ref2 + 0.0013 <= 2.5 * ref3"
           * "ref1 >= 2.9 * ref2"
           * "ref1 <= ref2"
           * "ref1 >= ref1"

        With "ref1", "ref2" ... the assets names or the groups names provided
        in the parameter `groups`. Assets names can be referenced without the need of
        `groups` if the input `X` of the `fit` method is a DataFrame with these
        assets names in columns.

        Examples:

            * "SPX >= 0.10" --> SPX weight must be greater than 10% (note that you can also use `min_weights`)
            * "SX5E + TLT >= 0.2" --> the sum of SX5E and TLT weights must be greater than 20%
            * "US >= 0.7" --> the sum of all US weights must be greater than 70%
            * "Equity <= 3 * Bond" --> the sum of all Equity weights must be less or equal to 3 times the sum of all Bond weights.
            * "2*SPX + 3*Europe <= Bond + 0.05" --> mixing assets and group constraints

    groups : dict[str, list[str]] or array-like of shape (n_groups, n_assets), optional
        The assets groups referenced in `linear_constraints`.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset groups) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.

        Examples:

            * groups = {"SX5E": ["Equity", "Europe"], "SPX": ["Equity", "US"], "TLT": ["Bond", "US"]}
            * groups = [["Equity", "Equity", "Bond"], ["Europe", "US", "US"]]

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
        CVPXY expression or a list of CVPXY expressions.

    overwrite_expected_return : Callable[[cp.Variable], cp.Expression], optional
        Overwrite the expected return :math:`\mu \cdot w` with a custom expression.
        It is a function that must take as argument the weights `w` and returns a
        CVPXY expression.

    solver : str, default="CLARABEL"
        The solver to use. The default is "CLARABEL" which is written in Rust and has
        better numerical stability and performance than ECOS and SCS. Cvxpy will replace
        its default solver "ECOS" by "CLARABEL" in future releases.
        For more details about available solvers, check the CVXPY documentation:
        https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver

    solver_params : dict, optional
        Solver parameters. For example, `solver_params=dict(verbose=True)`.
        The default (`None`) is use `{"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9}`
        for the solver "CLARABEL" and the CVXPY default otherwise.
        For more details about solver arguments, check the CVXPY documentation:
        https://www.cvxpy.org/tutorial/advanced/index.html#setting-solver-options

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

    raise_on_failure : bool, default=True
        If this is set to True, an error is raised when the optimization fail otherwise
        it passes with a warning.

    portfolio_params :  dict, optional
        Portfolio parameters passed to the portfolio evaluated by the `predict` and
        `score` methods. If not provided, the `name`, `transaction_costs`,
        `management_fees`, `previous_weights` and `risk_free_rate` are copied from the
        optimization model and passed to the portfolio.

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
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
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
        raise_on_failure: bool = True,
        add_objective: skt.ExpressionFunction | None = None,
        add_constraints: skt.ExpressionFunction | None = None,
        overwrite_expected_return: skt.ExpressionFunction | None = None,
        portfolio_params: dict | None = None,
    ):
        super().__init__(portfolio_params=portfolio_params)
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
        self.min_acceptable_return = min_acceptable_return
        self.transaction_costs = transaction_costs
        self.management_fees = management_fees
        self.previous_weights = previous_weights
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
        self.raise_on_failure = raise_on_failure
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

    def _clean_input(
        self,
        value: float | dict | npt.ArrayLike | None,
        n_assets: int,
        fill_value: Any,
        name: str,
    ) -> float | np.ndarray:
        """Convert input to cleaned float or ndarray.

        Parameters
        ----------
        value : float, dict, array-like or None.
            Input value to clean.

        n_assets : int
            Number of assets. Used to verify the shape of the converted array.

        fill_value : Any
            When `items` is a dictionary, elements that are not in `asset_names` are
            filled with `fill_value` in the converted array.

        name : str
            Name used for error messages.

        Returns
        -------
        value :  float or ndarray of shape (n_assets,)
            The cleaned float or 1D array.
        """
        if value is None:
            return fill_value
        if np.isscalar(value):
            return float(value)
        return input_to_array(
            items=value,
            n_assets=n_assets,
            fill_value=fill_value,
            dim=1,
            assets_names=(
                self.feature_names_in_ if hasattr(self, "feature_names_in_") else None
            ),
            name=name,
        )

    def _clear_models_cache(self):
        """CLear the cache of CVX models"""
        self._cvx_cache = {}

    def _get_weight_constraints(
        self, n_assets: int, w: cp.Variable, factor: skt.Factor
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
        constrains : list[cvxpy Constrains]
            The list of weights constraints.
        """
        constraints = []

        if self.min_weights is not None:
            min_weights = self._clean_input(
                self.min_weights,
                n_assets=n_assets,
                fill_value=0,
                name="min_weights",
            )
            constraints.append(
                w * self._scale_constraints
                >= min_weights * factor * self._scale_constraints
            )

        if self.max_weights is not None:
            max_weights = self._clean_input(
                self.max_weights,
                n_assets=n_assets,
                fill_value=1,
                name="max_weights",
            )
            constraints.append(
                w * self._scale_constraints
                <= max_weights * factor * self._scale_constraints
            )

        if self.max_long is not None:
            max_long = float(self.max_long)
            if max_long <= 0:
                raise ValueError("`max_long` must be strictly positif")
            constraints.append(
                cp.sum(cp.pos(w)) * self._scale_constraints
                <= max_long * factor * self._scale_constraints
            )

        if self.max_short is not None:
            max_short = float(self.max_short)
            if max_short <= 0:
                raise ValueError("`max_short` must be strictly positif")
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

        if self.linear_constraints is not None:
            if self.groups is None:
                if not hasattr(self, "feature_names_in_"):
                    raise ValueError(
                        "If `linear_constraints` is provided you must provide either"
                        " `groups` or `X` as a DataFrame with asset names in columns"
                    )
                groups = np.asarray([self.feature_names_in_])
            else:
                groups = input_to_array(
                    items=self.groups,
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
            a, b = equations_to_matrix(
                groups=groups,
                equations=self.linear_constraints,
                raise_if_group_missing=False,
            )
            if np.any(a != 0):
                constraints.append(
                    a @ w * self._scale_constraints
                    - b * factor * self._scale_constraints
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
        """Returns the CVXPY expression evaluated by calling the `add_objective`
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
        """Returns the list of CVXPY expressions evaluated by calling the
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
        self, prior_model: PriorModel, w: cp.Variable
    ) -> cp.Expression:
        """Expected Return expression"""
        if self.overwrite_expected_return is None:
            expected_return = prior_model.mu @ w
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

        all_weights = []
        all_problem_values = []
        optimal = True
        for i in range(n_optimizations):
            for parameter, values in parameters_values:
                parameter.value = values[i]

            try:
                # We suppress cvxpy warning as it is redundant with our warning
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    problem.solve(solver=self.solver, **self._solver_params)

                if w.value is None:
                    raise cp.SolverError("No solution found")

                weights = w.value / factor.value
                problem_values = {
                    name: expression.value / factor.value
                    for name, expression in expressions.items()
                }
                problem_values["objective"] = (
                    problem.value / self._scale_objective.value
                )

                if (
                    self.risk_measure
                    in [RiskMeasure.VARIANCE, RiskMeasure.SEMI_VARIANCE]
                    and "risk" in problem_values
                ):
                    problem_values["risk"] /= factor.value

                all_problem_values.append(problem_values)
                all_weights.append(np.array(weights, dtype=float))

                if problem.status != cp.OPTIMAL:
                    optimal = False
            except (cp.SolverError, scl.ArpackNoConvergence):
                params_string = " ".join(
                    [f"{p.value:0g}" for p in problem.parameters()]
                )
                if len(params_string) != 0:
                    params_string = f" with parameters {params_string}"
                msg = (
                    f"Solver '{self.solver}' failed for {params_string}. Try another"
                    " solver, or solve with solver_params=dict(verbose=True) for more"
                    " information"
                )
                if self.raise_on_failure:
                    raise cp.SolverError(msg) from None
                else:
                    warnings.warn(msg, stacklevel=2)

        if not optimal:
            warnings.warn(
                "Solution may be inaccurate. Try changing the solver params or the"
                " scale. For more details, set `solver_params=dict(verbose=True)`",
                stacklevel=2,
            )

        if n_optimizations == 1:
            self.weights_ = all_weights[0]
            self.problem_values_ = all_problem_values[0]
        else:
            self.weights_ = np.array(all_weights, dtype=float)
            self.problem_values_ = all_problem_values

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
        self, prior_model: PriorModel, w: cp.Variable, factor: skt.Factor
    ) -> cp.Expression:
        """Transaction cost expression.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_assets = prior_model.returns.shape[1]

        transaction_costs = self._clean_input(
            self.transaction_costs,
            n_assets=n_assets,
            fill_value=0,
            name="transaction_costs",
        )
        if np.all(transaction_costs == 0):
            return cp.Constant(0)

        previous_weights = self._clean_input(
            self.previous_weights,
            n_assets=n_assets,
            fill_value=0,
            name="previous_weights",
        )
        if np.isscalar(previous_weights):
            previous_weights *= np.ones(n_assets)

        if np.isscalar(transaction_costs):
            return transaction_costs * cp.norm(previous_weights * factor - w, 1)
        return cp.norm(
            cp.multiply(transaction_costs, (previous_weights * factor - w)),
            1,
        )

    @cache_method("_cvx_cache")
    def _cvx_management_fee(
        self, prior_model: PriorModel, w: cp.Variable
    ) -> cp.Expression:
        """Management fee expression.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : cvxpy Expression
           The CVXPY Expression of management fee .
        """
        n_assets = prior_model.returns.shape[1]

        management_fees = self._clean_input(
            self.management_fees,
            n_assets=n_assets,
            fill_value=0,
            name="management_fees",
        )
        if np.all(management_fees == 0):
            return cp.Constant(0)

        if np.isscalar(management_fees):
            management_fees *= np.ones(n_assets)
        return management_fees @ w

    @cache_method("_cvx_cache")
    def _cvx_returns(self, prior_model: PriorModel, w: cp.Variable) -> cp.Expression:
        """Expression of the portfolio returns series.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distributions.

        w : cvxpy Variable
            The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : cvxpy Expression
            The CVXPY Expression the portfolio returns series.
        """
        returns = prior_model.returns @ w
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
            name="previous_weights",
        )
        if np.isscalar(previous_weights):
            previous_weights *= np.ones(n_assets)
        turnover = cp.abs(w - previous_weights * factor)
        return turnover

    @cache_method("_cvx_cache")
    def _cvx_min_acceptable_return(
        self,
        prior_model: PriorModel,
        w: cp.Variable,
        min_acceptable_return: skt.Target = None,
    ) -> cp.Expression:
        """Expression of the portfolio Minimum Acceptable Returns.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distributions..

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
            min_acceptable_return = prior_model.mu
        if not np.isscalar(min_acceptable_return) and min_acceptable_return.shape != (
            len(min_acceptable_return),
            1,
        ):
            min_acceptable_return = min_acceptable_return[np.newaxis, :]
        mar = (prior_model.returns - min_acceptable_return) @ w
        return mar

    @cache_method("_cvx_cache")
    def __cvx_drawdown(
        self, prior_model: PriorModel, w: cp.Variable, factor: skt.Factor
    ) -> tuple[cp.Variable, list[cp.Expression]]:
        """Expression of the portfolio drawdown.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        ptf_returns = self._cvx_returns(prior_model=prior_model, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            prior_model=prior_model, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(prior_model=prior_model, w=w)
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
        self, prior_model: PriorModel, w: cp.Variable, factor: skt.Factor
    ) -> tuple[cp.Variable, list[cp.Expression]]:
        """Expression of the portfolio drawdown.
        Wrapper around __cvx_drawdown to avoid re-adding the constraints when they
        have already been included in the problem.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distributions.

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
            v, _ = self.__cvx_drawdown(prior_model=prior_model, w=w, factor=factor)
            return v, []
        return self.__cvx_drawdown(prior_model=prior_model, w=w, factor=factor)

    def _tracking_error(
        self, prior_model: PriorModel, w: cp.Variable, y: np.ndarray, factor: skt.Factor
    ) -> cp.Expression:
        """Expression of the portfolio tracking error.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        ptf_returns = self._cvx_returns(prior_model=prior_model, w=w)
        tracking_error = cp.norm(ptf_returns - y * factor, "fro") / cp.sqrt(
            n_observations - 1
        )
        return tracking_error

    # Risk Measures risk models
    # They need to be named f'_{risk_measure}_risk' as they are loaded dynamically in
    # mean_risk_optimization()
    def _mean_absolute_deviation_risk(
        self, prior_model: PriorModel, w: cp.Variable, min_acceptable_return: skt.Target
    ) -> skt.RiskResult:
        """Expression and Constraints of the Mean Absolute Deviation risk measure.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        ptf_min_acceptable_return = self._cvx_min_acceptable_return(
            prior_model=prior_model, w=w, min_acceptable_return=min_acceptable_return
        )
        v = cp.Variable(n_observations, nonneg=True)
        risk = 2 * cp.sum(v) / n_observations
        constraints = [
            ptf_min_acceptable_return * self._scale_constraints
            >= -v * self._scale_constraints
        ]
        return risk, constraints

    def _first_lower_partial_moment_risk(
        self,
        prior_model: PriorModel,
        w: cp.Variable,
        min_acceptable_return: skt.Target,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the First Lower Partial Moment risk measure.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        ptf_min_acceptable_return = self._cvx_min_acceptable_return(
            prior_model=prior_model, w=w, min_acceptable_return=min_acceptable_return
        )
        v = cp.Variable(n_observations, nonneg=True)
        risk = cp.sum(v) / n_observations
        constraints = [
            self.risk_free_rate * factor * self._scale_constraints
            - ptf_min_acceptable_return * self._scale_constraints
            <= v * self._scale_constraints
        ]
        return risk, constraints

    def _standard_deviation_risk(
        self, prior_model: PriorModel, w: cp.Variable
    ) -> skt.RiskResult:
        """Expression and Constraints of the Standard Deviation risk measure.

        Parameters
        ----------
        prior_model : PriorModel
            The prior model of the assets distributions.

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
        if prior_model.cholesky is not None:
            z = prior_model.cholesky
        else:
            z = np.linalg.cholesky(prior_model.covariance)
        risk = v
        constraints = [
            cp.SOC(v * self._scale_constraints, z.T @ w * self._scale_constraints)
        ]
        return risk, constraints

    def _variance_risk(self, prior_model: PriorModel, w: cp.Variable) -> skt.RiskResult:
        """Expression and Constraints of the Variance risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

        w : cvxpy Variable
           The CVXPY Variable representing assets weights.

        Returns
        -------
        expression : tuple[cvxpy Expression , list[cvxpy Expression]]
           CVXPY Expression and Constraints the Variance risk measure.
        """
        risk, constraints = self._standard_deviation_risk(prior_model=prior_model, w=w)
        risk = cp.square(risk)
        return risk, constraints

    def _worst_case_variance_risk(
        self,
        prior_model: PriorModel,
        covariance_uncertainty_set: UncertaintySet,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Worst Case Variance.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_assets = prior_model.returns.shape[1]
        x = cp.Variable((n_assets, n_assets), symmetric=True)
        y = cp.Variable((n_assets, n_assets), symmetric=True)
        w_reshaped = cp.reshape(w, (n_assets, 1))
        factor_reshaped = cp.reshape(factor, (1, 1))
        z1 = cp.vstack([x, w_reshaped.T])
        z2 = cp.vstack([w_reshaped, factor_reshaped])

        risk = covariance_uncertainty_set.k * cp.pnorm(
            sc.linalg.sqrtm(covariance_uncertainty_set.sigma) @ (cp.vec(x) + cp.vec(y)),
            2,
        ) + cp.trace(prior_model.covariance @ (x + y))
        # semi-definite positive constraints
        # noinspection PyTypeChecker
        constraints = [
            cp.hstack([z1, z2]) * self._scale_constraints >> 0,
            y * self._scale_constraints >> 0,
        ]
        return risk, constraints

    def _semi_variance_risk(
        self,
        prior_model: PriorModel,
        w: cp.Variable,
        min_acceptable_return: skt.Target = None,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Semi Variance risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        ptf_min_acceptable_return = self._cvx_min_acceptable_return(
            prior_model=prior_model, w=w, min_acceptable_return=min_acceptable_return
        )
        v = cp.Variable(n_observations, nonneg=True)
        risk = cp.sum_squares(v) / (n_observations - 1)
        constraints = [
            ptf_min_acceptable_return * self._scale_constraints
            >= -v * self._scale_constraints
        ]
        return risk, constraints

    def _semi_deviation_risk(
        self,
        prior_model: PriorModel,
        w: cp.Variable,
        min_acceptable_return: skt.Target = None,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Semi Standard Deviation risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        ptf_min_acceptable_return = self._cvx_min_acceptable_return(
            prior_model=prior_model, w=w, min_acceptable_return=min_acceptable_return
        )
        v = cp.Variable(n_observations, nonneg=True)
        risk = cp.norm(v, 2) / np.sqrt(n_observations - 1)
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
        self, prior_model: PriorModel, w: cp.Variable, factor: skt.Factor
    ) -> skt.RiskResult:
        """Expression and Constraints of the Worst Realization risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        ptf_returns = self._cvx_returns(prior_model=prior_model, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            prior_model=prior_model, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(prior_model=prior_model, w=w)
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
        prior_model: PriorModel,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the CVaR risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        ptf_returns = self._cvx_returns(prior_model=prior_model, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            prior_model=prior_model, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(prior_model=prior_model, w=w)
        alpha = cp.Variable()
        v = cp.Variable(n_observations, nonneg=True)
        risk = alpha + 1.0 / (n_observations * (1 - self.cvar_beta)) * cp.sum(v)
        # noinspection PyTypeChecker
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
        prior_model: PriorModel,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the EVaR risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        ptf_returns = self._cvx_returns(prior_model=prior_model, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            prior_model=prior_model, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(prior_model=prior_model, w=w)
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
        self, prior_model: PriorModel, w: cp.Variable, factor: skt.Factor
    ) -> skt.RiskResult:
        """Expression and Constraints of the EVaR risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        v, constraints = self._cvx_drawdown(prior_model=prior_model, w=w, factor=factor)
        u = cp.Variable()
        risk = u
        constraints += [u * self._scale_constraints >= v[1:] * self._scale_constraints]
        return risk, constraints

    def _average_drawdown_risk(
        self, prior_model: PriorModel, w: cp.Variable, factor: skt.Factor
    ) -> skt.RiskResult:
        """Expression and Constraints of the Average Drawdown risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        v, constraints = self._cvx_drawdown(prior_model=prior_model, w=w, factor=factor)
        risk = cp.sum(v[1:]) / n_observations
        return risk, constraints

    def _cdar_risk(
        self,
        prior_model: PriorModel,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the CDaR risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        v, constraints = self._cvx_drawdown(prior_model=prior_model, w=w, factor=factor)
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
        prior_model: PriorModel,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the EDaR risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        n_observations = prior_model.returns.shape[0]
        v, constraints = self._cvx_drawdown(prior_model=prior_model, w=w, factor=factor)
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
        prior_model: PriorModel,
        w: cp.Variable,
        factor: skt.Factor,
    ) -> skt.RiskResult:
        """Expression and Constraints of the Ulcer Index risk measure.

        Parameters
        ----------
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        v, constraints = self._cvx_drawdown(prior_model=prior_model, w=w, factor=factor)
        n_observations = prior_model.returns.shape[0]
        risk = cp.norm(v[1:], 2) / (np.sqrt(n_observations))
        return risk, constraints

    def _gini_mean_difference_risk(
        self, prior_model: PriorModel, w: cp.Variable, factor: skt.Factor
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
        prior_model : PriorModel
           The prior model of the assets distributions.

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
        ptf_returns = self._cvx_returns(prior_model=prior_model, w=w)
        ptf_transaction_cost = self._cvx_transaction_cost(
            prior_model=prior_model, w=w, factor=factor
        )
        ptf_management_fee = self._cvx_management_fee(prior_model=prior_model, w=w)
        observation_nb = prior_model.returns.shape[0]
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
            == cp.reshape(z, (observation_nb,)) * self._scale_constraints,
            z @ gmd_w.T <= ones @ x.T + y @ ones.T,
        ]
        return risk, constraints

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None):
        pass
