"""Risk Budgeting Optimization estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause
# The optimization features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.

import cvxpy as cp
import numpy as np
import numpy.typing as npt

import skfolio.typing as skt
from skfolio.measures import RiskMeasure
from skfolio.optimization.convex._base import ConvexOptimization
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.utils.tools import args_names, check_estimator


class RiskBudgeting(ConvexOptimization):
    r"""Risk Budgeting Optimization estimator.

    The Risk Budgeting estimator solves the below convex problem:

        .. math::   \begin{cases}
                    \begin{aligned}
                    &\min_{w} & & risk_{i}(w) \\
                    &\text{s.t.} & & b^T \cdot log(w) \ge c \\
                    & & & w^T \cdot \mu \ge min\_return \\
                    & & & A \cdot w \ge b \\
                    & & & w \ge 0
                    \end{aligned}
                    \end{cases}

    with :math:`b` the risk budget vector and :math:`c` an auxiliary variable of
    the log barrier.

    And :math:`risk_{i}` a risk measure among:

        * Mean Absolute Deviation
        * First Lower Partial Moment
        * Variance
        * Semi-Variance
        * CVaR (Conditional Value at Risk)
        * EVaR (Entropic Value at Risk)
        * Worst Realization (worst return)
        * CDaR (Conditional Drawdown at Risk)
        * Maximum Drawdown
        * Average Drawdown
        * EDaR (Entropic Drawdown at Risk)
        * Ulcer Index
        * Gini Mean Difference

    Cost and additional constraints can also be added to the optimization problem  (see
    the parameters description).

    Limitations are imposed on some constraints including long only weights to ensure
    convexity.

    The assets expected returns, covariance matrix and returns are estimated from the
    :ref:`prior estimator <prior>`.

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

    risk_budget : dict[str, float] | array-like of shape (n_assets,), optional
        Risk budget allocated to each asset.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset risk budget) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.
        The default (`None`) is to use the identity vector, reducing the risk
        budgeting to a risk-parity (each asset contributing equally to the total risk).

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

    min_return : float | array-like of shape (n_optimization), optional
        Lower bound constraint on the expected return.

    min_acceptable_return : float, optional
        The minimum acceptable return used to distinguish "downside" and "upside"
        returns for the computation of lower partial moments:

            * First Lower Partial Moment
            * Semi-Variance
            * Semi-Deviation

        The default (`None`) is to use the mean.

    cvar_beta : float, default=0.95
        CVaR (Conditional Value at Risk) confidence level.

    evar_beta : float, default=0
        EVaR (Entropic Value at Risk) confidence level.

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

    problem_: cvxpy.Problem
        CVXPY problem used for the optimization. Only when `save_problem` is set to
        `True`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    def __init__(
        self,
        risk_measure: RiskMeasure = RiskMeasure.VARIANCE,
        risk_budget: np.ndarray | None = None,
        prior_estimator: BasePrior | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        transaction_costs: skt.MultiInput = 0.0,
        management_fees: skt.MultiInput = 0.0,
        previous_weights: skt.MultiInput | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        risk_free_rate: float = 0.0,
        min_return: skt.Target | None = None,
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
        super().__init__(
            risk_measure=risk_measure,
            prior_estimator=prior_estimator,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=1,
            transaction_costs=transaction_costs,
            management_fees=management_fees,
            previous_weights=previous_weights,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            risk_free_rate=risk_free_rate,
            min_acceptable_return=min_acceptable_return,
            cvar_beta=cvar_beta,
            evar_beta=evar_beta,
            cdar_beta=cdar_beta,
            edar_beta=edar_beta,
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
        )
        self.min_return = min_return
        self.risk_budget = risk_budget

    def _validation(self) -> None:
        if not isinstance(self.risk_measure, RiskMeasure):
            raise TypeError("risk_measure must be of type `RiskMeasure`")
        if self.min_weights < 0:
            raise ValueError(
                "Risk Budgeting must have non negative `min_weights` constraint"
                " otherwise the problem becomes non-convex."
            )

    def fit(self, X: npt.ArrayLike, y=None) -> "RiskBudgeting":
        """Fit the Risk Budgeting Optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors.
            The default is `None`.


        Returns
        -------
        self : RiskBudgeting
           Fitted estimator.
        """
        self._check_feature_names(X, reset=True)
        # Validate
        self._validation()
        # Used to avoid adding multiple times similar constrains linked to identical
        # risk models
        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        self.prior_estimator_.fit(X, y)
        prior_model = self.prior_estimator_.prior_model_
        n_observations, n_assets = prior_model.returns.shape

        # set solvers params
        if self.solver == "CLARABEL":
            self._set_solver_params(default={"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9})
        else:
            self._set_solver_params(default=None)

        # set scale
        self._set_scale_objective(default=1)
        self._set_scale_constraints(default=1)

        # Risk budget
        risk_budget = self.risk_budget
        if risk_budget is None:
            risk_budget = np.ones(n_assets)
        else:
            risk_budget = self._clean_input(
                self.risk_budget,
                n_assets=n_assets,
                fill_value=1e-10,
                name="risk_budget",
            )
            risk_budget[risk_budget == 0] = 1e-10

        # Variables
        w = cp.Variable(n_assets)
        factor = cp.Variable()
        c = cp.Variable(nonneg=True)

        # Expected returns
        expected_return = (
            self._cvx_expected_return(prior_model=prior_model, w=w)
            - self._cvx_transaction_cost(prior_model=prior_model, w=w, factor=factor)
            - self._cvx_management_fee(prior_model=prior_model, w=w)
        )

        # risk budgeting constraint
        constraints = [
            risk_budget @ cp.log(w) * self._scale_constraints
            >= c * self._scale_constraints
        ]

        # weight constraints
        constraints += self._get_weight_constraints(
            n_assets=n_assets, w=w, factor=factor
        )

        parameters_values = []

        # min_return constraint
        if self.min_return is not None:
            parameter = cp.Parameter(nonneg=False)
            constraints += [
                expected_return * self._scale_constraints
                >= parameter * factor * self._scale_constraints
            ]
            parameters_values.append((parameter, self.min_return))

        # risk and risk constraints
        risk_func = getattr(self, f"_{self.risk_measure.value}_risk")
        args = {}
        for arg_name in args_names(risk_func):
            if arg_name == "prior_model":
                args[arg_name] = prior_model
            elif arg_name == "w":
                args[arg_name] = w
            elif arg_name == "factor":
                if self.risk_measure in [RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT]:
                    args[arg_name] = factor
                else:
                    args[arg_name] = cp.Constant(1)
            else:
                args[arg_name] = getattr(self, arg_name)
        risk, constraints_i = risk_func(**args)
        constraints += constraints_i

        # custom objectives and constraints
        custom_objective = self._get_custom_objective(w=w)
        constraints += self._get_custom_constraints(w=w)

        objective = cp.Minimize(
            risk * self._scale_objective + custom_objective * self._scale_objective
        )

        # problem
        # noinspection PyTypeChecker
        problem = cp.Problem(objective, constraints)

        # results
        self._solve_problem(
            problem=problem,
            w=w,
            factor=factor,
            parameters_values=parameters_values,
            expressions={
                "expected_return": expected_return,
                "risk": risk,
                "factor": factor,
            },
        )

        return self
