"""Mean Risk Optimization estimator."""

import warnings

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause
# The optimization features are derived
# from Riskfolio-Lib, Copyright (c) 2020-2023, Dany Cajas, Licensed under BSD 3 clause.
import cvxpy as cp
import numpy as np
import numpy.typing as npt
import pandas as pd
import sklearn as sk
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio.measures import RiskMeasure
from skfolio.optimization.convex._base import ConvexOptimization, ObjectiveFunction
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.uncertainty_set import BaseCovarianceUncertaintySet, BaseMuUncertaintySet
from skfolio.utils.tools import args_names, check_estimator

# noinspection PyUnresolvedReferences
_NON_ANNUALIZED_RISK_MEASURES = [rm for rm in RiskMeasure if not rm.is_annualized]


class MeanRisk(ConvexOptimization):
    r"""Mean-Risk Optimization estimator.

    The below 4 objective functions can be optimized:

        * Minimize Risk:

        .. math::   \begin{cases}
                    \begin{aligned}
                    &\min_{w} & & risk_{i}(w) \\
                    &\text{s.t.} & & w^T \cdot \mu \ge min\_return \\
                    & & & A \cdot w \ge b \\
                    & & & risk_{j}(w) \le max\_risk_{j} \quad \forall \; j \ne i
                    \end{aligned}
                    \end{cases}

        * Maximize Expected Return:

        .. math::   \begin{cases}
                    \begin{aligned}
                    &\max_{w} & & w^T \cdot \mu \\
                    &\text{s.t.} & & risk_{i}(w) \le max\_risk_{i} \\
                    & & & A \cdot w \ge b \\
                    & & & risk_{j}(w) \le max\_risk_{j} \quad \forall \; j \ne i
                    \end{aligned}
                    \end{cases}

        * Maximize Utility:

        .. math::   \begin{cases}
                    \begin{aligned}
                    &\max_{w} & & w^T \cdot \mu - \lambda \times risk_{i}(w)\\
                    &\text{s.t.} & & risk_{i}(w) \le max\_risk_{i} \\
                    & & & w^T \cdot \mu \ge min\_return \\
                    & & & A \cdot w \ge b \\
                    & & & risk_{j}(w) \le max\_risk_{j} \quad \forall \; j \ne i
                    \end{aligned}
                    \end{cases}

        * Maximize Ratio:

        .. math::   \begin{cases}
                    \begin{aligned}
                    &\max_{w} & & \frac{w^T \cdot \mu - r_{f}}{risk_{i}(w)}\\
                    &\text{s.t.} & & risk_{i}(w) \le max\_risk_{i} \\
                    & & & w^T \cdot \mu \ge min\_return \\
                    & & & A \cdot w \ge b \\
                    & & & risk_{j}(w) \le max\_risk_{j} \quad \forall \; j \ne i
                    \end{aligned}
                    \end{cases}

    With :math:`risk_{i}` a risk measure among:

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

    Cost, regularization, uncertainty set, and additional constraints can also be added
    to the optimization problem (see the parameters description).

    The assets expected returns, covariance matrix and returns are estimated from the
    :ref:`prior estimator <prior>`.

    Parameters
    ----------
    objective_function : ObjectiveFunction, default=ObjectiveFunction.MINIMIZE_RISK
        :class:`~skfolio.optimization.ObjectiveFunction` of the optimization.
        Can be any of:

            * MINIMIZE_RISK
            * MAXIMIZE_RETURN
            * MAXIMIZE_UTILITY
            * MAXIMIZE_RATIO

        The default is `ObjectiveFunction.MINIMIZE_RISK`.

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

    risk_aversion : float, default=1.0
        Risk aversion factor :math:`\lambda` of the utility function. Only used for
        `objective_function=ObjectiveFunction.MAXIMIZE_UTILITY`.
        The default value is `1.0`.

    prior_estimator : BasePrior, optional
        :ref:`Prior estimator <prior>`.
        The prior estimator is used to estimate the :class:`~skfolio.prior.PriorModel`
        containing the estimation of assets expected returns, covariance matrix,
        returns and Cholesky decomposition of the covariance.
        The default (`None`) is to use :class:`~skfolio.prior.EmpiricalPrior`.

    efficient_frontier_size : int, optional
        If provided, it represents the number of Pareto-optimal portfolios along the
        efficient frontier to be computed. This parameter can only be used with
        `objective_function = ObjectiveFunction.MINIMIZE_RISK`.

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

        .. math:: w^T \cdot \hat{\mu} - \kappa_{\mu} \lVert S_{\mu}^\frac{1}{2} \cdot w \rVert_{2}

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
           * "ref1 == ref2"
           * "ref1 >= ref1"

        With "ref1", "ref2" ... the assets names or the groups names provided
        in the parameter `groups`. Assets names can be referenced without the need of
        `groups` if the input `X` of the `fit` method is a DataFrame with these
        assets names in columns.

        For example:

            * "SPX >= 0.10" --> SPX weight must be greater than 10% (note that you can also use `min_weights`)
            * "SX5E + TLT >= 0.2" --> the sum of SX5E and TLT weights must be greater than 20%
            * "US == 0.7" --> the sum of all US weights must be equal to 70%
            * "Equity == 3 * Bond" --> the sum of all Equity weights must be equal to 3 times the sum of all Bond weights.
            * "2*SPX + 3*Europe <= Bond + 0.05" --> mixing assets and group constraints

    groups : dict[str, list[str]] or array-like of shape (n_groups, n_assets), optional
        The assets groups referenced in `linear_constraints`.
        If a dictionary is provided, its (key/value) pair must be the
        (asset name/asset groups) and the input `X` of the `fit` method must be a
        DataFrame with the assets names in columns.

        For example:

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

    max_tracking_error : float, optional
        Upper bound constraint on the tracking error.
        The tracking error is defined as the RMSE (root-mean-square error) of the
        portfolio returns compared to a target returns. If `max_tracking_error` is
        provided, the target returns `y` must be provided in the `fit` method.

    max_turnover : float, optional
        Upper bound constraint of the turnover.
        The turnover is defined as the absolute difference between the portfolio weights
        and the `previous_weights`. Note that another way to control for turnover is by
        using the `transaction_costs` parameter.

    max_mean_absolute_deviation : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Mean Absolute Deviation.

    max_first_lower_partial_moment : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the First Lower Partial Moment.

    max_variance : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Variance.

    max_standard_deviation : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Standard deviation.

    max_semi_variance : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Semi-Variance (Second Lower Partial Moment or
        Downside Variance).

    max_semi_deviation : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Semi-Standard deviation.

    max_worst_realization : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Worst Realization (Worst Return).

    max_cvar : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the CVaR (Conditional Value-at-Risk or Expected
        Shortfall).

    max_evar : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the EVaR (Entropic Value at Risk).

    max_max_drawdown : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Maximum Drawdown.

    max_average_drawdown : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Average Drawdown.

    max_cdar : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the CDaR (Conditional Drawdown at Risk).

    max_edar : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the EDaR (Entropic Drawdown at Risk).

    max_ulcer_index : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Ulcer Index.

    max_gini_mean_difference : float | array-like of shape (n_optimization), optional
        Upper bound constraint on the Gini Mean Difference.

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
        for "CLARABEL", `{"numerics/feastol": 1e-8, "limits/gap": 1e-8}` for SCIP
        and the solver default otherwise.
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

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.
    """

    def __init__(
        self,
        objective_function: ObjectiveFunction = ObjectiveFunction.MINIMIZE_RISK,
        risk_measure: RiskMeasure = RiskMeasure.VARIANCE,
        risk_aversion: float = 1.0,
        efficient_frontier_size: int | None = None,
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
        min_return: skt.Target | None = None,
        max_tracking_error: skt.Target | None = None,
        max_turnover: skt.Target | None = None,
        max_mean_absolute_deviation: skt.Target | None = None,
        max_first_lower_partial_moment: skt.Target | None = None,
        max_variance: skt.Target | None = None,
        max_standard_deviation: skt.Target | None = None,
        max_semi_variance: skt.Target | None = None,
        max_semi_deviation: skt.Target | None = None,
        max_worst_realization: skt.Target | None = None,
        max_cvar: skt.Target | None = None,
        max_evar: skt.Target | None = None,
        max_max_drawdown: skt.Target | None = None,
        max_average_drawdown: skt.Target | None = None,
        max_cdar: skt.Target | None = None,
        max_edar: skt.Target | None = None,
        max_ulcer_index: skt.Target | None = None,
        max_gini_mean_difference: skt.Target | None = None,
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
            mu_uncertainty_set_estimator=mu_uncertainty_set_estimator,
            covariance_uncertainty_set_estimator=covariance_uncertainty_set_estimator,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            min_budget=min_budget,
            max_budget=max_budget,
            max_short=max_short,
            max_long=max_long,
            cardinality=cardinality,
            group_cardinalities=group_cardinalities,
            threshold_long=threshold_long,
            threshold_short=threshold_short,
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
        self.objective_function = objective_function
        self.risk_aversion = risk_aversion
        self.efficient_frontier_size = efficient_frontier_size
        self.min_return = min_return
        self.max_tracking_error = max_tracking_error
        self.max_turnover = max_turnover
        self.max_mean_absolute_deviation = max_mean_absolute_deviation
        self.max_first_lower_partial_moment = max_first_lower_partial_moment
        self.max_variance = max_variance
        self.max_standard_deviation = max_standard_deviation
        self.max_semi_variance = max_semi_variance
        self.max_semi_deviation = max_semi_deviation
        self.max_worst_realization = max_worst_realization
        self.max_cvar = max_cvar
        self.max_evar = max_evar
        self.max_max_drawdown = max_max_drawdown
        self.max_average_drawdown = max_average_drawdown
        self.max_cdar = max_cdar
        self.max_edar = max_edar
        self.max_ulcer_index = max_ulcer_index
        self.max_gini_mean_difference = max_gini_mean_difference

    def _validation(self) -> None:
        """Validate the input parameters."""
        if not isinstance(self.risk_measure, RiskMeasure):
            raise TypeError("risk_measure must be of type `RiskMeasure`")
        if not isinstance(self.objective_function, ObjectiveFunction):
            raise TypeError("objective_function must be of type `ObjectiveFunction`")
        if self.efficient_frontier_size is not None:
            if self.efficient_frontier_size <= 1:
                raise ValueError(
                    "`efficient_frontier_size` must be strictly greater than one"
                )
            if self.objective_function != ObjectiveFunction.MINIMIZE_RISK:
                raise ValueError(
                    "`efficient_frontier_size` must be used only with "
                    "`objective_function = ObjectiveFunction.MINIMIZE_RISK`"
                )

    def get_metadata_routing(self):
        # noinspection PyTypeChecker
        router = (
            super()
            .get_metadata_routing()
            .add(
                mu_uncertainty_set_estimator=self.mu_uncertainty_set_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                covariance_uncertainty_set_estimator=self.covariance_uncertainty_set_estimator,
                method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
            )
        )
        return router

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params
    ) -> "MeanRisk":
        """Fit the Mean-Risk Optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : array-like of shape (n_observations, n_targets), optional
            Price returns of factors or a target benchmark.
            The default is `None`.

        Returns
        -------
        self : MeanRisk
           Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        # `X` is unchanged and only `feature_names_in_` is performed
        _ = skv.validate_data(self, X, skip_check_array=True)

        # Validate
        self._validation()
        # Used to avoid adding multiple times similar constrains linked to identical
        # risk models
        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        self.prior_estimator_.fit(X, y, **routed_params.prior_estimator.fit)
        prior_model = self.prior_estimator_.prior_model_
        n_observations, n_assets = prior_model.returns.shape

        # set solvers params
        match self.solver:
            case "CLARABEL":
                self._set_solver_params(
                    default={"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9}
                )
            case "SCIP":
                self._set_solver_params(
                    default={"numerics/feastol": 1e-8, "limits/gap": 1e-8}
                )
            case _:
                self._set_solver_params(default=None)

        # set scales and check measure
        if self.objective_function == ObjectiveFunction.MAXIMIZE_RATIO:
            if self.overwrite_expected_return is not None:
                if self.risk_measure == RiskMeasure.VARIANCE:
                    warnings.warn(
                        "When selecting 'MAXIMIZE_RATIO' with 'VARIANCE', the "
                        "optimization will return the maximum Sharpe Ratio portfolio. "
                        "This is because the mean/variance ratio is not a "
                        "1-homogeneous function, unlike the mean/std. To suppress this"
                        "warning, replace 'VARIANCE' by 'STANDARD_DEVIATION'",
                        stacklevel=2,
                    )

                elif self.risk_measure == RiskMeasure.SEMI_VARIANCE:
                    warnings.warn(
                        "When selecting 'MAXIMIZE_RATIO' with 'SEMI_VARIANCE', the "
                        "optimization will return the maximum Sortino Ratio portfolio. "
                        "This is because the mean/semi-variance ratio is not a "
                        "1-homogeneous function, unlike the mean/semi-std ratio. To "
                        "suppress this warning, replace 'SEMI_VARIANCE' by "
                        "'SEMI_DEVIATION'",
                        stacklevel=2,
                    )

            self._set_scale_objective(default=1)
            self._set_scale_constraints(default=1)
        else:
            match self.risk_measure:
                case (
                    RiskMeasure.MEAN_ABSOLUTE_DEVIATION
                    | RiskMeasure.FIRST_LOWER_PARTIAL_MOMENT
                    | RiskMeasure.CVAR
                    | RiskMeasure.WORST_REALIZATION
                    | RiskMeasure.AVERAGE_DRAWDOWN
                    | RiskMeasure.MAX_DRAWDOWN
                    | RiskMeasure.CDAR
                    | RiskMeasure.ULCER_INDEX
                ):
                    self._set_scale_objective(default=1e-1)
                    self._set_scale_constraints(default=1e2)

                case RiskMeasure.EVAR:
                    self._set_scale_objective(default=1)
                    self._set_scale_constraints(default=1e-2)

                case RiskMeasure.EDAR:
                    self._set_scale_objective(default=1)
                    self._set_scale_constraints(default=1e2)

                case _:
                    self._set_scale_objective(default=1)
                    self._set_scale_constraints(default=1)

        # Init weight variable and constraints
        w = cp.Variable(n_assets)
        constraints = []

        if self.objective_function == ObjectiveFunction.MAXIMIZE_RATIO:
            factor = cp.Variable()
        else:
            factor = cp.Constant(1)

        # Mu uncertainty set
        if self.mu_uncertainty_set_estimator is None:
            mu_uncertainty_set = cp.Constant(0)
        else:
            # noinspection PyTypeChecker
            self.mu_uncertainty_set_estimator_ = sk.clone(
                self.mu_uncertainty_set_estimator
            )
            self.mu_uncertainty_set_estimator_.fit(
                X, y, **routed_params.mu_uncertainty_set_estimator.fit
            )
            mu_uncertainty_set = self._cvx_mu_uncertainty_set(
                mu_uncertainty_set=self.mu_uncertainty_set_estimator_.uncertainty_set_,
                w=w,
            )

        # Expected returns
        expected_return = (
            self._cvx_expected_return(prior_model=prior_model, w=w)
            - self._cvx_transaction_cost(prior_model=prior_model, w=w, factor=factor)
            - self._cvx_management_fee(prior_model=prior_model, w=w)
            - mu_uncertainty_set
        )

        # Regularization
        regularization = self._cvx_regularization(w=w)

        # Tracking error
        if self.max_tracking_error is not None:
            if y is None:
                raise ValueError(
                    "If `max_tracking_error` is provided, `y` must also be provided"
                )
            if isinstance(y, pd.DataFrame):
                if y.shape[1] > 1:
                    raise ValueError(
                        "If `max_tracking_error` is provided, `y` must be a"
                        " 1d-array, a single-column DataFrame or a Series"
                    )
                y = y[y.columns[0]]
            _, y = skv.validate_data(self, X, y)
            tracking_error = self._tracking_error(
                prior_model=prior_model, w=w, y=y, factor=factor
            )
            constraints += [
                tracking_error * self._scale_constraints
                <= self.max_tracking_error * factor * self._scale_constraints
            ]

        # Turnover
        if self.max_turnover is not None:
            turnover = self._turnover(n_assets=n_assets, w=w, factor=factor)
            constraints += [
                turnover * self._scale_constraints
                <= self.max_turnover * factor * self._scale_constraints
            ]

        # weight constraints
        constraints += self._get_weight_constraints(
            n_assets=n_assets, w=w, factor=factor
        )

        parameters_values = []

        # Efficient frontier
        if self.efficient_frontier_size is not None:
            # We find the lower and upper bounds of the expected returns.
            # noinspection PyTypeChecker
            model: MeanRisk = sk.clone(self)
            # noinspection PyTypeChecker
            model.set_params(
                objective_function=ObjectiveFunction.MINIMIZE_RISK,
                efficient_frontier_size=None,
                portfolio_params=dict(annualized_factor=1),
            )
            model.fit(X, y, **fit_params)
            min_return = model.problem_values_["expected_return"]
            # noinspection PyTypeChecker
            model.set_params(objective_function=ObjectiveFunction.MAXIMIZE_RETURN)
            model.fit(X, y, **fit_params)
            max_return = model.problem_values_["expected_return"]
            if max_return <= 0:
                raise ValueError(
                    "Unable to compute the Efficient Frontier with only negative"
                    " expected returns"
                )
            targets = np.linspace(
                max(min_return, 1e-10) * 1.01,
                max_return,
                num=self.efficient_frontier_size,
            )
            parameter = cp.Parameter(nonneg=False)
            constraints += [expected_return >= parameter * factor]
            parameters_values.append((parameter, targets))

        # min_return constraint
        if self.min_return is not None:
            parameter = cp.Parameter(nonneg=False)
            constraints += [
                expected_return * self._scale_constraints
                >= parameter * factor * self._scale_constraints
            ]
            parameters_values.append((parameter, self.min_return))

        # risk and risk constraints
        risk = None
        for r_m in _NON_ANNUALIZED_RISK_MEASURES:
            risk_limit = getattr(self, f"max_{r_m.value}")

            if self.risk_measure == r_m or risk_limit is not None:
                # Add covariance uncertainty set if provided
                if (
                    r_m == RiskMeasure.VARIANCE
                    and self.covariance_uncertainty_set_estimator is not None
                ):
                    risk_func = self._worst_case_variance_risk
                else:
                    risk_func = getattr(self, f"_{r_m.value}_risk")

                args = {}
                for arg_name in args_names(risk_func):
                    if arg_name == "prior_model":
                        args[arg_name] = prior_model
                    elif arg_name == "w":
                        args[arg_name] = w
                    elif arg_name == "factor":
                        args[arg_name] = factor
                    elif arg_name == "covariance_uncertainty_set":
                        # noinspection PyTypeChecker
                        self.covariance_uncertainty_set_estimator_ = sk.clone(
                            self.covariance_uncertainty_set_estimator
                        )
                        self.covariance_uncertainty_set_estimator_.fit(
                            X,
                            y,
                            **routed_params.covariance_uncertainty_set_estimator.fit,
                        )
                        args[arg_name] = (
                            self.covariance_uncertainty_set_estimator_.uncertainty_set_
                        )
                    else:
                        args[arg_name] = getattr(self, arg_name)

                risk_i, constraints_i = risk_func(**args)
                constraints += constraints_i
                if risk_limit is not None:
                    parameter = cp.Parameter(nonneg=True)
                    constraints += [
                        risk_i * self._scale_constraints
                        <= parameter * factor * self._scale_constraints
                    ]
                    parameters_values.append((parameter, risk_limit))
                if self.risk_measure == r_m:
                    risk = risk_i

        # custom objectives and constraints
        custom_objective = self._get_custom_objective(w=w)
        constraints += self._get_custom_constraints(w=w)

        match self.objective_function:
            case ObjectiveFunction.MAXIMIZE_RETURN:
                objective = cp.Maximize(
                    expected_return * self._scale_objective
                    - regularization * self._scale_objective
                    + custom_objective * self._scale_objective
                )
            case ObjectiveFunction.MINIMIZE_RISK:
                objective = cp.Minimize(
                    risk * self._scale_objective
                    + regularization * self._scale_objective
                    + custom_objective * self._scale_objective
                )
            case ObjectiveFunction.MAXIMIZE_UTILITY:
                objective = cp.Maximize(
                    expected_return * self._scale_objective
                    - self.risk_aversion * risk * self._scale_objective
                    - regularization * self._scale_objective
                    + custom_objective * self._scale_objective
                )
            case ObjectiveFunction.MAXIMIZE_RATIO:
                # Capture common obvious mistake before solver failure to help user
                if np.isscalar(self.min_weights) and self.min_weights >= 0:
                    if np.max(prior_model.mu) - self.risk_free_rate <= 0:
                        raise ValueError(
                            "Cannot optimize for Maximum Ratio with your current "
                            "constraints and input. This is because your assets' "
                            "expected returns are all under-performing your risk-free "
                            f"rate {self.risk_free_rate:.2%}."
                        )
                homogenization_factor = _optimal_homogenization_factor(
                    mu=prior_model.mu
                )

                if expected_return.is_affine():
                    # Charnes-Cooper's variable transformation for Fractional
                    # Programming problem Max(f1/f2) with f2 linear and with
                    # 1-homogeneous function (homogeneous technique)
                    constraints += [
                        expected_return * self._scale_constraints
                        - cp.Constant(self.risk_free_rate)
                        * factor
                        * self._scale_constraints
                        == cp.Constant(homogenization_factor) * self._scale_constraints
                    ]
                else:
                    # Schaible's generalization of Charnes-Cooper's variable
                    # transformation for Fractional Programming problem :Max(f1/f2)
                    # with f1 concave instead of linear and with 1-homogeneous function.
                    # (homogeneous technique)
                    # Schaible,"Parameter-free Convex Equivalent and Dual Programs of
                    # Fractional Programming Problems".
                    # The condition to work is f1 >= 0, so we need to raise an user
                    # warning when it's not the case.

                    constraints += [
                        expected_return * self._scale_constraints
                        - cp.Constant(self.risk_free_rate)
                        * factor
                        * self._scale_constraints
                        >= cp.Constant(homogenization_factor) * self._scale_constraints
                    ]
                objective = cp.Minimize(
                    risk * self._scale_objective
                    + regularization * self._scale_objective
                    + custom_objective * self._scale_objective
                )
            case _:
                raise ValueError(
                    f"objective_function {self.objective_function} is not valid"
                )

        # problem
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
                "mu_uncertainty_set": mu_uncertainty_set,
                "regularization": regularization,
                "factor": factor,
            },
        )

        return self


def _optimal_homogenization_factor(mu: np.ndarray) -> float:
    """
    Compute the optimal homogenization factor for ratio optimization based on expected
    returns.

    While a default value of 1 is commonly used in textbooks for simplicity,
    fine-tuning this factor based on the underlying data can enhance convergence.
    Additionally, using a data-driven approach to determine this factor can improve the
    robustness of certain constraints, such as the calibration of big M methods.

    Parameters
    ----------
    mu : ndarray of shape (n_assets,)
        Vector of expected returns.

    Returns
    -------
    value : float
        Homogenization factor.
    """
    return min(1e3, max(1e-3, np.mean(np.abs(mu))))
