"""Distributionally Robust CVaR Optimization estimator."""

# Copyright (c) 2023
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# License: BSD 3 clause

import cvxpy as cp
import numpy as np
import numpy.typing as npt

import skfolio.typing as skt
from skfolio.measures import RiskMeasure
from skfolio.optimization.convex._base import ConvexOptimization
from skfolio.prior import BasePrior, EmpiricalPrior
from skfolio.utils.tools import check_estimator


class DistributionallyRobustCVaR(ConvexOptimization):
    r"""Distributionally Robust CVaR.

    The Distributionally Robust CVaR model constructs a Wasserstein ball in the space of
    multivariate and non-discrete probability distributions centered at the uniform
    distribution on the training samples and finds the allocation that minimizes the
    CVaR of the worst-case distribution within this Wasserstein ball.
    Esfahani and Kuhn [1]_ proved that for piecewise linear objective functions,
    which is the case of CVaR [2]_, the distributionally robust optimization problem
    over a Wasserstein ball can be reformulated as finite convex programs.

    Only piecewise linear functions are supported, which means that transaction costs
    and regularization are not permitted.

    A solver like `Mosek` that can handle a high number of constraints is preferred.

    Parameters
    ----------
    cvar_beta : float, default=0.95
        CVaR (Conditional Value at Risk) confidence level.

    risk_aversion : float, default=1.0
        Risk aversion factor of the utility function: return - risk_aversion * cvar.

    wasserstein_ball_radius: float, default=0.02
        Radius of the Wasserstein ball.

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

    max_short : float, optional
        Maximum short position. The short position is defined as the sum of negative
        weights (in absolute term).
        The default (`None`) means no maximum short position.

    max_long : float, optional
        Maximum long position. The long position is defined as the sum of positive
        weights.
        The default (`None`) means no maximum long position.

    max_budget :  float, optional
        Maximum budget. It is the upper bound of the sum of long and short positions
        (sum of all weights). If provided, you must set `budget=None`.
        The default (`None`) means no maximum budget constraint.

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

    References
    ----------
    .. [1] "Data-driven distributionally robust optimization using the Wasserstein
        metric: performance guarantees and tractable reformulations".
        Esfahani and Kuhn (2018).

    .. [2] "Optimization of conditional value-at-risk".
        Rockafellar and Uryasev (2000).
    """

    def __init__(
        self,
        risk_aversion: float = 1.0,
        cvar_beta: float = 0.95,
        wasserstein_ball_radius: float = 0.02,
        prior_estimator: BasePrior | None = None,
        min_weights: skt.MultiInput | None = 0.0,
        max_weights: skt.MultiInput | None = 1.0,
        budget: float | None = 1,
        min_budget: float | None = None,
        max_budget: float | None = None,
        max_short: float | None = None,
        max_long: float | None = None,
        groups: skt.Groups | None = None,
        linear_constraints: skt.LinearConstraints | None = None,
        left_inequality: skt.Inequality | None = None,
        right_inequality: skt.Inequality | None = None,
        risk_free_rate: float = 0.0,
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
            risk_measure=RiskMeasure.CVAR,
            prior_estimator=prior_estimator,
            min_weights=min_weights,
            max_weights=max_weights,
            budget=budget,
            min_budget=min_budget,
            max_budget=max_budget,
            max_short=max_short,
            max_long=max_long,
            groups=groups,
            linear_constraints=linear_constraints,
            left_inequality=left_inequality,
            right_inequality=right_inequality,
            risk_free_rate=risk_free_rate,
            cvar_beta=cvar_beta,
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
        self.risk_aversion = risk_aversion
        self.wasserstein_ball_radius = wasserstein_ball_radius

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> "DistributionallyRobustCVaR":
        """Fit the Distributionally Robust CVaR Optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : array-like of shape (n_observations, n_factors), optional
            Price returns of factors.
            The default is `None`.

        Returns
        -------
        self : DistributionallyRobustCVaR
           Fitted estimator.
        """
        self._check_feature_names(X, reset=True)
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

        a1 = -1
        b1 = cp.Constant(self.risk_aversion)
        a2 = -1 - self.risk_aversion / (1 - self.cvar_beta)
        b2 = cp.Constant(self.risk_aversion * (1 - 1 / (1 - self.cvar_beta)))
        ones = np.ones(n_assets)
        w = cp.Variable(n_assets)
        u = cp.Variable((n_observations, n_assets))
        v = cp.Variable((n_observations, n_assets))
        lb = cp.Variable()
        tau = cp.Variable()
        s = cp.Variable(n_observations)

        factor = cp.Constant(1)

        # constraints
        constraints = self._get_weight_constraints(
            n_assets=n_assets, w=w, factor=factor
        )
        constraints += [
            u * self._scale_constraints >= cp.Constant(0),
            v * self._scale_constraints >= cp.Constant(0),
            b1 * tau * self._scale_constraints
            + a1 * (prior_model.returns @ w) * self._scale_constraints
            + cp.multiply(u, (1 + prior_model.returns)) @ ones * self._scale_constraints
            <= s * self._scale_constraints,
            b2 * tau * self._scale_constraints
            + a2 * (prior_model.returns @ w) * self._scale_constraints
            + cp.multiply(v, (1 + prior_model.returns)) @ ones * self._scale_constraints
            <= s * self._scale_constraints,
        ]

        for i in range(n_observations):
            # noinspection PyTypeChecker
            constraints.append(
                cp.norm(-u[i] - a1 * w, np.inf) * self._scale_constraints
                <= lb * self._scale_constraints
            )
            # noinspection PyTypeChecker
            constraints.append(
                cp.norm(-v[i] - a2 * w, np.inf) * self._scale_constraints
                <= lb * self._scale_constraints
            )

        # custom objectives and constraints
        custom_objective = self._get_custom_objective(w=w)
        constraints += self._get_custom_constraints(w=w)

        objective = cp.Minimize(
            cp.Constant(self.wasserstein_ball_radius) * lb * self._scale_objective
            + (1 / n_observations) * cp.sum(s) * self._scale_objective
            + custom_objective * self._scale_objective
        )

        # problem
        problem = cp.Problem(objective, constraints)

        # results
        self._solve_problem(problem=problem, w=w, factor=factor)

        return self
