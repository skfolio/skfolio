"""Maximum Diversification Optimization estimator."""

# Copyright (c) 2023-2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import sklearn.utils.validation as skv

import skfolio.typing as skt
from skfolio.measures import RiskMeasure
from skfolio.optimization.convex._base import ObjectiveFunction
from skfolio.optimization.convex._mean_risk import MeanRisk
from skfolio.prior import BasePrior


class MaximumDiversification(MeanRisk):
    r"""Maximum Diversification Optimization estimator.

    Maximizes the diversification ratio which is the ratio of the weighted volatilities
    over the total volatility.

    It is a special case of the :class:`~skfolio.optimization.MeanRisk` estimator where
    the expected return from the objective function is replaced by the weighted
    volatilities.

    Parameters
    ----------
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
            needs to be homogeneous to the periodicity of :math:`\mu`. For example, if
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
            be homogeneous to the periodicity of :math:`\mu`. For example, if the input
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

    linear_constraints : array-like of shape (n_constraints,), optional
        Linear constraints.
        The linear constraints must match any of following patterns:

           * `"2.5 * ref1 + 0.10 * ref2 + 0.0013 <= 2.5 * ref3"`
           * `"ref1 >= 2.9 * ref2"`
           * `"ref1 == ref2"`
           * `"ref1 >= ref1"`

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

    min_return : float | array-like of shape (n_optimization), optional
        Lower bound constraint on the expected return.

    add_objective : Callable[[cp.Variable], cp.Expression], optional
        Add a custom objective to the existing objective expression.
        It is a function that must take as argument the weights `w` and returns a
        CVXPY expression.

    add_constraints : Callable[[cp.Variable], cp.Expression|list[cp.Expression]], optional
        Add a custom constraint or a list of constraints to the existing constraints.
        It is a function that must take as argument the weights `w` and returns a
        CVPXY expression or a list of CVPXY expressions.

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

    problem_: cvxpy.Problem
        CVXPY problem used for the optimization. Only when `save_problem` is set to
        `True`.

    n_features_in_ : int
        Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of assets seen during `fit`. Defined only when `X`
        has assets names that are all strings.

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

    def __init__(
        self,
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
        risk_free_rate: float = 0.0,
        min_return: skt.Target | None = None,
        max_tracking_error: skt.Target | None = None,
        max_turnover: skt.Target | None = None,
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
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            risk_measure=RiskMeasure.STANDARD_DEVIATION,
            prior_estimator=prior_estimator,
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
            min_return=min_return,
            max_tracking_error=max_tracking_error,
            max_turnover=max_turnover,
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
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None, **fit_params
    ) -> MaximumDiversification:
        """Fit the Maximum Diversification Optimization estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : array-like of shape (n_observations, n_targets), optional
            Price returns of factors or a target benchmark.
            The default is `None`.

        **fit_params : dict
            Parameters to pass to the underlying estimators.
            Only available if `enable_metadata_routing=True`, which can be
            set by using ``sklearn.set_config(enable_metadata_routing=True)``.
            See :ref:`Metadata Routing User Guide <metadata_routing>` for
            more details.

        Returns
        -------
        self : MaximumDiversification
           Fitted estimator.
        """
        # `X` is unchanged and only `feature_names_in_` is performed
        _ = skv.validate_data(self, X, skip_check_array=True)

        def func(w, obj):
            """Weighted volatilities."""
            covariance = obj.prior_estimator_.return_distribution_.covariance
            return np.sqrt(np.diag(covariance)) @ w

        self.overwrite_expected_return = func
        super().fit(X, y, **fit_params)
        return self
