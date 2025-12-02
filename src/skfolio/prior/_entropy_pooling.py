"""Entropy Pooling estimator."""

# Copyright (c) 2025
# Author: Hugo Delatte <delatte.hugo@gmail.com>
# Credits: Vincent Maladière, Matteo Manzi, Carlo Nicolini
# SPDX-License-Identifier: BSD-3-Clause

import operator
import re
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import cvxpy as cp
import numpy as np
import numpy.typing as npt
import scipy.optimize as sco
import scipy.sparse.linalg as scl
import scipy.special as scs
import scipy.stats as sts
import sklearn.utils.metadata_routing as skm
import sklearn.utils.validation as skv

import skfolio.measures as sm
import skfolio.typing as skt
from skfolio.exceptions import SolverError
from skfolio.measures import (
    ExtraRiskMeasure,
    PerfMeasure,
    RiskMeasure,
)
from skfolio.prior._base import BasePrior, ReturnDistribution
from skfolio.prior._empirical import EmpiricalPrior
from skfolio.utils.equations import equations_to_matrix
from skfolio.utils.tools import check_estimator, default_asset_names, input_to_array


class EntropyPooling(BasePrior):
    r"""Entropy Pooling estimator.

    Entropy Pooling, introduced by Attilio Meucci in 2008 as a generalization of the
    Black-Litterman framework, is a nonparametric method for adjusting a baseline
    ("prior") probability distribution to incorporate user-defined views by finding the
    posterior distribution closest to the prior while satisfying those views.

    User-defined views can be **elicited** from domain experts or **derived** from
    quantitative analyses.

    Grounded in information theory, it updates the distribution in the least-informative
    way by minimizing the Kullback-Leibler divergence (relative entropy) under the
    specified view constraints.

    Mathematically, the problem is formulated as:

    .. math::

       \begin{aligned}
       \min_{\mathbf{q}} \quad & \sum_{i=1}^T q_i \log\left(\frac{q_i}{p_i}\right) \\
       \text{subject to} \quad & \sum_{i=1}^T q_i = 1 \quad \text{(normalization constraint)} \\
                               & \mathbb{E}_q[f_j(X)] = v_j \quad(\text{or } \le v_j, \text{ or } \ge v_j), \quad j = 1,\dots,k, \text{(view constraints)} \\
                               & q_i \ge 0, \quad i = 1, \dots, T
       \end{aligned}

    Where:

    - :math:`T` is the number of observations (number of scenarios).
    - :math:`p_i` is the prior probability of scenario :math:`x_i`.
    - :math:`q_i` is the posterior probability of scenario :math:`x_i`.
    - :math:`X` is the scenario matrix of shape (n_observations, n_assets).
    - :math:`f_j` is the j :sup:`th` view function.
    - :math:`v_j` is the target value imposed by the j :sup:`th` view.
    - :math:`k` is the total number of views.

    The `skfolio` implementation supports the following views:
        * Equalities
        * Inequalities
        * Ranking
        * Linear combinations (e.g. relative views)
        * Views on groups of assets

    On the following measures:
        * Mean
        * Variance
        * Skew
        * Kurtosis
        * Correlation
        * Value-at-Risk (VaR)
        * Conditional Value-at-Risk (CVaR)

    Notes
    -----
    Entropy Pooling re-weights the sample probabilities of the prior distribution and is
    therefore constrained by the support (completeness) of that distribution. For
    example, if the historical distribution contains no returns below -10% for a given
    asset, we cannot impose a CVaR view of 15%: no matter how we adjust the sample
    probabilities, such tail data do not exist.

    Therefore, to impose extreme views on a sparse historical distribution, one must
    generate synthetic data. In that case, the EP posterior is only as reliable as the
    synthetic scenarios. It is thus essential to use a generator capable of
    extrapolating tail dependencies, such as :class:`~skfolio.distribution.VineCopula`,
    to model joint extreme events accurately.

    Two methods are available:
        * Dual form: solves the Fenchel dual of the EP problem using Truncated
          Newton Constrained method.
        * Primal form: solves the original relative-entropy projection in
          probability-space via interior-point algorithms.

    See the solver parameter's docstring for full details on available solvers and
    options.

    To handle nonlinear views, constraints are linearized by fixing the relevant asset
    moments (e.g., means or variances) and then solved via **nested entropic tilting**.
    At each stage, the KL-divergence is minimized relative to the original prior,
    while nesting all previously enforced (linearized) views into the feasible set:

    * Stage 1: impose views on asset means, VaR and CVaR.
    * Stage 2: carry forward Stage 1 constraints and add variance, fixing the
      mean at its Stage 1 value.
    * Stage 3: carry forward Stage 2 constraints and add skewness, kurtosis and
      pairwise correlations, fixing both mean and variance at their Stage 2 values.

    Because each entropic projection nests the prior views, every constraint from
    earlier stages is preserved as new ones are added, yielding a final distribution
    that satisfies all original nonlinear views while staying as close as possible to
    the original prior.

    Only the necessary moments are fixed. Slack variables with an L1 norm penalty are
    introduced to avoid solver infeasibility that may arise from overly tight
    constraints.

    CVaR view constraints cannot be directly expressed as linear functions of the
    posterior probabilities. Therefore, when CVaR views are present, the EP problem is
    solved by recursively solving a series of convex programs that approximate the
    nonlinear CVaR constraint.

    This implementation improves upon Meucci's algorithm [1]_ by formulating the problem
    in continuous space as a function of the dual variables etas (VaR levels), rather
    than searching over discrete tail sizes. This formulation not only handles the
    CVaR constraint more directly but also supports multiple CVaR views on different
    assets.

    Although the overall problem is convex in the dual variables etas, it remains
    non-smooth due to the presence of the positive-part operator in the CVaR
    definition. Consequently, we employ derivative-free optimization methods.
    Specifically, for a single CVaR view we use a one-dimensional root-finding
    method (Brent's method), and for the multivariate case (supporting multiple
    CVaR views) we use Powell's method for derivative-free convex descent.

    Parameters
    ----------
    prior_estimator : BasePrior, optional
        Estimator of the asset's prior distribution, fitted from a
        :ref:`prior estimator <prior>`. The default (`None`) is to use the
        empirical prior :class:`~skfolio.prior.EmpiricalPrior()`. To perform Entropy
        Pooling on synthetic data, you can use :class:`~skfolio.prior.SyntheticData`
        by setting `prior_estimator = SyntheticData()`.

    mean_views : list[str], optional
        Views on asset means.
        The views must match any of following patterns:

            * `"ref1 >= a"`
            * `"ref1 == b"`
            * `"ref1 <= ref1"`
            * `"ref1 >= a * prior(ref1)"`
            * `"ref1 == b * prior(ref2)"`
            * `"a * ref1 + b * ref2 + c <= d * ref3"`

        With `"ref1"`, `"ref2"` ... the assets names or the groups names provided
        in the parameter `groups`. Assets names can be referenced without the need of
        `groups` if the input `X` of the `fit` method is a DataFrame with assets names
        in columns. Otherwise, the default asset names `x0, x1, ...` are assigned.
        By using the term `prior(...)`, you can reference the asset prior mean.

        For example:

            * `"SPX >= 0.0015"` --> The mean of SPX is greater than 0.15% (daily mean if
              `X` is daily)
            * `"SX5E == 0.002"` --> The mean of SX5E equals 0.2%
            * `"AAPL <= 0.003"` --> The mean of AAPL is less than to 0.3%
            * `"SPX <= SX5E"` --> Ranking view: the mean of SPX is less than SX5E
            * `"SPX >= 1.5 * prior(SPX)"` --> The mean of SPX increases by at least 50% (versus its prior)
            * `"SX5E == 2 * prior(SX5E)"` --> The mean of SX5E doubles (versus its prior)
            * `"AAPL <= 0.8 * prior(SPX)"` --> The mean of AAPL is less than 0.8 times the SPX prior
            * `"SX5E + SPX >= 0"` --> The sum of SX5E and SPX mean is greater than zero
            * `"US == 0.007"` --> The sum of means of US assets equals 0.7%
            * `"Equity == 3 * Bond"` --> The sum of means of Equity assets equals
              three times the  sum of means of Bond assets.
            * `"2*SPX + 3*Europe <= Bond + 0.05"` --> Mixing assets and group mean views

    variance_views : list[str], optional
        Views on asset variances.
        It supports the same patterns as `mean_views`.

        For example:

            * `"SPX >= 0.0009"` --> SPX variance is greater than 0.0009 (daily)
            * `"SX5E == 1.5 * prior(SX5E)"` --> SX5E variance increases by 150% (versus
              its prior)

    skew_views : list[str], optional
        Views on asset skews.
        It supports the same patterns as `mean_views`.

        For example:

            * `"SPX >= 2.0"` --> SPX skew is greater than 2.0
            * `"SX5E == 1.5 * prior(SX5E)"` --> SX5E skew increases by 150% (versus its
              prior)

    kurtosis_views : list[str], optional
        Views on asset kurtosis.
        It supports the same patterns as `mean_views`.

        For example:

            * `"SPX >= 9.0"` --> SPX kurtosis is greater than 9.0
            * `"SX5E == 1.5 * prior(SX5E)"` --> SX5E kurtosis increases by 150% (versus
              its prior)

    correlation_views : list[str], optional
        Views on asset correlations.
        The views must match any of following patterns:

            * `"(asset1, asset2) >= a"`
            * `"(asset1, asset2) == a"`
            * `"(asset1, asset2) <= a"`
            * `"(asset1, asset2) >= a * prior(asset1, asset2)"`
            * `"(asset1, asset2) == a * prior(asset1, asset2)"`
            * `"(asset1, asset2) <= a * prior(asset1, asset2)"`

        For example:

            * `"(SPX, SX5E) >= 0.8"` --> the correlation between SPX and SX5E is greater than 80%
            * `"(SPX, SX5E) == 1.5 * prior(SPX, SX5E)"` --> the correlation between SPX
              and SX5E increases by 150% (versus its prior)

    value_at_risk_views : list[str], optional
        Views on asset Value-at-Risks (VaR).

        For example:

            * `"SPX >= 0.03"` --> SPX VaR is greater than 3%
            * `"SX5E == 1.5 * prior(SX5E)"` --> SX5E VaR increases by 150% (versus its prior)

    cvar_views : list[str], optional
        Views on asset Conditional Value-at-Risks (CVaR).
        It only supports equalities.

        For example:

            * `"SPX == 0.05"` --> SPX CVaR equals 5%
            * `"SX5E == 1.5 * prior(SX5E)"` --> SX5E CVaR increases by 150% (versus its prior)

    value_at_risk_beta : float, default=0.95
        Confidence level for VaR views, by default 95%.

    cvar_beta : float, default=0.95
        Confidence level for CVaR views, by default 95%.

    groups : dict[str, list[str]] or array-like of strings of shape (n_groups, n_assets), optional
        Asset grouping for use in group-based views. If a dict is provided, keys are
        asset names and values are lists of group labels; then `X` must be a DataFrame
        whose columns match those asset names.

        For example:

            * `groups = {"SX5E": ["Equity", "Europe"], "SPX": ["Equity", "US"], "TLT": ["Bond", "US"]}`
            * `groups = [["Equity", "Equity", "Bond"], ["Europe", "US", "US"]]`

    solver : str, default="TNC"
        The solver to use.

        - "TNC" (default) solves the entropic-pooling dual via SciPy's Truncated Newton
          Constrained method. By exploiting the smooth Fenchel dual and its
          closed-form gradient, it operates in :math:`\mathbb{R}^k` (the number of
          constraints) rather than :math:`\mathbb{R}^T` (the number of scenarios),
          yielding an order-of-magnitude speedup over primal CVXPY interior-point
          solvers.

        - CVXPY solvers (e.g. "CLARABEL") solve the entropic-pooling problem in its
          primal form using interior-point methods. While they tend to be slower than
          the dual-based approach, they often achieve higher accuracy by enforcing
          stricter primal feasibility and duality-gap tolerances. See the CVXPY
          documentation for supported solvers:
          https://www.cvxpy.org/tutorial/advanced/index.html#choosing-a-solver.

    solver_params : dict, optional
        Additional parameters to pass to the chosen solver.

        - When using **SciPy TNC**, supported options include (but are not limited to)
          `gtol`, `ftol`, `eps`, `maxfun`, `maxCGit`, `stepmx`, `disp`. See the SciPy
          documentation for a full list and descriptions:
          https://docs.scipy.org/doc/scipy/reference/optimize.minimize-tnc.html

        - When using a **CVXPY** solver (e.g. ``"CLARABEL"``), supply any
          solver-specific parameters here. Refer to the CVXPY solver guide for
          details: https://www.cvxpy.org/tutorial/solvers

    Attributes
    ----------
    return_distribution_ : ReturnDistribution
        Fitted :class:`~skfolio.prior.ReturnDistribution` to be used by the optimization
        estimators, containing the assets distribution, moments estimation and the EP
        posterior probabilities (sample weights).

    relative_entropy_ : float
        The KL-divergence between the posterior and prior distributions.

    effective_number_of_scenarios_ : float
        Effective number of scenarios defined as the perplexity of sample weight
        (exponential of entropy).

    prior_estimator_ : BasePrior
        Fitted `prior_estimator`.

    n_features_in_ : int
       Number of assets seen during `fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
       Names of features seen during `fit`. Defined only when `X`
       has feature names that are all strings.

    References
    ----------
    .. [1] "Fully Flexible Extreme Views",
            Journal of Risk, Meucci, Ardia & Keel (2011)

    .. [2] "Fully Flexible Views: Theory and Practice",
            Risk, Meucci (2013).

    .. [3] "Effective Number of Scenarios in Fully Flexible Probabilities",
            GARP Risk Professional, Meucci (2012)

    .. [4] "I-Divergence Geometry of Probability Distributions and Minimization
            Problems", The Annals of Probability, Csiszar (1975)


    Examples
    --------
    For a full tutorial on entropy pooling, see :ref:`sphx_glr_auto_examples_entropy_pooling_plot_1_entropy_pooling.py`.

    >>> from skfolio import RiskMeasure
    >>> from skfolio.datasets import load_sp500_dataset
    >>> from skfolio.preprocessing import prices_to_returns
    >>> from skfolio.prior import EntropyPooling
    >>> from skfolio.optimization import HierarchicalRiskParity
    >>>
    >>> prices = load_sp500_dataset()
    >>> prices = prices[["AMD", "BAC", "GE", "JNJ", "JPM", "LLY", "PG"]]
    >>> X = prices_to_returns(prices)
    >>>
    >>> groups = {
    ...     "AMD": ["Technology", "Growth"],
    ...     "BAC": ["Financials", "Value"],
    ...     "GE": ["Industrials", "Value"],
    ...     "JNJ": ["Healthcare", "Defensive"],
    ...     "JPM": ["Financials", "Income"],
    ...     "LLY": ["Healthcare", "Defensive"],
    ...     "PG": ["Consumer", "Defensive"],
    ... }
    >>>
    >>> entropy_pooling = EntropyPooling(
    ...     mean_views=[
    ...         "JPM == -0.002",
    ...         "PG >= LLY",
    ...         "BAC >= prior(BAC) * 1.2",
    ...         "Financials == 2 * Growth",
    ...     ],
    ...     variance_views=[
    ...         "BAC == prior(BAC) * 4",
    ...     ],
    ...     correlation_views=[
    ...         "(BAC,JPM) == 0.80",
    ...         "(BAC,JNJ) <= prior(BAC,JNJ) * 0.5",
    ...     ],
    ...     skew_views=[
    ...         "BAC == -0.05",
    ...     ],
    ...     cvar_views=[
    ...         "GE == 0.08",
    ...     ],
    ...     cvar_beta=0.90,
    ...     groups=groups,
    ... )
    >>>
    >>> entropy_pooling.fit(X)
    EntropyPooling(correlation_views=...
    >>>
    >>> print(entropy_pooling.relative_entropy_)
    0.18...
    >>> print(entropy_pooling.effective_number_of_scenarios_)
    6876.67...
    >>> print(entropy_pooling.return_distribution_.sample_weight)
    [0.000103...  0.000093... ... 0.000103...  0.000108...]
    >>>
    >>> # CVaR Hierarchical Risk Parity optimization on Entropy Pooling
    >>> model = HierarchicalRiskParity(
    ...     risk_measure=RiskMeasure.CVAR,
    ...     prior_estimator=entropy_pooling
    ... )
    >>> model.fit(X)
    HierarchicalRiskParity(prior_estimator=...
    >>> print(model.weights_)
    [0.073... 0.0541... ... 0.200...]
    >>>
    >>> # Stress Test the Portfolio
    >>> entropy_pooling = EntropyPooling(cvar_views=["AMD == 0.10"])
    >>> entropy_pooling.fit(X)
    EntropyPooling(cvar_views=['AMD == 0.10'])
    >>>
    >>> stressed_dist = entropy_pooling.return_distribution_
    >>>
    >>> stressed_ptf = model.predict(stressed_dist)
    """

    relative_entropy_: float
    effective_number_of_scenarios_: float
    prior_estimator_: BasePrior
    n_features_in_: int
    feature_names_in_: np.ndarray

    if TYPE_CHECKING:
        _returns: np.ndarray
        _prior_sample_weight: np.ndarray
        _groups: np.ndarray
        _is_fixed_mean: np.ndarray
        _is_fixed_variance: np.ndarray
        _constraints: dict[str, list[np.ndarray] | None]

    def __init__(
        self,
        prior_estimator: BasePrior | None = None,
        mean_views: list[str] | None = None,
        variance_views: list[str] | None = None,
        correlation_views: list[str] | None = None,
        skew_views: list[str] | None = None,
        kurtosis_views: list[str] | None = None,
        value_at_risk_views: list[str] | None = None,
        cvar_views: list[str] | None = None,
        value_at_risk_beta: float = 0.95,
        cvar_beta: float = 0.95,
        groups: skt.Groups | None = None,
        solver: str = "TNC",
        solver_params: dict | None = None,
    ):
        self.prior_estimator = prior_estimator
        self.mean_views = mean_views
        self.variance_views = variance_views
        self.correlation_views = correlation_views
        self.skew_views = skew_views
        self.kurtosis_views = kurtosis_views
        self.value_at_risk_views = value_at_risk_views
        self.cvar_views = cvar_views
        self.value_at_risk_beta = value_at_risk_beta
        self.cvar_beta = cvar_beta
        self.groups = groups
        self.solver = solver
        self.solver_params = solver_params

    def get_metadata_routing(self):
        router = skm.MetadataRouter(owner=self.__class__.__name__).add(
            prior_estimator=self.prior_estimator,
            method_mapping=skm.MethodMapping().add(caller="fit", callee="fit"),
        )
        return router

    def fit(self, X: npt.ArrayLike, y=None, **fit_params) -> "EntropyPooling":
        """Fit the Entropy Pooling estimator.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_assets)
           Price returns of the assets.

        y : Ignored
           Not used, present for API consistency by convention.

        **fit_params : dict
           Parameters to pass to the underlying estimators.
           Only available if `enable_metadata_routing=True`, which can be
           set by using ``sklearn.set_config(enable_metadata_routing=True)``.
           See :ref:`Metadata Routing User Guide <metadata_routing>` for
           more details.

        Returns
        -------
        self : EntropyPooling
           Fitted estimator.
        """
        routed_params = skm.process_routing(self, "fit", **fit_params)

        # Validation
        skv.validate_data(self, X)

        self.prior_estimator_ = check_estimator(
            self.prior_estimator,
            default=EmpiricalPrior(),
            check_type=BasePrior,
        )
        # Fitting prior estimator
        self.prior_estimator_.fit(X, y, **routed_params.prior_estimator.fit)
        # Prior distribution
        self._returns = self.prior_estimator_.return_distribution_.returns
        self._mu = self.prior_estimator_.return_distribution_.mu
        self._covariance = self.prior_estimator_.return_distribution_.covariance
        n_observations, n_assets = self._returns.shape
        self._prior_sample_weight = (
            self.prior_estimator_.return_distribution_.sample_weight
        )
        if self._prior_sample_weight is None:
            self._prior_sample_weight = np.ones(n_observations) / n_observations

        assets_names = getattr(self, "feature_names_in_", default_asset_names(n_assets))
        if self.groups is None:
            self._groups = np.asarray([assets_names])
        else:
            self._groups = input_to_array(
                items=self.groups,
                n_assets=n_assets,
                fill_value="",
                dim=2,
                assets_names=assets_names,
                name="groups",
            )

        # Init problem variables
        self._is_fixed_mean = np.zeros(n_assets, dtype=bool)
        self._is_fixed_variance = np.zeros(n_assets, dtype=bool)
        self._constraints = {
            "equality": None,
            "inequality": None,
            "fixed_equality": None,
            "cvar_equality": None,
        }

        # Step 1: Mean, VaR and CVaR
        self._add_mean_views()
        self._add_value_at_risk_views()
        sample_weight = self._solve_with_cvar()

        # Step 2:  Mean, VaR, CVaR and Variance
        if self.variance_views is not None:
            # Get mean from Step 1
            mean = sm.mean(self._returns, sample_weight=sample_weight)
            # Add new views and solve
            self._add_variance_views(mean=mean)
            sample_weight = self._solve_with_cvar()

        # Step 3: Mean, VaR, CVaR, Variance, Correlation, Skew and Kurtosis
        if (
            self.correlation_views is not None
            or self.skew_views is not None
            or self.kurtosis_views is not None
        ):
            # Get mean and variance from Step 2
            mean = sm.mean(self._returns, sample_weight=sample_weight)
            variance = sm.variance(
                self._returns, sample_weight=sample_weight, biased=True
            )
            # Add new views and solve
            self._add_correlation_views(mean=mean, variance=variance)
            self._add_skew_views(mean=mean, variance=variance)
            self._add_kurtosis_views(mean=mean, variance=variance)
            sample_weight = self._solve_with_cvar()

        self.relative_entropy_ = float(
            np.sum(scs.rel_entr(sample_weight, self._prior_sample_weight))
        )
        self.effective_number_of_scenarios_ = np.exp(sts.entropy(sample_weight))
        self.return_distribution_ = ReturnDistribution(
            mu=sm.mean(self._returns, sample_weight=sample_weight),
            covariance=np.cov(self._returns, rowvar=False, aweights=sample_weight),
            returns=self._returns,
            sample_weight=sample_weight,
        )

        # Manage memory
        del self._returns
        del self._mu
        del self._covariance
        del self._constraints
        del self._prior_sample_weight

        return self

    def _add_constraint(self, a: np.ndarray, b: np.ndarray, name: str) -> None:
        """Add the left matrix `a` and right vector `b` of linear equality constraints
        `x @ a == b` and linear inequality constraints `x @ a <= b` to the
        `_constraints` dict.

        Parameters
        ----------
        a : ndarray of shape (n_observations, n_constraints)
            Left matrix in `x @ a == b` or `x @ a <= b`.

        a : ndarray of shape (n_observations, n_constraints)
            Right vector in `x @ a == b` or `x @ a <= b`.

        Returns
        -------
        None
        """
        if b.size == 0:
            return

        # Init constraints dict
        if self._constraints[name] is None:
            n_observations, _ = self._returns.shape
            self._constraints[name] = [np.empty((n_observations, 0)), np.empty(0)]

        # Rescaling: views can be on different scales, by rescaling we avoid high
        # disparity, have better conditioning, uniform stopping criteria and slack
        # penalties.
        scales = np.linalg.norm(a, axis=0)
        a /= scales
        b /= scales

        for i, x in enumerate([a, b]):
            self._constraints[name][i] = np.hstack((self._constraints[name][i], x))

    def _add_mean_views(self) -> None:
        """Add mean view constraints to the optimization problem."""
        if self.mean_views is None:
            return

        a_eq, b_eq, a_ineq, b_ineq = self._process_views(measure=PerfMeasure.MEAN)

        for a, b, name in [(a_eq, b_eq, "equality"), (a_ineq, b_ineq, "inequality")]:
            if a.size != 0:
                self._add_constraint(a=self._returns @ a.T, b=b, name=name)

    def _add_variance_views(self, mean: np.ndarray) -> None:
        """Add variance view constraints to the optimization problem.

        Parameters
        ----------
        mean : ndarray of shape (n_assets,)
            The fixed mean vector used to compute variance as a linear function of
            sample weights.
        """
        if self.variance_views is None:
            return

        a_eq, b_eq, a_ineq, b_ineq = self._process_views(measure=RiskMeasure.VARIANCE)

        _, n_assets = self._returns.shape

        fix = np.zeros(n_assets, dtype=bool)
        for a, b, name in [(a_eq, b_eq, "equality"), (a_ineq, b_ineq, "inequality")]:
            if a.size != 0:
                self._add_constraint(
                    a=(self._returns - mean) ** 2 @ a.T, b=b, name=name
                )
                fix |= np.any(a != 0, axis=0)

        self._fix_mean(fix=fix, mean=mean)

    def _add_skew_views(self, mean: np.ndarray, variance: np.ndarray) -> None:
        """Add skew view constraints to the optimization problem.

        Parameters
        ----------
        mean : ndarray of shape (n_assets,)
            The fixed mean vector used to compute skew as a linear function of sample
            weights.

        variance : ndarray of shape (n_assets,)
            The fixed variance vector used to compute skew as a linear function of
            sample weights.
        """
        if self.skew_views is None:
            return

        a_eq, b_eq, a_ineq, b_ineq = self._process_views(measure=ExtraRiskMeasure.SKEW)

        _, n_assets = self._returns.shape

        fix = np.zeros(n_assets, dtype=bool)
        for a, b, name in [(a_eq, b_eq, "equality"), (a_ineq, b_ineq, "inequality")]:
            if a.size != 0:
                self._add_constraint(
                    a=(self._returns**3 - mean**3 - 3 * mean * variance)
                    / variance**1.5
                    @ a.T,
                    b=b,
                    name=name,
                )
                fix |= np.any(a != 0, axis=0)

        # Fix the mean and the variance of the correlation views  in order for the
        # correlation to match exactly the views
        self._fix_mean(fix=fix, mean=mean)
        self._fix_variance(fix=fix, mean=mean, variance=variance)

    def _add_kurtosis_views(self, mean: np.ndarray, variance: np.ndarray) -> None:
        """Add kurtosis view constraints to the optimization problem.

        Parameters
        ----------
        mean : ndarray of shape (n_assets,)
           The fixed mean vector used to compute kurtosis as a linear function of
           sample weights.

        variance : ndarray of shape (n_assets,)
           The fixed variance vector used to compute kurtosis as a linear function of
           sample weights.
        """
        if self.kurtosis_views is None:
            return

        a_eq, b_eq, a_ineq, b_ineq = self._process_views(
            measure=ExtraRiskMeasure.KURTOSIS
        )

        _, n_assets = self._returns.shape
        fix = np.zeros(n_assets, dtype=bool)
        for a, b, name in [(a_eq, b_eq, "equality"), (a_ineq, b_ineq, "inequality")]:
            if a.size != 0:
                self._add_constraint(
                    a=(
                        (
                            self._returns**4
                            - 4 * mean * self._returns**3
                            + 6 * mean**2 * self._returns**2
                            - 3 * mean**4
                        )
                        / variance**2
                        @ a.T
                    ),
                    b=b,
                    name=name,
                )
                fix |= np.any(a != 0, axis=0)

        self._fix_mean(fix=fix, mean=mean)
        self._fix_variance(fix=fix, mean=mean, variance=variance)

    def _add_value_at_risk_views(self) -> None:
        """Add Value-at-Risk (VaR) view constraints to the optimization problem."""
        if self.value_at_risk_views is None:
            return

        a_eq, b_eq, a_ineq, b_ineq = self._process_views(
            measure=ExtraRiskMeasure.VALUE_AT_RISK
        )

        if (a_eq.size != 0 and np.any(np.count_nonzero(a_eq, axis=1) != 1)) or (
            a_ineq.size != 0 and np.any(np.count_nonzero(a_ineq, axis=1) != 1)
        ):
            raise ValueError(
                "You cannot mix multiple assets in a single Value-at-Risk view."
            )

        if np.any(b_eq < 0) | np.any(a_ineq * b_ineq[:, np.newaxis] < 0):
            raise ValueError("Value-at-Risk views must be strictly positive.")

        n_observations, _ = self._returns.shape

        for a, b, name in [(a_eq, b_eq, "equality"), (a_ineq, b_ineq, "inequality")]:
            for ai, bi in zip(a, b, strict=True):
                idx = np.where(self._returns[:, ai.astype(bool)].flatten() <= -abs(bi))[
                    0
                ]
                if idx.size == 0:
                    raise ValueError(
                        f"The Value-at-Risk view of {bi:0.3%} is excessively extreme. "
                        "Consider lowering the view or adjusting your prior "
                        "distribution to include more extreme values."
                    )
                sign = 1 if name == "equality" or bi > 0 else -1
                ai = np.zeros(n_observations)
                ai[idx] = 1.0
                self._add_constraint(
                    a=sign * ai.reshape(-1, 1),
                    b=sign * np.array([1 - self.value_at_risk_beta]),
                    name=name,
                )

    def _add_correlation_views(self, mean: np.ndarray, variance: np.ndarray) -> None:
        """Add correlation view constraints to the optimization problem.

        Parameters
        ----------
        mean : ndarray of shape (n_assets,)
           The fixed mean vector used to compute correlation as a linear function of
           sample weights.

        variance : ndarray of shape (n_assets,)
           The fixed variance vector used to compute correlation as a linear function of
           sample weights.
        """
        if self.correlation_views is None:
            return

        assets = self._groups[0]
        _, n_assets = self._returns.shape
        asset_to_index = {asset: i for i, asset in enumerate(assets)}
        try:
            views = []
            for view in self.correlation_views:
                res = _parse_correlation_view(view, assets=assets)
                expression = res["expression"]
                corr_view = expression["constant"]
                if "prior_assets" in expression:
                    i, j = (asset_to_index[a] for a in expression["prior_assets"])
                    corr_view += (
                        self._covariance[i, j]
                        / np.sqrt(self._covariance[i, i] * self._covariance[j, j])
                        * expression["multiplier"]
                    )
                    corr_view = np.clip(corr_view, 0 + 1e-8, 1 - 1e-8)
                views.append(
                    (
                        (asset_to_index[a] for a in res["assets"]),
                        res["operator"],
                        corr_view,
                    )
                )
        except KeyError as e:
            raise ValueError(f"Asset {e.args[0]} is missing from the assets.") from None

        fix = np.zeros(n_assets, dtype=bool)
        for (i, j), op, corr_view in views:
            if not 0 <= corr_view <= 1:
                raise ValueError("Correlation views must be between 0 and 1.")
            ai = self._returns[:, i] * self._returns[:, j]
            bi = mean[i] * mean[j] + corr_view * np.sqrt(variance[i] * variance[j])
            sign = 1 if op in [operator.eq, operator.lt, operator.le] else -1
            self._add_constraint(
                a=sign * ai.reshape(-1, 1),
                b=sign * np.array([bi]),
                name="equality" if op == operator.eq else "inequality",
            )
            fix[[i, j]] = True

        self._fix_mean(fix=fix, mean=mean)
        self._fix_variance(fix=fix, mean=mean, variance=variance)

    def _solve_with_cvar(self) -> np.ndarray:
        """Solve the entropy pooling problem handling CVaR view constraints.

        CVaR view constraints cannot be directly expressed as linear functions of the
        posterior probabilities. Therefore, when CVaR views are present, the EP problem
        must be solved by recursively solving a series of convex programs that
        approximate the non-linear CVaR constraint.

        Our approach improves upon Meucci's algorithm [1]_ by formulating the problem
        in continuous space as a function of the dual variables etas (VaR levels)
        rather than searching over discrete tail sizes. This formulation not only
        handles the CVaR constraint more directly but also supports multiple CVaR views
        on different assets.

        Although the overall problem is convex in the dual variables etas, it remains
        non-smooth due to the presence of the positive-part operator in the CVaR
        definition. Consequently, we employ derivative-free optimization methods.
        Specifically, for a single CVaR view we use a one-dimensional root-finding
        method (Brent's method), and for the multidimensional case (supporting multiple
        CVaR views) we utilize Powell's method for derivative-free convex descent.

        Returns
        -------
        sample_weight : ndarray of shape (n_observations,)
            The updated probability vector satisfying all view constraints.

        References
        ----------
        .. [1] "Fully Flexible Extreme Views",
                Journal of Risk, Meucci, Ardia & Keel (2011)
        """
        if self.cvar_views is None:
            sample_weight = self._solve()
            return sample_weight

        a_eq, b_eq, _, b_ineq = self._process_views(measure=RiskMeasure.CVAR)

        if b_ineq.size != 0:
            raise ValueError(
                "CVaR view inequalities are not supported, use equalities views `==`"
            )

        if np.any(np.count_nonzero(a_eq, axis=1) != 1):
            raise ValueError("You cannot mix multiple assets in a single CVaR view")

        if np.any(b_eq < 0):
            raise ValueError("CVaR view must be strictly positive")

        n_observations, _ = self._returns.shape
        asset_returns = self._returns[:, np.where(a_eq.sum(axis=0) != 0)[0]].T
        views = b_eq
        n_views = len(views)

        min_ret = -np.min(asset_returns, axis=1)
        invalid_views = views >= min_ret
        if np.any(invalid_views):
            msg = ""
            for v, m in zip(views[invalid_views], min_ret[invalid_views], strict=True):
                msg += (
                    "The CVaR views of "
                    + ", ".join([f"{v:.2%}" for v in views[invalid_views]])
                    + f" is excessively extreme and cannot exceed {m:.2%} which is the "
                    "worst realization. Consider lowering the view or adjusting your "
                    "prior distribution to include more extreme values."
                )
            raise ValueError(msg)

        def func(etas: list[float]) -> tuple[np.ndarray, float]:
            """Solve the EP with CVaR constraints for a given list of etas (VaR levels).

            Parameters
            ----------
            etas : list[float]
                The list of etas (VaR levels) of each asset with a CVaR view.

            Returns
            -------
            sample_weight : ndarray of shape (n_observations,)
                Sample weight of the CVaR EP problem.
            error : float
                For one-dimensional (a single eta), the error is the difference
                between the target CVaR beta and the effective CVaR beta.
                For multidimensional, the error is the RMSE of the difference between
                the target CVaR and the effective CVaR.
            """
            # Init CVaR constraints
            self._constraints["cvar_equality"] = [
                np.empty((n_observations, 0)),
                np.empty(0),
            ]
            pos_part = None
            for i in range(n_views):
                if not (0 < etas[i] < views[i]):
                    raise ValueError(
                        f"eta[{i}] must be between 0 and the CVaR view {views[i]}"
                    )
                pos_part = np.maximum(-asset_returns[i] - etas[i], 0)
                self._add_constraint(
                    a=(pos_part / (1 - self.cvar_beta)).reshape(-1, 1),
                    b=np.array([views[i] - etas[i]]),
                    name="cvar_equality",
                )
            w = self._solve()

            if n_views == 1:
                error = np.sum(w[pos_part != 0]) - (1 - self.cvar_beta)
            else:
                error = np.linalg.norm(
                    sm.cvar(asset_returns.T, beta=self.cvar_beta, sample_weight=w)
                    - views
                ) / np.sqrt(n_views)
            return w, error

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="^Solution may be inaccurate")

            # One-dimensional: we use one-dimensional root-finding method
            if n_views == 1:
                sol = sco.root_scalar(
                    lambda x: func([x])[-1],
                    bracket=[1e-3, views[0] - 1e-8],
                    method="brentq",  # Doesn't required a smooth derivative
                    xtol=1e-4,
                    maxiter=50,
                )
                if not sol.converged:
                    raise RuntimeError(
                        "Failed to solve the CVaR view problem. Consider relaxing your "
                        "constraints or substituting with VaR views."
                    )
                res = [sol.root]
            # Multidimensional: we use derivative-free convex descent
            else:
                sol = sco.minimize(
                    lambda x: func(x)[-1],
                    x0=np.array([view * 0.5 for view in views]),
                    bounds=[(1e-3, view - 1e-8) for view in views],
                    method="Powell",
                    options={"xtol": 1e-4, "maxiter": 80},
                )
                if not sol.success:
                    raise ValueError(
                        "Failed to solve the multi-CVaR view problem. Consider "
                        "relaxing your constraints, using a single CVaR view, or "
                        "substituting with VaR views."
                    )
                res = sol.x

        sample_weight = func(res)[0]
        return sample_weight

    def _solve(self) -> np.ndarray:
        """Solve the base entropy pooling problem.
        Dispatch either to the EP dual (TNC) solver or the EP primal (CVXPY) solver
        based on the `solver` parameter.

        Returns
        -------
        sample_weight : ndarray of shape (n_observations,)
           The updated posterior probability vector.
        """
        if all(v is None for v in self._constraints.values()):
            # No view constraints so we don't need to solve the EP problem.
            return self._prior_sample_weight

        if self.solver == "TNC":
            return self._solve_dual()
        return self._solve_primal()

    def _solve_dual(self) -> np.ndarray:
        r"""Solves the entropic-pooling dual via SciPy's Truncated Newton
        Constrained method. By exploiting the smooth Fenchel dual and its
        closed-form gradient, it operates in :math:`\mathbb{R}^k` (the number of
        constraints) rather than :math:`\mathbb{R}^T` (the number of scenarios),
        yielding an order-of-magnitude speedup over primal CVXPY interior-point
        solvers.

        Returns
        -------
        sample_weight : ndarray of shape (n_observations,)
           The updated posterior probability vector.

        References
        ----------
        .. [1] "Convex Optimization", Cambridge University Press, Section 5.2.3,
            Entropy maximization, Boyd & Vandenberghe (2004)

        .. [2] "I-Divergence Geometry of Probability Distributions and Minimization
            Problems", The Annals of Probability, Csiszar (1975)

        .. [3] "Fully Flexible Views: Theory and Practice",
                Risk, Meucci (2013).

        """
        n_observations, _ = self._returns.shape
        # Init constraints with sum(p)==1, rescaled by its norm
        # Has better convergence than the normalized form done inside the dual.
        a = [np.ones(n_observations).reshape(-1, 1) / np.sqrt(n_observations)]
        b = [np.array([1.0]) / np.sqrt(n_observations)]
        bounds = [(None, None)]
        for name, constraints in self._constraints.items():
            if constraints is not None:
                a.append(constraints[0])
                b.append(constraints[1])
                s = constraints[1].size
                match name:
                    case "equality" | "cvar_equality":
                        bounds += [(None, None)] * s
                    case "fixed_equality":
                        # Equivalent to relaxing the problem with slack variables with
                        # a norm penalty to avoid solver infeasibility that may arise
                        # from overly tight constraints from fixing the moments.
                        bounds += [(-1000, 1000)] * s
                    case "inequality":
                        bounds += [(0, None)] * s
                    case _:
                        raise KeyError(f"constrain {name}")

        a = np.hstack(a)
        b = np.hstack(b)

        def func(x: np.ndarray) -> tuple[float, np.ndarray]:
            """Computes the Fenchel dual of the entropic-pooling problem in its
            unnormalized form.

            Parameters
            ----------
            x : ndarray of shape (n_constraints,)
               Dual variables (Lagrange multipliers for each constraint).

            Returns
            -------
            obj : float
               Value of the dual objective.
            grad : ndarray of shape  (n_constraints,)
               Gradient of the dual objective.
            """
            z = self._prior_sample_weight * np.exp(-a @ x - 1)
            obj = z.sum() + x @ b
            grad = b - a.T @ z
            return obj, grad

        # We use TNC as it often outperforms L-BFGS-B on EP problems because
        # it builds (implicitly) true curvature information via Hessian-vector products,
        # whereas L-BFGS-B only ever uses a low-rank quasi-Newton approximation.
        # We set stepmx=1 to avoid large solver defaults when the bounds are not None.
        solver_params = (
            self.solver_params
            if self.solver_params is not None
            else {
                "maxfun": 5000,
                "ftol": 1e-11,
                "xtol": 1e-8,
                "gtol": 1e-8,
                "stepmx": 1,
            }
        )

        sol = sco.minimize(
            func,
            x0=np.zeros_like(b),
            jac=True,
            bounds=bounds,
            method="TNC",
            options=solver_params,
        )
        if not sol.success:
            raise SolverError(
                "Dual problem with Solver 'TNC' failed. This typically occurs when the "
                "specified views conflict or are overly extreme. Consider using a "
                "prior that generates more synthetic data for extreme views. You can "
                "also change `solver_params` or try another `solver` such as "
                f"'CLARABEL'. Solver error: {sol.message}"
            )
        sample_weight = self._prior_sample_weight * np.exp(-1 - a @ sol.x)
        # Handles numerical precision errors
        sample_weight = np.clip(sample_weight, 0, 1)
        sample_weight /= sample_weight.sum()
        return sample_weight

    def _solve_primal(self) -> np.ndarray:
        """Solve the base entropy-pooling problem in its primal form by minimizing KL
        divergence to the prior.

        Returns
        -------
        sample_weight : ndarray of shape (n_observations,)
           The updated posterior probability vector.
        """
        n_observations, _ = self._returns.shape

        solver_params = self.solver_params if self.solver_params is not None else {}

        posterior = cp.Variable(n_observations)  # Posterior probas
        objective = cp.sum(cp.kl_div(posterior, self._prior_sample_weight))
        constraints = [posterior >= 0, cp.sum(posterior) == 1]

        if self._constraints["equality"] is not None:
            a, b = self._constraints["equality"]
            constraints.append(posterior @ a - b == 0)

        if self._constraints["inequality"] is not None:
            a, b = self._constraints["inequality"]
            constraints.append(posterior @ a - b <= 0)

        if self._constraints["cvar_equality"] is not None:
            a, b = self._constraints["cvar_equality"]
            constraints.append(posterior @ a - b == 0)

        if self._constraints["fixed_equality"] is not None:
            a, b = self._constraints["fixed_equality"]
            # Relax the problem with slack variables with a norm1 penalty to avoid
            # solver infeasibility that may arise from overly tight constraints from
            # fixing the moments.
            slack = cp.Variable(b.size)
            constraints.append(posterior @ a - b == slack)
            objective += 1e5 * cp.norm1(slack)

        problem = cp.Problem(cp.Minimize(objective), constraints)
        try:
            # We suppress cvxpy warning as it is redundant with our warning
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem.solve(solver=self.solver, **solver_params)

            if posterior.value is None:
                raise cp.SolverError
            if problem.status != cp.OPTIMAL:
                warnings.warn(
                    "Solution may be inaccurate. Try changing the solver params. For "
                    "more details, set `solver_params=dict(verbose=True)`",
                    stacklevel=2,
                )
            # Handles numerical precision errors
            sample_weight = np.clip(posterior.value, 0, 1)
            sample_weight /= sample_weight.sum()
            return sample_weight
        except (cp.SolverError, scl.ArpackNoConvergence):
            raise SolverError(
                f"Primal problem with Solver '{self.solver}' failed. This typically "
                "occurs when the specified views conflict or are overly extreme. "
                "Consider using a prior that generates more synthetic data for extreme "
                "views. You can also change `solver_params` or try another `solver` "
                "such as 'TNC'."
            ) from None

    def _process_views(self, measure: PerfMeasure | RiskMeasure | ExtraRiskMeasure):
        """Process and convert view equations into constraint matrices.

        This method uses the provided view strings and groups to generate the equality
        and inequality matrices (a_eq, b_eq, a_ineq, b_ineq) needed to formulate the
        constraints. Prior asset expressions are replaced using the current prior.

        Parameters
        ----------
        measure : {PerfMeasure, RiskMeasure, ExtraRiskMeasure}
           The type of view measure to process.

        Returns
        -------
        a_eq, b_eq, a_ineq, b_ineq : tuple of ndarray
           Matrices and vectors corresponding to equality and inequality constraints.
        """
        assets = self._groups[0]
        name = f"{measure.value}_views"
        views = getattr(self, name)
        views = np.asarray(views)
        if views.ndim != 1:
            raise ValueError(f"{name} must be a list of strings")
        required_prior_assets = _extract_prior_assets(views, assets=assets)

        if len(required_prior_assets) != 0:
            prior_values = {}
            if measure == PerfMeasure.MEAN:
                prior_values = {
                    k: self._mu[i]
                    for i, k in enumerate(assets)
                    if k in required_prior_assets
                }
            elif measure == RiskMeasure.VARIANCE:
                prior_values = {
                    k: self._covariance[i, i]
                    for i, k in enumerate(assets)
                    if k in required_prior_assets
                }
            else:
                measure_func = getattr(sm, str(measure.value))
                prior_values = {
                    k: measure_func(self._returns[:, i])
                    for i, k in enumerate(assets)
                    if k in required_prior_assets
                }
            views = _replace_prior_views(views=views, prior_values=prior_values)

        a_eq, b_eq, a_ineq, b_ineq = equations_to_matrix(
            groups=self._groups,
            equations=views,
            raise_if_group_missing=True,
            names=("groups", "views"),
        )

        return a_eq, b_eq, a_ineq, b_ineq

    def _fix_mean(self, fix: np.ndarray, mean: np.ndarray) -> None:
        """Add constraints to fix the mean for assets where view constraints have been
        applied.

        The method introduces slack variables to avoid solver infeasibility from
        conflicting tight constraints.

        Parameters
        ----------
        fix : ndarray of bool of shape (n_assets,)
            Boolean mask indicating which assets to fix.

        mean : ndarray of shape (n_assets,)
            Fixed mean values for the assets.
        """
        fix &= ~self._is_fixed_mean
        if np.any(fix):
            self._add_constraint(
                a=self._returns[:, fix], b=mean[fix], name="fixed_equality"
            )
            self._is_fixed_mean |= fix

    def _fix_variance(
        self, fix: np.ndarray, mean: np.ndarray, variance: np.ndarray
    ) -> None:
        """Add constraints to fix the variance for assets where view constraints have
        been applied.

        The method introduces slack variables to avoid solver infeasibility from
        conflicting tight constraints.

        Parameters
        ----------
        fix : ndarray of bool of shape (n_assets,)
            Boolean mask indicating which assets to fix.

        mean : ndarray of shape (n_assets,)
            Fixed mean values used for the linearization of the variance.

        variance : np.ndarray
            Fixed variance values.
        """
        fix &= ~self._is_fixed_variance
        if np.any(fix):
            self._add_constraint(
                a=(self._returns[:, fix] - mean[fix]) ** 2,
                b=variance[fix],
                name="fixed_equality",
            )
            self._is_fixed_variance |= fix


def _extract_prior_assets(views: Sequence[str], assets: Sequence[str]) -> set[str]:
    """
    Given a list of views, return a set of asset names referenced within any
    'prior(ASSET)' pattern. Only asset names in the provided 'assets' list
    will be recognized.

    Supported format:
      - "prior(ASSET)"
      - "prior(ASSET) * a"
      - "a * prior(ASSET)"

    Parameters
    ----------
    views : list[str]
        The list of views to scan.

    assets : list[str]
        The list of allowed asset names.

    Returns
    -------
    prior_assets : set[str]
        Set of asset names that appear in prior() pattern.
    """
    allowed_assets = "|".join(re.escape(asset) for asset in assets)
    pattern = r"prior\(\s*(" + allowed_assets + r")\s*\)"

    prior_assets = set()
    for view in views:
        matches = re.findall(pattern, view)
        prior_assets.update(matches)

    return set(prior_assets)


def _replace_prior_views(
    views: Sequence[str], prior_values: dict[str, float]
) -> list[str]:
    """
    Replace occurrences of the below prior patterns using `prior_values`.

    Supported patterns:
        - "prior(ASSET)"
        - "prior(ASSET) * a"
        - "a * prior(ASSET)"

    Parameters
    ----------
    views : list[str]
        The list of views.

    prior_values : dict[str, float]
        A dictionary mapping asset names (str) to their prior values.

    Returns
    -------
    views : list[str]
        The views with each prior pattern instance replaced by its computed value.
    """
    # Build a regex pattern for allowed asset names.
    allowed_assets = "|".join(re.escape(asset) for asset in prior_values)

    # Pattern captures:
    #   - An optional multiplier before: ([0-9\.]+)\s*\*\s*
    #   - The prior() function with an allowed asset: prior\(\s*(allowed_asset)\s*\)
    #   - An optional multiplier after: \s*\*\s*([0-9\.]+)
    pattern = (
        r"(?:([0-9\.]+)\s*\*\s*)?"  # Optional pre-multiplier
        r"prior\(\s*(" + allowed_assets + r")\s*\)"  # prior(ASSET) with allowed asset
        r"(?:\s*\*\s*([0-9\.]+))?"  # Optional post-multiplier
    )

    def repl(match) -> str:
        pre_multiplier = float(match.group(1)) if match.group(1) else 1.0
        asset = match.group(2)
        post_multiplier = float(match.group(3)) if match.group(3) else 1.0
        result_value = prior_values[asset] * pre_multiplier * post_multiplier

        # Cast the float to a string making sure to use :16f to avoid scientific notation
        return f"{result_value:.16f}".rstrip("0").rstrip(".")

    new_views = [re.sub(pattern, repl, view) for view in views]

    # After substitution, check for any unresolved 'prior(ASSET)'
    unresolved_pattern = r"prior\(\s*([^)]+?)\s*\)"
    for view in new_views:
        m = re.search(unresolved_pattern, view)
        if m is not None:
            missing_asset = m.group(1)
            raise ValueError(
                "Unresolved 'prior' expression found in view. "
                f"Asset '{missing_asset}' is not available in prior_values."
            )

    return new_views


def _parse_correlation_view(view: str, assets: Sequence[str]) -> dict:
    """
    Parse a correlation view and return its structured representation.

    The view string should follow one of the formats:
      (asset1, asset2) <operator> <float>
      (asset1, asset2) <operator> <float> * prior(asset3, asset4) * <float> + <float>

    Only asset names from the provided `assets` list are accepted.
    If the view contains a "prior(...)" expression, it is replaced by its computed value
    using multipliers and constant additions. If the asset referenced in any prior
    expression is not allowed, an error is raised.

    Parameters
    ----------
    view : str
        The correlation view string.

    assets : list[str]
        A list of allowed asset names.

    Returns
    -------
    views : dict
        A dictionary with keys:
         - "assets": tuple(asset1, asset2)
         - "operator": the comparison operator (==, >=, <=, >, or <)
         - "expression": a dict representing the right-hand side expression.
             * If there is no prior expression, "expression" contains {"constant": value}.
             * If a prior expression is present, it returns a dict with keys:
                   "prior_assets": tuple(asset3, asset4),
                   "multiplier": overall multiplier (default 1.0),
                   "constant": constant term (default 0.0).
    """
    operator_map = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
    }

    # Build a regex pattern that only accepts the allowed asset names.
    allowed_assets = "|".join(re.escape(asset) for asset in assets)

    main_pattern = (
        r"^\(\s*(" + allowed_assets + r")\s*,\s*(" + allowed_assets + r")\s*\)\s*"
        r"(==|>=|<=|>|<)\s*(.+)$"
    )
    main_match = re.match(main_pattern, view)
    if not main_match:
        raise ValueError(
            f"Invalid correlation view format or unknown asset in view: {view}"
        )

    asset1, asset2, operator_str, expression = main_match.groups()
    expression = expression.strip()

    if "prior(" not in expression:
        try:
            constant = float(expression)
        except ValueError as e:
            raise ValueError(
                f"Could not convert constant '{expression}' to float in view: {view}"
            ) from e
        parsed_expression = {"constant": constant}
    else:
        # Pattern to capture:
        #   - An optional pre multiplier followed by "*" (e.g. "2 *")
        #   - The pattern prior( asset3 , asset4 )
        #   - An optional post multiplier preceded by "*" (e.g. "* 3")
        #   - An optional constant preceded by "+" (e.g. "+ 4")
        prior_pattern = (
            r"^(?:\s*([A-Za-z0-9_.-]+)\s*\*\s*)?"  # Optional pre-multiplier
            r"prior\(\s*("
            + allowed_assets
            + r")\s*,\s*("
            + allowed_assets
            + r")\s*\)"  # prior(asset3, asset4)
            r"(?:\s*\*\s*([A-Za-z0-9_.-]+))?"  # Optional post-multiplier
            r"(?:\s*\+\s*([A-Za-z0-9_.-]+))?"  # Optional constant addition
            r"$"
        )
        prior_match = re.match(prior_pattern, expression)
        if not prior_match:
            raise ValueError(
                "Invalid prior expression format or unknown asset in expression: "
                f"{expression}"
            )

        (pre_mult, asset3, asset4, post_mult, constant) = prior_match.groups()
        try:
            pre_mult = float(pre_mult) if pre_mult is not None else 1.0
        except ValueError as e:
            raise ValueError(
                f"Invalid pre-multiplier '{pre_mult}' in view: {view}"
            ) from e
        try:
            post_mult = float(post_mult) if post_mult is not None else 1.0
        except ValueError as e:
            raise ValueError(
                f"Invalid post-multiplier '{post_mult}' in view: {view}"
            ) from e
        try:
            constant = float(constant) if constant is not None else 0.0
        except ValueError as e:
            raise ValueError(f"Invalid constant '{constant}' in view: {view}") from e

        parsed_expression = {
            "prior_assets": (asset3, asset4),
            "multiplier": pre_mult * post_mult,
            "constant": constant,
        }

    return {
        "assets": (asset1, asset2),
        "operator": operator_map[operator_str],
        "expression": parsed_expression,
    }
