.. _optimization:

.. currentmodule:: skfolio.optimization

============
Optimization
============

The optimization module implements a set of methods intended for portfolio optimization.
They follow the same API as scikit-learn's `estimator`: the `fit` method takes `X` as
the assets returns and stores the portfolio weights in its `weights_` attribute.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc.)

All optimization inputs (expected returns, covariance, return scenarios) are expressed
in the periodicity of `X`: with daily returns, the optimizer works with daily moments
and scenarios rather than annualized ones. Parameters that share the unit of expected
returns, such as `transaction_costs` and `management_fees`, must be expressed in the
same periodicity. See :ref:`Periodicity Convention <periodicity_convention>` for the
rationale and the cost conversion rules.

Naive Allocation
****************

The naive module implements a set of naive allocations commonly used as benchmarks for
comparing different models:

    * :class:`EqualWeighted`
    * :class:`InverseVolatility`
    * :class:`Random`

**Example:**

Naive inverse-volatility allocation:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import InverseVolatility
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = InverseVolatility()
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)


Mean-Risk Optimization
**********************

The :class:`MeanRisk` estimator can solve the below 4 objective functions:

    * Minimize Risk:

    .. math::   \begin{cases}
                \begin{aligned}
                &\min_{w} & & risk_{i}(w) \\
                &\text{s.t.} & & w^T\mu \ge min\_return \\
                & & & A w \ge b \\
                & & & risk_{j}(w) \le max\_risk_{j} \quad \forall \; j \ne i
                \end{aligned}
                \end{cases}

    * Maximize Expected Return:

    .. math::   \begin{cases}
                \begin{aligned}
                &\max_{w} & & w^T\mu \\
                &\text{s.t.} & & risk_{i}(w) \le max\_risk_{i} \\
                & & & A w \ge b \\
                & & & risk_{j}(w) \le max\_risk_{j} \quad \forall \; j \ne i
                \end{aligned}
                \end{cases}

    * Maximize Utility:

    .. math::   \begin{cases}
                \begin{aligned}
                &\max_{w} & & w^T\mu - \lambda \times risk_{i}(w)\\
                &\text{s.t.} & & risk_{i}(w) \le max\_risk_{i} \\
                & & & w^T\mu \ge min\_return \\
                & & & A w \ge b \\
                & & & risk_{j}(w) \le max\_risk_{j} \quad \forall \; j \ne i
                \end{aligned}
                \end{cases}

    * Maximize Ratio:

    .. math::   \begin{cases}
                \begin{aligned}
                &\max_{w} & & \frac{w^T\mu - r_{f}}{risk_{i}(w)}\\
                &\text{s.t.} & & risk_{i}(w) \le max\_risk_{i} \\
                & & & w^T\mu \ge min\_return \\
                & & & A w \ge b \\
                & & & risk_{j}(w) \le max\_risk_{j} \quad \forall \; j \ne i
                \end{aligned}
                \end{cases}

With :math:`risk_{i}` a risk measure among:

    * Variance
    * Semi-Variance
    * Standard-Deviation
    * Semi-Deviation
    * Mean Absolute Deviation
    * First Lower Partial Moment
    * CVaR (Conditional Value at Risk)
    * EVaR (Entropic Value at Risk)
    * Worst Realization (worst return)
    * CDaR (Conditional Drawdown at Risk)
    * Maximum Drawdown
    * Average Drawdown
    * EDaR (Entropic Drawdown at Risk)
    * Ulcer Index
    * Gini Mean Difference

It supports the following parameters:

    * Weight Constraints
    * Budget Constraints
    * Group Constraints
    * Transaction Costs
    * Management Fees
    * L1 and L2 Regularization
    * Turnover Constraint
    * Tracking Error Constraint
    * Uncertainty Set on Expected Returns
    * Uncertainty Set on Covariance
    * Expected Return Constraints
    * Risk Measure Constraints
    * Custom Objective
    * Custom Constraints
    * Prior Estimator

**Example:**

Maximum Sharpe Ratio portfolio:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
    )
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.sharpe_ratio)

Prior Estimator
===============

Every portfolio optimization has a parameter named `prior_estimator`.
The :ref:`prior estimator <prior>` fits a :class:`~skfolio.prior.ReturnDistribution` containing
estimates of expected asset returns, covariance matrix, returns and Cholesky
decomposition of the covariance. It represents the investor’s prior beliefs about the
model used to estimate such distribution.

When the prior follows the native NaN-aware convention, compatible optimizers solve the
optimization problem on the investable subset and expand `weights_` back to the full
input universe. See :ref:`Missing Data and Changing Universes <missing_data>` for
details.

The available prior estimators are:

    * :class:`~skfolio.prior.EmpiricalPrior`
    * :class:`~skfolio.prior.BlackLitterman`
    * :class:`~skfolio.prior.TimeSeriesFactorModel`

**Example:**

Minimum Variance portfolio using a Factor Model:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio.datasets import load_factors_dataset, load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import TimeSeriesFactorModel

    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()

    X, factors = prices_to_returns(prices, factor_prices)
    X_train, X_test, factors_train, factors_test = train_test_split(X, factors, test_size=0.33, shuffle=False)

    model = MeanRisk(prior_estimator=TimeSeriesFactorModel())
    model.fit(X_train, factors=factors_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)



Combining Prior Estimators
==========================

Prior estimators can be combined together, making it possible to design complex models:

**Example:**

This example is **purposely complex** to demonstrate how multiple estimators can be
combined.

The model below is a Maximum Sharpe Ratio optimization using a Factor Model for the
estimation of the **assets** expected returns and covariance matrix. A Black & Litterman
model is used for the estimation of the **factors** expected returns and covariance matrix,
incorporating the analysts' views on the factors. Finally, the Black & Litterman prior
expected returns are estimated using an equal-weighted market equilibrium with a risk
aversion of 2 and a denoised prior covariance matrix:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio.datasets import load_factors_dataset, load_sp500_dataset
    from skfolio.moments import DenoiseCovariance, EquilibriumMu
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import BlackLitterman, EmpiricalPrior, TimeSeriesFactorModel

    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()

    X, factors = prices_to_returns(prices, factor_prices)
    X_train, X_test, factors_train, factors_test = train_test_split(X, factors, test_size=0.33, shuffle=False)

    factor_views = ["MTUM - QUAL == 0.0003 ",
                    "SIZE - USMV == 0.0004",
                    "VLUE == 0.0006"]

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        prior_estimator=TimeSeriesFactorModel(
            factor_prior_estimator=BlackLitterman(
                prior_estimator=EmpiricalPrior(
                    mu_estimator=EquilibriumMu(risk_aversion=2),
                    covariance_estimator=DenoiseCovariance()
                ),
                views=factor_views)
        )
    )

    model.fit(X_train, factors=factors_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)



Custom Estimator
================
It is very common to use a custom implementation for the moments estimators. For
example, you may want to use an in-house estimation for the covariance or a predictive
model for the expected returns.

Below is a simple example of how you would implement a custom covariance estimator.
For more complex cases and estimators, check the :ref:`API Reference <api>`.

.. code-block:: python

    import numpy as np

    from skfolio.datasets import load_sp500_dataset
    from skfolio.moments import BaseCovariance
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import EmpiricalPrior

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)


    class MyCustomCovariance(BaseCovariance):
        def __init__(self, my_param=0):
            super().__init__()
            self.my_param = my_param

        def fit(self, X, y=None):
            X = self._validate_data(X)
            # Your custom implementation goes here
            covariance = np.cov(X.T, ddof=self.my_param)
            self._set_covariance(covariance)
            return self


    model = MeanRisk(
        prior_estimator=EmpiricalPrior(covariance_estimator=MyCustomCovariance(my_param=1)),
    )
    model.fit(X)



Worst-Case Optimization
=======================
With the `mu_uncertainty_set_estimator` parameter, the expected returns of the assets
are modeled with a :ref:`norm-ball uncertainty set <uncertainty_set_estimator>`. This
approach is known as worst-case optimization and falls under the class of robust
optimization. It mitigates the instability that arises from estimation errors of the
expected returns.

**Example:**

Worst-case maximum Mean/CDaR ratio (Conditional Drawdown at Risk) with an ellipsoidal
uncertainty set for the expected returns of the assets:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.preprocessing import prices_to_returns
    from skfolio.uncertainty_set import BootstrapMuUncertaintySet

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.CDAR,
        mu_uncertainty_set_estimator=BootstrapMuUncertaintySet(confidence_level=0.9),
    )
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)
    print(portfolio.cdar_ratio)

Covariance uncertainty is configured with `covariance_uncertainty_set_estimator`.
It is applied to the variance risk measure or a `max_variance` constraint. Generic
estimators use a lifted semidefinite formulation, while
:class:`~skfolio.uncertainty_set.OrthogonalCovarianceUncertaintySet` uses a compact
representation in the factor model's orthogonal space.


Going Further
=============
You can explore the remaining parameters (constraints, L1 and L2 regularization, costs,
turnover, tracking error, etc.) with the
:ref:`Mean-Risk examples <mean_risk_examples>` and the :class:`MeanRisk` API.

Risk Budgeting
**************

The :class:`RiskBudgeting` solves the below convex problem:

    .. math::   \begin{cases}
                \begin{aligned}
                &\min_{w} & & risk_{i}(w) \\
                &\text{s.t.} & & b^T log(w) \ge c \\
                & & & w^T\mu \ge min\_return \\
                & & & A w \ge b \\
                & & & w \ge0
                \end{aligned}
                \end{cases}

with :math:`b` the risk budget vector and :math:`c` an auxiliary variable of the log
barrier.

And :math:`risk_{i}` a risk measure among:

    * Variance
    * Semi-Variance
    * Standard-Deviation
    * Semi-Deviation
    * Mean Absolute Deviation
    * First Lower Partial Moment
    * CVaR (Conditional Value at Risk)
    * EVaR (Entropic Value at Risk)
    * Worst Realization (worst return)
    * CDaR (Conditional Drawdown at Risk)
    * Maximum Drawdown
    * Average Drawdown
    * EDaR (Entropic Drawdown at Risk)
    * Ulcer Index
    * Gini Mean Difference
    * First Lower Partial Moment

It supports the following parameters:

    * Weight Constraints
    * Budget Constraints
    * Group Constrains
    * Transaction Costs
    * Management Fees
    * Expected Return Constraints
    * Custom Objective
    * Custom constraints
    * Prior Estimator

Limitations are imposed on certain constraints, such as long-only weights, to ensure the
problem remains convex.

**Example:**

CVaR (Conditional Value at Risk) Risk Parity portfolio:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import RiskBudgeting
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = RiskBudgeting(risk_measure=RiskMeasure.CVAR)
    model.fit(X_train)
    print(model.weights_)

    portfolio_train = model.predict(X_train)
    print(portfolio_train.annualized_sharpe_ratio)
    print(portfolio_train.contribution(measure=RiskMeasure.CVAR))

    portfolio_test = model.predict(X_test)
    print(portfolio_test.annualized_sharpe_ratio)
    print(portfolio_test.contribution(measure=RiskMeasure.CVAR))


Maximum Diversification
***********************

The :class:`MaximumDiversification` maximizes the diversification ratio, which is the
ratio of the weighted volatilities over the total volatility.

**Example:**

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MaximumDiversification
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = MaximumDiversification()
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.diversification)



Distributionally Robust CVaR
****************************

The :class:`DistributionallyRobustCVaR` constructs a Wasserstein ball in the space of
multivariate and non-discrete probability distributions centered at the uniform
distribution on the training samples and finds the allocation that minimizes the CVaR
of the worst-case distribution within this Wasserstein ball.
Esfahani and Kuhn proved that for piecewise linear objective functions,
which is the case of CVaR, the distributionally robust optimization problem
over a Wasserstein ball can be reformulated as finite convex programs.

A solver like `Mosek` that can handle a high number of constraints is preferred.

**Example:**

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import DistributionallyRobustCVaR
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X = X["2020":]
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = DistributionallyRobustCVaR(wasserstein_ball_radius=0.01)
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.cvar)


Hierarchical Risk Parity
************************

The :class:`HierarchicalRiskParity` (HRP) is a portfolio optimization method developed
by Marcos Lopez de Prado.

This algorithm uses a distance matrix to compute hierarchical clusters using the
Hierarchical Tree Clustering algorithm then employs seriation to rearrange the assets
in the dendrogram, minimizing the distance between leaves.

The final step is the recursive bisection where each cluster is split between two
sub-clusters by starting with the topmost cluster and traversing in a top-down
manner. For each sub-cluster, we compute the total cluster risk of an inverse-risk
allocation. A weighting factor is then computed from these two sub-cluster risks,
which is used to update the cluster weight.

.. note ::

    The original paper uses the variance as the risk measure and the single-linkage
    method for the Hierarchical Tree Clustering algorithm. Here we generalize it to
    multiple risk measures and linkage methods.
    The default linkage method is set to the Ward
    variance minimization algorithm, which is more stable and has better properties
    than the single-linkage method.


It supports all :ref:`prior estimators <prior>` and :ref:`risk measures <measures_ref>`
as well as weight constraints.

It also supports all :ref:`distance estimators <distance>` through the
`distance_estimator` parameter. It fits a distance model for the
estimation of the codependence and the distance matrix used to compute the linkage
matrix:

    * :class:`~skfolio.distance.PearsonDistance`
    * :class:`~skfolio.distance.KendallDistance`
    * :class:`~skfolio.distance.SpearmanDistance`
    * :class:`~skfolio.distance.CovarianceDistance`
    * :class:`~skfolio.distance.DistanceCorrelation`
    * :class:`~skfolio.distance.MutualInformation`

**Example:**

Hierarchical Risk Parity with semi (downside) standard-deviation as the risk measure and
mutual information as the distance estimator:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.distance import MutualInformation
    from skfolio.optimization import HierarchicalRiskParity
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = HierarchicalRiskParity(
        risk_measure=RiskMeasure.SEMI_DEVIATION, distance_estimator=MutualInformation()
    )
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)
    print(portfolio.contribution(measure=RiskMeasure.SEMI_DEVIATION))


.. _asset_seriation:

Seriation for HRP and Schur
***************************

:class:`~skfolio.optimization.HierarchicalRiskParity` and
:class:`~skfolio.optimization.SchurComplementary` accept a `seriation_estimator`
to choose the asset order used by recursive allocation. The default,
:class:`~skfolio.seriation.HierarchicalSeriation`, preserves Ward linkage and
optimal leaf ordering. :class:`~skfolio.seriation.SpectralSeriation` orders a
distance matrix using a spectral coordinate and aligns its orientation across
`partial_fit` calls.

.. figure:: /_static/seriation/distance_inputs.svg
   :alt: The optimizer's input X goes to the prior and, optionally, the distance
         estimator. One of three routes supplies distances for seriation: prior
         return scenarios, prior covariance, or X. Allocation always uses the
         prior's moments and return scenarios.
   :width: 100%
   :align: center

   Choose one of the three routes to compute distances. The prior supplies the
   allocation inputs in every case. Online compatibility is described below.

The distance estimator can use three different inputs:

* **Prior return scenarios (default).** With `distance_from_prior=True`, the
  distance estimator is fitted on the return scenarios produced by the prior.
  This accepts any distance estimator fitted from returns, such as `PearsonDistance`,
  `SpearmanDistance` or `CovarianceDistance` configured with a covariance estimator.

* **Prior covariance.** Set
  `distance_estimator=CovarianceDistance(covariance_estimator="precomputed")`
  to convert the prior's covariance into distances without estimating another
  covariance. This configuration requires `distance_from_prior=True` for both
  `fit` and `partial_fit`.

* **Input returns.** Set `distance_from_prior=False` to fit the distance estimator
  on the `X` argument supplied to the optimization estimator's `fit(X)` or
  `partial_fit(X)`, before the prior processes it. In a pipeline, `X` is the output
  of the earlier steps. For `fit`, the same distance estimators as in the prior
  return scenarios option are supported. For `partial_fit`, the distance must
  support incremental learning, as described below.

In all three cases, portfolio allocation uses the prior's moments and return
scenarios.

For example, a factor-model prior can supply the allocation covariance while a
separate covariance estimator measures dependence from input returns:

.. code-block:: python

    from skfolio.distance import CovarianceDistance
    from skfolio.moments import EWCovariance
    from skfolio.optimization import SchurComplementary
    from skfolio.seriation import SpectralSeriation

    model = SchurComplementary(
        prior_estimator=factor_prior,
        distance_estimator=CovarianceDistance(EWCovariance()),
        distance_from_prior=False,
        seriation_estimator=SpectralSeriation(),
    )
    model.fit(X)

Here `factor_prior` is a separately configured factor model.

Custom distances inherit from :class:`~skfolio.distance.BaseDistance`. Override
its read-only `requires_covariance_input` property to return `True` when `X` must
contain covariance. The property must reflect the current parameters before
fitting. The base class derives the scikit-learn `pairwise` input tag from it.
HRP and Schur supply the prior covariance to these distances, which require
`distance_from_prior=True`.

See :ref:`seriation` for the available algorithms, input requirements and online
updates, and :ref:`seriation_turnover` for their potential effect on turnover.

HERC and NCO use clustering estimators because their allocations require clusters.

Online Learning
===============

HRP and Schur support `partial_fit` when their prior and its components support
incremental learning. Each call updates the prior once from new observations, then
computes distances, seriation and portfolio weights. Distance updates depend on the
chosen input:

* With `distance_from_prior=True`, a distance estimated from prior return scenarios
  is fitted afresh on the current scenarios. This supports batch-only distances,
  such as `PearsonDistance`, and priors that truncate or regenerate scenarios.
* With `CovarianceDistance("precomputed")`, the optimizer calls the distance's `fit`
  on the current prior covariance. No additional covariance is estimated, and the
  distance itself does not need to support `partial_fit`.
* With `distance_from_prior=False`, the distance must support `partial_fit`, for
  example `CovarianceDistance(EWCovariance())`. It consumes each new batch once,
  including observations for assets that are not yet investable. Batch-only
  configurations, such as `PearsonDistance` and `CovarianceDistance()` with its
  default `GerberCovariance`, require `distance_from_prior=True` during online
  learning.

The optimizer updates `SpectralSeriation` through `partial_fit` and refits
`HierarchicalSeriation` through `fit`. Refitting the distance does not reset the
spectral estimator's history. See :ref:`seriation` for their respective behavior.

The prior needs enough warm-up observations to produce valid moments. A separate
distance estimator must also provide the required distances for every investable
asset.

With the default `PearsonDistance`, update cost grows with the prior's stored history.
For long runs, `CovarianceDistance("precomputed")` avoids repeated estimation, or the
prior's `max_history` can limit the number of scenarios.

For a worked example with late listings, delistings, holidays and asset warm-up,
see :ref:`sphx_glr_auto_examples_online_learning_plot_online_schur_changing_universe.py`.

Hierarchical Equal Risk Contribution
************************************

The :class:`HierarchicalEqualRiskContribution` (HERC) is a portfolio optimization method
developed by Thomas Raffinot.

This algorithm uses a distance matrix to compute hierarchical clusters using the
Hierarchical Tree Clustering algorithm. It then computes, for each cluster, the total
cluster risk of an inverse-risk allocation.

The final step is the top-down recursive division of the dendrogram, where the assets
weights are updated using a naive risk parity within clusters.

It differs from the Hierarchical Risk Parity by exploiting the dendrogram shape
during the top-down recursive division instead of bisecting it.

.. note ::

    The default linkage method is set to the Ward
    variance minimization algorithm, which is more stable and has better properties
    than the single-linkage method.


It supports all :ref:`prior estimators <prior>` and :ref:`risk measures <measures_ref>`
as well as weight constraints.

It also supports all :ref:`distance estimators <distance>` through the
`distance_estimator` parameter. It fits a distance model for the
estimation of the codependence and the distance matrix used to compute the linkage
matrix:

    * :class:`~skfolio.distance.PearsonDistance`
    * :class:`~skfolio.distance.KendallDistance`
    * :class:`~skfolio.distance.SpearmanDistance`
    * :class:`~skfolio.distance.CovarianceDistance`
    * :class:`~skfolio.distance.DistanceCorrelation`
    * :class:`~skfolio.distance.MutualInformation`

**Example:**

Hierarchical Equal Risk Contribution with CVaR (Conditional Value at Risk) as the risk
measure and mutual information as the distance estimator:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.distance import MutualInformation
    from skfolio.optimization import HierarchicalEqualRiskContribution
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = HierarchicalEqualRiskContribution(
        risk_measure=RiskMeasure.CVAR,
        distance_estimator = MutualInformation()
    )
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)
    print(portfolio.contribution(measure=RiskMeasure.CVAR))


Nested Clusters Optimization
****************************

The :class:`NestedClustersOptimization` (NCO) is a portfolio optimization method
developed by Marcos Lopez de Prado.

It uses a distance matrix to compute clusters using a clustering algorithm (
Hierarchical Tree Clustering, KMeans, etc.). For each cluster, the inner-cluster
weights are computed by fitting the inner-estimator on each cluster using the whole
training data. Then the outer-cluster weights are computed by training the
outer-estimator using out-of-sample estimates of the inner-estimators with
cross-validation. Finally, the final assets weights are the dot-product of the
inner-weights and outer-weights.

.. note ::

    The original paper uses KMeans as the clustering algorithm, minimum Variance for
    the inner-estimator and equal-weighted for the outer-estimator. Here we generalize
    it to all `sklearn` and `skfolio` clustering algorithms (Hierarchical Tree
    Clustering, KMeans, etc.), all portfolio optimizations (Mean-Variance, HRP, etc.)
    and risk measures (variance, CVaR, etc.).
    To avoid data leakage at the outer-estimator, we use out-of-sample estimates to
    fit the outer estimator.

It supports all :ref:`distance estimators <distance>`
and :ref:`clustering estimator <cluster>` (both `skfolio` and `sklearn`)

**Example:**

Nested Clusters Optimization with KMeans as the clustering algorithm, Kendall Distance
as the distance estimator, Minimum Semi-Variance as the inner estimator, and CVaR Risk
Parity as the outer (meta) estimator trained on the out-of-sample estimates from the
KFold cross-validation and run with parallelization:

.. code-block:: python

    from sklearn.cluster import KMeans
    from sklearn.model_selection import KFold, train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.distance import KendallDistance
    from skfolio.optimization import MeanRisk, NestedClustersOptimization, RiskBudgeting
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = NestedClustersOptimization(
        inner_estimator=MeanRisk(risk_measure=RiskMeasure.SEMI_VARIANCE),
        outer_estimator=RiskBudgeting(risk_measure=RiskMeasure.CVAR),
        distance_estimator=KendallDistance(),
        clustering_estimator=KMeans(n_init="auto"),
        cv=KFold(),
        n_jobs=-1,
    )
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)
    print(portfolio.contribution(measure=RiskMeasure.CVAR))


The `cv` parameter can also be a combinatorial cross-validation, such as
:class:`CombinatorialPurgedCV`, in which case each cluster's
out-of-sample outputs are a collection of multiple paths instead of one single path.
The selected out-of-sample path among this collection of paths is chosen according to
the `quantile` and `quantile_measure` parameters.

Stacking Optimization
*********************

:class:`StackingOptimization` is an ensemble method that consists of stacking the outputs
of individual portfolio optimizations with a final portfolio optimization.

The final weights are the dot product of the individual optimizations' weights and the final
optimization's weights.

Stacking leverages the strengths of each individual portfolio optimization by
using their outputs as inputs to a final portfolio optimization.

To avoid data leakage, out-of-sample estimates are used to fit the outer
optimization.

**Example:**

Stacking Optimization with Minimum Semi-Variance and CVaR Risk Parity
stacked together using Minimum Variance as the final (meta) estimator.

.. code-block:: python

    from sklearn.model_selection import KFold, train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk, RiskBudgeting, StackingOptimization
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    estimators = [
        ('model1', MeanRisk(risk_measure=RiskMeasure.SEMI_VARIANCE)),
        ('model2', RiskBudgeting(risk_measure=RiskMeasure.CVAR))
    ]

    model = StackingOptimization(
        estimators=estimators,
        final_estimator=MeanRisk(),
        cv=KFold(),
        n_jobs=-1
    )
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)


The `cv` parameter can also be a combinatorial cross-validation, such as
:class:`CombinatorialPurgedCV`, in which case each out-of-sample outputs are a
collection of multiple paths instead of one single path. The selected out-of-sample path
among this collection of paths is chosen according to the `quantile` and
`quantile_measure` parameters.

.. _tracking_error_optimization:

Tracking Error Optimization
****************************

Tracking error measures the deviation between a portfolio's performance and a benchmark.
`skfolio` provides three approaches for tracking error optimization:

1. **Return-based tracking error constraint** (via `max_tracking_error`):
   Constrains the tracking error while optimizing another objective (e.g., minimize CVaR).

2. **Weight-based target** (via `target_weights`):
   Minimizes tracking error by finding weights that minimize deviation from a target
   portfolio allocation.

3. **Return-based target** (via :class:`BenchmarkTracker`):
   Minimizes tracking error by optimizing on excess returns (portfolio returns
   minus benchmark returns).

**Example 1: Return-based tracking error constraint**

Minimize CVaR while constraining the tracking error to 0.30% vs a benchmark:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset, load_sp500_index
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    spx_prices = load_sp500_index()

    X, y = prices_to_returns(prices, spx_prices)
    X_train, X_test, factors_train, factors_test = train_test_split(X, factors, test_size=0.33, shuffle=False)

    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.CVAR,
        max_tracking_error=0.003,  # 0.30% tracking error constraint
    )
    model.fit(X_train, factors=factors_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.cvar)

**Example 2: Weight-based target**

Minimize tracking error vs an equal-weighted target portfolio:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    import numpy as np
    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()

    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    # Define target portfolio (e.g., equal-weighted)
    n_assets = X.shape[1]
    target_weights = np.ones(n_assets) / n_assets

    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        target_weights=target_weights,
    )
    model.fit(X_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)

**Example 3: Return-based target**

Minimize tracking error vs a benchmark's returns:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset, load_sp500_index
    from skfolio.optimization import BenchmarkTracker
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    benchmark_prices = load_sp500_index()

    X, y = prices_to_returns(prices, benchmark_prices)
    X_train, X_test, factors_train, factors_test = train_test_split(
        X, y["SP500"], test_size=0.33, shuffle=False
    )

    model = BenchmarkTracker(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
    )
    model.fit(X_train, factors=factors_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    # Compare portfolio returns to benchmark
    excess_returns = portfolio.returns - y_test.values
    tracking_error = np.std(excess_returns, ddof=1)
    print(f"Tracking Error: {tracking_error:0.2%}")

.. _optimization_fallbacks:

Fallbacks
*********

Optimization can sometimes fail during a given rebalancing. For example, a convex
mean-variance problem with strict risk or sector constraints may become infeasible on
specific dates.

All optimization estimators accept a `fallback` parameter. It can be None, a fallback
estimator, `"previous_weights"`, or a list combining estimators and
`"previous_weights"`. If the primary estimator raises an error during `fit`, the
configured fallbacks are tried in order until one succeeds. This includes errors from
input validation or from fitting the prior.

Each fallback estimator is cloned and fitted independently on the original inputs. Its
weights, asset names and asset count are copied to the primary estimator, so `fit` still
returns the original estimator instance. The fitted fallback is available through
`fallback_`. Its estimates remain separate from those of the primary estimator.

The `"previous_weights"` fallback reuses the allocation supplied through
`previous_weights`. If the current investable universe is known, assets outside it
receive zero weight. The remaining weights stay unchanged, without rescaling or checking
them against the primary optimizer's constraints, so part of the portfolio may remain in
cash. In a manual fitting loop, the caller supplies the holdings to retain through
`previous_weights` before each call.

Each attempt is recorded in `fallback_chain_`. After a successful fallback,
`fallback_` contains the fitted estimator or the string `"previous_weights"`.

This mechanism allows automated production pipelines to continue when the primary
optimization fails and a fallback succeeds. The configured fallback sequence supports
reproducibility, while the recorded attempts provide traceability and auditability.
Fallbacks can also switch allocation methods or deliberately relax constraints in a
controlled sequence when the original problem is infeasible or a solver cannot converge.

If every fallback fails, `raise_on_failure=True` raises the final error. With
`raise_on_failure=False`, the estimator emits a warning and sets `weights_` to None,
so subsequent calls to `predict` return a :class:`~skfolio.portfolio.FailedPortfolio`.
See :ref:`optimization_failure_handling`.

Example: The primary model is a minimum-variance optimization made intentionally
infeasible (the assets' minimum weights are set to 10%, which exceeds the feasible
upper bound of 1/n_assets = 5%). As a fallback, we provide a feasible minimum-variance
model with a 2% minimum weight constraint:

.. code-block:: python

    model = MeanRisk(
        min_weights=0.1,  # intentionally infeasible
        fallback=MeanRisk(min_weights=0.02),  # feasible fallback
    )
    model.fit(X_train)
    print(model.weights_)

    # Let's retrieve the fitted fallback that produced the final result:
    print(model.fallback_)
    # Let's display the sequence of attempts and their outcomes:
    print(model.fallback_chain_)
    # The fallback audit trail is also propagated to the predicted portfolio:
    portfolio = model.predict(X_test)
    assert portfolio.fallback_chain == model.fallback_chain_

When calling `predict`, the selected fallback and the full attempt log are propagated
to the resulting portfolio via `fallback_chain`.


For a step-by-step tutorial and more details, see
:ref:`sphx_glr_auto_examples_mean_risk_plot_17_failure_and_fallbacks.py`.


.. _optimization_failure_handling:

Failure Handling
****************
In research, cross-validation and hyperparameter tuning (e.g., walk-forward, multiple
randomized cross-validation), it's often useful to let all runs complete while keeping
a full record of failures instead of stopping on the first failed rebalancing.

During `fit`, the configured fallbacks can handle input validation errors, failures
while fitting the prior or other estimators, and optimization failures. If a fallback
succeeds, `fit` returns normally with its allocation for either value of
`raise_on_failure`. When no fallback succeeds:

- If `raise_on_failure=True` (default), the final error is raised after all
  configured fallbacks have failed. Without a fallback, this is the primary error.
  This setting is useful in production when the primary optimization or a fallback
  is expected to succeed. After a raised error, further predictions or updates
  require a fresh estimator or a new successful `fit`.
- If `raise_on_failure=False`, the estimator emits a warning and sets `weights_` to
  None. Subsequent calls to `predict` return a
  :class:`~skfolio.portfolio.FailedPortfolio` containing the failure diagnostics.

`FailedPortfolio` behaves like an augmented NaN: it marks a failed period while
retaining diagnostics and compatibility with downstream portfolio analytics.
Research evaluations can finish while preserving the full timeline of successful
and failed rebalances.

The estimator records the outcome in the following attributes:

- `error_` contains the final error message, or None after a successful allocation
  or fallback. For multiple portfolios, it is a list as described below.
- `fallback_` contains the successful fallback estimator or `"previous_weights"`.
  It is None when no fallback succeeds.
- `fallback_chain_` contains the sequence of attempts, starting with the primary
  estimator. Each entry records `"success"` or an error message. This attribute is
  None when no fallback chain was attempted.

During `partial_fit`, the estimator first updates the prior and other estimators, then
computes portfolio weights. The `fallback` and `raise_on_failure` settings apply only to
optimization failures after those updates have completed. Errors from input validation
or from updating the prior and other estimators are always raised. Online fitting
supports only `fallback=None` or `fallback="previous_weights"`, because fallback
estimators have not accumulated the primary model's online history. Estimator fallbacks
are available during batch `fit`, where each one is fitted independently on the supplied
history.

See :ref:`Updates and Failure Handling <online_failure_handling>` for the online
continuation and restart rules.

Expected failures while computing portfolio weights raise
:class:`~skfolio.exceptions.OptimizationError`. Convex optimization steps, including
HERC's constraint adjustment, raise its subtype
:class:`~skfolio.exceptions.ConvexOptimizationError`. Catching `OptimizationError`
allows the same error handler to cover portfolio optimization failures across
estimators. Catching `ConvexOptimizationError` limits it to failures of convex
optimization steps.

A successful batch fallback can provide weights even if the primary model did not finish
fitting. A batch error suppressed with `raise_on_failure=False` can also leave the
primary model incompletely fitted. In both cases, online learning requires a fresh
estimator or a successful `fit` of the primary model before the next `partial_fit`.

Example: proceed without raising and retrieve failure diagnostics

.. code-block:: python

    from skfolio.optimization import MeanRisk

    # Configure an intentionally infeasible problem
    model = MeanRisk(
        min_weights=1.0,
        raise_on_failure=False,  # do not raise; collect diagnostics instead
    )

    model.fit(X_train)  # does not raise; weights_ is None on failure
    print(model.error_)          # stringified error message
    print(model.fallback_chain_) # None because no fallback was configured

    ptf = model.predict(X_test)  # returns a FailedPortfolio sentinel
    print(type(ptf).__name__)    # "FailedPortfolio"
    print(ptf.optimization_error)
    print(ptf.fallback_chain)


.. _optimization_multiple_results:

Multiple Portfolios
===================

Convex optimizers can compute several portfolios in one call, for example along
an efficient frontier or with array-valued constraints. With
`raise_on_failure=False`, successful portfolios are retained and each failed
portfolio is recorded as an all-NaN row in `weights_`. The corresponding `error_`
entry contains its error message, or None for a successful portfolio. `predict`
returns a :class:`~skfolio.population.Population` containing a
:class:`~skfolio.portfolio.FailedPortfolio` for each failed row.

With `raise_on_failure=False`, fallback is attempted only when every requested portfolio
fails. If some portfolios succeed, individual failures remain recorded in the result
without separate fallback attempts. With `raise_on_failure=True`, the first optimization
failure stops the calculation, and the configured fallbacks are tried for the fitting
call as a whole.


For a complete tutorial illustrating failure handling and fallbacks, see
:ref:`sphx_glr_auto_examples_mean_risk_plot_17_failure_and_fallbacks.py`.
