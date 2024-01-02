.. _optimization:

.. currentmodule:: skfolio.optimization

============
Optimization
============

The optimization module implements a set of methods intended for portfolio optimization.
They follow the same API as scikit-learn's `estimator`: the `fit` method takes `X` as
the assets returns and stores the portfolio weights in its `weights_` attribute.

`X` can be any array-like structure (numpy array, pandas DataFrame, etc.)

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
    * Group Constrains
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

Every optimization estimator has a parameter named `prior_estimator`.
The :ref:`prior estimator <prior>` fits a :class:`~skfolio.prior.PriorModel` containing
the estimation of assets' expected returns, covariance matrix, returns and Cholesky
decomposition of the covariance. It represents the investor’s prior beliefs about the
model used to estimate such distribution.

The available prior estimators are:

    * :class:`~skfolio.prior.EmpiricalPrior`
    * :class:`~skfolio.prior.BlackLitterman`
    * :class:`~skfolio.prior.FactorModel`

**Example:**

Minimum Variance portfolio using a Factor Model:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio.datasets import load_factors_dataset, load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import FactorModel

    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()

    X, y = prices_to_returns(prices, factor_prices)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    model = MeanRisk(prior_estimator=FactorModel())
    model.fit(X_train, y_train)
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
estimation of the assets' expected reruns and covariance matrix. A Black & Litterman
model is used for the estimation of the factors' expected reruns and covariance matrix,
incorporating the analyst' views on the factors. Finally, the Black & Litterman prior
expected returns are estimated using an equal-weighted market equilibrium with a risk
aversion of 2 and a denoised prior covariance matrix:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio.datasets import load_factors_dataset, load_sp500_dataset
    from skfolio.moments import DenoiseCovariance, EquilibriumMu
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import BlackLitterman, EmpiricalPrior, FactorModel

    prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()

    X, y = prices_to_returns(prices, factor_prices)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

    factor_views = ["MTUM - QUAL == 0.0003 ",
                    "SIZE - USMV == 0.0004",
                    "VLUE == 0.0006"]

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        prior_estimator=FactorModel(
            factor_prior_estimator=BlackLitterman(
                prior_estimator=EmpiricalPrior(
                    mu_estimator=EquilibriumMu(risk_aversion=2),
                    covariance_estimator=DenoiseCovariance()
                ),
                views=factor_views)
        )
    )

    model.fit(X_train, y_train)
    print(model.weights_)

    portfolio = model.predict(X_test)
    print(portfolio.annualized_sharpe_ratio)



Custom Estimator
================
It is very common to use a custom implementation for the prior estimator. For
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
are modeled with an ellipsoidal uncertainty set. This approach is known as worst-case
optimization and falls under the class of robust optimization. It mitigates the
instability that arises from estimation errors of the expected returns.

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
in the dendrogram, minimizing the distance between leafs.

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

It also supports all :ref:`distance estimator <distance>` through the
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
Hierarchical Tree Clustering, KMeans, etc..). For each cluster, the inner-cluster
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

It supports all :ref:`distance estimator <distance>`
and :ref:`clustering estimator <cluster>` (both `skfolio` and `sklearn`)

**Example:**

Nested Clusters Optimization with KMeans as the clustering algorithm, Kendall Distance
as the distance estimator, Minimum Semi-Variance as the inner estimator, and CVaR Risk
Parity as the outer (meta) estimator trained on the out-of-sample estimates from the
KFolds cross-validation and run with parallelization:

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

:class:`StackingOptimization` is an ensemble method that consists in stacking the output
of individual optimization estimators with a final optimization estimator.

The weights are the dot-product of individual estimators' weights with the final
estimator's weights. Stacking allows to use the strength of each individual estimator
by using their output as input of a final estimator.

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
