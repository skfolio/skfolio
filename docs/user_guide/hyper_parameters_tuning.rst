.. _hyper_parameters_tuning:

.. currentmodule:: skfolio.model_selection

***********************
Hyper-Parameters Tuning
***********************

Hyper-parameters tuning in `skfolio` follows the same API as `scikit-learn`.

Hyper-parameters are parameters that are not directly learnt within estimators.
They are passed as arguments to the constructor of the estimator classes.

It is possible and recommended to search the hyper-parameter space for the
best :ref:`cross validation <cross_validation>` score.

Any parameter provided when constructing an estimator may be optimized in this
manner. Specifically, to find the names and current values for all parameters
for a given estimator, use::

  estimator.get_params()

A search consists of:

- an estimator (such as :class:`~skfolio.optimization.MeanRisk`)
- a parameter space
- a method for searching or sampling candidates
- a cross-validation scheme
- a :ref:`score function <gridsearch_scoring>`



Two generic approaches to parameter search are provided in
scikit-learn: for given values, `GridSearchCV` exhaustively considers
all parameter combinations, while `RandomizedSearchCV` can sample a
given number of candidates from a parameter space with a specified
distribution.

After describing these tools we detail :ref:`best practices
<grid_search_tips>` applicable to these approaches.

Exhaustive Grid Search
**********************

The grid search provided by `GridSearchCV` exhaustively generates
candidates from a grid of parameter values specified with the `param_grid`
parameter. For instance, the following `param_grid`::

    param_grid = [
        {'l1_coef': [0.001, 0.01, 0.1], 'risk_measure': [RiskMeasure.SEMI_VARIANCE]},
        {'l1_coef': [0.001, 0.01, 0.1], 'l2_coef': [0.01, 0.1, 1], 'risk_measure': [RiskMeasure.CVAR]},
    ]

specifies that two grids should be explored: one with a Semi-Variance risk measure and
l1_coef values in [0.001, 0.01, 0.1], and the second one with a CVaR risk measure,
and the cross-product of l1_coef values ranging in [0.001, 0.01, 0.1] and l2_coef
values in  [0.01, 0.1, 1].

The `GridSearchCV` instance implements the usual estimator API: when
"fitting" it on a dataset all the possible combinations of parameter values are
evaluated and the best combination is retained.


**Example:**

.. code-block:: python

    from sklearn.model_selection import GridSearchCV, KFold, train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    param_grid = [
        {'l1_coef': [0.001, 0.01, 0.1], 'risk_measure': [RiskMeasure.SEMI_VARIANCE]},
        {'l1_coef': [0.001, 0.01, 0.1], 'l2_coef': [0.01, 0.1, 1], 'risk_measure': [RiskMeasure.CVAR]},
    ]

    grid_search = GridSearchCV(
        estimator=MeanRisk(min_weights=-1),
        cv=KFold(),
        param_grid=param_grid,
        n_jobs=-1  # using all cores
    )
    grid_search.fit(X_train)
    print(grid_search.cv_results_)

    best_model = grid_search.best_estimator_
    print(best_model.weights_)


.. currentmodule:: sklearn.model_selection

.. _randomized_parameter_search:

Randomized Parameter Optimization
*********************************

While using a grid of parameter settings is currently the most widely used
method for parameter optimization, other search methods have more
favorable properties.
`RandomizedSearchCV` implements a randomized search over parameters,
where each setting is sampled from a distribution over possible parameter values.
This has two main benefits over an exhaustive search:

* A budget can be chosen independent of the number of parameters and possible values.
* Adding parameters that do not influence the performance does not decrease efficiency.

Specifying how parameters should be sampled is done using a dictionary, very
similar to specifying parameters for `GridSearchCV`. Additionally,
a computation budget, being the number of sampled candidates or sampling
iterations, is specified using the `n_iter` parameter.
For each parameter, either a distribution over possible values or a list of
discrete choices (which will be sampled uniformly) can be specified.

In principle, any function can be passed that provides a `rvs` (random
variate sample) method to sample a value. A call to the `rvs` function should
provide independent random samples from possible parameter values on
consecutive calls.

The `scipy.stats` module contains many useful
distributions for sampling parameters, such as `expon`, `gamma`,
`uniform`, `loguniform` or `randint`.

For continuous parameters, such as `l1_coef` above, it is important to specify
a continuous distribution to take full advantage of the randomization. This way,
increasing `n_iter` will always lead to a finer search.

A continuous log-uniform random variable is the continuous version of
a log-spaced parameter. For example to specify the equivalent of `l2_coef` from above,
`loguniform(0.01,  1)` can be used instead of `[0.01, 0.1, 1]`.

Mirroring the example above in grid search, we can specify a continuous random
variable that is log-uniformly distributed between `0.01` and `1`::

    import scipy.stats as stats
    {'l1_coef': stats.loguniform(0.01,  1), 'risk_measure': [RiskMeasure.SEMI_VARIANCE]}

**Example:**

.. code-block:: python

    import scipy.stats as stats
    from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    param_dist = {'l2_coef': stats.loguniform(0.01,  1), 'risk_measure': [RiskMeasure.CVAR]}

    rd_search = RandomizedSearchCV(
        estimator=MeanRisk(min_weights=-1),
        cv=KFold(),
        n_iter=10,
        param_distributions=param_dist,
        n_jobs=-1  # using all cores
    )
    rd_search.fit(X_train)
    print(rd_search.cv_results_)

    best_model = rd_search.best_estimator_
    print(best_model.weights_)

.. _grid_search_tips:

Tips for Parameter Search
*************************

.. _gridsearch_scoring:

Specifying an Objective Metric
------------------------------

By default, all portfolio optimization estimators have the same score function which is
the **Sharpe Ratio**. This score function can be customized with
:func:`~skfolio.metrics.make_scorer` by using another :ref:`measure <measures_ref>` or
by writing your own score function.

**Example:**

In the below example, the Sortino Ratio is used instead of the default Sharpe Ratio:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV, KFold, train_test_split

    from skfolio import RatioMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.metrics import make_scorer
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    scoring = make_scorer(RatioMeasure.SORTINO_RATIO)

    grid_search = GridSearchCV(
        estimator=MeanRisk(min_weights=-1),
        cv=KFold(),
        param_grid={'l2_coef': [0.0001, 0.001,  0.01, 1]},
        scoring=scoring
    )

    grid_search.fit(X_train)
    print(grid_search.cv_results_)

    best_model = grid_search.best_estimator_
    print(best_model.weights_)

    pred = best_model.predict(X_test)
    print(pred.sortino_ratio)


**Example:**

In this example, we use a custom score function:

.. code-block:: python

    def custom_score(pred):
        return pred.mean - 2 * pred.variance - 3 * pred.semi_variance

    scoring = make_scorer(custom_score)


.. _composite_grid_search:

Composite Estimators and Parameter Spaces
-----------------------------------------
`GridSearchCV` and `RandomizedSearchCV` allow searching over
parameters of composite or nested estimators using a dedicated
`<estimator>__<parameter>` syntax.

**Example:**

In the below example, we search the optimal parameter `alpha` of the nested estimator
:class:`~skfolio.moments.EWMu`:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV, KFold, train_test_split

    from skfolio.datasets import load_sp500_dataset
    from skfolio.moments import EWMu
    from skfolio.optimization import MeanRisk, ObjectiveFunction
    from skfolio.preprocessing import prices_to_returns
    from skfolio.prior import EmpiricalPrior

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        prior_estimator=EmpiricalPrior(mu_estimator=EWMu(alpha=0.2)),
    )

    print(model.get_params(deep=True))

    param_grid = {"prior_estimator__mu_estimator__alpha": [0.001, 0.01, 0.01, 0.1]}

    grid_search = GridSearchCV(
        estimator=model,
        cv=KFold(),
        param_grid=param_grid,
    )

    grid_search.fit(X_train)
    print(grid_search.best_estimator_)



**Example:**

The same logic applies for `Pipeline`. Here we search the optimal risk measure of
:class:`~skfolio.optimization.MeanRisk` which is part of a `Pipeline`:

.. code-block:: python

    from sklearn.model_selection import GridSearchCV, KFold, train_test_split
    from sklearn.pipeline import Pipeline

    from skfolio import RiskMeasure
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio.pre_selection import SelectKExtremes
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = Pipeline(
        [
            ("pre_selection", SelectKExtremes(k=10, highest=True)),
            ("optimization", MeanRisk()),
        ]
    )

    param_grid = {
        "optimization__risk_measure": [RiskMeasure.SEMI_VARIANCE, RiskMeasure.CVAR]
    }

    grid_search = GridSearchCV(
        estimator=model,
        cv=KFold(),
        param_grid=param_grid,
    )

    grid_search.fit(X_train)
    print(grid_search.best_estimator_)


Parallelism
-----------

The parameter search tools evaluate each parameter combination on each data
fold independently. Computations can be run in parallel by using the keyword
`n_jobs=-1`. See function signature for more details.



