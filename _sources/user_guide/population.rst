.. _population:

.. currentmodule:: skfolio.population

.. role:: python(code)
   :language: python

==========
Population
==========

A :class:`Population` is a list of portfolios (:class:`~skfolio.portfolio.Portfolio`
or :class:`~skfolio.portfolio.MultiPeriodPortfolio` or both).
`Population` inherits from the build-in `list` class and extend it by adding new
functionalities to improve portfolio manipulation and analysis.


**Example:**

In this example, we create a Population of 100 random Portfolios:

.. code-block:: python

    from skfolio import (
        PerfMeasure,
        Population,
        Portfolio,
        RatioMeasure,
        RiskMeasure,
    )
    from skfolio.datasets import load_sp500_dataset
    from skfolio.preprocessing import prices_to_returns
    from skfolio.utils.stats import rand_weights

    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices)

    population = Population([])

    n_assets = X.shape[1]
    for i in range(100):
        weights = rand_weights(n=n_assets)
        portfolio = Portfolio(X=X, weights=weights, name=str(i))
        population.append(portfolio)


Let's explore some of the methods:

.. code-block:: python

    print(population.composition())

    print(population.summary())

    portfolio = population.quantile(measure=RiskMeasure.VARIANCE, q=0.95)

    population.set_portfolio_params(compounded=True)

    fronts = population.non_denominated_sort()

    population.plot_measures(
        x=RiskMeasure.ANNUALIZED_VARIANCE,
        y=PerfMeasure.ANNUALIZED_MEAN,
        z=RiskMeasure.MAX_DRAWDOWN,
        show_fronts=True,
    )

    population[:2].plot_cumulative_returns()

    population.plot_distribution(
        measure_list=[RatioMeasure.SHARPE_RATIO, RatioMeasure.SORTINO_RATIO]
    )

    population.plot_composition()


A `Population` is returned by the `predict` method of some Optimization estimators that
supports multi-outputs.

For example, fitting :class:`~skfolio.optimization.MeanRisk` with parameter
`efficient_frontier_size=30` will find the weights of 30 portfolios belonging to the
efficient frontier. Calling the method `predict(X_test)` on that model will return a
`Population` containing these 30 `Portfolio` predicted on the test set:

.. code-block:: python

    from sklearn.model_selection import train_test_split

    from skfolio import (
        RiskMeasure,
    )
    from skfolio.datasets import load_sp500_dataset
    from skfolio.optimization import MeanRisk
    from skfolio.preprocessing import prices_to_returns

    prices = load_sp500_dataset()
    X = prices_to_returns(X=prices)
    X_train, X_test = train_test_split(X, test_size=0.33, shuffle=False)

    model = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=30,
    )
    model.fit(X_train)
    print(model.weights_.shape)

    population = model.predict(X_test)


