import datetime as dt

import pandas as pd
import pytest

from skfolio import (
    MultiPeriodPortfolio,
    PerfMeasure,
    Population,
    Portfolio,
    RatioMeasure,
    RiskMeasure,
)
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.utils.stats import rand_weights


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2017, 1, 1) :]
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="function")
def population(X):
    # Create a population of portfolios with 3 objectives
    population = Population([])
    n_assets = X.shape[1]
    for i in range(100):
        weights = rand_weights(n=n_assets, zeros=n_assets - 10)
        portfolio = Portfolio(
            X=X,
            weights=weights,
            fitness_measures=[
                PerfMeasure.MEAN,
                RiskMeasure.SEMI_DEVIATION,
                RiskMeasure.MAX_DRAWDOWN,
            ],
            name=str(i),
        )
        population.append(portfolio)
    return population


@pytest.fixture(scope="function")
def small_population(X):
    n_assets = X.shape[1]
    population = Population(
        [
            Portfolio(X=X, weights=rand_weights(n=n_assets, zeros=n_assets - 10))
            for _ in range(10)
        ]
    )
    return population


@pytest.fixture(scope="function")
def multi_period_portfolio(X):
    # Add the multi period portfolio
    periods = [
        (dt.date(2017, 1, 1), dt.date(2017, 3, 1)),
        (dt.date(2017, 3, 15), dt.date(2017, 5, 1)),
        (dt.date(2017, 5, 1), dt.date(2017, 8, 1)),
    ]

    multi_period_portfolio = MultiPeriodPortfolio(
        name="mmp",
        fitness_measures=[
            PerfMeasure.MEAN,
            RiskMeasure.SEMI_DEVIATION,
            RiskMeasure.MAX_DRAWDOWN,
        ],
    )
    n_assets = X.shape[1]
    for i, period in enumerate(periods):
        portfolio = Portfolio(
            X=X[period[0] : period[1]],
            weights=rand_weights(n=n_assets, zeros=n_assets - 5),
            name=f"ptf_period_{i}",
        )
        multi_period_portfolio.append(portfolio)
    return multi_period_portfolio


def test_magic_methods(population):
    assert len(population) == 100
    assert population[0].name == "0"
    assert population[-1].name == "99"
    assert len(population[1:3]) == 2
    for i, ptf in enumerate(population):
        assert ptf.name == str(i)
    ptf = population[5]
    assert ptf in population
    del population[5]
    assert len(population) == 99
    assert ptf not in population
    population.append(ptf)
    assert len(population) == 100
    assert ptf in population
    ptfs = list(population).copy()
    population = ptfs
    assert list(population) == ptfs
    ptfs.append(ptf)
    ptf = population[10]
    population[10] = ptf
    ptf.fitness_measures = [
        PerfMeasure.MEAN,
        RiskMeasure.SEMI_DEVIATION,
        RatioMeasure.SORTINO_RATIO,
    ]
    population.append(ptf)


def test_non_dominated_sorting(population):
    fronts = population.non_denominated_sort()
    assert sorted([i for j in fronts for i in j]) == list(range(len(population)))
    for i, front in enumerate(fronts):
        dominates = False
        if i == len(fronts) - 1:
            dominates = True
        for idx_1 in front:
            for j in range(i + 1, len(fronts)):
                for idx_2 in fronts[j]:
                    assert not population[idx_2].dominates(population[idx_1])
                    if population[idx_1].dominates(population[idx_2]):
                        dominates = True
        assert dominates


@pytest.mark.parametrize("to_surface", [False, True])
def test_population_plot_measures(population, to_surface):
    assert population.plot_measures(
        x=RiskMeasure.SEMI_DEVIATION,
        y=PerfMeasure.MEAN,
        z=RiskMeasure.MAX_DRAWDOWN,
        to_surface=to_surface,
        show_fronts=True,
    )


def test_population_multi_period_portfolio(population, multi_period_portfolio):
    population.append(multi_period_portfolio)
    assert len(population) == 101
    assert population.plot_measures(
        x=RiskMeasure.STANDARD_DEVIATION, y=PerfMeasure.MEAN, show_fronts=True
    )
    assert population.filter(tags="random").plot_measures(
        x=RiskMeasure.STANDARD_DEVIATION,
        y=PerfMeasure.MEAN,
        hover_measures=[RatioMeasure.SHARPE_RATIO],
        title="Portfolios -- with sharpe ration",
    )

    assert (
        population.min_measure(measure=PerfMeasure.MEAN).mean
        <= population.max_measure(measure=PerfMeasure.MEAN).mean
    )

    # composition
    assert isinstance(population.composition(), pd.DataFrame)
    assert population.plot_composition()
    assert population.plot_returns_distribution()


def test_slicing(population, multi_period_portfolio):
    new_population = population[:2]
    assert isinstance(new_population, Population)

    portfolio = population[2]
    assert isinstance(portfolio, Portfolio)

    population.append(multi_period_portfolio)
    mpp = population[-1]
    assert mpp == multi_period_portfolio


def test_population_plot_cumulative_returns(population):
    assert population[:2].plot_cumulative_returns()

    with pytest.raises(ValueError):
        population[:2].plot_cumulative_returns(log_scale=True)

    population.set_portfolio_params(compounded=True)
    assert population[:2].plot_cumulative_returns()
    assert population[:2].plot_cumulative_returns(log_scale=True)


def test_population_rolling_measure(small_population):
    df = small_population.rolling_measure()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == len(small_population)

    assert small_population.plot_rolling_measure(
        measure=RiskMeasure.STANDARD_DEVIATION, window=50
    )


def test_population_filter_chaining(population):
    res = population.filter(names=["1", "2"]).composition()
    assert res.shape[1] == 2


def test_portfolio_contribution(small_population):
    contribution = small_population.contribution(measure=RiskMeasure.CVAR)
    assert isinstance(contribution, pd.DataFrame)
    assert contribution.shape[1] == 10
    assert contribution.shape[0] <= 20

    assert small_population.plot_contribution(measure=RiskMeasure.STANDARD_DEVIATION)
