from __future__ import annotations

import operator

import numpy as np
import pandas as pd
import pytest

from skfolio import (
    FailedPortfolio,
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
    prices = prices.loc[pd.Timestamp(2017, 1, 1) :]
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
            Portfolio(
                X=X, weights=rand_weights(n=n_assets, zeros=n_assets - 10, seed=i)
            )
            for i in range(10)
        ]
    )
    return population


@pytest.fixture(scope="function")
def multi_period_portfolio(X):
    # Add the multi period portfolio
    periods = [
        (pd.Timestamp(2017, 1, 1), pd.Timestamp(2017, 3, 1)),
        (pd.Timestamp(2017, 3, 15), pd.Timestamp(2017, 5, 1)),
        (pd.Timestamp(2017, 5, 1), pd.Timestamp(2017, 8, 1)),
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


@pytest.fixture(scope="function")
def failed_portfolio(X):
    failed_portfolio = FailedPortfolio(X)
    return failed_portfolio


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
    fronts = population.non_dominated_sort()
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


def test_population_collection_contracts(small_population):
    empty = Population([])
    first = small_population[:1]
    second = small_population[1:2]

    assert repr(empty) == "<Population([])>"
    with pytest.raises(TypeError, match="Cannot add a Population"):
        operator.add(empty, [])

    combined = first + second
    assert isinstance(combined, Population)
    assert list(combined) == [first[0], second[0]]

    empty.insert(0, first[0])
    empty.extend(second)
    assert list(empty) == [first[0], second[0]]
    assert empty.set_portfolio_params() is empty

    with pytest.raises(ValueError, match="Invalid parameter"):
        empty.set_portfolio_params(unknown_parameter=True)
    with pytest.raises(TypeError, match="inherit from BasePortfolio"):
        Population([object()])


def test_set_portfolio_params_updates_population_and_returns_self(small_population):
    result = small_population.set_portfolio_params(compounded=True)

    assert result is small_population
    assert all(portfolio.compounded for portfolio in small_population)


def test_population_compounded_validation(small_population):
    with pytest.raises(ValueError, match="population is empty"):
        Population([])._validate_compounded()

    mixed_population = small_population[:2]
    mixed_population[0].compounded = True
    mixed_population[1].compounded = False

    with pytest.raises(ValueError, match="mix of compounded"):
        mixed_population._validate_compounded()


def test_population_sorting_requires_common_fitness_measures(small_population):
    mixed_population = small_population[:2]
    mixed_population[0].fitness_measures = [PerfMeasure.MEAN]

    with pytest.raises(ValueError, match="mixed `fitness_measures`"):
        mixed_population.non_dominated_sort()


def test_deprecated_non_denominated_sort_alias(small_population):
    expected = small_population.non_dominated_sort(first_front_only=True)

    with pytest.warns(FutureWarning, match="non_denominated_sort"):
        result = small_population.non_denominated_sort(first_front_only=True)

    assert result == expected


def test_population_filter_contracts_and_tagged_returns(small_population):
    small_population[0].name = "first"
    small_population[0].tag = "selected"
    small_population[1].name = "second"
    small_population[1].tag = "selected"

    assert small_population.filter() is small_population
    assert small_population.filter(names="first") == [small_population[0]]
    assert small_population.filter(tags="selected") == small_population[:2]
    assert small_population.filter(names="first", tags="selected") == [
        small_population[0]
    ]

    returns = small_population[:1].returns_df()
    assert list(returns.columns) == ["first_selected"]


def test_population_quantile_rejects_values_outside_unit_interval(small_population):
    with pytest.raises(ValueError, match="between 0 and 1"):
        small_population.quantile(RatioMeasure.SHARPE_RATIO, q=-0.1)


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


def test_population_returns(population):
    df = population.returns_df()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == len(population)

    df_no_tag = population.returns_df(use_tag_in_column_name=False)
    assert isinstance(df_no_tag, pd.DataFrame)
    assert df_no_tag.shape[1] == len(population)


def test_population_cumulative_returns(population):
    assert isinstance(population.cumulative_returns_df(), pd.DataFrame)
    assert isinstance(
        population.cumulative_returns_df(use_tag_in_column_name=False), pd.DataFrame
    )
    assert population[:2].plot_cumulative_returns()

    with pytest.raises(ValueError):
        population[:2].plot_cumulative_returns(log_scale=True)

    population.set_portfolio_params(compounded=True)
    assert population[:2].plot_cumulative_returns()
    assert population[:2].plot_cumulative_returns(log_scale=True)


def test_population_drawdowns(population):
    assert isinstance(population.drawdowns_df(), pd.DataFrame)
    assert isinstance(
        population.drawdowns_df(use_tag_in_column_name=False), pd.DataFrame
    )
    assert population[:2].plot_drawdowns()
    population.set_portfolio_params(compounded=True)
    assert population[:2].plot_drawdowns()


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


def test_plot_distribution(population):
    assert population.plot_distribution(measure_list=[RiskMeasure.STANDARD_DEVIATION])
    assert population.plot_distribution(
        measure_list=[RiskMeasure.STANDARD_DEVIATION, RatioMeasure.SHARPE_RATIO]
    )

    for i in range(len(population)):
        population[i].tag = f"Tag_{int(i > len(population) / 2)}"

    assert population.plot_distribution(
        measure_list=[RiskMeasure.STANDARD_DEVIATION], tag_list=["Tag_0", "Tag_1"]
    )
    assert population.plot_distribution(
        measure_list=[RiskMeasure.STANDARD_DEVIATION, RatioMeasure.SHARPE_RATIO],
        tag_list=["Tag_0", "Tag_1"],
    )


def test_boxplot(population):
    assert population.boxplot_measure(measure=RiskMeasure.STANDARD_DEVIATION)

    for i in range(len(population)):
        population[i].tag = f"Tag_{int(i > len(population) / 2)}"

    assert population.boxplot_measure(
        measure=RatioMeasure.SHARPE_RATIO, tag_list=["Tag_0", "Tag_1"]
    )


def test_population_failed_portfolio(small_population, failed_portfolio):
    pop = small_population
    pop[5] = failed_portfolio
    assert len(pop) == 10

    assert not np.isnan(pop.returns_df()).all().all()
    assert not np.isnan(pop.cumulative_returns_df()).all().all()
    assert not np.isnan(pop.drawdowns_df()).all().all()
    fronts = pop.non_dominated_sort()
    assert fronts == [[5, 6, 8], [1, 3, 4], [9, 7], [2], [0]]
    np.testing.assert_almost_equal(
        pop.measures(measure=RatioMeasure.SHARPE_RATIO),
        [
            0.0325762,
            0.05448077,
            0.04138013,
            0.05535799,
            0.05914441,
            np.nan,
            0.05972074,
            0.05036052,
            0.06193033,
            0.04598217,
        ],
    )
    np.testing.assert_almost_equal(
        pop.measures_mean(measure=RatioMeasure.SHARPE_RATIO), 0.051214808
    )
    np.testing.assert_almost_equal(
        pop.measures_std(measure=RatioMeasure.SHARPE_RATIO), 0.0091293978
    )
    assert pop.sort_measure(measure=RatioMeasure.SHARPE_RATIO)
    assert pop.sort_measure(measure=RatioMeasure.SHARPE_RATIO, reverse=True)
    assert pop.quantile(measure=RatioMeasure.SHARPE_RATIO, q=0.05)
    assert pop.min_measure(measure=RatioMeasure.SHARPE_RATIO)
    assert pop.max_measure(measure=RatioMeasure.SHARPE_RATIO)
    assert isinstance(pop.summary(), pd.DataFrame)
    summary = pop.summary(formatted=False)
    assert not np.isnan(summary.values[:, 6:]).any()
    assert np.isnan(summary.values[:-1, 5]).all()

    comp = pop.composition()
    assert not np.isnan(comp.values[:, 6:]).any()
    assert np.isnan(comp.values[:, 5]).all()
    contrib = pop.contribution(measure=RatioMeasure.SHARPE_RATIO)
    assert not np.isnan(contrib.values[:, 6:]).any()
    assert np.isnan(contrib.values[:, 5]).all()
    assert pop.plot_distribution(measure_list=[RatioMeasure.SHARPE_RATIO])
    assert pop.boxplot_measure(measure=RatioMeasure.SHARPE_RATIO)
    assert pop.plot_cumulative_returns()
    assert pop.plot_drawdowns()
    assert pop.plot_composition()
    assert pop.plot_contribution(measure=RatioMeasure.SHARPE_RATIO)
    assert pop.plot_measures(x=PerfMeasure.MEAN, y=RiskMeasure.STANDARD_DEVIATION)
    assert pop.plot_rolling_measure(measure=RatioMeasure.SHARPE_RATIO)
    assert pop.plot_returns_distribution()


def test_population_composition_and_contribution_sub_ptf_names(
    small_population, multi_period_portfolio
):
    small_population.append(multi_period_portfolio)

    comp = small_population.composition(display_sub_ptf_name=True)
    assert all(
        f"{multi_period_portfolio.name}_{p.name}" in comp.columns
        for p in multi_period_portfolio
    )
    # Without sub-portfolio names, the duplicated columns are de-duplicated by
    # `pd.concat` with a positional suffix.
    expected = {multi_period_portfolio.name} | {
        f"{multi_period_portfolio.name}_{i}"
        for i in range(1, len(multi_period_portfolio))
    }
    comp = small_population.composition(display_sub_ptf_name=False)
    assert set(comp.columns) - {p.name for p in small_population[:-1]} == expected

    contrib = small_population.contribution(
        measure=RiskMeasure.VARIANCE, display_sub_ptf_name=True
    )
    assert all(
        f"{multi_period_portfolio.name}_{p.name}" in contrib.columns
        for p in multi_period_portfolio
    )
    contrib = small_population.contribution(
        measure=RiskMeasure.VARIANCE, display_sub_ptf_name=False
    )
    assert set(contrib.columns) - {p.name for p in small_population[:-1]} == expected


def test_population_plot_measures_color_scale_and_tags(X):
    n_assets = X.shape[1]
    population = Population(
        [
            Portfolio(
                X=X,
                weights=rand_weights(n=n_assets, zeros=n_assets - 10, seed=i),
                name=f"ptf_{i}",
                tag="odd" if i % 2 else None,
            )
            for i in range(6)
        ]
    )

    # Tags drive the color when neither fronts nor a color scale are requested and
    # the legend is placed outside the plotting area.
    fig = population.plot_measures(x=RiskMeasure.STANDARD_DEVIATION, y=PerfMeasure.MEAN)
    # Untagged portfolios become the empty tag, and whether plotly gives that
    # group a trace of its own varies by version, so assert on the set of names.
    trace_names = {trace.name for trace in fig.data}
    assert "odd" in trace_names
    assert trace_names <= {"", "odd"}
    assert fig.layout.legend.x == 1.02

    # A measure color scale is added to the hover data, drives the color and moves
    # the legend inside the plotting area.
    fig = population.plot_measures(
        x=RiskMeasure.STANDARD_DEVIATION,
        y=PerfMeasure.MEAN,
        color_scale=RatioMeasure.SHARPE_RATIO,
    )
    assert fig.layout.coloraxis.colorbar.title.text == str(RatioMeasure.SHARPE_RATIO)
    assert fig.layout.legend.x == 0.02
