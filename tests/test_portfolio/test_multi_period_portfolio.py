import datetime as dt

import numpy as np
import pandas as pd
import pytest

import skfolio.measures as mt
from skfolio import (
    ExtraRiskMeasure,
    FailedPortfolio,
    MultiPeriodPortfolio,
    PerfMeasure,
    Portfolio,
    RatioMeasure,
    RiskMeasure,
)
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.utils.stats import rand_weights
from skfolio.utils.tools import args_names


def _portfolio_returns(asset_returns: np.ndarray, weights: np.array) -> np.array:
    r"""
    Compute the portfolio returns from its assets returns and weights.
    """
    n, m = asset_returns.shape
    returns = np.zeros(n)
    for i in range(m):
        returns += asset_returns[:, i] * weights[i]
    return returns


def _dominate(fitness_1: np.ndarray, fitness_2: np.ndarray) -> bool:
    return np.all(fitness_1 >= fitness_2) and np.any(fitness_1 > fitness_2)


@pytest.fixture(scope="module")
def prices():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2017, 1, 1) :]
    return prices


@pytest.fixture(scope="module")
def X(prices):
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="module")
def periods():
    periods = [
        (dt.date(2018, 1, 1), dt.date(2018, 3, 1)),
        (dt.date(2018, 3, 15), dt.date(2018, 5, 1)),
        (dt.date(2018, 5, 1), dt.date(2018, 8, 1)),
    ]
    return periods


@pytest.fixture(scope="module")
def weights():
    weights = [
        np.array(
            [
                0.13045922,
                0.0,
                0.07275738,
                0.0,
                0.0,
                0.0,
                0.0,
                0.10378508,
                0.0,
                0.0,
                0.06514792,
                0.1572522,
                0.0,
                0.04561998,
                0.0,
                0.13172688,
                0.0,
                0.12010884,
                0.06686275,
                0.10627975,
            ]
        ),
        np.array(
            [
                0.16601113,
                0.22600576,
                0.10415873,
                0.15929996,
                0.0,
                0.0,
                0.03297379,
                0.17318809,
                0.0,
                0.01196659,
                0.06460301,
                0.0,
                0.03044458,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.03134836,
                0.0,
            ]
        ),
        np.array(
            [
                0.03336826,
                0.08888789,
                0.0,
                0.11553431,
                0.0,
                0.09538946,
                0.0,
                0.0,
                0.0,
                0.0,
                0.02055812,
                0.13314598,
                0.17740991,
                0.04196778,
                0.0,
                0.0,
                0.17062177,
                0.0,
                0.12311653,
                0.0,
            ]
        ),
    ]
    return weights


@pytest.fixture(scope="function")
def portfolio_and_returns(prices, periods, weights):
    returns = np.array([])
    portfolios = []
    for i, (period, weight) in enumerate(zip(periods, weights, strict=True)):
        X = prices_to_returns(X=prices[period[0] : period[1]])
        returns = np.concatenate([returns, _portfolio_returns(X.to_numpy(), weight)])
        portfolios.append(
            Portfolio(
                X=X,
                weights=weight,
                previous_weights=weights[i - 1] if i > 0 else None,
                name=f"portfolio_{i}",
            ),
        )
    portfolio = MultiPeriodPortfolio(
        portfolios=portfolios,
        name="mpp",
        tag="my_tag",
    )
    return portfolio, returns


@pytest.fixture(scope="function")
def portfolio_and_returns_with_failed_ptf(prices, periods, weights):
    returns = np.array([])
    portfolios = []
    for i, (period, weight) in enumerate(zip(periods, weights, strict=True)):
        X = prices_to_returns(X=prices[period[0] : period[1]])
        if i != 1:
            returns = np.concatenate(
                [returns, _portfolio_returns(X.to_numpy(), weight)]
            )
            portfolios.append(Portfolio(X=X, weights=weight, name=f"portfolio_{i}"))
        else:
            returns = np.concatenate([returns, np.full(len(X), np.nan)])
            portfolios.append(FailedPortfolio(X=X, name=f"failed_portfolio_{i}"))

    portfolio = MultiPeriodPortfolio(
        portfolios=portfolios,
        name="mpp",
        tag="my_tag",
    )
    return portfolio, returns


@pytest.fixture(scope="function")
def portfolio_and_returns_with_full_failed_ptf(prices, periods, weights):
    returns = np.array([])
    portfolios = []
    for i, period in enumerate(periods):
        X = prices_to_returns(X=prices[period[0] : period[1]])
        returns = np.concatenate([returns, np.full(len(X), np.nan)])
        portfolios.append(FailedPortfolio(X=X, name=f"failed_portfolio_{i}"))

    portfolio = MultiPeriodPortfolio(
        portfolios=portfolios,
        name="mpp",
        tag="my_tag",
    )
    return portfolio, returns


@pytest.fixture(scope="function")
def portfolio(portfolio_and_returns):
    portfolio, _ = portfolio_and_returns
    return portfolio


@pytest.fixture(
    scope="module",
    params=list(PerfMeasure)
    + list(RiskMeasure)
    + list(RiskMeasure)
    + list(ExtraRiskMeasure),
)
def measure(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[None, 100, 1],
)
def annualized_factor(request):
    return request.param


def test_portfolio_annualized(portfolio, annualized_factor):
    if annualized_factor is not None:
        portfolio.annualized_factor = annualized_factor

    if annualized_factor is None:
        annualized_factor = 252.0
    assert portfolio.annualized_factor == annualized_factor

    np.testing.assert_almost_equal(
        portfolio.annualized_mean, portfolio.mean * annualized_factor
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_variance, portfolio.variance * annualized_factor
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_semi_variance, portfolio.semi_variance * annualized_factor
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_standard_deviation,
        portfolio.standard_deviation * np.sqrt(annualized_factor),
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_semi_deviation,
        portfolio.semi_deviation * np.sqrt(annualized_factor),
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_sharpe_ratio,
        portfolio.sharpe_ratio * np.sqrt(annualized_factor),
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_sortino_ratio,
        portfolio.sortino_ratio * np.sqrt(annualized_factor),
    )


def test_portfolio_methods(portfolio_and_returns, periods):
    portfolio, returns = portfolio_and_returns
    assert portfolio.n_observations == returns.shape[0]
    assert len(portfolio) == len(periods)
    np.testing.assert_almost_equal(returns, portfolio.returns)
    np.testing.assert_almost_equal(returns.mean(), portfolio.mean)
    np.testing.assert_almost_equal(returns.std(ddof=1), portfolio.standard_deviation)
    np.testing.assert_almost_equal(
        np.sqrt(
            np.sum(np.minimum(0, returns - returns.mean()) ** 2) / (len(returns) - 1)
        ),
        portfolio.semi_deviation,
    )
    np.testing.assert_almost_equal(
        portfolio.mean / portfolio.standard_deviation, portfolio.sharpe_ratio
    )
    np.testing.assert_almost_equal(
        portfolio.mean / portfolio.semi_deviation, portfolio.sortino_ratio
    )
    np.testing.assert_almost_equal(
        portfolio.fitness, np.array([portfolio.mean, -portfolio.variance])
    )
    portfolio.fitness_measures = [PerfMeasure.MEAN, RiskMeasure.SEMI_DEVIATION]
    np.testing.assert_almost_equal(
        portfolio.fitness, np.array([portfolio.mean, -portfolio.semi_deviation])
    )
    portfolio.fitness_measures = [
        PerfMeasure.MEAN,
        RiskMeasure.SEMI_DEVIATION,
        RiskMeasure.MAX_DRAWDOWN,
    ]
    np.testing.assert_almost_equal(
        portfolio.fitness,
        np.array([portfolio.mean, -portfolio.semi_deviation, -portfolio.max_drawdown]),
    )
    assert len(portfolio.assets) == len(periods)
    assert portfolio.composition.shape[1] == len(periods)
    assert isinstance(portfolio.cumulative_returns_df, pd.Series)
    assert isinstance(portfolio.drawdowns_df, pd.Series)
    portfolio.clear()
    assert portfolio.plot_returns()
    assert portfolio.plot_cumulative_returns()
    assert portfolio.plot_drawdowns()
    assert isinstance(portfolio.composition, pd.DataFrame)
    assert portfolio.plot_composition()
    assert isinstance(portfolio.summary(), pd.Series)
    assert isinstance(portfolio.summary(formatted=False), pd.Series)
    assert portfolio.plot_weights_per_observation()
    assert isinstance(portfolio.weights_dict, dict)
    assert isinstance(portfolio.previous_weights_dict, dict)


def test_mpp_magic_methods(portfolio, periods):
    mpp = portfolio
    assert mpp[1] == mpp.portfolios[1]
    for i, p in enumerate(mpp):
        assert p.name == f"portfolio_{i}"
    p_1 = mpp[1]
    assert mpp == mpp
    assert p_1 in mpp
    assert 3 not in mpp
    assert -mpp[1] == -p_1
    assert abs(mpp)[1] == abs(p_1)
    assert round(mpp, 2)[1] == round(p_1, 2)
    assert (mpp + mpp)[1] == p_1 * 2
    assert (mpp - mpp * 0.5)[1] == p_1 * 0.5
    assert (mpp - mpp * 0.4)[1] != p_1 * 0.5
    assert (mpp - mpp * 0.4)[1] != p_1 * 0.5
    assert (mpp / 2)[1] == p_1 * 0.5
    assert (mpp // 2)[1] == p_1 // 2
    del mpp[1]
    assert p_1 not in mpp
    mpp[1] = p_1
    assert p_1 in mpp
    mpp.portfolios = [mpp[0], p_1]
    assert mpp[0] != p_1
    assert mpp[1] == p_1


def test_portfolio_dominate(X):
    n_assets = X.shape[1]
    for _ in range(1000):
        weights_1 = rand_weights(n=n_assets)
        weights_2 = rand_weights(n=n_assets)
        portfolio_1 = Portfolio(
            X=X,
            weights=weights_1,
            fitness_measures=[
                PerfMeasure.MEAN,
                RiskMeasure.SEMI_DEVIATION,
                RiskMeasure.MAX_DRAWDOWN,
            ],
        )
        portfolio_2 = Portfolio(
            X=X,
            weights=weights_2,
            fitness_measures=[
                PerfMeasure.MEAN,
                RiskMeasure.SEMI_DEVIATION,
                RiskMeasure.MAX_DRAWDOWN,
            ],
        )

        # Doesn't dominate itself (same front)
        assert portfolio_1.dominates(portfolio_1) is False
        assert _dominate(
            portfolio_1.fitness, portfolio_2.fitness
        ) == portfolio_1.dominates(portfolio_2)


def test_portfolio_measure(portfolio, measure):
    m = getattr(portfolio, measure.value)
    assert isinstance(m, float)
    assert not np.isnan(m)


def test_portfolio_slots(portfolio):
    for attr in portfolio._slots():
        if attr[0] == "_":
            try:
                getattr(portfolio, attr[1:])
            except AttributeError:
                pass
        getattr(portfolio, attr)


def test_portfolio_clear_cache(portfolio, periods, measure):
    if measure.is_ratio:
        r = measure.linked_risk_measure
    else:
        r = measure
    if r.is_annualized:
        r = r.non_annualized_measure
    func = getattr(mt, r.value)

    args = [
        arg if arg in Portfolio._measure_global_args else f"{r.value}_{arg}"
        for arg in args_names(func)
        if arg not in ["biased", "sample_weight"]
    ]
    args = [arg for arg in args if arg not in Portfolio._read_only_attrs]
    # default
    m = getattr(portfolio, measure.value)
    for arg in args:
        if arg == "drawdowns":
            arg = "compounded"
        if arg == "compounded":
            a = not getattr(portfolio, arg)
        else:
            a = np.random.uniform(0.2, 1)
        setattr(portfolio, arg, a)
        assert getattr(portfolio, arg) == a
        new_m = getattr(portfolio, str(measure.value))
        if measure != ExtraRiskMeasure.VALUE_AT_RISK:
            assert m != new_m
        if isinstance(measure, RatioMeasure):
            assert getattr(portfolio, measure.value) == portfolio.mean / new_m


def test_portfolio_read_only(portfolio, periods):
    for attr in MultiPeriodPortfolio._read_only_attrs:
        try:
            setattr(portfolio, attr, 0)
            raise
        except AttributeError as e:
            assert str(e) == f"can't set attribute '{attr}' because it is read-only"


def test_portfolio_delete_attr(portfolio, periods):
    try:
        delattr(portfolio, "dummy")
        raise
    except AttributeError as e:
        assert str(e) == "`MultiPeriodPortfolio` object has no attribute 'dummy'"


def test_portfolio_summary(portfolio, periods):
    df = portfolio.summary(formatted=False)
    assert df.loc["Number of Portfolios"] == 3.0
    assert df.loc["Number of Failed Portfolios"] == 0
    assert df.loc["Number of Fallback Portfolios"] == 0


def test_portfolio_contribution(portfolio):
    contribution = portfolio.contribution(measure=RiskMeasure.CVAR)
    assert isinstance(contribution, pd.DataFrame)
    assert contribution.shape == (17, 3)

    contribution = portfolio.contribution(
        measure=RiskMeasure.STANDARD_DEVIATION, to_df=False
    )
    assert isinstance(contribution, list)
    assert len(contribution) == 3
    assert contribution[0].shape == (20,)

    assert portfolio.plot_contribution(measure=RiskMeasure.STANDARD_DEVIATION)


def test_weights_per_observation(portfolio):
    df = portfolio.weights_per_observation
    np.testing.assert_array_equal(df.index.values, portfolio.observations)
    assert len(df.columns) == 17
    assert len(set(df.columns)) == 17
    assert portfolio.plot_weights_per_observation()


def test_mpp_with_failed_ptf_methods(portfolio_and_returns_with_failed_ptf, periods, X):
    portfolio, returns = portfolio_and_returns_with_failed_ptf

    assert portfolio.n_failed_portfolios == 1
    assert portfolio.n_fallback_portfolios == 0
    assert portfolio.n_observations == returns.shape[0]
    assert len(portfolio) == len(periods)
    np.testing.assert_almost_equal(returns, portfolio.returns)
    assert np.isnan(portfolio.drawdowns).any()
    assert not np.isnan(portfolio.drawdowns[-1])
    assert np.isnan(portfolio.cumulative_returns).any()
    assert not np.isnan(portfolio.cumulative_returns[-1])

    assert len(portfolio.assets) == len(periods)
    assert portfolio.composition.shape[1] == len(periods)
    assert isinstance(portfolio.cumulative_returns_df, pd.Series)
    assert isinstance(portfolio.drawdowns_df, pd.Series)
    portfolio.clear()
    assert portfolio.plot_returns()
    assert portfolio.plot_cumulative_returns()
    assert portfolio.plot_drawdowns()
    assert isinstance(portfolio.composition, pd.DataFrame)
    assert portfolio.plot_composition()
    assert isinstance(portfolio.summary(), pd.Series)
    summary = portfolio.summary(formatted=False)
    assert summary.loc["Number of Failed Portfolios"] == 1
    assert summary.loc["Number of Fallback Portfolios"] == 0
    assert not np.isnan(summary.values).any()
    assert portfolio.plot_weights_per_observation()
    contrib = portfolio.contribution(measure=RatioMeasure.SHARPE_RATIO)
    assert not np.isnan(contrib).all().all()
    assert np.isnan(contrib["failed_portfolio_1"]).all()


def test_portfolio_measure_nan(portfolio_and_returns_with_failed_ptf, measure):
    portfolio, _ = portfolio_and_returns_with_failed_ptf

    m = getattr(portfolio, measure.value)
    assert isinstance(m, float)
    assert not np.isnan(m)


def test_mpp_with_full_failed_ptf_methods(
    portfolio_and_returns_with_full_failed_ptf, periods, X
):
    portfolio, returns = portfolio_and_returns_with_full_failed_ptf

    assert portfolio.n_failed_portfolios == 3
    assert portfolio.n_fallback_portfolios == 0
    assert portfolio.n_observations == returns.shape[0]
    assert len(portfolio) == len(periods)
    np.testing.assert_almost_equal(returns, portfolio.returns)
    assert np.isnan(portfolio.drawdowns).all()
    assert np.isnan(portfolio.cumulative_returns).all()

    assert len(portfolio.assets) == len(periods)
    assert portfolio.composition.shape[1] == len(periods)
    assert isinstance(portfolio.cumulative_returns_df, pd.Series)
    assert isinstance(portfolio.drawdowns_df, pd.Series)
    portfolio.clear()
    assert portfolio.plot_returns()
    assert portfolio.plot_cumulative_returns()
    assert portfolio.plot_drawdowns()
    assert isinstance(portfolio.composition, pd.DataFrame)
    assert portfolio.plot_composition()
    assert isinstance(portfolio.summary(), pd.Series)
    summary = portfolio.summary(formatted=False)
    assert summary.loc["Number of Failed Portfolios"] == 3
    assert summary.loc["Number of Fallback Portfolios"] == 0
    assert np.isnan(summary.values[:-4]).all()
    assert portfolio.plot_weights_per_observation()
    contrib = portfolio.contribution(measure=RatioMeasure.SHARPE_RATIO)
    assert np.isnan(contrib).all().all()


def test_portfolio_measure_all_nan(portfolio_and_returns_with_full_failed_ptf, measure):
    portfolio, _ = portfolio_and_returns_with_full_failed_ptf

    m = getattr(portfolio, measure.value)
    assert isinstance(m, float)
    assert np.isnan(m)


def test_weight_dict(X, weights, portfolio_and_returns):
    portfolio, _ = portfolio_and_returns

    for i in range(len(weights)):
        np.testing.assert_almost_equal(
            [portfolio.weights_dict[f"portfolio_{i}"][x] for x in X.columns], weights[i]
        )

    for i in range(len(weights)):
        np.testing.assert_almost_equal(
            [portfolio.previous_weights_dict[f"portfolio_{i}"][x] for x in X.columns],
            weights[i - 1] if i > 0 else np.zeros(20),
        )


def test_fallback_portfolios_include_failed(prices, periods, weights):
    # Build three portfolios: one normal without fallback, one normal with fallback,
    # and one FailedPortfolio with fallback. The failed one must be included.
    portfolios = []

    # p0: normal, no fallback
    X0 = prices_to_returns(X=prices[periods[0][0] : periods[0][1]])
    p0 = Portfolio(X=X0, weights=weights[0], name="p0")
    portfolios.append(p0)

    # p1: normal, with fallback_chain
    X1 = prices_to_returns(X=prices[periods[1][0] : periods[1][1]])
    p1 = Portfolio(
        X=X1,
        weights=weights[1],
        name="p1",
        fallback_chain=[("EqualWeighted()", "success")],
    )
    portfolios.append(p1)

    # p2: failed, with fallback_chain
    X2 = prices_to_returns(X=prices[periods[2][0] : periods[2][1]])
    p2 = FailedPortfolio(
        X=X2,
        name="p2_failed",
        fallback_chain=[
            ("MeanVariance()", "solver_error"),
            ("EqualWeighted()", "success"),
        ],
    )
    portfolios.append(p2)

    mpp = MultiPeriodPortfolio(portfolios=portfolios, name="mpp_fallback")

    # failed_portfolios should only contain the FailedPortfolio
    assert mpp.n_failed_portfolios == 1
    assert mpp.failed_portfolios == [p2]

    # fallback_portfolios should include both p1 and p2 (including the failed one)
    assert mpp.n_fallback_portfolios == 2
    fps = mpp.fallback_portfolios
    assert p1 in fps and p2 in fps
    assert len(fps) == 2

    # Summary should reflect counts
    summary = mpp.summary(formatted=False)
    assert summary.loc["Number of Failed Portfolios"] == 1
    assert summary.loc["Number of Fallback Portfolios"] == 2
