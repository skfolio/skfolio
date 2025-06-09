import datetime as dt
import pickle
import timeit
import tracemalloc
from copy import copy

import numpy as np
import pandas as pd
import pytest

import skfolio.measures as mt
from skfolio import (
    ExtraRiskMeasure,
    MultiPeriodPortfolio,
    PerfMeasure,
    Portfolio,
    RatioMeasure,
    RiskMeasure,
)
from skfolio.datasets import load_sp500_dataset
from skfolio.portfolio._base import _MEASURES
from skfolio.preprocessing import prices_to_returns
from skfolio.utils.stats import rand_weights
from skfolio.utils.tools import args_names


@pytest.fixture(scope="module")
def X() -> pd.DataFrame:
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2017, 1, 1) :]
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="module")
def weights() -> np.ndarray:
    weights = np.array(
        [
            0.12968013,
            0.09150399,
            0.12715628,
            0.0,
            0.0,
            0.05705225,
            0.0,
            0.0,
            0.1094415,
            0.30989117,
            0.0,
            0.0,
            0.09861857,
            0.0,
            0.0,
            0.00224294,
            0.06412114,
            0.0,
            0.0,
            0.01029202,
        ]
    )
    return weights


@pytest.fixture
def portfolio(X: pd.DataFrame, weights: np.ndarray) -> Portfolio:
    portfolio = Portfolio(X=X, weights=weights, annualized_factor=252)
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


def _portfolio_returns(asset_returns: np.ndarray, weights: np.array) -> np.array:
    r"""
    Compute the portfolio returns from its assets returns and weights.
    """
    n, m = asset_returns.shape
    returns = np.zeros(n)
    for i in range(m):
        returns += asset_returns[:, i] * weights[i]
    return returns


@pytest.fixture(scope="module")
def sample_weight(X):
    rng = np.random.default_rng(42)
    sample_weight = rng.random(len(X))
    sample_weight /= sample_weight.sum()
    return sample_weight


def test_pickle(portfolio):
    portfolio.sharpe_ratio = 5
    pickled = pickle.dumps(portfolio)
    unpickled = pickle.loads(pickled)
    assert unpickled.name == portfolio.name
    assert portfolio.sharpe_ratio != unpickled.sharpe_ratio

    mmp = MultiPeriodPortfolio(portfolios=[portfolio, portfolio])
    pickled = pickle.dumps(mmp)
    unpickled = pickle.loads(pickled)
    assert unpickled.portfolios[0].sharpe_ratio


def test_concatenate(X, weights):
    portfolios = [Portfolio(X=X, weights=weights), Portfolio(X=X, weights=weights)]
    c = np.concatenate(portfolios)
    assert c.shape == (X.shape[0] * 2,)


def _estimate_portfolio_memory(X, weights, n: int) -> float:
    tracemalloc.start()
    tracemalloc.clear_traces()
    start = tracemalloc.get_traced_memory()
    for _ in range(n):
        portfolio = Portfolio(X=X, weights=weights)
        _ = portfolio.returns
        _ = portfolio.standard_deviation
        _ = portfolio.fitness
        _ = portfolio.mean_absolute_deviation_ratio
    end = tracemalloc.get_traced_memory()
    tracemalloc.clear_traces()
    return end[0] - start[0]


def test_garbage_collection(X, weights):
    m1 = _estimate_portfolio_memory(X, weights, n=1)
    m10 = _estimate_portfolio_memory(X, weights, n=10)
    m100 = _estimate_portfolio_memory(X, weights, n=100)
    m1000 = _estimate_portfolio_memory(X, weights, n=1000)

    assert m10 < 2 * m1
    assert m100 < 2 * m1
    assert m1000 < 2 * m1


def test_portfolio_annualized(X, weights, annualized_factor):
    if annualized_factor is None:
        portfolio = Portfolio(X=X, weights=weights)
    else:
        portfolio = Portfolio(X=X, weights=weights, annualized_factor=annualized_factor)

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


def test_portfolio_methods(X, weights):
    portfolio = Portfolio(X=X, weights=weights)
    returns = _portfolio_returns(asset_returns=X.to_numpy(), weights=weights)
    assert portfolio.n_observations == X.shape[0]
    assert portfolio.n_assets == X.shape[1]
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

    assert len(portfolio.nonzero_assets_index) == 10
    assert len(portfolio.nonzero_assets) == 10
    assert len(portfolio.composition) == 10
    idx = np.nonzero(weights)[0]
    np.testing.assert_almost_equal(portfolio.nonzero_assets_index, idx)
    names_1 = np.array(X.columns[idx])
    assert np.array_equal(portfolio.nonzero_assets, names_1)
    names_2 = portfolio.composition.index.to_numpy()
    names_2.sort()
    names_1.sort()
    assert np.array_equal(names_1, names_2)
    portfolio.clear()
    assert portfolio.plot_returns()
    assert portfolio.plot_returns_distribution()
    assert portfolio.plot_cumulative_returns()
    assert portfolio.plot_rolling_measure(measure=RatioMeasure.SHARPE_RATIO, window=20)
    assert isinstance(portfolio.composition, pd.DataFrame)
    assert portfolio.plot_composition()
    assert isinstance(portfolio.summary(), pd.Series)
    assert isinstance(portfolio.summary(formatted=False), pd.Series)
    assert portfolio.get_weight(asset=portfolio.nonzero_assets[5])
    portfolio.annualized_factor = 252
    assert isinstance(portfolio.summary(), pd.Series)


def test_portfolio_magic_methods(X, weights):
    n_assets = X.shape[1]
    ptf_1 = Portfolio(X=X, weights=rand_weights(n=n_assets))
    ptf_2 = Portfolio(X=X, weights=rand_weights(n=n_assets))
    assert ptf_1.n_observations == X.shape[0]
    assert ptf_1.n_assets == X.shape[1]
    ptf = ptf_1 + ptf_2
    assert np.array_equal(ptf.weights, ptf_1.weights + ptf_2.weights)
    ptf = ptf_1 - ptf_2
    assert np.array_equal(ptf.weights, ptf_1.weights - ptf_2.weights)
    ptf = -ptf_1
    assert np.array_equal(ptf.weights, -ptf_1.weights)
    ptf = ptf_1 * 2.3
    assert np.array_equal(ptf.weights, 2.3 * ptf_1.weights)
    ptf = ptf_1 / 2.3
    assert np.array_equal(ptf.weights, ptf_1.weights / 2.3)
    ptf = abs(ptf_1)
    assert np.array_equal(ptf.weights, abs(ptf_1.weights))
    ptf = round(ptf_1, 2)
    assert np.array_equal(ptf.weights, np.round(ptf_1.weights, 2))
    ptf = ptf_1 // 2
    assert np.array_equal(ptf.weights, ptf_1.weights // 2)
    assert ptf_1 == ptf_1
    assert ptf_1 != ptf_2
    assert (ptf_1 > ptf_2) is ptf_1.dominates(ptf_2)
    assert (ptf_1 < ptf_2) is ptf_2.dominates(ptf_1)


def test_portfolio_dominate(X):
    n_assets = X.shape[1]
    for _ in range(1000):
        weights_1 = rand_weights(n=n_assets)
        weights_2 = rand_weights(n=n_assets)
        portfolio_1 = Portfolio(
            weights=weights_1,
            fitness_measures=[
                PerfMeasure.MEAN,
                RiskMeasure.SEMI_DEVIATION,
                RiskMeasure.MAX_DRAWDOWN,
            ],
            X=X,
        )
        portfolio_2 = Portfolio(
            weights=weights_2,
            fitness_measures=[
                PerfMeasure.MEAN,
                RiskMeasure.SEMI_DEVIATION,
                RiskMeasure.MAX_DRAWDOWN,
            ],
            X=X,
        )
        # Doesn't dominate itself (same front)
        assert portfolio_1.dominates(portfolio_1) is False
        assert (
            np.all(portfolio_1.fitness >= portfolio_2.fitness)
            and np.any(portfolio_1.fitness > portfolio_2.fitness)
        ) == portfolio_1.dominates(portfolio_2)


def test_portfolio_metrics(portfolio, measure):
    m = getattr(portfolio, measure.value)
    assert isinstance(m, float)
    assert not np.isnan(m)
    assert portfolio.sric
    assert portfolio.skew
    assert portfolio.kurtosis
    assert portfolio.diversification
    assert portfolio.effective_number_assets


def test_portfolio_effective_number_assets(portfolio):
    np.testing.assert_almost_equal(portfolio.effective_number_assets, 6.00342169912319)


def test_portfolio_sric(portfolio):
    np.testing.assert_almost_equal(portfolio.sric, -0.20309958369097764)


def test_portfolio_diversification(portfolio):
    np.testing.assert_almost_equal(portfolio.diversification, 1.449839842913199)


def test_portfolio_slots(X, weights):
    portfolio = Portfolio(X=X, weights=weights, annualized_factor=252)
    for attr in portfolio._slots():
        if attr[0] == "_":
            try:
                getattr(portfolio, attr[1:])
            except AttributeError:
                pass
        getattr(portfolio, attr)


def test_copy(X, weights):
    portfolio = Portfolio(X=X, weights=weights, annualized_factor=252)
    with pytest.raises(AttributeError):
        _ = portfolio._assets_names
    _ = portfolio.nonzero_assets
    _ = copy(portfolio)


def test_portfolio_cache(X, weights, measure):
    portfolio = Portfolio(X=X, weights=weights, annualized_factor=252)
    # time for accessing cached attributes
    n = int(1e5)
    ref = timeit.timeit(lambda: portfolio.name, number=n) / n
    first_access_time = timeit.timeit(
        lambda: getattr(portfolio, measure.value), number=1
    )
    cached_access_time = (
        timeit.timeit(lambda: getattr(portfolio, measure.value), number=n) / n
    )
    assert first_access_time > 10 * cached_access_time
    assert ref > cached_access_time / 10


def test_portfolio_clear_cache(X, weights, measure):
    portfolio = Portfolio(X=X, weights=weights, annualized_factor=1)
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


def test_portfolio_read_only(X, weights):
    portfolio = Portfolio(X=X, weights=weights, annualized_factor=252)
    for attr in Portfolio._read_only_attrs:
        try:
            setattr(portfolio, attr, 0)
            raise
        except AttributeError as e:
            assert str(e) == f"can't set attribute '{attr}' because it is read-only"


def test_portfolio_delete_attr(X, weights):
    portfolio = Portfolio(X=X, weights=weights, annualized_factor=252)
    try:
        delattr(portfolio, "dummy")
        raise
    except AttributeError as e:
        assert str(e) == "`Portfolio` object has no attribute 'dummy'"


def test_portfolio_rolling_measure(X, weights):
    window = 30
    portfolio = Portfolio(X=X[:50], weights=weights, annualized_factor=252)
    ref = Portfolio(X=X.iloc[50 - window : 50], weights=weights, annualized_factor=252)

    for measure in _MEASURES:
        res = portfolio.rolling_measure(measure=measure, window=30)
        np.testing.assert_almost_equal(res.iloc[-1], getattr(ref, measure.value))


def test_portfolio_expected_returns_from_assets(X, weights):
    portfolio = Portfolio(X=X, weights=weights)
    rets = X.to_numpy()
    mus = np.mean(rets, axis=0)
    ptf_rets = _portfolio_returns(asset_returns=rets, weights=weights)
    np.testing.assert_almost_equal(
        portfolio.expected_returns_from_assets(assets_expected_returns=mus),
        np.mean(ptf_rets),
    )


def test_portfolio_variance_from_assets(X, weights):
    portfolio = Portfolio(X=X, weights=weights)
    rets = X.to_numpy()
    cov = np.cov(rets.T)
    ptf_rets = _portfolio_returns(asset_returns=rets, weights=weights)
    np.testing.assert_almost_equal(
        portfolio.variance_from_assets(assets_covariance=cov), np.var(ptf_rets)
    )


def test_portfolio_plot_cumulative_returns(X, weights):
    portfolio = Portfolio(X=X, weights=weights, annualized_factor=252)

    assert portfolio.plot_cumulative_returns()

    with pytest.raises(ValueError):
        portfolio.plot_cumulative_returns(log_scale=True)

    portfolio.compounded = True
    assert portfolio.plot_cumulative_returns()
    assert portfolio.plot_cumulative_returns(log_scale=True)


def test_portfolio_contribution(portfolio):
    contribution = portfolio.contribution(measure=RiskMeasure.CVAR, to_df=True)
    assert isinstance(contribution, pd.DataFrame)
    assert contribution.shape == (10, 1)
    assert np.isclose(contribution.sum().sum(), portfolio.cvar)

    contribution = portfolio.contribution(measure=RiskMeasure.STANDARD_DEVIATION)
    assert isinstance(contribution, np.ndarray)
    assert contribution.shape == (20,)

    assert np.isclose(np.sum(contribution), portfolio.standard_deviation)

    assert portfolio.plot_contribution(measure=RiskMeasure.STANDARD_DEVIATION)


def test_weights_per_observation(portfolio):
    df = portfolio.weights_per_observation
    np.testing.assert_array_equal(df.index.values, portfolio.observations)
    assert len(df.columns) == 10
    np.testing.assert_array_equal(df.columns.values, portfolio.nonzero_assets)
    np.testing.assert_array_equal(
        df.values[0], portfolio.weights[portfolio.nonzero_assets_index]
    )


def test_sample_weight(portfolio, sample_weight):
    ref = portfolio.cvar
    portfolio.sample_weight = np.ones(len(sample_weight)) / len(sample_weight)
    v1 = portfolio.cvar
    np.testing.assert_almost_equal(ref, v1)
    portfolio.sample_weight = sample_weight
    v2 = portfolio.cvar
    _ = portfolio.summary()
    assert abs(v1 - v2) > 0.001
    portfolio.sample_weight = None
    v3 = portfolio.cvar
    np.testing.assert_almost_equal(ref, v3)


def test_sample_weight_error(portfolio, sample_weight):
    with pytest.raises(ValueError, match="sample_weight must have the same length as"):
        portfolio.sample_weight = np.ones(5)

    with pytest.raises(ValueError, match="sample_weight must sum to one"):
        portfolio.sample_weight = np.ones(len(sample_weight))

    with pytest.raises(ValueError, match="sample_weight must be a 1D array"):
        portfolio.sample_weight = [[1]]
