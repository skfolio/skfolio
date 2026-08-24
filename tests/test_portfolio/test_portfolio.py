from __future__ import annotations

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
from skfolio.typing import FloatArray
from skfolio.utils.stats import rand_weights
from skfolio.utils.tools import args_names


@pytest.fixture(scope="module")
def X() -> pd.DataFrame:
    prices = load_sp500_dataset()
    prices = prices.loc[pd.Timestamp(2017, 1, 1) :]
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="module")
def weights() -> FloatArray:
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
def portfolio(X: pd.DataFrame, weights: FloatArray) -> Portfolio:
    portfolio = Portfolio(X=X, weights=weights, annualization_factor=252)
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
def annualization_factor(request):
    return request.param


def _portfolio_returns(asset_returns: FloatArray, weights: FloatArray) -> FloatArray:
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


def test_portfolio_annualized(X, weights, annualization_factor):
    if annualization_factor is None:
        portfolio = Portfolio(X=X, weights=weights)
    else:
        portfolio = Portfolio(
            X=X, weights=weights, annualization_factor=annualization_factor
        )

    if annualization_factor is None:
        annualization_factor = 252.0
    assert portfolio.annualization_factor == annualization_factor

    np.testing.assert_almost_equal(
        portfolio.annualized_mean, portfolio.mean * annualization_factor
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_variance, portfolio.variance * annualization_factor
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_semi_variance,
        portfolio.semi_variance * annualization_factor,
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_standard_deviation,
        portfolio.standard_deviation * np.sqrt(annualization_factor),
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_semi_deviation,
        portfolio.semi_deviation * np.sqrt(annualization_factor),
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_sharpe_ratio,
        portfolio.sharpe_ratio * np.sqrt(annualization_factor),
    )
    np.testing.assert_almost_equal(
        portfolio.annualized_sortino_ratio,
        portfolio.sortino_ratio * np.sqrt(annualization_factor),
    )


def test_portfolio_deprecated_annualized_factor(X, weights):
    with pytest.warns(FutureWarning, match="annualized_factor"):
        portfolio = Portfolio(X=X, weights=weights, annualized_factor=12)

    assert portfolio.annualization_factor == 12

    with pytest.warns(FutureWarning, match="annualized_factor"):
        assert portfolio.annualized_factor == 12

    with pytest.warns(FutureWarning, match="annualized_factor"):
        portfolio.annualized_factor = 52

    assert portfolio.annualization_factor == 52


def test_portfolio_annualization_factor_conflict(X, weights):
    with pytest.raises(ValueError, match="annualized_factor"):
        Portfolio(
            X=X,
            weights=weights,
            annualization_factor=252,
            annualized_factor=252,
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
    assert isinstance(portfolio.cumulative_returns_df, pd.Series)
    assert isinstance(portfolio.drawdowns_df, pd.Series)
    portfolio.clear()
    assert portfolio.plot_returns()
    assert portfolio.plot_returns_distribution()
    assert portfolio.plot_cumulative_returns()
    assert portfolio.plot_drawdowns()
    assert portfolio.plot_rolling_measure(measure=RatioMeasure.SHARPE_RATIO, window=20)
    assert isinstance(portfolio.composition, pd.DataFrame)
    assert portfolio.plot_composition()
    assert isinstance(portfolio.summary(), pd.Series)
    assert isinstance(portfolio.summary(formatted=False), pd.Series)
    assert portfolio.get_weight(asset=portfolio.nonzero_assets[5])
    portfolio.annualization_factor = 252
    assert isinstance(portfolio.summary(), pd.Series)
    assert isinstance(portfolio.weights_dict, dict)
    assert isinstance(portfolio.previous_weights_dict, dict)


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


def test_portfolio_metrics_2(portfolio, measure):
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


def test_portfolio_slots(portfolio):
    for attr in portfolio._slots():
        if attr[0] == "_":
            try:
                getattr(portfolio, attr[1:])
            except AttributeError:
                pass
        getattr(portfolio, attr)


def test_copy(portfolio):
    with pytest.raises(AttributeError):
        _ = portfolio._assets_names
    _ = portfolio.nonzero_assets
    _ = copy(portfolio)


def test_portfolio_cache(portfolio, measure):
    # time for accessing cached attributes
    n = int(1e5)
    first_access_time = timeit.timeit(
        lambda: getattr(portfolio, measure.value), number=1
    )
    cached_access_time = (
        timeit.timeit(lambda: getattr(portfolio, measure.value), number=n) / n
    )
    assert first_access_time > 10 * cached_access_time


def test_portfolio_clear_cache(portfolio, measure):
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


def test_portfolio_read_only(portfolio):
    for attr in Portfolio._read_only_attrs:
        with pytest.raises(
            AttributeError,
            match=f"can't set attribute '{attr}' because it is read-only",
        ):
            setattr(portfolio, attr, 0)


def test_portfolio_delete_attr(portfolio):
    with pytest.raises(
        AttributeError, match="`Portfolio` object has no attribute 'dummy'"
    ):
        delattr(portfolio, "dummy")


def test_portfolio_rolling_measure(X, weights):
    window = 30
    portfolio = Portfolio(X=X[:50], weights=weights, annualization_factor=252)
    ref = Portfolio(
        X=X.iloc[50 - window : 50], weights=weights, annualization_factor=252
    )

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


def test_portfolio_plot_cumulative_returns(portfolio):
    assert portfolio.plot_cumulative_returns()

    with pytest.raises(ValueError):
        portfolio.plot_cumulative_returns(log_scale=True)

    portfolio.compounded = True
    assert portfolio.plot_cumulative_returns()
    assert portfolio.plot_cumulative_returns(log_scale=True)


def test_portfolio_plot_drawdowns(portfolio):
    assert portfolio.plot_drawdowns()
    portfolio.compounded = True
    assert portfolio.plot_drawdowns()


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


def test_weight_dict(X, weights):
    portfolio = Portfolio(X=X, weights=weights, previous_weights=np.arange(20))
    np.testing.assert_almost_equal(
        [portfolio.weights_dict[x] for x in X.columns], weights
    )
    np.testing.assert_almost_equal(
        [portfolio.previous_weights_dict[x] for x in X.columns], np.arange(20)
    )


def test_portfolio_nan_handling(X, weights):
    """Test that NaN values in X are handled gracefully."""
    X_with_nan = X.to_numpy().copy()

    # Asset index 3 has zero weight, asset index 0 has non-zero weight
    assert weights[3] == 0.0
    assert weights[0] != 0.0

    # NaN in asset returns is treated as zero for portfolio return computation.
    X_nan_zero_weight = X_with_nan.copy()
    X_nan_zero_weight[5, 3] = np.nan  # Day 5, asset 3 (zero weight)
    portfolio = Portfolio(X=X_nan_zero_weight, weights=weights)
    assert not np.any(np.isnan(portfolio.returns))

    # NaN in asset with non-zero weight contributes zero for that day.
    X_nan_nonzero_weight = X_with_nan.copy()
    X_nan_nonzero_weight[10, 0] = np.nan  # Day 10, asset 0 (non-zero weight)
    portfolio = Portfolio(X=X_nan_nonzero_weight, weights=weights)
    X_expected = np.nan_to_num(X_nan_nonzero_weight, nan=0.0)
    expected_returns = weights @ X_expected.T
    assert not np.any(np.isnan(portfolio.returns))
    np.testing.assert_array_almost_equal(portfolio.returns, expected_returns)

    # Multiple NaNs on different days
    X_multi_nan = X_with_nan.copy()
    X_multi_nan[10, 0] = np.nan  # Day 10, asset 0 (non-zero weight)
    X_multi_nan[20, 1] = np.nan  # Day 20, asset 1 (non-zero weight)
    X_multi_nan[30, 3] = np.nan  # Day 30, asset 3 (zero weight) - should be ignored
    portfolio = Portfolio(X=X_multi_nan, weights=weights)
    X_expected = np.nan_to_num(X_multi_nan, nan=0.0)
    expected_returns = weights @ X_expected.T
    assert not np.any(np.isnan(portfolio.returns))
    np.testing.assert_array_almost_equal(portfolio.returns, expected_returns)

    # NaN in asset with zero weight, verify returns match clean computation
    X_nan_zero_only = X_with_nan.copy()
    X_nan_zero_only[5, 3] = np.nan
    X_nan_zero_only[15, 4] = np.nan  # Asset 4 also has zero weight
    portfolio_nan = Portfolio(X=X_nan_zero_only, weights=weights)
    portfolio_ref = Portfolio(X=X, weights=weights)
    np.testing.assert_array_almost_equal(portfolio_nan.returns, portfolio_ref.returns)


class TestPortfolioFactorAttribution:
    """Tests for Portfolio.predicted_attribution and realized_attribution."""

    @pytest.fixture()
    def factor_model_and_portfolio(self):
        """Build a small synthetic factor model and a matching portfolio."""
        from skfolio.prior import FactorModel

        rng = np.random.default_rng(42)

        n_obs = 60
        n_assets = 4
        n_factors = 2
        asset_names = np.array(["A", "B", "C", "D"])
        factor_names = np.array(["Mom", "Val"])
        observations = pd.bdate_range("2023-01-01", periods=n_obs)

        loading = rng.standard_normal((n_assets, n_factors)) * 0.5
        A = rng.standard_normal((n_factors, n_factors))
        factor_cov = A @ A.T / n_factors
        factor_mu = rng.standard_normal(n_factors) * 0.001
        idio_cov = rng.uniform(0.001, 0.01, size=n_assets)

        factor_returns = rng.multivariate_normal(factor_mu, factor_cov, size=n_obs)
        exposures = np.tile(loading, (n_obs, 1, 1))
        exposures += rng.standard_normal(exposures.shape) * 0.05
        idio_returns = rng.standard_normal((n_obs, n_assets)) * np.sqrt(idio_cov)

        fm = FactorModel(
            observations=np.asarray(observations),
            asset_names=asset_names,
            factor_names=factor_names,
            factor_families=None,
            loading_matrix=loading,
            exposures=exposures,
            factor_covariance=factor_cov,
            factor_mu=factor_mu,
            factor_returns=factor_returns,
            idio_covariance=idio_cov,
            idio_mu=None,
            idio_returns=idio_returns,
            regression_weights=np.ones((n_obs, n_assets)),
            idio_variances=np.broadcast_to(idio_cov, (n_obs, n_assets)).copy(),
        )

        weights = np.array([0.4, 0.3, 0.2, 0.1])
        X = pd.DataFrame(
            rng.standard_normal((n_obs, n_assets)) * 0.01,
            columns=asset_names,
            index=observations,
        )
        ptf = Portfolio(X=X, weights=weights)
        return fm, ptf

    # --- predicted_attribution ---

    def test_predicted_attribution_returns_attribution(
        self, factor_model_and_portfolio
    ):
        from skfolio.attribution import Attribution

        fm, ptf = factor_model_and_portfolio
        result = ptf.predicted_attribution(fm)
        assert isinstance(result, Attribution)

    def test_predicted_attribution_uses_portfolio_annualization_factor(
        self, factor_model_and_portfolio
    ):
        fm, ptf = factor_model_and_portfolio
        result = ptf.predicted_attribution(fm)
        result_from_fm = fm.predicted_attribution(
            weights=ptf.weights, annualization_factor=ptf.annualization_factor
        )
        np.testing.assert_almost_equal(result.total.vol, result_from_fm.total.vol)

    def test_predicted_attribution_consistent_with_factor_model(
        self, factor_model_and_portfolio
    ):
        fm, ptf = factor_model_and_portfolio
        result = ptf.predicted_attribution(fm)
        result_from_fm = fm.predicted_attribution(
            weights=ptf.weights, annualization_factor=ptf.annualization_factor
        )
        np.testing.assert_almost_equal(
            result.total.mu_contrib, result_from_fm.total.mu_contrib
        )

    def test_predicted_attribution_asset_not_in_model_raises(
        self, factor_model_and_portfolio
    ):
        fm, _ = factor_model_and_portfolio
        X_bad = pd.DataFrame(
            np.zeros((60, 2)),
            columns=["UNKNOWN_1", "UNKNOWN_2"],
            index=fm.observations,
        )
        ptf_bad = Portfolio(X=X_bad, weights=np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="not in the factor model"):
            ptf_bad.predicted_attribution(fm)

    def test_predicted_attribution_subset_assets(self, factor_model_and_portfolio):
        """Portfolio holds a subset of the factor model's assets."""
        from skfolio.attribution import Attribution

        fm, _ = factor_model_and_portfolio
        X_sub = pd.DataFrame(
            np.zeros((60, 2)),
            columns=np.array(["A", "C"]),
            index=fm.observations,
        )
        ptf_sub = Portfolio(X=X_sub, weights=np.array([0.6, 0.4]))
        result = ptf_sub.predicted_attribution(fm)
        assert isinstance(result, Attribution)

    # --- realized_attribution ---

    def test_realized_attribution_returns_attribution(self, factor_model_and_portfolio):
        from skfolio.attribution import Attribution

        fm, ptf = factor_model_and_portfolio
        result = ptf.realized_attribution(fm)
        assert isinstance(result, Attribution)

    def test_realized_attribution_auto_aligns(self, factor_model_and_portfolio):
        """Factor model covers 60 obs, portfolio only first 30."""
        fm, ptf = factor_model_and_portfolio

        X_short = pd.DataFrame(
            ptf.X[:30],
            columns=ptf.assets,
            index=fm.observations[:30],
        )
        ptf_short = Portfolio(X=X_short, weights=ptf.weights)

        result = ptf_short.realized_attribution(fm)
        from skfolio.attribution import Attribution

        assert isinstance(result, Attribution)

    def test_realized_attribution_trims_factor_model_warmup(
        self, factor_model_and_portfolio
    ):
        """Portfolio can start before the factor model realized time series."""
        fm, ptf = factor_model_and_portfolio
        fm_warmup = fm.select_observations(fm.observations[10:50])

        result = ptf.realized_attribution(fm_warmup)
        result_from_fm = fm_warmup.realized_attribution(
            weights=ptf.weights,
            portfolio_returns=ptf.returns[10:50],
            annualization_factor=ptf.annualization_factor,
            compute_uncertainty=True,
        )

        np.testing.assert_almost_equal(result.total.vol, result_from_fm.total.vol)

    def test_realized_attribution_internal_missing_observation_raises(
        self, factor_model_and_portfolio
    ):
        """Missing dates inside the overlap remain an alignment error."""
        fm, ptf = factor_model_and_portfolio
        observations = np.concatenate([fm.observations[:20], fm.observations[21:]])
        fm_gap = fm.select_observations(observations)

        with pytest.raises(ValueError, match="inside the overlapping"):
            ptf.realized_attribution(fm_gap)

    def test_realized_attribution_asset_not_in_model_raises(
        self, factor_model_and_portfolio
    ):
        fm, _ = factor_model_and_portfolio
        X_bad = pd.DataFrame(
            np.zeros((60, 2)),
            columns=["UNKNOWN_1", "UNKNOWN_2"],
            index=fm.observations,
        )
        ptf_bad = Portfolio(X=X_bad, weights=np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="not in the factor model"):
            ptf_bad.realized_attribution(fm)

    def test_realized_attribution_missing_observations_raises(
        self, factor_model_and_portfolio
    ):
        fm, ptf = factor_model_and_portfolio
        bad_dates = pd.bdate_range("2099-01-01", periods=60)
        X_bad = pd.DataFrame(
            np.asarray(ptf.X),
            columns=ptf.assets,
            index=bad_dates,
        )
        ptf_bad = Portfolio(X=X_bad, weights=ptf.weights)
        with pytest.raises(ValueError, match="not found in FactorModel"):
            ptf_bad.realized_attribution(fm)

    def test_realized_attribution_uses_portfolio_annualization_factor(
        self, factor_model_and_portfolio
    ):
        factor_model, ptf = factor_model_and_portfolio
        result = ptf.realized_attribution(factor_model)
        aligned_factor_model = factor_model.select_observations(ptf.observations)
        result_from_factor_model = aligned_factor_model.realized_attribution(
            weights=ptf.weights,
            portfolio_returns=ptf.returns,
            annualization_factor=ptf.annualization_factor,
            compute_uncertainty=True,
        )
        np.testing.assert_almost_equal(
            result.total.vol, result_from_factor_model.total.vol
        )

    def test_realized_attribution_compute_uncertainty_false_no_regression_inputs(
        self, factor_model_and_portfolio
    ):
        from dataclasses import replace

        from skfolio.attribution import Attribution

        fm, ptf = factor_model_and_portfolio
        fm_no_unc = replace(fm, regression_weights=None, idio_variances=None)
        result = ptf.realized_attribution(fm_no_unc, compute_uncertainty=False)
        assert isinstance(result, Attribution)
        assert result.systematic.mu_uncertainty is None

    # --- rolling_realized_attribution ---

    def test_rolling_realized_returns_attribution(self, factor_model_and_portfolio):
        from skfolio.attribution import Attribution

        fm, ptf = factor_model_and_portfolio
        result = ptf.rolling_realized_attribution(fm, window_size=20, step=10)
        assert isinstance(result, Attribution)
        assert result.is_rolling is True

    def test_rolling_realized_window_count(self, factor_model_and_portfolio):
        fm, ptf = factor_model_and_portfolio
        result = ptf.rolling_realized_attribution(fm, window_size=20, step=10)
        assert len(result.observations) > 0
        assert result.total.vol.shape[0] == len(result.observations)

    def test_rolling_realized_auto_aligns(self, factor_model_and_portfolio):
        """Factor model covers 60 obs, portfolio only first 40."""
        fm, ptf = factor_model_and_portfolio
        X_short = pd.DataFrame(
            ptf.X[:40],
            columns=ptf.assets,
            index=fm.observations[:40],
        )
        ptf_short = Portfolio(X=X_short, weights=ptf.weights)
        result = ptf_short.rolling_realized_attribution(fm, window_size=15, step=10)
        from skfolio.attribution import Attribution

        assert isinstance(result, Attribution)

    def test_rolling_realized_trims_factor_model_warmup(
        self, factor_model_and_portfolio
    ):
        """Rolling windows are computed over the overlapping realized window."""
        fm, ptf = factor_model_and_portfolio
        fm_warmup = fm.select_observations(fm.observations[5:45])

        result = ptf.rolling_realized_attribution(fm_warmup, window_size=15, step=10)

        expected = len(np.arange(0, 39 - 15 + 1, 10))
        assert len(result.observations) == expected
        np.testing.assert_array_equal(
            result.observations, fm.observations[[20, 30, 40]]
        )

    def test_rolling_realized_uses_portfolio_annualization_factor(
        self, factor_model_and_portfolio
    ):
        factor_model, ptf = factor_model_and_portfolio
        result = ptf.rolling_realized_attribution(factor_model, window_size=20, step=10)
        aligned_factor_model = factor_model.select_observations(ptf.observations)
        result_from_factor_model = aligned_factor_model.rolling_realized_attribution(
            weights=ptf.weights,
            portfolio_returns=ptf.returns,
            annualization_factor=ptf.annualization_factor,
            window_size=20,
            step=10,
            compute_uncertainty=True,
        )
        np.testing.assert_array_almost_equal(
            result.total.vol, result_from_factor_model.total.vol
        )

    def test_rolling_realized_asset_not_in_model_raises(
        self, factor_model_and_portfolio
    ):
        fm, _ = factor_model_and_portfolio
        X_bad = pd.DataFrame(
            np.zeros((60, 2)),
            columns=["UNKNOWN_1", "UNKNOWN_2"],
            index=fm.observations,
        )
        ptf_bad = Portfolio(X=X_bad, weights=np.array([0.5, 0.5]))
        with pytest.raises(ValueError, match="not in the factor model"):
            ptf_bad.rolling_realized_attribution(fm, window_size=20, step=10)

    def test_rolling_realized_failed_portfolio_raises(self, factor_model_and_portfolio):
        from skfolio.portfolio._failed_portfolio import FailedPortfolio

        fm, ptf = factor_model_and_portfolio
        failed = FailedPortfolio(X=ptf.X)
        with pytest.raises(ValueError, match="failed portfolio"):
            failed.rolling_realized_attribution(fm, window_size=20, step=10)

    def test_rolling_realized_decomposition_additive(self, factor_model_and_portfolio):
        fm, ptf = factor_model_and_portfolio
        result = ptf.rolling_realized_attribution(fm, window_size=20, step=10)
        for i in range(len(result.observations)):
            sum_vol = (
                np.sum(result.factors.vol_contrib[i])
                + result.idio.vol_contrib[i]
                + result.unattributed.vol_contrib[i]
            )
            np.testing.assert_almost_equal(sum_vol, result.total.vol[i], decimal=8)


class TestPortfolioNaNReturns:
    def test_zero_weight_nan_returns_treated_as_zero(self):
        rets = np.array([[0.01, np.nan], [0.02, np.nan], [-0.01, np.nan]])
        weights = np.array([1.0, 0.0])
        ptf = Portfolio(X=rets, weights=weights)
        np.testing.assert_array_almost_equal(ptf.returns, [0.01, 0.02, -0.01])

    def test_all_nan_with_zero_weights(self):
        rets = np.full((5, 3), np.nan)
        weights = np.zeros(3)
        ptf = Portfolio(X=rets, weights=weights)
        np.testing.assert_array_equal(ptf.returns, np.zeros(5))

    def test_mixed_nan_and_finite(self):
        rets = np.array([[0.01, np.nan, 0.03], [0.02, 0.05, np.nan]])
        weights = np.array([0.5, 0.0, 0.5])
        ptf = Portfolio(X=rets, weights=weights)
        np.testing.assert_array_almost_equal(
            ptf.returns, [0.5 * 0.01 + 0.5 * 0.03, 0.5 * 0.02]
        )

    def test_original_X_preserved_with_nan(self):
        rets = np.array([[0.01, np.nan], [0.02, np.nan]])
        weights = np.array([1.0, 0.0])
        ptf = Portfolio(X=rets, weights=weights)
        np.testing.assert_array_equal(np.asarray(ptf.X), rets)
