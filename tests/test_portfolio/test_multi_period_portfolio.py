from __future__ import annotations

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
from skfolio.typing import FloatArray
from skfolio.utils.stats import rand_weights
from skfolio.utils.tools import args_names


def _portfolio_returns(asset_returns: FloatArray, weights: FloatArray) -> FloatArray:
    r"""
    Compute the portfolio returns from its assets returns and weights.
    """
    n, m = asset_returns.shape
    returns = np.zeros(n)
    for i in range(m):
        returns += asset_returns[:, i] * weights[i]
    return returns


def _dominate(fitness_1: FloatArray, fitness_2: FloatArray) -> bool:
    return np.all(fitness_1 >= fitness_2) and np.any(fitness_1 > fitness_2)


@pytest.fixture(scope="module")
def prices():
    prices = load_sp500_dataset()
    prices = prices.loc[pd.Timestamp(2017, 1, 1) :]
    return prices


@pytest.fixture(scope="module")
def X(prices):
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="module")
def periods():
    periods = [
        (pd.Timestamp(2018, 1, 1), pd.Timestamp(2018, 3, 1)),
        (pd.Timestamp(2018, 3, 15), pd.Timestamp(2018, 5, 1)),
        (pd.Timestamp(2018, 5, 1), pd.Timestamp(2018, 8, 1)),
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
def annualization_factor(request):
    return request.param


def test_portfolio_annualized(portfolio, annualization_factor):
    if annualization_factor is not None:
        portfolio.annualization_factor = annualization_factor

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


def test_long_short_exposure():
    X = pd.DataFrame(
        np.zeros((6, 3)),
        index=pd.date_range("2024-01-01", periods=6),
        columns=["A", "B", "C"],
    )
    portfolio = MultiPeriodPortfolio(
        portfolios=[
            Portfolio(X=X.iloc[:2], weights=[0.4, -0.25, -0.15]),
            FailedPortfolio(X=X.iloc[2:4]),
            Portfolio(X=X.iloc[4:], weights=[0.1, 0.0, -0.1]),
        ]
    )

    expected = pd.DataFrame(
        {
            "Long": [0.4, 0.4, np.nan, np.nan, 0.1, 0.1],
            "Short": [-0.4, -0.4, np.nan, np.nan, -0.1, -0.1],
            "Net": [0.0, 0.0, np.nan, np.nan, 0.0, 0.0],
            "Gross": [0.8, 0.8, np.nan, np.nan, 0.2, 0.2],
        },
        index=X.index,
    )
    pd.testing.assert_frame_equal(
        portfolio.long_short_exposure,
        expected,
        check_freq=False,
    )

    fig = portfolio.plot_long_short_exposure()
    assert len(fig.data) == 4
    assert [trace.name for trace in fig.data] == ["Long", "Short", "Net", "Gross"]


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


def _make_factor_model(asset_names, n_obs, rng):
    """Build a minimal FactorModel for attribution tests."""
    from skfolio.prior import FactorModel

    n_assets = len(asset_names)
    n_factors = 2
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

    return FactorModel(
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


class TestMultiPeriodPortfolioFactorAttribution:
    """Tests for MultiPeriodPortfolio.predicted_attribution and realized_attribution."""

    @pytest.fixture()
    def fm_and_mpp(self):
        """Build a factor model and a multi-period portfolio with two periods."""
        rng = np.random.default_rng(99)
        asset_names = np.array(["A", "B", "C", "D"])
        n_obs = 60

        fm = _make_factor_model(asset_names, n_obs, rng)
        obs = fm.observations

        w1 = np.array([0.4, 0.3, 0.2, 0.1])
        X1 = pd.DataFrame(
            rng.standard_normal((30, 4)) * 0.01,
            columns=asset_names,
            index=obs[:30],
        )
        ptf1 = Portfolio(X=X1, weights=w1)

        w2 = np.array([0.1, 0.2, 0.3, 0.4])
        X2 = pd.DataFrame(
            rng.standard_normal((30, 4)) * 0.01,
            columns=asset_names,
            index=obs[30:60],
        )
        ptf2 = Portfolio(X=X2, weights=w2)

        mpp = MultiPeriodPortfolio(portfolios=[ptf1, ptf2])
        return fm, mpp

    # --- predicted_attribution ---

    def test_predicted_returns_attribution(self, fm_and_mpp):
        from skfolio.attribution import Attribution

        fm, mpp = fm_and_mpp
        result = mpp.predicted_attribution(fm)
        assert isinstance(result, Attribution)

    def test_predicted_uses_last_portfolio_weights(self, fm_and_mpp):
        fm, mpp = fm_and_mpp
        result_mpp = mpp.predicted_attribution(fm)
        result_last = mpp[-1].predicted_attribution(fm)
        np.testing.assert_almost_equal(result_mpp.total.vol, result_last.total.vol)

    def test_predicted_empty_raises(self):
        mpp = MultiPeriodPortfolio()
        rng = np.random.default_rng(0)
        fm = _make_factor_model(np.array(["A"]), 10, rng)
        with pytest.raises(ValueError, match="empty"):
            mpp.predicted_attribution(fm)

    def test_predicted_last_failed_raises(self, fm_and_mpp):
        fm, mpp = fm_and_mpp
        obs = fm.observations
        failed = FailedPortfolio(
            X=pd.DataFrame(
                np.zeros((5, 4)),
                columns=["A", "B", "C", "D"],
                index=obs[55:60],
            ),
        )
        mpp_fail = MultiPeriodPortfolio(portfolios=[mpp[0], failed])
        with pytest.raises(ValueError, match="FailedPortfolio"):
            mpp_fail.predicted_attribution(fm)

    def test_predicted_asset_not_in_model_raises(self, fm_and_mpp):
        fm, _ = fm_and_mpp
        ptf = Portfolio(
            X=pd.DataFrame(
                np.zeros((10, 2)),
                columns=["UNKNOWN", "OTHER"],
                index=fm.observations[:10],
            ),
            weights=np.array([0.5, 0.5]),
        )
        mpp = MultiPeriodPortfolio(portfolios=[ptf])
        with pytest.raises(ValueError, match="not in the factor model"):
            mpp.predicted_attribution(fm)

    # --- realized_attribution ---

    def test_realized_returns_attribution(self, fm_and_mpp):
        from skfolio.attribution import Attribution

        fm, mpp = fm_and_mpp
        result = mpp.realized_attribution(fm)
        assert isinstance(result, Attribution)

    def test_realized_skips_failed_portfolios(self, fm_and_mpp):
        from skfolio.attribution import Attribution

        fm, mpp = fm_and_mpp
        obs = fm.observations
        failed = FailedPortfolio(
            X=pd.DataFrame(
                np.zeros((5, 4)),
                columns=["A", "B", "C", "D"],
                index=obs[25:30],
            ),
        )
        ptf1 = mpp[0]
        ptf2 = mpp[1]
        mpp_with_failed = MultiPeriodPortfolio(
            portfolios=[ptf1, failed, ptf2],
            check_observations_order=False,
        )
        result = mpp_with_failed.realized_attribution(fm)
        assert isinstance(result, Attribution)

    def test_realized_empty_raises(self):
        mpp = MultiPeriodPortfolio()
        rng = np.random.default_rng(0)
        fm = _make_factor_model(np.array(["A"]), 10, rng)
        with pytest.raises(ValueError, match="empty"):
            mpp.realized_attribution(fm)

    def test_realized_all_failed_raises(self, fm_and_mpp):
        fm, _ = fm_and_mpp
        obs = fm.observations
        failed = FailedPortfolio(
            X=pd.DataFrame(
                np.zeros((10, 4)),
                columns=["A", "B", "C", "D"],
                index=obs[:10],
            ),
        )
        mpp = MultiPeriodPortfolio(portfolios=[failed])
        with pytest.raises(ValueError, match="All child portfolios"):
            mpp.realized_attribution(fm)

    def test_realized_subset_assets(self, fm_and_mpp):
        """Child portfolios hold subsets of the factor model's assets."""
        from skfolio.attribution import Attribution

        fm, _ = fm_and_mpp
        rng = np.random.default_rng(7)
        obs = fm.observations

        ptf1 = Portfolio(
            X=pd.DataFrame(
                rng.standard_normal((30, 2)) * 0.01,
                columns=np.array(["A", "B"]),
                index=obs[:30],
            ),
            weights=np.array([0.6, 0.4]),
        )
        ptf2 = Portfolio(
            X=pd.DataFrame(
                rng.standard_normal((30, 2)) * 0.01,
                columns=np.array(["C", "D"]),
                index=obs[30:60],
            ),
            weights=np.array([0.5, 0.5]),
        )
        mpp = MultiPeriodPortfolio(portfolios=[ptf1, ptf2])
        result = mpp.realized_attribution(fm)
        assert isinstance(result, Attribution)

    def test_realized_trims_factor_model_warmup(self, fm_and_mpp):
        """Aggregated returns are restricted to the factor model overlap."""
        from skfolio.attribution import Attribution

        fm, mpp = fm_and_mpp
        fm_warmup = fm.select_observations(fm.observations[10:50])

        result = mpp.realized_attribution(fm_warmup)

        assert isinstance(result, Attribution)

    def test_realized_internal_missing_observation_raises(self, fm_and_mpp):
        """Missing dates inside the overlap remain an alignment error."""
        fm, mpp = fm_and_mpp
        observations = np.concatenate([fm.observations[:20], fm.observations[21:]])
        fm_gap = fm.select_observations(observations)

        with pytest.raises(ValueError, match="inside the overlapping"):
            mpp.realized_attribution(fm_gap)

    def test_realized_asset_not_in_model_raises(self, fm_and_mpp):
        fm, _ = fm_and_mpp
        ptf = Portfolio(
            X=pd.DataFrame(
                np.zeros((10, 2)),
                columns=["UNKNOWN", "OTHER"],
                index=fm.observations[:10],
            ),
            weights=np.array([0.5, 0.5]),
        )
        mpp = MultiPeriodPortfolio(portfolios=[ptf])
        with pytest.raises(ValueError, match="not in the factor model"):
            mpp.realized_attribution(fm)

    # --- rolling_realized_attribution ---

    def test_rolling_realized_returns_attribution(self, fm_and_mpp):
        from skfolio.attribution import Attribution

        fm, mpp = fm_and_mpp
        result = mpp.rolling_realized_attribution(fm, window_size=15, step=10)
        assert isinstance(result, Attribution)
        assert result.is_rolling is True

    def test_rolling_realized_window_count(self, fm_and_mpp):
        fm, mpp = fm_and_mpp
        result = mpp.rolling_realized_attribution(fm, window_size=15, step=10)
        n_obs = len(mpp.observations)
        expected = len(np.arange(0, n_obs - 15 + 1, 10))
        assert len(result.observations) == expected

    def test_rolling_realized_trims_factor_model_warmup(self, fm_and_mpp):
        """Rolling windows use only overlapping factor model observations."""
        fm, mpp = fm_and_mpp
        fm_warmup = fm.select_observations(fm.observations[5:45])

        result = mpp.rolling_realized_attribution(fm_warmup, window_size=15, step=10)

        expected = len(np.arange(0, 39 - 15 + 1, 10))
        assert len(result.observations) == expected

    def test_rolling_realized_skips_failed_portfolios(self, fm_and_mpp):
        from skfolio.attribution import Attribution

        fm, mpp = fm_and_mpp
        obs = fm.observations
        failed = FailedPortfolio(
            X=pd.DataFrame(
                np.zeros((5, 4)),
                columns=["A", "B", "C", "D"],
                index=obs[25:30],
            ),
        )
        ptf1 = mpp[0]
        ptf2 = mpp[1]
        mpp_with_failed = MultiPeriodPortfolio(
            portfolios=[ptf1, failed, ptf2],
            check_observations_order=False,
        )
        result = mpp_with_failed.rolling_realized_attribution(
            fm, window_size=15, step=10
        )
        assert isinstance(result, Attribution)

    def test_rolling_realized_empty_raises(self):
        mpp = MultiPeriodPortfolio()
        rng = np.random.default_rng(0)
        fm = _make_factor_model(np.array(["A"]), 10, rng)
        with pytest.raises(ValueError, match="empty"):
            mpp.rolling_realized_attribution(fm, window_size=5, step=2)

    def test_rolling_realized_all_failed_raises(self, fm_and_mpp):
        fm, _ = fm_and_mpp
        obs = fm.observations
        failed = FailedPortfolio(
            X=pd.DataFrame(
                np.zeros((10, 4)),
                columns=["A", "B", "C", "D"],
                index=obs[:10],
            ),
        )
        mpp = MultiPeriodPortfolio(portfolios=[failed])
        with pytest.raises(ValueError, match="All child portfolios"):
            mpp.rolling_realized_attribution(fm, window_size=5, step=2)

    def test_rolling_realized_asset_not_in_model_raises(self, fm_and_mpp):
        fm, _ = fm_and_mpp
        ptf = Portfolio(
            X=pd.DataFrame(
                np.zeros((10, 2)),
                columns=["UNKNOWN", "OTHER"],
                index=fm.observations[:10],
            ),
            weights=np.array([0.5, 0.5]),
        )
        mpp = MultiPeriodPortfolio(portfolios=[ptf])
        with pytest.raises(ValueError, match="not in the factor model"):
            mpp.rolling_realized_attribution(fm, window_size=5, step=2)

    def test_rolling_realized_decomposition_additive(self, fm_and_mpp):
        fm, mpp = fm_and_mpp
        result = mpp.rolling_realized_attribution(fm, window_size=15, step=10)
        for i in range(len(result.observations)):
            sum_vol = (
                np.sum(result.factors.vol_contrib[i])
                + result.idio.vol_contrib[i]
                + result.unexplained.vol_contrib[i]
            )
            np.testing.assert_almost_equal(sum_vol, result.total.vol[i], decimal=8)
