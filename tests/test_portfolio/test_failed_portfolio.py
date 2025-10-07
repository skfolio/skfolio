import pickle
import timeit
import tracemalloc
from copy import copy

import numpy as np
import pandas as pd
import pytest
import sklearn.utils.validation as skv

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
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import BaseOptimization
from skfolio.portfolio._base import _MEASURES
from skfolio.utils.stats import rand_weights
from skfolio.utils.tools import args_names


@pytest.fixture(
    scope="module",
    params=list(PerfMeasure)
    + list(RiskMeasure)
    + list(RiskMeasure)
    + list(ExtraRiskMeasure),
)
def measure(request):
    return request.param


@pytest.fixture
def portfolio(X: pd.DataFrame) -> FailedPortfolio:
    portfolio = FailedPortfolio(X=X)
    return portfolio


class CustomOptimization(BaseOptimization):
    """Dummy optimization that forces a `fit` failure with proba `failure_proba`."""

    def __init__(
        self,
        failure_proba: float = 0.5,
        portfolio_params: dict | None = None,
        fallback=None,
        previous_weights=None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
            previous_weights=previous_weights,
        )
        self.failure_proba = failure_proba

    def fit(self, X, y=None):
        X = skv.validate_data(self, X)
        # Fail with probability equal to `failure_proba`
        if np.random.rand() < self.failure_proba:
            raise RuntimeError("Forced failure")
        n_assets = X.shape[1]
        self.weights_ = rand_weights(n_assets)
        return self


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


def test_concatenate(X):
    portfolios = [FailedPortfolio(X=X), FailedPortfolio(X=X)]
    c = np.concatenate(portfolios)
    assert c.shape == (X.shape[0] * 2,)


def _estimate_portfolio_memory(X, n: int) -> float:
    tracemalloc.start()
    tracemalloc.clear_traces()
    start = tracemalloc.get_traced_memory()
    for _ in range(n):
        portfolio = FailedPortfolio(X=X)
        _ = portfolio.returns
        _ = portfolio.standard_deviation
        _ = portfolio.fitness
        _ = portfolio.mean_absolute_deviation_ratio
    end = tracemalloc.get_traced_memory()
    tracemalloc.clear_traces()
    return end[0] - start[0]


def test_garbage_collection(X):
    m1 = _estimate_portfolio_memory(X, n=1)
    m10 = _estimate_portfolio_memory(X, n=10)
    m100 = _estimate_portfolio_memory(X, n=100)
    m1000 = _estimate_portfolio_memory(X, n=1000)

    assert m10 < 2 * m1
    assert m100 < 2 * m1
    assert m1000 < 2 * m1


def test_portfolio_methods(X, portfolio):
    assert portfolio.n_observations == X.shape[0]
    assert portfolio.n_assets == X.shape[1]
    assert portfolio.returns.shape == (X.shape[0],)
    assert np.all(np.isnan(portfolio.returns))
    np.testing.assert_array_equal(portfolio.fitness, np.array([np.nan, np.nan]))
    portfolio.fitness_measures = [
        PerfMeasure.MEAN,
        RiskMeasure.SEMI_DEVIATION,
        RiskMeasure.MAX_DRAWDOWN,
    ]
    np.testing.assert_almost_equal(
        portfolio.fitness,
        np.array([np.nan, np.nan, np.nan]),
    )
    assert len(portfolio.nonzero_assets_index) == 20
    assert len(portfolio.nonzero_assets) == 20
    assert len(portfolio.composition) == 20
    assert np.isnan(portfolio.composition).all().all()

    contrib = portfolio.contribution(measure=RatioMeasure.SHARPE_RATIO)
    assert len(contrib) == 20
    assert np.isnan(contrib).all()

    assert isinstance(portfolio.cumulative_returns_df, pd.Series)
    assert isinstance(portfolio.drawdowns_df, pd.Series)
    portfolio.clear()
    assert portfolio.plot_returns()
    assert portfolio.plot_cumulative_returns()
    assert portfolio.plot_drawdowns()
    assert portfolio.plot_rolling_measure(measure=RatioMeasure.SHARPE_RATIO, window=20)
    assert isinstance(portfolio.composition, pd.DataFrame)
    assert portfolio.plot_composition()
    assert isinstance(portfolio.summary(), pd.Series)
    assert isinstance(portfolio.summary(formatted=False), pd.Series)
    assert isinstance(portfolio.summary(), pd.Series)


def test_portfolio_magic_methods(X):
    for ptf_1, ptf_2 in [
        (FailedPortfolio(X=X), FailedPortfolio(X=X)),
        (FailedPortfolio(X=X), Portfolio(X=X, weights=rand_weights(n=20))),
        (Portfolio(X=X, weights=rand_weights(n=20)), FailedPortfolio(X=X)),
    ]:
        assert ptf_1.n_observations == X.shape[0]
        assert ptf_1.n_assets == X.shape[1]
        ptf = ptf_1 + ptf_2
        assert isinstance(ptf, FailedPortfolio)
        ptf = ptf_1 - ptf_2

    for ptf_1, ptf_2 in [
        (FailedPortfolio(X=X), FailedPortfolio(X=X)),
        (FailedPortfolio(X=X), Portfolio(X=X, weights=rand_weights(n=20))),
    ]:
        assert isinstance(ptf, FailedPortfolio)
        ptf = -ptf_1
        assert isinstance(ptf, FailedPortfolio)
        ptf = ptf_1 * 2.3
        assert isinstance(ptf, FailedPortfolio)
        ptf = ptf_1 / 2.3
        assert isinstance(ptf, FailedPortfolio)
        ptf = abs(ptf_1)
        assert isinstance(ptf, FailedPortfolio)
        ptf = round(ptf_1, 2)
        assert isinstance(ptf, FailedPortfolio)
        ptf = ptf_1 // 2
        assert isinstance(ptf, FailedPortfolio)
        assert ptf_1 != ptf_1
        assert ptf_1 != ptf_2
        assert (ptf_1 > ptf_2) is ptf_1.dominates(ptf_2)
        assert (ptf_1 < ptf_2) is ptf_2.dominates(ptf_1)


def test_portfolio_metrics(portfolio, measure):
    m = getattr(portfolio, measure.value)
    assert np.isnan(m)


def test_portfolio_metrics2(portfolio):
    assert np.isnan(portfolio.sric)
    assert np.isnan(portfolio.skew)
    assert np.isnan(portfolio.kurtosis)
    assert np.isnan(portfolio.diversification)
    assert np.isnan(portfolio.effective_number_assets)


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
    ref = timeit.timeit(lambda: portfolio.name, number=n) / n
    first_access_time = timeit.timeit(
        lambda: getattr(portfolio, measure.value), number=1
    )
    cached_access_time = (
        timeit.timeit(lambda: getattr(portfolio, measure.value), number=n) / n
    )
    assert first_access_time > 10 * cached_access_time
    assert ref > cached_access_time / 10


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
        AttributeError, match="`FailedPortfolio` object has no attribute 'dummy'"
    ):
        delattr(portfolio, "dummy")


def test_portfolio_rolling_measure(portfolio):
    for measure in _MEASURES:
        res = portfolio.rolling_measure(measure=measure, window=30)
        assert isinstance(res, pd.Series)
        assert np.isnan(res).all()


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
    assert contribution.shape == (20, 1)

    contribution = portfolio.contribution(measure=RiskMeasure.STANDARD_DEVIATION)
    assert isinstance(contribution, np.ndarray)
    assert contribution.shape == (20,)

    assert portfolio.plot_contribution(measure=RiskMeasure.STANDARD_DEVIATION)


def test_weights_per_observation(portfolio):
    df = portfolio.weights_per_observation
    np.testing.assert_array_equal(df.index.values, portfolio.observations)
    assert len(df.columns) == 20
    assert np.isnan(df).all().all()


def test_cross_val_predict(X):
    walk_forward = WalkForward(test_size=1, train_size=12, freq="WOM-3FRI")

    model = CustomOptimization(failure_proba=0.5, raise_on_failure=False)

    with pytest.warns(UserWarning):
        pred = cross_val_predict(model, X, cv=walk_forward)

    assert len(pred.composition) == 20
    assert not np.isnan(pred.composition).all().all()

    contrib = pred.contribution(measure=RatioMeasure.SHARPE_RATIO)
    assert len(contrib) == 20
    assert not np.isnan(contrib).all().all()

    assert isinstance(pred.cumulative_returns_df, pd.Series)
    assert isinstance(pred.drawdowns_df, pd.Series)
    pred.clear()
    assert pred.plot_returns()
    assert pred.plot_cumulative_returns()
    assert pred.plot_drawdowns()
    assert pred.plot_rolling_measure(measure=RatioMeasure.SHARPE_RATIO, window=20)
    assert isinstance(pred.composition, pd.DataFrame)
    assert pred.plot_composition()
    assert isinstance(pred.summary(), pd.Series)
    assert isinstance(pred.summary(formatted=False), pd.Series)
    assert isinstance(pred.summary(), pd.Series)
