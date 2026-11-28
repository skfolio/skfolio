"""Tests for online_score portfolio scoring and OnlineGridSearch / OnlineRandomizedSearch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.exceptions import NotFittedError, UnsetMetadataPassedError
from sklearn.pipeline import make_pipeline

from skfolio.measures import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.model_selection import (
    OnlineGridSearch,
    OnlineRandomizedSearch,
    online_score,
)
from skfolio.model_selection._online._search import (
    _check_refit_for_multimetric,
    _rank_scores,
)
from skfolio.model_selection._online._validation import (
    _score_multi_period_portfolio,
    _validate_sizes,
)
from skfolio.model_selection._validation import _is_portfolio_optimization_estimator
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior


def _make_online_estimator(**kwargs):
    return MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40),
            covariance_estimator=EWCovariance(half_life=40),
        ),
        **kwargs,
    )


WARMUP = 400
TEST_SIZE = 200


def _single_candidate_search(search_cls, estimator, **kwargs):
    """Use the same candidate for both online search implementations."""
    search_params = (
        {"param_grid": {}}
        if search_cls is OnlineGridSearch
        else {"param_distributions": {}, "n_iter": 1, "random_state": 0}
    )
    return search_cls(estimator, **search_params, error_score="raise", **kwargs)


def _mean_squared_error(estimator, rows):
    """Score a mean estimate before its next online update."""
    return -float(np.mean((rows - estimator.mu_) ** 2))


@pytest.mark.parametrize("search_cls", [OnlineGridSearch, OnlineRandomizedSearch])
@pytest.mark.parametrize(
    (
        "n_observations",
        "warmup_size",
        "test_size",
        "purged_size",
        "reduce_test",
        "freq",
    ),
    [
        (25, 8, 5, 0, False, None),
        (27, 8, 5, 2, False, None),
        (23, 8, 5, 0, False, None),
        (25, 8, 1, 0, False, None),
        (25, 8, 5, 0, True, None),
        (54, 2, 2, 0, False, "W-MON"),
    ],
    ids=["trailing", "purged-trailing", "aligned", "one-period", "reduced", "weekly"],
)
def test_online_search_final_state(
    search_cls, n_observations, warmup_size, test_size, purged_size, reduce_test, freq
):
    """Final training includes unscored rows without changing evaluation policy."""
    X = np.column_stack(
        [
            np.linspace(-0.02, 0.03, n_observations),
            np.linspace(0.01, -0.01, n_observations),
        ]
    )
    X[-2:] += [0.15, -0.08]
    if freq is not None:
        # Start on a calendar boundary so no initial observations are excluded.
        X = pd.DataFrame(X, index=pd.bdate_range("2024-01-01", periods=n_observations))
    kwargs = dict(
        warmup_size=warmup_size,
        test_size=test_size,
        purged_size=purged_size,
        reduce_test=reduce_test,
        freq=freq,
        scoring=_mean_squared_error,
    )
    search = _single_candidate_search(
        search_cls, EWMu(half_life=4, min_observations=2), **kwargs
    ).fit(X)

    # Bias-corrected EW means are normalized scalar geometric sums, independent
    # of both the estimator recurrence and the walk-forward split construction.
    decay = 0.5**0.25
    weights = [decay ** (n_observations - 1 - i) for i in range(n_observations)]
    expected = [
        sum(w * float(row[j]) for w, row in zip(weights, np.asarray(X), strict=True))
        / sum(weights)
        for j in range(2)
    ]
    np.testing.assert_allclose(
        search.best_estimator_.mu_, expected, rtol=1e-12, atol=1e-15
    )

    no_refit = _single_candidate_search(
        search_cls, EWMu(half_life=4, min_observations=2), refit=False, **kwargs
    ).fit(X)
    assert not hasattr(no_refit, "best_estimator_")
    np.testing.assert_array_equal(
        no_refit.cv_results_["mean_score"], search.cv_results_["mean_score"]
    )


@pytest.mark.parametrize("search_cls", [OnlineGridSearch, OnlineRandomizedSearch])
def test_online_search_trailing_metadata(search_cls):
    """The final update slices aliased observation metadata with labeled returns."""
    X = pd.DataFrame(
        {"A": np.linspace(-0.02, 0.03, 25), "B": np.linspace(0.01, -0.01, 25)},
        index=pd.bdate_range("2024-01-01", periods=25),
    )
    active = pd.DataFrame(True, index=X.index, columns=X.columns)
    active.iloc[-1, 0] = False
    original_X, original_active = X.copy(deep=True), active.copy(deep=True)
    with config_context(enable_metadata_routing=True):
        estimator = EWMu(half_life=4, min_observations=2).set_partial_fit_request(
            active_mask="activity"
        )
        search = _single_candidate_search(
            search_cls,
            estimator,
            warmup_size=8,
            test_size=5,
            scoring=_mean_squared_error,
        ).fit(X, activity=active)

    assert np.isnan(search.best_estimator_.mu_[0])
    weights = [0.5 ** ((24 - i) / 4) for i in range(25)]
    expected_b = sum(w * value for w, value in zip(weights, X["B"], strict=True)) / sum(
        weights
    )
    np.testing.assert_allclose(
        search.best_estimator_.mu_[1], expected_b, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_array_equal(search.best_estimator_.feature_names_in_, ["A", "B"])
    pd.testing.assert_frame_equal(X, original_X)
    pd.testing.assert_frame_equal(active, original_active)


@pytest.mark.parametrize("search_cls", [OnlineGridSearch, OnlineRandomizedSearch])
@pytest.mark.parametrize("weight_drift", [False, True])
def test_online_search_trailing_returns_and_predictions(search_cls, weight_drift):
    """Tail data changes the next allocation, preserving the scored backtest."""
    t = np.arange(32)
    calm = pd.DataFrame(
        {"A": 0.01 * np.sin(t * 1.3), "B": 0.012 * np.cos(t * 0.7)},
        index=pd.bdate_range("2024-01-01", periods=32),
    )
    X = calm.copy()
    X.iloc[-4:] = [[0.15, 0.01], [-0.12, -0.01], [0.18, 0.005], [-0.1, -0.005]]
    model = MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=4, min_observations=2),
            covariance_estimator=EWCovariance(
                half_life=4, min_observations=2, assume_centered=True, nearest=False
            ),
        )
    )
    kwargs = dict(
        warmup_size=8,
        test_size=5,
        scoring=PerfMeasure.MEAN,
        return_predictions=True,
        portfolio_params={"weight_drift": weight_drift, "compounded": True},
    )
    baseline = _single_candidate_search(search_cls, model, **kwargs).fit(calm)
    search = _single_candidate_search(search_cls, model, **kwargs).fit(X)

    # Independent centered EW second moments and the closed-form fully invested,
    # long-only two-asset minimum-variance allocation.
    weights = [0.5 ** ((31 - i) / 4) for i in range(32)]
    values = X.to_numpy()
    covariance = np.array(
        [
            [
                sum(
                    w * float(row[j]) * float(row[k])
                    for w, row in zip(weights, values, strict=True)
                )
                / sum(weights)
                for k in range(2)
            ]
            for j in range(2)
        ]
    )
    weight_a = np.clip(
        (covariance[1, 1] - covariance[0, 1])
        / (covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]),
        0,
        1,
    )
    expected_weights = np.array([weight_a, 1 - weight_a])
    fitted = search.best_estimator_
    np.testing.assert_allclose(
        fitted.prior_estimator_.return_distribution_.covariance,
        covariance,
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(fitted.weights_, expected_weights, rtol=0, atol=2e-5)
    assert not np.allclose(
        baseline.best_estimator_.weights_, fitted.weights_, atol=2e-5
    )
    for key in ["mean_score", "rank"]:
        np.testing.assert_array_equal(
            search.cv_results_[key], baseline.cv_results_[key]
        )
    prediction = search.cv_results_["predictions"][0]
    original_prediction = baseline.cv_results_["predictions"][0]
    assert prediction.n_observations == 20
    np.testing.assert_array_equal(prediction.returns, original_prediction.returns)
    np.testing.assert_array_equal(prediction.observations, X.index[8:28])
    for portfolio, original in zip(prediction, original_prediction, strict=True):
        np.testing.assert_array_equal(portfolio.weights, original.weights)

    if weight_drift:
        # The last scored portfolio supplies holdings; consuming the suffix does
        # not add an unscored rebalance or replace its independently drifted state.
        expected_holdings = prediction[-1].weights * np.prod(1 + values[23:28], axis=0)
        expected_holdings /= expected_holdings.sum()
        np.testing.assert_allclose(
            [fitted.previous_weights[name] for name in X.columns],
            expected_holdings,
            rtol=1e-12,
            atol=1e-15,
        )
    else:
        assert fitted.previous_weights is None

    future = pd.DataFrame([[0.03, -0.01], [-0.02, 0.02]], columns=X.columns)
    np.testing.assert_allclose(
        search.predict(future).returns, future.to_numpy() @ expected_weights, atol=1e-6
    )


def _make_inactive_block(X, start: int, stop: int, assets: tuple[int, ...] = (0, 1)):
    """Create a NaN block with a matching active mask."""
    X_masked = X.copy()
    active_mask = np.ones(X.shape, dtype=bool)
    active_mask[start:stop, list(assets)] = False
    if hasattr(X_masked, "iloc"):
        X_masked.iloc[start:stop, list(assets)] = np.nan
    else:
        X_masked[start:stop, list(assets)] = np.nan
    return X_masked, active_mask


class TestIsPortfolioEstimator:
    def test_optimization_estimator(self):
        assert _is_portfolio_optimization_estimator(_make_online_estimator()) is True

    def test_component_estimator(self):
        assert _is_portfolio_optimization_estimator(EWCovariance()) is False

    def test_pipeline(self):
        from sklearn.pipeline import make_pipeline

        pipe = make_pipeline(
            EmpiricalPrior(
                covariance_estimator=EWCovariance(),
                mu_estimator=EWMu(),
            ),
            _make_online_estimator(),
        )
        assert _is_portfolio_optimization_estimator(pipe) is True


class TestScoreMultiPeriodPortfolio:
    @pytest.fixture()
    def multi_period_portfolio(self, X):
        from skfolio.model_selection import online_predict

        model = _make_online_estimator()
        return online_predict(
            model,
            X,
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
        )

    def test_default_sharpe(self, multi_period_portfolio):
        score = _score_multi_period_portfolio(multi_period_portfolio, None)
        assert isinstance(score, float)
        assert score == pytest.approx(
            multi_period_portfolio.sharpe_ratio,
            rel=1e-6,
        )

    def test_risk_measure_negated(self, multi_period_portfolio):
        score = _score_multi_period_portfolio(
            multi_period_portfolio,
            RiskMeasure.VARIANCE,
        )
        assert score == pytest.approx(
            -multi_period_portfolio.variance,
            rel=1e-6,
        )

    def test_perf_measure_not_negated(self, multi_period_portfolio):
        score = _score_multi_period_portfolio(
            multi_period_portfolio,
            PerfMeasure.MEAN,
        )
        assert score == pytest.approx(multi_period_portfolio.mean, rel=1e-6)


class TestOnlineScorePortfolio:
    def test_default_portfolio_scoring(self, X):
        """Portfolio estimator with default scoring returns Sharpe ratio."""
        model = _make_online_estimator()
        score = online_score(
            model,
            X,
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
        )
        assert isinstance(score, float)

    def test_measure_enum_scoring(self, X):
        """Portfolio estimator scored with a BaseMeasure enum."""
        model = _make_online_estimator()
        score = online_score(
            model,
            X,
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
            scoring=RatioMeasure.SORTINO_RATIO,
        )
        assert isinstance(score, float)

    def test_multi_metric_portfolio(self, X):
        """Portfolio estimator with dict scoring returns dict of floats."""
        model = _make_online_estimator()
        scoring = {
            "sharpe": RatioMeasure.SHARPE_RATIO,
            "variance": RiskMeasure.VARIANCE,
        }
        result = online_score(
            model,
            X,
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
            scoring=scoring,
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {"sharpe", "variance"}
        for v in result.values():
            assert isinstance(v, float)

    def test_per_step_portfolio_raises(self, X):
        """per_step=True with portfolio estimator raises ValueError."""
        model = _make_online_estimator()
        with pytest.raises(ValueError, match="per_step=True is not supported"):
            online_score(
                model,
                X,
                warmup_size=WARMUP,
                test_size=TEST_SIZE,
                per_step=True,
            )

    def test_aggregate_equals_mean_per_step(self, X):
        """For component estimators, aggregate score equals mean of per-step."""
        est = EWCovariance(half_life=30)
        agg = online_score(est, X, warmup_size=WARMUP, test_size=50)
        per = online_score(
            est,
            X,
            warmup_size=WARMUP,
            test_size=50,
            per_step=True,
        )
        assert agg == pytest.approx(float(np.mean(per)), rel=1e-6)


class TestOnlineGridSearch:
    @pytest.mark.filterwarnings(
        "ignore:The covariance matrix is not positive definite:UserWarning"
    )
    def test_metadata_routing_active_mask(self, X):
        """Grid search routes active_mask metadata to partial_fit."""
        X_masked, active_mask = _make_inactive_block(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            routed_search = OnlineGridSearch(
                EWCovariance(),
                param_grid={"half_life": [20, 40]},
                warmup_size=WARMUP,
                test_size=50,
            )
            plain_search = OnlineGridSearch(
                EWCovariance(),
                param_grid={"half_life": [20, 40]},
                warmup_size=WARMUP,
                test_size=50,
            )
            routed_search.fit(X_masked, active_mask=active_mask)
            plain_search.fit(X_masked)

        assert not np.allclose(
            routed_search.cv_results_["mean_score"],
            plain_search.cv_results_["mean_score"],
            equal_nan=True,
        )

    def test_metadata_routing_requires_request(self, X):
        """Grid search respects an explicit None request."""
        X_masked, active_mask = _make_inactive_block(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            search = OnlineGridSearch(
                EWCovariance().set_partial_fit_request(active_mask=None),
                param_grid={"half_life": [20, 40]},
                warmup_size=WARMUP,
                test_size=50,
            )
            with pytest.raises(
                UnsetMetadataPassedError,
                match="OnlineGridSearch\\.fit",
            ):
                search.fit(X_masked, active_mask=active_mask)

    def test_component_estimator(self, X):
        """Grid search over a component estimator."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
        )
        search.fit(X)

        assert hasattr(search, "cv_results_")
        assert hasattr(search, "best_estimator_")
        assert search.best_params_ in [{"half_life": 20}, {"half_life": 40}]
        assert isinstance(search.best_score_, float)
        assert len(search.cv_results_["params"]) == 2
        assert search.cv_results_["rank"].min() == 1
        assert search.cv_results_["rank"].max() == 2

    def test_portfolio_estimator(self, X):
        """Grid search over a portfolio optimization estimator."""
        search = OnlineGridSearch(
            _make_online_estimator(),
            param_grid={
                "prior_estimator__covariance_estimator__half_life": [20, 60],
            },
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
        )
        search.fit(X)

        assert isinstance(search.best_score_, float)
        assert hasattr(search.best_estimator_, "weights_")

    def test_predict_delegates(self, X):
        """predict delegates to best_estimator_."""
        search = OnlineGridSearch(
            _make_online_estimator(),
            param_grid={
                "prior_estimator__covariance_estimator__half_life": [20, 60],
            },
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
        )
        search.fit(X)
        pred = search.predict(X)
        assert pred is not None

    def test_score_delegates(self, X):
        """score delegates to best_estimator_."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
        )
        search.fit(X)
        score = search.score(X)
        assert isinstance(score, float)

    def test_refit_false(self, X):
        """refit=False does not store best_estimator_."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
            refit=False,
        )
        search.fit(X)
        assert not hasattr(search, "best_estimator_")
        assert hasattr(search, "best_params_")

    def test_callable_refit_receives_full_cv_results(self, X):
        """Callable refit receives the full `cv_results_` dictionary."""
        seen = {}

        def refit(results):
            seen["keys"] = set(results)
            return int(np.nanargmax(results["mean_score"]))

        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
            refit=refit,
        )
        search.fit(X)

        assert "mean_score" in seen["keys"]
        assert "fit_time" in seen["keys"]
        assert search.best_index_ == int(np.nanargmax(search.cv_results_["mean_score"]))

    def test_multi_metric(self, X):
        """Grid search with multi-metric scoring."""
        from skfolio.metrics import (
            diagonal_calibration_ratio,
            mahalanobis_calibration_ratio,
        )

        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            scoring={
                "mahalanobis": mahalanobis_calibration_ratio,
                "diagonal": diagonal_calibration_ratio,
            },
            warmup_size=WARMUP,
            test_size=50,
            refit="mahalanobis",
        )
        search.fit(X)

        assert "mean_score_mahalanobis" in search.cv_results_
        assert "mean_score_diagonal" in search.cv_results_
        assert "rank_mahalanobis" in search.cv_results_
        assert "rank_diagonal" in search.cv_results_
        assert search.best_score_ == pytest.approx(
            search.cv_results_["mean_score_mahalanobis"][search.best_index_]
        )

    def test_multi_metric_refit_false(self, X):
        """Multi-metric search with refit=False keeps only cv_results_."""
        from skfolio.metrics import (
            diagonal_calibration_ratio,
            mahalanobis_calibration_ratio,
        )

        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            scoring={
                "mahalanobis": mahalanobis_calibration_ratio,
                "diagonal": diagonal_calibration_ratio,
            },
            warmup_size=WARMUP,
            test_size=50,
            refit=False,
        )
        search.fit(X)

        assert hasattr(search, "cv_results_")
        assert not hasattr(search, "best_index_")
        assert not hasattr(search, "best_params_")
        assert not hasattr(search, "best_score_")
        assert not hasattr(search, "best_estimator_")

    def test_multi_metric_refit_true_raises(self, X):
        """Multi-metric scoring requires an explicit refit metric name."""
        from skfolio.metrics import (
            diagonal_calibration_ratio,
            mahalanobis_calibration_ratio,
        )

        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            scoring={
                "mahalanobis": mahalanobis_calibration_ratio,
                "diagonal": diagonal_calibration_ratio,
            },
            warmup_size=WARMUP,
            test_size=50,
            refit=True,
        )

        with pytest.raises(ValueError, match="For multi-metric scoring"):
            search.fit(X)

    def test_return_predictions(self, X):
        """return_predictions stores MultiPeriodPortfolio per candidate."""
        search = OnlineGridSearch(
            _make_online_estimator(),
            param_grid={
                "prior_estimator__covariance_estimator__half_life": [20, 60],
            },
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
            return_predictions=True,
        )
        search.fit(X)
        preds = search.cv_results_.get("predictions", [])
        assert len(preds) == 2
        from skfolio.portfolio import MultiPeriodPortfolio

        for p in preds:
            assert isinstance(p, MultiPeriodPortfolio)

    def test_weight_drift_routing_retains_fitted_best_estimator(self, X):
        model = _make_online_estimator()
        search = OnlineGridSearch(
            model,
            param_grid={
                "prior_estimator__covariance_estimator__half_life": [40],
            },
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
            return_predictions=True,
            portfolio_params={
                "weight_drift": True,
                "compounded": True,
                "risk_free_rate": 0.001,
            },
        )
        search.fit(X)

        assert model.portfolio_params is None
        assert hasattr(search.best_estimator_, "weights_")
        assert search.best_estimator_.portfolio_params == {"weight_drift": True}
        [prediction] = search.cv_results_["predictions"]
        assert prediction.compounded is True
        assert prediction.risk_free_rate == 0.001
        assert all(portfolio.weight_drift for portfolio in prediction)
        assert all(portfolio.compounded for portfolio in prediction)
        assert all(portfolio.risk_free_rate == 0.001 for portfolio in prediction)

    @pytest.mark.filterwarnings("ignore:Estimator fit failed:UserWarning")
    def test_return_predictions_aligns_failed_candidates(self, X):
        """Failed candidates keep a `None` placeholder in predictions."""
        from skfolio.portfolio import MultiPeriodPortfolio

        search = OnlineGridSearch(
            _make_online_estimator(),
            param_grid={
                "prior_estimator__covariance_estimator__half_life": [20, -1],
            },
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
            return_predictions=True,
            error_score=np.nan,
        )
        with pytest.warns(UserWarning, match="Estimator fit failed"):
            search.fit(X)

        preds = search.cv_results_["predictions"]
        assert len(preds) == 2
        assert isinstance(preds[0], MultiPeriodPortfolio)
        assert preds[1] is None

    def test_return_predictions_ignored_for_component_estimators(self, X):
        """Component estimators do not expose predictions in cv_results_."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
            return_predictions=True,
        )
        search.fit(X)

        assert "predictions" not in search.cv_results_

    @pytest.mark.filterwarnings("ignore:Estimator fit failed:UserWarning")
    def test_multi_metric_error_score(self, X):
        """Multi-metric failures use per-metric error scores instead of crashing."""
        from skfolio.metrics import (
            diagonal_calibration_ratio,
            mahalanobis_calibration_ratio,
        )

        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, -1]},
            scoring={
                "mahalanobis": mahalanobis_calibration_ratio,
                "diagonal": diagonal_calibration_ratio,
            },
            warmup_size=WARMUP,
            test_size=50,
            refit="mahalanobis",
            error_score=np.nan,
        )
        with pytest.warns(UserWarning, match="Estimator fit failed"):
            search.fit(X)

        assert np.isnan(search.cv_results_["mean_score_mahalanobis"][1])
        assert np.isnan(search.cv_results_["mean_score_diagonal"][1])

    @pytest.mark.filterwarnings("ignore:Estimator fit failed:UserWarning")
    def test_all_candidates_failed_raises(self, X):
        """All failing candidates raise instead of selecting a bogus best one."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [-1, -2]},
            warmup_size=WARMUP,
            test_size=50,
            error_score=np.nan,
        )

        with pytest.warns(UserWarning, match="Estimator fit failed"):
            with pytest.raises(ValueError, match="All parameter candidates failed"):
                search.fit(X)

    def test_best_estimator_is_refitted(self, X):
        """Best estimator from search is fully trained (refit_last=True)."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
        )
        search.fit(X)
        assert hasattr(search.best_estimator_, "covariance_")

    def test_not_fitted_predict_raises(self, X):
        """predict before fit raises."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
        )
        with pytest.raises(
            NotFittedError, match="This OnlineGridSearch instance is not fitted yet"
        ):
            search.predict(X)

    def test_pipeline_raises(self, X):
        """Pipelines are rejected explicitly for online search."""
        search = OnlineGridSearch(
            make_pipeline(EWCovariance()),
            param_grid={"ewcovariance__half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
        )
        with pytest.raises(TypeError, match="Pipeline is not supported"):
            search.fit(X)

    def test_list_param_grid(self, X):
        """param_grid as a list of dicts works."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid=[{"half_life": [20]}, {"half_life": [40]}],
            warmup_size=WARMUP,
            test_size=50,
        )
        search.fit(X)
        assert len(search.cv_results_["params"]) == 2


class TestOnlineRandomizedSearch:
    @pytest.mark.filterwarnings(
        "ignore:The covariance matrix is not positive definite:UserWarning"
    )
    def test_metadata_routing_active_mask(self, X):
        """Randomized search routes active_mask metadata to partial_fit."""
        X_masked, active_mask = _make_inactive_block(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            routed_search = OnlineRandomizedSearch(
                EWCovariance(),
                param_distributions={"half_life": [20, 40, 60]},
                n_iter=2,
                warmup_size=WARMUP,
                test_size=50,
                random_state=0,
            )
            plain_search = OnlineRandomizedSearch(
                EWCovariance(),
                param_distributions={"half_life": [20, 40, 60]},
                n_iter=2,
                warmup_size=WARMUP,
                test_size=50,
                random_state=0,
            )
            routed_search.fit(X_masked, active_mask=active_mask)
            plain_search.fit(X_masked)

        assert not np.allclose(
            routed_search.cv_results_["mean_score"],
            plain_search.cv_results_["mean_score"],
            equal_nan=True,
        )

    def test_metadata_routing_requires_request(self, X):
        """Randomized search respects an explicit None request."""
        X_masked, active_mask = _make_inactive_block(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            search = OnlineRandomizedSearch(
                EWCovariance().set_partial_fit_request(active_mask=None),
                param_distributions={"half_life": [20, 40, 60]},
                n_iter=2,
                warmup_size=WARMUP,
                test_size=50,
                random_state=0,
            )
            with pytest.raises(
                UnsetMetadataPassedError,
                match="OnlineRandomizedSearch\\.fit",
            ):
                search.fit(X_masked, active_mask=active_mask)

    def test_basic(self, X):
        """Randomized search evaluates n_iter candidates."""
        search = OnlineRandomizedSearch(
            EWCovariance(),
            param_distributions={"half_life": [10, 20, 30, 40, 50, 60]},
            n_iter=3,
            warmup_size=WARMUP,
            test_size=50,
            random_state=42,
        )
        search.fit(X)

        assert len(search.cv_results_["params"]) == 3
        assert isinstance(search.best_score_, float)
        assert hasattr(search, "best_estimator_")

    def test_random_state_reproducible(self, X):
        """Same random_state produces same candidates."""
        kwargs = dict(
            param_distributions={"half_life": [10, 20, 30, 40, 50, 60]},
            n_iter=3,
            warmup_size=WARMUP,
            test_size=50,
            random_state=42,
        )
        s1 = OnlineRandomizedSearch(EWCovariance(), **kwargs)
        s2 = OnlineRandomizedSearch(EWCovariance(), **kwargs)
        s1.fit(X)
        s2.fit(X)

        assert s1.cv_results_["params"] == s2.cv_results_["params"]
        np.testing.assert_array_equal(
            s1.cv_results_["mean_score"],
            s2.cv_results_["mean_score"],
        )

    def test_portfolio_estimator(self, X):
        """Randomized search works with portfolio estimators."""
        search = OnlineRandomizedSearch(
            _make_online_estimator(),
            param_distributions={
                "prior_estimator__covariance_estimator__half_life": [20, 40, 60],
            },
            n_iter=2,
            warmup_size=WARMUP,
            test_size=TEST_SIZE,
            random_state=0,
        )
        search.fit(X)
        assert isinstance(search.best_score_, float)


@pytest.mark.parametrize("search_cls", [OnlineGridSearch, OnlineRandomizedSearch])
def test_search_entry_rebalancing_params(search_cls, X):
    """Search entry parameters only affect the first candidate portfolio."""
    transaction_costs = 0.001
    search_params = (
        {"param_grid": {"transaction_costs": [transaction_costs]}}
        if search_cls is OnlineGridSearch
        else {
            "param_distributions": {"transaction_costs": [transaction_costs]},
            "n_iter": 1,
            "random_state": 0,
        }
    )
    search = search_cls(
        _make_online_estimator(),
        **search_params,
        warmup_size=WARMUP,
        test_size=TEST_SIZE,
        return_predictions=True,
        entry_rebalancing_params={"transaction_costs": 0.0, "fallback": None},
    )

    search.fit(X)

    prediction = search.cv_results_["predictions"][0]
    assert len(prediction) >= 2
    assert prediction[0].transaction_costs == 0.0
    assert all(ptf.transaction_costs == transaction_costs for ptf in prediction[1:])
    assert search.best_estimator_.transaction_costs == transaction_costs


def test_search_invalid_entry_rebalancing_param_raises(X):
    """Invalid entry parameter names raise before candidate evaluation."""
    search = OnlineGridSearch(
        _make_online_estimator(),
        param_grid={"risk_aversion": [1.0]},
        warmup_size=WARMUP,
        test_size=TEST_SIZE,
        entry_rebalancing_params={"not_a_parameter": 1},
    )

    with pytest.raises(ValueError, match="contains invalid parameter names"):
        search.fit(X)


def test_search_entry_rebalancing_params_rejects_component_estimator(X):
    """Search entry parameters require a portfolio optimization estimator."""
    search = OnlineGridSearch(
        EWCovariance(),
        param_grid={"half_life": [20]},
        warmup_size=WARMUP,
        test_size=50,
        entry_rebalancing_params={"half_life": 10},
    )

    with pytest.raises(ValueError, match="entry_rebalancing_params"):
        search.fit(X)


class TestValidateSizes:
    def test_float_warmup_size_raises(self):
        with pytest.raises(TypeError, match="warmup_size must be an integer"):
            _validate_sizes(252.5, 5)

    def test_float_test_size_raises(self):
        with pytest.raises(TypeError, match="test_size must be an integer"):
            _validate_sizes(252, 5.0)

    def test_bool_warmup_size_raises(self):
        with pytest.raises(TypeError, match="warmup_size must be an integer"):
            _validate_sizes(True, 5)

    def test_bool_test_size_raises(self):
        with pytest.raises(TypeError, match="test_size must be an integer"):
            _validate_sizes(252, False)

    def test_zero_warmup_size_raises(self):
        with pytest.raises(ValueError, match="warmup_size must be >= 1"):
            _validate_sizes(0, 5)

    def test_zero_test_size_raises(self):
        with pytest.raises(ValueError, match="test_size must be >= 1"):
            _validate_sizes(252, 0)

    def test_valid_sizes_passes(self):
        _validate_sizes(252, 5)

    def test_numpy_int_passes(self):
        _validate_sizes(np.int64(252), np.int32(5))


class TestRankScores:
    def test_ties_share_the_same_rank(self):
        ranks = _rank_scores(np.array([0.7, 0.7, np.nan, 0.5]))
        np.testing.assert_array_equal(ranks, np.array([1, 1, 4, 3], dtype=np.int32))


class TestSearchFloatSizeRejection:
    def test_grid_search_float_warmup_raises(self, X):
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20]},
            warmup_size=252.5,
            test_size=50,
        )
        with pytest.raises(TypeError, match="warmup_size must be an integer"):
            search.fit(X)

    def test_grid_search_float_test_size_raises(self, X):
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20]},
            warmup_size=WARMUP,
            test_size=50.0,
        )
        with pytest.raises(TypeError, match="test_size must be an integer"):
            search.fit(X)


def test_rank_scores_empty():
    ranks = _rank_scores(np.array([]))
    assert ranks.dtype == np.int32
    assert ranks.shape == (0,)


def test_check_refit_for_multimetric_false_is_accepted():
    assert _check_refit_for_multimetric(False, {"a": None, "b": None}) is None


class TestOnlineSearchReuse:
    @pytest.mark.parametrize("search_cls", [OnlineGridSearch, OnlineRandomizedSearch])
    @pytest.mark.parametrize(
        "mode",
        [
            "single_false",
            "multi_false",
            "single_callable",
            "multi_callable",
            "multi_named",
            "enabled",
        ],
    )
    def test_current_fit_attributes(self, search_cls, mode):
        """Reused searches expose the documented current-mode attributes only."""
        X = np.random.default_rng(931).normal(0, 0.02, (24, 3))
        current = 0.5 * X + 0.005
        search = _single_candidate_search(
            search_cls,
            EWMu(half_life=3),
            scoring=_mean_squared_error,
            warmup_size=8,
            test_size=4,
        ).fit(X)
        previous = search.best_estimator_
        previous_mu = previous.mu_.copy()
        config = {"refit": True}
        if "multi" in mode:
            config["scoring"] = {"a": _mean_squared_error, "b": _mean_squared_error}
        if mode.endswith("false"):
            config["refit"] = False
        elif mode.endswith("callable"):
            config["refit"] = lambda results: 0
        elif mode == "multi_named":
            config["refit"] = "a"
        fresh = clone(search).set_params(**config).fit(current)
        search.set_params(**config).fit(current)
        expected = {
            "best_index_": mode != "multi_false",
            "best_params_": mode != "multi_false",
            "best_score_": mode
            not in ("multi_false", "single_callable", "multi_callable"),
            "best_estimator_": not mode.endswith("false"),
        }
        for name, present in expected.items():
            assert hasattr(fresh, name) == present
            assert hasattr(search, name) == present
        assert search.cv_results_.keys() == fresh.cv_results_.keys()
        for key, value in fresh.cv_results_.items():
            if key != "fit_time":  # Wall-clock durations are not fitted results.
                np.testing.assert_equal(search.cv_results_[key], value)
        if expected["best_index_"]:
            assert search.best_index_ == 0
            assert search.best_params_ == {}
        if expected["best_score_"]:
            key = "mean_score_a" if "multi" in mode else "mean_score"
            assert search.best_score_ == pytest.approx(fresh.cv_results_[key][0])
        if expected["best_estimator_"]:
            # Independent normalized geometric sums, not the estimator recurrence.
            decay = 0.5 ** (1 / 3)
            weights = np.array([decay ** (23 - i) for i in range(24)])
            np.testing.assert_allclose(
                search.best_estimator_.mu_,
                weights @ current / weights.sum(),
                rtol=1e-12,
                atol=1e-15,
            )
            assert search.best_estimator_ is not previous
        np.testing.assert_array_equal(previous.mu_, previous_mu)

    @pytest.mark.parametrize("search_cls", [OnlineGridSearch, OnlineRandomizedSearch])
    @pytest.mark.parametrize("multimetric", [False, True])
    def test_selected_failed_candidate_removes_old_model(self, search_cls, multimetric):
        """A completed warning path must not expose an earlier successful model."""
        X = np.random.default_rng(931).normal(0, 0.02, (24, 3))
        search = _single_candidate_search(
            search_cls,
            EWMu(half_life=3),
            scoring=_mean_squared_error,
            warmup_size=8,
            test_size=4,
        ).fit(X)
        params = {"half_life": [3, -1]}
        config = (
            {"param_grid": params}
            if search_cls is OnlineGridSearch
            else {"param_distributions": params, "n_iter": 2}
        )
        config.update(
            error_score=np.nan,
            refit=lambda results: next(
                i for i, p in enumerate(results["params"]) if p["half_life"] == -1
            ),
        )
        if multimetric:
            config["scoring"] = {"a": _mean_squared_error, "b": _mean_squared_error}
        with pytest.warns(UserWarning) as observed:
            search.set_params(**config).fit(0.5 * X + 0.005)
        assert any(
            "`best_estimator_` is not available" in str(w.message) for w in observed
        )
        assert search.best_params_ == {"half_life": -1}
        assert search.cv_results_["params"][search.best_index_] == {"half_life": -1}
        assert not hasattr(search, "best_score_")
        assert not hasattr(search, "best_estimator_")

    @pytest.mark.parametrize("search_cls", [OnlineGridSearch, OnlineRandomizedSearch])
    def test_disabled_refit_cannot_delegate_to_previous_optimizer(self, search_cls):
        """Public prediction and score cannot use a previously selected allocation."""
        X = pd.DataFrame(
            np.random.default_rng(931).normal(0, 0.02, (24, 3)),
            index=pd.bdate_range("2024-01-01", periods=24),
            columns=["A", "B", "C"],
        )
        current = 0.5 * X + 0.005
        estimator = _make_online_estimator().set_params(
            prior_estimator__mu_estimator__half_life=3,
            prior_estimator__covariance_estimator__half_life=3,
        )
        search = _single_candidate_search(
            search_cls,
            estimator,
            scoring=PerfMeasure.MEAN,
            warmup_size=8,
            test_size=4,
            return_predictions=True,
        ).fit(X)
        previous = search.best_estimator_
        previous_weights = previous.weights_.copy()
        expected_returns = current.to_numpy() @ previous_weights
        np.testing.assert_allclose(search.predict(current).returns, expected_returns)
        search.set_params(refit=False).fit(current)
        for method in (search.predict, search.score):
            with pytest.raises(NotFittedError):
                method(current)
        fresh = clone(search).fit(current)
        prediction = search.cv_results_["predictions"][0]
        np.testing.assert_allclose(
            prediction.returns, fresh.cv_results_["predictions"][0].returns
        )
        np.testing.assert_array_equal(prediction.observations, current.index[8:])
        assert search.best_score_ == pytest.approx(np.mean(prediction.returns))
        search.set_params(refit=True).fit(current)
        assert search.best_estimator_ is not previous
        np.testing.assert_allclose(
            search.predict(current).returns,
            current.to_numpy() @ search.best_estimator_.weights_,
        )
        np.testing.assert_array_equal(previous.weights_, previous_weights)
        np.testing.assert_allclose(previous.predict(current).returns, expected_returns)


class TestOnlineGridSearchRefitEdgeCases:
    def test_multi_metric_callable_refit(self, X):
        """Multi-metric search with a callable refit selects by the callable."""
        from skfolio.metrics import (
            diagonal_calibration_ratio,
            mahalanobis_calibration_ratio,
        )

        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            scoring={
                "mahalanobis": mahalanobis_calibration_ratio,
                "diagonal": diagonal_calibration_ratio,
            },
            warmup_size=WARMUP,
            test_size=50,
            refit=lambda cv_results: 1,
        )
        search.fit(X)

        assert search.best_index_ == 1
        assert search.best_params_ == {"half_life": 40}
        assert not hasattr(search, "best_score_")
        assert search.best_estimator_.half_life == 40

    def test_callable_refit_non_integer_raises(self, X):
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
            refit=lambda cv_results: 0.5,
        )
        with pytest.raises(TypeError, match="best_index_ returned is not an integer"):
            search.fit(X)

    def test_callable_refit_out_of_range_raises(self, X):
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, 40]},
            warmup_size=WARMUP,
            test_size=50,
            refit=lambda cv_results: 2,
        )
        with pytest.raises(IndexError, match="best_index_ index out of range"):
            search.fit(X)

    @pytest.mark.filterwarnings("ignore:Estimator fit failed:UserWarning")
    def test_callable_refit_selecting_failed_candidate_warns(self, X):
        """Selecting a failed candidate leaves `best_estimator_` unset."""
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20, -1]},
            warmup_size=WARMUP,
            test_size=50,
            refit=lambda cv_results: 1,
            error_score=np.nan,
        )
        with pytest.warns(UserWarning, match="`best_estimator_` is not available"):
            search.fit(X)

        assert search.best_index_ == 1
        assert not hasattr(search, "best_estimator_")

    def test_error_score_raise_propagates(self, X):
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [-1]},
            warmup_size=WARMUP,
            test_size=50,
            error_score="raise",
        )
        with pytest.raises(ValueError, match="half_life must be positive"):
            search.fit(X)

    @pytest.mark.parametrize("error_score", ["foo", True])
    def test_invalid_error_score_raises(self, X, error_score):
        search = OnlineGridSearch(
            EWCovariance(),
            param_grid={"half_life": [20]},
            warmup_size=WARMUP,
            test_size=50,
            error_score=error_score,
        )
        with pytest.raises(
            ValueError, match="error_score must be the string 'raise' or a real number"
        ):
            search.fit(X)
