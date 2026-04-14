"""Tests for online_score portfolio scoring and OnlineGridSearch / OnlineRandomizedSearch."""

import numpy as np
import pytest
from sklearn import config_context
from sklearn.exceptions import NotFittedError, UnsetMetadataPassedError
from sklearn.pipeline import make_pipeline

from skfolio.measures import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.model_selection import (
    OnlineGridSearch,
    OnlineRandomizedSearch,
    online_score,
)
from skfolio.model_selection._online._search import _rank_scores
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
                EWCovariance().set_partial_fit_request(active_mask=True),
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
        """Grid search raises when metadata is passed without a request."""
        X_masked, active_mask = _make_inactive_block(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            search = OnlineGridSearch(
                EWCovariance(),
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
        """Callable refit receives the full ``cv_results_`` dictionary."""
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

    @pytest.mark.filterwarnings("ignore:Estimator fit failed:UserWarning")
    def test_return_predictions_aligns_failed_candidates(self, X):
        """Failed candidates keep a ``None`` placeholder in predictions."""
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
        with pytest.raises(NotFittedError):
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
                EWCovariance().set_partial_fit_request(active_mask=True),
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
        """Randomized search raises when metadata is passed without a request."""
        X_masked, active_mask = _make_inactive_block(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            search = OnlineRandomizedSearch(
                EWCovariance(),
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
