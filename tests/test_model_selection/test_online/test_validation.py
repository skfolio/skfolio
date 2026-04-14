"""Test online_predict and online_score modules."""

from functools import partial

import numpy as np
import pytest
import sklearn as sk
from sklearn import config_context
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.pipeline import make_pipeline

from skfolio import MultiPeriodPortfolio
from skfolio.metrics import (
    diagonal_calibration_ratio,
    mahalanobis_calibration_ratio,
    portfolio_variance_calibration_ratio,
    portfolio_variance_qlike_loss,
)
from skfolio.model_selection import (
    WalkForward,
    cross_val_predict,
    online_predict,
    online_score,
)
from skfolio.moments import EWCovariance, EWMu, RegimeAdjustedEWCovariance
from skfolio.optimization import InverseVolatility, MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior
from skfolio.uncertainty_set import (
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)


def _make_online_estimator(**kwargs):
    """Build a MeanRisk estimator with online (EW) sub-estimators."""
    return MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40),
            covariance_estimator=EWCovariance(half_life=40),
        ),
        **kwargs,
    )


def _make_regime_online_estimator(**kwargs):
    """Build a MeanRisk estimator with regime-adjusted covariance."""
    return MeanRisk(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(half_life=40),
            covariance_estimator=RegimeAdjustedEWCovariance(),
        ),
        **kwargs,
    )


def _make_inactive_block(
    X,
    start: int,
    stop: int,
    assets: tuple[int, ...] = (0, 1),
    *,
    insert_nan: bool = True,
):
    """Create a NaN block with a matching active mask."""
    X_masked = X.copy()
    active_mask = np.ones(X.shape, dtype=bool)
    active_mask[start:stop, list(assets)] = False
    if insert_nan:
        if hasattr(X_masked, "iloc"):
            X_masked.iloc[start:stop, list(assets)] = np.nan
        else:
            X_masked[start:stop, list(assets)] = np.nan
    return X_masked, active_mask


def _make_estimation_mask(
    X, start: int, stop: int, assets: tuple[int, ...] = (0, 1)
) -> np.ndarray:
    """Create an estimation mask that excludes some assets on a block."""
    estimation_mask = np.ones(X.shape, dtype=bool)
    estimation_mask[start:stop, list(assets)] = False
    return estimation_mask


class TestOnlinePredict:
    def test_basic(self, X):
        """online_predict returns a MultiPeriodPortfolio with correct structure."""
        model = _make_online_estimator()
        warmup = 400
        test_size = 100

        pred = online_predict(
            model,
            X,
            warmup_size=warmup,
            test_size=test_size,
            portfolio_params=dict(name="online_test"),
        )

        assert isinstance(pred, MultiPeriodPortfolio)
        assert pred.name == "online_test"
        assert len(pred.portfolios) > 0
        for ptf in pred.portfolios:
            assert ptf.n_observations > 0

    def test_matches_cross_val_predict(self, X):
        """online_predict matches cross_val_predict with expanding WalkForward."""
        warmup = 400
        test_size = 300
        model = _make_online_estimator()

        cv = WalkForward(
            test_size=test_size,
            train_size=warmup,
            expand_train=True,
            reduce_test=True,
        )
        ref = cross_val_predict(sk.clone(model), X, cv=cv)

        pred = online_predict(
            model, X, warmup_size=warmup, test_size=test_size, reduce_test=True
        )

        assert len(pred.portfolios) == len(ref.portfolios)
        for p_pred, p_ref in zip(pred.portfolios, ref.portfolios, strict=True):
            np.testing.assert_array_almost_equal(p_pred.weights, p_ref.weights)
            np.testing.assert_array_almost_equal(np.asarray(p_pred), np.asarray(p_ref))

    def test_expanding_equivalence(self, X):
        """online_predict reproduces a manual expanding fit/predict loop."""
        warmup = 400
        test_size = 300

        model = _make_online_estimator()
        pred = online_predict(
            model, X, warmup_size=warmup, test_size=test_size, reduce_test=True
        )

        model_manual = sk.clone(model)
        cv = WalkForward(
            test_size=test_size,
            train_size=warmup,
            expand_train=True,
            reduce_test=True,
        )
        splits = list(cv.split(X))

        train_0 = splits[0][0]
        warmup_end = int(train_0[-1]) + 1
        model_manual.fit(X.iloc[:warmup_end])

        manual_portfolios = []
        update_frontier = warmup_end
        for train_idx, test_idx in splits:
            train_end = int(train_idx[-1]) + 1
            test_start = int(test_idx[0])
            test_end = int(test_idx[-1]) + 1

            if train_end > update_frontier:
                model_manual.partial_fit(X.iloc[update_frontier:train_end])
                update_frontier = train_end

            ptf = model_manual.predict(X.iloc[test_start:test_end])
            manual_portfolios.append(ptf)

        assert len(pred.portfolios) == len(manual_portfolios)
        for p_pred, p_man in zip(pred.portfolios, manual_portfolios, strict=True):
            np.testing.assert_array_almost_equal(p_pred.weights, p_man.weights)
            np.testing.assert_array_almost_equal(np.asarray(p_pred), np.asarray(p_man))

    def test_test_size_1(self, X):
        """online_predict works with daily rebalancing (test_size=1)."""
        model = _make_online_estimator()
        n_obs = X.shape[0]
        warmup = n_obs - 10

        pred = online_predict(model, X, warmup_size=warmup, test_size=1)

        assert isinstance(pred, MultiPeriodPortfolio)
        assert len(pred.portfolios) == 10
        for ptf in pred.portfolios:
            assert ptf.n_observations == 1

    def test_reduce_test(self, X):
        """reduce_test=True includes a partial last window."""
        model = _make_online_estimator()
        n_obs = X.shape[0]
        warmup = 400
        test_size = 300

        remaining = n_obs - warmup
        n_full = remaining // test_size
        has_partial = remaining % test_size > 0

        pred_no_reduce = online_predict(
            model, X, warmup_size=warmup, test_size=test_size, reduce_test=False
        )
        pred_reduce = online_predict(
            model, X, warmup_size=warmup, test_size=test_size, reduce_test=True
        )

        assert len(pred_no_reduce.portfolios) == n_full
        if has_partial:
            assert len(pred_reduce.portfolios) == n_full + 1
            assert pred_reduce.portfolios[-1].n_observations < test_size
        else:
            assert len(pred_reduce.portfolios) == n_full

    def test_purged_size(self, X):
        """purged_size creates a gap between model knowledge and test window."""
        model = _make_online_estimator()
        warmup = 400
        test_size = 50

        pred_no_purge = online_predict(
            model, X, warmup_size=warmup, test_size=test_size, purged_size=0
        )
        pred_purge = online_predict(
            model, X, warmup_size=warmup, test_size=test_size, purged_size=5
        )

        assert len(pred_purge.portfolios) > 0

        returns_no_purge = np.concatenate(
            [np.asarray(p) for p in pred_no_purge.portfolios]
        )
        returns_purge = np.concatenate([np.asarray(p) for p in pred_purge.portfolios])
        assert not np.array_equal(returns_no_purge, returns_purge)

    def test_previous_weights(self, X):
        """previous_weights propagation works correctly across steps."""
        model = _make_online_estimator(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            transaction_costs=0.001,
        )
        assert model.needs_previous_weights is True

        warmup = 400
        test_size = 300

        pred = online_predict(model, X, warmup_size=warmup, test_size=test_size)

        assert len(pred.portfolios) >= 2
        assert np.all(pred[0].previous_weights == 0)

        for i in range(1, len(pred.portfolios)):
            np.testing.assert_almost_equal(
                pred[i - 1].weights, pred[i].previous_weights
            )

    def test_previous_weights_differ(self, X):
        """Transaction costs with previous_weights produce different portfolios."""
        warmup = 400
        test_size = 300

        model_ref = _make_online_estimator(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        )
        model_tc = _make_online_estimator(
            objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
            transaction_costs=0.001,
        )

        pred_ref = online_predict(model_ref, X, warmup_size=warmup, test_size=test_size)
        pred_tc = online_predict(model_tc, X, warmup_size=warmup, test_size=test_size)

        any_diff = any(
            not np.allclose(p1.weights, p2.weights)
            for p1, p2 in zip(pred_ref.portfolios, pred_tc.portfolios, strict=True)
        )
        assert any_diff

    def test_estimator_not_mutated(self, X):
        """The input estimator is not mutated (cloned internally)."""
        model = _make_online_estimator()
        assert not hasattr(model, "weights_")

        online_predict(model, X, warmup_size=400, test_size=300)

        assert not hasattr(model, "weights_")

    def test_no_partial_fit_raises(self, X):
        """Raises TypeError when the estimator lacks partial_fit."""
        model = InverseVolatility()
        with pytest.raises(TypeError, match="partial_fit"):
            online_predict(model, X, warmup_size=400, test_size=100)

    def test_non_portfolio_estimator_raises(self, X):
        """Raises TypeError when the estimator is not a portfolio optimizer."""
        with pytest.raises(
            TypeError,
            match="online_predict` only supports portfolio optimization estimators",
        ):
            online_predict(EWCovariance(half_life=30), X, warmup_size=400, test_size=50)

    def test_pipeline_raises(self, X):
        """Pipelines are rejected explicitly for online evaluation."""
        model = make_pipeline(_make_online_estimator())
        with pytest.raises(TypeError, match="Pipeline is not supported"):
            online_predict(model, X, warmup_size=400, test_size=100)

    @pytest.mark.parametrize(
        ("param_name", "estimator"),
        [
            ("mu_uncertainty_set_estimator", EmpiricalMuUncertaintySet()),
            (
                "covariance_uncertainty_set_estimator",
                EmpiricalCovarianceUncertaintySet(),
            ),
        ],
    )
    def test_uncertainty_set_without_partial_fit_raises(self, X, param_name, estimator):
        model = _make_online_estimator(**{param_name: estimator})

        with pytest.raises(TypeError, match=param_name):
            online_predict(model, X, warmup_size=400, test_size=100)

    def test_bad_warmup_raises(self, X):
        """Raises ValueError for invalid warmup_size."""
        model = _make_online_estimator()
        with pytest.raises(ValueError, match="warmup_size"):
            online_predict(model, X, warmup_size=0, test_size=1)

    def test_bad_test_size_raises(self, X):
        """Raises ValueError for invalid test_size."""
        model = _make_online_estimator()
        with pytest.raises(ValueError, match="test_size"):
            online_predict(model, X, warmup_size=400, test_size=0)

    def test_float_warmup_raises(self, X):
        """Raises TypeError for float warmup_size."""
        model = _make_online_estimator()
        with pytest.raises(TypeError, match="warmup_size must be an integer"):
            online_predict(model, X, warmup_size=252.5, test_size=1)

    def test_float_test_size_raises(self, X):
        """Raises TypeError for float test_size."""
        model = _make_online_estimator()
        with pytest.raises(TypeError, match="test_size must be an integer"):
            online_predict(model, X, warmup_size=400, test_size=1.0)

    def test_insufficient_data_raises(self, X):
        """Raises ValueError when data is too short."""
        model = _make_online_estimator()
        with pytest.raises(ValueError):
            online_predict(model, X, warmup_size=X.shape[0], test_size=1)

    def test_freq(self, X):
        """online_predict works with freq-based rebalancing."""
        model = _make_online_estimator()

        pred = online_predict(
            model,
            X,
            warmup_size=6,
            test_size=1,
            freq="MS",
        )

        assert isinstance(pred, MultiPeriodPortfolio)
        assert len(pred.portfolios) > 0

    def test_metadata_routing_active_mask(self, X):
        """online_predict routes estimation_mask metadata to partial_fit."""
        estimation_mask = _make_estimation_mask(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            routed_model = _make_regime_online_estimator()
            routed_model.prior_estimator.covariance_estimator.set_partial_fit_request(
                estimation_mask=True
            )
            pred_routed = online_predict(
                routed_model,
                X,
                warmup_size=400,
                test_size=50,
                reduce_test=True,
                params={"estimation_mask": estimation_mask},
            )
            pred_plain = online_predict(
                _make_regime_online_estimator(),
                X,
                warmup_size=400,
                test_size=50,
                reduce_test=True,
            )

        assert len(pred_routed.portfolios) == len(pred_plain.portfolios)
        assert any(
            not np.allclose(
                p_routed.weights,
                p_plain.weights,
                equal_nan=True,
            )
            for p_routed, p_plain in zip(
                pred_routed.portfolios, pred_plain.portfolios, strict=True
            )
        )

    def test_metadata_routing_requires_request(self, X):
        """online_predict raises when metadata is passed without a request."""
        estimation_mask = _make_estimation_mask(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            with pytest.raises(UnsetMetadataPassedError, match="partial_fit"):
                online_predict(
                    _make_regime_online_estimator(),
                    X,
                    warmup_size=400,
                    test_size=50,
                    reduce_test=True,
                    params={"estimation_mask": estimation_mask},
                )


class TestOnlineScore:
    def test_default_scoring_aggregate(self, X):
        """online_score returns an aggregate float by default."""
        est = EWCovariance(half_life=30)
        score = online_score(est, X, warmup_size=400, test_size=50)

        assert isinstance(score, float)

    def test_default_scoring_per_step(self, X):
        """online_score with per_step=True returns an ndarray."""
        est = EWCovariance(half_life=30)
        scores = online_score(
            est,
            X,
            warmup_size=400,
            test_size=50,
            per_step=True,
        )

        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
        assert len(scores) > 0

    def test_single_scorer(self, X):
        """online_score with a single scorer and per_step returns an ndarray."""
        est = EWCovariance(half_life=30)
        scores = online_score(
            est,
            X,
            warmup_size=400,
            test_size=50,
            scoring=mahalanobis_calibration_ratio,
            per_step=True,
        )

        assert isinstance(scores, np.ndarray)
        assert scores.ndim == 1
        assert len(scores) > 0
        assert all(s > 0 for s in scores)

    def test_multi_scorer(self, X):
        """online_score with a dict of scorers returns a dict."""
        est = EWCovariance(half_life=30)
        scoring = {
            "mahalanobis": mahalanobis_calibration_ratio,
            "diagonal": diagonal_calibration_ratio,
            "portfolio": portfolio_variance_calibration_ratio,
            "qlike": portfolio_variance_qlike_loss,
        }
        result = online_score(
            est,
            X,
            warmup_size=400,
            test_size=50,
            scoring=scoring,
            per_step=True,
        )

        assert isinstance(result, dict)
        assert set(result.keys()) == set(scoring.keys())
        for _name, arr in result.items():
            assert isinstance(arr, np.ndarray)
            assert arr.ndim == 1
            assert len(arr) > 0

    def test_multi_horizon(self, X):
        """online_score with different test_size produces different results."""
        est = EWCovariance(half_life=30)
        scores_1d = online_score(
            est,
            X,
            warmup_size=400,
            test_size=1,
            scoring=portfolio_variance_calibration_ratio,
            per_step=True,
        )
        scores_5d = online_score(
            est,
            X,
            warmup_size=400,
            test_size=5,
            scoring=portfolio_variance_calibration_ratio,
            per_step=True,
        )

        assert len(scores_1d) > len(scores_5d)

    def test_estimator_not_mutated(self, X):
        """The input estimator is not mutated."""
        est = EWCovariance(half_life=30)
        assert not hasattr(est, "covariance_")

        online_score(est, X, warmup_size=400, test_size=50)

        assert not hasattr(est, "covariance_")

    def test_no_partial_fit_raises(self, X):
        """Raises TypeError when the estimator lacks partial_fit."""
        from skfolio.moments import EmpiricalCovariance

        est = EmpiricalCovariance()
        with pytest.raises(TypeError, match="partial_fit"):
            online_score(est, X, warmup_size=400, test_size=50)

    def test_pipeline_raises(self, X):
        """Pipelines are rejected explicitly for online evaluation."""
        est = make_pipeline(EWCovariance(half_life=30))
        with pytest.raises(TypeError, match="Pipeline is not supported"):
            online_score(est, X, warmup_size=400, test_size=50)

    def test_bad_warmup_raises(self, X):
        """Raises ValueError for invalid warmup_size."""
        est = EWCovariance(half_life=30)
        with pytest.raises(ValueError, match="warmup_size"):
            online_score(est, X, warmup_size=0, test_size=1)

    def test_insufficient_data_raises(self, X):
        """Raises ValueError when data is too short."""
        est = EWCovariance(half_life=30)
        with pytest.raises(ValueError):
            online_score(est, X, warmup_size=X.shape[0], test_size=1)

    def test_float_warmup_raises(self, X):
        """Raises TypeError for float warmup_size."""
        est = EWCovariance(half_life=30)
        with pytest.raises(TypeError, match="warmup_size must be an integer"):
            online_score(est, X, warmup_size=252.5, test_size=1)

    def test_float_test_size_raises(self, X):
        """Raises TypeError for float test_size."""
        est = EWCovariance(half_life=30)
        with pytest.raises(TypeError, match="test_size must be an integer"):
            online_score(est, X, warmup_size=400, test_size=50.0)

    def test_partial_weights_scorer(self, X):
        """Scorer with extra args via functools.partial works correctly."""
        est = EWCovariance(half_life=30)
        n_assets = X.shape[1]
        w = np.ones(n_assets) / n_assets
        scorer = partial(portfolio_variance_calibration_ratio, portfolio_weights=w)

        scores = online_score(
            est,
            X,
            warmup_size=400,
            test_size=50,
            scoring=scorer,
            per_step=True,
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) > 0

    def test_with_prior_estimator(self, X):
        """online_score works with a prior estimator (EmpiricalPrior)."""
        est = EmpiricalPrior(
            covariance_estimator=EWCovariance(half_life=30),
            mu_estimator=EWMu(half_life=30),
        )
        scores = online_score(
            est,
            X,
            warmup_size=400,
            test_size=50,
            scoring=mahalanobis_calibration_ratio,
            per_step=True,
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) > 0
        assert all(s > 0 for s in scores)

    def test_calibration_reasonable(self, X):
        """Mean calibration ratio should be reasonably close to 1.0."""
        est = EWCovariance(half_life=30)
        scores = online_score(
            est,
            X,
            warmup_size=400,
            test_size=1,
            scoring=mahalanobis_calibration_ratio,
            per_step=True,
        )
        mean_score = np.mean(scores)
        assert 0.5 < mean_score < 2.0

    def test_freq(self, X):
        """online_score works with freq-based rebalancing."""
        est = EWCovariance(half_life=30)
        scores = online_score(
            est,
            X,
            warmup_size=6,
            test_size=1,
            freq="MS",
            scoring=mahalanobis_calibration_ratio,
            per_step=True,
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) > 0

    def test_purged_size(self, X):
        """online_score with purged_size produces different scores."""
        est = EWCovariance(half_life=30)
        scores_no_purge = online_score(
            est,
            X,
            warmup_size=400,
            test_size=50,
            purged_size=0,
            scoring=mahalanobis_calibration_ratio,
            per_step=True,
        )
        scores_purge = online_score(
            est,
            X,
            warmup_size=400,
            test_size=50,
            purged_size=5,
            scoring=mahalanobis_calibration_ratio,
            per_step=True,
        )

        assert len(scores_purge) > 0
        assert not np.array_equal(scores_no_purge, scores_purge)

    @pytest.mark.filterwarnings(
        "ignore:The covariance matrix is not positive definite:UserWarning"
    )
    def test_metadata_routing_active_mask(self, X):
        """online_score routes active_mask metadata to partial_fit."""
        X_masked, active_mask = _make_inactive_block(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            routed_est = EWCovariance(half_life=30).set_partial_fit_request(
                active_mask=True
            )
            scores_routed = online_score(
                routed_est,
                X_masked,
                warmup_size=400,
                test_size=50,
                scoring=mahalanobis_calibration_ratio,
                per_step=True,
                params={"active_mask": active_mask},
            )
            scores_plain = online_score(
                EWCovariance(half_life=30),
                X_masked,
                warmup_size=400,
                test_size=50,
                scoring=mahalanobis_calibration_ratio,
                per_step=True,
            )

        assert scores_routed.shape == scores_plain.shape
        assert not np.allclose(scores_routed, scores_plain, equal_nan=True)

    def test_metadata_routing_requires_request(self, X):
        """online_score raises when metadata is passed without a request."""
        X_masked, active_mask = _make_inactive_block(X, start=420, stop=480)

        with config_context(enable_metadata_routing=True):
            with pytest.raises(UnsetMetadataPassedError, match="online_score"):
                online_score(
                    EWCovariance(half_life=30),
                    X_masked,
                    warmup_size=400,
                    test_size=50,
                    scoring=mahalanobis_calibration_ratio,
                    per_step=True,
                    params={"active_mask": active_mask},
                )


class TestOnlineScoringValidation:
    """Validate that mismatched scoring types are rejected early."""

    def test_portfolio_estimator_rejects_make_scorer(self, X):
        """make_scorer with response_method='predict' raises for portfolio."""
        from skfolio.measures import RatioMeasure
        from skfolio.metrics import make_scorer

        model = _make_online_estimator()
        scorer = make_scorer(RatioMeasure.SHARPE_RATIO)

        with pytest.raises(TypeError, match="make_scorer is not supported"):
            online_score(model, X, warmup_size=400, test_size=50, scoring=scorer)

    def test_portfolio_estimator_rejects_make_scorer_custom(self, X):
        """make_scorer with a custom callable raises for portfolio."""
        from skfolio.metrics import make_scorer

        model = _make_online_estimator()
        scorer = make_scorer(lambda pred: pred.mean)

        with pytest.raises(TypeError, match="make_scorer is not supported"):
            online_score(model, X, warmup_size=400, test_size=50, scoring=scorer)

    def test_portfolio_estimator_rejects_make_scorer_dict(self, X):
        """Dict of make_scorer values raises for portfolio."""
        from skfolio.measures import RatioMeasure, RiskMeasure
        from skfolio.metrics import make_scorer

        model = _make_online_estimator()
        scoring = {
            "sharpe": make_scorer(RatioMeasure.SHARPE_RATIO),
            "var": make_scorer(RiskMeasure.VARIANCE),
        }

        with pytest.raises(TypeError, match="make_scorer is not supported"):
            online_score(model, X, warmup_size=400, test_size=50, scoring=scoring)

    def test_portfolio_estimator_rejects_plain_callable(self, X):
        """Plain callables are not accepted for portfolio online scoring."""
        model = _make_online_estimator()

        with pytest.raises(TypeError, match="must be `None`, a `BaseMeasure`"):
            online_score(
                model,
                X,
                warmup_size=400,
                test_size=50,
                scoring=lambda portfolio: portfolio.mean,
            )

    def test_component_estimator_rejects_base_measure(self, X):
        """BaseMeasure raises for component estimators."""
        from skfolio.measures import RatioMeasure

        est = EWCovariance(half_life=30)

        with pytest.raises(TypeError, match="BaseMeasure scoring is only"):
            online_score(
                est,
                X,
                warmup_size=400,
                test_size=50,
                scoring=RatioMeasure.SHARPE_RATIO,
            )

    def test_component_estimator_rejects_base_measure_dict(self, X):
        """Dict of BaseMeasure raises for component estimators."""
        from skfolio.measures import RatioMeasure

        est = EWCovariance(half_life=30)

        with pytest.raises(TypeError, match="BaseMeasure scoring is only"):
            online_score(
                est,
                X,
                warmup_size=400,
                test_size=50,
                scoring={"a": RatioMeasure.SHARPE_RATIO},
            )

    def test_component_estimator_rejects_portfolio_scorer(self, X):
        """Portfolio scorers must not be used for component estimators."""
        from skfolio.metrics import make_scorer

        est = EWCovariance(half_life=30)
        scorer = make_scorer(lambda pred: pred.mean)

        with pytest.raises(TypeError, match="response_method=None"):
            online_score(est, X, warmup_size=400, test_size=50, scoring=scorer)
