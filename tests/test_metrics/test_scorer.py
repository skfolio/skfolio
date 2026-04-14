"""Test Scorer module."""

import numpy as np
import pytest
import sklearn.model_selection as sks

from skfolio import RatioMeasure, RiskMeasure
from skfolio.metrics import (
    mahalanobis_calibration_loss,
    make_scorer,
    portfolio_variance_calibration_loss,
    portfolio_variance_qlike_loss,
)
from skfolio.metrics._scorer import _BaseScorer, _EstimatorScorer, _PortfolioScorer
from skfolio.moments import EWCovariance
from skfolio.optimization import MeanRisk, ObjectiveFunction


def test_default_score(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)
    grid_search = sks.GridSearchCV(
        estimator=model, cv=cv, n_jobs=-1, param_grid={"l2_coef": l2_coefs}
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.sharpe_ratio
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_ratio(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)

    # ratio measure
    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(RatioMeasure.CDAR_RATIO),
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cdar_ratio
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_risk_measure(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)
    # risk measure
    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(RiskMeasure.CVAR),
    )
    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cvar
        res[f"split{i}_test_score"] = -d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


def test_measure_score_custom(X):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        max_variance=0.3**2 / 252,
    )
    l2_coefs = [0.001, 0.01]
    cv = sks.KFold(3)

    # Custom
    def custom(prediction):
        return prediction.cvar - 2 * prediction.cdar

    grid_search = sks.GridSearchCV(
        estimator=model,
        cv=cv,
        n_jobs=-1,
        param_grid={"l2_coef": l2_coefs},
        scoring=make_scorer(custom),
    )

    grid_search.fit(X)

    res = {}
    for i, (train, test) in enumerate(cv.split(X)):
        d = np.zeros(2)
        for j, l2_coef in enumerate(l2_coefs):
            model.set_params(l2_coef=l2_coef)
            model.fit(X.take(train))
            pred = model.predict(X.take(test))
            d[j] = pred.cvar - 2 * pred.cdar
        res[f"split{i}_test_score"] = d

    for k, v in res.items():
        np.testing.assert_almost_equal(grid_search.cv_results_[k], v)

    np.testing.assert_almost_equal(
        grid_search.cv_results_["mean_test_score"],
        np.array(list(res.values())).mean(axis=0),
    )

    assert (
        grid_search.best_params_["l2_coef"]
        == l2_coefs[np.argmax(grid_search.cv_results_["mean_test_score"])]
    )


# ---------------------------------------------------------------------------
# _BaseScorer / _PortfolioScorer / _EstimatorScorer unit tests
# ---------------------------------------------------------------------------


class TestMakeScorerResponseMethod:
    """Validate the ``response_method`` parameter."""

    def test_default_response_method_is_predict(self):
        scorer = make_scorer(lambda pred: 1.0)
        assert isinstance(scorer, _PortfolioScorer)

    def test_response_method_predict(self):
        scorer = make_scorer(lambda pred: 1.0, response_method="predict")
        assert isinstance(scorer, _PortfolioScorer)

    def test_response_method_none(self):
        scorer = make_scorer(lambda est, X: 1.0, response_method=None)
        assert isinstance(scorer, _EstimatorScorer)

    def test_invalid_response_method_raises(self):
        with pytest.raises(ValueError, match="response_method"):
            make_scorer(lambda pred: 1.0, response_method="transform")

    def test_measure_with_response_method_none_raises(self):
        with pytest.raises(ValueError, match="response_method=None"):
            make_scorer(RatioMeasure.SHARPE_RATIO, response_method=None)

    def test_measure_with_response_method_predict(self):
        scorer = make_scorer(RatioMeasure.SHARPE_RATIO, response_method="predict")
        assert isinstance(scorer, _PortfolioScorer)

    def test_non_callable_non_measure_raises(self):
        with pytest.raises(TypeError, match="callable or a measure"):
            make_scorer("not_a_func")


class TestBaseScorerInheritance:
    def test_portfolio_scorer_is_base(self):
        scorer = make_scorer(lambda pred: 1.0)
        assert isinstance(scorer, _BaseScorer)

    def test_estimator_scorer_is_base(self):
        scorer = make_scorer(lambda est, X: 1.0, response_method=None)
        assert isinstance(scorer, _BaseScorer)


class TestEstimatorScorerCall:
    """Test that ``_EstimatorScorer`` calls score_func correctly."""

    def test_passes_estimator_and_x(self):
        received = {}

        def spy(estimator, X_test, **kw):
            received["estimator"] = estimator
            received["X_test"] = X_test
            received["kwargs"] = kw
            return 42.0

        scorer = make_scorer(spy, response_method=None)
        est = object()
        X = np.zeros((5, 3))
        result = scorer(est, X)

        assert received["estimator"] is est
        np.testing.assert_array_equal(received["X_test"], X)
        assert received["kwargs"] == {}
        assert result == 42.0

    def test_binds_kwargs(self):
        def fn(estimator, X_test, weights=None):
            return float(np.sum(weights)) if weights is not None else 0.0

        w = np.array([0.5, 0.5])
        scorer = make_scorer(fn, response_method=None, weights=w)
        result = scorer(object(), np.zeros((3, 2)))
        assert result == 1.0

    def test_sign_flip(self):
        def loss(estimator, X_test):
            return 10.0

        scorer = make_scorer(loss, greater_is_better=False, response_method=None)
        result = scorer(object(), np.zeros((3, 2)))
        assert result == -10.0

    def test_sign_positive_default(self):
        def score(estimator, X_test):
            return 5.0

        scorer = make_scorer(score, response_method=None)
        result = scorer(object(), np.zeros((3, 2)))
        assert result == 5.0

    def test_accepts_y_parameter(self):
        """Scorer accepts ``y`` for sklearn protocol compatibility."""

        def fn(estimator, X_test):
            return 1.0

        scorer = make_scorer(fn, response_method=None)
        result = scorer(object(), np.zeros((3, 2)), y=np.zeros(3))
        assert result == 1.0


class TestEstimatorScorerRepr:
    def test_repr_shows_response_method(self):
        def my_loss(estimator, X_test):
            return 0.0

        scorer = make_scorer(my_loss, greater_is_better=False, response_method=None)
        r = repr(scorer)
        assert "response_method=None" in r
        assert "my_loss" in r
        assert "greater_is_better=False" in r

    def test_repr_with_kwargs(self):
        def my_loss(estimator, X_test, weights=None):
            return 0.0

        scorer = make_scorer(
            my_loss, greater_is_better=False, response_method=None, weights="w"
        )
        r = repr(scorer)
        assert "weights=w" in r

    def test_portfolio_scorer_repr_no_response_method(self):
        def my_score(pred):
            return 0.0

        scorer = make_scorer(my_score)
        r = repr(scorer)
        assert "response_method" not in r


class TestPortfolioScorerReprMeasure:
    """Validate that measure-based portfolio scorers show the measure name."""

    def test_repr_shows_measure_name(self):
        scorer = make_scorer(RatioMeasure.SHARPE_RATIO)
        r = repr(scorer)
        assert "Sharpe Ratio" in r
        assert "score_func" not in r

    def test_repr_risk_measure_shows_greater_is_better_false(self):
        scorer = make_scorer(RiskMeasure.CVAR)
        r = repr(scorer)
        assert "CVaR" in r
        assert "greater_is_better=False" in r

    def test_repr_custom_callable_shows_function_name(self):
        def my_custom(pred):
            return 0.0

        scorer = make_scorer(my_custom)
        r = repr(scorer)
        assert "my_custom" in r


class TestPortfolioScorerAcceptsY:
    """Ensure the y=None fix doesn't break existing _PortfolioScorer."""

    def test_portfolio_scorer_ignores_y(self, X):
        model = MeanRisk(
            objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
            max_variance=0.3**2 / 252,
        )
        model.fit(X)
        scorer = make_scorer(RatioMeasure.SHARPE_RATIO)
        score_no_y = scorer(model, X)
        score_with_y = scorer(model, X, y=None)
        assert score_no_y == score_with_y


class TestEstimatorScorerWithCovarianceMetrics:
    """Integration tests with real covariance scoring functions."""

    @pytest.fixture()
    def cov_estimator(self):
        rng = np.random.default_rng(42)
        X_train = rng.standard_normal((300, 5)) * 0.01
        est = EWCovariance(half_life=30)
        est.fit(X_train)
        return est

    @pytest.fixture()
    def X_test(self):
        rng = np.random.default_rng(123)
        return rng.standard_normal((50, 5)) * 0.01

    def test_qlike_scorer(self, cov_estimator, X_test):
        scorer = make_scorer(
            portfolio_variance_qlike_loss,
            greater_is_better=False,
            response_method=None,
        )
        score = scorer(cov_estimator, X_test)
        raw = portfolio_variance_qlike_loss(cov_estimator, X_test)
        assert score == pytest.approx(-raw)

    def test_qlike_scorer_with_portfolio_weights(self, cov_estimator, X_test):
        w = np.ones(5) / 5
        scorer = make_scorer(
            portfolio_variance_qlike_loss,
            greater_is_better=False,
            response_method=None,
            portfolio_weights=w,
        )
        score = scorer(cov_estimator, X_test)
        raw = portfolio_variance_qlike_loss(cov_estimator, X_test, portfolio_weights=w)
        assert score == pytest.approx(-raw)

    def test_calibration_loss_scorer(self, cov_estimator, X_test):
        scorer = make_scorer(
            mahalanobis_calibration_loss,
            greater_is_better=False,
            response_method=None,
        )
        score = scorer(cov_estimator, X_test)
        raw = mahalanobis_calibration_loss(cov_estimator, X_test)
        assert score == pytest.approx(-raw)

    def test_portfolio_calibration_loss_with_portfolio_weights(
        self, cov_estimator, X_test
    ):
        w = np.array([0.4, 0.2, 0.2, 0.1, 0.1])
        scorer = make_scorer(
            portfolio_variance_calibration_loss,
            greater_is_better=False,
            response_method=None,
            portfolio_weights=w,
        )
        score = scorer(cov_estimator, X_test)
        raw = portfolio_variance_calibration_loss(
            cov_estimator, X_test, portfolio_weights=w
        )
        assert score == pytest.approx(-raw)

    def test_dict_scoring(self, cov_estimator, X_test):
        """Multiple _EstimatorScorer instances in a dict."""
        scorers = {
            "qlike": make_scorer(
                portfolio_variance_qlike_loss,
                greater_is_better=False,
                response_method=None,
            ),
            "calibration": make_scorer(
                mahalanobis_calibration_loss,
                greater_is_better=False,
                response_method=None,
            ),
        }
        scores = {name: fn(cov_estimator, X_test) for name, fn in scorers.items()}
        raw_qlike = portfolio_variance_qlike_loss(cov_estimator, X_test)
        raw_calib = mahalanobis_calibration_loss(cov_estimator, X_test)
        assert scores["qlike"] == pytest.approx(-raw_qlike)
        assert scores["calibration"] == pytest.approx(-raw_calib)


class TestScorerReprEdgeCases:
    def test_repr_with_functools_partial(self):
        from functools import partial

        fn = partial(portfolio_variance_qlike_loss, portfolio_weights=np.ones(5) / 5)
        scorer = make_scorer(fn, greater_is_better=False, response_method=None)
        r = repr(scorer)
        assert "make_scorer(" in r
        assert "greater_is_better=False" in r

    def test_repr_with_lambda(self):
        scorer = make_scorer(lambda est, X: 0.0, response_method=None)
        r = repr(scorer)
        assert "<lambda>" in r


class TestEstimatorScorerOnlineIntegration:
    """End-to-end test with :func:`online_score`."""

    def test_online_score_with_estimator_scorer(self, X):
        from skfolio.model_selection import online_score

        scorer = make_scorer(
            portfolio_variance_qlike_loss,
            greater_is_better=False,
            response_method=None,
        )
        score = online_score(
            EWCovariance(half_life=60),
            X,
            warmup_size=252,
            test_size=5,
            scoring=scorer,
        )
        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_online_score_dict_scoring(self, X):
        from skfolio.model_selection import online_score

        scoring = {
            "qlike": make_scorer(
                portfolio_variance_qlike_loss,
                greater_is_better=False,
                response_method=None,
            ),
            "calibration": make_scorer(
                mahalanobis_calibration_loss,
                greater_is_better=False,
                response_method=None,
            ),
        }
        scores = online_score(
            EWCovariance(half_life=60),
            X,
            warmup_size=252,
            test_size=5,
            scoring=scoring,
        )
        assert isinstance(scores, dict)
        assert set(scores.keys()) == {"qlike", "calibration"}
        for v in scores.values():
            assert isinstance(v, float)
            assert np.isfinite(v)
