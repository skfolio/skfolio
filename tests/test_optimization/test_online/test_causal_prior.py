"""Causal prior integration and metadata alignment."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context

from skfolio.optimization.online._winner import FollowTheWinner
from skfolio.prior import BasePrior, ReturnDistribution


class RecordingPrior(BasePrior):
    def __init__(self):
        pass

    def fit(self, X, y=None, **params):
        raise AssertionError("A causal prior must not be batch fitted")

    def partial_fit(self, X, y=None, signals=None):
        assert len(X) == 1
        self.rows_ = [*getattr(self, "rows_", []), np.asarray(X).copy()]
        self.signals_ = [*getattr(self, "signals_", []), signals]
        self.targets_ = [*getattr(self, "targets_", []), y]
        n = np.shape(X)[1]
        self.return_distribution_ = ReturnDistribution(
            mu=np.zeros(n),
            covariance=np.eye(n) * len(self.rows_) / 100,
            returns=np.asarray(X),
        )
        return self


def test_prior_is_cloned_updated_causally_and_reset():
    prior = RecordingPrior()
    X = np.array([[0.1, 0.02], [-0.03, 0.1], [0.05, 0.03]])
    estimator = FollowTheWinner(prior_estimator=prior, max_variance=1)
    estimator.fit(X)
    assert len(estimator.prior_estimator_.rows_) == 3
    assert not hasattr(prior, "rows_")
    assert_allclose(estimator._projector.config.covariance, np.eye(2) * 0.03)
    streamed = FollowTheWinner(prior_estimator=RecordingPrior(), max_variance=1)
    for row in X:
        streamed.partial_fit(row)
    assert_allclose(estimator.weights_, streamed.weights_)
    estimator.fit(X[:1])
    assert len(estimator.prior_estimator_.rows_) == 1


def test_routed_metadata_and_y_are_sliced_per_row():
    X = np.ones((3, 2)) * 0.01
    with config_context(enable_metadata_routing=True):
        prior = RecordingPrior().set_partial_fit_request(signals=True)
        estimator = FollowTheWinner(prior_estimator=prior, max_variance=1)
        estimator.fit(X, np.arange(3), signals=np.arange(3) * 10)
    assert_allclose(np.concatenate(estimator.prior_estimator_.signals_), [0, 10, 20])
    assert_allclose(np.concatenate(estimator.prior_estimator_.targets_), [0, 1, 2])


def test_no_prior_means_no_statistical_estimator():
    estimator = FollowTheWinner().fit([[0.1, 0.2]])
    assert not hasattr(estimator, "prior_estimator_")


def test_conflicting_covariance_sources_raise():
    with pytest.raises(ValueError, match="not both"):
        FollowTheWinner(prior_estimator=RecordingPrior(), covariance=np.eye(2)).fit(
            [[0.1, 0.2]]
        )


def test_empirical_prior_can_warm_up_before_first_execution():
    from skfolio.moments import EWCovariance, EWMu
    from skfolio.prior import EmpiricalPrior

    X = np.random.default_rng(42).normal(0, 0.01, (8, 3))
    prior = EmpiricalPrior(
        mu_estimator=EWMu(min_observations=3),
        covariance_estimator=EWCovariance(min_observations=3),
        max_history=10,
    )
    estimator = FollowTheWinner(prior_estimator=prior, max_variance=0.01)
    estimator.fit(X)
    covariance = estimator.prior_estimator_.return_distribution_.covariance
    assert np.isfinite(covariance).all()
    assert estimator.weights_ @ covariance @ estimator.weights_ <= 0.01 + 1e-7
    assert_allclose(estimator.initial_weights_, np.ones(3) / 3)
    with pytest.raises(ValueError, match="before an execution decision"):
        FollowTheWinner(prior_estimator=prior, max_variance=0.01).partial_fit(X[:1])


class WeightedPrior(RecordingPrior):
    def partial_fit(self, X, y=None, sample_weight=None):
        return super().partial_fit(X, y, signals=sample_weight)


def test_sample_weight_is_routed_only_to_requested_prior():
    with config_context(enable_metadata_routing=True):
        prior = WeightedPrior().set_partial_fit_request(sample_weight=True)
        estimator = FollowTheWinner(prior_estimator=prior)
        estimator.fit(np.zeros((3, 2)), sample_weight=np.array([1, 2, 3]))
    assert_allclose(np.concatenate(estimator.prior_estimator_.signals_), [1, 2, 3])


class FailingPrior(RecordingPrior):
    def partial_fit(self, X, y=None, signals=None):
        super().partial_fit(X, y, signals=signals)
        if np.any(np.asarray(X) > 0.5):
            raise ValueError("Injected prior failure")
        return self


def test_prior_failure_blocks_reuse_until_fit_resets_all_state():
    source = FailingPrior()
    estimator = FollowTheWinner(prior_estimator=source).fit([[0.1, 0.02]])
    with pytest.raises(ValueError, match="Injected prior failure"):
        estimator.partial_fit([[0.6, 0.01]])
    with pytest.raises(RuntimeError, match="call fit"):
        estimator.partial_fit([[0.01, 0.02]])
    with pytest.raises(RuntimeError, match="call fit"):
        estimator.predict([[0.01, 0.02]])
    estimator.fit([[0.01, 0.02]])
    assert len(estimator.prior_estimator_.rows_) == 1
    assert estimator.n_observations_ == 1
    assert not hasattr(source, "rows_")


class AlignedPrior(RecordingPrior):
    def partial_fit(self, X, y=None, signals=None):
        assert_allclose(signals, np.asarray(X)[:, 0])
        assert_allclose(y, np.asarray(X)[:, 1])
        return super().partial_fit(X, y, signals)


def test_online_evaluation_routes_prior_metadata_through_warmup():
    from skfolio.model_selection import online_predict

    X = np.random.default_rng(261).normal(0, 0.01, (8, 2))
    with config_context(enable_metadata_routing=True):
        model = FollowTheWinner(
            prior_estimator=AlignedPrior().set_partial_fit_request(signals=True),
            portfolio_params={"weight_drift": True},
        )
        result = online_predict(
            model,
            X,
            y=X[:, 1],
            warmup_size=3,
            test_size=1,
            params={"signals": X[:, 0]},
        )
    assert len(result.returns) == 5
