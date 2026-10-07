"""Integration tests for the three distance paths and online allocation lifecycle."""

import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.base import clone
from sklearn.utils import get_tags
from sklearn.utils.validation import validate_data

from skfolio.cluster import HierarchicalClustering
from skfolio.distance import BaseDistance, CovarianceDistance, PearsonDistance
from skfolio.exceptions import OptimizationError
from skfolio.measures import RiskMeasure
from skfolio.moments import EWCovariance, EWMu, EmpiricalCovariance
from skfolio.optimization import HierarchicalRiskParity, SchurComplementary
from skfolio.prior import EmpiricalPrior, ReturnDistribution
from skfolio.seriation import HierarchicalSeriation, SpectralSeriation


@pytest.fixture(params=[HierarchicalRiskParity, SchurComplementary])
def optimizer(request):
    return request.param


@pytest.fixture
def returns():
    return pd.DataFrame(
        np.random.default_rng(9).normal(0, 0.01, (60, 4)), columns=list("abcd")
    )


def prior():
    return EmpiricalPrior(
        mu_estimator=EWMu(half_life=5, min_observations=3),
        covariance_estimator=EWCovariance(half_life=5, min_observations=3),
    )


class RecordingDistance(PearsonDistance):
    def fit(self, X, y=None, sample_weight=None):
        self.received_ = X.copy()
        self.target_ = y
        self.sample_weight_ = sample_weight
        return super().fit(X)

    def partial_fit(self, X, y=None):
        raise AssertionError("Prior snapshots must use fit")


class RecordingCovarianceDistance(BaseDistance):
    def __init__(self):
        pass

    @property
    def requires_covariance_input(self):
        return True

    def fit(self, X, y=None):
        validate_data(self, X)
        self.received_ = X.copy()
        distance = CovarianceDistance("precomputed").fit(X)
        self.codependence_ = distance.codependence_
        self.distance_ = distance.distance_
        return self


class ArrayCovarianceDistance(CovarianceDistance):
    def _fit(self, X, y, method, **fit_params):
        return super()._fit(np.asarray(X), y, method, **fit_params)


class RebuiltPrior(EmpiricalPrior):
    def _fit(self, X, y, method, **fit_params):
        super()._fit(X, y, method, **fit_params)
        distribution = self.return_distribution_
        # A prior can rebuild past scenarios and expose independent scenario weights.
        scenarios = distribution.returns[::-1].copy()
        scenarios[:, 0] *= 2
        self.return_distribution_ = ReturnDistribution(
            mu=distribution.mu,
            covariance=distribution.covariance,
            returns=scenarios,
            sample_weight=np.arange(1, len(scenarios) + 1, dtype=float)
            / (len(scenarios) * (len(scenarios) + 1) / 2),
        )
        return self


class WeightedCovariance(EmpiricalCovariance):
    def fit(self, X, y=None, sample_weight=None):
        X = validate_data(self, X)
        self.sample_weight_ = sample_weight
        self.covariance_ = np.cov(X.T, aweights=sample_weight)
        return self


class OnlineWeightedCovariance(WeightedCovariance):
    def partial_fit(self, X, y=None, sample_weight=None):
        first_call = not hasattr(self, "covariance_")
        X = validate_data(self, X, reset=first_call)
        if sample_weight is None:
            sample_weight = np.ones(len(X))
        self.returns_ = X if first_call else np.concatenate([self.returns_, X])
        self.sample_weight_ = (
            sample_weight
            if first_call
            else np.concatenate([self.sample_weight_, sample_weight])
        )
        self.covariance_ = np.cov(self.returns_.T, aweights=self.sample_weight_)
        return self


class ExcludedCovariancePrior(EmpiricalPrior):
    def partial_fit(self, X, y=None, **fit_params):
        super().partial_fit(X, y, **fit_params)
        distribution = self.return_distribution_
        mu = distribution.mu.copy()
        covariance = distribution.covariance.copy()
        mu[2] = np.nan
        covariance[2, 0] = np.inf
        self.return_distribution_ = ReturnDistribution(
            mu=mu, covariance=covariance, returns=distribution.returns
        )
        return self


@pytest.mark.parametrize("fit_first", [False, True])
@pytest.mark.parametrize("path", ["scenarios", "covariance", "raw"])
def test_online_children_and_batch_equivalence(optimizer, returns, fit_first, path):
    distance = (
        CovarianceDistance(EWCovariance(half_life=6, min_observations=3))
        if path == "raw"
        else CovarianceDistance("precomputed")
        if path == "covariance"
        else PearsonDistance()
    )
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=distance,
        distance_from_prior=path != "raw",
    )
    getattr(model, "fit" if fit_first else "partial_fit")(returns[:20])
    fitted_prior = model.prior_estimator_
    fitted_distance = model.distance_estimator_
    for start in [20, 40]:
        model.partial_fit(returns[start : start + 20])
        assert model.prior_estimator_ is fitted_prior
        if path == "raw":
            assert model.distance_estimator_ is fitted_distance
        reference = clone(model).fit(returns[: start + 20])
        np.testing.assert_allclose(model.weights_, reference.weights_, atol=1e-10)
    assert len(fitted_prior.return_distribution_.returns) == 60
    model.set_params(prior_estimator__covariance_estimator__half_life=8).fit(
        returns[-20:]
    )
    assert model.prior_estimator_ is not fitted_prior
    assert model.prior_estimator_.covariance_estimator_.half_life == 8
    assert len(model.prior_estimator_.return_distribution_.returns) == 20
    np.testing.assert_allclose(model.weights_, clone(model).fit(returns[-20:]).weights_)


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
@pytest.mark.parametrize("path", ["scenarios", "covariance", "raw"])
def test_distance_without_feature_names_preserves_asset_order(
    optimizer, returns, method, path
):
    returns = returns.copy()
    returns.iloc[:20, 2] = np.nan
    covariance = (
        "precomputed"
        if path == "covariance"
        else EWCovariance(half_life=5, min_observations=3)
    )
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=ArrayCovarianceDistance(covariance),
        distance_from_prior=path != "raw",
    )
    reference = clone(model).set_params(
        distance_estimator=CovarianceDistance(covariance)
    )
    for start in [0, 20]:
        batch = returns.iloc[start : start + 20]
        update = method if start == 0 else "partial_fit"
        getattr(model, update)(batch)
        getattr(reference, update)(batch)
        np.testing.assert_allclose(model.weights_, reference.weights_)
        np.testing.assert_array_equal(model.feature_names_in_, returns.columns)
        assert not hasattr(model.distance_estimator_, "feature_names_in_")


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_custom_covariance_distance_receives_prior_covariance(
    optimizer, returns, method
):
    model = optimizer(
        prior_estimator=prior(), distance_estimator=RecordingCovarianceDistance()
    )
    getattr(model, method)(returns[:20])
    model.partial_fit(returns[20:40])
    distance = model.distance_estimator_
    assert get_tags(distance).input_tags.pairwise
    np.testing.assert_allclose(
        distance.received_, model.prior_estimator_.return_distribution_.covariance
    )
    reference = (
        clone(model)
        .set_params(distance_estimator=CovarianceDistance("precomputed"))
        .fit(returns[:40])
    )
    np.testing.assert_allclose(model.weights_, reference.weights_)


def test_raw_distance_reports_unsupported_covariance(optimizer, returns):
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=CovarianceDistance(),
        distance_from_prior=False,
    )
    with pytest.raises(
        TypeError, match="GerberCovariance does not implement partial_fit"
    ):
        model.partial_fit(returns[:20])


def test_prior_scenarios_are_full_snapshots_with_aligned_weights(optimizer, returns):
    p = RebuiltPrior(
        mu_estimator=EWMu(half_life=5, min_observations=3),
        covariance_estimator=EWCovariance(half_life=5, min_observations=3),
        max_history=12,
    )
    model = optimizer(
        prior_estimator=p,
        distance_estimator=RecordingDistance(),
        seriation_estimator=SpectralSeriation(),
    )
    model.partial_fit(returns[:20], y=np.ones(20))
    seriator = model.seriation_estimator_
    model.partial_fit(returns[20:30], y=np.ones(10))
    assert model.seriation_estimator_ is seriator
    learned = model.prior_estimator_.return_distribution_
    np.testing.assert_array_equal(model.distance_estimator_.received_, learned.returns)
    np.testing.assert_array_equal(
        model.distance_estimator_.sample_weight_, learned.sample_weight
    )
    assert model.distance_estimator_.target_ is None
    assert isinstance(model.distance_estimator_.received_.index, pd.RangeIndex)


@pytest.mark.parametrize("path", ["covariance", "raw", "scenarios"])
def test_singleton_and_returning_assets(optimizer, returns, path):
    distance = (
        CovarianceDistance(EWCovariance(half_life=5, min_observations=3))
        if path == "raw"
        else CovarianceDistance("precomputed")
        if path == "covariance"
        else PearsonDistance()
    )
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=distance,
        distance_from_prior=path != "raw",
        seriation_estimator=SpectralSeriation(),
        raise_on_failure=False,
    )
    batch = returns[:5].copy()
    batch.iloc[:, 1:] = np.nan
    model.partial_fit(batch)
    child = model.prior_estimator_
    raw = model.distance_estimator_
    np.testing.assert_array_equal(model.weights_, [1, 0, 0, 0])
    if optimizer is SchurComplementary:
        assert model.effective_gamma_ == 0
    if path == "raw":
        assert model.distance_estimator_ is raw
    else:
        assert model.distance_estimator_ is None
    model.partial_fit(returns[5:20])
    assert model.prior_estimator_ is child
    assert len(child.return_distribution_.returns) == 20
    assert len(model.seriation_estimator_.ordering_) == 4
    if path == "raw":
        assert model.distance_estimator_ is raw


@pytest.mark.parametrize("fallback", [None, "previous_weights"])
def test_empty_universe_raises_online(optimizer, returns, fallback):
    model = optimizer(
        prior_estimator=prior(),
        fallback=fallback,
        previous_weights=0.25,
        raise_on_failure=False,
    )
    with pytest.raises(ValueError, match="All assets are non-investable"):
        model.partial_fit(returns[:1])


@pytest.mark.parametrize("fallback", [None, "previous_weights"])
def test_handled_allocation_failure_preserves_learning(optimizer, returns, fallback):
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=CovarianceDistance("precomputed"),
        seriation_estimator=SpectralSeriation(),
        fallback=fallback,
        previous_weights={"a": 0.2, "b": 0.3, "c": 0.1, "d": 0.4},
        raise_on_failure=False,
    )
    model.partial_fit(returns[:20])
    learned_prior = model.prior_estimator_
    with config_context(enable_metadata_routing=True):
        model.max_weights = 0.4
        active = np.zeros((5, 4), dtype=bool)
        active[:, 0] = True
        if fallback is None:
            with pytest.warns(UserWarning, match="batch was consumed"):
                model.partial_fit(returns[20:25], active_mask=active)
            assert model.weights_ is None
        else:
            model.partial_fit(returns[20:25], active_mask=active)
            np.testing.assert_allclose(model.weights_, [0.2, 0, 0, 0])
            assert model.fallback_ == "previous_weights"
            assert model.error_ is None
            assert len(model.fallback_chain_) == 2
            assert model.fallback_chain_[-1] == ("previous_weights", "success")
        assert model.prior_estimator_ is learned_prior
        if optimizer is SchurComplementary:
            assert model.effective_gamma_ is None
        model.max_weights = 1
        model.partial_fit(returns[25:35], active_mask=np.ones((10, 4), dtype=bool))
        assert len(model.prior_estimator_.return_distribution_.returns) == 35
        assert model.error_ is None
        assert model.fallback_ is None
        assert model.fallback_chain_ is None


@pytest.mark.parametrize("fallback", [None, "previous_weights"])
@pytest.mark.parametrize("raise_on_failure", [False, True])
def test_online_allocation_failure_without_usable_fallback(
    optimizer, returns, fallback, raise_on_failure
):
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=CovarianceDistance("precomputed"),
        fallback=fallback,
        raise_on_failure=raise_on_failure,
    ).partial_fit(returns[:20])
    learned_prior = model.prior_estimator_
    model.max_weights = 0.4
    active = np.zeros((5, 4), dtype=bool)
    active[:, 0] = True
    error = OptimizationError if fallback is None else RuntimeError
    message = "infeasible" if fallback is None else "previous_weights.*None"
    expectation = (
        pytest.raises(error, match=message)
        if raise_on_failure
        else pytest.warns(UserWarning, match="batch was consumed")
    )
    with config_context(enable_metadata_routing=True), expectation:
        model.partial_fit(returns[20:25], active_mask=active)
    assert model.prior_estimator_ is learned_prior
    assert len(learned_prior.return_distribution_.returns) == 25
    assert model.fallback_ is None
    if fallback is None:
        assert model.fallback_chain_ is None
        assert "infeasible" in model.error_
    else:
        assert len(model.fallback_chain_) == 2
        assert "infeasible" in model.fallback_chain_[0][1]
        assert model.fallback_chain_[1] == ("previous_weights", model.error_)
        assert "previous_weights" in model.error_
    if not raise_on_failure:
        assert model.weights_ is None


@pytest.mark.parametrize("change", ["names", "bounds", "fallback"])
def test_invalid_inputs_are_rejected_before_learning(optimizer, returns, change):
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=CovarianceDistance(
            EWCovariance(half_life=5, min_observations=3)
        ),
        distance_from_prior=False,
    )
    model.partial_fit(returns[:20])
    weights = model.weights_.copy()
    distribution = model.prior_estimator_.return_distribution_
    diagnostics = (model.error_, model.fallback_, model.fallback_chain_)
    batch = returns[20:30]
    if change == "names":
        batch = batch.iloc[:, ::-1]
    elif change == "bounds":
        model.max_weights = 0.1
    else:
        model.fallback = optimizer()
    with pytest.raises((ValueError, TypeError)):
        model.partial_fit(batch)
    assert model.prior_estimator_.return_distribution_ is distribution
    np.testing.assert_array_equal(model.weights_, weights)
    assert diagnostics == (model.error_, model.fallback_, model.fallback_chain_)


@pytest.mark.parametrize("bound", ["min_weights", "max_weights"])
@pytest.mark.parametrize("excess", [0, 5e-9, 1e-8, 2e-8])
def test_weight_bound_sum_tolerance(optimizer, returns, bound, excess):
    model = optimizer(prior_estimator=prior()).partial_fit(returns[:20])
    distribution = model.prior_estimator_.return_distribution_
    weights = model.weights_.copy()
    diagnostics = (model.error_, model.fallback_, model.fallback_chain_)
    total = 1 + excess if bound == "min_weights" else 1 - excess
    model.set_params(**{bound: total / 4})

    if excess > 1e-8:
        with pytest.raises(ValueError, match=f"Invalid `{bound}`"):
            model.partial_fit(returns[20:30])
        assert model.prior_estimator_.return_distribution_ is distribution
        np.testing.assert_array_equal(model.weights_, weights)
        assert (model.error_, model.fallback_, model.fallback_chain_) == diagnostics
    else:
        model.partial_fit(returns[20:30])
        assert len(model.prior_estimator_.return_distribution_.returns) == 30
        np.testing.assert_allclose(model.weights_, 0.25, rtol=0, atol=1e-8)


@pytest.mark.parametrize("excess", [0, 5e-9, 1e-8, 2e-8])
def test_weight_bound_sum_tolerance_after_delisting(optimizer, returns, excess):
    model = optimizer(prior_estimator=prior()).partial_fit(returns[:20])
    model.max_weights = (1 - excess) / 2
    active = np.zeros((10, 4), dtype=bool)
    active[:, :2] = True

    with config_context(enable_metadata_routing=True):
        if excess > 1e-8:
            with pytest.raises(OptimizationError, match="infeasible"):
                model.partial_fit(returns[20:30], active_mask=active)
        else:
            model.partial_fit(returns[20:30], active_mask=active)
            np.testing.assert_allclose(
                model.weights_, [0.5, 0.5, 0, 0], rtol=0, atol=1e-8
            )
    assert len(model.prior_estimator_.return_distribution_.returns) == 30


def test_raw_batch_distance_rejected_when_called(optimizer, returns):
    model = optimizer(prior_estimator=prior(), distance_from_prior=False).fit(
        returns[:20]
    )
    with pytest.raises(
        TypeError, match="PearsonDistance does not implement partial_fit"
    ):
        model.partial_fit(returns[20:30])
    np.testing.assert_array_equal(
        model.prior_estimator_.return_distribution_.returns, returns[:30]
    )
    model.set_params(
        distance_estimator=CovarianceDistance(
            EWCovariance(half_life=5, min_observations=3)
        )
    ).fit(returns[:20])
    model.partial_fit(returns[20:30])
    np.testing.assert_array_equal(
        model.prior_estimator_.return_distribution_.returns, returns[:30]
    )


def test_raw_precomputed_distance_rejected_before_learning(optimizer, returns):
    with pytest.raises(ValueError, match="requires distance_from_prior=True"):
        optimizer(
            prior_estimator=prior(),
            distance_from_prior=False,
            distance_estimator=CovarianceDistance("precomputed"),
        ).partial_fit(returns)


@pytest.mark.parametrize(
    "failure,error_type",
    [
        ("prior", RuntimeError),
        ("prior", ValueError),
        ("prior", OptimizationError),
        ("distance", RuntimeError),
        ("distance", ValueError),
        ("distance", OptimizationError),
        ("seriation", RuntimeError),
        ("seriation", ValueError),
        ("seriation", OptimizationError),
        ("allocation", RuntimeError),
        ("allocation", ValueError),
    ],
)
def test_unexpected_errors_propagate(
    optimizer, returns, monkeypatch, failure, error_type
):
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=CovarianceDistance(
            EWCovariance(half_life=5, min_observations=3)
        ),
        distance_from_prior=False,
        seriation_estimator=SpectralSeriation(),
        raise_on_failure=False,
        fallback="previous_weights",
        previous_weights=0.25,
    )
    model.partial_fit(returns[:20])
    targets = {
        "prior": (model.prior_estimator_, "partial_fit"),
        "distance": (model.distance_estimator_, "partial_fit"),
        "seriation": (model.seriation_estimator_, "partial_fit"),
        "allocation": (model, "_compute_weights"),
    }
    target, method = targets[failure]

    def fail(*args, **kwargs):
        raise error_type("unexpected update error")

    monkeypatch.setattr(target, method, fail)
    with pytest.raises(error_type, match="unexpected update error"):
        model.partial_fit(returns[20:30])


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_nonfinite_allocation_uses_fallback(optimizer, returns, monkeypatch, method):
    model = optimizer(
        prior_estimator=prior(), fallback="previous_weights", previous_weights=0.2
    )
    compute_weights = model._compute_weights

    def nonfinite_weights(*args, **kwargs):
        weights = compute_weights(*args, **kwargs)
        weights[0] = np.nan
        return weights

    monkeypatch.setattr(model, "_compute_weights", nonfinite_weights)
    getattr(model, method)(returns)
    np.testing.assert_array_equal(model.weights_, np.full(4, 0.2))
    assert "non-finite" in model.fallback_chain_[0][1]
    if optimizer is SchurComplementary:
        assert model.effective_gamma_ is None


def test_strict_raw_distance_readiness(optimizer, returns):
    model = optimizer(
        prior_estimator=prior(),
        distance_from_prior=False,
        distance_estimator=CovarianceDistance(
            EWCovariance(half_life=5, min_observations=30)
        ),
        raise_on_failure=False,
    )
    with pytest.raises(ValueError, match="unavailable for prior-investable"):
        model.partial_fit(returns[:20])


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_legacy_alias_and_parameter_conflict(optimizer, returns, method):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        legacy = optimizer(
            prior_estimator=prior(),
            hierarchical_clustering_estimator=HierarchicalClustering(),
        )
        assert not hasattr(legacy, "hierarchical_clustering_estimator_")
        cloned = clone(legacy)
        cloned.get_params()
        cloned.get_metadata_routing()
    with pytest.warns(
        FutureWarning, match="hierarchical_clustering_estimator.*2.0"
    ) as caught:
        getattr(cloned, method)(returns[:20])
    assert len(caught) == 1
    assert caught[0].filename == __file__
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        cloned.partial_fit(returns[20:40])
    with pytest.warns(
        FutureWarning, match="hierarchical_clustering_estimator_.*2.0"
    ) as caught:
        assert (
            cloned.hierarchical_clustering_estimator_
            is cloned.seriation_estimator_.hierarchical_clustering_estimator_
        )
    assert caught[0].filename == __file__
    with warnings.catch_warnings(), pytest.raises(ValueError, match="either"):
        warnings.simplefilter("error", FutureWarning)
        optimizer(
            hierarchical_clustering_estimator=HierarchicalClustering(),
            seriation_estimator=SpectralSeriation(),
        ).fit(returns)
    with pytest.warns(FutureWarning, match="hierarchical_clustering_estimator.*2.0"):
        cloned.fit(returns)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        cloned.set_params(
            hierarchical_clustering_estimator=None,
            seriation_estimator=SpectralSeriation(),
        ).fit(returns)
        assert not hasattr(cloned, "hierarchical_clustering_estimator_")


def test_legacy_clustering_alias_for_singleton(optimizer, returns):
    model = optimizer(prior_estimator=prior(), raise_on_failure=False)
    returns = returns.copy()
    returns.iloc[:, 1:] = np.nan
    model.partial_fit(returns)
    with pytest.warns(FutureWarning, match="hierarchical_clustering_estimator_.*2.0"):
        assert model.hierarchical_clustering_estimator_ is None


@pytest.mark.parametrize(
    "seriation", [None, HierarchicalSeriation(), SpectralSeriation()]
)
def test_seriation_api_has_no_deprecation_warnings(optimizer, returns, seriation):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = optimizer(prior_estimator=prior(), seriation_estimator=seriation)
        model.fit(returns[:20])
        model.partial_fit(returns[20:40])
        assert len(model.seriation_estimator_.ordering_) == returns.shape[1]


def test_online_scenario_metadata_is_rejected_before_learning(optimizer, returns):
    with config_context(enable_metadata_routing=True):
        model = optimizer(
            prior_estimator=prior(),
            distance_estimator=RecordingDistance().set_fit_request(sample_weight=True),
        )
        model.partial_fit(returns[:20])
        distribution = model.prior_estimator_.return_distribution_
        with pytest.raises(ValueError, match="Observation metadata"):
            model.partial_fit(returns[20:30], sample_weight=np.ones(10))
        assert model.prior_estimator_.return_distribution_ is distribution


@pytest.mark.parametrize("input_format", ["dict", "array", "scalar", "none"])
@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_costs_and_constraints_match_compact_allocation(returns, input_format, method):
    full = returns.copy()
    full.iloc[:, 2] = np.nan
    inputs = dict(
        transaction_costs={"a": 0.0001, "b": 0.0002, "c": 10.0, "d": 0.0003},
        management_fees={"a": 0.0001, "b": 0.0001, "c": 10.0, "d": 0.0002},
        previous_weights={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
        min_weights={"a": 0.05, "d": 0.1},
        max_weights={"a": 0.4, "b": 0.5, "d": 0.5},
    )
    if input_format == "array":
        inputs = {
            name: np.array(
                [value.get(asset, 1 if name == "max_weights" else 0) for asset in full]
            )
            for name, value in inputs.items()
        }
    elif input_format == "scalar":
        inputs = dict(
            transaction_costs=0.0001,
            management_fees=0.0001,
            previous_weights=0.25,
            min_weights=0.05,
            max_weights=0.5,
        )
    elif input_format == "none":
        inputs = dict.fromkeys(inputs)

    model = HierarchicalRiskParity(
        prior_estimator=prior(),
        risk_measure=RiskMeasure.CVAR,
        distance_estimator=CovarianceDistance("precomputed"),
        **inputs,
    )
    compact_inputs = {
        name: value[[0, 1, 3]] if isinstance(value, np.ndarray) else value
        for name, value in inputs.items()
    }
    reference = clone(model).set_params(**compact_inputs)
    for start in [0, 20, 40]:
        update = method if start == 0 else "partial_fit"
        batch = full.iloc[start : start + 20]
        getattr(model, update)(batch)
        getattr(reference, update)(batch.drop(columns="c"))
        np.testing.assert_allclose(model.weights_[[0, 1, 3]], reference.weights_)
        assert model.weights_[2] == 0


@pytest.mark.parametrize("name", ["min_weights", "max_weights"])
def test_invalid_bounds_for_excluded_assets_raise_before_learning(
    optimizer, returns, name
):
    batch = returns[:20].copy()
    batch.iloc[:, 2] = np.nan
    model = optimizer(
        prior_estimator=prior(), distance_estimator=CovarianceDistance("precomputed")
    ).partial_fit(batch)
    distribution = model.prior_estimator_.return_distribution_
    weights = model.weights_.copy()
    mask = model.investable_mask_.copy()
    invalid = np.full(4, 0.5 if name == "max_weights" else 0.0)
    invalid[2] = np.nan
    model.set_params(**{name: invalid})

    with pytest.raises(ValueError, match=name):
        model.partial_fit(returns[20:40])

    assert model.prior_estimator_.return_distribution_ is distribution
    np.testing.assert_array_equal(model.weights_, weights)
    np.testing.assert_array_equal(model.investable_mask_, mask)


@pytest.mark.parametrize(
    "name", ["transaction_costs", "management_fees", "previous_weights"]
)
@pytest.mark.parametrize("value", [np.nan, np.inf, [0.0, 0.0]])
@pytest.mark.parametrize("risk_measure", [RiskMeasure.VARIANCE, RiskMeasure.CVAR])
def test_hrp_invalid_risk_inputs_do_not_trigger_online_fallback(
    returns, name, value, risk_measure
):
    model = HierarchicalRiskParity(
        prior_estimator=prior(),
        risk_measure=risk_measure,
        fallback="previous_weights",
        previous_weights=0.25,
        raise_on_failure=False,
    ).partial_fit(returns[:20])
    model.set_params(**{name: value})

    with pytest.raises(ValueError, match=name):
        model.partial_fit(returns[20:40])


@pytest.mark.parametrize(
    "name", ["transaction_costs", "management_fees", "previous_weights"]
)
def test_schur_portfolio_inputs_are_validated_when_used(returns, name):
    reference = SchurComplementary(prior_estimator=prior()).partial_fit(returns[:20])
    model = clone(reference).set_params(**{name: [0.0, 0.0]})
    model.partial_fit(returns[:20])
    np.testing.assert_allclose(model.weights_, reference.weights_)

    with pytest.raises(ValueError, match=name):
        model.predict(returns[20:40])


@pytest.mark.parametrize("method", ["fit", "partial_fit"])
def test_full_universe_inputs_after_asset_listing(optimizer, returns, method):
    initial = returns[:20].copy()
    initial.iloc[:, 2] = np.nan
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=CovarianceDistance("precomputed"),
        min_weights=np.array([0.05, 0.05, 0.3, 0.05]),
        max_weights=np.array([0.5, 0.5, 0.4, 0.5]),
        transaction_costs=np.array([0.0001, 0.0002, 0.0003, 0.0004]),
        management_fees=np.array([0.0001, 0.0002, 0.0003, 0.0004]),
        previous_weights=np.array([0.2, 0.3, 0.1, 0.4]),
    ).partial_fit(initial)
    assert model.weights_[2] == 0

    batch = returns[20:40]
    getattr(model, method)(batch)
    history = pd.concat([initial, batch]) if method == "partial_fit" else batch
    reference = clone(model).fit(history)
    np.testing.assert_allclose(model.weights_, reference.weights_, atol=1e-10)
    assert model.weights_[2] >= 0.3 - 1e-8


def test_precomputed_masks_prior_exclusions_without_mutation(optimizer, returns):
    model = optimizer(
        prior_estimator=ExcludedCovariancePrior(**prior().get_params(deep=False)),
        distance_estimator=CovarianceDistance("precomputed"),
        seriation_estimator=SpectralSeriation(),
    )
    model.partial_fit(returns[:20])
    assert model.weights_[2] == 0
    assert np.isinf(model.prior_estimator_.return_distribution_.covariance[2, 0])
    assert np.isnan(model.distance_estimator_.distance_[2]).all()
    np.testing.assert_array_equal(
        model.distance_estimator_.feature_names_in_, returns.columns
    )


def test_nested_scenario_weight_routing_and_conflict(optimizer, returns):
    with config_context(enable_metadata_routing=True):
        covariance = WeightedCovariance().set_fit_request(sample_weight=True)
        model = optimizer(
            prior_estimator=RebuiltPrior(**prior().get_params(deep=False)),
            distance_estimator=CovarianceDistance(covariance),
        )
        model.partial_fit(returns[:20])
        learned = model.prior_estimator_.return_distribution_
        np.testing.assert_array_equal(
            model.distance_estimator_.covariance_estimator_.sample_weight_,
            learned.sample_weight,
        )
        expected = np.cov(learned.returns.T, aweights=learned.sample_weight)
        np.testing.assert_allclose(
            model.distance_estimator_.covariance_estimator_.covariance_, expected
        )
        # Explicit distance weights must not override internally aligned scenarios.
        with pytest.raises(ValueError, match="Conflicting parameters"):
            model.fit(returns[:20], sample_weight=np.ones(20) / 20)


@pytest.mark.parametrize("weight_name", ["sample_weight", "observation_weights"])
def test_raw_sample_weights_follow_partial_fit_request(optimizer, returns, weight_name):
    with config_context(enable_metadata_routing=True):
        covariance = (
            OnlineWeightedCovariance()
            .set_fit_request(sample_weight=False)
            .set_partial_fit_request(
                sample_weight=True if weight_name == "sample_weight" else weight_name
            )
        )
        model = optimizer(
            prior_estimator=prior(),
            distance_estimator=CovarianceDistance(covariance),
            distance_from_prior=False,
        )
        sample_weight = np.arange(1, 41, dtype=float)
        for start in [0, 20]:
            model.partial_fit(
                returns[start : start + 20],
                **{weight_name: sample_weight[start : start + 20]},
            )
        learned = model.distance_estimator_.covariance_estimator_
        np.testing.assert_array_equal(learned.sample_weight_, sample_weight)
        np.testing.assert_allclose(
            learned.covariance_, np.cov(returns[:40].T, aweights=sample_weight)
        )


@pytest.mark.parametrize("weight_request", [False, None])
def test_scenario_weights_are_not_forwarded_when_unrequested(
    optimizer, returns, weight_request
):
    with config_context(enable_metadata_routing=True):
        model = optimizer(
            prior_estimator=RebuiltPrior(**prior().get_params(deep=False)),
            distance_estimator=CovarianceDistance(
                WeightedCovariance().set_fit_request(sample_weight=weight_request)
            ),
        ).partial_fit(returns[:20])
        learned = model.distance_estimator_.covariance_estimator_
        assert learned.sample_weight_ is None
        np.testing.assert_allclose(
            learned.covariance_,
            np.cov(model.prior_estimator_.return_distribution_.returns.T),
        )


def test_raw_nested_activity_metadata_updates_each_learner(optimizer, returns):
    with config_context(enable_metadata_routing=True):
        p = prior()
        covariance = EWCovariance(half_life=6, min_observations=3)
        model = optimizer(
            prior_estimator=p,
            distance_estimator=CovarianceDistance(covariance),
            distance_from_prior=False,
        )
        active = np.ones((20, 4), dtype=bool)
        active[:, 1] = False
        model.partial_fit(returns[:20], active_mask=active)
        assert model.weights_[1] == 0
        assert np.isnan(
            model.distance_estimator_.covariance_estimator_.covariance_[1]
        ).all()
        reference = clone(covariance).partial_fit(returns[:20], active_mask=active)
        np.testing.assert_allclose(
            model.distance_estimator_.covariance_estimator_.covariance_,
            reference.covariance_,
            equal_nan=True,
        )


@pytest.mark.parametrize("seriation", [HierarchicalSeriation, SpectralSeriation])
@pytest.mark.parametrize("path", ["covariance", "raw", "scenarios"])
@pytest.mark.filterwarnings("ignore:More than 5%.*zero-filled:UserWarning")
def test_listing_holiday_delisting_and_reentry(optimizer, returns, seriation, path):
    active = np.ones(returns.shape, dtype=bool)
    active[:20, 2] = False
    active[25:30, 3] = False
    returns = returns.copy()
    returns[~active] = np.nan
    returns.iloc[23:25, 1] = np.nan  # Holiday, still active.
    distance = (
        CovarianceDistance(EWCovariance(half_life=5, min_observations=3))
        if path == "raw"
        else CovarianceDistance("precomputed")
        if path == "covariance"
        else PearsonDistance()
    )
    model = optimizer(
        prior_estimator=prior(),
        distance_estimator=distance,
        distance_from_prior=path != "raw",
        seriation_estimator=seriation(),
    )
    steps = [
        (0, 20, [True, True, False, True]),
        (20, 22, [True, True, False, True]),
        (22, 23, [True, True, True, True]),
        (23, 25, [True, True, True, True]),
        (25, 30, [True, True, True, False]),
        (30, 32, [True, True, True, False]),
        (32, 33, [True, True, True, True]),
    ]
    with config_context(enable_metadata_routing=True):
        for start, end, expected in steps:
            if start == 23:
                holiday_mu = model.prior_estimator_.mu_estimator_.mu_[1]
            model.partial_fit(returns.iloc[start:end], active_mask=active[start:end])
            expected = np.array(expected)
            np.testing.assert_array_equal(
                model.seriation_estimator_.investable_mask_, expected
            )
            np.testing.assert_array_equal(model.weights_[~expected], 0)
            assert np.all(model.weights_[expected] > 0)
            assert model.weights_.sum() == pytest.approx(1)
            if start == 23:
                assert model.prior_estimator_.mu_estimator_.mu_[1] == holiday_mu


def test_scenario_covariance_activity_request_requires_opt_out(optimizer, returns):
    with config_context(enable_metadata_routing=True):
        covariance = EWCovariance(half_life=5, min_observations=3)
        model = optimizer(
            prior_estimator=prior(),
            distance_estimator=CovarianceDistance(covariance),
        )
        model.partial_fit(returns[:20])
        distribution = model.prior_estimator_.return_distribution_
        active = np.ones((10, 4), dtype=bool)
        active[:, 1] = False
        with pytest.raises(ValueError, match="Observation metadata"):
            model.partial_fit(returns[20:30], active_mask=active)
        assert model.prior_estimator_.return_distribution_ is distribution

        # The distance learns prior scenarios, whose mask is not the batch mask.
        covariance.set_fit_request(active_mask=False)
        model.fit(returns[:20])
        model.partial_fit(returns[20:30], active_mask=active)
        assert model.weights_[1] == 0
        assert model.distance_estimator_.n_features_in_ == 3


@pytest.mark.parametrize("risk", [RiskMeasure.CVAR, RiskMeasure.WORST_REALIZATION])
def test_positive_returns_keep_existing_signed_tail_risks(returns, risk):
    model = HierarchicalRiskParity(risk_measure=risk).fit(returns + 0.1)
    assert np.isfinite(model.weights_).all()
    assert model.weights_.sum() == pytest.approx(1)


def test_online_updates_after_serialization(optimizer, returns):
    model = optimizer(
        prior_estimator=prior(), distance_estimator=CovarianceDistance("precomputed")
    )
    model.partial_fit(returns[:20])
    restored = pickle.loads(pickle.dumps(model))
    model.partial_fit(returns[20:30])
    restored.partial_fit(returns[20:30])
    np.testing.assert_allclose(restored.weights_, model.weights_)
