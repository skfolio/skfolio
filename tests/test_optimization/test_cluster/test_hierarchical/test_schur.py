from __future__ import annotations

import warnings

import numpy as np
import pytest

from skfolio.cluster import HierarchicalClustering
from skfolio.datasets import (
    load_sp500_dataset,
)
from skfolio.moments import EWCovariance
from skfolio.optimization import (
    HierarchicalRiskParity,
    MeanRisk,
    SchurComplementary,
)
from skfolio.optimization.cluster.hierarchical import _schur
from skfolio.optimization.cluster.hierarchical._schur import _compute_weights
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior, TimeSeriesFactorModel


@pytest.fixture(scope="module")
def full_X():
    prices = load_sp500_dataset()
    full_X = prices_to_returns(prices)
    return full_X


@pytest.mark.parametrize(
    "date_range",
    [slice(None), slice("2010", None), slice("2015", None), slice("2010", "2015")],
)
def test_schur_frontier(full_X, date_range):
    rets = full_X[date_range]
    prev_ptf = None
    for gamma in np.linspace(0, 1.0, 50):
        schur = SchurComplementary(gamma=gamma)
        ptf = schur.fit_predict(rets)
        if prev_ptf is not None:
            assert ptf.variance <= prev_ptf.variance + 1e-8
            assert ptf.mean <= prev_ptf.mean + 1e-5
        prev_ptf = ptf


def test_schur_default(X):
    model = SchurComplementary()
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.00290673,
                0.00080765,
                0.00947158,
                0.02841936,
                0.02198382,
                0.05570452,
                0.0727225,
                0.10366611,
                0.01190645,
                0.06332227,
                0.07082402,
                0.11164499,
                0.00610297,
                0.06378797,
                0.04052787,
                0.11672992,
                0.00708368,
                0.0310986,
                0.14080058,
                0.0404884,
            ]
        ),
    )


def test_schur_hrp_min_var(X):
    hrp = HierarchicalRiskParity()
    min_var = MeanRisk()
    schur_0 = SchurComplementary(gamma=0)
    schur_05 = SchurComplementary(gamma=0.5)

    ptf_hrp = hrp.fit_predict(X)
    ptf_min_var = min_var.fit_predict(X)
    ptf_schur_0 = schur_0.fit_predict(X)
    ptf_schur_05 = schur_05.fit_predict(X)

    np.testing.assert_array_almost_equal(ptf_hrp.weights, ptf_schur_0.weights)

    assert ptf_hrp.variance > ptf_min_var.variance
    assert ptf_hrp.mean > ptf_min_var.mean

    assert ptf_schur_05.variance > ptf_min_var.variance
    assert ptf_schur_05.mean > ptf_min_var.mean

    assert ptf_schur_05.variance < ptf_hrp.variance
    assert ptf_schur_05.mean < ptf_hrp.mean


def test_schur_prior_estimator(X):
    model = SchurComplementary(
        prior_estimator=EmpiricalPrior(covariance_estimator=EWCovariance())
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.00003212,
                0.00001797,
                0.01774978,
                0.00850949,
                0.04422037,
                0.06736142,
                0.01204828,
                0.12571592,
                0.02177091,
                0.07554259,
                0.0660629,
                0.1241542,
                0.00006829,
                0.07776382,
                0.03005696,
                0.13499451,
                0.0098029,
                0.03833465,
                0.09975659,
                0.04603632,
            ]
        ),
    )


def test_schur_factor_model(X, factors):
    model = SchurComplementary(prior_estimator=TimeSeriesFactorModel())
    model.fit(X, factors=factors)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.0412263,
                0.0053729,
                0.0013345,
                0.0085172,
                0.0157929,
                0.000797,
                0.0415737,
                0.1538852,
                0.0438251,
                0.1222923,
                0.0262726,
                0.0444741,
                0.0487746,
                0.1262759,
                0.0876273,
                0.088709,
                0.0001549,
                0.0460946,
                0.054206,
                0.0427938,
            ]
        ),
    )


def test_schur_linkage(X, linkage_method):
    model = SchurComplementary(
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=linkage_method
        ),
    )
    model.fit(X)


def test_hrp_weight_constraints(X):
    model = SchurComplementary()
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.0029067297577)
    np.testing.assert_almost_equal(model.weights_[-1], 0.040488396016)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Min Weights
    model.set_params(min_weights={"AAPL": 0.05, "XOM": 0.08})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.05)
    np.testing.assert_almost_equal(model.weights_[-1], 0.08)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(min_weights=0.05)
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) * 0.05)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Max Weights
    model.set_params(min_weights=0)
    model.set_params(max_weights={"AAPL": 0.001, "XOM": 0.03})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.001)
    np.testing.assert_almost_equal(model.weights_[-1], 0.03)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(max_weights=0.05)
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) * 0.05)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Both
    model.set_params(min_weights={"AAPL": 0.05}, max_weights={"XOM": 0.03})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.05)
    np.testing.assert_almost_equal(model.weights_[-1], 0.03)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(min_weights=0.01, max_weights=0.08)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.01,
                0.01,
                0.0136739,
                0.0227483,
                0.0317375,
                0.08,
                0.0582109,
                0.08,
                0.0171891,
                0.08,
                0.08,
                0.08,
                0.01,
                0.08,
                0.0666322,
                0.08,
                0.0102266,
                0.0511294,
                0.08,
                0.0584522,
            ]
        ),
    )
    assert model.effective_gamma_ == 0.5
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)


def test_hrp_weight_constraints_error(X):
    model = SchurComplementary(min_weights=0.03, max_weights=0.06)
    model.fit(X)
    assert model.effective_gamma_ == 0.0
    assert model.weights_ is not None
    assert not np.any(np.isnan(model.weights_))

    model = SchurComplementary(min_weights=0.03, max_weights=0.06, keep_monotonic=False)
    model.fit(X)
    assert model.effective_gamma_ == 0.5
    assert not np.any(np.isnan(model.weights_))


def test_schur_invalid_gamma(X):
    model = SchurComplementary(gamma=1.5)
    with pytest.raises(ValueError, match=r"gamma must be between 0 and 1\. Got 1\.5"):
        model.fit(X)


@pytest.mark.parametrize("turning_gamma", [0.0, 0.03])
def test_schur_first_turning_point(monkeypatch, turning_gamma):
    def compute_weights(gamma, **kwargs):
        # Variance first turns at turning_gamma, then decreases again after 0.1.
        offset = (
            gamma - turning_gamma if gamma <= 0.1 else 0.15 - turning_gamma - gamma / 2
        )
        return np.array([0.5 + offset, 0.5 - offset])

    monkeypatch.setattr(_schur, "_compute_weights", compute_weights)
    weights, gamma = _schur._compute_monotonic_weights(
        max_gamma=0.2,
        sorted_assets=np.arange(2),
        covariance=np.eye(2),
        min_weights=np.zeros(2),
        max_weights=np.ones(2),
    )

    assert gamma == pytest.approx(turning_gamma, abs=1e-4)
    np.testing.assert_allclose(weights, [0.5, 0.5], atol=1e-4)


@pytest.mark.parametrize("turning_gamma", [1e-7, 1e-5, 0.025, 0.03])
def test_binary_search_gamma_and_weights(turning_gamma):
    def objective(gamma):
        assert 0 <= gamma <= 0.1
        return (gamma - turning_gamma) ** 2, np.array([gamma])

    weights, gamma = _schur._binary_search(
        objective,
        low_gamma=0,
        high_gamma=0.1,
        low_variance=turning_gamma**2,
        low_weights=np.array([0.0]),
    )

    # Retain small positive improvements even when the whole interval is below tol.
    assert gamma > 0
    assert abs(gamma - turning_gamma) <= min(1e-4, turning_gamma)
    np.testing.assert_array_equal(weights, [gamma])


@pytest.mark.parametrize("feasible", [True, False])
def test_binary_search_skips_rejected_slope_probes(feasible):
    calls = []

    def objective(gamma):
        calls.append(gamma)
        assert 0 <= gamma <= 0.1
        return (gamma**2, np.array([gamma])) if feasible else (np.inf, None)

    initial_weights = np.array([0.0]) if feasible else None
    weights, gamma = _schur._binary_search(
        objective,
        low_gamma=0,
        high_gamma=0.1,
        low_variance=0 if feasible else np.inf,
        low_weights=initial_weights,
    )
    assert weights is initial_weights
    assert gamma == 0

    # The search keeps its refinement depth without evaluating a slope for each
    # rejected midpoint. Previously this required 42 evaluations.
    assert len(calls) <= 21


@pytest.mark.parametrize("boundary", [0.025, 0.03, 0.034])
def test_binary_search_feasibility_boundary(boundary):
    def objective(gamma):
        if gamma > boundary:
            return np.inf, None
        return (gamma - 0.08) ** 2, np.array([gamma])

    weights, gamma = _schur._binary_search(
        objective,
        low_gamma=0,
        high_gamma=0.1,
        low_variance=0.08**2,
        low_weights=np.array([0.0]),
    )

    assert 0 <= boundary - gamma <= 1e-4
    np.testing.assert_array_equal(weights, [gamma])


@pytest.mark.parametrize("max_gamma", [0.2, 0.3], ids=["terminal", "mid_sweep"])
def test_schur_refinement_retains_lower_endpoint(monkeypatch, max_gamma):
    def compute_weights(gamma, **kwargs):
        # The sweep decreases through 0.2, but no refinement midpoint passes
        # both the variance and slope checks. The lower endpoint is gamma=0.1.
        t = (gamma - 0.1) / 0.1
        if t < 0:
            risk = 2 - 3 * t
        elif t <= 1:
            risk = 2 + t + 10 * t**2 - 30 * t**3 + 18 * t**4
        else:
            risk = 1 + 3 * (t - 1)
        variance = 0.55 + 0.025 * risk
        offset = np.sqrt((variance - 0.5) / 2)
        return np.array([0.5 + offset, 0.5 - offset])

    monkeypatch.setattr(_schur, "_compute_weights", compute_weights)
    weights, gamma = _schur._compute_monotonic_weights(
        max_gamma=max_gamma,
        sorted_assets=np.arange(2),
        covariance=np.eye(2),
        min_weights=np.zeros(2),
        max_weights=np.ones(2),
    )

    assert gamma == pytest.approx(0.1)
    np.testing.assert_allclose(weights, compute_weights(gamma))


@pytest.mark.parametrize("max_gamma", [1e-5, 1e-4])
@pytest.mark.parametrize("turning_fraction", [0.0, 0.3, 2.0])
def test_schur_small_gamma(monkeypatch, max_gamma, turning_fraction):
    calls = []
    turning_gamma = turning_fraction * max_gamma

    def compute_weights(gamma, **kwargs):
        calls.append(gamma)
        assert 0 <= gamma <= max_gamma
        offset = gamma - turning_gamma
        return np.array([0.5 + offset, 0.5 - offset])

    monkeypatch.setattr(_schur, "_compute_weights", compute_weights)
    weights, gamma = _schur._compute_monotonic_weights(
        max_gamma=max_gamma,
        sorted_assets=np.arange(2),
        covariance=np.eye(2),
        min_weights=np.zeros(2),
        max_weights=np.ones(2),
    )

    offset = gamma - turning_gamma
    np.testing.assert_array_equal(weights, [0.5 + offset, 0.5 - offset])
    if turning_fraction == 0:
        assert gamma == 0
    else:
        assert gamma > 0
        assert weights @ weights < 0.5 + 2 * turning_gamma**2
    if turning_fraction == 2:
        assert gamma == max_gamma
        # The terminal slope check reuses the initial variance at gamma=0.
        assert calls == [0, max_gamma]


def test_schur_weights_match_effective_gamma(X):
    model = SchurComplementary(
        gamma=1,
        prior_estimator=EmpiricalPrior(
            covariance_estimator=EWCovariance(half_life=30, min_observations=20)
        ),
    ).fit(X.iloc[:252])
    assert 0 < model.effective_gamma_ < model.gamma
    weights = model.weights_.copy()

    model.set_params(gamma=model.effective_gamma_, keep_monotonic=False).fit(
        X.iloc[:252]
    )
    np.testing.assert_allclose(weights, model.weights_, rtol=1e-12, atol=1e-14)


@pytest.fixture
def non_spd_schur_inputs():
    # The covariance is positive definite, but nearly rank one. Its left Schur
    # block has negative variances, which correlation clipping cannot repair.
    v = np.array([1.0, 2.0, 3.0, 4.0])
    covariance = np.outer(v, v) + 1e-6 * np.eye(4)
    sorted_assets = np.array([2, 3, 0, 1])
    return covariance, sorted_assets


def _compute_weights_from(inputs, force_spd):
    covariance, sorted_assets = inputs
    return _compute_weights(
        gamma=0.5,
        sorted_assets=sorted_assets,
        covariance=covariance,
        max_weights=np.ones(4),
        min_weights=np.zeros(4),
        force_spd=force_spd,
    )


def test_compute_weights_rejects_unrepairable_block(non_spd_schur_inputs):
    assert _compute_weights_from(non_spd_schur_inputs, force_spd=False) is None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with pytest.raises(
            ValueError, match=r"Schur complement failed with gamma=0\.5000"
        ):
            _compute_weights_from(non_spd_schur_inputs, force_spd=True)
    assert not [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    "bad_left,bad_right", [(True, False), (False, True), (True, True)]
)
def test_compute_weights_force_spd_repairs_blocks(monkeypatch, bad_left, bad_right):
    # Simulate indefinite augmented blocks with positive variances so the real
    # correlation-clipping repair can run. Exercise left, right, and both repairs.
    bad_block = np.array([[1.0, 1.1], [1.1, 1.0]])
    blocks = [bad_block if bad else np.eye(2) for bad in (bad_left, bad_right)]
    augmented = iter(blocks)
    monkeypatch.setattr(_schur, "_schur_augmentation", lambda *a, **kw: next(augmented))
    calls = []
    cov_nearest = _schur.cov_nearest

    def record_repair(cov):
        calls.append(cov.copy())
        return cov_nearest(cov)

    monkeypatch.setattr(_schur, "cov_nearest", record_repair)
    weights = _compute_weights_from((np.eye(4), np.arange(4)), force_spd=True)

    assert len(calls) == bad_left + bad_right
    for block in calls:
        np.testing.assert_array_equal(block, bad_block)
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert np.all((weights >= 0) & (weights <= 1))


def test_compute_weights_force_spd_failure_raises(non_spd_schur_inputs, monkeypatch):
    def failing_cov_nearest(cov):
        raise np.linalg.LinAlgError("cannot repair")

    monkeypatch.setattr(_schur, "cov_nearest", failing_cov_nearest)
    with pytest.raises(ValueError, match=r"Schur complement failed with gamma=0\.5000"):
        _compute_weights_from(non_spd_schur_inputs, force_spd=True)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_compute_weights_uses_repaired_blocks_in_later_splits(monkeypatch):
    bad_block = np.eye(4)
    bad_block[0, 1] = bad_block[1, 0] = 1.1
    root_blocks = iter([bad_block, np.eye(4)])
    schur_augmentation = _schur._schur_augmentation
    later_blocks = []

    def augment(a, b, d, gamma):
        if len(a) == 4:
            return next(root_blocks)
        later_blocks.append(a.copy())
        return schur_augmentation(a, b, d, gamma=gamma)

    monkeypatch.setattr(_schur, "_schur_augmentation", augment)
    weights = _compute_weights(
        gamma=0.5,
        sorted_assets=np.arange(8),
        covariance=np.eye(8),
        max_weights=np.ones(8),
        min_weights=np.zeros(8),
        force_spd=True,
    )

    assert len(later_blocks) == 4
    for block in later_blocks:
        assert np.all(np.linalg.eigvalsh(block) > 0)
    assert np.all(np.isfinite(weights))
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert np.all((weights >= 0) & (weights <= 1))
