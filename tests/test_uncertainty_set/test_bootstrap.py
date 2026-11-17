from __future__ import annotations

import tracemalloc

import numpy as np
import pytest
import scipy.linalg as sla
from sklearn import config_context

from skfolio.moments import ImpliedCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
)
from skfolio.utils import bootstrap


class TestBootstrapMuUncertaintySet:
    def test_fit(self, X):
        model = BootstrapMuUncertaintySet(seed=42)
        model.fit(X)
        np.testing.assert_almost_equal(model.uncertainty_set_.radius, 5.604501123581913)
        np.testing.assert_allclose(
            (model.uncertainty_set_.geometry @ model.uncertainty_set_.geometry.T)[
                :10, :10
            ],
            np.diag(
                [
                    1.39919678e-07,
                    5.58537502e-07,
                    1.67361847e-07,
                    2.66162131e-07,
                    1.50144227e-07,
                    2.26308221e-07,
                    9.34421406e-08,
                    4.79060931e-08,
                    1.15293461e-07,
                    5.25031279e-08,
                ]
            ),
            rtol=1e-6,
            atol=0,
        )

        model = BootstrapMuUncertaintySet(diagonal=False, seed=42)
        model.fit(X)

    def test_seed_makes_the_bootstrap_reproducible(self, X):
        """The same seed produces identical bootstrap geometry."""
        first = BootstrapMuUncertaintySet(seed=42).fit(X)
        second = BootstrapMuUncertaintySet(seed=42).fit(X)

        np.testing.assert_array_equal(
            first.uncertainty_set_.geometry, second.uncertainty_set_.geometry
        )

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = BootstrapMuUncertaintySet(
                n_bootstrap_samples=20,
                prior_estimator=EmpiricalPrior(
                    covariance_estimator=ImpliedCovariance().set_fit_request(
                        implied_vol=True
                    )
                ),
            )

            with pytest.raises(ValueError, match="`implied_vol` cannot be None"):
                model.fit(X)

            model.fit(X, implied_vol=implied_vol)

        assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)


class TestBootstrapCovarianceUncertaintySet:
    def test_fit(self, X):
        model = BootstrapCovarianceUncertaintySet(seed=42)
        model.fit(X)
        np.testing.assert_almost_equal(model.uncertainty_set_.radius, 21.15732657569969)
        np.testing.assert_allclose(
            (model.uncertainty_set_.geometry @ model.uncertainty_set_.geometry.T)[
                :10, :10
            ],
            np.diag(
                [
                    7.45191137e-10,
                    9.61007843e-10,
                    9.17238818e-10,
                    6.30304305e-10,
                    7.59943831e-10,
                    7.14995484e-10,
                    6.73925644e-10,
                    2.29038255e-10,
                    7.87354582e-10,
                    2.49854202e-10,
                ]
            ),
            rtol=1e-6,
            atol=0,
        )

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = BootstrapCovarianceUncertaintySet(
                n_bootstrap_samples=20,
                prior_estimator=EmpiricalPrior(
                    covariance_estimator=ImpliedCovariance().set_fit_request(
                        implied_vol=True
                    )
                ),
            )

            with pytest.raises(ValueError, match="`implied_vol` cannot be None"):
                model.fit(X)

            model.fit(X, implied_vol=implied_vol)

        assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)


def test_bootstrap_covariance_full_geometry(X_small):
    X_sub = X_small.iloc[:, :4]
    model = BootstrapCovarianceUncertaintySet(
        diagonal=False, n_bootstrap_samples=20, seed=42
    )
    model.fit(X_sub)
    geometry = model.uncertainty_set_.geometry
    assert geometry.shape == (16, 16)
    assert np.all(np.isfinite(geometry))
    np.testing.assert_allclose(geometry, geometry.T, atol=1e-12)


@pytest.mark.parametrize(
    "estimator", [BootstrapMuUncertaintySet, BootstrapCovarianceUncertaintySet]
)
@pytest.mark.parametrize("diagonal", [True, False])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.int32])
def test_fit_matches_dense_bootstrap(estimator, diagonal, dtype):
    """Uncertainty fits match dense-sample statistics and preserve caller ownership."""
    returns = np.random.default_rng(7).normal(size=(32, 3)).astype(dtype)
    before = returns.copy()
    prior = EmpiricalPrior().fit(returns)
    samples = bootstrap.stationary_bootstrap(
        prior.return_distribution_.returns, 20, block_size=3, seed=42
    )
    expected_state = np.random.get_state()
    if estimator is BootstrapMuUncertaintySet:
        mus = np.mean(samples, axis=1).astype(float)
        deviations = mus - prior.return_distribution_.mu
    else:
        deviations = np.array([np.cov(sample.T).ravel(order="F") for sample in samples])
    expected = (
        np.diag(np.sqrt(np.var(deviations, axis=0, ddof=1)))
        if diagonal
        else sla.sqrtm(np.cov(deviations, rowvar=False)).real
    )
    model = estimator(
        prior_estimator=prior,
        n_bootstrap_samples=20,
        block_size=3,
        diagonal=diagonal,
        seed=42,
    ).fit(returns)

    # Exact equality also checks that mean accumulation precision is unchanged.
    np.testing.assert_array_equal(model.uncertainty_set_.geometry, expected)
    assert model.uncertainty_set_.norm == 2
    actual_state = np.random.get_state()
    np.testing.assert_array_equal(actual_state[1], expected_state[1])
    assert actual_state[0] == expected_state[0]
    assert actual_state[2:] == expected_state[2:]
    np.testing.assert_array_equal(returns, before)
    assert model.prior_estimator_ is not prior


@pytest.mark.parametrize(
    "estimator", [BootstrapMuUncertaintySet, BootstrapCovarianceUncertaintySet]
)
def test_fit_avoids_bootstrap_cube_allocation(estimator):
    """Fitting does not allocate all bootstrap samples at once."""
    returns = np.random.default_rng(7).normal(size=(256, 10))
    n_bootstrap_samples = 1000
    model = estimator(n_bootstrap_samples=n_bootstrap_samples, block_size=3, seed=42)

    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        model.fit(returns)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()

    # Peak allocations must stay below half the size of all resampled returns.
    assert peak - baseline < n_bootstrap_samples * returns.nbytes // 2


@pytest.mark.parametrize(
    "estimator", [BootstrapMuUncertaintySet, BootstrapCovarianceUncertaintySet]
)
@pytest.mark.parametrize("seed", [42, None])
def test_fit_preserves_global_random_state(estimator, seed):
    """Fitting does not change NumPy's global random state."""
    returns = np.random.default_rng(7).normal(size=(32, 3))
    previous_state = np.random.get_state()
    try:
        # An unrelated initial seed prevents consecutive identical fits from
        # hiding a reseed followed by the same bootstrap draw sequence.
        np.random.seed(7)
        expected_state = np.random.get_state()

        estimator(n_bootstrap_samples=20, block_size=3, seed=seed).fit(returns)

        actual_state = np.random.get_state()
        np.testing.assert_array_equal(actual_state[1], expected_state[1])
        assert actual_state[0] == expected_state[0]
        assert actual_state[2:] == expected_state[2:]
    finally:
        np.random.set_state(previous_state)
