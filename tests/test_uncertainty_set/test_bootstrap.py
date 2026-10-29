from __future__ import annotations

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
def test_fit_reduces_one_sample_at_a_time(monkeypatch, estimator, diagonal, dtype):
    """Uncertainty fits preserve dense-sample results without constructing the cube."""
    returns = np.random.default_rng(7).normal(size=(32, 3)).astype(dtype)
    before = returns.copy()
    prior = EmpiricalPrior().fit(returns)
    samples = bootstrap.stationary_bootstrap(
        prior.return_distribution_.returns, 20, block_size=3, seed=42
    )
    expected_state = np.random.get_state()
    if estimator is BootstrapMuUncertaintySet:
        deviations = np.mean(samples, axis=1) - prior.return_distribution_.mu
    else:
        deviations = np.array([np.cov(sample.T).ravel(order="F") for sample in samples])
    expected = (
        np.diag(np.sqrt(np.var(deviations, axis=0, ddof=1)))
        if diagonal
        else sla.sqrtm(np.cov(deviations, rowvar=False)).real
    )
    indexed_shapes = []
    vstack = np.vstack

    class SampleReturns(np.ndarray):
        """Reject all-sample indexing at the shared allocation owner."""

        def __getitem__(self, key):
            """Observe indexing before a bootstrap cube can be allocated."""
            if isinstance(key, tuple) and isinstance(key[0], np.ndarray):
                indexed_shapes.append(key[0].shape)
                assert key[0].ndim == 1, "fit must reduce one sample at a time"
            return super().__getitem__(key)

    def tracked_vstack(arrays):
        """Keep ordinary stacking while observing its later advanced indexing."""
        return vstack(arrays).view(SampleReturns)

    monkeypatch.setattr(bootstrap.np, "vstack", tracked_vstack)
    model = estimator(
        prior_estimator=prior,
        n_bootstrap_samples=20,
        block_size=3,
        diagonal=diagonal,
        seed=42,
    ).fit(returns)

    assert indexed_shapes == [(32,)] * 20
    # Exact dense-reference equality catches changed float32 mean accumulation.
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
@pytest.mark.parametrize("seed", [42, None])
def test_fit_preserves_global_random_state(estimator, seed):
    """Sample-local reductions retain the native bootstrap's isolated RNG."""
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
