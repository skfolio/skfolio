from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context

from skfolio.moments import ImpliedCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
)


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

            with pytest.raises(ValueError):
                model.fit(X)

            model.fit(X, implied_vol=implied_vol)

        # noinspection PyUnresolvedReferences
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

            with pytest.raises(ValueError):
                model.fit(X)

            model.fit(X, implied_vol=implied_vol)

        # noinspection PyUnresolvedReferences
        assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)
