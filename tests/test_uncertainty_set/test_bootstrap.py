import numpy as np
import pytest
from sklearn import config_context

from skfolio.moments import ImpliedCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.uncertainty_set import (
    BootstrapCovarianceUncertaintySet,
    BootstrapMuUncertaintySet,
    EmpiricalMuUncertaintySet,
)


class TestBootstrapMuUncertaintySet:
    def test_fit(self, X):
        model = BootstrapMuUncertaintySet()
        model.fit(X)
        np.testing.assert_almost_equal(model.uncertainty_set_.k, 5.604501123581913)
        np.testing.assert_almost_equal(
            model.uncertainty_set_.sigma[:10, :10],
            np.array(
                [
                    [
                        1.30559015e-07,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        5.26628755e-07,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        1.69447847e-07,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        2.59822980e-07,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        1.54717635e-07,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        2.24606860e-07,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        9.55288679e-08,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        4.79065829e-08,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        1.18267526e-07,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        5.61971181e-08,
                    ],
                ]
            ),
        )

        model = EmpiricalMuUncertaintySet(diagonal=False)
        model.fit(X)

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = BootstrapMuUncertaintySet(
                prior_estimator=EmpiricalPrior(
                    covariance_estimator=ImpliedCovariance().set_fit_request(
                        implied_vol=True
                    )
                )
            )

            with pytest.raises(ValueError):
                model.fit(X)

            model.fit(X, implied_vol=implied_vol)

        # noinspection PyUnresolvedReferences
        assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)


class TestBootstrapCovarianceUncertaintySet:
    def test_fit(self, X):
        model = BootstrapCovarianceUncertaintySet()
        model.fit(X)
        np.testing.assert_almost_equal(model.uncertainty_set_.k, 21.15732657569969)
        np.testing.assert_almost_equal(
            model.uncertainty_set_.sigma[:10, :10],
            np.array(
                [
                    [
                        7.43059442e-10,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        9.81212261e-10,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        8.86266409e-10,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        5.79691855e-10,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        7.19047519e-10,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        6.68327332e-10,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        6.51800272e-10,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        2.21389783e-10,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        7.52747813e-10,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        2.39404329e-10,
                    ],
                ]
            ),
            9,
        )

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = BootstrapCovarianceUncertaintySet(
                prior_estimator=EmpiricalPrior(
                    covariance_estimator=ImpliedCovariance().set_fit_request(
                        implied_vol=True
                    )
                )
            )

            with pytest.raises(ValueError):
                model.fit(X)

            model.fit(X, implied_vol=implied_vol)

        # noinspection PyUnresolvedReferences
        assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)
