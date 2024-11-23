import numpy as np
import pytest
import scipy as sc
from sklearn import config_context

from skfolio.moments import ImpliedCovariance
from skfolio.prior import EmpiricalPrior
from skfolio.uncertainty_set import (
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)


class TestEmpiricalMuUncertaintySet:
    def test_fit(self, X):
        model = EmpiricalMuUncertaintySet()
        model.fit(X)
        w = np.array(
            [
                0.77788332,
                0.85449662,
                0.79007352,
                0.03013433,
                0.16967223,
                0.80579971,
                0.25336184,
                0.11030346,
                0.11823236,
                0.94095405,
                0.22376926,
                0.86762532,
                0.98096903,
                0.30642242,
                0.94903522,
                0.49107811,
                0.64132217,
                0.67429886,
                0.01153626,
                0.98177423,
            ]
        )
        c1 = model.uncertainty_set_.k * np.linalg.norm(
            sc.linalg.sqrtm(model.uncertainty_set_.sigma) @ w, 2
        )
        np.testing.assert_almost_equal(c1, 0.007086160726324358)

        np.testing.assert_almost_equal(model.uncertainty_set_.k, 5.604501123581913)
        np.testing.assert_almost_equal(
            model.uncertainty_set_.sigma[:10, :10],
            np.array(
                [
                    [
                        1.48851935e-07,
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
                        6.15255522e-07,
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
                        1.73617772e-07,
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
                        2.71863454e-07,
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
                        1.59398354e-07,
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
                        2.13290775e-07,
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
                        1.03923694e-07,
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
                        5.78584708e-08,
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
                        1.33088061e-07,
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
                        5.92709773e-08,
                    ],
                ]
            ),
        )

        model = EmpiricalMuUncertaintySet(diagonal=False)
        model.fit(X)

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = EmpiricalMuUncertaintySet(
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


class TestEmpiricalCovarianceUncertaintySet:
    def test_fit(self, X):
        model = EmpiricalCovarianceUncertaintySet()
        model.fit(X)
        np.testing.assert_almost_equal(model.uncertainty_set_.k, 21.15732657569969)
        np.testing.assert_almost_equal(
            model.uncertainty_set_.sigma[:10, :10],
            np.array(
                [
                    [
                        1.00282123e-10,
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
                        2.07250009e-10,
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
                        5.84834814e-11,
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
                        9.15777287e-11,
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
                        5.36936429e-11,
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
                        7.18474089e-11,
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
                        3.50068966e-11,
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
                        1.94897375e-11,
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
                        4.48309701e-11,
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
                        1.99655430e-11,
                    ],
                ]
            ),
            9,
        )

        model = EmpiricalCovarianceUncertaintySet(diagonal=False)
        model.fit(X)

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = EmpiricalCovarianceUncertaintySet(
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
