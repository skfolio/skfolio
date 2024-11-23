import numpy as np
import pytest
from sklearn import config_context

from skfolio.moments import (
    ImpliedCovariance,
)
from skfolio.optimization.naive import EqualWeighted, InverseVolatility, Random
from skfolio.prior import EmpiricalPrior, FactorModel


class TestInverseVolatility:
    def test_fit(self, X, y):
        model = InverseVolatility()
        model.fit(X)
        np.testing.assert_almost_equal(sum(model.weights_), 1)
        w = 1 / np.std(np.asarray(X), axis=0)
        w /= sum(w)
        np.testing.assert_almost_equal(model.weights_, w)

        model = InverseVolatility(prior_estimator=FactorModel())
        model.fit(X, y)

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = InverseVolatility(
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


class TestEqualWeighted:
    def test_fit(self, X):
        model = EqualWeighted()
        model.fit(X)
        weights = model.weights_
        np.testing.assert_almost_equal(sum(weights), 1)
        w = 1 / X.shape[1]
        np.testing.assert_almost_equal(weights, w)


class TestRandom:
    def test_fit(self, X):
        model = Random()
        weights = model.fit(X).weights_
        np.testing.assert_almost_equal(sum(weights), 1)
