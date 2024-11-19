import numpy as np
import pytest
from sklearn import config_context

from skfolio.moments import ImpliedCovariance
from skfolio.optimization.convex import (
    MaximumDiversification,
)
from skfolio.prior import EmpiricalPrior, FactorModel


def test_maximum_diversification(X):
    model = MaximumDiversification()
    model.fit(X)
    ptf = model.predict(X)
    diversification = (
        model.problem_values_["expected_return"] / model.problem_values_["risk"]
    )
    np.testing.assert_almost_equal(ptf.diversification, diversification, 3)


def test_maximum_diversification_factor(X, y):
    model = MaximumDiversification(prior_estimator=FactorModel())
    model.fit(X, y)
    ptf = model.predict(X)
    diversification = (
        model.problem_values_["expected_return"] / model.problem_values_["risk"]
    )

    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(ptf.diversification, diversification, 3)


def test_metadata_routing(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = MaximumDiversification(
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
