import numpy as np
import pytest
from sklearn import config_context

from skfolio.moments import ImpliedCovariance
from skfolio.prior import EmpiricalPrior


def test_empirical_prior(X):
    model = EmpiricalPrior()
    model.fit(X)
    res = model.return_distribution_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    np.testing.assert_almost_equal(
        res.mu,
        np.array(
            [
                1.04344495e-03,
                1.90156515e-03,
                5.80763817e-04,
                7.36759751e-04,
                5.06726281e-04,
                -9.94537558e-05,
                8.04565487e-04,
                4.65724603e-04,
                6.21142195e-04,
                3.91538834e-04,
                1.10235332e-03,
                5.95227119e-04,
                1.03408770e-03,
                5.35320353e-04,
                4.93494909e-04,
                4.64948611e-04,
                2.10707897e-04,
                1.05905502e-03,
                4.34667892e-04,
                3.66200428e-04,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        res.covariance[:5, :5],
        np.array(
            [
                [0.00033685, 0.00028313, 0.00015618, 0.00017581, 0.00012206],
                [0.00028313, 0.00139232, 0.00022058, 0.0002515, 0.00017624],
                [0.00015618, 0.00022058, 0.0003929, 0.00019494, 0.00022071],
                [0.00017581, 0.0002515, 0.00019494, 0.00061523, 0.00014713],
                [0.00012206, 0.00017624, 0.00022071, 0.00014713, 0.00036072],
            ]
        ),
    )
    np.testing.assert_almost_equal(res.returns, np.asarray(X))
    assert res.cholesky is None


def test_empirical_prior_log_normal(X):
    model1 = EmpiricalPrior()
    model2 = EmpiricalPrior(is_log_normal=True, investment_horizon=1)

    model1.fit(X)
    model2.fit(X)

    np.testing.assert_almost_equal(
        model1.return_distribution_.mu, model2.return_distribution_.mu, 4
    )

    np.testing.assert_almost_equal(
        model1.return_distribution_.covariance,
        model2.return_distribution_.covariance,
        4,
    )


def test_empirical_prior_log_normal_investment_horizon(X):
    model = EmpiricalPrior(is_log_normal=True, investment_horizon=252)
    model.fit(X)
    res = model.return_distribution_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    np.testing.assert_almost_equal(
        res.mu,
        np.array(
            [
                0.30067519,
                0.61216954,
                0.15758953,
                0.20469753,
                0.13647218,
                -0.02469321,
                0.2249461,
                0.12453549,
                0.16940725,
                0.10375612,
                0.3198692,
                0.16179255,
                0.29758174,
                0.14443353,
                0.13237546,
                0.12428224,
                0.05359132,
                0.3058039,
                0.11575736,
                0.09670825,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        res.covariance[:5, :5],
        np.array(
            [
                [0.15002375, 0.15579818, 0.06082337, 0.07137514, 0.04671937],
                [0.15579818, 1.05213292, 0.1079466, 0.12822831, 0.08449074],
                [0.06082337, 0.1079466, 0.13914677, 0.07028907, 0.0754463],
                [0.07137514, 0.12822831, 0.07028907, 0.24792103, 0.05187185],
                [0.04671937, 0.08449074, 0.0754463, 0.05187185, 0.12443769],
            ]
        ),
    )
    np.testing.assert_almost_equal(res.returns, np.asarray(X))
    assert res.cholesky is None


def test_metadata_routing(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = EmpiricalPrior(
            covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
        )

        with pytest.raises(ValueError):
            model.fit(X)

        model.fit(X, implied_vol=implied_vol)

    # noinspection PyUnresolvedReferences
    assert model.covariance_estimator_.r2_scores_.shape == (20,)
