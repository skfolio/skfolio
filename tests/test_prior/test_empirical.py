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
                1.30067519,
                1.61216954,
                1.15758953,
                1.20469753,
                1.13647218,
                0.97530679,
                1.2249461,
                1.12453549,
                1.16940725,
                1.10375612,
                1.3198692,
                1.16179255,
                1.29758174,
                1.14443353,
                1.13237546,
                1.12428224,
                1.05359132,
                1.3058039,
                1.11575736,
                1.09670825,
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
