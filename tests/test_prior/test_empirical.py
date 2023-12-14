import datetime as dt
import numpy as np
import pytest

from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2014, 1, 1) :]
    X = prices_to_returns(X=prices)
    return X


def test_empirical_prior(X):
    model = EmpiricalPrior()
    model.fit(X)
    res = model.prior_model_
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
    model = EmpiricalPrior(is_log_normal=True, investment_horizon=255)
    model.fit(X)
    res = model.prior_model_
    assert hash(res)
    assert res.mu.shape == (20,)
    assert res.covariance.shape == (20, 20)
    np.testing.assert_almost_equal(
        res.mu,
        np.array(
            [
                1.30475211,
                1.62136161,
                1.15960797,
                1.20737132,
                1.13820431,
                0.97501653,
                1.22790845,
                1.12610786,
                1.17158795,
                1.10505405,
                1.32423721,
                1.16386854,
                1.30161207,
                1.14627304,
                1.13405258,
                1.12585123,
                1.05424632,
                1.30995827,
                1.11721323,
                1.09791416,
            ]
        ),
    )
    np.testing.assert_almost_equal(
        res.covariance[:5, :5],
        np.array(
            [
                [0.15284131, 0.15911746, 0.06186271, 0.07263143, 0.04750486],
                [0.15911746, 1.07913865, 0.11008322, 0.1308346, 0.0861384],
                [0.06186271, 0.11008322, 0.14137923, 0.07142908, 0.07661981],
                [0.07263143, 0.1308346, 0.07142908, 0.25223024, 0.05269778],
                [0.04750486, 0.0861384, 0.07661981, 0.05269778, 0.12637346],
            ]
        ),
    )
    np.testing.assert_almost_equal(res.returns, np.asarray(X))
    assert res.cholesky is None
