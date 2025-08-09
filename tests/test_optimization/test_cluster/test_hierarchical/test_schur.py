import numpy as np
import pytest

from skfolio.cluster import HierarchicalClustering
from skfolio.datasets import (
    load_sp500_dataset,
)
from skfolio.moments import EWCovariance
from skfolio.optimization import (
    HierarchicalRiskParity,
    MeanRisk,
    SchurComplementary,
)
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior, FactorModel


@pytest.fixture(scope="module")
def full_X():
    prices = load_sp500_dataset()
    full_X = prices_to_returns(prices)
    return full_X


@pytest.mark.parametrize(
    "date_range",
    [slice(None), slice("2010", None), slice("2015", None), slice("2010", "2015")],
)
def test_schur_frontier(full_X, date_range):
    rets = full_X[date_range]
    prev_ptf = None
    for gamma in np.linspace(0, 1.0, 50):
        schur = SchurComplementary(gamma=gamma)
        ptf = schur.fit_predict(rets)
        if prev_ptf is not None:
            assert ptf.variance <= prev_ptf.variance + 1e-8
            assert ptf.mean <= prev_ptf.mean + 1e-5
        prev_ptf = ptf


def test_schur_default(X):
    model = SchurComplementary()
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.00290673,
                0.00080765,
                0.00947158,
                0.02841936,
                0.02198382,
                0.05570452,
                0.0727225,
                0.10366611,
                0.01190645,
                0.06332227,
                0.07082402,
                0.11164499,
                0.00610297,
                0.06378797,
                0.04052787,
                0.11672992,
                0.00708368,
                0.0310986,
                0.14080058,
                0.0404884,
            ]
        ),
    )


def test_schur_hrp_min_var(X):
    hrp = HierarchicalRiskParity()
    min_var = MeanRisk()
    schur_0 = SchurComplementary(gamma=0)
    schur_05 = SchurComplementary(gamma=0.5)

    ptf_hrp = hrp.fit_predict(X)
    ptf_min_var = min_var.fit_predict(X)
    ptf_schur_0 = schur_0.fit_predict(X)
    ptf_schur_05 = schur_05.fit_predict(X)

    np.testing.assert_array_almost_equal(ptf_hrp.weights, ptf_schur_0.weights)

    assert ptf_hrp.variance > ptf_min_var.variance
    assert ptf_hrp.mean > ptf_min_var.mean

    assert ptf_schur_05.variance > ptf_min_var.variance
    assert ptf_schur_05.mean > ptf_min_var.mean

    assert ptf_schur_05.variance < ptf_hrp.variance
    assert ptf_schur_05.mean < ptf_hrp.mean


def test_schur_prior_estimator(X):
    model = SchurComplementary(
        prior_estimator=EmpiricalPrior(covariance_estimator=EWCovariance())
    )
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.0017058,
                0.0007265,
                0.0853799,
                0.0231435,
                0.0550234,
                0.0159805,
                0.028233,
                0.1062284,
                0.1140945,
                0.045891,
                0.0509539,
                0.1137293,
                0.0056005,
                0.109078,
                0.0218608,
                0.1001613,
                0.0072586,
                0.0357706,
                0.0445501,
                0.0346304,
            ]
        ),
    )


def test_schur_factor_model(X, y):
    model = SchurComplementary(prior_estimator=FactorModel())
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.0412263,
                0.0053729,
                0.0013345,
                0.0085172,
                0.0157929,
                0.000797,
                0.0415737,
                0.1538852,
                0.0438251,
                0.1222923,
                0.0262726,
                0.0444741,
                0.0487746,
                0.1262759,
                0.0876273,
                0.088709,
                0.0001549,
                0.0460946,
                0.054206,
                0.0427938,
            ]
        ),
    )


def test_schur_linkage(X, linkage_method):
    model = SchurComplementary(
        hierarchical_clustering_estimator=HierarchicalClustering(
            linkage_method=linkage_method
        ),
    )
    model.fit(X)


def test_hrp_weight_constraints(X):
    model = SchurComplementary()
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.0029067297577)
    np.testing.assert_almost_equal(model.weights_[-1], 0.040488396016)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Min Weights
    model.set_params(min_weights={"AAPL": 0.05, "XOM": 0.08})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.05)
    np.testing.assert_almost_equal(model.weights_[-1], 0.08)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(min_weights=0.05)
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) * 0.05)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Max Weights
    model.set_params(min_weights=0)
    model.set_params(max_weights={"AAPL": 0.001, "XOM": 0.03})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.001)
    np.testing.assert_almost_equal(model.weights_[-1], 0.03)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(max_weights=0.05)
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) * 0.05)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    # Both
    model.set_params(min_weights={"AAPL": 0.05}, max_weights={"XOM": 0.03})
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[0], 0.05)
    np.testing.assert_almost_equal(model.weights_[-1], 0.03)
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)

    model.set_params(min_weights=0.01, max_weights=0.08)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.01,
                0.01,
                0.0136739,
                0.0227483,
                0.0317375,
                0.08,
                0.0582109,
                0.08,
                0.0171891,
                0.08,
                0.08,
                0.08,
                0.01,
                0.08,
                0.0666322,
                0.08,
                0.0102266,
                0.0511294,
                0.08,
                0.0584522,
            ]
        ),
    )
    assert model.effective_gamma_ == 0.5
    np.testing.assert_almost_equal(sum(model.weights_), 1.0)


def test_hrp_weight_constraints_error(X):
    model = SchurComplementary(min_weights=0.03, max_weights=0.06)
    model.fit(X)
    assert model.effective_gamma_ == 0.0
    assert model.weights_ is not None
    assert not np.any(np.isnan(model.weights_))

    model = SchurComplementary(min_weights=0.03, max_weights=0.06, keep_monotonic=False)
    model.fit(X)
    assert model.effective_gamma_ == 0.5
    assert not np.any(np.isnan(model.weights_))
