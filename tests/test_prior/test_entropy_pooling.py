import operator

import numpy as np
import pytest

import skfolio.measures as sm
from skfolio.distribution import Gaussian, GaussianCopula, VineCopula
from skfolio.exceptions import GroupNotFoundError
from skfolio.prior import EntropyPooling, FactorModel, SyntheticData
from skfolio.prior._entropy_pooling import (
    _extract_prior_assets,
    _parse_correlation_view,
    _replace_prior_views,
)


# 23
@pytest.fixture(
    scope="module",
    params=["TNC", "CLARABEL"],
)
def solver(request):
    return request.param


@pytest.fixture(scope="module")
def views():
    views = [
        "AAPL>= prior(AAPL)*3.25",
        "AAPL >=3.25*prior(ABL)",
        "AAPL >= 3.5 * prior(XXL)",
        r"AAPL >=prior(A_1) * 3.5",
        r"AAPL ==prior(A B)* 3.5",
        r"AAPL <=   prior(A 2) *3.5 ",
        "AAPL >=   A3 ",
        "BA/ LN Equity ==   prior(BA/ LN Equity)",
    ]
    return views


def test_no_views(X, solver):
    model = EntropyPooling(solver=solver)
    model.fit(X)
    np.testing.assert_almost_equal(
        model.return_distribution_.sample_weight, np.ones(len(X)) / len(X)
    )
    np.testing.assert_almost_equal(model.relative_entropy_, 0.0)
    np.testing.assert_almost_equal(model.effective_number_of_scenarios_, len(X))


def test_mean_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AAPL >= 0.002",
            "AMD == 0.003",
            "BAC <= 0.0001",
            "UNH >= 0.0",
            "WMT <= 0.5",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    mean = sm.mean(np.array(X), sw)

    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.00298, 5)
    np.testing.assert_almost_equal(model.effective_number_of_scenarios_, 2256.26377, 1)
    np.testing.assert_almost_equal(mean[0], 0.002, 5)
    np.testing.assert_almost_equal(mean[1], 0.003, 5)
    np.testing.assert_almost_equal(mean[2], 0.0001, 5)
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 5)
    np.testing.assert_array_almost_equal(
        mean,
        [
            2.00e-03,
            3.00e-03,
            1.00e-04,
            8.10e-04,
            5.40e-04,
            4.00e-05,
            1.01e-03,
            5.70e-04,
            3.40e-04,
            4.70e-04,
            1.28e-03,
            6.70e-04,
            1.50e-03,
            7.00e-04,
            5.90e-04,
            6.00e-04,
            1.60e-04,
            1.21e-03,
            5.80e-04,
            3.60e-04,
        ],
        5,
    )


def test_mean_views_prior(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AAPL >= prior(AAPL) * 1.2",
            "AMD == prior(AMD)",
            "BAC <= prior(BAC)*0.8",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    prior_mean = np.mean(x, axis=0)
    mean = sm.mean(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.000175, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    assert mean[0] >= prior_mean[0] * 1.2
    np.testing.assert_almost_equal(mean[1], prior_mean[1], 5)
    assert mean[2] <= prior_mean[2] * 0.8
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 8)


def test_value_at_risk_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        value_at_risk_views=[
            "AAPL == 0.03",
            "AMD == 0.06",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    value_at_risk = sm.value_at_risk(x, sample_weight=sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.002375, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(value_at_risk[0], 0.03, 3)
    np.testing.assert_almost_equal(value_at_risk[1], 0.06, 3)


def test_value_at_risk_views_prior(X, solver):
    model = EntropyPooling(
        solver=solver,
        value_at_risk_views=[
            "AAPL >= prior(AAPL) * 1.2",
            "AMD == prior(AMD)",
            "BAC <= prior(BAC)*0.8",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    value_at_risk_prior = sm.value_at_risk(x)
    value_at_risk = sm.value_at_risk(x, sample_weight=sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.014598, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(value_at_risk[0], value_at_risk_prior[0] * 1.2, 3)
    np.testing.assert_almost_equal(value_at_risk[1], value_at_risk_prior[1], 3)
    np.testing.assert_almost_equal(value_at_risk[2], value_at_risk_prior[2] * 0.8, 3)


def test_cvar_views(X, solver):
    model = EntropyPooling(solver=solver, cvar_views=["AAPL == 0.05"])
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    cvar = sm.cvar(x, sample_weight=sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.002751, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(cvar[0], 0.05, 5)


def test_cvar_prior_views(X, solver):
    model = EntropyPooling(solver=solver, cvar_views=["AAPL == prior(AAPL)*1.3"])
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    cvar_prior = sm.cvar(x)
    cvar = sm.cvar(x, sample_weight=sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.006599, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(cvar[0], cvar_prior[0] * 1.3, 5)


def test_cvar_multiple_views(X, solver):
    model = EntropyPooling(solver=solver, cvar_views=["AAPL == 0.05", "XOM == 0.06"])
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    cvar = sm.cvar(x, sample_weight=sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.013235, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(cvar[0], 0.05, 4)
    np.testing.assert_almost_equal(cvar[19], 0.06, 4)


def test_mean_var_cvar_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AAPL >= 0.002",
            "AMD == 0.003",
            "BAC <= 0.0001",
            "UNH >= 0.0",
            "WMT <= 0.5",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
        value_at_risk_views=["AAPL == 0.03", "AMD == 0.06"],
        cvar_views=["AAPL == 0.05"],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    np.testing.assert_almost_equal(
        model.relative_entropy_, sw @ np.log(sw / (np.ones(len(sw)) / len(sw)))
    )
    x = np.array(X)
    mean = sm.mean(x, sample_weight=sw)
    value_at_risk = sm.value_at_risk(x, sample_weight=sw)
    cvar = sm.cvar(x, sample_weight=sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.011436, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(mean[0], 0.002, 5)
    np.testing.assert_almost_equal(mean[1], 0.003, 5)
    np.testing.assert_almost_equal(mean[2], 0.0001, 5)
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 5)
    np.testing.assert_almost_equal(value_at_risk[0], 0.03, 4)
    np.testing.assert_almost_equal(value_at_risk[1], 0.06, 3)
    np.testing.assert_almost_equal(cvar[0], 0.05, 5)


def test_variance_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        variance_views=[
            "AAPL >= 0.0004",
            "AMD == 0.002",
            "BAC <= 0.0003",
            "UNH >= 0.0",
            "WMT <= 0.5",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    variance = sm.variance(x, sample_weight=sw, biased=True)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.026435, 3)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(variance[0], 0.0004, 5)
    np.testing.assert_almost_equal(variance[1], 0.002, 5)
    np.testing.assert_almost_equal(variance[2], 0.0003, 5)
    np.testing.assert_almost_equal(
        1.5 * variance[3] - (2 * variance[4] + 3 * variance[5]), 0, 5
    )


def test_variance_views_prior(X, solver):
    model = EntropyPooling(
        solver=solver,
        variance_views=[
            "AAPL >= prior(AAPL)*2",
            "AMD == 0.002",
            "BAC <= 0.5 * prior(BAC)",
        ],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    variance_prior = sm.variance(x)
    variance = sm.variance(x, sample_weight=sw, biased=True)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.116059, 4)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(variance[0], variance_prior[0] * 2, 5)
    np.testing.assert_almost_equal(variance[1], 0.002, 5)
    np.testing.assert_almost_equal(variance[2], variance_prior[2] * 0.5, 5)


def test_mean_variance_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AAPL >= 0.002",
            "AMD == 0.003",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
        variance_views=[
            "AAPL >= 0.0004",
            "AMD == 0.002",
            "BAC <= 0.0003",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    mean = sm.mean(x, sample_weight=sw)
    variance = sm.variance(x, sample_weight=sw, biased=True)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.027149, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(mean[0], 0.002, 5)
    np.testing.assert_almost_equal(mean[1], 0.003, 5)
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 5)
    np.testing.assert_almost_equal(variance[0], 0.0004, 5)
    np.testing.assert_almost_equal(variance[1], 0.002, 5)
    np.testing.assert_almost_equal(variance[2], 0.0003, 5)
    np.testing.assert_almost_equal(
        1.5 * variance[3] - (2 * variance[4] + 3 * variance[5]), 0, 5
    )


def test_mean_cvar_variance_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AMD == 0.003",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
        variance_views=[
            "AAPL >= 0.0004",
            "AMD == 0.002",
            "BAC <= 0.0003",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
        cvar_views=["AAPL == 0.10"],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight

    np.testing.assert_almost_equal(model.relative_entropy_, 0.208164, 4)
    x = np.array(X)
    mean = sm.mean(x, sample_weight=sw)
    variance = sm.variance(x, sample_weight=sw, biased=True)
    cvar = sm.cvar(x, sample_weight=sw)

    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(mean[0], -0.002055, 5)
    np.testing.assert_almost_equal(mean[1], 0.003, 5)
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 5)
    np.testing.assert_almost_equal(variance[0], 0.00077226, 5)
    np.testing.assert_almost_equal(variance[1], 0.002, 5)
    np.testing.assert_almost_equal(variance[2], 0.0003, 5)
    np.testing.assert_almost_equal(
        1.5 * variance[3] - (2 * variance[4] + 3 * variance[5]), 0, 5
    )
    np.testing.assert_almost_equal(cvar[0], 0.10, 5)


def test_cvar_variance_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        variance_views=[
            "AAPL >= 0.0005",
            "AMD == 0.003",
        ],
        cvar_views=["AAPL == 0.10"],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    np.testing.assert_almost_equal(model.relative_entropy_, 0.09407, 5)
    x = np.array(X)
    mean = sm.mean(x, sample_weight=sw)
    variance = sm.variance(x, sample_weight=sw, biased=True)
    cvar = sm.cvar(x, sample_weight=sw)

    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)

    np.testing.assert_almost_equal(mean[0], -0.0033, 4)
    np.testing.assert_almost_equal(mean[1], -0.0021, 4)
    assert variance[0] >= 0.0005
    np.testing.assert_almost_equal(variance[1], 0.003, 4)
    np.testing.assert_almost_equal(cvar[0], 0.10, 5)
    np.testing.assert_almost_equal(cvar[1], 0.121, 3)


def test_correlation_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        correlation_views=[
            "(AAPL,AMD) == 0.5",
            "(AMD, BAC) >= 0.6",
            "(WMT, XOM) <= 0.10",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    corr = sm.correlation(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.071610, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(corr[0, 1], 0.5, 5)
    np.testing.assert_almost_equal(corr[1, 2], 0.6, 5)
    np.testing.assert_almost_equal(corr[18, 19], 0.1, 5)


def test_correlation_views_prior(X, solver):
    model = EntropyPooling(
        solver=solver,
        correlation_views=[
            "(AAPL,AMD) == prior(AAPL,AMD)*1.5",
            "(AMD, BAC) >= 0.6",
            "(WMT, XOM) <= 0.3 * prior(WMT, XOM)",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    corr_prior = np.corrcoef(x.T)
    corr = sm.correlation(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.075878, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(corr[0, 1], corr_prior[0, 1] * 1.5, 5)
    np.testing.assert_almost_equal(corr[1, 2], 0.6, 5)
    np.testing.assert_almost_equal(corr[18, 19], corr_prior[18, 19] * 0.3, 5)


def test_mean_correlation_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AMD == 0.003",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
        correlation_views=[
            "(AAPL,AMD) == 0.5",
            "(AMD, BAC) >= 0.6",
            "(WMT, XOM) <= 0.10",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    mean = sm.mean(x, sample_weight=sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.076290976, 5)
    corr = sm.correlation(x, sw)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(mean[1], 0.003, 5)
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 5)
    np.testing.assert_almost_equal(corr[0, 1], 0.5, 5)
    np.testing.assert_almost_equal(corr[1, 2], 0.6, 5)
    np.testing.assert_almost_equal(corr[18, 19], 0.1, 5)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_mean_variance_correlation_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AMD == 0.003",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
        variance_views=[
            "AAPL == 0.0005",
            "AMD == 0.003",
        ],
        correlation_views=[
            "(AAPL,AMD) == 0.5",
            "(AMD, BAC) >= 0.6",
            "(WMT, XOM) <= 0.10",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    mean = sm.mean(x, sw)
    variance = sm.variance(x, sample_weight=sw, biased=True)
    corr = sm.correlation(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.185747, 3)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(mean[1], 0.003, 5)
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 7)
    np.testing.assert_almost_equal(variance[0], 0.0005)
    np.testing.assert_almost_equal(variance[1], 0.003, 5)
    np.testing.assert_almost_equal(corr[0, 1], 0.5, 4)
    np.testing.assert_almost_equal(corr[1, 2], 0.6, 4)
    np.testing.assert_almost_equal(corr[18, 19], 0.1, 3)


def test_cvar_correlation_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        correlation_views=[
            "(AAPL,AMD) == 0.5",
            "(AMD, BAC) >= 0.6",
            "(WMT, XOM) <= 0.10",
        ],
        cvar_views=["AAPL == 0.10"],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    np.testing.assert_almost_equal(model.relative_entropy_, 0.220522, 4)
    x = np.array(X)
    mean = sm.mean(x, sw)
    corr = sm.correlation(x, sw)

    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)

    np.testing.assert_almost_equal(mean[0], -0.0033, 4)
    np.testing.assert_almost_equal(mean[1], -0.0021, 4)
    np.testing.assert_almost_equal(corr[0, 1], 0.5, 5)
    np.testing.assert_almost_equal(corr[1, 2], 0.6, 5)
    np.testing.assert_almost_equal(corr[18, 19], 0.1, 5)
    np.testing.assert_almost_equal(sm.cvar(returns=x[:, 0], sample_weight=sw), 0.10, 5)
    np.testing.assert_almost_equal(sm.cvar(returns=x[:, 1], sample_weight=sw), 0.109, 3)


def test_skew_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        skew_views=[
            "AAPL == -0.1",
            "BAC >= 2.0",
            "WMT <= -0.3",
            "2.5 * BBY == 1.4*CVX + 4*GE",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    skew = sm.skew(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.010719, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1)
    np.testing.assert_almost_equal(skew[0], -0.1, 5)
    np.testing.assert_almost_equal(skew[2], 2.0, 4)
    np.testing.assert_almost_equal(skew[18], -0.3, 5)
    np.testing.assert_almost_equal(2.5 * skew[3] - (1.4 * skew[4] + 4 * skew[5]), 0, 4)


def test_skew_views_prior(X, solver):
    model = EntropyPooling(
        solver=solver,
        skew_views=[
            "AAPL == prior(AAPL)*1.5",
            "BAC >= 2.0",
            "WMT <= prior(WMT)*0.3",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    skew_prior = sm.skew(x)
    skew = sm.skew(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.007809, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(skew[0], skew_prior[0] * 1.5, 3)
    np.testing.assert_almost_equal(skew[2], 2.0, 3)
    np.testing.assert_almost_equal(skew[18], skew_prior[18] * 0.3, 3)


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_mean_variance_correlation_skew_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AMD == 0.003",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
        variance_views=[
            "AAPL == 0.0005",
            "AMD == 0.003",
        ],
        correlation_views=[
            "(AAPL,AMD) == 0.5",
            "(AMD, BAC) >= 0.6",
            "(WMT, XOM) <= 0.10",
        ],
        skew_views=[
            "AAPL == -0.1",
            "BAC >= 2.0",
            "WMT <= -0.3",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    mean = sm.mean(x, sw)
    variance = sm.variance(x, sample_weight=sw, biased=True)
    corr = sm.correlation(x, sw)
    skew = sm.skew(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.215806, 3)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(mean[1], 0.003, 5)
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 5)
    np.testing.assert_almost_equal(variance[0], 0.0005)
    np.testing.assert_almost_equal(variance[1], 0.003, 5)
    np.testing.assert_almost_equal(corr[0, 1], 0.5, 4)
    np.testing.assert_almost_equal(corr[1, 2], 0.6, 4)
    np.testing.assert_almost_equal(corr[18, 19], 0.1, 3)
    np.testing.assert_almost_equal(skew[0], -0.1, 4)
    np.testing.assert_almost_equal(skew[2], 2.0, 3)
    assert skew[18] < -0.3


def test_cvar_skew_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        skew_views=[
            "AAPL == -0.1",
            "BAC >= 2.0",
            "WMT <= -0.3",
        ],
        cvar_views=["AAPL == 0.07"],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    skew = sm.skew(x, sw)
    cvar = sm.cvar(x, sample_weight=sw)
    # np.testing.assert_almost_equal(model.relative_entropy_, 0.093231, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(skew[0], -0.1, 2)
    np.testing.assert_almost_equal(skew[2], 2.0, 2)
    np.testing.assert_almost_equal(skew[18], -0.3, 2)
    np.testing.assert_almost_equal(cvar[0], 0.07, 4)


def test_kurtosis_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        kurtosis_views=[
            "AAPL == 9.0",
            "BAC >= 25",
            "WMT <= 10",
            "2.5 * BBY == 1.4*CVX + 4*GE",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    kurtosis = sm.kurtosis(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.026512, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1)
    np.testing.assert_almost_equal(kurtosis[0], 9.0, 4)
    np.testing.assert_almost_equal(kurtosis[2], 25, 4)
    np.testing.assert_almost_equal(kurtosis[18], 10, 3)
    np.testing.assert_almost_equal(
        2.5 * kurtosis[3] - (1.4 * kurtosis[4] + 4 * kurtosis[5]), 0, 2
    )


def test_kurtosis_views_prior(X, solver):
    model = EntropyPooling(
        solver=solver,
        kurtosis_views=[
            "AAPL == prior(AAPL)*1.5",
            "BAC >= 25",
            "WMT <= prior(WMT)*0.3",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    kurtosis_prior = sm.kurtosis(x)
    kurtosis = sm.kurtosis(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.046041, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(kurtosis[0], kurtosis_prior[0] * 1.5, 4)
    np.testing.assert_almost_equal(kurtosis[2], 25.0, 3)
    np.testing.assert_almost_equal(kurtosis[18], kurtosis_prior[18] * 0.3, 4)


def test_mean_variance_correlation_kurtosis_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "AMD == 0.003",
            "1.5 * BBY == 2*CVX + 3*GE",
        ],
        variance_views=[
            "AAPL == 0.0005",
            "AMD == 0.003",
        ],
        correlation_views=[
            "(AAPL,AMD) == 0.5",
            "(AMD, BAC) >= 0.6",
            "(WMT, XOM) <= 0.10",
        ],
        kurtosis_views=[
            "AAPL == 9.0",
            "BAC >= 25",
            "WMT <= 10",
        ],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    mean = sm.mean(x, sw)
    variance = sm.variance(x, sample_weight=sw, biased=True)
    corr = sm.correlation(x, sw)
    kurtosis = sm.kurtosis(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.274519, 2)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(mean[1], 0.003, 5)
    np.testing.assert_almost_equal(1.5 * mean[3] - (2 * mean[4] + 3 * mean[5]), 0, 5)
    np.testing.assert_almost_equal(variance[0], 0.0005, 5)
    np.testing.assert_almost_equal(variance[1], 0.003, 5)
    np.testing.assert_almost_equal(corr[0, 1], 0.5, 4)
    np.testing.assert_almost_equal(corr[1, 2], 0.6, 2)
    np.testing.assert_almost_equal(corr[18, 19], 0.1, 3)
    np.testing.assert_almost_equal(kurtosis[0], 9.0, 3)
    np.testing.assert_almost_equal(kurtosis[2], 25, 2)
    np.testing.assert_almost_equal(kurtosis[18], 10, 2)


def test_cvar_kurtosis_views(X, solver):
    model = EntropyPooling(
        solver=solver,
        kurtosis_views=[
            "AAPL == 9.0",
            "BAC >= 25",
            "WMT <= 10",
        ],
        cvar_views=["AAPL == 0.05"],
    )
    model.fit(X)

    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    kurtosis = sm.kurtosis(x, sw)
    cvar = sm.cvar(x, sample_weight=sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.027908, 3)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(kurtosis[0], 9.0, 1)
    np.testing.assert_almost_equal(kurtosis[2], 25, 1)
    np.testing.assert_almost_equal(kurtosis[18], 10, 1)
    np.testing.assert_almost_equal(cvar[0], 0.05, 5)


def test_view_on_groups(X, groups_dict, solver):
    model = EntropyPooling(
        solver=solver,
        mean_views=[
            "Equity >= 0.004",
            "Europe == 0.005",
            "Bond <= 0.5 * Equity",
        ],
        groups=groups_dict,
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    x = np.array(X)
    mean = sm.mean(x, sw)
    np.testing.assert_almost_equal(model.relative_entropy_, 0.005711, 5)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    assert mean[:3].sum() >= 0.004
    np.testing.assert_almost_equal(mean[2:10].sum(), 0.005)
    np.testing.assert_almost_equal(mean[6:].sum(), 0.5 * mean[:3].sum())


def test_extract_prior_assets(views):
    prior_assets = _extract_prior_assets(
        views=views,
        assets=[
            "AAPL",
            "ABL",
            "XXL",
            "A_1",
            "A B",
            "A 2",
            "A3",
            "BA/ LN Equity",
            "Dummy",
        ],
    )
    assert prior_assets == {"XXL", "A_1", "ABL", "AAPL", "A 2", "A B", "BA/ LN Equity"}


def test_replace_prior_views(views):
    res = _replace_prior_views(
        views=views,
        prior_values={
            "XXL": 2,
            "A_1": 1.5,
            "ABL": 152.5515,
            "AAPL": -5,
            "A 2": 0,
            "A B": 0.8,
            "BA/ LN Equity": 50,
        },
    )

    assert res == [
        "AAPL>= -16.25",
        "AAPL >=495.792375",
        "AAPL >= 7.0",
        "AAPL >=5.25",
        "AAPL ==2.8000000000000003",
        "AAPL <=   0.0 ",
        "AAPL >=   A3 ",
        "BA/ LN Equity ==   50.0",
    ]


def test_replace_prior_views_error(views):
    with pytest.raises(ValueError, match="Unresolved 'prior' expression"):
        _ = _replace_prior_views(
            views=views,
            prior_values={
                "XXL": 2,
            },
        )


@pytest.mark.parametrize(
    "view,expected",
    [
        (
            "(AAPL,A_1) >= 0.8",
            {
                "assets": ("AAPL", "A_1"),
                "operator": operator.ge,
                "expression": {"constant": 0.8},
            },
        ),
        (
            "(A 2,BA/ LN Equity) < 0.3",
            {
                "assets": ("A 2", "BA/ LN Equity"),
                "operator": operator.lt,
                "expression": {"constant": 0.3},
            },
        ),
        (
            "(AAPL,A 2) == 0.5",
            {
                "assets": ("AAPL", "A 2"),
                "operator": operator.eq,
                "expression": {"constant": 0.5},
            },
        ),
        (
            "(A_1, ABL) >= prior(A 2,BA/ LN Equity) * 2.5 + 1.0",
            {
                "assets": ("A_1", "ABL"),
                "operator": operator.ge,
                "expression": {
                    "prior_assets": ("A 2", "BA/ LN Equity"),
                    "multiplier": 2.5,
                    "constant": 1.0,
                },
            },
        ),
        (
            "(AAPL, BA/ LN Equity) >= 0.5 * prior(AAPL,A 2)",
            {
                "assets": ("AAPL", "BA/ LN Equity"),
                "operator": operator.ge,
                "expression": {
                    "prior_assets": ("AAPL", "A 2"),
                    "multiplier": 0.5,
                    "constant": 0.0,
                },
            },
        ),
        (
            "(a1,a2)>=2*prior(a1,a2)",
            {
                "assets": ("a1", "a2"),
                "operator": operator.ge,
                "expression": {
                    "prior_assets": ("a1", "a2"),
                    "multiplier": 2.0,
                    "constant": 0.0,
                },
            },
        ),
        (
            "(  a1  ,   a2  )   >=   2   *  prior(  a1   , a2  ) * 4 +   5",
            {
                "assets": ("a1", "a2"),
                "operator": operator.ge,
                "expression": {
                    "prior_assets": ("a1", "a2"),
                    "multiplier": 8.0,
                    "constant": 5.0,
                },
            },
        ),
    ],
)
def test_parse_correlation_view(view, expected):
    parsed_view = _parse_correlation_view(
        view,
        assets=[
            "AAPL",
            "ABL",
            "XXL",
            "A_1",
            "A B",
            "A 2",
            "A3",
            "BA/ LN Equity",
            "a1",
            "a2",
            "Dummy",
        ],
    )
    assert parsed_view == expected


def test_parse_correlation_view_error():
    with pytest.raises(ValueError, match="Invalid correlation view format"):
        _parse_correlation_view("(AAPL,A_1) >= 0.8", assets=["AAPL"])

    with pytest.raises(ValueError, match="Could not convert constant 'a'"):
        _parse_correlation_view("(AAPL,A_1) >= a", assets=["AAPL", "A_1"])

    with pytest.raises(ValueError, match="Invalid post-multiplier 'a'"):
        _parse_correlation_view(
            "(A_1, ABL) >=  prior(A_1, ABL) * a ", assets=["A_1", "ABL"]
        )


def test_synthetic_data_prior(X, solver):
    model = EntropyPooling(
        solver=solver,
        prior_estimator=SyntheticData(
            n_samples=10000,
            distribution_estimator=VineCopula(
                log_transform=True,
                marginal_candidates=[Gaussian()],
                copula_candidates=[GaussianCopula()],
            ),
        ),
        mean_views=["AMD == 0.003"],
    )
    model.fit(X)
    sw = model.return_distribution_.sample_weight
    x = model.return_distribution_.returns
    mean = sm.mean(x, sw)
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(mean[1], 0.003)


def test_factor_entropy_pooling(X, y, solver):
    ref = FactorModel()
    ref.fit(X, y)

    model = FactorModel(
        factor_prior_estimator=EntropyPooling(
            solver=solver,
            mean_views=["QUAL == 0.0005"],
        ),
    )
    model.fit(X, y)

    sw = model.factor_prior_estimator_.return_distribution_.sample_weight
    assert np.all(sw >= 0)
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(sm.mean(y["QUAL"], sample_weight=sw), 0.0005)
    np.testing.assert_almost_equal(model.return_distribution_.sample_weight, sw)

    assert model.return_distribution_.mu.sum() > ref.return_distribution_.mu.sum()


def test_factor_synthetic_data_entropy_pooling(X, y, solver):
    factor_synth = SyntheticData(
        n_samples=10000,
        distribution_estimator=VineCopula(
            log_transform=True,
            marginal_candidates=[Gaussian()],
            copula_candidates=[GaussianCopula()],
        ),
    )
    factor_view = EntropyPooling(
        solver=solver,
        prior_estimator=factor_synth,
        mean_views=["QUAL == 0.0005"],
    )
    model = FactorModel(factor_prior_estimator=factor_view)
    model.fit(X, y)

    sw = model.factor_prior_estimator_.return_distribution_.sample_weight
    ret = model.factor_prior_estimator_.return_distribution_.returns
    assert np.all(sw >= 0)
    assert len(sw) == 10000
    np.testing.assert_almost_equal(np.sum(sw), 1, 8)
    np.testing.assert_almost_equal(sm.mean(ret, sample_weight=sw)[1], 0.0005)
    np.testing.assert_almost_equal(model.return_distribution_.sample_weight, sw)


def test_entropy_pooling_with_array(X, solver):
    model = EntropyPooling(solver=solver, mean_views=["x2 == 0.0005"])
    model.fit(np.array(X))
    sw = model.return_distribution_.sample_weight
    mean = sm.mean(np.array(X), sw)
    np.testing.assert_almost_equal(mean[2], 0.0005, 5)


def test_view_error(X, solver):
    model = EntropyPooling(solver=solver, mean_views="AAPL == 0.0005")
    with pytest.raises(ValueError, match="mean_views must be a list of strings"):
        model.fit(X)


def test_value_at_risk_view_error(X, solver):
    model = EntropyPooling(solver=solver, value_at_risk_views=["AAPL == 2 * AMD"])
    with pytest.raises(ValueError, match="You cannot mix multiple assets"):
        model.fit(X)

    model = EntropyPooling(solver=solver, value_at_risk_views=["AAPL >= -0.5"])
    with pytest.raises(
        ValueError, match="Value-at-Risk views must be strictly positive"
    ):
        model.fit(X)

    model = EntropyPooling(solver=solver, value_at_risk_views=["AAPL == -0.5"])
    with pytest.raises(
        ValueError, match="Value-at-Risk views must be strictly positive"
    ):
        model.fit(X)

    model = EntropyPooling(solver=solver, value_at_risk_views=["AAPL <= -0.5"])
    with pytest.raises(
        ValueError, match="Value-at-Risk views must be strictly positive"
    ):
        model.fit(X)

    model = EntropyPooling(solver=solver, value_at_risk_views=["AAPL >= 0.9"])
    with pytest.raises(
        ValueError, match="The Value-at-Risk view of -90.000% is excessively extreme"
    ):
        model.fit(X)


def test_correlation_view_error(X, solver):
    model = EntropyPooling(solver=solver, correlation_views=["(AAPL,AMD) == 1.5"])
    with pytest.raises(ValueError, match="Correlation views must be between 0 and 1."):
        model.fit(X)


def test_cvar_view_error(X, solver):
    model = EntropyPooling(solver=solver, cvar_views=["AAPL >= 0.05"])
    with pytest.raises(ValueError, match="CVaR view inequalities are not supported"):
        model.fit(X)

    model = EntropyPooling(solver=solver, cvar_views=["AAPL == 2 * AMD"])
    with pytest.raises(
        ValueError, match="You cannot mix multiple assets in a single CVaR view"
    ):
        model.fit(X)

    model = EntropyPooling(solver=solver, cvar_views=["AAPL == -0.05"])
    with pytest.raises(ValueError, match="CVaR view must be strictly positive"):
        model.fit(X)

    model = EntropyPooling(solver=solver, cvar_views=["AAPL == 0.9"])
    with pytest.raises(
        ValueError,
        match="The CVaR views of 90.00% is excessively extreme and cannot exceed 12.87% which is the worst realization",
    ):
        model.fit(X)


def test_prior_error(X, solver):
    model = EntropyPooling(solver=solver, mean_views=["AAPL >= prior(dummy)"])
    with pytest.raises(GroupNotFoundError, match="Wrong pattern encountered"):
        model.fit(X)

    model = EntropyPooling(solver=solver, mean_views=["AAPL >= priorr(AAPL)"])
    with pytest.raises(GroupNotFoundError, match="Wrong pattern encountered"):
        model.fit(X)


def test_single_view(X):
    model = EntropyPooling(mean_views=["AMD <= BAC"])
    model.fit(X)


def test_complex_views(X):
    X = X[["AMD", "BAC", "GE", "JNJ", "JPM", "LLY", "PG"]]
    groups = {
        "AMD": ["Technology", "Growth"],
        "BAC": ["Financials", "Value"],
        "GE": ["Industrials", "Value"],
        "JNJ": ["Healthcare", "Defensive"],
        "JPM": ["Financials", "Income"],
        "LLY": ["Healthcare", "Defensive"],
        "PG": ["Consumer", "Defensive"],
    }

    entropy_pooling = EntropyPooling(
        mean_views=[
            "JPM == -0.002",
            "PG >= LLY",
            "BAC >= prior(BAC) * 1.2",
            "Financials == 2 * Growth",
        ],
        variance_views=[
            "BAC == prior(BAC) * 4",
        ],
        correlation_views=[
            "(BAC,JPM) == 0.80",
            "(BAC,JNJ) <= prior(BAC,JNJ) * 0.5",
        ],
        skew_views=[
            "BAC == -0.05",
        ],
        cvar_views=[
            "GE == 0.07",
        ],
        cvar_beta=0.90,
        groups=groups,
    )
    entropy_pooling.fit(X)
    np.testing.assert_almost_equal(entropy_pooling.relative_entropy_, 0.6739174515, 5)
