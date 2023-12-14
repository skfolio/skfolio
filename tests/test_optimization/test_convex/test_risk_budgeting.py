import datetime as dt

import numpy as np
import pytest

from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization.convex import RiskBudgeting
from skfolio.preprocessing import prices_to_returns


@pytest.fixture(scope="module")
def precisions():
    precisions = {e: 6 for e in RiskMeasure}
    precisions[RiskMeasure.EVAR] = 5
    precisions[RiskMeasure.MEAN_ABSOLUTE_DEVIATION] = 5
    precisions[RiskMeasure.CVAR] = 5
    precisions[RiskMeasure.AVERAGE_DRAWDOWN] = 4
    precisions[RiskMeasure.ULCER_INDEX] = 4
    precisions[RiskMeasure.WORST_REALIZATION] = 4
    precisions[RiskMeasure.EDAR] = 3
    precisions[RiskMeasure.MAX_DRAWDOWN] = 3
    precisions[RiskMeasure.CDAR] = 3
    return precisions


@pytest.fixture(scope="module")
def X():
    prices = load_sp500_dataset()
    prices = prices.loc[dt.date(2018, 1, 1) :]
    X = prices_to_returns(X=prices)
    return X


@pytest.fixture(scope="module")
def X2(X):
    X2 = X["2022":]
    return X2


@pytest.fixture(
    scope="module",
    params=[
        rm
        for rm in RiskMeasure
        if not rm.is_annualized
        and rm
        not in [
            RiskMeasure.GINI_MEAN_DIFFERENCE,  # Too slow without MOSEK
        ]
    ],
)
def risk_measure(request):
    return request.param


@pytest.fixture(scope="module")
def groups():
    groups = [
        ["Equity"] * 3 + ["Fund"] * 5 + ["Bond"] * 12,
        ["US"] * 2 + ["Europe"] * 8 + ["Japan"] * 10,
    ]
    return groups


@pytest.fixture(scope="module")
def linear_constraints():
    linear_constraints = [
        "Equity <= 0.5 * Bond",
        "US >= 0.1",
        "Europe >= 0.5 * Fund",
        "Japan <= 1",
    ]
    return linear_constraints


def test_risk_budgeting_contribution(X):
    n_assets = X.shape[1]
    model = RiskBudgeting()
    ptf = model.fit_predict(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.04201237,
                0.032168,
                0.03687683,
                0.03977124,
                0.03973885,
                0.03821871,
                0.04646103,
                0.06765447,
                0.04063898,
                0.06398907,
                0.0555448,
                0.06813484,
                0.0427472,
                0.0597675,
                0.06133195,
                0.06726793,
                0.0314916,
                0.04680726,
                0.07486711,
                0.04451026,
            ]
        ),
    )
    rc = ptf.contribution(measure=RiskMeasure.STANDARD_DEVIATION)
    np.testing.assert_almost_equal(
        rc, np.ones(n_assets) * ptf.standard_deviation / n_assets, 6
    )


def test_risk_budgeting(X2, risk_measure, precisions):
    precision = precisions[risk_measure]
    n_assets = X2.shape[1]
    model = RiskBudgeting(risk_measure=risk_measure)
    ptf = model.fit_predict(X2)
    np.testing.assert_almost_equal(ptf.mean, model.problem_values_["expected_return"])
    np.testing.assert_almost_equal(
        getattr(ptf, risk_measure.value), model.problem_values_["risk"], precision
    )
    rc = ptf.contribution(measure=risk_measure)
    np.testing.assert_almost_equal(rc, np.ones(n_assets) * np.mean(rc), precision - 1)


def test_risk_budgeting_groups(X, groups, linear_constraints):
    model = RiskBudgeting(groups=groups, linear_constraints=linear_constraints)

    ptf = model.fit_predict(X)
    w = model.weights_
    assert w[:3].sum() <= 0.5 * w[8:].sum()
    assert w[:2].sum() >= 0.099
    assert w[2:10].sum() >= 0.5 * w[3:8].sum()
    assert w[10:].sum() <= 1

    np.testing.assert_almost_equal(
        ptf.contribution(measure=RiskMeasure.STANDARD_DEVIATION),
        np.array(
            [
                0.00091078,
                0.00082199,
                0.00061383,
                0.00061259,
                0.00061238,
                0.00061318,
                0.00060932,
                0.00059876,
                0.00061196,
                0.00060047,
                0.00060463,
                0.00059849,
                0.00061136,
                0.00060261,
                0.00060179,
                0.00059898,
                0.00061652,
                0.00060902,
                0.00059545,
                0.00060999,
            ]
        ),
    )


def test_risk_budgeting_transaction_costs_and_management_fees(X2, risk_measure):
    model = RiskBudgeting(risk_measure=risk_measure)
    ptf = model.fit_predict(X2)
    model = RiskBudgeting(
        risk_measure=risk_measure,
        min_return=ptf.mean * 1.05,
        transaction_costs=0.01 / 255,
        management_fees=0.01 / 255,
    )
    ptf2 = model.fit_predict(X2)
    np.testing.assert_almost_equal(ptf2.mean, ptf.mean * 1.05)
