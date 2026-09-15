from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pytest
from sklearn import config_context

from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.moments import ImpliedCovariance
from skfolio.optimization.convex import (
    RiskBudgeting,
)
from skfolio.optimization.convex import _base as convex_base
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EmpiricalPrior, EntropyPooling, TimeSeriesFactorModel


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
def X(X):
    return X["2018-01-03":]


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


def test_risk_budgeting_contribution(X):
    n_assets = X.shape[1]
    model = RiskBudgeting()
    ptf = model.fit_predict(X)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.0420088,
                0.0321681,
                0.0368757,
                0.0397691,
                0.0397394,
                0.0382183,
                0.0464549,
                0.0676572,
                0.0406381,
                0.0639892,
                0.055542,
                0.0681354,
                0.0427415,
                0.0597638,
                0.0613296,
                0.067267,
                0.0314903,
                0.0468024,
                0.0749026,
                0.0445065,
            ]
        ),
        5,
    )  # Precision is 5 due to diff between linux and windows
    rc = ptf.contribution(measure=RiskMeasure.STANDARD_DEVIATION)
    np.testing.assert_almost_equal(
        rc, np.ones(n_assets) * ptf.standard_deviation / n_assets, 6
    )


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_risk_budgeting(X_small, risk_measure, precisions):
    precision = precisions[risk_measure]
    n_assets = X_small.shape[1]
    model = RiskBudgeting(risk_measure=risk_measure)
    ptf = model.fit_predict(X_small)
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


def test_risk_budgeting_factor_constraint(X, factors):
    factor_returns = factors.loc[X.index].rename(columns={"MTUM": "Momentum"})
    model = RiskBudgeting(
        prior_estimator=TimeSeriesFactorModel(),
        linear_constraints=["Momentum == 0"],
    )
    model.fit(X, factors=factor_returns)

    factor_model = model.prior_estimator_.return_distribution_.factor_model
    momentum_exposure = model.weights_ @ factor_model.loading_matrix[:, 0]

    np.testing.assert_almost_equal(momentum_exposure, 0.0)


def test_risk_budgeting_factor_family_constraint(X, factors):
    factor_returns = factors.loc[X.index].rename(columns={"MTUM": "Momentum"})
    factor_families = ["style", "quality", "style", "defensive", "style"]
    model = RiskBudgeting(
        prior_estimator=TimeSeriesFactorModel(factor_families=factor_families),
        linear_constraints=["style <= -0.05"],
    )
    model.fit(X, factors=factor_returns)

    factor_model = model.prior_estimator_.return_distribution_.factor_model
    style_mask = factor_model.factor_families == "style"
    family_exposure = (
        model.weights_ @ factor_model.loading_matrix[:, style_mask]
    ).sum()

    assert family_exposure <= -0.05


@pytest.mark.filterwarnings("ignore:The EVaR problem will be relaxed")
def test_risk_budgeting_transaction_costs_and_management_fees(X_small, risk_measure):
    model = RiskBudgeting(risk_measure=risk_measure)
    ptf = model.fit_predict(X_small)
    model = RiskBudgeting(
        risk_measure=risk_measure,
        min_return=ptf.mean * 1.05,
        transaction_costs=0.01 / 252,
        management_fees=0.01 / 252,
    )
    ptf2 = model.fit_predict(X_small)
    np.testing.assert_almost_equal(ptf2.mean, ptf.mean * 1.05)


def test_metadata_routing(X_small, implied_vol_small):
    with config_context(enable_metadata_routing=True):
        model = RiskBudgeting(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        with pytest.raises(ValueError):
            model.fit(X_small)

        model.fit(X_small, implied_vol=implied_vol_small)

    # noinspection PyUnresolvedReferences
    assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)


def test_risk_budgeting_non_investable_nan_assets(
    nan_investable_test_data, fixed_return_distribution_prior
):
    X, mu, covariance, investable_mask = nan_investable_test_data

    model = RiskBudgeting(
        risk_budget=np.array([1.0, 2.0, 99.0, 3.0]),
        prior_estimator=fixed_return_distribution_prior(mu=mu, covariance=covariance),
    )
    model.fit(X)

    return_distribution = model.prior_estimator_.return_distribution_
    assert return_distribution.n_assets == X.shape[1]
    assert return_distribution.n_investable_assets == np.count_nonzero(investable_mask)
    np.testing.assert_array_equal(model.investable_mask_, investable_mask)
    assert model.weights_.shape == (X.shape[1],)
    assert np.isfinite(model.weights_).all()
    np.testing.assert_allclose(model.weights_[~investable_mask], 0)
    np.testing.assert_allclose(model.weights_.sum(), 1)
    assert np.all(model.weights_[investable_mask] > 0)

    portfolio = model.predict(X)
    expected_returns = (
        X.iloc[:, investable_mask].to_numpy() @ model.weights_[investable_mask]
    )
    np.testing.assert_allclose(portfolio.returns, expected_returns)


@pytest.mark.parametrize("weights", [0.05, np.ones(20) / 20, list(np.ones(20) / 20)])
def test_risk_budgeting_equal_weight_constraints(X_small, weights):
    model = RiskBudgeting(min_weights=weights)
    model.fit(X_small)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) / 20)

    model = RiskBudgeting(max_weights=weights)
    model.fit(X_small)
    np.testing.assert_almost_equal(model.weights_, np.ones(20) / 20)


def test_risk_budgeting_weight_constraints_dict(X_small):
    model = RiskBudgeting(
        min_weights={"AAPL": 0.5, "UNH": 0.3}, max_weights={"BAC": 0.01}
    )
    model.fit(X_small)
    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                0.5,
                0.00903176,
                0.01,
                0.01046216,
                0.01166263,
                0.01081226,
                0.01087975,
                0.01202612,
                0.01099958,
                0.01162496,
                0.01148655,
                0.01207474,
                0.01011174,
                0.01159145,
                0.01162177,
                0.01161292,
                0.01056359,
                0.3,
                0.01177834,
                0.01165966,
            ]
        ),
        3,
    )


def test_risk_budgeting_negative_weight_constraints(X_small):
    model = RiskBudgeting(
        min_weights={"AAPL": -0.5, "UNH": 0.3}, max_weights={"BAC": 0.01}
    )

    with pytest.raises(
        ValueError,
        match=(
            r"RiskBudgeting must have non negative `min_weights` "
            r"constraint otherwise the problem becomes non-convex."
        ),
    ):
        model.fit(X_small)


@pytest.fixture(scope="module")
def X_full():
    """The full price history, unsliced.

    The ill-conditioning covered below needs the whole sample: the shorter windows the
    other fixtures use are not concentrated enough to reproduce it.
    """
    return prices_to_returns(load_sp500_dataset())


@pytest.fixture(scope="module")
def concentrated_prior():
    """A prior whose views concentrate `sample_weight` onto few scenarios.

    CVaR risk budgeting on this distribution is bounded and feasible, but so
    ill-conditioned that CLARABEL 0.11 freezes with a non-zero dual residual and
    terminates in `InsufficientProgress`. CLARABEL 0.10 solves it, so which solver
    gets there is a property of the installed version, not of skfolio. See issue #292.
    """
    return EntropyPooling(
        mean_views=["AMD >= BAC", "JPM <= prior(JPM) * 0.8"],
        cvar_views=["GE == 0.12"],
    )


def test_cvar_solves_on_concentrated_sample_weight(X_full, concentrated_prior):
    """The problem must be solved, whichever solver manages it.

    This used to raise `SolverError` on CLARABEL 0.11 with no way through. The
    assertion is deliberately on the outcome and not on the retry: on CLARABEL 0.10
    the primary solver succeeds and no fallback is needed.
    """
    model = RiskBudgeting(
        risk_measure=RiskMeasure.CVAR, prior_estimator=concentrated_prior
    )
    model.fit(X_full)

    assert model.solver_ in ("CLARABEL", "SCS")
    assert model.fallback_ is None
    np.testing.assert_almost_equal(model.weights_.sum(), 1.0)
    assert np.all(model.weights_ > 0)


def test_failed_solve_is_retried_with_the_fallback_solver(X_small, monkeypatch):
    """A failing primary solver is retried once, and `solver_` names the winner.

    The primary failure is injected rather than provoked with an ill-conditioned
    problem, so the test pins the retry itself and not a solver version's behaviour.
    """
    real_solve_once = convex_base._solve_once
    calls = []

    def failing_first_call(*args, **kwargs):
        calls.append(kwargs["solver"])
        if len(calls) == 1:
            raise cp.SolverError(f"Solver '{kwargs['solver']}' failed")
        return real_solve_once(*args, **kwargs)

    monkeypatch.setattr(convex_base, "_solve_once", failing_first_call)

    model = RiskBudgeting(risk_measure=RiskMeasure.CVAR)
    with pytest.warns(
        UserWarning,
        match=r"Solver 'CLARABEL' failed\. Retrying with the fallback solver 'SCS'\.",
    ):
        model.fit(X_small)

    assert calls == ["CLARABEL", "SCS"]
    assert model.solver_ == "SCS"
    np.testing.assert_almost_equal(model.weights_.sum(), 1.0)


def test_fallback_solver_does_not_inherit_the_primary_solver_params(
    X_small, monkeypatch
):
    """`solver_params` are tuned per solver, so the retry must not reuse them.

    `tol_gap_abs` is a CLARABEL key that SCS would reject.
    """
    real_solve_once = convex_base._solve_once
    seen = {}

    def failing_first_call(*args, **kwargs):
        seen[kwargs["solver"]] = kwargs["solver_params"]
        if len(seen) == 1:
            raise cp.SolverError(f"Solver '{kwargs['solver']}' failed")
        return real_solve_once(*args, **kwargs)

    monkeypatch.setattr(convex_base, "_solve_once", failing_first_call)

    model = RiskBudgeting(
        risk_measure=RiskMeasure.CVAR, solver_params={"tol_gap_abs": 1e-9}
    )
    with pytest.warns(UserWarning):
        model.fit(X_small)

    assert seen["CLARABEL"] == {"tol_gap_abs": 1e-9}
    assert seen["SCS"] == {}


def test_solver_reports_the_primary_solver_when_it_succeeds(X_small):
    """A problem CLARABEL solves is not retried and keeps its own solution."""
    model = RiskBudgeting(risk_measure=RiskMeasure.CVAR)
    model.fit(X_small)

    assert model.solver_ == "CLARABEL"


def test_no_fallback_when_the_solver_is_already_the_fallback(X_small, monkeypatch):
    """Choosing the fallback solver explicitly must not retry it a second time."""
    real_solve_once = convex_base._solve_once
    calls = []

    def counting(*args, **kwargs):
        calls.append(kwargs["solver"])
        return real_solve_once(*args, **kwargs)

    monkeypatch.setattr(convex_base, "_solve_once", counting)

    model = RiskBudgeting(risk_measure=RiskMeasure.CVAR, solver="SCS")
    model.fit(X_small)

    assert calls == ["SCS"]
    assert model.solver_ == "SCS"


def test_infeasible_problem_is_not_retried(X_small):
    """An infeasible problem carries a certificate, so the retry is skipped.

    `min_weights=1.0` on 20 assets cannot meet the unit budget. The error must name
    the primary solver, not the fallback, and no retry warning may be emitted.
    """
    model = RiskBudgeting(min_weights=1.0)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(cp.SolverError, match=r"Solver 'CLARABEL' failed"):
            model.fit(X_small)


def test_risk_budgeting_invalid_risk_measure_type(X):
    # `set_params` bypasses the enum conversion performed in `__init__`.
    model = RiskBudgeting().set_params(risk_measure="variance")
    with pytest.raises(TypeError, match="risk_measure must be of type `RiskMeasure`"):
        model.fit(X)


def test_risk_budgeting_non_default_solver():
    # Any solver other than CLARABEL falls through to empty params. Risk budgeting
    # needs an exponential cone, which SCIPY does not support, so the primary solve
    # fails and the fallback solver finishes the problem. The params are set before
    # the solve either way, so the branch is exercised.
    rng = np.random.default_rng(0)
    X = rng.normal(0.0005, 0.01, (60, 6))
    model = RiskBudgeting(solver="SCIPY")
    with pytest.warns(
        UserWarning,
        match=r"Solver 'SCIPY' failed\. Retrying with the fallback solver 'SCS'\.",
    ):
        model.fit(X)
    assert model._solver_params == {}
    assert model.solver_ == "SCS"
