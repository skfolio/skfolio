"""Tests for the `solver_path` solver sequence shared by the convex estimators."""

from __future__ import annotations

import warnings

import cvxpy as cp
import numpy as np
import pytest

from skfolio import RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, RiskBudgeting
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import EntropyPooling

# "SCIPY" ships with cvxpy, so it is available wherever skfolio is installed. It only
# handles linear programs, which makes it a dependable stand-in for a solver that fails
# on a given problem: risk budgeting needs an exponential cone and mean-variance a
# quadratic one.
FAILING_SOLVER = "SCIPY"


@pytest.fixture(scope="module")
def X():
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.01, (60, 6))


@pytest.fixture(scope="module")
def X_full():
    """The full price history, unsliced.

    The ill-conditioning of `concentrated_prior` needs the whole sample.
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


def test_default_is_a_single_attempt(X):
    """Without `solver_path`, nothing about the solve changes."""
    model = MeanRisk()
    model.fit(X)

    assert model._solver_path == [
        ("CLARABEL", {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9})
    ]
    assert model._solver_params == {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9}
    assert model.solver_ == "CLARABEL"


def test_a_failed_solve_advances_to_the_next_solver(X):
    """The sequence moves on, and `solver_` names the one that finished the problem."""
    model = RiskBudgeting(solver_path=[FAILING_SOLVER, "CLARABEL"])

    with pytest.warns(
        UserWarning,
        match=(
            r"Solver 'SCIPY' failed\. Trying the next solver of `solver_path`:"
            r" 'CLARABEL'\."
        ),
    ):
        model.fit(X)

    assert model.solver_ == "CLARABEL"
    np.testing.assert_almost_equal(model.weights_.sum(), 1.0)


def test_the_first_success_wins(X):
    """A solver that succeeds ends the sequence, with no warning and no second solve."""
    model = RiskBudgeting(solver_path=["CLARABEL", FAILING_SOLVER])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.fit(X)

    assert model.solver_ == "CLARABEL"


def test_a_string_entry_takes_the_default_parameters_of_its_solver(X):
    """`"CLARABEL"` in a path means the same thing as `solver="CLARABEL"`."""
    model = MeanRisk(solver_path=["CLARABEL"])
    model.fit(X)

    assert model._solver_path == [
        ("CLARABEL", {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9})
    ]


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_a_string_entry_of_an_untuned_solver_falls_through_to_empty_params(X):
    """Solvers skfolio does not tune keep the CVXPY defaults."""
    model = MeanRisk(risk_measure=RiskMeasure.CVAR, solver_path=[FAILING_SOLVER])
    model.fit(X)

    assert model._solver_path == [(FAILING_SOLVER, {})]


def test_a_tuple_entry_carries_its_own_parameters(X):
    """Explicit parameters replace the tuned defaults, as `solver_params` does."""
    model = MeanRisk(solver_path=[("CLARABEL", {"tol_gap_abs": 1e-7})])
    model.fit(X)

    assert model._solver_path == [("CLARABEL", {"tol_gap_abs": 1e-7})]
    assert model._solver_params == {"tol_gap_abs": 1e-7}


def test_each_entry_keeps_its_own_parameters(X):
    """Parameters are per solver: keys of one are not portable to another."""
    model = RiskBudgeting(
        solver_path=[
            (FAILING_SOLVER, {}),
            ("CLARABEL", {"tol_gap_abs": 1e-7, "tol_gap_rel": 1e-7}),
        ]
    )

    with pytest.warns(UserWarning, match="Trying the next solver"):
        model.fit(X)

    assert model._solver_path[1][1] == {"tol_gap_abs": 1e-7, "tol_gap_rel": 1e-7}
    assert model.solver_ == "CLARABEL"


def test_an_exhausted_path_reports_every_attempt(X):
    """When no solver succeeds, the error names the whole path."""
    model = RiskBudgeting(solver_path=[FAILING_SOLVER, "SCIP"])

    with (
        pytest.warns(UserWarning, match="Trying the next solver"),
        pytest.raises(
            cp.SolverError,
            match=r"All solvers of `solver_path` failed \('SCIPY', 'SCIP'\)",
        ),
    ):
        model.fit(X)


def test_a_single_solver_path_raises_the_plain_error(X):
    """One entry must behave exactly like `solver`: same message, no warning."""
    model = RiskBudgeting(solver_path=[FAILING_SOLVER])

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(cp.SolverError, match=r"Solver 'SCIPY' failed"):
            model.fit(X)


def test_an_infeasible_problem_stops_the_sequence(X):
    """A certificate is a property of the problem, so the next solver is not tried.

    `min_weights=0.5` with `max_weights=0.4` cannot be satisfied. The error must name
    the solver that proved it, and no advance warning may be emitted.
    """
    model = MeanRisk(
        min_weights=0.5, max_weights=0.4, solver_path=["CLARABEL", FAILING_SOLVER]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(cp.SolverError, match=r"Solver 'CLARABEL' failed"):
            model.fit(X)


def test_the_path_applies_to_each_optimization_of_a_sweep(X):
    """On an efficient frontier, one hard point does not decide the whole sweep."""
    model = MeanRisk(efficient_frontier_size=5, solver_path=["CLARABEL"])
    model.fit(X)

    assert model.solver_ == ["CLARABEL"] * 5
    assert model.weights_.shape == (5, 6)


def test_a_mixed_integer_problem_skips_the_solvers_that_cannot_express_it(X):
    """Cardinality needs integer variables, which CLARABEL cannot encode."""
    model = MeanRisk(cardinality=3, solver_path=["CLARABEL", "SCIP"])

    with pytest.warns(
        UserWarning,
        match=(
            r"The problem is mixed-integer and 'CLARABEL' cannot express integer"
            r" variables, so it is skipped in `solver_path`\."
        ),
    ):
        model.fit(X)

    assert model.solver_ == "SCIP"
    assert np.count_nonzero(~np.isclose(model.weights_, 0)) <= 3


def test_a_mixed_integer_problem_without_a_capable_solver_raises(X):
    """With nothing left in the path, the existing error stands."""
    model = MeanRisk(cardinality=3, solver_path=["CLARABEL"])

    with pytest.raises(ValueError, match="require a mixed-integer solver"):
        model.fit(X)


def test_an_uninstalled_solver_in_the_path_raises(X):
    model = MeanRisk(solver_path=["CLARABEL", "NOT_A_SOLVER"])

    with pytest.raises(ValueError, match="The solver NOT_A_SOLVER is not installed"):
        model.fit(X)


@pytest.mark.parametrize(
    "params,expected",
    [
        (
            dict(solver="SCS"),
            r"`solver_path` cannot be used together with solver,",
        ),
        (
            dict(solver_params={"tol_gap_abs": 1e-9}),
            r"`solver_path` cannot be used together with solver_params,",
        ),
        (
            dict(solver="SCS", solver_params={"tol_gap_abs": 1e-9}),
            r"`solver_path` cannot be used together with solver and solver_params,",
        ),
    ],
)
def test_solver_path_conflicts_with_solver_and_solver_params(X, params, expected):
    """A second way to name the first attempt would make it ambiguous."""
    model = MeanRisk(solver_path=["CLARABEL"], **params)

    with pytest.raises(ValueError, match=expected):
        model.fit(X)


@pytest.mark.parametrize("solver_path", [[], "CLARABEL", (), {}])
def test_an_empty_or_non_list_solver_path_raises(X, solver_path):
    model = MeanRisk(solver_path=solver_path)

    with pytest.raises(ValueError, match="must be a non-empty list of solver names"):
        model.fit(X)


@pytest.mark.parametrize("element", [123, None, ("CLARABEL", "SCS"), ("CLARABEL",)])
def test_a_malformed_solver_path_element_raises(X, element):
    model = MeanRisk(solver_path=[element])

    with pytest.raises(ValueError, match="must be a solver name or a"):
        model.fit(X)


@pytest.mark.skipif("SCS" not in cp.installed_solvers(), reason="SCS is not installed")
def test_cvar_on_a_concentrated_sample_weight_is_recovered_by_scs(
    X_full, concentrated_prior
):
    """The motivating instance of #292, driven through `solver_path`.

    The assertion is on the outcome and not on the advance: CLARABEL 0.10 solves this
    on its own, and only 0.11 needs SCS.
    """
    model = RiskBudgeting(
        risk_measure=RiskMeasure.CVAR,
        prior_estimator=concentrated_prior,
        solver_path=[
            "CLARABEL",
            ("SCS", {"eps_abs": 1e-6, "eps_rel": 1e-6, "max_iters": 100_000}),
        ],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model.fit(X_full)

    assert model.solver_ in ("CLARABEL", "SCS")
    np.testing.assert_almost_equal(model.weights_.sum(), 1.0)
    assert np.all(model.weights_ > 0)
