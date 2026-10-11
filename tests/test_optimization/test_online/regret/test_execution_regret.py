"""Independent execution and validation checks for historical regret."""

from copy import deepcopy

import numpy as np
import pytest

from skfolio.optimization.online import ExponentiatedGradient, RegretType, regret
from skfolio.optimization.online._benchmark import CRP
from skfolio.optimization.online._regret import _losses_from_weights


def test_regret_uses_execution_costs_and_preserves_fitted_state():
    X = np.array([[0.1, -0.05], [-0.02, 0.08], [0.04, -0.01]])
    model = ExponentiatedGradient(
        initial_weights=[0.8, 0.2],
        previous_weights=[0.5, 0.5],
        transaction_costs=0.01,
        management_fees=0.001,
        max_turnover=0.4,
    ).fit(X[:1])
    weights = model.weights_.copy()
    count = model._t
    portfolio = deepcopy(model).fit_predict(X)
    actual = regret(model, X, regret_type=RegretType.DYNAMIC_WORST_CASE)
    expected = np.cumsum(-np.log1p(portfolio.returns) + np.log1p(X.max(axis=1)))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(model.weights_, weights)
    assert model._t == count
    # First observation uses the explicit initial allocation, before learning.
    assert actual[0] > -np.log1p(X[0] @ np.array([0.8, 0.2])) + np.log1p(X[0].max())


def test_static_comparator_is_copied_and_cash_is_included():
    X = np.array([[0.1, -0.05], [-0.02, 0.08]])
    model = ExponentiatedGradient()
    comp = CRP(weights=[0.2, 0.3])
    actual = regret(model, X, comparator=comp)
    online_returns = deepcopy(model).fit_predict(X).returns
    expected = np.cumsum(-np.log1p(online_returns) + np.log1p(X @ [0.2, 0.3]))
    np.testing.assert_allclose(actual, expected)
    assert not hasattr(comp, "weights_")
    assert not hasattr(model, "weights_")


@pytest.mark.parametrize("X", [[], [[np.nan, 0]], [[np.inf, 0]], [[-1, 0]], [[-2, 0]]])
def test_invalid_returns_rejected(X):
    with pytest.raises(ValueError):
        regret(ExponentiatedGradient(), X)


@pytest.mark.parametrize(
    "config",
    [
        {"path_length": -1},
        {"path_length": np.nan},
        {"path_penalty": np.inf},
        {"path_length": 1, "path_penalty": 1},
        {},
        {"path_length": 1, "norm": "bad"},
    ],
)
def test_invalid_dynamic_configuration_rejected(config):
    with pytest.raises(ValueError):
        regret(
            ExponentiatedGradient(),
            [[0.01, 0.02]],
            regret_type=RegretType.DYNAMIC_UNIVERSAL,
            dynamic_config=config,
        )


@pytest.mark.parametrize(
    "options", [{"window": 1.5}, {"window": True}, {"average": "bad"}]
)
def test_invalid_aggregation_rejected(options):
    with pytest.raises(ValueError):
        regret(ExponentiatedGradient(), [[0.01, 0.02]], **options)


def test_invalid_solver_is_not_silently_replaced():
    import cvxpy as cp

    with pytest.raises(cp.error.SolverError):
        regret(
            ExponentiatedGradient(),
            [[0.01, 0.02]],
            regret_type=RegretType.DYNAMIC_UNIVERSAL,
            dynamic_config={"path_length": 1, "solver": "NOT_A_SOLVER"},
        )


def test_invalid_gross_wealth_is_not_clipped():
    with pytest.raises(ValueError, match="strictly positive"):
        _losses_from_weights(np.array([[0.5, 1.0]]), np.array([3.0, 0.0]))


def test_constant_comparator_matches_analytic_optimum():
    from skfolio.optimization.online._regret import _solve_bcrp_constant

    # Symmetric opposing returns have their unique optimum at equal weights.
    relatives = np.array([[1.2, 0.9], [0.9, 1.2]])
    weights = _solve_bcrp_constant(relatives)
    # Solver tolerances control the objective, not individual near-optimal weights.
    assert abs(np.log(relatives @ weights).sum() - 2 * np.log(1.05)) < 1e-9
    assert np.all(weights >= 0)
    assert weights.sum() == pytest.approx(1)
    # A strictly dominant asset puts the optimum on the simplex boundary.
    weights = _solve_bcrp_constant(np.array([[1.1, 1.0], [1.2, 1.01]]))
    np.testing.assert_allclose(weights, [1.0, 0.0], atol=1e-7)


def test_plot_preserves_supplied_values_and_figure():
    from skfolio.optimization.online import plot_regret_curve

    values = np.array([0.1, 0.3])
    fig = plot_regret_curve(values, average=False)
    result = plot_regret_curve(values / [1, 2], fig=fig, label="Running average")
    assert result is fig
    assert len(fig.data) == 2
    np.testing.assert_array_equal(fig.data[0].y, values)
    assert fig.data[0].name == "Regret"


@pytest.mark.parametrize("norm", ["l1", "l2"])
def test_dynamic_penalty_interpolates_between_hindsight_comparators(norm):
    X = np.array([[0.2, -0.1], [-0.1, 0.2], [0.15, -0.05], [-0.03, 0.08]])
    model = ExponentiatedGradient()
    unrestricted = regret(
        model,
        X,
        regret_type="dynamic_universal",
        dynamic_config={"path_penalty": 0.0, "norm": norm},
    )
    worst_case = regret(model, X, regret_type="dynamic_worst_case")
    np.testing.assert_allclose(unrestricted, worst_case, atol=1e-7)
    penalized = regret(
        model,
        X,
        regret_type="dynamic_universal",
        dynamic_config={"path_penalty": 100.0, "norm": norm},
    )
    static = regret(model, X, regret_type="static")
    # Compare optimized total log wealth; near-equal optimizers can have
    # different prefix curves at solver precision.
    np.testing.assert_allclose(penalized[-1], static[-1], atol=1e-7)


def test_explicit_legacy_comparator_matches_manual_prefix_losses():
    from skfolio.optimization.online._benchmark import BCRP

    X = np.array([[0.1, -0.05], [-0.02, 0.08], [0.04, -0.01]])
    comparator = BCRP(max_weights=0.7)
    model = ExponentiatedGradient()
    curve = regret(model, X, comparator=comparator, regret_type="dynamic_legacy")
    weights = deepcopy(comparator).fit_dynamic(X).all_weights_
    online = deepcopy(model).fit_predict(X).returns
    expected = np.cumsum(-np.log1p(online) + np.log1p(np.sum(X * weights, axis=1)))
    np.testing.assert_allclose(curve, expected, atol=1e-10)
    assert not hasattr(comparator, "all_weights_")


@pytest.mark.parametrize("regret_type", ["unknown", 123])
def test_invalid_regret_type_rejected(regret_type):
    with pytest.raises(ValueError):
        regret(ExponentiatedGradient(), [[0.1, 0.2]], regret_type=regret_type)


def test_weight_shape_validation_prevents_broadcasted_loss():
    with pytest.raises(ValueError, match="weights must have shape"):
        _losses_from_weights(np.ones((3, 2)), np.ones((1, 2)))


def test_bankrupt_online_execution_rejected():
    model = ExponentiatedGradient(management_fees=0.8)
    with pytest.raises(ValueError, match="strictly positive"):
        regret(model, [[-0.5, -0.6]])
