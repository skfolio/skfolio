import cvxpy as cp
import numpy as np
import pytest
import sklearn.model_selection as sks
from sklearn import config_context

from skfolio import (
    MultiPeriodPortfolio,
    Population,
    Portfolio,
    RatioMeasure,
    RiskMeasure,
)
from skfolio.model_selection import cross_val_predict
from skfolio.moments import EmpiricalMu, ImpliedCovariance
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.prior import BlackLitterman, EmpiricalPrior, FactorModel
from skfolio.uncertainty_set import (
    EmpiricalCovarianceUncertaintySet,
    EmpiricalMuUncertaintySet,
)
from skfolio.utils.equations import equations_to_matrix


@pytest.fixture(scope="module")
def precisions():
    precisions = {e: 7 for e in RiskMeasure}
    precisions[RiskMeasure.CDAR] = 6
    precisions[RiskMeasure.SEMI_DEVIATION] = 6
    precisions[RiskMeasure.MAX_DRAWDOWN] = 5
    precisions[RiskMeasure.AVERAGE_DRAWDOWN] = 5
    precisions[RiskMeasure.ULCER_INDEX] = 5
    precisions[RiskMeasure.EVAR] = 6
    precisions[RiskMeasure.GINI_MEAN_DIFFERENCE] = 6
    return precisions


@pytest.fixture(scope="module")
def precisions2(precisions):
    precisions[RiskMeasure.EVAR] = 3
    precisions[RiskMeasure.GINI_MEAN_DIFFERENCE] = 5
    return precisions


@pytest.fixture(scope="module")
def precisions3(precisions):
    precisions[RiskMeasure.VARIANCE] = 6
    precisions[RiskMeasure.WORST_REALIZATION] = 5
    precisions[RiskMeasure.CDAR] = 5
    precisions[RiskMeasure.EDAR] = 5
    return precisions


@pytest.fixture(scope="module")
def X(X):
    return X["2018-01-03":]


@pytest.fixture(scope="module")
def y(y):
    return y["2018-01-03":]


@pytest.fixture(
    scope="module",
    params=[
        rm
        for rm in RiskMeasure
        if not rm.is_annualized
        and rm
        not in [
            RiskMeasure.GINI_MEAN_DIFFERENCE,
            RiskMeasure.EDAR,
            RiskMeasure.EVAR,
        ]
    ],
)
def risk_measure(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[RiskMeasure.GINI_MEAN_DIFFERENCE, RiskMeasure.EDAR],
)
def risk_measure2(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        dict(min_weights=-1, max_weights=1),
        dict(
            min_weights=0,
            max_weights=None,
            budget=0.5,
        ),
        dict(
            min_weights=-0.2,
            max_weights=10,
            budget=None,
            min_budget=-1,
            max_budget=1,
        ),
        dict(
            min_weights=-1,
            max_weights=1,
            min_budget=-1,
            budget=None,
            max_budget=1,
            max_short=0.5,
            max_long=2,
        ),
    ],
)
def mean_risk_params(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[True, False],
)
def mean_risk_params_transaction_costs(request, previous_weights, transaction_costs):
    if request.param:
        return dict(
            previous_weights=previous_weights,
            transaction_costs=transaction_costs,
        )
    else:
        return dict()


@pytest.fixture(
    scope="module",
    params=[True, False],
)
def mean_risk_params_linear_constraints(request, groups, linear_constraints):
    if request.param:
        return dict(groups=groups, linear_constraints=linear_constraints)
    else:
        return dict()


@pytest.fixture(
    scope="module",
    params=[True, False],
)
def mean_risk_params_inequalities(request, groups, linear_constraints):
    _, _, left_inequality, right_inequality = equations_to_matrix(
        groups=np.array(groups), equations=linear_constraints
    )
    if request.param:
        return dict(left_inequality=left_inequality, right_inequality=right_inequality)
    else:
        return dict()


@pytest.fixture(
    scope="module",
    params=[{"l1_coef": 0.01, "l2_coef": 0}, {"l1_coef": 0, "l2_coef": 0.1}],
)
def mean_risk_params_coef(request):
    return request.param


def test_cvx_cache(X):
    n_observations, n_assets = X.shape

    model = MeanRisk()
    model.fit(X)
    w = cp.Variable(n_assets)
    factor = cp.Constant(1)
    model._clear_models_cache()
    res = model._cvx_drawdown(
        prior_model=model.prior_estimator_.prior_model_, w=w, factor=factor
    )
    a = res[0]
    assert len(res[1]) != 0
    res = model._cvx_drawdown(
        prior_model=model.prior_estimator_.prior_model_, w=w, factor=factor
    )
    b = res[0]
    assert len(res[1]) == 0
    assert "__cvx_drawdown" in model._cvx_cache
    assert hash(a) == hash(b)
    model._clear_models_cache()
    assert len(model._cvx_cache) == 0
    res = model._cvx_returns(prior_model=model.prior_estimator_.prior_model_, w=w)
    assert "_cvx_returns" in model._cvx_cache
    res2 = model._cvx_returns(prior_model=model.prior_estimator_.prior_model_, w=w)
    assert hash(res) == hash(res2)
    model._clear_models_cache()
    res3 = model._cvx_returns(prior_model=model.prior_estimator_.prior_model_, w=w)
    assert hash(res) != hash(res3)


def test_mean_risk_minimize_risk(
    X,
    precisions,
    risk_measure,
    mean_risk_params,
    mean_risk_params_transaction_costs,
    mean_risk_params_linear_constraints,
    mean_risk_params_inequalities,
):
    params = {
        **mean_risk_params,
        **mean_risk_params_transaction_costs,
        **mean_risk_params_linear_constraints,
        **mean_risk_params_inequalities,
    }
    precision = precisions[risk_measure]

    # Minimize risk
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk_measure,
        **params,
    )

    p = model.fit_predict(X)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure.value), model.problem_values_["risk"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, model.problem_values_["expected_return"], precision
    )


def test_mean_risk_minimize_risk_2(
    X_small,
    precisions,
    risk_measure2,
):
    precision = precisions[risk_measure2]

    # Minimize risk
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk_measure2,
    )

    p = model.fit_predict(X_small)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure2.value), model.problem_values_["risk"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, model.problem_values_["expected_return"], precision
    )


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_mean_risk_under_risk_and_return_constraint(
    X,
    precisions3,
    risk_measure,
    mean_risk_params,
    mean_risk_params_transaction_costs,
    mean_risk_params_linear_constraints,
):
    params = {
        **mean_risk_params,
        **mean_risk_params_transaction_costs,
        **mean_risk_params_linear_constraints,
    }
    precision = precisions3[risk_measure]
    max_risk_arg = f"max_{risk_measure.value}"

    # Minimize risk
    min_risk_model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk_measure,
        **params,
    )

    min_risk_model_ptf = min_risk_model.fit_predict(X)
    risk_constraint = min_risk_model.problem_values_["risk"] * 1.05

    # Maximize return under upper risk constraint
    max_return_model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        risk_measure=risk_measure,
        **params,
        **{max_risk_arg: risk_constraint},
    )

    p = max_return_model.fit_predict(X)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure.value), risk_constraint, precision
    )
    np.testing.assert_almost_equal(
        getattr(p, risk_measure.value),
        max_return_model.problem_values_["risk"],
        precision,
    )
    np.testing.assert_almost_equal(
        p.mean, max_return_model.problem_values_["expected_return"], precision
    )
    assert (
        max_return_model.problem_values_["expected_return"]
        >= min_risk_model_ptf.mean - 1e-6
    )

    # Minimize risk under lower return constraint
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk_measure,
        min_return=max_return_model.problem_values_["expected_return"],
        **params,
    )

    p = model.fit_predict(X)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure.value), model.problem_values_["risk"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, model.problem_values_["expected_return"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, max_return_model.problem_values_["expected_return"], precision
    )


def test_mean_risk_under_risk_and_return_constraint_2(
    X_small,
    precisions2,
    risk_measure2,
):
    precision = precisions2[risk_measure2]
    max_risk_arg = f"max_{risk_measure2.value}"

    # Minimize risk
    min_risk_model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk_measure2,
    )

    min_risk_model_ptf = min_risk_model.fit_predict(X_small)
    risk_constraint = (min_risk_model.problem_values_["risk"] + 1e-7) * 1.05

    # Maximize return under upper risk constraint
    max_return_model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        risk_measure=risk_measure2,
        **{max_risk_arg: risk_constraint},
    )

    p = max_return_model.fit_predict(X_small)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure2.value), risk_constraint, precision
    )
    np.testing.assert_almost_equal(
        getattr(p, risk_measure2.value),
        max_return_model.problem_values_["risk"],
        precision,
    )
    np.testing.assert_almost_equal(
        p.mean, max_return_model.problem_values_["expected_return"], precision
    )
    assert (
        max_return_model.problem_values_["expected_return"]
        >= min_risk_model_ptf.mean - 1e-6
    )

    # Minimize risk under lower return constraint
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk_measure2,
        min_return=max_return_model.problem_values_["expected_return"],
    )

    p = model.fit_predict(X_small)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure2.value), model.problem_values_["risk"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, model.problem_values_["expected_return"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, max_return_model.problem_values_["expected_return"], precision
    )


def test_mean_risk_utility(
    X,
    precisions,
    risk_measure,
    mean_risk_params,
    mean_risk_params_transaction_costs,
    mean_risk_params_linear_constraints,
):
    params = {
        **mean_risk_params,
        **mean_risk_params_transaction_costs,
        **mean_risk_params_linear_constraints,
    }
    precision = precisions[risk_measure]

    # Maximize utility
    risk_aversion = 3
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=risk_measure,
        risk_aversion=risk_aversion,
        **params,
    )

    p = model.fit_predict(X)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure.value), model.problem_values_["risk"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, model.problem_values_["expected_return"], precision
    )
    utility = model.problem_values_["objective"]
    p_utility = p.mean - risk_aversion * getattr(p, risk_measure.value)
    np.testing.assert_almost_equal(p_utility, utility, precision)


def test_mean_risk_utility2(
    X_small,
    precisions2,
    risk_measure2,
):
    precision = precisions2[risk_measure2]

    # Maximize utility
    risk_aversion = 3
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=risk_measure2,
        risk_aversion=risk_aversion,
    )

    p = model.fit_predict(X_small)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure2.value), model.problem_values_["risk"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, model.problem_values_["expected_return"], precision
    )
    utility = model.problem_values_["objective"]
    p_utility = p.mean - risk_aversion * getattr(p, risk_measure2.value)
    np.testing.assert_almost_equal(p_utility, utility, precision)


def test_mean_risk_ratio(
    X,
    precisions,
    risk_measure,
    mean_risk_params,
    mean_risk_params_transaction_costs,
    mean_risk_params_linear_constraints,
):
    params = {
        **mean_risk_params,
        **mean_risk_params_transaction_costs,
        **mean_risk_params_linear_constraints,
    }
    precision = precisions[risk_measure]

    # Maximize ratio
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=risk_measure,
        **params,
    )
    p = model.fit_predict(X)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure.value), model.problem_values_["risk"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, model.problem_values_["expected_return"], precision
    )


def test_mean_risk_ratio2(
    X_small,
    precisions2,
    risk_measure2,
):
    precision = precisions2[risk_measure2]

    # Maximize ratio
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=risk_measure2,
    )
    p = model.fit_predict(X_small)
    np.testing.assert_almost_equal(
        getattr(p, risk_measure2.value), model.problem_values_["risk"], precision
    )
    np.testing.assert_almost_equal(
        p.mean, model.problem_values_["expected_return"], precision
    )


def test_mean_risk_feature_names_in_(X):
    model = MeanRisk()
    model.fit(X)
    assert np.all(model.feature_names_in_ == X.columns)


def test_mean_risk_get_params():
    model = MeanRisk()
    params = model.get_params(deep=True)
    assert "prior_estimator" in params
    assert params["prior_estimator"] is None
    assert "prior_estimator__mu_estimator" not in params
    model = MeanRisk(prior_estimator=EmpiricalPrior(mu_estimator=EmpiricalMu()))
    params = model.get_params(deep=True)
    assert "prior_estimator" in params
    assert params["prior_estimator"] is not None
    assert "prior_estimator__mu_estimator" in params
    assert "prior_estimator__mu_estimator__window_size" in params


def test_mean_risk_set_params():
    model = MeanRisk()
    with pytest.raises(AttributeError):
        # noinspection PyTypeChecker
        model.set_params(prior_estimator__mu_estimator__window_size=30)

    model = MeanRisk(prior_estimator=EmpiricalPrior(mu_estimator=EmpiricalMu()))
    # noinspection PyTypeChecker
    model.set_params(prior_estimator__mu_estimator__window_size=30)

    params = model.get_params(deep=True)
    assert "prior_estimator__mu_estimator__window_size" in params
    assert params["prior_estimator__mu_estimator__window_size"] == 30


def test_mean_risk_cross_val_predict(X):
    prediction_mpp = cross_val_predict(
        MeanRisk(), X, cv=sks.KFold(n_splits=5), n_jobs=None
    )
    assert isinstance(prediction_mpp, MultiPeriodPortfolio)
    assert np.asarray(prediction_mpp).shape == (X.shape[0],)
    pop = Population([prediction_mpp])
    assert np.asarray(pop).shape == (1, X.shape[0])


def test_mean_risk_predict(X):
    model = MeanRisk()
    portfolio = model.fit_predict(X)
    assert isinstance(portfolio, Portfolio)
    sharpe = model.score(X)
    assert portfolio.sharpe_ratio == sharpe

    model = MeanRisk(min_return=[0.0005, 0.0001])
    model.fit(X)
    print(model.weights_)
    population = model.predict(X)
    assert isinstance(population, Population)

    model = MeanRisk(min_return=[0.0005, 0.0001], max_cdar=0.15)
    model.fit(X.to_numpy())
    print(model.weights_)
    population = model.predict(X.to_numpy())
    assert isinstance(population, Population)
    assert population[0].cdar <= 0.15
    assert population[1].cdar <= 0.15

    model = MeanRisk(efficient_frontier_size=10)
    model.fit(X)
    population = model.predict(X)
    assert isinstance(population, Population)
    sharpe = model.score(X)
    assert population.measures_mean(RatioMeasure.SHARPE_RATIO) == sharpe


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_regularization(X, risk_measure, mean_risk_params_coef):
    diff = 0.01

    max_risk_arg = f"max_{risk_measure.value}"

    # MINIMIZE_RISK
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk_measure,
    )
    model.fit(X)
    obj_ref = model.problem_values_["objective"]
    w_ref = model.weights_

    model.set_params(**mean_risk_params_coef)
    model.fit(X)
    obj = model.problem_values_["objective"]
    w = model.weights_

    assert obj > obj_ref
    if mean_risk_params_coef["l2_coef"] != 0:
        assert sum(np.square(w_ref)) - sum(np.square(w)) > diff

    # MAXIMIZE_RETURN
    risk = obj_ref * 1.2
    # noinspection PyTypeChecker
    model.set_params(
        l1_coef=0,
        l2_coef=0,
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        **{max_risk_arg: risk},
    )
    model.fit(X)
    obj_ref = model.problem_values_["objective"]
    w_ref = model.weights_

    model.set_params(**mean_risk_params_coef)
    model.fit(X)
    obj = model.problem_values_["objective"]
    w = model.weights_

    assert obj < obj_ref
    if mean_risk_params_coef["l2_coef"] != 0:
        assert sum(np.square(w_ref)) - sum(np.square(w)) > diff

    # MAXIMIZE_UTILITY
    risk_aversion = 3
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_UTILITY,
        risk_measure=risk_measure,
        risk_aversion=risk_aversion,
    )
    model.fit(X)
    obj_ref = model.problem_values_["objective"]
    w_ref = model.weights_

    model.set_params(**mean_risk_params_coef)
    model.fit(X)
    obj = model.problem_values_["objective"]
    w = model.weights_
    if risk_measure not in [RiskMeasure.EDAR]:
        assert obj < obj_ref
    if mean_risk_params_coef["l2_coef"] != 0:
        assert sum(np.square(w_ref)) - sum(np.square(w)) > diff

    # MAXIMIZE_RATIO
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=risk_measure,
    )
    model.fit(X)
    w_ref = model.weights_

    model.set_params(**mean_risk_params_coef)
    model.fit(X)
    w = model.weights_
    if risk_measure not in [RiskMeasure.EDAR]:
        if mean_risk_params_coef["l2_coef"] != 0:
            assert sum(np.square(w_ref)) - sum(np.square(w)) > diff


def test_worst_case_mean_variance(X):
    diff = 0.001
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.VARIANCE,
        mu_uncertainty_set_estimator=EmpiricalMuUncertaintySet(confidence_level=0.5),
    )
    # MINIMIZE_RISK
    model.fit(X)
    obj_ref = model.problem_values_["objective"]
    w_ref = model.weights_
    # noinspection PyTypeChecker
    model.set_params(
        covariance_uncertainty_set_estimator=EmpiricalCovarianceUncertaintySet(
            confidence_level=0.5
        ),
    )
    model.fit(X)
    obj = model.problem_values_["objective"]
    w = model.weights_

    assert obj != obj_ref
    assert sum(np.square(w_ref)) - sum(np.square(w)) > diff

    # MAXIMIZE_RETURN
    risk = obj_ref * 1.5
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        risk_measure=RiskMeasure.VARIANCE,
        max_variance=risk,
    )

    model.fit(X)
    obj_ref = model.problem_values_["objective"]
    w_ref = model.weights_
    # noinspection PyTypeChecker
    model.set_params(
        covariance_uncertainty_set_estimator=EmpiricalCovarianceUncertaintySet(
            confidence_level=0.5
        )
    )
    model.fit(X)
    obj = model.problem_values_["objective"]
    w = model.weights_

    assert obj < obj_ref
    assert sum(np.square(w_ref)) - sum(np.square(w)) > diff

    # MAXIMIZE_RATIO
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=RiskMeasure.VARIANCE,
    )

    model.fit(X)
    obj_ref = model.problem_values_["objective"]
    w_ref = model.weights_
    # noinspection PyTypeChecker
    model.set_params(
        scale_objective=1e-3,
        covariance_uncertainty_set_estimator=EmpiricalCovarianceUncertaintySet(
            confidence_level=0.5
        ),
    )

    model.fit(X)
    obj = model.problem_values_["objective"]
    w = model.weights_

    assert obj != obj_ref
    assert sum(np.square(w_ref)) - sum(np.square(w)) > diff


def test_transaction_costs(X, risk_measure):
    n_assets = X.shape[1]
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        risk_measure=risk_measure,
        budget=1,
        min_weights=0,
    )

    # Ref with no transaction_costs
    p_ref = model.fit_predict(X)
    w_ref = model.weights_
    # uniform transaction_costs for all assets and empty prev_weight --> no impact on
    # weights
    # noinspection PyTypeChecker
    model.set_params(
        transaction_costs=0.1 / 252, previous_weights=np.ones(n_assets) / n_assets
    )
    model.fit(X)
    w = model.weights_
    assert abs(w_ref - w).sum() > 5e-2

    # transaction_costs on top two invested assets and uniform prev_weight --> impact on
    # the two invested assets weight
    asset_1 = p_ref.composition.index[0]
    asset_2 = p_ref.composition.index[1]
    above = 5e-2
    transaction_costs = {asset_1: 0.2, asset_2: 0.5}
    # noinspection PyTypeChecker
    model.set_params(
        transaction_costs=transaction_costs,
        previous_weights=np.ones(n_assets) / n_assets,
    )
    p = model.fit_predict(X)
    assert abs(p.get_weight(asset=asset_1)) < abs(p_ref.get_weight(asset=asset_1))
    assert abs(p.get_weight(asset=asset_2)) < abs(p_ref.get_weight(asset=asset_2))
    assert abs(p_ref.weights - p.weights).sum() > above


def test_mean_risk_methods(X, risk_measure, precisions):
    n_assets = X.shape[1]
    target_risk_arg = f"max_{risk_measure.value}"
    precision = precisions[risk_measure]

    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=risk_measure,
        min_weights=0,
    )

    model.fit(X)
    min_risk = model.problem_values_["risk"]

    risk = min_risk * 1.1
    # noinspection PyTypeChecker
    model.set_params(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        **{target_risk_arg: risk},
    )

    p = model.fit_predict(X)
    ret = model.problem_values_["expected_return"]
    np.testing.assert_almost_equal(risk, getattr(p, risk_measure.value), precision)
    # noinspection PyTypeChecker
    model.set_params(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        min_return=ret,
        **{target_risk_arg: None},
    )
    p = model.fit_predict(X)
    np.testing.assert_almost_equal(ret, p.mean, precision)

    risks = [risk, risk * 1.1]
    # noinspection PyTypeChecker
    model.set_params(
        objective_function=ObjectiveFunction.MAXIMIZE_RETURN,
        min_return=None,
        **{target_risk_arg: risks},
    )
    pop = model.fit_predict(X)
    assert isinstance(pop, Population)
    assert len(model.problem_values_) == 2
    assert "objective" in model.problem_values_[0]
    assert model.weights_.shape == (2, n_assets)
    p0 = pop[0]
    p1 = pop[1]
    np.testing.assert_almost_equal(risks[0], getattr(p0, risk_measure.value), precision)
    np.testing.assert_almost_equal(risks[1], getattr(p1, risk_measure.value), precision)

    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        efficient_frontier_size=5,
        risk_measure=risk_measure,
        min_weights=0,
    )
    pop = model.fit_predict(X)
    assert isinstance(pop, Population)
    assert len(model.problem_values_) == 5
    assert "objective" in model.problem_values_[0]
    assert model.weights_.shape == (5, n_assets)
    p = pop[0]
    for i in range(1, 5):
        pi = pop[i]
        assert getattr(pi, risk_measure.value) > getattr(p, risk_measure.value)
        p = pi


def test_groups(X, groups, linear_constraints):
    model = MeanRisk(groups=groups, linear_constraints=linear_constraints)

    model.fit(X)
    w1 = model.weights_
    p1 = model.fit_predict(X)

    np.testing.assert_almost_equal(sum(w1), 1)
    assert np.all(w1 >= 0)
    assert w1[:3].sum() <= 0.5 * w1[-12:].sum()
    assert w1[:2].sum() >= 0.1
    assert w1[2:10].sum() >= 0.5 * w1[3:8].sum()
    assert w1[-10:].sum() <= 1

    model.fit(np.array(X))
    w2 = model.weights_
    p2 = model.fit_predict(X)
    np.testing.assert_almost_equal(w1, w2)

    new_groups = {X.columns[i]: [groups[0][i], groups[1][i]] for i in range(20)}

    model = MeanRisk(
        groups=new_groups,
        linear_constraints=linear_constraints,
    )
    model.fit(X)
    w3 = model.weights_
    p3 = model.fit_predict(X)
    with pytest.raises(ValueError):
        model.fit(np.array(X))
    np.testing.assert_almost_equal(w1, w3)
    np.testing.assert_almost_equal(p1.returns, p2.returns)
    np.testing.assert_almost_equal(p1.returns, p3.returns)


def test_tracking_error(X, y):
    model = MeanRisk(max_tracking_error=0.005)
    bench = y["SIZE"]
    p = model.fit(X, bench).predict(X)
    tracking_error = np.sqrt(
        np.sum((p.returns - np.asarray(bench)) ** 2) / (len(p.returns) - 1)
    )
    np.testing.assert_almost_equal(tracking_error, 0.005)


def test_turnover(X, y):
    previous_weights = np.ones(20) / 20
    model = MeanRisk(max_turnover=0.02, previous_weights=previous_weights)
    p = model.fit(X, y).predict(X)
    assert np.all(np.abs(p.weights - previous_weights) <= 0.02)


def test_mean_risk_factor_model(X, y):
    model = MeanRisk(prior_estimator=FactorModel())
    portfolio = model.fit(X, y).predict(X)
    assert isinstance(portfolio, Portfolio)


def test_optimization_factor_black_litterman(X, y):
    n_observations, n_assets = X.shape
    factor_views = ["MTUM - QUAL == 0.03 ", "SIZE - USMV== 0.04", "VLUE == 0.06"]

    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
        prior_estimator=FactorModel(
            factor_prior_estimator=BlackLitterman(
                views=np.asarray(factor_views), tau=1 / n_observations
            ),
            higham=True,
            residual_variance=False,
        ),
    )
    model.fit(X, y)
    np.testing.assert_almost_equal(
        model.prior_estimator_.prior_model_.mu,
        np.array(
            [
                0.04573766,
                0.07949394,
                0.05322793,
                0.04596431,
                0.04373008,
                0.05400534,
                0.03507889,
                0.00876922,
                0.04644162,
                0.00956013,
                0.01111351,
                0.01195771,
                0.04378188,
                0.01147724,
                0.0107007,
                0.00863476,
                0.06883195,
                0.02906054,
                0.01349415,
                0.04004494,
            ]
        ),
    )

    assert model.prior_estimator_.prior_model_.covariance.shape == (n_assets, n_assets)
    np.testing.assert_almost_equal(
        model.prior_estimator_.prior_model_.covariance[:5, 15:],
        np.array(
            [
                [
                    1.25634529e-04,
                    2.32021683e-04,
                    1.95783005e-04,
                    1.03417221e-04,
                    1.80971856e-04,
                ],
                [
                    1.41305357e-04,
                    3.33556413e-04,
                    2.47133368e-04,
                    1.27595433e-04,
                    2.35518998e-04,
                ],
                [
                    1.22471913e-04,
                    3.59441727e-04,
                    2.05669219e-04,
                    1.06975737e-04,
                    2.70975567e-04,
                ],
                [
                    1.24972237e-04,
                    2.79875907e-04,
                    1.97230854e-04,
                    1.04532010e-04,
                    2.19851561e-04,
                ],
                [
                    1.12629723e-04,
                    2.88745927e-04,
                    1.81882528e-04,
                    9.56316906e-05,
                    2.23238864e-04,
                ],
            ]
        ),
    )

    np.testing.assert_almost_equal(
        model.weights_,
        np.array(
            [
                3.23889556e-09,
                4.37353839e-01,
                3.50041391e-09,
                3.55794921e-09,
                3.73876428e-09,
                4.67673649e-09,
                3.69469609e-09,
                2.81118740e-09,
                3.57468357e-09,
                2.36272378e-09,
                2.51196427e-09,
                3.73429426e-09,
                3.55000669e-09,
                2.69088597e-09,
                2.56698189e-09,
                2.84579902e-09,
                5.62646099e-01,
                3.55289624e-09,
                5.57309135e-09,
                4.32763989e-09,
            ]
        ),
    )


def test_metadata_routing(X_small, implied_vol_small):
    with config_context(enable_metadata_routing=True):
        model = MeanRisk(
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


def test_mean_risk_linear_constraints_equalities(
    X,
):
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.VARIANCE,
        linear_constraints=["AMD == 0.2", "UNH==0.6"],
    )
    model.fit(X)
    np.testing.assert_almost_equal(model.weights_[1], 0.2)
    np.testing.assert_almost_equal(model.weights_[17], 0.6)
