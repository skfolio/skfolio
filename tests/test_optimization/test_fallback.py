from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn as sk
import sklearn.model_selection as sks
import sklearn.utils.validation as skv
from sklearn import config_context
from sklearn.pipeline import Pipeline

from skfolio.model_selection import cross_val_predict
from skfolio.optimization import (
    BaseOptimization,
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    ObjectiveFunction,
)
from skfolio.portfolio import FailedPortfolio, Portfolio
from skfolio.pre_selection import (
    SelectKExtremes,
)
from skfolio.prior import TimeSeriesFactorModel
from skfolio.typing import FloatArray


def assert_weights_dict_subset_equal(d1: dict, d2: dict, tol: float = 1e-15) -> None:
    """True iff for every key k in d2 d1.get(k, 0.0) matches d2[k] within tol."""
    for k, b in d2.items():
        assert abs(d1.get(k, 0.0) - b) < tol


class CustomOptimization(BaseOptimization):
    """Simple custom optimizer forcing fit failure to test fallback"""

    def __init__(
        self,
        fail: bool = False,
        portfolio_params: dict | None = None,
        fallback=None,
        previous_weights: FloatArray | None = None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
            previous_weights=previous_weights,
        )
        self.fail = fail

    def fit(self, X, y=None, **fit_params):
        X = skv.validate_data(self, X)
        if self.fail:
            raise RuntimeError("CustomOptimization forced failure")
        n_assets = X.shape[1]
        self.weights_ = np.arange(n_assets, dtype=float)
        self.weights_ /= np.sum(self.weights_)
        return self


class CustomOptimizationWithoutFallback(BaseOptimization):
    """Simple custom optimizer without 'fallback' in __init__ to test
    backward-compatibility of the fallback mechanism.
    """

    def __init__(self, portfolio_params: dict | None = None):
        super().__init__(portfolio_params=portfolio_params)

    def fit(self, X, y=None):
        X = skv.validate_data(self, X)
        n_assets = X.shape[1]
        self.weights_ = np.arange(1, 1 + n_assets, dtype=float)
        self.weights_ /= np.sum(self.weights_)
        return self


class OptimizationFailingBeforeValidation(CustomOptimization):
    def fit(self, X, y=None):
        raise RuntimeError("Failure before input validation")


@pytest.fixture(params=["dataframe", "numpy", "list"])
def fallback_data(request):
    X = np.random.default_rng(0).normal(0, 0.01, (30, 4))
    if request.param == "dataframe":
        return pd.DataFrame(X, columns=["A", "B", "C", "D"])
    if request.param == "list":
        return X.tolist()
    return X


@pytest.mark.parametrize(
    "optimizer", [CustomOptimization, OptimizationFailingBeforeValidation]
)
@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_fallback_array_like_input(fallback_data, optimizer, raise_on_failure):
    model = optimizer(
        fail=True, fallback=EqualWeighted(), raise_on_failure=raise_on_failure
    )

    assert model.fit(fallback_data) is model

    np.testing.assert_array_equal(model.weights_, np.full(4, 0.25))
    assert model.n_features_in_ == 4
    assert isinstance(model.fallback_, EqualWeighted)
    assert model.fallback_chain_[-1] == ("EqualWeighted()", "success")
    assert len(model.fallback_chain_) == 2
    assert model.error_ is None
    if isinstance(fallback_data, pd.DataFrame):
        np.testing.assert_array_equal(model.feature_names_in_, fallback_data.columns)
    else:
        assert not hasattr(model, "feature_names_in_")
    portfolio = model.predict(fallback_data)
    assert isinstance(portfolio, Portfolio)
    assert not isinstance(portfolio, FailedPortfolio)
    np.testing.assert_allclose(portfolio.returns, np.mean(fallback_data, axis=1))


def test_fallback_refit_clears_stale_feature_names():
    X = pd.DataFrame(
        np.random.default_rng(0).normal(0, 0.01, (30, 4)), columns=list("ABCD")
    )
    model = OptimizationFailingBeforeValidation(fallback=EqualWeighted()).fit(X)
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)

    X_array = X.iloc[:, :3].to_numpy()
    model.fit(X_array)

    assert not hasattr(model, "feature_names_in_")
    assert model.n_features_in_ == 3
    np.testing.assert_array_equal(model.weights_, np.full(3, 1 / 3))
    np.testing.assert_allclose(model.predict(X_array).returns, X_array.mean(axis=1))


@pytest.mark.parametrize("previous_weights", [None, [0.5, 0.5]])
@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_fallback_previous_weights_failure_recorded_once(
    fallback_data, previous_weights, raise_on_failure
):
    model = CustomOptimization(
        fail=True,
        fallback="previous_weights",
        previous_weights=previous_weights,
        raise_on_failure=raise_on_failure,
    )
    if raise_on_failure:
        with pytest.raises((RuntimeError, ValueError), match="previous_weights"):
            model.fit(fallback_data)
    else:
        with pytest.warns(UserWarning, match="previous_weights"):
            model.fit(fallback_data)
        assert model.weights_ is None

    assert model.fallback_ is None
    assert model.fallback_chain_ == [
        (str(model), "CustomOptimization forced failure"),
        ("previous_weights", model.error_),
    ]


def test_fallback_previous_weights_failure_continues(fallback_data):
    model = CustomOptimization(
        fail=True, fallback=["previous_weights", EqualWeighted()]
    ).fit(fallback_data)

    assert len(model.fallback_chain_) == 3
    assert model.fallback_chain_[1][0] == "previous_weights"
    assert "'previous_weights' is None" in model.fallback_chain_[1][1]
    assert model.fallback_chain_[2] == ("EqualWeighted()", "success")
    assert model.error_ is None
    np.testing.assert_array_equal(model.weights_, np.full(4, 0.25))


def test_fallback_without_weights_continues(fallback_data):
    failed_fallback = CustomOptimization(fail=True, raise_on_failure=False)
    model = CustomOptimization(fail=True, fallback=[failed_fallback, EqualWeighted()])

    with pytest.warns(UserWarning, match="CustomOptimization forced failure"):
        model.fit(fallback_data)

    assert model.fallback_chain_ == [
        (str(model), "CustomOptimization forced failure"),
        (str(failed_fallback), "CustomOptimization forced failure"),
        ("EqualWeighted()", "success"),
    ]
    assert isinstance(model.fallback_, EqualWeighted)
    assert model.error_ is None
    np.testing.assert_array_equal(model.weights_, np.full(4, 0.25))
    assert not isinstance(model.predict(fallback_data), FailedPortfolio)


@pytest.mark.parametrize("raise_on_failure", [True, False])
def test_fallback_without_weights_exhausted(fallback_data, raise_on_failure):
    failed_fallback = CustomOptimization(fail=True, raise_on_failure=False)
    model = CustomOptimization(
        fail=True, fallback=failed_fallback, raise_on_failure=raise_on_failure
    )

    with pytest.warns(UserWarning, match="CustomOptimization forced failure"):
        if raise_on_failure:
            with pytest.raises(RuntimeError, match="CustomOptimization forced failure"):
                model.fit(fallback_data)
        else:
            model.fit(fallback_data)
            assert model.weights_ is None
            assert isinstance(model.predict(fallback_data), FailedPortfolio)

    assert model.fallback_ is None
    assert model.error_ == "CustomOptimization forced failure"
    assert model.fallback_chain_ == [
        (str(model), "CustomOptimization forced failure"),
        (str(failed_fallback), "CustomOptimization forced failure"),
    ]


def test_custom_optimization_no_fallback_param_in_init_still_works(X):
    model = CustomOptimizationWithoutFallback()
    model.fit(X)
    assert hasattr(model, "weights_")
    assert np.isclose(model.weights_.sum(), 1.0)
    assert model.fallback_ is None
    assert model.fallback_chain_ is None
    assert model.error_ is None
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)

    ptf = model.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


def test_fallback(X):
    # Primary estimator fails; first fallback also fails; second fallback succeeds
    model = CustomOptimization(fail=True, fallback=EqualWeighted())
    model.fit(X)
    assert hasattr(model, "weights_")
    np.testing.assert_array_equal(model.weights_, EqualWeighted().fit(X).weights_)
    assert isinstance(model.fallback_, EqualWeighted)
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback=EqualWeighted())",
            "CustomOptimization forced failure",
        ),
        ("EqualWeighted()", "success"),
    ]
    assert model.error_ is None
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    ptf = model.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


def test_fallback_inverse_volatility(X):
    model = MeanRisk(solver="NOT_A_SOLVER", fallback=InverseVolatility()).fit(X)
    expected_weights = 1 / np.std(X.to_numpy(), axis=0)
    expected_weights /= expected_weights.sum()

    assert isinstance(model.fallback_, InverseVolatility)
    assert model.fallback_chain_[-1] == ("InverseVolatility()", "success")
    assert model.n_features_in_ == X.shape[1]
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    np.testing.assert_allclose(model.weights_, expected_weights)

    portfolio = model.predict(X)
    assert isinstance(portfolio, Portfolio)
    np.testing.assert_allclose(portfolio.returns, X.to_numpy() @ expected_weights)


def test_fallback_with_clone(X):
    # Primary estimator fails; first fallback also fails; second fallback succeeds
    model = CustomOptimization(fail=True, fallback=EqualWeighted())
    model = sk.clone(model)
    model.fit(X)
    assert hasattr(model, "weights_")
    np.testing.assert_array_equal(model.weights_, EqualWeighted().fit(X).weights_)
    assert isinstance(model.fallback_, EqualWeighted)
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback=EqualWeighted())",
            "CustomOptimization forced failure",
        ),
        ("EqualWeighted()", "success"),
    ]
    assert model.error_ is None
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    ptf = model.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


def test_fallback_list_first_fails_then_succeeds(X):
    # Primary estimator fails; first fallback also fails; second fallback succeeds
    model = CustomOptimization(
        fail=True, fallback=[CustomOptimization(fail=True), EqualWeighted()]
    )

    model.fit(X)
    assert hasattr(model, "weights_")
    np.testing.assert_array_equal(model.weights_, EqualWeighted().fit(X).weights_)
    assert isinstance(model.fallback_, EqualWeighted)
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True,\n                   fallback=[CustomOptimization(fail=True), EqualWeighted()])",
            "CustomOptimization forced failure",
        ),
        ("CustomOptimization(fail=True)", "CustomOptimization forced failure"),
        ("EqualWeighted()", "success"),
    ]
    assert model.error_ is None
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)

    ptf = model.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


def test_fallback_chain_first_fails_then_succeeds(X):
    # Primary estimator fails; first fallback also fails; second fallback succeeds
    model = CustomOptimization(fail=True)
    model.fallback = CustomOptimization(fail=True)
    model.fallback.fallback = EqualWeighted()

    model.fit(X)
    assert hasattr(model, "weights_")
    np.testing.assert_array_equal(model.weights_, EqualWeighted().fit(X).weights_)
    assert isinstance(model.fallback_, CustomOptimization)
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True,\n                   fallback=CustomOptimization(fail=True,\n                                               fallback=EqualWeighted()))",
            "CustomOptimization forced failure",
        ),
        ("CustomOptimization(fail=True, fallback=EqualWeighted())", "success"),
    ]
    assert model.error_ is None
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    ptf = model.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


def test_predict_after_fallback_returns_portfolio(X):
    model = CustomOptimization(fail=True, fallback=EqualWeighted())
    ptf = model.fit(X).predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback=EqualWeighted())",
            "CustomOptimization forced failure",
        ),
        ("EqualWeighted()", "success"),
    ]
    assert not hasattr(ptf, "optimization_error")
    assert ptf.weights is not None and np.isclose(ptf.weights.sum(), 1.0)


def test_fallback_factor_model(X, factors):
    model = CustomOptimization(
        fail=True, fallback=MeanRisk(prior_estimator=TimeSeriesFactorModel())
    )
    model.fit(X, factors=factors)
    assert hasattr(model, "weights_")
    assert isinstance(model.fallback_, MeanRisk)
    assert model.fallback_chain_ == [
        (str(model), "CustomOptimization forced failure"),
        ("MeanRisk(prior_estimator=TimeSeriesFactorModel())", "success"),
    ]
    assert model.error_ is None
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    ptf = model.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


def test_cross_val_predict_with_fallback(X):
    model = CustomOptimization(fail=True, fallback=EqualWeighted())
    mpp = cross_val_predict(model, X, cv=sks.KFold(n_splits=3))
    assert mpp.n_failed_portfolios == 0
    assert mpp.n_fallback_portfolios == 3
    for ptf in mpp:
        assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
        assert ptf.fallback_chain == [
            (
                "CustomOptimization(fail=True, fallback=EqualWeighted())",
                "CustomOptimization forced failure",
            ),
            ("EqualWeighted()", "success"),
        ]
        assert not hasattr(ptf, "optimization_error")
        assert ptf.weights is not None and np.isclose(ptf.weights.sum(), 1.0)

    assert np.asarray(mpp).shape[0] == X.shape[0]
    assert not np.isnan(np.asarray(mpp)).any()
    assert isinstance(mpp.summary(), pd.Series)
    summary = mpp.summary()
    assert summary.loc["Number of Portfolios"] == "3"
    assert summary.loc["Number of Failed Portfolios"] == "0"
    assert summary.loc["Number of Fallback Portfolios"] == "3"
    summary = mpp.summary(formatted=False)
    assert summary.loc["Number of Portfolios"] == 3
    assert summary.loc["Number of Failed Portfolios"] == 0
    assert summary.loc["Number of Fallback Portfolios"] == 3


def test_failed_portfolio_when_raise_off(X):
    model = CustomOptimization(fail=True, raise_on_failure=False)
    with pytest.warns(UserWarning):
        model.fit(X)
    assert hasattr(model, "weights_")
    assert model.weights_ is None
    assert model.fallback_ is None
    assert model.fallback_chain_ is None
    assert model.error_ == "CustomOptimization forced failure"
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    ptf = model.predict(X)
    assert isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain is None


def test_failed_portfolio_when_raise_off_with_fallback(X):
    model = CustomOptimization(
        fail=True, fallback=CustomOptimization(fail=True), raise_on_failure=False
    )
    with pytest.warns(UserWarning):
        model.fit(X)
    assert hasattr(model, "weights_")
    assert model.weights_ is None
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback=CustomOptimization(fail=True),\n                   raise_on_failure=False)",
            "CustomOptimization forced failure",
        ),
        ("CustomOptimization(fail=True)", "CustomOptimization forced failure"),
    ]
    assert model.error_ == "CustomOptimization forced failure"
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    ptf = model.predict(X)
    assert isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


def test_fallback_with_raise_off(X):
    model = CustomOptimization(
        fail=True, fallback=CustomOptimization(fail=False), raise_on_failure=False
    )
    model.fit(X)
    assert hasattr(model, "weights_")
    assert model.weights_ is not None
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback=CustomOptimization(),\n                   raise_on_failure=False)",
            "CustomOptimization forced failure",
        ),
        ("CustomOptimization()", "success"),
    ]
    assert model.error_ is None
    assert model.n_features_in_ == 20
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    ptf = model.predict(X)
    assert ptf.fallback_chain == model.fallback_chain_


def test_cross_val_predict_failed_portfolio_when_raise_off(X):
    model = CustomOptimization(fail=True, raise_on_failure=False)
    with pytest.warns(UserWarning):
        mpp = cross_val_predict(model, X, cv=sks.KFold(n_splits=3))
    for ptf in mpp:
        assert isinstance(ptf, FailedPortfolio)
        assert ptf.optimization_error == "CustomOptimization forced failure"
        assert ptf.fallback_chain is None

    # All folds fail -> returns should be NaN
    arr = np.asarray(mpp)
    assert arr.shape[0] == X.shape[0]
    assert np.all(np.isnan(arr))


def test_fallback_previous_weights_array(fallback_data):
    n_assets = np.shape(fallback_data)[1]
    prev = np.arange(1, n_assets + 1, dtype=float)
    prev /= prev.sum()
    model = CustomOptimization(
        fail=True, fallback="previous_weights", previous_weights=prev
    )
    model.fit(fallback_data)
    np.testing.assert_allclose(model.weights_, prev)
    assert model.fallback_ == "previous_weights"
    assert model.fallback_chain_ == [
        (str(model), "CustomOptimization forced failure"),
        ("previous_weights", "success"),
    ]
    assert model.error_ is None
    ptf = model.predict(fallback_data)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


def test_fallback_previous_weights_dict(X):
    prev = {"AAPL": 0.4, "AMD": 0.2, "UNH": 0.4}
    model = CustomOptimization(
        fail=True, previous_weights=prev, fallback="previous_weights"
    )
    model.fit(X)
    np.testing.assert_allclose(
        model.weights_,
        [
            0.4,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.4,
            0.0,
            0.0,
        ],
    )
    assert model.fallback_ == "previous_weights"
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback='previous_weights',\n                   previous_weights={'AAPL': 0.4, 'AMD': 0.2, 'UNH': 0.4})",
            "CustomOptimization forced failure",
        ),
        ("previous_weights", "success"),
    ]
    assert model.error_ is None
    ptf = model.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == model.fallback_chain_


@pytest.mark.filterwarnings("ignore:Solution may be inaccurate")
def test_hyperparam_tuning_on_fallback_param(X):
    # Force primary to fail; tune which fallback to use
    model = MeanRisk(
        objective_function=ObjectiveFunction.MAXIMIZE_RATIO, min_weights=1.0
    )
    param_grid = {
        "fallback": [EqualWeighted(), HierarchicalRiskParity()],
    }
    gs = sks.GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
    gs.fit(X)

    best_model = gs.best_estimator_
    assert hasattr(best_model, "weights_")
    assert isinstance(best_model.fallback_, HierarchicalRiskParity)
    assert best_model.fallback_chain_ == [
        (
            "MeanRisk(fallback=HierarchicalRiskParity(), min_weights=1.0,\n         objective_function=MAXIMIZE_RATIO)",
            "Solver 'CLARABEL' failed. Try another solver, or solve with solver_params=dict(verbose=True) for more information",
        ),
        ("HierarchicalRiskParity()", "success"),
    ]
    assert best_model.error_ is None
    ptf = best_model.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == best_model.fallback_chain_


def test_pipeline_on_fallback(X):
    model = CustomOptimization(fail=True, fallback=EqualWeighted())
    pipe = Pipeline([("pre_selection", SelectKExtremes(k=10)), ("optim", model)])

    with config_context(transform_output="pandas"):
        pipe.fit(X)

    om = pipe.named_steps["optim"]
    assert om.weights_ is not None
    assert isinstance(om.fallback_, EqualWeighted)
    assert om.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback=EqualWeighted())",
            "CustomOptimization forced failure",
        ),
        ("EqualWeighted()", "success"),
    ]
    assert om.error_ is None

    ptf = pipe.predict(X)
    assert isinstance(ptf, Portfolio) and not isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == om.fallback_chain_


def test_pipeline_on_fallback_raise_off(X):
    model = CustomOptimization(
        fail=True, fallback=CustomOptimization(fail=True), raise_on_failure=False
    )
    pipe = Pipeline([("pre_selection", SelectKExtremes(k=10)), ("optim", model)])

    with pytest.warns(UserWarning):
        with config_context(transform_output="pandas"):
            pipe.fit(X)

    om = pipe.named_steps["optim"]
    assert om.weights_ is None
    assert om.fallback_ is None
    assert om.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback=CustomOptimization(fail=True),\n                   raise_on_failure=False)",
            "CustomOptimization forced failure",
        ),
        ("CustomOptimization(fail=True)", "CustomOptimization forced failure"),
    ]
    assert om.error_ == "CustomOptimization forced failure"

    ptf = pipe.predict(X)
    assert isinstance(ptf, FailedPortfolio)
    assert ptf.fallback_chain == om.fallback_chain_
    assert ptf.optimization_error == om.error_


def test_previous_weights_propagation(X):
    prev = {"AAPL": 0.4, "AMD": 0.2, "UNH": 0.4}
    model = CustomOptimization(
        fail=True, previous_weights=prev, fallback=CustomOptimization(fail=False)
    )
    model.fit(X)
    assert model.fallback_.previous_weights == prev
    ptf = model.predict(X)
    assert_weights_dict_subset_equal(ptf.previous_weights_dict, prev)

    model = CustomOptimization(
        fail=True,
        previous_weights=prev,
        fallback=CustomOptimization(fail=True, fallback="previous_weights"),
    )
    model.fit(X)
    assert model.fallback_.previous_weights == prev
    np.testing.assert_array_equal(
        model.weights_,
        [
            0.4,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.4,
            0.0,
            0.0,
        ],
    )
    ptf = model.predict(X)
    assert_weights_dict_subset_equal(ptf.previous_weights_dict, prev)

    model = CustomOptimization(
        fail=True,
        previous_weights=prev,
        fallback=[CustomOptimization(fail=True), "previous_weights"],
    )
    model.fit(X)
    np.testing.assert_array_equal(
        model.weights_,
        [
            0.4,
            0.2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.4,
            0.0,
            0.0,
        ],
    )
    ptf = model.predict(X)
    assert_weights_dict_subset_equal(ptf.previous_weights_dict, prev)


def test_fallback_needs_previous_weights(X):
    model = MeanRisk(
        fallback=MeanRisk(
            transaction_costs=0.001,
        ),
    )
    assert model.needs_previous_weights is True

    model = MeanRisk(
        fallback=MeanRisk(),
    )
    assert model.needs_previous_weights is False

    model = MeanRisk(
        fallback="previous_weights",
    )
    assert model.needs_previous_weights is True

    model = MeanRisk(
        fallback=[MeanRisk(), "previous_weights"],
    )
    assert model.needs_previous_weights is True

    model = MeanRisk(
        fallback=[MeanRisk(), MeanRisk()],
    )
    assert model.needs_previous_weights is False

    model = MeanRisk(
        fallback=[MeanRisk(), MeanRisk(max_turnover=0.5)],
    )
    assert model.needs_previous_weights is True


def test_subclass_without_fit_keeps_parent_wrapped_fit():
    class ChildWithoutFit(CustomOptimization):
        pass

    class ChildReusingWrappedFit(CustomOptimization):
        fit = CustomOptimization.fit

    assert ChildWithoutFit.fit is CustomOptimization.fit
    assert ChildReusingWrappedFit.fit is CustomOptimization.fit
    assert ChildWithoutFit.fit._fallback_wrapped is True


def test_fallback_empty_list_raises_primary_error(X):
    model = CustomOptimization(fail=True, fallback=[])
    with pytest.raises(RuntimeError, match="CustomOptimization forced failure"):
        model.fit(X)
    assert model.fallback_chain_ == [(str(model), "CustomOptimization forced failure")]


def test_fallback_previous_weights_conflict_warns(X):
    prev = np.full(X.shape[1], 1 / X.shape[1])
    model = CustomOptimization(
        fail=True,
        previous_weights=prev,
        fallback=CustomOptimization(fail=False, previous_weights=np.zeros(X.shape[1])),
    )
    with pytest.warns(
        UserWarning, match="previous_weights are automatically propagated"
    ):
        model.fit(X)
    np.testing.assert_array_equal(model.fallback_.previous_weights, prev)


def test_predict_copies_portfolio_params(X):
    model = CustomOptimization(portfolio_params={"name": "custom_ptf"})
    ptf = model.fit(X).predict(X)
    assert ptf.name == "custom_ptf"
    assert model.portfolio_params == {"name": "custom_ptf"}


def test_invalid_string_fallback_raises(X):
    model = CustomOptimization(fail=True, fallback="bad")
    with pytest.raises(ValueError, match="Unsupported string fallback: 'bad'"):
        model.fit(X)
    with pytest.raises(ValueError, match="Unsupported string fallback: 'bad'"):
        _ = model.needs_previous_weights


def test_invalid_type_fallback_raises(X):
    model = CustomOptimization(fail=True, fallback=5)
    with pytest.raises(
        TypeError, match=r"must inherit from BaseOptimization \(got int\)"
    ):
        model.fit(X)
    with pytest.raises(
        TypeError, match=r"must inherit from BaseOptimization \(got int\)"
    ):
        _ = model.needs_previous_weights


@pytest.mark.parametrize(
    "transaction_costs,expected",
    [
        ({"AAPL": 0.01}, True),
        ({"AAPL": 0.0}, False),
        ({}, False),
        ([], False),
        (["not-a-number"], True),
    ],
)
def test_needs_previous_weights_transaction_costs(transaction_costs, expected):
    model = MeanRisk(transaction_costs=transaction_costs)
    assert model.needs_previous_weights is expected


def test_weight_drift_needs_previous_weights():
    assert MeanRisk().needs_previous_weights is False
    assert (
        MeanRisk(portfolio_params={"weight_drift": True}).needs_previous_weights is True
    )
    assert (
        MeanRisk(portfolio_params={"weight_drift": False}).needs_previous_weights
        is False
    )
