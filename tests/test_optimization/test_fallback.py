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
    MeanRisk,
    ObjectiveFunction,
)
from skfolio.portfolio import FailedPortfolio, Portfolio
from skfolio.pre_selection import (
    SelectKExtremes,
)
from skfolio.prior import FactorModel


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
        previous_weights: np.ndarray | None = None,
        raise_on_failure: bool = True,
    ):
        super().__init__(
            portfolio_params=portfolio_params,
            fallback=fallback,
            raise_on_failure=raise_on_failure,
            previous_weights=previous_weights,
        )
        self.fail = fail

    def fit(self, X, y=None):
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


def test_fallback_factor_model(X, y):
    model = CustomOptimization(
        fail=True, fallback=MeanRisk(prior_estimator=FactorModel())
    )
    model.fit(X, y)
    assert hasattr(model, "weights_")
    assert isinstance(model.fallback_, MeanRisk)
    assert model.fallback_chain_ == [
        (
            "CustomOptimization(fail=True, fallback=MeanRisk(prior_estimator=FactorModel()))",
            "CustomOptimization forced failure",
        ),
        ("MeanRisk(prior_estimator=FactorModel())", "success"),
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


def test_fallback_previous_weights_array(X):
    n_assets = X.shape[1]
    prev = np.arange(1, n_assets + 1, dtype=float)
    prev /= prev.sum()
    model = CustomOptimization(
        fail=True, fallback="previous_weights", previous_weights=prev
    )
    model.fit(X)
    np.testing.assert_allclose(model.weights_, prev)
    assert model.fallback_ == "previous_weights"
    assert model.error_ is None
    ptf = model.predict(X)
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
