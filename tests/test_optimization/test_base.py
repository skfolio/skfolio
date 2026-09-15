from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from skfolio.optimization import EqualWeighted, InverseVolatility, MeanRisk
from skfolio.portfolio import FailedPortfolio, Portfolio
from skfolio.portfolio import _portfolio as portfolio_module


@pytest.fixture(params=[MeanRisk, InverseVolatility], ids=lambda cls: cls.__name__)
def model(request, X):
    return request.param().fit(X)


def test_predict_feature_names(model, X):
    weights = model.weights_.copy()
    expected_returns = X.to_numpy() @ weights

    portfolio = model.predict(X)

    assert isinstance(portfolio, Portfolio)
    np.testing.assert_array_equal(portfolio.assets, X.columns)
    np.testing.assert_allclose(portfolio.returns, expected_returns)
    np.testing.assert_array_equal(model.weights_, weights)


def test_predict_without_feature_names(model, X):
    weights = model.weights_.copy()
    expected_returns = X.to_numpy() @ weights

    with pytest.warns(UserWarning, match="X does not have valid feature names"):
        portfolio = model.predict(X.to_numpy())

    assert isinstance(portfolio, Portfolio)
    np.testing.assert_allclose(portfolio.returns, expected_returns)
    np.testing.assert_array_equal(model.weights_, weights)


@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda X: X.iloc[:, ::-1], id="reordered"),
        pytest.param(lambda X: X.iloc[:, :-1], id="missing"),
        pytest.param(lambda X: X.assign(unexpected_asset=0.0), id="additional"),
        pytest.param(
            lambda X: X.rename(columns={X.columns[0]: "unexpected_asset"}),
            id="renamed",
        ),
    ],
)
def test_predict_invalid_feature_names(model, X, transform):
    weights = model.weights_.copy()
    feature_names = model.feature_names_in_.copy()
    n_features = model.n_features_in_

    with pytest.raises(ValueError, match="feature names should match"):
        model.predict(transform(X))

    np.testing.assert_array_equal(model.weights_, weights)
    np.testing.assert_array_equal(model.feature_names_in_, feature_names)
    assert model.n_features_in_ == n_features


@pytest.mark.parametrize("weight_drift", [False, True])
def test_population_nullable_returns(weight_drift):
    X = pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.04], [-0.01, 0.02]],
        columns=["A", "B"],
        dtype="Float64",
    )
    model = EqualWeighted(portfolio_params={"weight_drift": weight_drift}).fit(X)
    model.weights_ = np.array([[np.nan, np.nan], [0.4, 0.6], [0.2, 0.8]])
    model.error_ = ["failed", None, None]
    X.iloc[0, 0] = pd.NA
    expected = [
        Portfolio(
            X.to_numpy(dtype=float, na_value=np.nan), w, weight_drift=weight_drift
        )
        for w in model.weights_[1:]
    ]
    population = model.predict(X)

    assert isinstance(population[0], FailedPortfolio)
    assert all(p.X is X for p in population)
    for actual, reference in zip(population[1:], expected, strict=True):
        np.testing.assert_allclose(actual.returns, reference.returns)
        np.testing.assert_allclose(actual.ending_weights, reference.ending_weights)
        assert np.isnan(actual.diversification)


@pytest.mark.parametrize("bad", [np.inf, "bad"])
def test_failed_optimization_preserves_invalid_returns(bad):
    X = pd.DataFrame([[bad, 0.02], [0.03, 0.04]], columns=["A", "B"])
    with pytest.warns(UserWarning, match="failed"):
        portfolio = EqualWeighted(raise_on_failure=False).fit_predict(X)
    assert isinstance(portfolio, FailedPortfolio)
    assert portfolio.X is X
    assert np.isnan(portfolio.returns).all()


def test_all_failed_population_skips_numeric_validation(monkeypatch):
    X = pd.DataFrame([[0.01, 0.02], [0.03, 0.04]], columns=["A", "B"])
    model = EqualWeighted().fit(X)
    model.weights_ = np.full((2, 2), np.nan)
    model.error_ = ["failed", "failed"]
    invalid = X.astype(object)
    invalid.iloc[0, 0] = "invalid"

    def unexpected_validation(*args, **kwargs):
        pytest.fail("An all-failed population must not validate numeric returns")

    monkeypatch.setattr(portfolio_module, "_to_numpy_returns", unexpected_validation)
    population = model.predict(invalid)

    assert len(population) == 2
    assert all(isinstance(p, FailedPortfolio) for p in population)
    assert all(p.X is invalid for p in population)
