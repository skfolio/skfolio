from __future__ import annotations

import numpy as np
import pytest

from skfolio.optimization import InverseVolatility, MeanRisk
from skfolio.portfolio import Portfolio


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
