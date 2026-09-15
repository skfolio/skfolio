from __future__ import annotations

import gc
import pickle
import weakref

import numpy as np
import pandas as pd
import pytest

from skfolio import FailedPortfolio, Portfolio, RiskMeasure
from skfolio.portfolio import _portfolio as portfolio_module


@pytest.fixture
def returns():
    return pd.DataFrame(
        [[0.01, 0.02], [0.03, 0.04], [-0.01, 0.02]],
        columns=["A", "B"],
        index=pd.date_range("2024-01-01", periods=3),
    )


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64", "bool"])
@pytest.mark.parametrize("dataframe", [False, True])
def test_float64_calculations_preserve_original_input(returns, dtype, dataframe):
    X = returns.astype(dtype)
    if not dataframe:
        X = X.to_numpy().copy()
    original = np.asarray(X).copy()
    writeable = np.asarray(X).flags.writeable

    portfolio = Portfolio(X, [0.5, 0.5], weight_drift=True)

    assert portfolio.X is X
    values = portfolio_module._to_numpy_returns(X)
    assert values.dtype == np.dtype(float)
    if dtype == "float64":
        assert np.shares_memory(values, np.asarray(X))
    np.testing.assert_array_equal(X, original)
    assert np.asarray(X).flags.writeable == writeable
    if dataframe:
        np.testing.assert_array_equal(portfolio.assets, X.columns)
        np.testing.assert_array_equal(portfolio.observations, X.index)


@pytest.mark.parametrize("dtype", ["Float32", "Float64", "Int64", "boolean"])
def test_nullable_numeric_input(dtype):
    X = pd.DataFrame({"A": [0, 1, None], "B": [1, None, 0]}, dtype=dtype)
    before = X.copy(deep=True)
    expected = Portfolio(X.to_numpy(dtype=float, na_value=np.nan), [0.4, 0.6])

    actual = Portfolio(X, [0.4, 0.6])

    assert actual.X is X
    np.testing.assert_allclose(actual.returns, expected.returns)
    np.testing.assert_allclose(actual._get_weights_path(), expected._get_weights_path())
    assert np.isnan(actual.diversification)
    pd.testing.assert_frame_equal(X, before)


@pytest.mark.parametrize("sparse", ["all", "mixed", "nullable"])
def test_sparse_dataframe(returns, sparse):
    returns.iloc[1, 0] = np.nan
    X = returns.astype("float32")
    X["A"] = X["A"].astype(pd.SparseDtype("float32", 0))
    if sparse == "all":
        X["B"] = X["B"].astype(pd.SparseDtype("float32", 0))
    elif sparse == "nullable":
        X["B"] = X["B"].astype("Float32")
        X.loc[X.index[1], "B"] = pd.NA
        returns.iloc[1, 1] = np.nan
    before = X.copy(deep=True)
    expected_values = returns.to_numpy(dtype=np.float32)
    expected = Portfolio(expected_values, [0.4, 0.6], weight_drift=True)

    actual = Portfolio(X, [0.4, 0.6], weight_drift=True)

    assert actual.X is X
    np.testing.assert_allclose(actual.returns, expected.returns)
    np.testing.assert_allclose(actual._get_weights_path(), expected._get_weights_path())
    assert np.isnan(actual.diversification)
    pd.testing.assert_frame_equal(X, before)


@pytest.mark.parametrize("array_like", ["list", "object", "numeric_strings"])
def test_numeric_array_like(returns, array_like):
    if array_like == "list":
        X = returns.to_numpy().tolist()
    elif array_like == "object":
        X = returns.astype(object)
    else:
        X = returns.astype(str).astype(object)
    expected = Portfolio(returns, [0.4, 0.6])

    actual = Portfolio(X, [0.4, 0.6])

    assert actual.X is X
    np.testing.assert_allclose(actual.returns, expected.returns)
    np.testing.assert_allclose(actual.diversification, expected.diversification)
    np.testing.assert_allclose(actual._get_weights_path(), expected._get_weights_path())


@pytest.mark.parametrize(
    "invalid, match",
    [(np.inf, "infinity"), (-np.inf, "infinity"), (1j, "Complex"), ("bad", "float")],
)
def test_invalid_returns(returns, invalid, match):
    X = returns.astype(complex if invalid == 1j else object)
    X.iloc[0, 0] = invalid
    with pytest.raises(ValueError, match=match):
        Portfolio(X, [0.5, 0.5])


@pytest.mark.parametrize("shape", [(0, 2), (3, 0), (0, 0)])
@pytest.mark.parametrize("dataframe", [False, True])
def test_empty_returns(shape, dataframe):
    X = np.empty(shape)
    if dataframe:
        X = pd.DataFrame(X)
    portfolio = Portfolio(X, np.zeros(shape[1]))
    np.testing.assert_array_equal(portfolio.returns, np.zeros(shape[0]))
    assert portfolio._get_weights_path().shape == shape


@pytest.mark.parametrize("X", [np.zeros(2), np.zeros((2, 2, 2)), 0.0])
def test_returns_require_two_dimensions(X):
    with pytest.raises(ValueError):
        Portfolio(X, [0.5, 0.5])


def test_nullable_portfolio_contribution(returns):
    expected = Portfolio(returns, [0.4, 0.6])
    X = returns.astype("Float64")
    before = X.copy(deep=True)
    portfolio = Portfolio(X, [0.4, 0.6])

    np.testing.assert_allclose(
        portfolio.contribution(RiskMeasure.VARIANCE),
        expected.contribution(RiskMeasure.VARIANCE),
    )
    assert portfolio.X is X
    pd.testing.assert_frame_equal(X, before)


def test_pickle_preserves_original_input(returns):
    portfolio = Portfolio(returns.astype("Float64"), [0.4, 0.6])
    restored = pickle.loads(pickle.dumps(portfolio))
    np.testing.assert_allclose(restored.returns, portfolio.returns)
    np.testing.assert_allclose(restored.diversification, portfolio.diversification)
    pd.testing.assert_frame_equal(restored.X, portfolio.X)


@pytest.mark.parametrize("missing", [False, True])
def test_converted_returns_not_retained(returns, missing, monkeypatch):
    X = returns.astype("Float64")
    if missing:
        X.iloc[0, 0] = pd.NA
    buffers = []
    convert = portfolio_module._to_numpy_returns

    def tracked(*args, **kwargs):
        values = convert(*args, **kwargs)
        buffers.append(weakref.ref(values))
        return values

    monkeypatch.setattr(portfolio_module, "_to_numpy_returns", tracked)
    portfolio = Portfolio(X, [0.4, 0.6])
    portfolio.contribution(RiskMeasure.VARIANCE)
    gc.collect()

    assert portfolio.X is X
    assert buffers and all(ref() is None for ref in buffers)


def test_failed_portfolio_skips_numeric_validation(returns, monkeypatch):
    X = returns.astype(object)
    X.iloc[0, 0] = "invalid"

    def unexpected_validation(*args, **kwargs):
        pytest.fail("Failed portfolios must not validate numeric returns")

    monkeypatch.setattr(portfolio_module, "_to_numpy_returns", unexpected_validation)
    portfolio = FailedPortfolio(X, optimization_error="invalid returns")

    assert portfolio.X is X
    np.testing.assert_array_equal(portfolio.assets, X.columns)
    np.testing.assert_array_equal(portfolio.observations, X.index)
    assert np.isnan(portfolio.returns).all()
    assert np.isnan(portfolio.diversification)
    assert np.isnan(portfolio._get_weights_path()).all()
