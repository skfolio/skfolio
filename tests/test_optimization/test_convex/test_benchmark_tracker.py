from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context

from skfolio.measures import RiskMeasure
from skfolio.optimization import BenchmarkTracker, MeanRisk, ObjectiveFunction
from skfolio.prior import TimeSeriesFactorModel


@pytest.fixture
def benchmark_returns(y):
    return y["MTUM"]


def test_benchmark_tracker(X, benchmark_returns):
    model = BenchmarkTracker(min_weights=0)
    model.fit(X, benchmark_returns)
    portfolio = model.predict(X)

    excess_returns = portfolio.returns - benchmark_returns.values
    tracking_error = np.std(excess_returns, ddof=1)
    np.testing.assert_almost_equal(
        tracking_error, model.problem_values_["risk"], decimal=4
    )


def test_benchmark_tracker_vs_manual(X, benchmark_returns):
    model1 = BenchmarkTracker(
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        min_weights=0,
    )
    model1.fit(X, benchmark_returns)

    excess_returns = X.copy()
    excess_returns.iloc[:, :] = X.values - benchmark_returns.values[:, np.newaxis]

    model2 = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.STANDARD_DEVIATION,
        min_weights=0,
    )
    model2.fit(excess_returns)

    np.testing.assert_almost_equal(model1.weights_, model2.weights_, decimal=6)


def test_benchmark_tracker_factor_constraint(X, y, benchmark_returns):
    factor_returns = y.rename(columns={"MTUM": "Momentum"})
    with config_context(enable_metadata_routing=True):
        model = BenchmarkTracker(
            prior_estimator=TimeSeriesFactorModel().set_fit_request(factors=True),
            linear_constraints=["Momentum == 0"],
        )
        model.fit(X, benchmark_returns, factors=factor_returns)

    factor_model = model.prior_estimator_.return_distribution_.factor_model
    momentum_exposure = model.weights_ @ factor_model.loading_matrix[:, 0]

    np.testing.assert_almost_equal(momentum_exposure, 0.0)


def test_benchmark_tracker_factor_family_constraint(X, y, benchmark_returns):
    factor_returns = y.rename(columns={"MTUM": "Momentum"})
    factor_families = ["style", "quality", "style", "defensive", "style"]
    with config_context(enable_metadata_routing=True):
        model = BenchmarkTracker(
            prior_estimator=TimeSeriesFactorModel(
                factor_families=factor_families
            ).set_fit_request(factors=True),
            linear_constraints=["style <= -0.05"],
        )
        model.fit(X, benchmark_returns, factors=factor_returns)

    factor_model = model.prior_estimator_.return_distribution_.factor_model
    style_mask = factor_model.factor_families == "style"
    family_exposure = (
        model.weights_ @ factor_model.loading_matrix[:, style_mask]
    ).sum()

    assert family_exposure <= -0.05


@pytest.mark.parametrize(
    "y_input",
    [
        "array",
        "series",
        "dataframe",
        "2d_array",
    ],
)
def test_benchmark_tracker_input_formats(X, y_input):
    if y_input == "array":
        benchmark_returns = np.random.randn(len(X)) * 0.01
    elif y_input == "series":
        benchmark_returns = pd.Series(np.random.randn(len(X)) * 0.01, index=X.index)
    elif y_input == "dataframe":
        benchmark_returns = pd.DataFrame(
            {"benchmark": np.random.randn(len(X)) * 0.01}, index=X.index
        )
    else:
        benchmark_returns = np.random.randn(len(X), 1) * 0.01

    model = BenchmarkTracker(min_weights=0)
    portfolio = model.fit(X, benchmark_returns).predict(X)

    assert portfolio.weights.shape == (X.shape[1],)


def test_benchmark_tracker_dict_weights(X, benchmark_returns):
    """Dict-based min/max weights require feature_names_in_ to survive the
    internal excess-returns transformation."""
    min_w = {name: 0.0 for name in X.columns}
    max_w = {name: 0.5 for name in X.columns}

    model = BenchmarkTracker(min_weights=min_w, max_weights=max_w)
    model.fit(X, benchmark_returns)
    portfolio = model.predict(X)

    assert portfolio.weights.shape == (X.shape[1],)
    np.testing.assert_array_less(portfolio.weights - 1e-6, 0.5)
    assert hasattr(model, "feature_names_in_")
    np.testing.assert_array_equal(model.feature_names_in_, X.columns.to_numpy())


def test_benchmark_tracker_dict_min_weights_with_dataframe_y(X, benchmark_returns):
    """Dict min_weights must work when y is a single-column DataFrame."""
    y_df = pd.DataFrame(
        {"benchmark": benchmark_returns.values}, index=benchmark_returns.index
    )
    min_w = {name: 0.01 for name in X.columns}

    model = BenchmarkTracker(min_weights=min_w)
    model.fit(X, y_df)
    portfolio = model.predict(X)

    assert portfolio.weights.shape == (X.shape[1],)
    np.testing.assert_array_less(0.01 - 1e-6, portfolio.weights)


def test_benchmark_tracker_errors(X, benchmark_returns):
    model = BenchmarkTracker()

    with pytest.raises(ValueError, match=r"benchmark returns.*must be provided"):
        model.fit(X, y=None)

    with pytest.raises(
        ValueError, match="Found input variables with inconsistent numbers of samples"
    ):
        model.fit(X, np.random.randn(len(X) - 10) * 0.01)

    multi_column_y = pd.DataFrame(
        {
            "b1": np.random.randn(len(X)) * 0.01,
            "b2": np.random.randn(len(X)) * 0.01,
        }
    )
    with pytest.raises(
        ValueError,
        match=r"y \(benchmark returns\) must be 1-dimensional or a single-column DataFrame/array, got shape \(2263, 2\)\.",
    ):
        model.fit(X, multi_column_y)

    model.budget = 2.0

    with pytest.raises(ValueError, match=r"Budget must be 1.0 for BenchmarkTracker"):
        model.fit(X, benchmark_returns)
