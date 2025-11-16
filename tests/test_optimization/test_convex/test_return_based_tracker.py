import numpy as np
import pandas as pd
import pytest

from skfolio.measures import RiskMeasure
from skfolio.optimization import MeanRisk, ReturnBasedTracker, ObjectiveFunction


def test_return_based_tracker(X):
    benchmark_returns = X.mean(axis=1)
    model = ReturnBasedTracker(min_weights=0)
    model.fit(X, benchmark_returns)
    portfolio = model.predict(X)

    excess_returns = portfolio.returns - benchmark_returns.values
    tracking_error = np.std(excess_returns, ddof=1)

    np.testing.assert_almost_equal(
        tracking_error, model.problem_values_["risk"], decimal=4
    )


def test_return_based_tracker_vs_manual(X):
    benchmark_returns = X.mean(axis=1)

    model1 = ReturnBasedTracker(
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


@pytest.mark.parametrize(
    "y_input",
    [
        "array",
        "series",
        "dataframe",
        "2d_array",
    ],
)
def test_return_based_tracker_input_formats(X, y_input):
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

    model = ReturnBasedTracker(min_weights=0)
    portfolio = model.fit(X, benchmark_returns).predict(X)

    assert portfolio.weights.shape == (X.shape[1],)


def test_return_based_tracker_errors(X):
    model = ReturnBasedTracker()

    with pytest.raises(ValueError, match=r"benchmark returns.*must be provided"):
        model.fit(X, y=None)

    with pytest.raises(ValueError, match="Found input variables with inconsistent numbers of samples"):
        model.fit(X, np.random.randn(len(X) - 10) * 0.01)

    multi_column_y = pd.DataFrame(
        {
            "b1": np.random.randn(len(X)) * 0.01,
            "b2": np.random.randn(len(X)) * 0.01,
        }
    )
    with pytest.raises(ValueError, match="y should be a 1d array, got an array of shape"):
        model.fit(X, multi_column_y)
