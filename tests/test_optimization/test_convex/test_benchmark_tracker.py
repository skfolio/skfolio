from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import clone, config_context

from skfolio.measures import RiskMeasure
from skfolio.moments import EWCovariance, EWMu
from skfolio.optimization import BenchmarkTracker, MeanRisk, ObjectiveFunction
from skfolio.prior import EmpiricalPrior, TimeSeriesFactorModel


@pytest.mark.parametrize("first_method", ["fit", "partial_fit"])
@pytest.mark.parametrize("target_format", ["array", "column", "series", "dataframe"])
def test_partial_fit_matches_benchmark_excess_returns(first_method, target_format):
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(0, 0.01, (120, 4)), columns=list("ABCD"))
    y = rng.normal(0, 0.01, 120)
    excess_returns = X.subtract(y, axis=0)
    if target_format == "column":
        y = y[:, None]
    elif target_format == "series":
        y = pd.Series(y)
    elif target_format == "dataframe":
        y = pd.DataFrame(y)
    prior = EmpiricalPrior(mu_estimator=EWMu(), covariance_estimator=EWCovariance())
    model = BenchmarkTracker(prior_estimator=prior)
    reference = MeanRisk(
        prior_estimator=clone(prior), risk_measure=RiskMeasure.STANDARD_DEVIATION
    )
    getattr(model, first_method)(X[:60], y[:60])
    getattr(reference, first_method)(excess_returns[:60])
    fitted_prior = model.prior_estimator_
    model.partial_fit(X[60:], y[60:])
    reference.partial_fit(excess_returns[60:])
    assert model.prior_estimator_ is fitted_prior
    np.testing.assert_array_equal(
        fitted_prior.return_distribution_.returns, excess_returns
    )
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    np.testing.assert_allclose(model.weights_, reference.weights_, atol=1e-8)


@pytest.mark.parametrize("invalid", ["missing_y", "length", "budget", "schema"])
def test_partial_fit_rejects_invalid_benchmark_input_before_learning(invalid):
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(0, 0.01, (60, 4)), columns=list("ABCD"))
    y = rng.normal(0, 0.01, 60)
    model = BenchmarkTracker(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(), covariance_estimator=EWCovariance()
        )
    ).partial_fit(X, y)
    distribution = model.prior_estimator_.return_distribution_
    weights = model.weights_.copy()
    if invalid == "missing_y":
        y = None
    elif invalid == "length":
        y = y[:-1]
    elif invalid == "budget":
        model.budget = 0.5
    else:
        X = X.iloc[:, ::-1]
    with pytest.raises(ValueError):
        model.partial_fit(X, y)
    assert model.prior_estimator_.return_distribution_ is distribution
    np.testing.assert_array_equal(model.weights_, weights)
    np.testing.assert_array_equal(model.feature_names_in_, list("ABCD"))
    assert model.error_ is None


@pytest.mark.parametrize("first_method", ["fit", "partial_fit"])
def test_benchmark_tracker_dataframe_protocol(first_method):
    class ProtocolFrame:
        def __init__(self, frame):
            self.frame = frame

        def __dataframe__(self):
            return self

        def column_names(self):
            return self.frame.columns

        def __array__(self, dtype=None, copy=None):
            return self.frame.to_numpy(dtype=dtype, copy=copy or False)

    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(0, 0.01, (120, 4)), columns=list("ABCD"))
    y = rng.normal(0, 0.01, 120)
    model = BenchmarkTracker(
        prior_estimator=EmpiricalPrior(
            mu_estimator=EWMu(), covariance_estimator=EWCovariance()
        ),
        max_weights={"A": 0.1},
    )
    getattr(model, first_method)(ProtocolFrame(X[:60]), y[:60])
    model.partial_fit(ProtocolFrame(X[60:]), y[60:])

    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    assert model.weights_[0] <= 0.1 + 1e-6
    distribution = model.prior_estimator_.return_distribution_
    np.testing.assert_array_equal(distribution.returns, X.subtract(y, axis=0))

    weights = model.weights_.copy()
    with pytest.raises(ValueError, match="feature names"):
        model.partial_fit(ProtocolFrame(X.iloc[:, ::-1]), y)
    assert model.prior_estimator_.return_distribution_ is distribution
    np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    np.testing.assert_array_equal(model.weights_, weights)


def test_benchmark_tracker_fallback_receives_original_returns():
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(0, 0.01, (60, 4)), columns=list("ABCD"))
    y = rng.normal(0, 0.01, 60)
    model = BenchmarkTracker(min_weights=1.0, fallback=MeanRisk()).fit(X, y)
    assert isinstance(model.fallback_, MeanRisk)
    np.testing.assert_array_equal(
        model.fallback_.prior_estimator_.return_distribution_.returns, X
    )


@pytest.fixture
def benchmark_returns(factors):
    return factors["MTUM"]


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


def test_benchmark_tracker_factor_constraint(X, factors, benchmark_returns):
    factor_returns = factors.rename(columns={"MTUM": "Momentum"})
    with config_context(enable_metadata_routing=True):
        model = BenchmarkTracker(
            prior_estimator=TimeSeriesFactorModel(),
            linear_constraints=["Momentum == 0"],
        )
        model.fit(X, benchmark_returns, factors=factor_returns)

    factor_model = model.prior_estimator_.return_distribution_.factor_model
    momentum_exposure = model.weights_ @ factor_model.loading_matrix[:, 0]

    np.testing.assert_almost_equal(momentum_exposure, 0.0)


def test_benchmark_tracker_factor_family_constraint(X, factors, benchmark_returns):
    factor_returns = factors.rename(columns={"MTUM": "Momentum"})
    factor_families = ["style", "quality", "style", "defensive", "style"]
    with config_context(enable_metadata_routing=True):
        model = BenchmarkTracker(
            prior_estimator=TimeSeriesFactorModel(factor_families=factor_families),
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


def test_benchmark_tracker_non_investable_nan_assets(
    nan_investable_test_data, fixed_return_distribution_prior
):
    X, mu, covariance, investable_mask = nan_investable_test_data
    benchmark_returns = pd.Series(np.full(len(X), 0.005), index=X.index)

    model = BenchmarkTracker(
        prior_estimator=fixed_return_distribution_prior(mu=mu, covariance=covariance),
    )
    model.fit(X, benchmark_returns)

    return_distribution = model.prior_estimator_.return_distribution_
    assert return_distribution.n_assets == X.shape[1]
    assert return_distribution.n_investable_assets == np.count_nonzero(investable_mask)
    np.testing.assert_array_equal(model.investable_mask_, investable_mask)
    np.testing.assert_array_equal(model.feature_names_in_, X.columns.to_numpy())
    assert model.weights_.shape == (X.shape[1],)
    assert np.isfinite(model.weights_).all()
    np.testing.assert_allclose(model.weights_[~investable_mask], 0)
    np.testing.assert_allclose(model.weights_.sum(), 1)

    portfolio = model.predict(X)
    expected_returns = (
        X.iloc[:, investable_mask].to_numpy() @ model.weights_[investable_mask]
    )
    np.testing.assert_allclose(portfolio.returns, expected_returns)


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
