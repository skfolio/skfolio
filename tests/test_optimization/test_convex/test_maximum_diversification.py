from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context

from skfolio.moments import ImpliedCovariance
from skfolio.optimization.convex import (
    MaximumDiversification,
)
from skfolio.prior import EmpiricalPrior, TimeSeriesFactorModel


def test_maximum_diversification(X):
    model = MaximumDiversification()
    model.fit(X)
    ptf = model.predict(X)
    diversification = (
        model.problem_values_["expected_return"] / model.problem_values_["risk"]
    )
    np.testing.assert_almost_equal(ptf.diversification, diversification, 3)


def test_maximum_diversification_factor(X, y):
    model = MaximumDiversification(prior_estimator=TimeSeriesFactorModel())
    model.fit(X, y)
    ptf = model.predict(X)
    diversification = (
        model.problem_values_["expected_return"] / model.problem_values_["risk"]
    )

    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(ptf.diversification, diversification, 3)


def test_maximum_diversification_factor_constraint(X, y):
    factor_returns = y.rename(columns={"MTUM": "Momentum"})
    model = MaximumDiversification(
        prior_estimator=TimeSeriesFactorModel(),
        linear_constraints=["Momentum == 0"],
    )
    model.fit(X, factor_returns)

    factor_model = model.prior_estimator_.return_distribution_.factor_model
    momentum_exposure = model.weights_ @ factor_model.loading_matrix[:, 0]

    np.testing.assert_almost_equal(momentum_exposure, 0.0)


def test_maximum_diversification_factor_family_constraint(X, y):
    factor_returns = y.rename(columns={"MTUM": "Momentum"})
    factor_families = ["style", "quality", "style", "defensive", "style"]
    model = MaximumDiversification(
        prior_estimator=TimeSeriesFactorModel(factor_families=factor_families),
        linear_constraints=["style <= -0.05"],
    )
    model.fit(X, factor_returns)

    factor_model = model.prior_estimator_.return_distribution_.factor_model
    style_mask = factor_model.factor_families == "style"
    family_exposure = (
        model.weights_ @ factor_model.loading_matrix[:, style_mask]
    ).sum()

    assert family_exposure <= -0.05


def test_metadata_routing(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = MaximumDiversification(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        with pytest.raises(ValueError):
            model.fit(X)

        model.fit(X, implied_vol=implied_vol)

    # noinspection PyUnresolvedReferences
    assert model.prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)


def test_maximum_diversification_non_investable_nan_assets(
    nan_investable_test_data, fixed_return_distribution_prior
):
    X, mu, covariance, investable_mask = nan_investable_test_data

    model = MaximumDiversification(
        prior_estimator=fixed_return_distribution_prior(mu=mu, covariance=covariance),
    )
    model.fit(X)

    return_distribution = model.prior_estimator_.return_distribution_
    assert return_distribution.n_assets == X.shape[1]
    assert return_distribution.n_investable_assets == np.count_nonzero(investable_mask)
    np.testing.assert_array_equal(model.investable_mask_, investable_mask)
    assert model.weights_.shape == (X.shape[1],)
    assert np.isfinite(model.weights_).all()
    np.testing.assert_allclose(model.weights_[~investable_mask], 0)
    np.testing.assert_allclose(model.weights_.sum(), 1)

    portfolio = model.predict(X)
    expected_returns = (
        X.iloc[:, investable_mask].to_numpy() @ model.weights_[investable_mask]
    )
    np.testing.assert_allclose(portfolio.returns, expected_returns)
