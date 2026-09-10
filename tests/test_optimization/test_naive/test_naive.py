from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context

from skfolio.moments import (
    ImpliedCovariance,
)
from skfolio.optimization import MeanRisk
from skfolio.optimization.naive import EqualWeighted, InverseVolatility, Random
from skfolio.portfolio import Portfolio
from skfolio.prior import EmpiricalPrior, TimeSeriesFactorModel


class TestInverseVolatility:
    def test_fit(self, X, factors):
        model = InverseVolatility()
        model.fit(X)
        np.testing.assert_almost_equal(sum(model.weights_), 1)
        w = 1 / np.std(np.asarray(X), axis=0)
        w /= sum(w)
        np.testing.assert_almost_equal(model.weights_, w)

        model = InverseVolatility(prior_estimator=TimeSeriesFactorModel())
        model.fit(X, factors=factors)

    def test_predict_feature_names(self, X):
        """Preserve the fitted asset-to-weight mapping during prediction."""
        assert isinstance(X, pd.DataFrame)
        model = InverseVolatility().fit(X)
        weights = model.weights_.copy()
        feature_names = X.columns.to_numpy()
        expected_returns = X.to_numpy() @ weights

        assert model.n_features_in_ == X.shape[1]
        np.testing.assert_array_equal(model.feature_names_in_, feature_names)
        portfolio = model.predict(X)
        assert isinstance(portfolio, Portfolio)
        np.testing.assert_allclose(portfolio.returns, expected_returns)

        # Unlabeled inputs remain positional when their feature count is unchanged.
        with pytest.warns(UserWarning, match="X does not have valid feature names"):
            portfolio = model.predict(X.to_numpy())
        assert isinstance(portfolio, Portfolio)
        np.testing.assert_allclose(portfolio.returns, expected_returns)

        invalid_inputs = [
            X.iloc[:, ::-1],
            X.iloc[:, :-1],
            X.assign(unexpected_asset=0.0),
        ]
        for invalid_X in invalid_inputs:
            assert isinstance(invalid_X, pd.DataFrame)
            with pytest.raises(ValueError, match="feature names should match"):
                model.predict(invalid_X)

            # Rejected predictions must not alter the fitted economic state.
            np.testing.assert_array_equal(model.weights_, weights)
            assert model.n_features_in_ == X.shape[1]
            np.testing.assert_array_equal(model.feature_names_in_, feature_names)

    def test_as_fallback(self, X):
        """Retain fitted feature metadata when used as a successful fallback."""
        assert isinstance(X, pd.DataFrame)
        model = MeanRisk(solver="NOT_A_SOLVER", fallback=InverseVolatility()).fit(X)

        assert isinstance(model.fallback_, InverseVolatility)
        assert model.fallback_chain_ is not None
        assert model.fallback_chain_[-1] == ("InverseVolatility()", "success")
        assert model.n_features_in_ == X.shape[1]
        np.testing.assert_array_equal(model.feature_names_in_, X.columns)
        portfolio = model.predict(X)
        assert isinstance(portfolio, Portfolio)
        np.testing.assert_allclose(portfolio.returns, X.to_numpy() @ model.weights_)

    def test_metadata_routing(self, X, implied_vol):
        with config_context(enable_metadata_routing=True):
            model = InverseVolatility(
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


class TestEqualWeighted:
    def test_fit(self, X):
        model = EqualWeighted()
        model.fit(X)
        weights = model.weights_
        np.testing.assert_almost_equal(sum(weights), 1)
        w = 1 / X.shape[1]
        np.testing.assert_almost_equal(weights, w)


class TestRandom:
    def test_fit(self, X):
        model = Random()
        weights = model.fit(X).weights_
        np.testing.assert_almost_equal(sum(weights), 1)
