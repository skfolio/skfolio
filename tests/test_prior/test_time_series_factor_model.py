from __future__ import annotations

import numpy as np
import pytest
from sklearn import config_context
from sklearn.linear_model import LassoCV

from skfolio.moments import ImpliedCovariance
from skfolio.prior import (
    BlackLitterman,
    EmpiricalPrior,
    LoadingMatrixRegression,
    TimeSeriesFactorModel,
)


def test_factor_model(X, y):
    model = TimeSeriesFactorModel()
    model.fit(X, y)
    assert model.return_distribution_
    assert model.return_distribution_.mu.shape == (20,)
    np.testing.assert_almost_equal(
        model.return_distribution_.cholesky @ model.return_distribution_.cholesky.T,
        model.return_distribution_.covariance,
        15,
    )

    model = TimeSeriesFactorModel(
        loading_matrix_estimator=LoadingMatrixRegression(
            linear_regressor=LassoCV(cv=5, fit_intercept=False), n_jobs=-1
        ),
    )
    model.fit(X, y)
    assert model.return_distribution_
    np.testing.assert_almost_equal(
        model.return_distribution_.cholesky @ model.return_distribution_.cholesky.T,
        model.return_distribution_.covariance,
        15,
    )


def test_black_litterman_factor_model(X, y):
    factor_views = ["MTUM - QUAL == 0.03 ", "SIZE - USMV== 0.04", "VLUE == 0.06 "]
    n_observations = X.shape[0]
    model = TimeSeriesFactorModel(
        factor_prior_estimator=BlackLitterman(
            views=factor_views, tau=1 / n_observations
        ),
    )
    model.fit(X, y)

    assert model.return_distribution_.mu.shape == (20,)
    assert model.return_distribution_.covariance.shape == (20, 20)
    np.testing.assert_almost_equal(
        model.return_distribution_.mu,
        np.array(
            [
                0.03913265,
                0.06901794,
                0.04743629,
                0.04119901,
                0.03839577,
                0.04114205,
                0.03060717,
                0.00924759,
                0.04197938,
                0.0095809,
                0.01440974,
                0.0130805,
                0.03724454,
                0.00999507,
                0.01208523,
                0.00583489,
                0.05676089,
                0.02747053,
                0.01263982,
                0.0330812,
            ]
        ),
    )

    np.testing.assert_almost_equal(
        model.return_distribution_.covariance[:5, :5],
        np.array(
            [
                [0.00033581, 0.00025468, 0.00017332, 0.00016387, 0.00014255],
                [0.00025468, 0.00137777, 0.00023824, 0.00022206, 0.00019555],
                [0.00017332, 0.00023824, 0.00038022, 0.00019973, 0.0001911],
                [0.00016387, 0.00022206, 0.00019973, 0.00060898, 0.00016214],
                [0.00014255, 0.00019555, 0.0001911, 0.00016214, 0.00034686],
            ]
        ),
    )


def test_metadata_routing_error(X, y, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = TimeSeriesFactorModel(
            factor_prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        with pytest.raises(
            ValueError, match="The following assets are missing from `implied_vol`"
        ):
            model.fit(X, y, implied_vol=implied_vol)


def test_metadata_routing(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = TimeSeriesFactorModel(
            factor_prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance().set_fit_request(
                    implied_vol=True
                )
            )
        )

        with pytest.raises(ValueError):
            model.fit(X, X)

        model.fit(X, X, implied_vol=implied_vol)

    # noinspection PyUnresolvedReferences
    assert model.factor_prior_estimator_.covariance_estimator_.r2_scores_.shape == (20,)
