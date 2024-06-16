import numpy as np
import pytest
import sklearn.linear_model as skl
from sklearn import config_context
from sklearn.exceptions import UnsetMetadataPassedError

from skfolio.moments import ImpliedCovariance, LedoitWolf
from skfolio.moments.covariance._implied_covariance import (
    _compute_implied_vol,
    _compute_realised_vol,
)


def test_compute_realised_vol(X):
    n_observations, n_assets = X.shape
    window = 30
    rv = _compute_realised_vol(np.array(X), window=window, ddof=0)
    assert rv.shape == (n_observations // window, n_assets)

    res = []
    k = 0
    while (k + 1) * window <= n_observations:
        res.append(
            np.asarray(
                X.iloc[
                    n_observations - (k + 1) * window : n_observations - k * window
                ].std(axis=0, ddof=0)
            )
        )
        k += 1

    res = np.flip(np.array(res), 0)

    np.testing.assert_almost_equal(res, rv)


def test_compute_implied_vol(implied_vol):
    n_observations, n_assets = implied_vol.shape
    window = 30

    iv = _compute_implied_vol(np.asarray(implied_vol), window=window)
    assert iv.shape == (n_observations // window, n_assets)

    res = []
    k = 0
    while (k + 1) * window <= n_observations:
        res.append(np.asarray(implied_vol.iloc[n_observations - k * window - 1]))
        k += 1
    res = np.flip(np.array(res), 0)

    np.testing.assert_almost_equal(res, iv)


def test_implied_covariance_without_vol(X):
    model = ImpliedCovariance()
    with pytest.raises(ValueError):
        model.fit(X)


def test_implied_covariance_default(X, implied_vol):
    model = ImpliedCovariance()
    model.fit(X, implied_vol=implied_vol)
    assert model.covariance_.shape == (20, 20)
    assert len(model.linear_regressors_) == 20
    assert model.coefs_.shape == (20, 2)
    assert model.intercepts_.shape == (20,)
    assert model.r2_scores_.shape == (20,)
    assert model.pred_realised_vols_.shape == (20,)

    res = np.exp(
        model.coefs_[:, 0] * np.log(np.asarray(implied_vol.iloc[-1] / np.sqrt(252)))
        + model.coefs_[:, 1] * np.log(np.asarray(X[-model.window :].std(ddof=1)))
        + model.intercepts_
    )

    np.testing.assert_almost_equal(model.pred_realised_vols_, res)

    np.testing.assert_almost_equal(
        model.covariance_[:2][:2],
        np.array(
            [
                [
                    6.57488641e-04,
                    2.83134035e-04,
                    1.56184082e-04,
                    1.75814250e-04,
                    1.22056078e-04,
                    1.38660138e-04,
                    1.47083022e-04,
                    8.10806087e-05,
                    1.42473991e-04,
                    8.01598741e-05,
                    9.68377637e-05,
                    7.98446492e-05,
                    2.15007353e-04,
                    9.97345580e-05,
                    8.60888400e-05,
                    8.56609339e-05,
                    1.37569074e-04,
                    1.32999989e-04,
                    8.21417385e-05,
                    1.06253398e-04,
                ],
                [
                    2.83134035e-04,
                    1.01675318e-03,
                    2.20576378e-04,
                    2.51503645e-04,
                    1.76240627e-04,
                    1.87494904e-04,
                    2.00154514e-04,
                    8.31528862e-05,
                    1.81429439e-04,
                    8.63395567e-05,
                    1.16628151e-04,
                    9.03413258e-05,
                    2.57876321e-04,
                    1.02332010e-04,
                    1.10774884e-04,
                    8.74268879e-05,
                    2.68487267e-04,
                    1.62195796e-04,
                    8.62525325e-05,
                    1.38431233e-04,
                ],
            ]
        ),
    )

    np.testing.assert_almost_equal(
        np.diag(model.covariance_), model.pred_realised_vols_**2
    )
    np.testing.assert_almost_equal(
        model.pred_realised_vols_,
        np.array(
            [
                0.02564154,
                0.03188657,
                0.02002452,
                0.02736693,
                0.01973539,
                0.02245573,
                0.01815133,
                0.00957966,
                0.01799403,
                0.01123385,
                0.01780456,
                0.01307078,
                0.02186195,
                0.01093394,
                0.0151937,
                0.01116985,
                0.03529737,
                0.01416595,
                0.0128378,
                0.01956481,
            ]
        ),
    )


def test_implied_covariance_no_intercept(X, implied_vol):
    model = ImpliedCovariance(
        linear_regressor=skl.LinearRegression(fit_intercept=False)
    )
    model.fit(X, implied_vol=implied_vol)
    np.testing.assert_almost_equal(model.intercepts_, np.zeros(20))


@pytest.mark.parametrize("volatility_risk_premium_adj", [0.1, 1, 1.5, 100])
def test_implied_covariance_volatility_risk_premium_adj(
    X, implied_vol, volatility_risk_premium_adj
):
    model = ImpliedCovariance(volatility_risk_premium_adj=volatility_risk_premium_adj)
    model.fit(X, implied_vol=implied_vol)

    res = np.asarray(implied_vol.iloc[-1] / np.sqrt(252) / volatility_risk_premium_adj)
    np.testing.assert_almost_equal(model.pred_realised_vols_, res)


@pytest.mark.parametrize("volatility_risk_premium_adj", [-0.5, 0])
def test_implied_covariance_volatility_risk_premium_adj_non_pos(
    X, implied_vol, volatility_risk_premium_adj
):
    model = ImpliedCovariance(volatility_risk_premium_adj=volatility_risk_premium_adj)
    with pytest.raises(ValueError):
        model.fit(X, implied_vol=implied_vol)


@pytest.mark.parametrize("n_folds", [0.1, 1, 2])
def test_implied_covariance_window_too_big(X, implied_vol, n_folds):
    model = ImpliedCovariance(window=len(X) // n_folds)
    with pytest.raises(ValueError):
        model.fit(X, implied_vol=implied_vol)


@pytest.mark.parametrize("window", [-1, 0, 1, 2])
def test_implied_covariance_small_error(X, implied_vol, window):
    model = ImpliedCovariance(window=window)
    with pytest.raises(ValueError):
        model.fit(X, implied_vol=implied_vol)


@pytest.mark.parametrize("window", [3, 4])
def test_implied_covariance_small(X, implied_vol, window):
    model = ImpliedCovariance(window=window)
    model.fit(X, implied_vol=implied_vol)
    assert np.any(model.coefs_ != 0)


def test_implied_covariance_meta_data_routing_error(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = ImpliedCovariance(covariance_estimator=ImpliedCovariance())
        with pytest.raises(UnsetMetadataPassedError):
            model.fit(X, implied_vol=implied_vol)


def test_implied_covariance_meta_data_routing(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = ImpliedCovariance(
            covariance_estimator=ImpliedCovariance().set_fit_request(implied_vol=True)
        )
        model.fit(X, implied_vol=implied_vol)

        model_ref = ImpliedCovariance()
        model_ref.fit(X, implied_vol=implied_vol)

        np.testing.assert_almost_equal(model.covariance_, model_ref.covariance_)


def test_implied_covariance_ledoit_wolf(X, implied_vol):
    model = ImpliedCovariance(covariance_estimator=LedoitWolf())
    model.fit(X, implied_vol=implied_vol)

    model_imp_ref = ImpliedCovariance()
    model_imp_ref.fit(X, implied_vol=implied_vol)

    model_led_ref = LedoitWolf()
    model_led_ref.fit(X)

    np.testing.assert_almost_equal(
        np.diag(model.covariance_), np.diag(model_imp_ref.covariance_)
    )

    np.fill_diagonal(model.covariance_, 0)
    np.fill_diagonal(model_led_ref.covariance_, 0)

    np.testing.assert_almost_equal(model.covariance_, model_led_ref.covariance_)
