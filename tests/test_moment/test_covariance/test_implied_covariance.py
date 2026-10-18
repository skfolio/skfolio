from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model as skl
from sklearn import config_context
from sklearn.exceptions import UnsetMetadataPassedError

from skfolio.moments import ImpliedCovariance, LedoitWolf
from skfolio.moments.covariance._implied_covariance import (
    _compute_implied_vol,
    _compute_realised_vol,
)
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("read_only", [False, True])
@pytest.mark.parametrize("premium", [None, 1.5])
@pytest.mark.parametrize("annualization_factor", [1, 252])
def test_implied_covariance_preserves_array(
    dtype, read_only, premium, annualization_factor
):
    """Annualization preserves caller units and repeat-fit results in both modes."""
    X = np.random.default_rng(52).normal(0, 0.01, (90, 3))
    implied_vol = np.tile([0.16, 0.24, 0.32], (90, 1)).astype(dtype)
    original = implied_vol.copy()
    implied_vol.flags.writeable = not read_only
    model = ImpliedCovariance(
        annualization_factor=annualization_factor,
        volatility_risk_premium_adj=premium,
    )

    for _ in range(2):
        model.fit(X, implied_vol=implied_vol)
        np.testing.assert_array_equal(implied_vol, original)
        assert implied_vol.flags.writeable == (not read_only)
        if premium is not None:
            assert model.pred_realised_vols_.dtype == dtype
            # Convert to the returns frequency, then adjust for the risk premium.
            expected_vol = (
                original[-1].astype(np.float64)
                / np.sqrt(annualization_factor)
                / premium
            )
            expected = np.corrcoef(X, rowvar=False) * np.outer(
                expected_vol, expected_vol
            )
            np.testing.assert_allclose(model.covariance_, expected, rtol=5e-7)
        if _ == 0:
            first = model.covariance_.copy()
        else:
            np.testing.assert_array_equal(model.covariance_, first)


@pytest.mark.parametrize(
    "params, error, message",
    [
        ({"volatility_risk_premium_adj": 0}, ValueError, "must be strictly positive"),
        ({"volatility_risk_premium_adj": np.nan}, ValueError, "must contain a value"),
        ({"volatility_risk_premium_adj": [1, 2]}, ValueError, "must be a of shape"),
        ({"window_size": 0}, ValueError, "window must be strictly greater than 2"),
        ({"window_size": 40}, ValueError, "Not enough observations"),
        ({"linear_regressor": 1}, TypeError, "Expected type"),
    ],
)
def test_implied_covariance_preserves_array_on_rejection(params, error, message):
    """Later premium and regression validation must not rescale caller metadata."""
    X = np.random.default_rng(52).normal(0, 0.01, (90, 3))
    implied_vol = np.tile([0.16, 0.24, 0.32], (90, 1))
    original = implied_vol.copy()

    with pytest.raises(error, match=message):
        ImpliedCovariance(**params).fit(X, implied_vol=implied_vol)

    np.testing.assert_array_equal(implied_vol, original)


@pytest.mark.parametrize("nested", [False, True])
def test_implied_covariance_preserves_routed_array(nested):
    """The first covariance consumer must preserve metadata for every later fit."""
    X = np.random.default_rng(52).normal(0, 0.01, (90, 3))
    implied_vol = np.tile([0.16, 0.24, 0.32], (90, 1))
    original = implied_vol.copy()
    expected_vol = original[-1] / np.sqrt(252)
    expected = np.corrcoef(X, rowvar=False) * np.outer(expected_vol, expected_vol)

    with config_context(enable_metadata_routing=True):
        child = (
            ImpliedCovariance(volatility_risk_premium_adj=1).set_fit_request(
                implied_vol=True
            )
            if nested
            else None
        )
        model = MeanRisk(
            prior_estimator=EmpiricalPrior(
                covariance_estimator=ImpliedCovariance(
                    prior_covariance_estimator=child, volatility_risk_premium_adj=1
                ).set_fit_request(implied_vol=True)
            )
        )
        for _ in range(2):
            model.fit(X, implied_vol=implied_vol)
            np.testing.assert_array_equal(implied_vol, original)
            np.testing.assert_allclose(
                model.prior_estimator_.return_distribution_.covariance, expected
            )


@pytest.mark.parametrize("representation", ["fortran", "slice", "frame", "nullable"])
def test_implied_covariance_preserves_representations(representation):
    """Copying retains native layout conversion and labeled-column alignment."""
    X = np.random.default_rng(52).normal(0, 0.01, (90, 3))
    values = np.tile([0.16, 0.24, 0.32], (90, 1))
    if representation == "fortran":
        implied_vol = np.asfortranarray(values)
    elif representation == "slice":
        implied_vol = np.repeat(values, 2, axis=1)[:, ::2]
    else:
        X = pd.DataFrame(X, columns=["a", "b", "c"])
        implied_vol = pd.DataFrame(values, columns=X.columns).reindex(
            columns=["c", "a", "b"]
        )
        if representation == "nullable":
            implied_vol = implied_vol.astype("Float64")
    original = implied_vol.copy()
    model = ImpliedCovariance(volatility_risk_premium_adj=1)
    model.fit(X, implied_vol=implied_vol)
    expected_vol = values[-1] / np.sqrt(252)
    expected = np.corrcoef(X, rowvar=False) * np.outer(expected_vol, expected_vol)
    np.testing.assert_allclose(model.covariance_, expected)
    if isinstance(implied_vol, pd.DataFrame):
        pd.testing.assert_frame_equal(implied_vol, original)
        np.testing.assert_array_equal(model.feature_names_in_, X.columns)
    else:
        np.testing.assert_array_equal(implied_vol, original)


def test_compute_realised_vol(X):
    n_observations, n_assets = X.shape
    window = 30
    rv = _compute_realised_vol(np.array(X), window_size=window, ddof=0)
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

    iv = _compute_implied_vol(np.asarray(implied_vol), window_size=window)
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
    with pytest.raises(ValueError, match="`implied_vol` cannot be None"):
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
        + model.coefs_[:, 1] * np.log(np.asarray(X[-model.window_size :].std(ddof=1)))
        + model.intercepts_
    )

    np.testing.assert_almost_equal(model.pred_realised_vols_, res)

    np.testing.assert_almost_equal(
        model.covariance_[:2][:2],
        np.array(
            [
                [
                    5.90463014e-04,
                    3.14255178e-04,
                    1.98741290e-04,
                    2.51797303e-04,
                    1.61212665e-04,
                    1.80530097e-04,
                    2.20630221e-04,
                    8.93487189e-05,
                    1.89943341e-04,
                    9.79986309e-05,
                    1.31366228e-04,
                    9.34134473e-05,
                    3.60097117e-04,
                    1.20963142e-04,
                    1.19939740e-04,
                    1.06662034e-04,
                    1.65682862e-04,
                    1.52434019e-04,
                    1.00953936e-04,
                    1.48998657e-04,
                ],
                [
                    3.14255178e-04,
                    9.78514216e-04,
                    1.77724392e-04,
                    2.28075160e-04,
                    1.47394907e-04,
                    1.54569628e-04,
                    1.90109818e-04,
                    5.80210200e-05,
                    1.53155622e-04,
                    6.68358535e-05,
                    1.00179549e-04,
                    6.69247480e-05,
                    2.73473028e-04,
                    7.85878850e-05,
                    9.77224291e-05,
                    6.89300792e-05,
                    2.04746823e-04,
                    1.17708165e-04,
                    6.71224688e-05,
                    1.22916537e-04,
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
                0.02429944,
                0.03128121,
                0.01905083,
                0.02683106,
                0.01894729,
                0.02160476,
                0.01737502,
                0.00952402,
                0.01747525,
                0.01069423,
                0.01711937,
                0.01207171,
                0.02173792,
                0.01078584,
                0.01494446,
                0.01100265,
                0.0348573,
                0.0140789,
                0.01244791,
                0.01860689,
            ]
        ),
    )


def test_implied_covariance_deprecated_annualized_factor():
    with pytest.warns(FutureWarning, match="annualized_factor"):
        model = ImpliedCovariance(annualized_factor=12)

    assert model.annualization_factor == 12
    assert model.annualized_factor is None

    with pytest.warns(FutureWarning, match="annualized_factor"):
        model.set_params(annualized_factor=52)

    assert model.annualization_factor == 52
    assert model.annualized_factor is None


def test_implied_covariance_annualization_factor_conflict():
    with pytest.raises(ValueError, match="annualized_factor"):
        ImpliedCovariance(annualization_factor=252, annualized_factor=252)


def test_implied_covariance_no_intercept(X, implied_vol):
    model = ImpliedCovariance(
        linear_regressor=skl.LinearRegression(fit_intercept=False)
    )
    model.fit(X, implied_vol=implied_vol)
    np.testing.assert_almost_equal(model.intercepts_, np.zeros(20))


@pytest.mark.parametrize(
    "volatility_risk_premium_adj, expected",
    [
        (0.1, 0.1),
        (1, 1),
        (1.5, 1.5),
        (100, 100),
        (np.arange(1, 21), np.arange(1, 21)),
        (
            {
                "GE": 6,
                "HD": 7,
                "JNJ": 8,
                "JPM": 9,
                "KO": 10,
                "LLY": 11,
                "MRK": 12,
                "MSFT": 13,
                "PEP": 14,
                "PFE": 15,
                "AAPL": 1,
                "AMD": 2,
                "BAC": 3,
                "BBY": 4,
                "CVX": 5,
                "PG": 16,
                "RRC": 17,
                "UNH": 18,
                "WMT": 19,
                "XOM": 20,
                "error": 100,
            },
            np.arange(1, 21),
        ),
    ],
)
def test_implied_covariance_volatility_risk_premium_adj(
    X, implied_vol, volatility_risk_premium_adj, expected
):
    model = ImpliedCovariance(volatility_risk_premium_adj=volatility_risk_premium_adj)
    model.fit(X, implied_vol=implied_vol)

    res = np.asarray(implied_vol.iloc[-1] / np.sqrt(252) / expected)
    np.testing.assert_almost_equal(model.pred_realised_vols_, res)


@pytest.mark.parametrize("volatility_risk_premium_adj", [-0.5, 0])
def test_implied_covariance_volatility_risk_premium_adj_non_pos(
    X, implied_vol, volatility_risk_premium_adj
):
    model = ImpliedCovariance(volatility_risk_premium_adj=volatility_risk_premium_adj)
    with pytest.raises(
        ValueError, match="volatility_risk_premium_adj must be strictly positive"
    ):
        model.fit(X, implied_vol=implied_vol)


@pytest.mark.parametrize("n_folds", [0.1, 1, 2])
def test_implied_covariance_window_too_big(X, implied_vol, n_folds):
    model = ImpliedCovariance(window_size=len(X) // n_folds)
    with pytest.raises(
        ValueError,
        match="Not enough observations to compute the volatility regression coefficients",
    ):
        model.fit(X, implied_vol=implied_vol)


@pytest.mark.parametrize("window_size", [-1, 0, 1, 2])
def test_implied_covariance_small_error(X, implied_vol, window_size):
    model = ImpliedCovariance(window_size=window_size)
    with pytest.raises(ValueError, match="window must be strictly greater than 2"):
        model.fit(X, implied_vol=implied_vol)


@pytest.mark.parametrize("window_size", [3, 4])
def test_implied_covariance_small(X, implied_vol, window_size):
    model = ImpliedCovariance(window_size=window_size)
    model.fit(X, implied_vol=implied_vol)
    assert np.any(model.coefs_ != 0)


def test_implied_covariance_meta_data_routing_error(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = ImpliedCovariance(prior_covariance_estimator=ImpliedCovariance())
        with pytest.raises(
            UnsetMetadataPassedError,
            match="are passed but are not explicitly set as requested",
        ):
            model.fit(X, implied_vol=implied_vol)


def test_implied_covariance_meta_data_routing(X, implied_vol):
    with config_context(enable_metadata_routing=True):
        model = ImpliedCovariance(
            prior_covariance_estimator=ImpliedCovariance().set_fit_request(
                implied_vol=True
            )
        )
        model.fit(X, implied_vol=implied_vol)

        model_ref = ImpliedCovariance()
        model_ref.fit(X, implied_vol=implied_vol)

        np.testing.assert_almost_equal(model.covariance_, model_ref.covariance_)


def test_implied_covariance_ledoit_wolf(X, implied_vol):
    model = ImpliedCovariance(prior_covariance_estimator=LedoitWolf())
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

    np.testing.assert_almost_equal(model.covariance_, model_led_ref.covariance_, 3)


@pytest.fixture
def synthetic_implied_vol(X):
    """Deterministic implied volatilities aligned with `X` (no download needed)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        rng.uniform(0.1, 0.4, size=X.shape), index=X.index, columns=X.columns
    )


def test_implied_covariance_set_params_annualization_conflict():
    model = ImpliedCovariance()
    with pytest.raises(
        ValueError,
        match="`annualized_factor` is deprecated; pass only `annualization_factor`",
    ):
        model.set_params(annualized_factor=12, annualization_factor=252)


def test_implied_covariance_volatility_risk_premium_adj_missing_assets(
    X, synthetic_implied_vol
):
    # A dict covering only some assets is rejected by `input_to_array`, which
    # fills the gaps with NaN and refuses them.
    model = ImpliedCovariance(volatility_risk_premium_adj={"AAPL": 1.0})
    with pytest.raises(ValueError, match=r"`volatility_risk_premium_adj` contains NaN"):
        model.fit(X, implied_vol=synthetic_implied_vol)


def test_implied_covariance_volatility_risk_premium_adj_scalar_nan(
    X, synthetic_implied_vol
):
    # A scalar NaN bypasses `input_to_array` and reaches the estimator's own guard.
    model = ImpliedCovariance(volatility_risk_premium_adj=np.nan)
    with pytest.raises(
        ValueError,
        match="volatility_risk_premium_adj must contain a value for each assets",
    ):
        model.fit(X, implied_vol=synthetic_implied_vol)


def test_implied_covariance_implied_vol_must_be_2d(X):
    model = ImpliedCovariance()
    with pytest.raises(ValueError, match="must be 2D array of shape"):
        model.fit(X, implied_vol=np.ones(len(X)))


def test_implied_covariance_implied_vol_shape_mismatch(X):
    model = ImpliedCovariance()
    with pytest.raises(ValueError, match=r"implied_vol.shape == .* expected"):
        model.fit(X, implied_vol=np.ones((len(X), 3)))
