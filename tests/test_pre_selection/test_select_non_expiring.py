import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import EqualWeighted
from skfolio.pre_selection import SelectComplete, SelectNonExpiring
from skfolio.preprocessing import prices_to_returns


def generate_prices(n: int) -> list[float]:
    # Just for example purposes
    return list(100 * np.cumprod(1 + np.random.normal(0, 0.01, n)))


@pytest.fixture
def X_df():
    X_df = pd.DataFrame(
        {
            "asset1": [1, 2, 3, 4],
            "asset2": [2, 3, 4, 5],
            "asset3": [3, 4, 5, 6],
            "asset4": [4, 5, np.nan, 7],
        },
        index=pd.date_range("2023-01-01", periods=4, freq="D"),
    )
    return X_df


@pytest.fixture
def prices():
    prices = pd.DataFrame(
        {
            "inception": [np.nan] * 3 + generate_prices(10),
            "defaulted": generate_prices(6) + [0.0] + [np.nan] * 6,
            "expired": generate_prices(10) + [np.nan] * 3,
            "complete": generate_prices(13),
        },
        index=pd.date_range(start="2024-01-03", end="2024-01-19", freq="B"),
    )
    return prices


@pytest.mark.parametrize(
    "expiration_dates,expected",
    [
        (
            {
                "asset1": pd.Timestamp("2023-01-10"),
                "asset2": pd.Timestamp("2023-01-02"),
                "asset3": pd.Timestamp("2023-01-06"),
                "asset4": dt.datetime(2023, 5, 1),
            },
            pd.DataFrame(
                {"asset1": [1, 2, 3, 4], "asset4": [4, 5, np.nan, 7]},
                index=pd.date_range("2023-01-01", periods=4, freq="D"),
            ),
        ),
    ],
)
def test_select_non_expiring(X_df, expiration_dates, expected):
    with config_context(transform_output="pandas"):
        selector = SelectNonExpiring(
            expiration_dates=expiration_dates,
            expiration_lookahead=pd.DateOffset(days=5),
        )
        res = selector.fit_transform(X_df)
        pd.testing.assert_frame_equal(res, expected)


def test_pipeline(prices):
    X = prices_to_returns(prices, drop_inceptions_nan=False, fill_nan=False)

    with config_context(transform_output="pandas"):
        model = Pipeline(
            [
                ("select_complete_assets", SelectComplete()),
                (
                    "select_non_expiring_assets",
                    SelectNonExpiring(
                        expiration_dates={"expired": dt.datetime(2024, 1, 16)},
                        expiration_lookahead=pd.offsets.BusinessDay(4),
                    ),
                ),
                ("zero_imputation", SimpleImputer(strategy="constant", fill_value=0)),
                ("optimization", EqualWeighted()),
            ]
        )
        pred = cross_val_predict(model, X, cv=WalkForward(train_size=4, test_size=4))
        expected = pd.DataFrame(
            {
                "EqualWeighted": {
                    "defaulted": 0.3333333333333333,
                    "expired": 0.3333333333333333,
                    "complete": 0.3333333333333333,
                    "inception": 0.0,
                },
                "EqualWeighted_1": {
                    "defaulted": 0.0,
                    "expired": 0.0,
                    "complete": 0.5,
                    "inception": 0.5,
                },
            },
        )
        expected.index.name = "asset"
        pd.testing.assert_frame_equal(pred.composition, expected)
        assert len(pred.returns) == 8
        assert np.all(~np.isnan(pred.returns))
