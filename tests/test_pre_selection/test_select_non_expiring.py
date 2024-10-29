import datetime as dt

import numpy as np
import pandas as pd
import pytest
from sklearn import config_context

from skfolio.pre_selection import SelectNonExpiring


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
