import numpy as np
import pandas as pd
import pytest
from sklearn import config_context

from skfolio.pre_selection import SelectComplete


@pytest.fixture
def X_df():
    X_df = pd.DataFrame(
        {
            "asset1": [np.nan, np.nan, 2, 3, 4],  # Starts late (inception)
            "asset2": [1, 2, 3, 4, 5],  # Complete data
            "asset3": [1, 2, 3, np.nan, 5],  # Missing values within data
            "asset4": [1, 2, 3, 4, np.nan],  # Ends early (expiration)
        }
    )
    return X_df


@pytest.mark.parametrize("use_df", [True, False])
@pytest.mark.parametrize(
    "drop_assets_with_internal_nan,expected",
    [
        (
            False,
            np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, np.nan], [5.0, 5.0]]),
        ),
        (True, np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])),
    ],
)
def test_select_complete(
    use_df,
    X_df,
    drop_assets_with_internal_nan,
    expected,
):
    if not use_df:
        X = np.asarray(X_df)
    else:
        X = X_df
    selector = SelectComplete(
        drop_assets_with_internal_nan=drop_assets_with_internal_nan
    )
    res = selector.fit_transform(X)
    np.testing.assert_almost_equal(res, expected)


@pytest.mark.parametrize(
    "drop_assets_with_internal_nan,expected",
    [
        (
            False,
            pd.DataFrame(
                {
                    "asset2": [1, 2, 3, 4, 5],
                    "asset3": [1, 2, 3, np.nan, 5],
                }
            ),
        ),
        (
            True,
            pd.DataFrame(
                {
                    "asset2": [1, 2, 3, 4, 5],
                }
            ),
        ),
    ],
)
def test_select_complete_pandas(X_df, drop_assets_with_internal_nan, expected):
    with config_context(transform_output="pandas"):
        selector = SelectComplete(
            drop_assets_with_internal_nan=drop_assets_with_internal_nan
        )
        res = selector.fit_transform(X_df)
        pd.testing.assert_frame_equal(res, expected)
