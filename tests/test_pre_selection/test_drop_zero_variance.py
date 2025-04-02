import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from skfolio.pre_selection import DropZeroVariance


def test_drop_zero_variance():
    X = pd.DataFrame(dict(a=[1, 1, 1], b=[1, 2, 3]))

    transformer = DropZeroVariance().set_output(transform="pandas")
    Xt = transformer.fit_transform(X)
    assert_frame_equal(Xt, X[["b"]])


def test_wrong_threshold():
    X = pd.DataFrame(dict(a=[1, 1, 1], b=[1, 2, 3]))
    transformer = DropZeroVariance(threshold=-1)
    with pytest.raises(ValueError, match="higher than 0"):
        transformer.fit(X)
