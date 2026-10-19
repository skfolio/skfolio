"""Explicit conversion from price relatives, without clipping or sign reversal."""

import numpy as np
import pytest

from skfolio.optimization.online._utils import net_to_relatives, relatives_to_net


@pytest.mark.parametrize("values", [[0.0, 1.0, 1.2], [[0.0, 1.0, 1.2]]])
def test_relatives_to_net(values):
    original = np.array(values)
    np.testing.assert_allclose(relatives_to_net(original), [[-1.0, 0.0, 0.2]])
    np.testing.assert_array_equal(original, values)


def test_return_conversion_round_trip():
    returns = np.array([[-0.3, 0.0, 0.5], [0.1, -0.2, 0.3]])
    np.testing.assert_allclose(relatives_to_net(net_to_relatives(returns)), returns)


@pytest.mark.parametrize("values", [[-0.1, 1], [np.nan], [np.inf], [-np.inf]])
def test_invalid_relatives(values):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        relatives_to_net(values)


@pytest.mark.parametrize("values", [1.0, np.ones((1, 1, 1))])
def test_invalid_dimensions(values):
    with pytest.raises(ValueError, match="1D or 2D"):
        relatives_to_net(values)
