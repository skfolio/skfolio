from __future__ import annotations

import pytest

from skfolio.distribution import BaseDistribution


def test_base_distribution_is_abstract():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseDistribution()
