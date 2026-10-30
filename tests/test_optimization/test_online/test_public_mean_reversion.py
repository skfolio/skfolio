"""Public mean-reversion parameter routing and independent update references."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from skfolio.optimization.online import PassiveAggressiveMeanReversion
from skfolio.optimization.online._loser import FollowTheLoser


@pytest.mark.parametrize("variant", ["simple", "slack_linear", "slack_quadratic"])
def test_pamr_public_variant_matches_closed_form(variant):
    model = PassiveAggressiveMeanReversion().set_params(
        variant=variant, aggressiveness=0.1
    )
    model = clone(model)
    model.partial_fit([[0.2, -0.1]])
    # At uniform weights the loss is 0.05 and squared centered norm is 0.045.
    tau = 0.05 / 0.045
    if variant == "slack_linear":
        tau = min(0.1, tau)
    elif variant == "slack_quadratic":
        tau = 0.05 / (0.045 + 1 / (2 * 0.1))
    expected = np.array([0.5, 0.5]) - tau * np.array([0.15, -0.15])
    assert_allclose(model.weights_, expected, atol=1e-10)
    reference = FollowTheLoser(
        strategy="pamr", pamr_variant=variant, pamr_C=0.1, update_mode="pa"
    ).fit([[0.2, -0.1]])
    assert_allclose(model.weights_, reference.weights_)


@pytest.mark.parametrize("params", [{"variant": "invalid"}, {"aggressiveness": 0}])
def test_pamr_public_parameter_validation(params):
    with pytest.raises(ValueError):
        PassiveAggressiveMeanReversion(**params).fit([[0.01, -0.01]])
