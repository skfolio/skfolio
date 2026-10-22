import numpy as np

from skfolio.optimization.online._loser import FTLStrategy, FollowTheLoser


def test_pamr_tiny_cnorm_no_update():
    # Make x_t nearly constant so c = x - mean(x) ~ 0
    x = np.zeros(3)  # net returns all zero
    est = FollowTheLoser(strategy=FTLStrategy.PAMR, epsilon=1.0)
    w0 = np.ones(3) / 3
    est.initial_weights = w0
    est.partial_fit(x)
    # With c_norm ~ 0, the PAMR update should be passive -> weights unchanged after projection
    assert np.allclose(est.weights_, w0)
